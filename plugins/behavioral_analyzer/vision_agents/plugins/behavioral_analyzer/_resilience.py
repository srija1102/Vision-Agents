"""Recovery and resilience tracking after stress spikes.

Measures how long a candidate takes to return within 0.5σ of their
baseline stress after a spike, and produces a normalized resilience score.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class _RecoveryRecord:
    spike_timestamp: float
    spike_stress: float
    recovery_timestamp: Optional[float] = None
    recovery_time_s: Optional[float] = None
    timed_out: bool = False


@dataclass
class ResilienceResult:
    """Resilience metrics computed from the full session.

    Attributes:
        resilience_score: Normalized score [0, 1]; higher = faster recovery.
        average_recovery_s: Mean seconds to recover after a spike.
        completed_recoveries: Number of spikes that led to measurable recovery.
        pending_recoveries: Spikes still within recovery window.
        timeout_recoveries: Spikes where recovery exceeded MAX_RECOVERY_S.
    """

    resilience_score: float
    average_recovery_s: float
    completed_recoveries: int
    pending_recoveries: int
    timeout_recoveries: int


class ResilienceTracker:
    """Tracks post-spike stress recovery and computes resilience score.

    Resilience formula:
        resilience = 1 - clamp(average_recovery_s / MAX_RECOVERY_S, 0, 1)

    Recovery is defined as stress returning to within RECOVERY_SIGMA_THRESHOLD
    standard deviations above the session baseline mean.

    A spike is considered a timeout if recovery_time > MAX_RECOVERY_S.
    Timed-out recoveries are included in the average at MAX_RECOVERY_S,
    penalizing the resilience score.
    """

    MAX_RECOVERY_S: float = 60.0
    RECOVERY_SIGMA_THRESHOLD: float = 0.5

    def __init__(self) -> None:
        self._pending: list[_RecoveryRecord] = []
        self._completed: list[_RecoveryRecord] = []

    def on_spike(
        self,
        spike_stress: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """Register a new spike to track recovery from.

        Args:
            spike_stress: Stress level at spike peak [0, 1].
            timestamp: Spike timestamp; defaults to time.monotonic().
        """
        ts = timestamp if timestamp is not None else time.monotonic()
        self._pending.append(_RecoveryRecord(spike_timestamp=ts, spike_stress=spike_stress))

    def update(
        self,
        current_stress: float,
        baseline_mean: float,
        baseline_std: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """Evaluate pending recoveries against the current stress level.

        Args:
            current_stress: Current stress score [0, 1].
            baseline_mean: Session baseline mean stress.
            baseline_std: Session baseline stress standard deviation.
            timestamp: Current time; defaults to time.monotonic().
        """
        now = timestamp if timestamp is not None else time.monotonic()
        recovery_threshold = baseline_mean + self.RECOVERY_SIGMA_THRESHOLD * max(baseline_std, 1e-6)

        still_pending: list[_RecoveryRecord] = []
        for record in self._pending:
            elapsed = now - record.spike_timestamp
            if current_stress <= recovery_threshold:
                record.recovery_timestamp = now
                record.recovery_time_s = elapsed
                self._completed.append(record)
                logger.debug("Recovery completed in %.1fs", elapsed)
            elif elapsed >= self.MAX_RECOVERY_S:
                record.timed_out = True
                record.recovery_time_s = self.MAX_RECOVERY_S
                self._completed.append(record)
                logger.info("Recovery timed out after %.1fs", elapsed)
            else:
                still_pending.append(record)
        self._pending = still_pending

    def compute(self) -> ResilienceResult:
        """Compute the current resilience score.

        Returns:
            ResilienceResult with score and breakdown stats.
        """
        if not self._completed:
            return ResilienceResult(
                resilience_score=1.0,
                average_recovery_s=0.0,
                completed_recoveries=0,
                pending_recoveries=len(self._pending),
                timeout_recoveries=0,
            )

        times = [r.recovery_time_s for r in self._completed if r.recovery_time_s is not None]
        avg_recovery_s = sum(times) / len(times)
        resilience_score = max(0.0, 1.0 - (avg_recovery_s / self.MAX_RECOVERY_S))
        timeout_count = sum(1 for r in self._completed if r.timed_out)

        return ResilienceResult(
            resilience_score=round(resilience_score, 4),
            average_recovery_s=round(avg_recovery_s, 2),
            completed_recoveries=len(self._completed),
            pending_recoveries=len(self._pending),
            timeout_recoveries=timeout_count,
        )
