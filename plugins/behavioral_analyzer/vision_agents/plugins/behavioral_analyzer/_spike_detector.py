"""Micro-spike anomaly detector for real-time stress monitoring.

Detects sudden stress changes exceeding 2σ from the rolling 30-second mean.
Each spike is tagged with its most likely cause (interruption, silence, question).
"""

import logging
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class SpikeCause(str, Enum):
    """Contextual cause of a detected stress spike."""

    QUESTION_RECEIVED = "question_received"
    INTERRUPTION = "interruption"
    SILENCE = "silence"
    UNKNOWN = "unknown"


@dataclass
class SpikeEvent:
    """A single detected micro-spike in candidate stress.

    Attributes:
        timestamp: Monotonic time of detection.
        stress_before: Stress level in the window preceding the spike.
        stress_after: Stress level at spike peak.
        delta_sigma: Magnitude of the spike in standard deviations.
        cause: Most likely contextual cause.
        window_id: Analysis window identifier where spike was detected.
    """

    timestamp: float
    stress_before: float
    stress_after: float
    delta_sigma: float
    cause: SpikeCause
    window_id: str


@dataclass
class _StressReading:
    timestamp: float
    stress: float


class SpikeDetector:
    """1-second sliding window micro-spike anomaly detector.

    Maintains a rolling 30-second buffer of stress readings. When a new
    reading deviates by more than SIGMA_THRESHOLD standard deviations from
    the rolling mean, a SpikeEvent is recorded.

    Thread-safety: not thread-safe; access must be serialised by caller.
    """

    SIGMA_THRESHOLD: float = 2.0
    ROLLING_WINDOW_S: float = 30.0
    MIN_SAMPLES_FOR_DETECTION: int = 5

    def __init__(self) -> None:
        self._readings: deque[_StressReading] = deque()
        self._spikes: list[SpikeEvent] = []
        self._pending_cause: SpikeCause = SpikeCause.UNKNOWN
        self._last_reading: Optional[_StressReading] = None

    def hint_cause(self, cause: SpikeCause) -> None:
        """Signal the most recent contextual event for cause tagging.

        Call this when an interviewer question arrives, an interruption
        is detected, or a silence period begins. The cause is consumed
        on the next detected spike and then reset to UNKNOWN.

        Args:
            cause: The contextual cause to associate with the next spike.
        """
        self._pending_cause = cause

    def ingest(self, stress: float, window_id: str) -> Optional[SpikeEvent]:
        """Record a new stress value and check for a spike.

        Args:
            stress: Current stress score [0, 1].
            window_id: Current analysis window identifier.

        Returns:
            SpikeEvent if a spike is detected, else None.
        """
        now = time.monotonic()
        cutoff = now - self.ROLLING_WINDOW_S
        while self._readings and self._readings[0].timestamp < cutoff:
            self._readings.popleft()

        spike: Optional[SpikeEvent] = None

        if len(self._readings) >= self.MIN_SAMPLES_FOR_DETECTION:
            values = [r.stress for r in self._readings]
            n = len(values)
            mean = sum(values) / n
            variance = sum((v - mean) ** 2 for v in values) / n
            std = variance ** 0.5

            if std > 1e-6:
                delta_sigma = abs(stress - mean) / std
                if delta_sigma >= self.SIGMA_THRESHOLD:
                    prev_stress = self._last_reading.stress if self._last_reading else mean
                    spike = SpikeEvent(
                        timestamp=now,
                        stress_before=prev_stress,
                        stress_after=stress,
                        delta_sigma=delta_sigma,
                        cause=self._pending_cause,
                        window_id=window_id,
                    )
                    self._spikes.append(spike)
                    logger.info(
                        "Stress spike: delta_sigma=%.2f cause=%s window=%s",
                        delta_sigma,
                        self._pending_cause.value,
                        window_id,
                    )
                    self._pending_cause = SpikeCause.UNKNOWN

        reading = _StressReading(timestamp=now, stress=stress)
        self._readings.append(reading)
        self._last_reading = reading
        return spike

    def spike_history(self) -> list[SpikeEvent]:
        """Return all spike events detected in the session."""
        return list(self._spikes)

    def spike_count(self) -> int:
        """Return total number of spikes detected."""
        return len(self._spikes)

    def recent_spikes(self, within_s: float = 60.0) -> list[SpikeEvent]:
        """Return spikes that occurred within the past N seconds.

        Args:
            within_s: Lookback window in seconds.

        Returns:
            List of SpikeEvents within the window.
        """
        cutoff = time.monotonic() - within_s
        return [s for s in self._spikes if s.timestamp >= cutoff]
