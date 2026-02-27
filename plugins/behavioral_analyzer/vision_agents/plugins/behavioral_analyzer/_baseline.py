import statistics
import time
from collections import deque
from dataclasses import dataclass

from ._types import AudioMetrics, VideoMetrics

_CALIBRATION_WINDOW_S = 60
_MIN_TURNS_FOR_BASELINE = 3
_MIN_SAMPLES = 5
_DEFAULT_WPM_STD = 25.0
_DEFAULT_DELAY_STD_FACTOR = 0.5


@dataclass
class BaselineStats:
    """Per-session behavioral baseline computed during calibration.

    All means and stds reflect the candidate's neutral/resting state,
    used to compute z-score deviations during the live session.
    """

    wpm_mean: float = 0.0
    wpm_std: float = _DEFAULT_WPM_STD
    filler_rate_mean: float = 0.0
    filler_rate_std: float = 1.0
    response_delay_mean_ms: float = 800.0
    response_delay_std_ms: float = 400.0
    posture_stability_mean: float = 0.8
    established: bool = False
    windows_recorded: int = 0
    turns_recorded: int = 0


class BaselineTracker:
    """Calibrates per-session behavioral baselines during the first ~60 seconds.

    Baseline is established when either:
    - 60 seconds have elapsed, OR
    - 3+ full conversation turns have been recorded
    AND at least 5 metric samples are available.

    Before baseline is established, the scorer falls back to absolute
    normalization thresholds, and confidence is capped at 0.5.
    """

    def __init__(self, calibration_window_s: float = _CALIBRATION_WINDOW_S) -> None:
        self._calibration_window_s = calibration_window_s
        self._start_time = time.monotonic()
        self._wpm_samples: deque[float] = deque(maxlen=20)
        self._filler_samples: deque[float] = deque(maxlen=20)
        self._delay_samples: deque[float] = deque(maxlen=20)
        self._posture_samples: deque[float] = deque(maxlen=20)
        self._stats = BaselineStats()

    @property
    def established(self) -> bool:
        return self._stats.established

    @property
    def stats(self) -> BaselineStats:
        return self._stats

    def record_window(self, audio: AudioMetrics, video: VideoMetrics) -> None:
        """Ingest one aggregated window into the calibration sample buffer.

        No-op once baseline is established.
        """
        if self._stats.established:
            return

        if audio.words_per_minute is not None:
            self._wpm_samples.append(audio.words_per_minute)
        if audio.filler_words_per_min is not None:
            self._filler_samples.append(audio.filler_words_per_min)
        if audio.avg_response_delay_ms is not None:
            self._delay_samples.append(audio.avg_response_delay_ms)
        if video.posture_stability_score is not None:
            self._posture_samples.append(video.posture_stability_score)

        self._stats.windows_recorded += 1
        self._try_establish()

    def record_turn(self) -> None:
        """Signal that a full conversation turn has completed."""
        self._stats.turns_recorded += 1
        self._try_establish()

    def z_score_normalized(self, value: float, mean: float, std: float) -> float:
        """Return z-score clamped to [0, 1], where positive z = above baseline.

        Used to detect deviations that indicate stress. Negative deviations
        (below baseline) are clamped to 0 to avoid negative contributions.
        """
        if std <= 0:
            return 0.0
        z = (value - mean) / std
        return min(max(z / 3.0, 0.0), 1.0)

    def _try_establish(self) -> None:
        elapsed = time.monotonic() - self._start_time
        enough_time = elapsed >= self._calibration_window_s
        enough_turns = self._stats.turns_recorded >= _MIN_TURNS_FOR_BASELINE
        enough_data = len(self._wpm_samples) >= _MIN_SAMPLES

        if (enough_time or enough_turns) and enough_data:
            self._compute_stats()

    def _compute_stats(self) -> None:
        wpm = list(self._wpm_samples)
        fillers = list(self._filler_samples)
        delays = list(self._delay_samples)
        postures = list(self._posture_samples)

        if wpm:
            self._stats.wpm_mean = statistics.mean(wpm)
            self._stats.wpm_std = statistics.stdev(wpm) if len(wpm) > 1 else _DEFAULT_WPM_STD
        if fillers:
            self._stats.filler_rate_mean = statistics.mean(fillers)
            self._stats.filler_rate_std = max(
                statistics.stdev(fillers) if len(fillers) > 1 else 1.0, 0.5
            )
        if delays:
            self._stats.response_delay_mean_ms = statistics.mean(delays)
            self._stats.response_delay_std_ms = max(
                statistics.stdev(delays) if len(delays) > 1 else delays[0] * _DEFAULT_DELAY_STD_FACTOR,
                100.0,
            )
        if postures:
            self._stats.posture_stability_mean = statistics.mean(postures)

        self._stats.established = True
