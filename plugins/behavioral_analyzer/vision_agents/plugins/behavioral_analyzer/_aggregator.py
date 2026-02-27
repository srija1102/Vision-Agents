"""5-second sliding window aggregator for behavioral metrics.

Raw event data (word counts, pose scores, turn durations) is ingested
continuously and flushed into a BehavioralPayload every analysis cycle.
A rolling deque of stress/aggression estimates tracks temporal trends.
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field

from ._types import (
    AudioMetrics,
    BehavioralPayload,
    ConversationMetrics,
    HistoricalContext,
    VideoMetrics,
)

_HISTORY_SIZE = 12          # 12 × 5s = 60s rolling history
_TREND_LOOKBACK = 6         # compare now vs 30s ago for delta


@dataclass
class _WindowBuffer:
    """Mutable accumulator for raw events within one 5-second window."""

    word_counts: list[int] = field(default_factory=list)
    filler_counts: list[int] = field(default_factory=list)
    response_delays_ms: list[float] = field(default_factory=list)
    pitch_scores: list[float] = field(default_factory=list)
    volume_spikes: int = 0
    silences_ms: list[float] = field(default_factory=list)
    interruptions_received: int = 0
    interruptions_given: int = 0
    speech_rate_changes: list[float] = field(default_factory=list)

    posture_scores: list[float] = field(default_factory=list)
    shoulder_scores: list[float] = field(default_factory=list)
    jitter_scores: list[float] = field(default_factory=list)
    slouch_scores: list[float] = field(default_factory=list)
    hand_movement_scores: list[float] = field(default_factory=list)
    gaze_scores: list[float] = field(default_factory=list)
    lean_forward_count: int = 0
    lean_backward_count: int = 0

    candidate_speaking_ms: float = 0.0
    interviewer_speaking_ms: float = 0.0
    interviewer_interruptions: int = 0
    candidate_interruptions: int = 0
    question_word_counts: list[int] = field(default_factory=list)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


class MetricsAggregator:
    """Thread-safe aggregator that batches raw events into 5-second windows.

    Callers feed data via `ingest_audio_event`, `ingest_video_event`, and
    `ingest_turn_event`. On each analysis cycle, `flush_window` is called to
    atomically snapshot the buffer, reset it, and return the aggregated payload.
    """

    def __init__(self, window_size_s: float = 5.0) -> None:
        self._window_size_s = window_size_s
        self._buffer = _WindowBuffer()
        self._window_start = time.monotonic()
        self._window_id = 0
        self._stress_history: deque[float] = deque(maxlen=_HISTORY_SIZE)
        self._aggression_history: deque[float] = deque(maxlen=_HISTORY_SIZE)
        self._lock = asyncio.Lock()

    async def ingest_audio_event(
        self,
        words_this_window: int | None = None,
        filler_count: int | None = None,
        response_delay_ms: float | None = None,
        pitch_variance: float | None = None,
        volume_spike: bool = False,
        silence_ms: float | None = None,
        interrupted: bool = False,
        speech_rate_change_pct: float | None = None,
    ) -> None:
        """Ingest one audio observation into the current window buffer."""
        async with self._lock:
            buf = self._buffer
            if words_this_window is not None:
                buf.word_counts.append(words_this_window)
            if filler_count is not None:
                buf.filler_counts.append(filler_count)
            if response_delay_ms is not None:
                buf.response_delays_ms.append(response_delay_ms)
            if pitch_variance is not None:
                buf.pitch_scores.append(pitch_variance)
            if volume_spike:
                buf.volume_spikes += 1
            if silence_ms is not None:
                buf.silences_ms.append(silence_ms)
            if interrupted:
                buf.interruptions_received += 1
            if speech_rate_change_pct is not None:
                buf.speech_rate_changes.append(speech_rate_change_pct)

    async def ingest_video_event(
        self,
        posture_stability: float | None = None,
        shoulder_tension: float | None = None,
        head_jitter: float | None = None,
        slouch: float | None = None,
        hand_movement: float | None = None,
        gaze_stability: float | None = None,
        lean_forward: bool = False,
        lean_backward: bool = False,
    ) -> None:
        """Ingest one video/pose observation into the current window buffer."""
        async with self._lock:
            buf = self._buffer
            if posture_stability is not None:
                buf.posture_scores.append(posture_stability)
            if shoulder_tension is not None:
                buf.shoulder_scores.append(shoulder_tension)
            if head_jitter is not None:
                buf.jitter_scores.append(head_jitter)
            if slouch is not None:
                buf.slouch_scores.append(slouch)
            if hand_movement is not None:
                buf.hand_movement_scores.append(hand_movement)
            if gaze_stability is not None:
                buf.gaze_scores.append(gaze_stability)
            if lean_forward:
                buf.lean_forward_count += 1
            if lean_backward:
                buf.lean_backward_count += 1

    async def ingest_turn_event(
        self,
        candidate_speaking_ms: float | None = None,
        interviewer_speaking_ms: float | None = None,
        interviewer_interrupted: bool = False,
        candidate_interrupted: bool = False,
        question_word_count: int | None = None,
    ) -> None:
        """Ingest one conversation-turn observation into the current window buffer."""
        async with self._lock:
            buf = self._buffer
            if candidate_speaking_ms is not None:
                buf.candidate_speaking_ms += candidate_speaking_ms
            if interviewer_speaking_ms is not None:
                buf.interviewer_speaking_ms += interviewer_speaking_ms
            if interviewer_interrupted:
                buf.interviewer_interruptions += 1
            if candidate_interrupted:
                buf.candidate_interruptions += 1
            if question_word_count is not None:
                buf.question_word_counts.append(question_word_count)

    def update_rolling_estimates(self, stress: float, aggression: float) -> None:
        """Push the latest analysis result into the rolling history deques.

        Called after each analysis cycle so that trend deltas are up to date
        before the next window is flushed.
        """
        self._stress_history.append(stress)
        self._aggression_history.append(aggression)

    async def flush_window(self) -> BehavioralPayload:
        """Atomically snapshot the current window buffer and reset it.

        Computes all derived metrics (WPM, speaking ratios, trend deltas)
        from the accumulated raw observations and returns a BehavioralPayload.
        """
        async with self._lock:
            now = time.monotonic()
            buf = self._buffer
            elapsed_min = self._window_size_s / 60.0

            total_words = sum(buf.word_counts)
            wpm = (total_words / elapsed_min) if total_words > 0 else None

            total_fillers = sum(buf.filler_counts)
            filler_per_min = (total_fillers / elapsed_min) if total_fillers > 0 else None

            total_speaking = buf.candidate_speaking_ms + buf.interviewer_speaking_ms
            if total_speaking > 0:
                candidate_ratio: float | None = buf.candidate_speaking_ms / total_speaking
                interviewer_ratio: float | None = buf.interviewer_speaking_ms / total_speaking
            else:
                candidate_ratio = None
                interviewer_ratio = None

            audio = AudioMetrics(
                words_per_minute=wpm,
                filler_words_count=total_fillers if buf.filler_counts else None,
                filler_words_per_min=filler_per_min,
                avg_response_delay_ms=_mean(buf.response_delays_ms),
                max_response_delay_ms=max(buf.response_delays_ms) if buf.response_delays_ms else None,
                speech_rate_change_percent=_mean(buf.speech_rate_changes),
                pitch_variance_score=_mean(buf.pitch_scores),
                volume_spike_count=buf.volume_spikes if buf.volume_spikes > 0 else None,
                silence_duration_before_response_ms=_mean(buf.silences_ms),
                interruption_count_received=buf.interruptions_received or None,
                interruption_count_given=buf.interruptions_given or None,
            )

            video = VideoMetrics(
                posture_stability_score=_mean(buf.posture_scores),
                shoulder_tension_score=_mean(buf.shoulder_scores),
                head_jitter_score=_mean(buf.jitter_scores),
                slouch_score=_mean(buf.slouch_scores),
                repetitive_hand_movement_score=_mean(buf.hand_movement_scores),
                lean_forward_frequency=(
                    buf.lean_forward_count / self._window_size_s
                    if buf.lean_forward_count > 0 else None
                ),
                lean_backward_frequency=(
                    buf.lean_backward_count / self._window_size_s
                    if buf.lean_backward_count > 0 else None
                ),
                gaze_stability_score=_mean(buf.gaze_scores),
            )

            conversation = ConversationMetrics(
                speaking_time_ratio_candidate=candidate_ratio,
                speaking_time_ratio_interviewer=interviewer_ratio,
                total_interruptions_by_interviewer=buf.interviewer_interruptions or None,
                total_interruptions_by_candidate=buf.candidate_interruptions or None,
                average_question_length=_mean([float(x) for x in buf.question_word_counts]),
                dominance_score=candidate_ratio,
            )

            stress_list = list(self._stress_history)
            aggression_list = list(self._aggression_history)

            stress_delta: float | None = None
            if len(stress_list) >= _TREND_LOOKBACK:
                stress_delta = stress_list[-1] - stress_list[-_TREND_LOOKBACK]

            aggression_delta: float | None = None
            if len(aggression_list) >= _TREND_LOOKBACK:
                aggression_delta = aggression_list[-1] - aggression_list[-_TREND_LOOKBACK]

            history = HistoricalContext(
                rolling_avg_stress_estimate=(
                    sum(stress_list) / len(stress_list) if stress_list else None
                ),
                rolling_avg_aggression_estimate=(
                    sum(aggression_list) / len(aggression_list) if aggression_list else None
                ),
                stress_trend_delta=stress_delta,
                aggression_trend_delta=aggression_delta,
                baseline_established=len(stress_list) >= _TREND_LOOKBACK,
            )

            payload = BehavioralPayload(
                window_id=f"w{self._window_id:04d}",
                window_start_ms=int(self._window_start * 1000),
                window_end_ms=int(now * 1000),
                audio=audio,
                video=video,
                conversation=conversation,
                history=history,
            )

            self._buffer = _WindowBuffer()
            self._window_start = now
            self._window_id += 1

            return payload
