from dataclasses import dataclass, field
from typing import Literal


@dataclass
class AudioMetrics:
    """Audio-derived behavioral metrics for a 3–5 second observation window.

    All fields are optional — None means the metric was not available
    in this window (e.g. no speech occurred).
    """

    words_per_minute: float | None = None
    filler_words_count: int | None = None
    filler_words_per_min: float | None = None
    avg_response_delay_ms: float | None = None
    max_response_delay_ms: float | None = None
    speech_rate_change_percent: float | None = None
    pitch_variance_score: float | None = None
    volume_spike_count: int | None = None
    silence_duration_before_response_ms: float | None = None
    interruption_count_received: int | None = None
    interruption_count_given: int | None = None


@dataclass
class VideoMetrics:
    """Video/pose-derived behavioral metrics for a 3–5 second observation window.

    Scores are in [0, 1]. None means the metric was not available
    (e.g. no video track, or pose not detected).
    """

    posture_stability_score: float | None = None
    shoulder_tension_score: float | None = None
    head_jitter_score: float | None = None
    slouch_score: float | None = None
    repetitive_hand_movement_score: float | None = None
    lean_forward_frequency: float | None = None
    lean_backward_frequency: float | None = None
    gaze_stability_score: float | None = None


@dataclass
class ConversationMetrics:
    """Turn and conversation dynamics metrics for a session window."""

    speaking_time_ratio_candidate: float | None = None
    speaking_time_ratio_interviewer: float | None = None
    total_interruptions_by_interviewer: int | None = None
    total_interruptions_by_candidate: int | None = None
    average_question_length: float | None = None
    dominance_score: float | None = None


@dataclass
class HistoricalContext:
    """Rolling historical metrics used for trend and drift detection.

    Covers approximately the last 30–60 seconds of session history.
    """

    rolling_avg_stress_estimate: float | None = None
    rolling_avg_aggression_estimate: float | None = None
    stress_trend_delta: float | None = None
    aggression_trend_delta: float | None = None
    baseline_established: bool = False


@dataclass
class BehavioralPayload:
    """Complete aggregated payload for one analysis window."""

    window_id: str
    window_start_ms: int
    window_end_ms: int
    audio: AudioMetrics
    video: VideoMetrics
    conversation: ConversationMetrics
    history: HistoricalContext


@dataclass
class DominanceAnalysis:
    """Conversational dominance breakdown between candidate and interviewer."""

    dominant_party: Literal["candidate", "interviewer", "balanced"]
    imbalance_score: float


@dataclass
class AnalysisResult:
    """Structured output from one behavioral analysis cycle."""

    candidate_stress: float | None
    interviewer_aggression: float | None
    pressure_trend: Literal["increasing", "stable", "decreasing"]
    stress_confidence: float
    aggression_confidence: float
    dominance_analysis: DominanceAnalysis
    key_contributing_signals: list[str]
    coaching_feedback: str
    risk_flag: Literal["none", "moderate", "high"]


@dataclass
class PrecomputedScores:
    """Deterministic pre-scores passed to the LLM as anchors."""

    stress_score: float
    aggression_score: float
    stress_confidence: float
    aggression_confidence: float
    active_stress_signals: list[str] = field(default_factory=list)
    active_aggression_signals: list[str] = field(default_factory=list)
    dominant_party: str = "balanced"
    imbalance_score: float = 0.0
