from dataclasses import dataclass, field
from typing import Optional

from vision_agents.core.events.base import PluginBaseEvent


@dataclass
class BehavioralAnalysisEvent(PluginBaseEvent):
    """Emitted every 5 seconds with the latest behavioral analysis results.

    All numeric fields are in [0, 1]. A value of None indicates insufficient
    data for reliable estimation (typically during the calibration window).
    """

    type: str = field(default="plugin.behavioral_analyzer.analysis", init=False)

    window_id: str = ""

    candidate_stress: float | None = None
    """Estimated candidate stress level. None = insufficient data."""

    interviewer_aggression: float | None = None
    """Estimated interviewer aggression level. None = insufficient data."""

    pressure_trend: str = "stable"
    """Trend direction: 'increasing', 'stable', or 'decreasing'."""

    stress_confidence: float = 0.0
    """Confidence in candidate_stress estimate [0, 1]."""

    aggression_confidence: float = 0.0
    """Confidence in interviewer_aggression estimate [0, 1]."""

    dominant_party: str = "balanced"
    """Dominant conversational party: 'candidate', 'interviewer', or 'balanced'."""

    imbalance_score: float = 0.0
    """Magnitude of speaking-time imbalance [0, 1]."""

    key_signals: list[str] = field(default_factory=list)
    """Top contributing signal names (max 4)."""

    coaching_feedback: str = ""
    """Short actionable coaching message for the candidate (max 20 words)."""

    risk_flag: str = "none"
    """Risk level: 'none', 'moderate', or 'high'."""

    aggression_class: str = "neutral"
    """Aggression style: 'neutral', 'challenging', 'rapid_fire', 'interruptive', or 'hostile'."""

    aggression_class_confidence: float = 0.0
    """Confidence in the aggression classification [0, 1]."""


@dataclass
class SpikeDetectedEvent(PluginBaseEvent):
    """Emitted immediately when a micro-spike in candidate stress is detected.

    A spike is defined as a change > 2σ from the rolling 30-second mean.
    """

    type: str = field(default="plugin.behavioral_analyzer.spike_detected", init=False)

    window_id: str = ""
    """Window in which the spike was detected."""

    stress_before: float = 0.0
    """Stress level immediately before the spike."""

    stress_after: float = 0.0
    """Stress level at spike peak."""

    delta_sigma: float = 0.0
    """Spike magnitude in standard deviations from the rolling mean."""

    cause: str = "unknown"
    """Most likely contextual cause: 'question_received', 'interruption', 'silence', 'unknown'."""


@dataclass
class SessionSummaryEvent(PluginBaseEvent):
    """Emitted once at the end of a session with the complete summary.

    Consumers can use this to persist, display, or export the session report.
    """

    type: str = field(default="plugin.behavioral_analyzer.session_summary", init=False)

    session_id: str = ""
    peak_stress: float = 0.0
    average_stress: float = 0.0
    resilience_score: float = 1.0
    average_recovery_s: float = 0.0
    spike_count: int = 0
    dominant_aggression_class: str = "neutral"
    filler_word_density_per_min: float = 0.0
    dominant_party_summary: str = "balanced"
    stress_difficulty_correlation: Optional[float] = None
    strengths: list[str] = field(default_factory=list)
    improvement_areas: list[str] = field(default_factory=list)
    improvement_plan: list[str] = field(default_factory=list)
