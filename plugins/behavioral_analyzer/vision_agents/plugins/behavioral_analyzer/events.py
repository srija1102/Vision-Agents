from dataclasses import dataclass, field

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
