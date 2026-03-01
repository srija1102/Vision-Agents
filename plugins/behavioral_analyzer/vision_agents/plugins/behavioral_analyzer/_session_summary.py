"""End-of-session summary generator.

Accumulates per-window analysis results throughout a session and produces
a structured SessionSummary with coaching plan, strengths, and improvement
areas. Outputs as a typed dataclass, JSON string, or Markdown report.
"""

import dataclasses
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from ._resilience import ResilienceResult
from ._question_analyzer import CorrelationResult
from ._spike_detector import SpikeEvent

logger = logging.getLogger(__name__)


@dataclass
class WindowRecord:
    """Snapshot of a single analysis window for session aggregation.

    Attributes:
        window_id: Identifier for the 5-second analysis window.
        timestamp: Monotonic time the window was emitted.
        candidate_stress: Stress score [0, 1] or None.
        interviewer_aggression: Aggression score [0, 1] or None.
        aggression_class: Aggression style classification string.
        risk_flag: 'none', 'moderate', or 'high'.
        coaching_feedback: Short coaching message from that window.
        dominant_party: 'candidate', 'interviewer', or 'balanced'.
    """

    window_id: str
    timestamp: float
    candidate_stress: float
    interviewer_aggression: float
    aggression_class: str
    risk_flag: str
    coaching_feedback: str
    dominant_party: str


@dataclass
class SessionSummary:
    """Complete end-of-session behavioral analysis summary.

    Fields are organized by analysis category. window_records is excluded
    from the default JSON output to avoid verbosity — call to_json(verbose=True)
    to include it.
    """

    session_id: str
    duration_s: float

    # Stress
    peak_stress: float
    peak_stress_timestamp: float
    average_stress: float

    # Aggression
    aggression_spike_window_ids: list[str]
    dominant_aggression_class: str

    # Resilience
    resilience_score: float
    average_recovery_s: float
    spike_count: int
    spike_causes: dict[str, int]

    # Conversation dominance
    dominant_party_summary: str

    # Linguistic
    filler_word_density_per_min: float

    # Question difficulty correlation
    stress_difficulty_correlation: Optional[float]
    stress_by_question_type: dict[str, float]

    # Coaching
    strengths: list[str]
    improvement_areas: list[str]
    improvement_plan: list[str]

    # Raw windows (excluded from default JSON)
    window_records: list[WindowRecord] = field(default_factory=list)


class SessionSummaryGenerator:
    """Accumulates session data and generates a SessionSummary on demand.

    Call record_window() after each BehavioralAnalysisEvent. Call generate()
    at the end of the session.

    Args:
        session_id: Unique identifier for the session.
    """

    def __init__(self, session_id: str) -> None:
        self._session_id = session_id
        self._windows: list[WindowRecord] = []
        self._spikes: list[SpikeEvent] = []
        self._total_fillers: int = 0
        self._session_start: Optional[float] = None
        self._session_end: Optional[float] = None
        self._resilience: Optional[ResilienceResult] = None
        self._correlation: Optional[CorrelationResult] = None

    def record_window(self, window: WindowRecord) -> None:
        """Add an analysis window to the session history.

        Args:
            window: WindowRecord from the most recent analysis cycle.
        """
        if self._session_start is None:
            self._session_start = window.timestamp
        self._session_end = window.timestamp
        self._windows.append(window)

    def record_spikes(self, spikes: list[SpikeEvent]) -> None:
        """Update the spike history from the SpikeDetector.

        Args:
            spikes: Full spike history from the current session.
        """
        self._spikes = list(spikes)

    def record_filler_total(self, total: int) -> None:
        """Record cumulative filler word count for the session.

        Args:
            total: Total filler words detected since session start.
        """
        self._total_fillers = total

    def record_resilience(self, result: ResilienceResult) -> None:
        """Update resilience computation result.

        Args:
            result: Latest ResilienceResult from ResilienceTracker.
        """
        self._resilience = result

    def record_correlation(self, result: Optional[CorrelationResult]) -> None:
        """Update question-difficulty correlation result.

        Args:
            result: Latest CorrelationResult or None if insufficient data.
        """
        self._correlation = result

    def generate(self) -> SessionSummary:
        """Generate the complete session summary.

        Returns:
            SessionSummary dataclass with all computed metrics.

        Raises:
            ValueError: If no windows have been recorded.
        """
        if not self._windows:
            raise ValueError("No windows recorded — cannot generate summary")

        start = self._session_start or 0.0
        end = self._session_end or start
        duration_s = max(end - start, 1.0)

        stress_values = [w.candidate_stress for w in self._windows]
        peak_stress = max(stress_values)
        peak_window = max(self._windows, key=lambda w: w.candidate_stress)
        average_stress = sum(stress_values) / len(stress_values)

        aggression_spike_ids = [
            w.window_id for w in self._windows if w.risk_flag in ("moderate", "high")
        ]

        class_counts: dict[str, int] = {}
        for w in self._windows:
            class_counts[w.aggression_class] = class_counts.get(w.aggression_class, 0) + 1
        dominant_agg = max(class_counts, key=lambda k: class_counts[k]) if class_counts else "neutral"

        party_counts: dict[str, int] = {}
        for w in self._windows:
            party_counts[w.dominant_party] = party_counts.get(w.dominant_party, 0) + 1
        dominant_party = max(party_counts, key=lambda k: party_counts[k]) if party_counts else "balanced"

        filler_density = (self._total_fillers / duration_s) * 60.0

        spike_causes: dict[str, int] = {}
        for spike in self._spikes:
            cause = spike.cause.value
            spike_causes[cause] = spike_causes.get(cause, 0) + 1

        resilience_score = self._resilience.resilience_score if self._resilience else 1.0
        avg_recovery_s = self._resilience.average_recovery_s if self._resilience else 0.0

        stress_diff_r = self._correlation.pearson_r if self._correlation else None
        stress_by_q = self._correlation.stress_by_complexity_bucket if self._correlation else {}

        strengths, improvements, plan = self._derive_coaching(
            average_stress=average_stress,
            peak_stress=peak_stress,
            resilience_score=resilience_score,
            filler_density=filler_density,
            dominant_aggression_class=dominant_agg,
        )

        return SessionSummary(
            session_id=self._session_id,
            duration_s=round(duration_s, 1),
            peak_stress=round(peak_stress, 4),
            peak_stress_timestamp=peak_window.timestamp,
            average_stress=round(average_stress, 4),
            aggression_spike_window_ids=aggression_spike_ids,
            dominant_aggression_class=dominant_agg,
            resilience_score=round(resilience_score, 4),
            average_recovery_s=round(avg_recovery_s, 2),
            spike_count=len(self._spikes),
            spike_causes=spike_causes,
            dominant_party_summary=dominant_party,
            filler_word_density_per_min=round(filler_density, 2),
            stress_difficulty_correlation=stress_diff_r,
            stress_by_question_type=stress_by_q,
            strengths=strengths,
            improvement_areas=improvements,
            improvement_plan=plan,
            window_records=list(self._windows),
        )

    def _derive_coaching(
        self,
        average_stress: float,
        peak_stress: float,
        resilience_score: float,
        filler_density: float,
        dominant_aggression_class: str,
    ) -> tuple[list[str], list[str], list[str]]:
        strengths: list[str] = []
        improvements: list[str] = []
        plan: list[str] = []

        if average_stress < 0.35:
            strengths.append("Maintained composure throughout the session")
        if resilience_score > 0.70:
            strengths.append("Recovered quickly after high-pressure moments")
        if filler_density < 2.0:
            strengths.append("Minimal filler words — clear and direct articulation")
        if peak_stress < 0.50:
            strengths.append("No extreme stress peaks detected")

        if average_stress > 0.60:
            improvements.append("Sustained elevated stress throughout the session")
            plan.append("Practice grounding techniques: box breathing before high-stakes answers")
        if peak_stress > 0.80:
            improvements.append("Extreme stress spikes under pressure questions")
            plan.append("Rehearse adversarial interview scenarios with a timer constraint")
        if resilience_score < 0.40:
            improvements.append("Slow recovery after high-pressure moments")
            plan.append("Build resilience through repeated mock adversarial practice sessions")
        if filler_density > 5.0:
            improvements.append("High frequency of filler words (um, uh, like, you know)")
            plan.append("Record and review speech; replace fillers with deliberate pauses")
        if dominant_aggression_class in ("rapid_fire", "hostile"):
            plan.append(
                "Practice acknowledging and calmly redirecting aggressive questioning styles"
            )

        return strengths, improvements, plan

    def to_json(self, summary: SessionSummary, verbose: bool = False) -> str:
        """Serialize the summary to a JSON string.

        Args:
            summary: SessionSummary to serialize.
            verbose: If True, includes window_records in the output.

        Returns:
            JSON string.
        """
        d = dataclasses.asdict(summary)
        if not verbose:
            d.pop("window_records", None)
        return json.dumps(d, indent=2, default=str)

    def to_markdown(self, summary: SessionSummary) -> str:
        """Generate a human-readable Markdown report.

        Args:
            summary: SessionSummary to format.

        Returns:
            Multi-line Markdown string.
        """
        lines = [
            f"# Interview Session Report — `{summary.session_id}`",
            "",
            f"**Duration**: {summary.duration_s:.0f}s",
            "",
            "## Stress Analysis",
            f"- **Peak Stress**: {summary.peak_stress:.2f}",
            f"- **Average Stress**: {summary.average_stress:.2f}",
            f"- **Spike Count**: {summary.spike_count}",
        ]

        if summary.spike_causes:
            lines.append("- **Spike Causes**:")
            for cause, count in summary.spike_causes.items():
                lines.append(f"  - {cause}: {count}")

        lines += [
            "",
            "## Resilience",
            f"- **Score**: {summary.resilience_score:.2f} / 1.00",
            f"- **Avg Recovery Time**: {summary.average_recovery_s:.1f}s",
            "",
            "## Interviewer Pressure",
            f"- **Style**: {summary.dominant_aggression_class}",
            f"- **High-Pressure Windows**: {len(summary.aggression_spike_window_ids)}",
            "",
            "## Communication",
            f"- **Filler Word Rate**: {summary.filler_word_density_per_min:.1f}/min",
            f"- **Dominant Party**: {summary.dominant_party_summary}",
        ]

        if summary.stress_difficulty_correlation is not None:
            lines += [
                "",
                "## Question Difficulty vs Stress",
                f"- **Pearson r**: {summary.stress_difficulty_correlation:.3f}",
            ]
            for bucket, avg in summary.stress_by_question_type.items():
                lines.append(f"  - {bucket} complexity: avg stress {avg:.2f}")

        if summary.strengths:
            lines += ["", "## Strengths", *[f"- {s}" for s in summary.strengths]]

        if summary.improvement_areas:
            lines += [
                "",
                "## Areas for Improvement",
                *[f"- {s}" for s in summary.improvement_areas],
            ]

        if summary.improvement_plan:
            lines += [
                "",
                "## Improvement Plan",
                *[f"{i + 1}. {p}" for i, p in enumerate(summary.improvement_plan)],
            ]

        return "\n".join(lines)
