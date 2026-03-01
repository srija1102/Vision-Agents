"""Historical session comparison engine.

Compares two SessionSummary objects and produces per-metric deltas,
percentage changes, and an overall improvement score.
Output is visualization-friendly (list of SessionDelta dataclasses).
"""

import logging
from dataclasses import dataclass

from ._session_summary import SessionSummary

logger = logging.getLogger(__name__)


@dataclass
class SessionDelta:
    """Per-metric change between two sessions.

    Attributes:
        metric: Human-readable metric name.
        session_1: Value in the baseline session.
        session_2: Value in the comparison session.
        delta_absolute: Arithmetic difference (session_2 - session_1).
        delta_percent: Percentage change relative to session_1.
        improved: True if the change is in the desired direction.
        direction_symbol: '↑' or '↓' for visualization.
    """

    metric: str
    session_1: float
    session_2: float
    delta_absolute: float
    delta_percent: float
    improved: bool
    direction_symbol: str


@dataclass
class SessionComparison:
    """Full comparison between two interview sessions.

    Attributes:
        session_1_id: Identifier for the baseline session.
        session_2_id: Identifier for the comparison session.
        deltas: Per-metric change breakdown.
        overall_improvement_score: Fraction of metrics that improved [0, 1].
        summary_lines: Human-readable comparison bullets.
    """

    session_1_id: str
    session_2_id: str
    deltas: list[SessionDelta]
    overall_improvement_score: float
    summary_lines: list[str]


class SessionComparisonEngine:
    """Compares two sessions and computes improvement metrics.

    Usage::

        engine = SessionComparisonEngine()
        comparison = engine.compare(baseline_summary, followup_summary)
        for line in comparison.summary_lines:
            print(line)
    """

    def compare(self, s1: SessionSummary, s2: SessionSummary) -> SessionComparison:
        """Compare two session summaries.

        Args:
            s1: Baseline (earlier) session.
            s2: Comparison (later) session.

        Returns:
            SessionComparison with per-metric deltas and improvement score.
        """
        deltas: list[SessionDelta] = [
            self._delta("average_stress", s1.average_stress, s2.average_stress, lower_is_better=True),
            self._delta("peak_stress", s1.peak_stress, s2.peak_stress, lower_is_better=True),
            self._delta(
                "resilience_score",
                s1.resilience_score,
                s2.resilience_score,
                lower_is_better=False,
            ),
            self._delta(
                "filler_word_density",
                s1.filler_word_density_per_min,
                s2.filler_word_density_per_min,
                lower_is_better=True,
            ),
            self._delta(
                "spike_count",
                float(s1.spike_count),
                float(s2.spike_count),
                lower_is_better=True,
            ),
        ]

        if s1.average_recovery_s > 0 or s2.average_recovery_s > 0:
            deltas.append(
                self._delta(
                    "average_recovery_s",
                    s1.average_recovery_s,
                    s2.average_recovery_s,
                    lower_is_better=True,
                )
            )

        if (
            s1.stress_difficulty_correlation is not None
            and s2.stress_difficulty_correlation is not None
        ):
            # Correlation closer to 0 is neither better nor worse; include as informational
            deltas.append(
                self._delta(
                    "stress_difficulty_correlation",
                    s1.stress_difficulty_correlation,
                    s2.stress_difficulty_correlation,
                    lower_is_better=True,  # lower r = less stress amplification by difficulty
                )
            )

        improved_count = sum(1 for d in deltas if d.improved)
        overall_score = round(improved_count / len(deltas), 4) if deltas else 0.5
        summary_lines = self._build_summary(deltas)

        return SessionComparison(
            session_1_id=s1.session_id,
            session_2_id=s2.session_id,
            deltas=deltas,
            overall_improvement_score=overall_score,
            summary_lines=summary_lines,
        )

    def _delta(
        self,
        metric: str,
        v1: float,
        v2: float,
        lower_is_better: bool,
    ) -> SessionDelta:
        abs_delta = v2 - v1
        pct = (abs_delta / v1 * 100.0) if abs(v1) > 1e-9 else 0.0
        improved = (abs_delta < 0) if lower_is_better else (abs_delta > 0)
        direction = "↓" if abs_delta < 0 else "↑"
        return SessionDelta(
            metric=metric,
            session_1=round(v1, 4),
            session_2=round(v2, 4),
            delta_absolute=round(abs_delta, 4),
            delta_percent=round(pct, 2),
            improved=improved,
            direction_symbol=direction,
        )

    def _build_summary(self, deltas: list[SessionDelta]) -> list[str]:
        lines = []
        for d in deltas:
            status = "✓" if d.improved else "✗"
            lines.append(
                f"{status} {d.metric}: {d.session_1:.3f} → {d.session_2:.3f} "
                f"({d.direction_symbol}{abs(d.delta_percent):.1f}%)"
            )
        return lines
