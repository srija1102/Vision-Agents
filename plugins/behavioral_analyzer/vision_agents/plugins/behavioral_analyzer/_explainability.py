"""Signal explainability and audit log generation.

Exposes which signals contributed most to stress and aggression scores,
their individual weights, and a per-window audit log. Designed for
transparency and bias review — no cross-session comparisons.

The weight maps here must stay in sync with _scorer.py.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Mirror of weights in _scorer.py — update both together.
_STRESS_WEIGHTS: dict[str, float] = {
    "response_delay": 0.20,
    "filler_rate": 0.15,
    "speech_rate_change": 0.12,
    "pitch_variance": 0.12,
    "silence_before_response": 0.10,
    "interruptions_received": 0.08,
    "posture_instability": 0.08,
    "volume_spikes": 0.05,
    "shoulder_tension": 0.05,
    "head_jitter": 0.03,
    "repetitive_movement": 0.02,
}

_AGGRESSION_WEIGHTS: dict[str, float] = {
    "interviewer_interruptions": 0.30,
    "dominance_imbalance": 0.25,
    "speaking_ratio_imbalance": 0.20,
    "rapid_questioning": 0.15,
    "volume_spikes": 0.10,
}


@dataclass
class SignalContribution:
    """Contribution of a single signal to the final score.

    Attributes:
        signal_name: Machine-readable signal identifier.
        raw_value: Normalised signal value [0, 1] used in scoring.
        weight: Fixed weight for this signal.
        weighted_contribution: raw_value × weight.
        percent_of_total: Fraction of the total weighted sum this explains.
    """

    signal_name: str
    raw_value: float
    weight: float
    weighted_contribution: float
    percent_of_total: float


@dataclass
class ExplainabilityReport:
    """Explainability output for a single analysis window.

    Attributes:
        window_id: Analysis window identifier.
        stress_score: Final candidate stress score.
        aggression_score: Final interviewer aggression score.
        top_stress_signals: Top 4 stress contributors, sorted descending.
        top_aggression_signals: Top 3 aggression contributors, sorted descending.
        confidence_breakdown: data_completeness, baseline_factor, signal_factor.
        audit_log: Full structured record for audit and bias review.
    """

    window_id: str
    stress_score: float
    aggression_score: float
    top_stress_signals: list[SignalContribution]
    top_aggression_signals: list[SignalContribution]
    confidence_breakdown: dict[str, float]
    audit_log: dict[str, object]


class Explainer:
    """Generates explainability reports for analysis windows.

    Reports expose the top contributing signals and a structured audit
    log so that candidates and reviewers can understand how scores were
    derived. Per-session normalization only — no cross-user comparison.

    Usage::

        explainer = Explainer()
        report = explainer.explain(
            window_id="w0042",
            stress_score=0.61,
            aggression_score=0.34,
            stress_signal_values={"response_delay": 0.8, "filler_rate": 0.6, ...},
            aggression_signal_values={"interviewer_interruptions": 0.5, ...},
            confidence_breakdown={"data_completeness": 0.9, "baseline_factor": 1.0},
        )
    """

    def explain(
        self,
        window_id: str,
        stress_score: float,
        aggression_score: float,
        stress_signal_values: dict[str, float],
        aggression_signal_values: dict[str, float],
        confidence_breakdown: dict[str, float],
    ) -> ExplainabilityReport:
        """Generate an explainability report for a single analysis window.

        Args:
            window_id: Window identifier.
            stress_score: Final stress score [0, 1].
            aggression_score: Final aggression score [0, 1].
            stress_signal_values: Raw signal values keyed by signal name.
            aggression_signal_values: Raw aggression signal values.
            confidence_breakdown: Confidence factor breakdown dict.

        Returns:
            ExplainabilityReport with top signals and audit log.
        """
        stress_contributions = self._compute_contributions(
            stress_signal_values, _STRESS_WEIGHTS
        )
        agg_contributions = self._compute_contributions(
            aggression_signal_values, _AGGRESSION_WEIGHTS
        )

        audit_log: dict[str, object] = {
            "window_id": window_id,
            "stress_score": stress_score,
            "aggression_score": aggression_score,
            "stress_signals": {
                k: v for k, v in stress_signal_values.items() if v is not None
            },
            "aggression_signals": {
                k: v for k, v in aggression_signal_values.items() if v is not None
            },
            "confidence": confidence_breakdown,
            "normalization_scope": "per_session_only",
            "cross_user_comparison": False,
        }

        return ExplainabilityReport(
            window_id=window_id,
            stress_score=stress_score,
            aggression_score=aggression_score,
            top_stress_signals=stress_contributions[:4],
            top_aggression_signals=agg_contributions[:3],
            confidence_breakdown=confidence_breakdown,
            audit_log=audit_log,
        )

    def _compute_contributions(
        self,
        signal_values: dict[str, float],
        weights: dict[str, float],
    ) -> list[SignalContribution]:
        contributions: list[SignalContribution] = []
        total_weighted = 0.0

        for signal, weight in weights.items():
            raw = signal_values.get(signal) or 0.0
            wc = raw * weight
            total_weighted += wc
            contributions.append(
                SignalContribution(
                    signal_name=signal,
                    raw_value=round(raw, 4),
                    weight=weight,
                    weighted_contribution=round(wc, 4),
                    percent_of_total=0.0,
                )
            )

        if total_weighted > 1e-9:
            for c in contributions:
                c.percent_of_total = round(c.weighted_contribution / total_weighted * 100.0, 2)

        contributions.sort(key=lambda c: c.weighted_contribution, reverse=True)
        return contributions
