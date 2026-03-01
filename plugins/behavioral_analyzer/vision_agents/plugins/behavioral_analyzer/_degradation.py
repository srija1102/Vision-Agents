"""Graceful degradation policy for partial signal availability.

When data sources (STT, video, LLM) become unavailable, this module
determines how to adjust confidence caps and which processing stages
to skip, rather than failing or silently producing misleading scores.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SignalAvailability:
    """Flags indicating which data sources are currently available.

    Attributes:
        audio_available: Raw audio track is being received.
        video_available: Video track is being received.
        stt_available: STT service is producing transcripts.
        llm_available: LLM service is reachable and responding.
    """

    audio_available: bool = True
    video_available: bool = True
    stt_available: bool = True
    llm_available: bool = True


@dataclass
class DegradationPolicy:
    """Analysis policy derived from current signal availability.

    Attributes:
        should_run_llm: Whether to attempt the LLM refinement call.
        confidence_cap: Maximum confidence value to report [0, 1].
        active_signal_categories: Which signal categories are usable.
        warnings: Human-readable descriptions of active degradations.
    """

    should_run_llm: bool
    confidence_cap: float
    active_signal_categories: list[str]
    warnings: list[str] = field(default_factory=list)

    @property
    def warning_message(self) -> str:
        """Concatenated warning summary for logging."""
        return "; ".join(self.warnings) if self.warnings else "All signals nominal"


class GracefulDegradationEngine:
    """Evaluates signal availability and produces a DegradationPolicy.

    The policy tells the analysis loop which stages to run, how to cap
    the reported confidence, and what to log. No stage silently fails —
    the policy is always explicit about what is missing.

    Decision tree (confidence caps are cumulative):
        - No audio           → cap 0.45 (major signal loss)
        - Audio, no STT      → cap 0.55 (transcript signals dropped)
        - No video           → cap 0.70 (pose signals dropped)
        - No LLM             → cap 0.75 (deterministic only)
        - <2 signal cats     → cap 0.30 (critically data-sparse)

    Usage::

        engine = GracefulDegradationEngine()
        availability = SignalAvailability(stt_available=False)
        policy = engine.evaluate(availability)
        if not policy.should_run_llm:
            result = deterministic_fallback(pre)
    """

    def evaluate(self, availability: SignalAvailability) -> DegradationPolicy:
        """Derive a degradation policy from current signal availability.

        Args:
            availability: Which data sources are currently reachable.

        Returns:
            DegradationPolicy with instructions for the analysis loop.
        """
        active: list[str] = []
        confidence_cap = 1.0
        warnings: list[str] = []

        if availability.audio_available and availability.stt_available:
            active += ["wpm", "filler_rate", "response_delay", "interruptions"]
        elif availability.audio_available:
            # Audio present but STT offline → volume and pitch only
            active += ["volume_spikes", "pitch_variance"]
            confidence_cap = min(confidence_cap, 0.55)
            warnings.append("STT unavailable — transcript-based signals dropped")
        else:
            confidence_cap = min(confidence_cap, 0.45)
            warnings.append("Audio unavailable — all audio signals dropped")

        if availability.video_available:
            active += ["posture_stability", "shoulder_tension", "head_jitter"]
        else:
            confidence_cap = min(confidence_cap, 0.70)
            warnings.append("Video unavailable — pose signals dropped")

        should_run_llm = availability.llm_available
        if not availability.llm_available:
            confidence_cap = min(confidence_cap, 0.75)
            warnings.append("LLM unavailable — using deterministic scoring only")

        if len(active) < 2:
            confidence_cap = min(confidence_cap, 0.30)
            warnings.append(
                f"Critically low signal count ({len(active)}) — confidence capped severely"
            )

        if warnings:
            logger.warning("Degradation active: %s", "; ".join(warnings))

        return DegradationPolicy(
            should_run_llm=should_run_llm,
            confidence_cap=round(confidence_cap, 2),
            active_signal_categories=active,
            warnings=warnings,
        )
