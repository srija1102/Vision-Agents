"""Deterministic feature scoring for behavioral signal fusion.

Design principles:
- Each signal is normalized to [0, 1] independently.
- Weights are summed to produce a raw score.
- A multi-signal gate reduces the score when too few signals are active,
  preventing single-feature false positives.
- Baseline-adjusted z-scores are used when available; absolute thresholds
  are the fallback during the calibration window.
- Confidence reflects data completeness, baseline quality, and signal count.
"""

from ._baseline import BaselineTracker
from ._types import (
    AudioMetrics,
    ConversationMetrics,
    DominanceAnalysis,
    PrecomputedScores,
    VideoMetrics,
)

# ── Absolute normalization ceilings (pre-baseline fallback) ────────────────────

_DELAY_MAX_MS = 3000.0
_FILLER_RATE_MAX = 20.0
_SPEECH_CHANGE_MAX_PCT = 50.0
_SILENCE_MAX_MS = 5000.0
_INTERRUPT_MAX = 5.0
_VOLUME_SPIKE_MAX = 5.0
_QUESTION_LEN_CALM = 20.0
_DOMINANCE_THRESHOLD = 0.60

# ── Signal activation threshold ────────────────────────────────────────────────
# A signal must exceed this to count as "active" for the multi-signal gate.
_SIGNAL_ACTIVE_THRESHOLD = 0.30

# ── Multi-signal gate thresholds ───────────────────────────────────────────────
# Stress must not fire on a single signal — require corroboration.
_GATE_FULL_SIGNALS = 3      # full score
_GATE_PARTIAL_SIGNALS = 2   # 75% of raw score
# Below 2 active signals → 50% of raw score (caps ~0.4 for typical inputs)

# ── Stress signal weights (must sum to 1.0) ────────────────────────────────────
_STRESS_WEIGHTS: dict[str, float] = {
    "response_delay":         0.20,
    "filler_rate":            0.15,
    "speech_rate_change":     0.12,
    "pitch_variance":         0.12,
    "silence_before_response": 0.10,
    "interruptions_received": 0.08,
    "volume_spikes":          0.05,
    "posture_instability":    0.08,
    "shoulder_tension":       0.05,
    "head_jitter":            0.03,
    "repetitive_movement":    0.02,
}

assert abs(sum(_STRESS_WEIGHTS.values()) - 1.0) < 1e-9, "Stress weights must sum to 1.0"

# ── Aggression signal weights (must sum to 1.0) ────────────────────────────────
_AGGRESSION_WEIGHTS: dict[str, float] = {
    "interviewer_interruptions": 0.30,
    "dominance_imbalance":       0.25,
    "speaking_ratio_imbalance":  0.20,
    "rapid_questioning":         0.15,
    "volume_spikes":             0.10,
}

assert abs(sum(_AGGRESSION_WEIGHTS.values()) - 1.0) < 1e-9, "Aggression weights must sum to 1.0"


def _norm(value: float | int | None, ceiling: float, invert: bool = False) -> float:
    """Normalize value to [0, 1] using an absolute ceiling.

    Returns 0.0 for None (missing data contributes nothing to the score).
    If invert=True, high value maps to 0 and low value maps to 1.
    """
    if value is None:
        return 0.0
    n = min(float(value) / ceiling, 1.0)
    return 1.0 - n if invert else n


def _norm_abs_change(value: float | None, ceiling: float = _SPEECH_CHANGE_MAX_PCT) -> float:
    """Normalize absolute percent change to [0, 1].

    Both faster and slower than baseline are treated as stress indicators.
    """
    if value is None:
        return 0.0
    return min(abs(value) / ceiling, 1.0)


def _gate_factor(active_count: int) -> float:
    if active_count >= _GATE_FULL_SIGNALS:
        return 1.0
    if active_count >= _GATE_PARTIAL_SIGNALS:
        return 0.75
    return 0.50


def _has_data(signal_key: str, audio: AudioMetrics, video: VideoMetrics) -> bool:
    """Return True if the signal has real data in this window."""
    lookup: dict[str, float | int | None] = {
        "response_delay": audio.avg_response_delay_ms,
        "filler_rate": audio.filler_words_per_min,
        "speech_rate_change": audio.speech_rate_change_percent,
        "pitch_variance": audio.pitch_variance_score,
        "silence_before_response": audio.silence_duration_before_response_ms,
        "interruptions_received": audio.interruption_count_received,
        "volume_spikes": audio.volume_spike_count,
        "posture_instability": video.posture_stability_score,
        "shoulder_tension": video.shoulder_tension_score,
        "head_jitter": video.head_jitter_score,
        "repetitive_movement": video.repetitive_hand_movement_score,
    }
    return lookup.get(signal_key) is not None


def _stress_signal_scores(
    audio: AudioMetrics,
    video: VideoMetrics,
    baseline: BaselineTracker,
) -> dict[str, float]:
    """Compute per-signal normalized stress scores."""
    stats = baseline.stats

    if stats.established and stats.response_delay_std_ms > 0:
        delay = baseline.z_score_normalized(
            audio.avg_response_delay_ms or 0.0,
            stats.response_delay_mean_ms,
            stats.response_delay_std_ms,
        )
    else:
        delay = _norm(audio.avg_response_delay_ms, _DELAY_MAX_MS)

    if stats.established and stats.filler_rate_std > 0:
        filler = baseline.z_score_normalized(
            audio.filler_words_per_min or 0.0,
            stats.filler_rate_mean,
            stats.filler_rate_std,
        )
    else:
        filler = _norm(audio.filler_words_per_min, _FILLER_RATE_MAX)

    return {
        "response_delay":         delay,
        "filler_rate":            filler,
        "speech_rate_change":     _norm_abs_change(audio.speech_rate_change_percent),
        "pitch_variance":         audio.pitch_variance_score or 0.0,
        "silence_before_response": _norm(audio.silence_duration_before_response_ms, _SILENCE_MAX_MS),
        "interruptions_received": _norm(audio.interruption_count_received, _INTERRUPT_MAX),
        "volume_spikes":          _norm(audio.volume_spike_count, _VOLUME_SPIKE_MAX),
        "posture_instability":    1.0 - (video.posture_stability_score or 1.0),
        "shoulder_tension":       video.shoulder_tension_score or 0.0,
        "head_jitter":            video.head_jitter_score or 0.0,
        "repetitive_movement":    video.repetitive_hand_movement_score or 0.0,
    }


def score_stress(
    audio: AudioMetrics,
    video: VideoMetrics,
    baseline: BaselineTracker,
) -> tuple[float, float, list[str]]:
    """Compute candidate stress score with confidence and active signal list.

    The multi-signal gate ensures that stress above 0.4 requires at least
    2 corroborating signals, preventing single-feature false positives.

    Returns:
        Tuple of (stress_score [0,1], confidence [0,1], active_signal_names).
    """
    signal_scores = _stress_signal_scores(audio, video, baseline)

    raw = sum(_STRESS_WEIGHTS[k] * v for k, v in signal_scores.items())
    active = [k for k, v in signal_scores.items() if v > _SIGNAL_ACTIVE_THRESHOLD]
    stress = min(raw * _gate_factor(len(active)), 1.0)

    available = sum(1 for k in signal_scores if _has_data(k, audio, video))
    data_completeness = available / len(signal_scores)
    baseline_factor = 1.0 if baseline.established else 0.65
    signal_factor = min(len(active) / 5.0, 1.0)
    confidence = data_completeness * baseline_factor * signal_factor

    return stress, confidence, active


def _aggression_signal_scores(
    audio: AudioMetrics,
    conversation: ConversationMetrics,
) -> dict[str, float]:
    """Compute per-signal normalized aggression scores."""
    interviewer_ratio = conversation.speaking_time_ratio_interviewer or 0.0

    dominance = max(
        0.0,
        (interviewer_ratio - _DOMINANCE_THRESHOLD) / (1.0 - _DOMINANCE_THRESHOLD),
    )

    ratio_imbalance = max(0.0, (interviewer_ratio - 0.5) * 2.0)

    avg_q = conversation.average_question_length
    rapid = max(0.0, 1.0 - (avg_q / _QUESTION_LEN_CALM)) if avg_q is not None else 0.0

    return {
        "interviewer_interruptions": _norm(
            conversation.total_interruptions_by_interviewer, _INTERRUPT_MAX * 2
        ),
        "dominance_imbalance":       dominance,
        "speaking_ratio_imbalance":  ratio_imbalance,
        "rapid_questioning":         rapid,
        "volume_spikes":             _norm(audio.volume_spike_count, _VOLUME_SPIKE_MAX),
    }


def score_aggression(
    audio: AudioMetrics,
    conversation: ConversationMetrics,
) -> tuple[float, float, list[str]]:
    """Compute interviewer aggression score with confidence and active signals.

    Aggression requires 2+ signals for a score above 0.5. A single loud
    moment or a single interruption is insufficient.

    Returns:
        Tuple of (aggression_score [0,1], confidence [0,1], active_signal_names).
    """
    signal_scores = _aggression_signal_scores(audio, conversation)

    raw = sum(_AGGRESSION_WEIGHTS[k] * v for k, v in signal_scores.items())
    active = [k for k, v in signal_scores.items() if v > _SIGNAL_ACTIVE_THRESHOLD]

    gate = 1.0 if len(active) >= 2 else 0.60
    aggression = min(raw * gate, 1.0)

    available = sum(1 for v in signal_scores.values() if v > 0.0)
    data_factor = available / len(signal_scores)
    signal_factor = 1.0 if len(active) >= 2 else 0.70
    confidence = data_factor * signal_factor

    return aggression, confidence, active


def score_dominance(conversation: ConversationMetrics) -> DominanceAnalysis:
    """Determine the dominant conversational party and imbalance magnitude."""
    candidate_ratio = conversation.speaking_time_ratio_candidate or 0.5
    interviewer_ratio = conversation.speaking_time_ratio_interviewer or 0.5
    imbalance = abs(candidate_ratio - interviewer_ratio)

    if imbalance < 0.15:
        return DominanceAnalysis(dominant_party="balanced", imbalance_score=imbalance)
    if interviewer_ratio > candidate_ratio:
        return DominanceAnalysis(dominant_party="interviewer", imbalance_score=imbalance)
    return DominanceAnalysis(dominant_party="candidate", imbalance_score=imbalance)


def build_precomputed(
    audio: AudioMetrics,
    video: VideoMetrics,
    conversation: ConversationMetrics,
    baseline: BaselineTracker,
) -> PrecomputedScores:
    """Run all deterministic scorers and bundle results for the LLM payload."""
    stress, stress_conf, stress_signals = score_stress(audio, video, baseline)
    aggression, agg_conf, agg_signals = score_aggression(audio, conversation)
    dominance = score_dominance(conversation)

    return PrecomputedScores(
        stress_score=round(stress, 3),
        aggression_score=round(aggression, 3),
        stress_confidence=round(stress_conf, 3),
        aggression_confidence=round(agg_conf, 3),
        active_stress_signals=stress_signals,
        active_aggression_signals=agg_signals,
        dominant_party=dominance.dominant_party,
        imbalance_score=round(dominance.imbalance_score, 3),
    )
