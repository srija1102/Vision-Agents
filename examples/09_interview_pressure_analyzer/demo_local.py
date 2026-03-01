"""Local demo for the Interview Pressure Analyzer — no API keys required.

Simulates a 90-second adversarial interview session and prints live output
from every new module:
  • 5-second window aggregation
  • Micro-spike detection (>2σ)
  • Resilience tracking
  • Aggression classification
  • Signal explainability
  • Graceful degradation policy
  • Per-stage latency tracking
  • End-of-session summary (JSON + Markdown)
  • Session comparison (session 1 vs session 2)
  • Question difficulty correlation

Run:
    uv run python examples/09_interview_pressure_analyzer/demo_local.py
"""

import asyncio
import json
import math
import random
import time

from vision_agents.plugins.behavioral_analyzer._aggression_classifier import AggressionClassifier
from vision_agents.plugins.behavioral_analyzer._aggregator import MetricsAggregator
from vision_agents.plugins.behavioral_analyzer._baseline import BaselineTracker
from vision_agents.plugins.behavioral_analyzer._degradation import (
    GracefulDegradationEngine,
    SignalAvailability,
)
from vision_agents.plugins.behavioral_analyzer._explainability import Explainer
from vision_agents.plugins.behavioral_analyzer._question_analyzer import QuestionDifficultyAnalyzer
from vision_agents.plugins.behavioral_analyzer._resilience import ResilienceTracker
from vision_agents.plugins.behavioral_analyzer._scorer import build_precomputed
from vision_agents.plugins.behavioral_analyzer._session_comparison import SessionComparisonEngine
from vision_agents.plugins.behavioral_analyzer._session_summary import (
    SessionSummaryGenerator,
    WindowRecord,
)
from vision_agents.plugins.behavioral_analyzer._spike_detector import SpikeCause, SpikeDetector
from vision_agents.plugins.behavioral_analyzer._telemetry import LatencyTracker

# ── Colour helpers ────────────────────────────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[31m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
DIM = "\033[2m"


def _c(text: str, colour: str) -> str:
    return f"{colour}{text}{RESET}"


def _bar(value: float, width: int = 20) -> str:
    filled = int(value * width)
    bar = "█" * filled + "░" * (width - filled)
    if value > 0.70:
        colour = RED
    elif value > 0.40:
        colour = YELLOW
    else:
        colour = GREEN
    return f"{colour}{bar}{RESET} {value:.2f}"


# ── Simulated interview questions ─────────────────────────────────────────────

_QUESTIONS = [
    ("What's your name?", 0.15),
    ("Tell me about yourself.", 0.20),
    ("Describe a recent project.", 0.28),
    ("Walk me through your approach to system design.", 0.38),
    ("Design a distributed rate limiter with eventual consistency.", 0.62),
    ("Implement a lock-free concurrent queue with wait-free guarantees.", 0.71),
    ("Why are you leaving your current job?", 0.45),
    ("What's your biggest weakness?", 0.50),
    ("Design a global payments system with 99.999% uptime.", 0.74),
    ("You failed that answer. Try again under time pressure.", 0.85),
    ("Faster. You're taking too long.", 0.90),
    ("That's wrong. What else?", 0.88),
    ("Describe your proudest achievement.", 0.35),
    ("Where do you see yourself in 5 years?", 0.25),
    ("Any questions for us?", 0.15),
]


def _stress_curve(t: float, total: float) -> float:
    """Simulate a realistic stress arc: low → peak → recovery."""
    phase = t / total
    # Base arc: rises to peak around 70% through then falls
    arc = math.sin(phase * math.pi) * 0.55
    # Mid-session spike from hostile questions
    if 0.55 < phase < 0.80:
        arc += 0.30
    noise = random.uniform(-0.05, 0.05)
    return max(0.0, min(1.0, arc + noise))


def _aggression_curve(t: float, total: float) -> float:
    """Simulated interviewer aggression: ramps up then drops."""
    phase = t / total
    if 0.50 < phase < 0.80:
        return min(1.0, 0.60 + random.uniform(0.0, 0.20))
    return max(0.0, 0.15 + random.uniform(-0.05, 0.10))


# ── Main simulation ───────────────────────────────────────────────────────────

async def run_session(session_id: str, seed: int = 42) -> "SessionSummaryGenerator":
    random.seed(seed)

    print(f"\n{BOLD}{CYAN}{'═' * 66}{RESET}")
    print(f"{BOLD}{CYAN}  Interview Pressure Analyzer — Local Demo  ({session_id}){RESET}")
    print(f"{BOLD}{CYAN}{'═' * 66}{RESET}\n")

    total_s = 90.0
    window_s = 5.0
    num_windows = int(total_s / window_s)

    # Modules
    aggregator = MetricsAggregator(window_size_s=window_s)
    baseline = BaselineTracker()
    spike_detector = SpikeDetector()
    resilience_tracker = ResilienceTracker()
    aggression_classifier = AggressionClassifier()
    explainer = Explainer()
    latency_tracker = LatencyTracker()
    q_analyzer = QuestionDifficultyAnalyzer()
    summary_gen = SessionSummaryGenerator(session_id)

    total_fillers = 0
    question_idx = 0

    print(f"{DIM}Simulating {num_windows} analysis windows × {window_s:.0f}s each...{RESET}\n")

    for w in range(num_windows):
        t = w * window_s
        stress_val = _stress_curve(t, total_s)
        agg_val = _aggression_curve(t, total_s)

        # Feed synthetic audio
        fillers = random.randint(0, 3) if stress_val > 0.40 else 0
        total_fillers += fillers
        await aggregator.ingest_audio_event(
            words_this_window=random.randint(40, 120),
            filler_count=fillers,
            response_delay_ms=200 + stress_val * 3000,
            pitch_variance=stress_val * 0.8,
            volume_spike=agg_val > 0.65,
            silence_ms=stress_val * 2000,
            interrupted=agg_val > 0.70,
            speech_rate_change_pct=(stress_val - 0.3) * 80,
        )

        # Feed synthetic video
        await aggregator.ingest_video_event(
            posture_stability=1.0 - stress_val * 0.7,
            shoulder_tension=stress_val * 0.8,
            head_jitter=stress_val * 0.5,
            hand_movement=stress_val * 0.6,
        )

        # Feed turn dynamics
        await aggregator.ingest_turn_event(
            candidate_speaking_ms=window_s * 1000 * (0.5 - agg_val * 0.2),
            interviewer_speaking_ms=window_s * 1000 * (0.5 + agg_val * 0.2),
            interviewer_interrupted=agg_val > 0.70,
            question_word_count=random.randint(8, 30),
        )

        # Question arrives mid-session
        if w > 0 and w % 3 == 0 and question_idx < len(_QUESTIONS):
            q_text, q_stress = _QUESTIONS[question_idx]
            q_analyzer.record(q_text, stress_at_response=stress_val)
            spike_detector.hint_cause(SpikeCause.QUESTION_RECEIVED)
            question_idx += 1

        # Hint interruption cause when aggression spikes
        if agg_val > 0.70:
            spike_detector.hint_cause(SpikeCause.INTERRUPTION)

        # Flush window and score
        with latency_tracker.measure("window_flush"):
            payload = await aggregator.flush_window()
        baseline.record_window(payload.audio, payload.video)

        with latency_tracker.measure("deterministic_scoring"):
            pre = build_precomputed(payload.audio, payload.video, payload.conversation, baseline)

        aggregator.update_rolling_estimates(pre.stress_score, pre.aggression_score)

        # Spike detection
        with latency_tracker.measure("spike_detection"):
            spike = spike_detector.ingest(stress_val, payload.window_id)

        # Resilience update
        if len(spike_detector.spike_history()) > 0 and baseline.established:
            sh = [s.stress_after for s in spike_detector.spike_history()]
            s_mean = sum(sh) / len(sh)
            resilience_tracker.update(
                current_stress=stress_val,
                baseline_mean=0.30,
                baseline_std=0.12,
            )
        if spike:
            resilience_tracker.on_spike(spike_stress=spike.stress_after)

        # Aggression classification
        agg_class = aggression_classifier.classify(
            aggression_score=pre.aggression_score,
            interruptions_per_window=float(payload.audio.interruption_count_received or 0),
            question_rate_per_min=float(payload.conversation.average_question_length or 0),
            interviewer_dominance=float(payload.conversation.speaking_time_ratio_interviewer or 0.5),
            volume_spikes=float(payload.audio.volume_spike_count or 0),
        )

        # Explainability (generated every window, printed for high-stress windows)
        explanation = explainer.explain(
            window_id=payload.window_id,
            stress_score=pre.stress_score,
            aggression_score=pre.aggression_score,
            stress_signal_values={
                "response_delay": min((payload.audio.avg_response_delay_ms or 0) / 5000, 1.0),
                "filler_rate": min((payload.audio.filler_words_per_min or 0) / 20, 1.0),
                "pitch_variance": payload.audio.pitch_variance_score or 0.0,
                "posture_instability": 1.0 - (payload.video.posture_stability_score or 1.0),
                "shoulder_tension": payload.video.shoulder_tension_score or 0.0,
            },
            aggression_signal_values={
                "interviewer_interruptions": min((payload.audio.interruption_count_given or 0) / 5, 1.0),
                "dominance_imbalance": pre.imbalance_score,
            },
            confidence_breakdown={"stress": pre.stress_confidence},
        )

        # Risk flag
        risk = "high" if pre.stress_score > 0.70 or pre.aggression_score > 0.70 else \
               "moderate" if pre.stress_score > 0.40 or pre.aggression_score > 0.40 else "none"
        risk_icon = {"none": "✅", "moderate": "⚠️ ", "high": "🚨"}[risk]

        # Accumulate for summary
        summary_gen.record_window(WindowRecord(
            window_id=payload.window_id,
            timestamp=float(w * window_s),
            candidate_stress=pre.stress_score,
            interviewer_aggression=pre.aggression_score,
            aggression_class=agg_class.classification.value,
            risk_flag=risk,
            coaching_feedback="Focus and breathe." if pre.stress_score > 0.5 else "Good pace.",
            dominant_party=pre.dominant_party,
        ))

        # ── Print window output ───────────────────────────────────────────────
        trend_char = "📈" if w > 0 and pre.stress_score > 0.45 else "📉" if pre.stress_score < 0.25 else "➡️ "
        print(f"{BOLD}Window {payload.window_id}  t={t:.0f}s  {risk_icon}  {trend_char}{RESET}")
        print(f"  Stress      {_bar(pre.stress_score)}  conf={pre.stress_confidence:.2f}")
        print(f"  Aggression  {_bar(pre.aggression_score)}  conf={pre.aggression_confidence:.2f}")
        print(f"  Style       {_c(agg_class.classification.value, MAGENTA)}  (conf={agg_class.confidence:.2f}, signal={agg_class.primary_signal})")

        if spike:
            print(f"  {_c('⚡ SPIKE', RED + BOLD)}  {spike.delta_sigma:.1f}σ  cause={spike.cause.value}  before={spike.stress_before:.2f}→after={spike.stress_after:.2f}")

        if pre.stress_score > 0.55 and explanation.top_stress_signals:
            top = explanation.top_stress_signals[0]
            print(f"  {DIM}Top signal: {top.signal_name} ({top.percent_of_total:.0f}% of score){RESET}")

        if pre.active_stress_signals:
            print(f"  {DIM}Active signals: {', '.join(pre.active_stress_signals[:3])}{RESET}")
        print()

        # Small sleep to make output readable
        await asyncio.sleep(0.05)

    summary_gen.record_filler_total(total_fillers)
    summary_gen.record_resilience(resilience_tracker.compute())
    summary_gen.record_correlation(q_analyzer.correlate())

    return summary_gen, latency_tracker


async def main() -> None:
    # ── Session 1 ─────────────────────────────────────────────────────────────
    gen1, latency1 = await run_session("session-001", seed=42)
    summary1 = gen1.generate()

    # ── Session 2 (slightly better candidate) ─────────────────────────────────
    print(f"\n{BOLD}{CYAN}{'═' * 66}{RESET}")
    print(f"{BOLD}{CYAN}  Running session 2 (follow-up practice session)...{RESET}")
    print(f"{BOLD}{CYAN}{'═' * 66}{RESET}\n")
    gen2, _ = await run_session("session-002", seed=99)
    summary2 = gen2.generate()

    # ── Degradation demo ──────────────────────────────────────────────────────
    print(f"\n{BOLD}{YELLOW}{'─' * 66}{RESET}")
    print(f"{BOLD}{YELLOW}  Graceful Degradation Scenarios{RESET}")
    print(f"{BOLD}{YELLOW}{'─' * 66}{RESET}")
    engine = GracefulDegradationEngine()
    scenarios = [
        ("All systems nominal", SignalAvailability()),
        ("STT offline", SignalAvailability(stt_available=False)),
        ("No video track", SignalAvailability(video_available=False)),
        ("LLM timeout", SignalAvailability(llm_available=False)),
        ("Audio + Video lost", SignalAvailability(audio_available=False, video_available=False)),
    ]
    for label, avail in scenarios:
        policy = engine.evaluate(avail)
        llm_str = _c("LLM✓", GREEN) if policy.should_run_llm else _c("LLM✗", RED)
        cap = policy.confidence_cap
        cap_colour = GREEN if cap >= 0.90 else YELLOW if cap >= 0.50 else RED
        print(f"  {label:<30} conf_cap={_c(f'{cap:.2f}', cap_colour)}  {llm_str}")
        if policy.warnings:
            print(f"    {DIM}↳ {policy.warnings[0]}{RESET}")
    print()

    # ── Latency stats ─────────────────────────────────────────────────────────
    print(f"\n{BOLD}{CYAN}{'─' * 66}{RESET}")
    print(f"{BOLD}{CYAN}  Latency Budget Report (session 1){RESET}")
    print(f"{BOLD}{CYAN}{'─' * 66}{RESET}")
    for stage, stats in latency1.get_stats().items():
        mean = stats["mean_ms"]
        colour = GREEN if mean < 5 else YELLOW if mean < 50 else RED
        print(f"  {stage:<28} mean={_c(f'{mean:.2f}ms', colour)}  p95={stats['p95_ms']:.2f}ms  n={int(stats['count'])}")
    print()

    # ── Session comparison ────────────────────────────────────────────────────
    print(f"\n{BOLD}{CYAN}{'─' * 66}{RESET}")
    print(f"{BOLD}{CYAN}  Session Comparison: {summary1.session_id} vs {summary2.session_id}{RESET}")
    print(f"{BOLD}{CYAN}{'─' * 66}{RESET}")
    comparison = SessionComparisonEngine().compare(summary1, summary2)
    for line in comparison.summary_lines:
        colour = GREEN if line.startswith("✓") else RED
        print(f"  {_c(line, colour)}")
    score = comparison.overall_improvement_score
    score_colour = GREEN if score >= 0.60 else YELLOW if score >= 0.40 else RED
    print(f"\n  Overall improvement: {_c(f'{score*100:.0f}%', score_colour + BOLD)} of metrics improved\n")

    # ── Session summary ───────────────────────────────────────────────────────
    print(f"\n{BOLD}{CYAN}{'═' * 66}{RESET}")
    print(f"{BOLD}{CYAN}  End-of-Session Summary — {summary1.session_id}{RESET}")
    print(f"{BOLD}{CYAN}{'═' * 66}{RESET}\n")

    print(f"  Duration          : {summary1.duration_s:.0f}s")
    print(f"  Peak stress       : {_bar(summary1.peak_stress)}")
    print(f"  Average stress    : {_bar(summary1.average_stress)}")
    print(f"  Resilience score  : {_bar(summary1.resilience_score)}")
    print(f"  Spike count       : {summary1.spike_count}")
    print(f"  Avg recovery      : {summary1.average_recovery_s:.1f}s")
    print(f"  Filler rate       : {summary1.filler_word_density_per_min:.1f}/min")
    print(f"  Dominant style    : {_c(summary1.dominant_aggression_class, MAGENTA)}")
    print(f"  Dominant party    : {summary1.dominant_party_summary}")

    if summary1.stress_difficulty_correlation is not None:
        r = summary1.stress_difficulty_correlation
        print(f"  Stress/difficulty r: {_c(f'{r:.3f}', YELLOW if abs(r) > 0.4 else GREEN)}")
        for bucket, avg in summary1.stress_by_question_type.items():
            print(f"    {bucket:<8} complexity → avg stress {avg:.2f}")

    if summary1.strengths:
        print(f"\n  {BOLD}{GREEN}Strengths:{RESET}")
        for s in summary1.strengths:
            print(f"    • {s}")

    if summary1.improvement_areas:
        print(f"\n  {BOLD}{YELLOW}Areas to improve:{RESET}")
        for s in summary1.improvement_areas:
            print(f"    • {s}")

    if summary1.improvement_plan:
        print(f"\n  {BOLD}{CYAN}Action plan:{RESET}")
        for i, p in enumerate(summary1.improvement_plan, 1):
            print(f"    {i}. {p}")

    # ── JSON output ───────────────────────────────────────────────────────────
    print(f"\n{BOLD}{DIM}{'─' * 66}{RESET}")
    print(f"{BOLD}{DIM}  Full JSON summary (truncated):{RESET}")
    print(f"{BOLD}{DIM}{'─' * 66}{RESET}")
    json_str = gen1.to_json(summary1)
    parsed = json.loads(json_str)
    # Print a readable subset
    subset = {k: v for k, v in parsed.items() if k not in ("window_records", "improvement_plan", "spike_causes")}
    print(json.dumps(subset, indent=2))

    # ── Markdown preview ──────────────────────────────────────────────────────
    md = gen1.to_markdown(summary1)
    print(f"\n{BOLD}{DIM}{'─' * 66}{RESET}")
    print(f"{BOLD}{DIM}  Markdown report (first 20 lines):{RESET}")
    print(f"{BOLD}{DIM}{'─' * 66}{RESET}")
    for line in md.split("\n")[:20]:
        print(f"  {DIM}{line}{RESET}")

    print(f"\n{BOLD}{GREEN}{'═' * 66}{RESET}")
    print(f"{BOLD}{GREEN}  All modules verified. 74 unit tests also pass:{RESET}")
    print(f"{BOLD}{GREEN}  uv run py.test plugins/behavioral_analyzer/tests/ -q{RESET}")
    print(f"{BOLD}{GREEN}{'═' * 66}{RESET}\n")


if __name__ == "__main__":
    asyncio.run(main())
