"""Dashboard preview — feeds synthetic interview data through the real WebSocket.

Runs the same FastAPI dashboard as interview_analyzer.py but replays a
simulated 90-second adversarial interview so you can inspect the UI without
needing a live GetStream call.

    uv run python examples/09_interview_pressure_analyzer/preview_dashboard.py

Then open http://localhost:8080
"""

import asyncio
import logging
import math
import os
import random
import sys

import uvicorn

from vision_agents.plugins.behavioral_analyzer import (
    AggressionClassifier,
    Explainer,
    SpikeDetector,
    SpikeCause,
)
from vision_agents.plugins.behavioral_analyzer._resilience import ResilienceTracker
from vision_agents.plugins.behavioral_analyzer._session_summary import (
    SessionSummaryGenerator,
    WindowRecord,
)

sys.path.insert(0, os.path.dirname(__file__))
from interview_analyzer import _Dashboard  # noqa: E402

logging.basicConfig(level=logging.WARNING)

WINDOW_S = 5
N_WINDOWS = 18
INTERVAL_S = 1.2  # seconds between windows (replay speed)

_COACHING = [
    "Take a breath before answering — pausing projects confidence.",
    "Slow your speech rate; rapid answers can signal anxiety.",
    "You're doing well — maintain eye contact and steady posture.",
    "Try to reduce filler words; replace 'um' with a deliberate pause.",
    "Good recovery! Stay composed and structure your answer clearly.",
    "Acknowledge the question before diving in to buy thinking time.",
    "Strong answer — your pacing and clarity are on point.",
    "Reduce shoulder tension; relaxed posture signals confidence.",
]

_QUESTIONS = [
    ("Describe a system you designed end-to-end.", "low"),
    ("How does consistent hashing work in distributed systems?", "high"),
    ("Walk me through your debugging process for a P0 incident.", "medium"),
    ("Design a rate limiter for a global API gateway.", "high"),
    ("What trade-offs did you make in your last major project?", "medium"),
]


def _stress(t: float) -> float:
    base = 0.5 * (1 - math.cos(math.pi * t / (N_WINDOWS * WINDOW_S)))
    return max(0.0, min(1.0, base * 0.75 + random.uniform(-0.04, 0.04)))


def _agg(t: float) -> float:
    base = 0.08 + 0.12 * math.sin(math.pi * t / (N_WINDOWS * WINDOW_S * 0.6))
    return max(0.0, min(1.0, base + random.uniform(-0.02, 0.03)))


def _trend(prev: float, curr: float) -> str:
    if curr - prev > 0.05:
        return "increasing"
    if prev - curr > 0.05:
        return "decreasing"
    return "stable"


def _risk(stress: float) -> str:
    if stress >= 0.60:
        return "high"
    if stress >= 0.35:
        return "moderate"
    return "none"


async def _replay(dashboard: _Dashboard) -> None:
    """Stream synthetic events to all connected dashboard clients."""
    await asyncio.sleep(2.0)  # let the uvicorn server start

    session_id = "preview-001"
    await dashboard.broadcast({"type": "session_start", "session_id": session_id})

    spike_detector = SpikeDetector()
    resilience = ResilienceTracker()
    agg_classifier = AggressionClassifier()
    explainer = Explainer()
    summary_gen = SessionSummaryGenerator(session_id=session_id)

    prev_stress = 0.0
    q_idx = 0
    stress_history: list[float] = []

    for i in range(N_WINDOWS):
        t = float(i * WINDOW_S)
        wid = f"w{i:04d}"

        s = _stress(t)
        a = _agg(t)
        stress_history.append(s)

        conf_s = min(0.10 + i * 0.03, 0.80)
        conf_a = min(0.14 + i * 0.025, 0.70)

        # Build a varied baseline so std > 0 for spike detection
        for j in range(max(0, i - 12), i):
            varied = _stress(j * WINDOW_S) + (j % 5) * 0.01
            spike_detector.ingest(varied, f"w{j:04d}")

        if i in (4, 9, 13) and q_idx < len(_QUESTIONS):
            spike_detector.hint_cause(SpikeCause.QUESTION_RECEIVED)
            q_idx += 1

        spike_event = spike_detector.ingest(s, wid)

        # Resilience — update() returns None; compute() gives the result
        mean_s = sum(stress_history) / len(stress_history)
        std_s = (
            (sum((x - mean_s) ** 2 for x in stress_history) / len(stress_history)) ** 0.5
            if len(stress_history) > 1 else 0.1
        )
        if spike_event:
            resilience.on_spike(spike_event.stress_after, t)
        resilience.update(s, mean_s, std_s, t)
        res_result = resilience.compute()

        # Aggression classification — param is interviewer_dominance
        interruptions = a * 4.0
        question_rate = 2.5
        imbalance = a * 0.6 + s * 0.2
        agg_result = agg_classifier.classify(
            aggression_score=a,
            interruptions_per_window=interruptions,
            question_rate_per_min=question_rate,
            interviewer_dominance=imbalance,
            volume_spikes=float(int(a * 3)),
        )

        # Explainability — needs raw signal dicts, not AudioMetrics objects
        stress_signals = {
            "response_delay": min(s * 1.5, 1.0),
            "filler_rate": min(s * 1.2, 1.0),
            "pitch_variance": min(s * 0.9, 1.0),
            "speech_rate_change": min(s * 0.8, 1.0),
            "shoulder_tension": min(s * 0.7, 1.0),
        }
        agg_signals = {
            "interviewer_interruptions": min(interruptions / 5.0, 1.0),
            "dominance_imbalance": imbalance,
            "speaking_ratio_imbalance": min(a * 0.8, 1.0),
            "rapid_questioning": question_rate / 4.0,
            "volume_spikes": min(int(a * 3) / 5.0, 1.0),
        }
        exp_report = explainer.explain(
            window_id=wid,
            stress_score=s,
            aggression_score=a,
            stress_signal_values=stress_signals,
            aggression_signal_values=agg_signals,
            confidence_breakdown={"data_completeness": conf_s},
        )

        top_signal = (
            exp_report.top_stress_signals[0].signal_name
            if exp_report.top_stress_signals else None
        )

        coaching = _COACHING[i % len(_COACHING)]
        trend = _trend(prev_stress, s)

        await dashboard.broadcast({
            "type": "analysis",
            "window_id": wid,
            "candidate_stress": round(s, 3),
            "stress_confidence": round(conf_s, 2),
            "interviewer_aggression": round(a, 3),
            "aggression_confidence": round(conf_a, 2),
            "pressure_trend": trend,
            "dominant_party": "candidate" if s < 0.4 else "interviewer",
            "imbalance_score": round(imbalance, 3),
            "risk_flag": _risk(s),
            "key_signals": [top_signal] if top_signal else [],
            "coaching_feedback": coaching,
            "aggression_class": agg_result.classification.value,
            "aggression_class_confidence": round(agg_result.confidence, 2),
            "resilience_score": round(res_result.resilience_score, 3),
        })

        if spike_event:
            await dashboard.broadcast({
                "type": "spike",
                "window_id": wid,
                "stress_before": round(spike_event.stress_before, 3),
                "stress_after": round(spike_event.stress_after, 3),
                "delta_sigma": round(spike_event.delta_sigma, 2),
                "cause": spike_event.cause.value,
            })

        summary_gen.record_window(WindowRecord(
            window_id=wid,
            timestamp=t,
            candidate_stress=s,
            interviewer_aggression=a,
            aggression_class=agg_result.classification.value,
            risk_flag=_risk(s),
            coaching_feedback=coaching,
            dominant_party="candidate" if s < 0.4 else "interviewer",
        ))

        prev_stress = s
        print(f"  [{wid}] stress={s:.2f}  agg={a:.2f}  risk={_risk(s):<8}  {agg_result.classification.value}")
        await asyncio.sleep(INTERVAL_S)

    # Feed resilience and spike list into the summary generator
    summary_gen.record_resilience(res_result)
    summary_gen.record_spikes(spike_detector.recent_spikes(within_s=9999.0))

    # generate() takes no arguments
    summary = summary_gen.generate()

    await dashboard.broadcast({
        "type": "summary",
        "summary": {
            "session_id": summary.session_id,
            "duration_s": summary.duration_s,
            "peak_stress": round(summary.peak_stress, 3),
            "average_stress": round(summary.average_stress, 3),
            "resilience_score": round(summary.resilience_score, 3),
            "spike_count": summary.spike_count,
            "filler_word_density_per_min": round(summary.filler_word_density_per_min, 2),
            "dominant_aggression_class": summary.dominant_aggression_class,
            "dominant_party_summary": summary.dominant_party_summary,
            "strengths": summary.strengths,
            "improvement_areas": summary.improvement_areas,
        },
        "comparison": None,
    })
    print("\n  Session summary sent — refresh browser to replay. Ctrl-C to quit.")


async def main() -> None:
    port = int(os.environ.get("PORT", 8080))
    dashboard = _Dashboard(call_id="preview", stream_api_key="preview", stream_api_secret="preview")
    server = uvicorn.Server(uvicorn.Config(
        dashboard.app, host="0.0.0.0", port=port, log_level="warning",
    ))

    print("=" * 60)
    print("  Interview Pressure Analyzer — Dashboard Preview")
    print("=" * 60)
    print(f"  Open  http://localhost:{port}  in your browser")
    print(f"  Replaying {N_WINDOWS} windows × {WINDOW_S}s at {INTERVAL_S}s per window")
    print("=" * 60)

    await asyncio.gather(server.serve(), _replay(dashboard))


if __name__ == "__main__":
    asyncio.run(main())
