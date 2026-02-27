"""Real-time AI Interview Pressure Analyzer.

Full pipeline:
  Candidate video  → PoseBridge (YOLO pose) → BehavioralAnalyzer
  Candidate audio  → Deepgram STT → InterviewBridge → BehavioralAnalyzer
  Turn events      → InterviewBridge → BehavioralAnalyzer
  Every 5 seconds  → BehavioralAnalyzer → Gemini 2.5 Flash → BehavioralAnalysisEvent

Run:
    STREAM_API_KEY=... STREAM_API_SECRET=... \
    GOOGLE_API_KEY=... \
    uv run python examples/09_interview_pressure_analyzer/interview_analyzer.py \
        --call-id <your-call-id>
"""

import argparse
import asyncio
import logging
import os
from datetime import datetime

from dotenv import load_dotenv
import vision_agents.plugins.deepgram as deepgram
import vision_agents.plugins.elevenlabs as elevenlabs
import vision_agents.plugins.gemini as gemini
import vision_agents.plugins.getstream as getstream
from vision_agents.core.agents import Agent
from vision_agents.core.edge.types import User
from vision_agents.plugins.behavioral_analyzer import (
    BehavioralAnalysisEvent,
    BehavioralAnalyzer,
    InterviewBridge,
)

try:
    from vision_agents.plugins.behavioral_analyzer import PoseBridge
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

_RISK_ICONS = {"none": "✅", "moderate": "⚠️ ", "high": "🚨"}
_TREND_ICONS = {"increasing": "📈", "stable": "➡️ ", "decreasing": "📉"}

INTERVIEWER_INSTRUCTIONS = """\
You are a professional technical interviewer conducting a software engineering interview.
Ask challenging but fair questions. Probe depth of knowledge with follow-ups.
Keep responses concise. Do not reveal you are an AI unless directly asked.
"""


def _format_analysis(event: BehavioralAnalysisEvent) -> str:
    risk = _RISK_ICONS.get(event.risk_flag, "")
    trend = _TREND_ICONS.get(event.pressure_trend, "")
    stress = f"{event.candidate_stress:.2f}" if event.candidate_stress is not None else "n/a"
    agg = f"{event.interviewer_aggression:.2f}" if event.interviewer_aggression is not None else "n/a"
    ts = datetime.now().strftime("%H:%M:%S")

    lines = [
        f"\n{'─'*60}",
        f"  {ts}  {risk} Window {event.window_id}",
        f"{'─'*60}",
        f"  candidate_stress       = {stress}  (conf={event.stress_confidence:.2f})",
        f"  interviewer_aggression = {agg}  (conf={event.aggression_confidence:.2f})",
        f"  pressure_trend         = {trend} {event.pressure_trend}",
        f"  dominant_party         = {event.dominant_party}  (imbalance={event.imbalance_score:.2f})",
        f"  risk_flag              = {risk} {event.risk_flag}",
        f"  key_signals            = {event.key_signals}",
        f"  coach → \"{event.coaching_feedback}\"",
        f"{'─'*60}",
    ]
    return "\n".join(lines)


async def run_interview(call_id: str, candidate_user_id: str | None) -> None:
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    logger.info("Starting interview analyzer (model=%s, call_id=%s)", model, call_id)

    # ── Analysis LLM (dedicated — separate from the interviewer LLM) ──────────
    analysis_llm = gemini.LLM(model)

    # ── Behavioral analysis stack ─────────────────────────────────────────────
    analyzer = BehavioralAnalyzer(llm=analysis_llm, analysis_interval_s=5.0)
    bridge = InterviewBridge(analyzer=analyzer, candidate_user_id=candidate_user_id)

    processors = [bridge, analyzer]

    if _YOLO_AVAILABLE:
        pose = PoseBridge(
            analyzer=analyzer,
            model_path="yolo11n-pose.pt",
            fps=15,
            enable_hand_tracking=False,
        )
        processors.insert(0, pose)
        logger.info("PoseBridge enabled (YOLO pose detection active)")
    else:
        logger.info("PoseBridge skipped (ultralytics not installed)")

    # ── Interviewer LLM ───────────────────────────────────────────────────────
    interviewer_llm = gemini.LLM(model)

    # ── Agent ─────────────────────────────────────────────────────────────────
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="AI Interviewer", id="ai_interviewer"),
        instructions=INTERVIEWER_INSTRUCTIONS,
        llm=interviewer_llm,
        stt=deepgram.STT(eager_turn_detection=True),
        tts=elevenlabs.TTS(model_id="eleven_flash_v2_5"),
        processors=processors,
    )

    # ── Subscribe to analysis results ─────────────────────────────────────────
    @agent.events.subscribe
    async def on_analysis(event: BehavioralAnalysisEvent):
        print(_format_analysis(event))

        if event.risk_flag == "high":
            logger.warning(
                "HIGH RISK detected — stress=%.2f aggression=%.2f",
                event.candidate_stress or 0.0,
                event.interviewer_aggression or 0.0,
            )

    # ── Join and run ──────────────────────────────────────────────────────────
    await agent.create_user()
    call = await agent.create_call("default", call_id)

    async with agent.join(call, participant_wait_timeout=None):
        await analyzer.start()
        logger.info("Agent joined call. Behavioral analysis active.")

        await agent.simple_response(
            "Hello! I'm ready to begin. Could you start by telling me "
            "a bit about your background and what you're looking for?"
        )

        await agent.finish()
        await analyzer.stop()

    logger.info("Interview session ended.")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="AI Interview Pressure Analyzer")
    parser.add_argument("--call-id", required=True, help="GetStream call ID to join")
    parser.add_argument(
        "--candidate-user-id",
        default=None,
        help="user_id of the candidate (auto-detected if omitted)",
    )
    args = parser.parse_args()

    asyncio.run(run_interview(args.call_id, args.candidate_user_id))


if __name__ == "__main__":
    main()
