"""Real-time interview behavioral pressure analysis plugin for Vision Agents.

Core components:
  BehavioralAnalyzer  — background analysis loop + LLM scoring (must be in processors)
  InterviewBridge     — routes STT / turn events → analyzer metrics
  PoseBridge          — YOLO subclass that routes pose metrics → analyzer

Minimal wiring example::

    from vision_agents.plugins.behavioral_analyzer import (
        BehavioralAnalyzer, InterviewBridge, PoseBridge, BehavioralAnalysisEvent
    )
    from vision_agents.plugins.gemini import LLM as GeminiLLM

    analyzer = BehavioralAnalyzer(llm=GeminiLLM("gemini-2.5-flash"))
    bridge   = InterviewBridge(analyzer=analyzer, candidate_user_id="candidate")
    pose     = PoseBridge(analyzer=analyzer, model_path="yolo11n-pose.pt")

    agent = Agent(
        edge=...,
        llm=...,
        stt=...,
        processors=[pose, bridge, analyzer],
    )

    @agent.events.subscribe
    async def on_analysis(event: BehavioralAnalysisEvent):
        print(f"[{event.risk_flag}] stress={event.candidate_stress:.2f}")
        print(f"Coach: {event.coaching_feedback}")

    async with agent.join(call):
        await analyzer.start()
        await agent.finish()
        await analyzer.stop()
"""

from .analyzer import BehavioralAnalyzer
from .events import BehavioralAnalysisEvent
from .interview_bridge import InterviewBridge
from ._types import (
    AnalysisResult,
    AudioMetrics,
    BehavioralPayload,
    ConversationMetrics,
    DominanceAnalysis,
    HistoricalContext,
    PrecomputedScores,
    VideoMetrics,
)

__all__ = [
    "BehavioralAnalyzer",
    "BehavioralAnalysisEvent",
    "InterviewBridge",
    "AnalysisResult",
    "AudioMetrics",
    "BehavioralPayload",
    "ConversationMetrics",
    "DominanceAnalysis",
    "HistoricalContext",
    "PrecomputedScores",
    "VideoMetrics",
]

try:
    from .pose_bridge import PoseBridge  # noqa: F401
    __all__.append("PoseBridge")
except ImportError:
    pass
