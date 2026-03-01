"""Real-time interview behavioral pressure analysis plugin for Vision Agents.

Core components:
  BehavioralAnalyzer  — background analysis loop + LLM scoring (must be in processors)
  InterviewBridge     — routes STT / turn events → analyzer metrics
  PoseBridge          — YOLO subclass that routes pose metrics → analyzer

Advanced intelligence modules:
  SpikeDetector       — detects >2σ stress spikes in a 1s sliding window
  ResilienceTracker   — measures post-spike recovery time and resilience score
  QuestionDifficultyAnalyzer — correlates question complexity with stress
  AggressionClassifier — classifies interviewer style
  SessionSummaryGenerator — produces end-of-session JSON / Markdown report
  SessionComparisonEngine — compares two sessions for improvement deltas
  Explainer           — signal contribution breakdown and audit log
  GracefulDegradationEngine — policy for missing signals or LLM unavailability
  LatencyTracker      — per-stage latency budgeting with optional OTEL export

Minimal wiring example::

    from vision_agents.plugins.behavioral_analyzer import (
        BehavioralAnalyzer, InterviewBridge, PoseBridge,
        BehavioralAnalysisEvent, SpikeDetectedEvent, SessionSummaryEvent,
    )
    from vision_agents.plugins.gemini import LLM as GeminiLLM

    analyzer = BehavioralAnalyzer(llm=GeminiLLM("gemini-2.5-flash"))
    analyzer.start_session(session_id="session-001")
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
        print(f"[{event.risk_flag}] stress={event.candidate_stress:.2f} ({event.aggression_class})")

    @agent.events.subscribe
    async def on_spike(event: SpikeDetectedEvent):
        print(f"SPIKE {event.delta_sigma:.1f}σ cause={event.cause}")

    @agent.events.subscribe
    async def on_summary(event: SessionSummaryEvent):
        print(f"Session over — resilience={event.resilience_score:.2f}")

    async with agent.join(call):
        await analyzer.start()
        await agent.finish()
        await analyzer.stop()
        summary = analyzer.generate_summary()
        print(summary_generator.to_markdown(summary))
"""

from .analyzer import BehavioralAnalyzer
from .events import BehavioralAnalysisEvent, SessionSummaryEvent, SpikeDetectedEvent
from .interview_bridge import InterviewBridge
from ._aggression_classifier import AggressionClass, AggressionClassification, AggressionClassifier
from ._degradation import GracefulDegradationEngine, SignalAvailability, DegradationPolicy
from ._explainability import Explainer, ExplainabilityReport, SignalContribution
from ._question_analyzer import QuestionDifficultyAnalyzer, QuestionComplexity, CorrelationResult
from ._resilience import ResilienceTracker, ResilienceResult
from ._session_comparison import SessionComparisonEngine, SessionComparison, SessionDelta
from ._session_summary import SessionSummaryGenerator, SessionSummary, WindowRecord
from ._spike_detector import SpikeDetector, SpikeEvent, SpikeCause
from ._telemetry import LatencyTracker, LATENCY_BUDGETS_MS
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
    # Core processors
    "BehavioralAnalyzer",
    "InterviewBridge",
    # Events
    "BehavioralAnalysisEvent",
    "SpikeDetectedEvent",
    "SessionSummaryEvent",
    # Aggression classification
    "AggressionClass",
    "AggressionClassification",
    "AggressionClassifier",
    # Degradation
    "DegradationPolicy",
    "GracefulDegradationEngine",
    "SignalAvailability",
    # Explainability
    "Explainer",
    "ExplainabilityReport",
    "SignalContribution",
    # Question difficulty
    "CorrelationResult",
    "QuestionComplexity",
    "QuestionDifficultyAnalyzer",
    # Resilience
    "ResilienceResult",
    "ResilienceTracker",
    # Session comparison
    "SessionComparison",
    "SessionComparisonEngine",
    "SessionDelta",
    # Session summary
    "SessionSummary",
    "SessionSummaryGenerator",
    "WindowRecord",
    # Spike detection
    "SpikeCause",
    "SpikeDetector",
    "SpikeEvent",
    # Telemetry
    "LATENCY_BUDGETS_MS",
    "LatencyTracker",
    # Data types
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
