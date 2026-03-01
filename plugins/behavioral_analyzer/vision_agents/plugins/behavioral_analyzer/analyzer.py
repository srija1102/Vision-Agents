"""BehavioralAnalyzer — production-grade real-time interview pressure processor.

Architecture overview:
  1. Upstream producers call ingest_audio_event / ingest_video_event /
     ingest_turn_event to feed raw observations into the 5-second buffer.
  2. Every analysis_interval_s, _analysis_loop flushes the buffer and runs
     the deterministic pre-scorer (_scorer.py).
  3. If confidence is high enough, the pre-scores are sent to the LLM alongside
     the aggregated metrics. The LLM returns JSON output.
  4. The parsed result (or a deterministic fallback on LLM failure) is emitted
     as a BehavioralAnalysisEvent on the agent's event bus.
  5. Spike detection, resilience tracking, aggression classification, and
     explainability are computed every cycle with no additional I/O.

LLM usage notes:
  - Pass a DEDICATED LLM instance — not the one used by the main agent.
    The analyzer calls set_instructions() on the LLM at startup.
  - The LLM is called at most once per analysis_interval_s.
  - If a previous LLM call is still in progress, the next cycle is skipped
    rather than queued, preventing cascading latency.
  - When the LLM is unavailable, DegradationPolicy routes to deterministic fallback.
"""

import asyncio
import dataclasses
import json
import logging
import time
from collections import deque
from typing import Literal, Optional, cast

from vision_agents.core.llm import LLM
from vision_agents.core.processors import Processor

from ._aggregator import MetricsAggregator
from ._aggression_classifier import AggressionClassifier
from ._baseline import BaselineTracker
from ._degradation import GracefulDegradationEngine, SignalAvailability
from ._explainability import Explainer
from ._prompt import SYSTEM_PROMPT
from ._resilience import ResilienceTracker
from ._scorer import build_precomputed
from ._session_summary import SessionSummaryGenerator, WindowRecord
from ._spike_detector import SpikeDetector, SpikeCause
from ._telemetry import LatencyTracker
from ._types import (
    AnalysisResult,
    BehavioralPayload,
    DominanceAnalysis,
    PrecomputedScores,
)
from .events import BehavioralAnalysisEvent, SessionSummaryEvent, SpikeDetectedEvent

logger = logging.getLogger(__name__)

_MIN_CONFIDENCE_FOR_LLM = 0.20
_MAX_LLM_RETRIES = 2
_DEFAULT_COACHING = "Stay focused. Take a breath before responding."


def _build_user_message(payload: BehavioralPayload, pre: PrecomputedScores) -> str:
    """Serialize the analysis window to a compact JSON string for the LLM.

    Only non-None fields are included to minimise token usage.
    The pre_computed block anchors the LLM to our deterministic scores.
    """

    def _clean(d: dict) -> dict:
        return {k: v for k, v in d.items() if v is not None}

    data: dict = {
        "window": payload.window_id,
        "pre_computed": dataclasses.asdict(pre),
        "audio": _clean(dataclasses.asdict(payload.audio)),
        "video": _clean(dataclasses.asdict(payload.video)),
        "conversation": _clean(dataclasses.asdict(payload.conversation)),
        "history": dataclasses.asdict(payload.history),
    }
    return json.dumps(data, separators=(",", ":"))


def _parse_result(raw: str) -> Optional[AnalysisResult]:
    """Parse the LLM JSON output into an AnalysisResult.

    Extracts the outermost JSON object from the response, tolerating
    any markdown code fences the LLM may have added.
    """
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start == -1 or end == 0:
        logger.error("LLM output contained no JSON object")
        return None

    try:
        data = json.loads(raw[start:end])
    except json.JSONDecodeError:
        logger.exception("LLM output was not valid JSON")
        return None

    try:
        dom = data.get("dominance_analysis") or {}
        return AnalysisResult(
            candidate_stress=data.get("candidate_stress"),
            interviewer_aggression=data.get("interviewer_aggression"),
            pressure_trend=cast(
                Literal["increasing", "stable", "decreasing"],
                data.get("pressure_trend", "stable"),
            ),
            stress_confidence=float(data.get("stress_confidence", 0.0)),
            aggression_confidence=float(data.get("aggression_confidence", 0.0)),
            dominance_analysis=DominanceAnalysis(
                dominant_party=cast(
                    Literal["candidate", "interviewer", "balanced"],
                    dom.get("dominant_party", "balanced"),
                ),
                imbalance_score=float(dom.get("imbalance_score", 0.0)),
            ),
            key_contributing_signals=data.get("key_contributing_signals", []),
            coaching_feedback=data.get("coaching_feedback", _DEFAULT_COACHING),
            risk_flag=cast(
                Literal["none", "moderate", "high"],
                data.get("risk_flag", "none"),
            ),
        )
    except (KeyError, ValueError, TypeError):
        logger.exception("LLM output had unexpected structure")
        return None


def _risk_flag(stress: Optional[float], aggression: Optional[float]) -> str:
    s = stress or 0.0
    a = aggression or 0.0
    if s > 0.70 or a > 0.70:
        return "high"
    if s > 0.40 or a > 0.40:
        return "moderate"
    return "none"


def _deterministic_fallback(pre: PrecomputedScores) -> AnalysisResult:
    """Build an AnalysisResult from pre-scores when the LLM is unavailable."""
    stress = pre.stress_score
    aggression = pre.aggression_score

    if stress > 0.60:
        coaching = "Slow down. Take a breath before your next answer."
    elif stress > 0.35:
        coaching = "Pause between thoughts. You're doing fine."
    else:
        coaching = "Good pace. Stay focused and keep going."

    return AnalysisResult(
        candidate_stress=stress,
        interviewer_aggression=aggression,
        pressure_trend="stable",
        stress_confidence=pre.stress_confidence,
        aggression_confidence=pre.aggression_confidence,
        dominance_analysis=DominanceAnalysis(
            dominant_party=cast(
                Literal["candidate", "interviewer", "balanced"],
                pre.dominant_party,
            ),
            imbalance_score=pre.imbalance_score,
        ),
        key_contributing_signals=pre.active_stress_signals[:4],
        coaching_feedback=coaching,
        risk_flag=cast(Literal["none", "moderate", "high"], _risk_flag(stress, aggression)),
    )


class BehavioralAnalyzer(Processor):
    """Real-time behavioral intelligence processor for interview analysis.

    Aggregates multimodal metrics over 5-second windows, runs deterministic
    pre-scoring, then optionally calls the LLM for holistic analysis and
    coaching output. Emits BehavioralAnalysisEvent (and SpikeDetectedEvent
    when spikes occur) on the agent's event bus after each cycle.

    Additional intelligence computed each cycle:
        - Micro-spike detection (>2σ deviations from 30s rolling mean)
        - Resilience tracking (recovery time after each spike)
        - Aggression style classification
        - Signal explainability report (audit log)
        - Session summary accumulation (call generate_summary() at session end)

    Args:
        llm: Dedicated LLM instance for analysis (must not be the agent's LLM).
        analysis_interval_s: Seconds between analysis cycles (default 5).
        signal_availability: Initial signal availability flags.

    Raises:
        ValueError: If llm is None.
    """

    name = "behavioral_analyzer"

    def __init__(
        self,
        llm: LLM,
        analysis_interval_s: float = 5.0,
        signal_availability: Optional[SignalAvailability] = None,
    ) -> None:
        if llm is None:
            raise ValueError("llm must not be None")
        self._llm = llm
        self._analysis_interval_s = analysis_interval_s
        self._signal_availability = signal_availability or SignalAvailability()

        self._aggregator = MetricsAggregator(window_size_s=analysis_interval_s)
        self._baseline = BaselineTracker()

        # Intelligence modules
        self._spike_detector = SpikeDetector()
        self._resilience_tracker = ResilienceTracker()
        self._aggression_classifier = AggressionClassifier()
        self._explainer = Explainer()
        self._degradation_engine = GracefulDegradationEngine()
        self._latency_tracker = LatencyTracker()

        # Session accumulation (must call start_session() before use)
        self._session_id: str = "default"
        self._summary_generator: Optional[SessionSummaryGenerator] = None
        self._total_fillers: int = 0

        # Rolling stress history for resilience baseline computation
        self._stress_history: deque[float] = deque(maxlen=60)

        self._analysis_task: Optional[asyncio.Task] = None
        self._analysis_lock = asyncio.Lock()
        self._running = False
        self._agent = None

    def start_session(self, session_id: str) -> None:
        """Initialize per-session accumulators. Call before start().

        Args:
            session_id: Unique identifier for this interview session.
        """
        self._session_id = session_id
        self._summary_generator = SessionSummaryGenerator(session_id)
        self._total_fillers = 0
        logger.info("Session started: %s", session_id)

    def update_signal_availability(self, availability: SignalAvailability) -> None:
        """Update which data sources are currently available.

        Call this when a data source goes offline or comes back online
        during a session. The next analysis cycle will apply the new policy.

        Args:
            availability: Current signal availability flags.
        """
        self._signal_availability = availability

    def attach_agent(self, agent) -> None:
        """Register all emitted events on the agent's event bus."""
        self._agent = agent
        agent.events.register(BehavioralAnalysisEvent)
        agent.events.register(SpikeDetectedEvent)
        agent.events.register(SessionSummaryEvent)

    async def start(self) -> None:
        """Start the background analysis loop and configure the LLM."""
        self._llm.set_instructions(SYSTEM_PROMPT)
        self._running = True
        if self._summary_generator is None:
            self._summary_generator = SessionSummaryGenerator(self._session_id)
        self._analysis_task = asyncio.create_task(
            self._analysis_loop(), name="behavioral_analysis_loop"
        )
        logger.info("BehavioralAnalyzer started (interval=%.1fs)", self._analysis_interval_s)

    async def stop(self) -> None:
        """Stop the background loop and emit a final SessionSummaryEvent."""
        self._running = False
        if self._analysis_task and not self._analysis_task.done():
            self._analysis_task.cancel()
            try:
                await self._analysis_task
            except asyncio.CancelledError:
                pass
        self._emit_session_summary()
        logger.info("BehavioralAnalyzer stopped — latency stats: %s", self._latency_tracker.get_stats())

    async def close(self) -> None:
        await self.stop()

    async def ingest_audio_event(self, filler_count: int = 0, **kwargs) -> None:
        """Feed an audio observation into the current window buffer.

        Args:
            filler_count: Number of filler words detected in this segment.
            **kwargs: Forwarded to MetricsAggregator.ingest_audio_event().
        """
        self._total_fillers += filler_count
        await self._aggregator.ingest_audio_event(filler_count=filler_count, **kwargs)

    async def ingest_video_event(self, **kwargs) -> None:
        """Feed a video/pose observation into the current window buffer."""
        await self._aggregator.ingest_video_event(**kwargs)

    async def ingest_turn_event(self, **kwargs) -> None:
        """Feed a conversation-turn observation into the current window buffer."""
        await self._aggregator.ingest_turn_event(**kwargs)

    def record_turn_completed(self) -> None:
        """Notify the baseline tracker that a full conversation turn has completed."""
        self._baseline.record_turn()

    def hint_spike_cause(self, cause: SpikeCause) -> None:
        """Tag the most likely cause for the next detected spike.

        Call this when an interviewer question is received, an interruption
        occurs, or a silence period begins. The tag is consumed by the next
        detected spike and then reset to UNKNOWN.

        Args:
            cause: Contextual cause to associate with the next spike.
        """
        self._spike_detector.hint_cause(cause)

    def get_latency_stats(self) -> dict[str, dict[str, float]]:
        """Return per-stage latency statistics for the current session."""
        return self._latency_tracker.get_stats()

    def generate_summary(self):
        """Generate the end-of-session summary synchronously.

        Returns:
            SessionSummary dataclass. Call after stop().
        """
        if self._summary_generator is None:
            raise RuntimeError("No session active — call start_session() first")
        self._summary_generator.record_spikes(self._spike_detector.spike_history())
        self._summary_generator.record_filler_total(self._total_fillers)
        self._summary_generator.record_resilience(self._resilience_tracker.compute())
        return self._summary_generator.generate()

    async def _analysis_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self._analysis_interval_s)

            if self._analysis_lock.locked():
                logger.debug("Skipping cycle: previous analysis still in progress")
                continue

            async with self._analysis_lock:
                try:
                    with self._latency_tracker.measure("end_to_end"):
                        await self._run_analysis()
                except asyncio.CancelledError:
                    raise
                except (ValueError, RuntimeError, OSError):
                    logger.exception("Behavioral analysis cycle failed")

    async def _run_analysis(self) -> None:
        with self._latency_tracker.measure("window_flush"):
            payload = await self._aggregator.flush_window()

        self._baseline.record_window(payload.audio, payload.video)

        policy = self._degradation_engine.evaluate(self._signal_availability)
        if policy.warnings:
            logger.info("Degradation policy: %s", policy.warning_message)

        with self._latency_tracker.measure("deterministic_scoring"):
            pre = build_precomputed(
                payload.audio,
                payload.video,
                payload.conversation,
                self._baseline,
            )

        # Apply degradation confidence cap
        pre.stress_confidence = min(pre.stress_confidence, policy.confidence_cap)
        pre.aggression_confidence = min(pre.aggression_confidence, policy.confidence_cap)

        payload.history.baseline_established = self._baseline.established
        payload.history.rolling_avg_stress_estimate = pre.stress_score
        payload.history.rolling_avg_aggression_estimate = pre.aggression_score

        self._aggregator.update_rolling_estimates(pre.stress_score, pre.aggression_score)

        skip_llm = (
            pre.stress_confidence < _MIN_CONFIDENCE_FOR_LLM
            and pre.aggression_confidence < _MIN_CONFIDENCE_FOR_LLM
        )

        if skip_llm:
            logger.debug(
                "Skipping LLM call: low confidence (stress=%.2f, agg=%.2f)",
                pre.stress_confidence,
                pre.aggression_confidence,
            )

        if policy.should_run_llm and not skip_llm:
            with self._latency_tracker.measure("llm_call"):
                result = await self._call_llm_with_retry(payload, pre)
        else:
            result = None

        if result is None:
            result = _deterministic_fallback(pre)

        # Spike detection
        spike = self._spike_detector.ingest(
            stress=result.candidate_stress or 0.0,
            window_id=payload.window_id,
        )
        if spike is not None:
            self._resilience_tracker.on_spike(spike_stress=spike.stress_after)
            self._emit_spike(spike, payload.window_id)

        # Update rolling stress history and resilience tracker
        current_stress = result.candidate_stress or 0.0
        self._stress_history.append(current_stress)
        if len(self._stress_history) >= 5:
            sh = list(self._stress_history)
            s_mean = sum(sh) / len(sh)
            s_var = sum((v - s_mean) ** 2 for v in sh) / len(sh)
            s_std = s_var ** 0.5
            self._resilience_tracker.update(
                current_stress=current_stress,
                baseline_mean=s_mean,
                baseline_std=s_std,
            )

        # Aggression classification
        agg_class = self._aggression_classifier.classify(
            aggression_score=result.interviewer_aggression or 0.0,
            interruptions_per_window=float(
                payload.audio.interruption_count_received or 0
            ),
            question_rate_per_min=float(
                payload.conversation.average_question_length or 0
            ),
            interviewer_dominance=float(
                payload.conversation.speaking_time_ratio_interviewer or 0.5
            ),
            volume_spikes=float(payload.audio.volume_spike_count or 0),
        )

        # Explainability (for audit — not emitted on bus but available via get_last_explanation)
        self._explainer.explain(
            window_id=payload.window_id,
            stress_score=result.candidate_stress or 0.0,
            aggression_score=result.interviewer_aggression or 0.0,
            stress_signal_values=self._build_stress_signal_map(payload),
            aggression_signal_values=self._build_aggression_signal_map(payload),
            confidence_breakdown={
                "stress_confidence": pre.stress_confidence,
                "aggression_confidence": pre.aggression_confidence,
                "baseline_established": float(self._baseline.established),
            },
        )

        with self._latency_tracker.measure("event_emission"):
            self._emit_result(result, payload.window_id, agg_class)

        # Accumulate for session summary
        if self._summary_generator is not None:
            self._summary_generator.record_window(
                WindowRecord(
                    window_id=payload.window_id,
                    timestamp=time.monotonic(),
                    candidate_stress=result.candidate_stress or 0.0,
                    interviewer_aggression=result.interviewer_aggression or 0.0,
                    aggression_class=agg_class.classification.value,
                    risk_flag=result.risk_flag,
                    coaching_feedback=result.coaching_feedback,
                    dominant_party=result.dominance_analysis.dominant_party,
                )
            )

        logger.info(
            "Analysis window=%s stress=%.2f(conf=%.2f) agg=%.2f(%s) risk=%s trend=%s",
            payload.window_id,
            result.candidate_stress or 0.0,
            result.stress_confidence,
            result.interviewer_aggression or 0.0,
            agg_class.classification.value,
            result.risk_flag,
            result.pressure_trend,
        )

    async def _call_llm_with_retry(
        self, payload: BehavioralPayload, pre: PrecomputedScores
    ) -> Optional[AnalysisResult]:
        user_msg = _build_user_message(payload, pre)

        for attempt in range(_MAX_LLM_RETRIES):
            try:
                response = await self._llm.simple_response(user_msg)
                if response.exception:
                    logger.error(
                        "LLM returned error on attempt %d: %s", attempt + 1, response.exception
                    )
                    return None

                result = _parse_result(response.text)
                if result is not None:
                    return result

                logger.warning("LLM output unparseable on attempt %d — retrying", attempt + 1)

            except (RuntimeError, TimeoutError, OSError, ConnectionError):
                logger.exception("LLM call failed on attempt %d", attempt + 1)
                return None

        return None

    def _emit_result(self, result: AnalysisResult, window_id: str, agg_class) -> None:
        if self._agent is None:
            return
        event = BehavioralAnalysisEvent(
            plugin_name=self.name,
            window_id=window_id,
            candidate_stress=result.candidate_stress,
            interviewer_aggression=result.interviewer_aggression,
            pressure_trend=result.pressure_trend,
            stress_confidence=result.stress_confidence,
            aggression_confidence=result.aggression_confidence,
            dominant_party=result.dominance_analysis.dominant_party,
            imbalance_score=result.dominance_analysis.imbalance_score,
            key_signals=result.key_contributing_signals,
            coaching_feedback=result.coaching_feedback,
            risk_flag=result.risk_flag,
            aggression_class=agg_class.classification.value,
            aggression_class_confidence=agg_class.confidence,
        )
        self._agent.events.send(event)

    def _emit_spike(self, spike, window_id: str) -> None:
        if self._agent is None:
            return
        event = SpikeDetectedEvent(
            plugin_name=self.name,
            window_id=window_id,
            stress_before=spike.stress_before,
            stress_after=spike.stress_after,
            delta_sigma=spike.delta_sigma,
            cause=spike.cause.value,
        )
        self._agent.events.send(event)

    def _emit_session_summary(self) -> None:
        if self._agent is None or self._summary_generator is None:
            return
        try:
            self._summary_generator.record_spikes(self._spike_detector.spike_history())
            self._summary_generator.record_filler_total(self._total_fillers)
            self._summary_generator.record_resilience(self._resilience_tracker.compute())
            summary = self._summary_generator.generate()
            event = SessionSummaryEvent(
                plugin_name=self.name,
                session_id=summary.session_id,
                peak_stress=summary.peak_stress,
                average_stress=summary.average_stress,
                resilience_score=summary.resilience_score,
                average_recovery_s=summary.average_recovery_s,
                spike_count=summary.spike_count,
                dominant_aggression_class=summary.dominant_aggression_class,
                filler_word_density_per_min=summary.filler_word_density_per_min,
                dominant_party_summary=summary.dominant_party_summary,
                stress_difficulty_correlation=summary.stress_difficulty_correlation,
                strengths=summary.strengths,
                improvement_areas=summary.improvement_areas,
                improvement_plan=summary.improvement_plan,
            )
            self._agent.events.send(event)
        except ValueError:
            logger.debug("Session summary skipped — no windows recorded")

    def _build_stress_signal_map(self, payload: BehavioralPayload) -> dict[str, float]:
        a = payload.audio
        v = payload.video
        return {
            "response_delay": min((a.avg_response_delay_ms or 0.0) / 5000.0, 1.0),
            "filler_rate": min((a.filler_words_per_min or 0.0) / 20.0, 1.0),
            "speech_rate_change": min(abs(a.speech_rate_change_percent or 0.0) / 100.0, 1.0),
            "pitch_variance": min(a.pitch_variance_score or 0.0, 1.0),
            "silence_before_response": min(
                (a.silence_duration_before_response_ms or 0.0) / 10000.0, 1.0
            ),
            "interruptions_received": min((a.interruption_count_received or 0) / 5.0, 1.0),
            "volume_spikes": min((a.volume_spike_count or 0) / 5.0, 1.0),
            "posture_instability": 1.0 - (v.posture_stability_score or 1.0),
            "shoulder_tension": v.shoulder_tension_score or 0.0,
            "head_jitter": v.head_jitter_score or 0.0,
            "repetitive_movement": v.repetitive_hand_movement_score or 0.0,
        }

    def _build_aggression_signal_map(self, payload: BehavioralPayload) -> dict[str, float]:
        c = payload.conversation
        a = payload.audio
        return {
            "interviewer_interruptions": min(
                (a.interruption_count_given or 0) / 5.0, 1.0
            ),
            "dominance_imbalance": c.dominance_score or 0.0,
            "speaking_ratio_imbalance": abs(
                (c.speaking_time_ratio_interviewer or 0.5) - 0.5
            ) * 2.0,
            "rapid_questioning": min((c.average_question_length or 0.0) / 50.0, 1.0),
            "volume_spikes": min((a.volume_spike_count or 0) / 5.0, 1.0),
        }

    async def __aenter__(self) -> "BehavioralAnalyzer":
        await self.start()
        return self

    async def __aexit__(self, *_) -> None:
        await self.stop()
