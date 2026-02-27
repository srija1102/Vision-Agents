"""BehavioralAnalyzer — production-grade real-time interview pressure processor.

Architecture overview:
  1. Upstream producers call ingest_audio_event / ingest_video_event /
     ingest_turn_event to feed raw observations into the 5-second buffer.
  2. Every analysis_interval_s, _analysis_loop flushes the buffer and runs
     the deterministic pre-scorer (_scorer.py).
  3. If confidence is high enough, the pre-scores are sent to the LLM alongside
     the aggregated metrics.  The LLM returns JSON output.
  4. The parsed result (or a deterministic fallback on LLM failure) is emitted
     as a BehavioralAnalysisEvent on the agent's event bus.

LLM usage notes:
  - Pass a DEDICATED LLM instance — not the one used by the main agent.
    The analyzer calls set_instructions() on the LLM at startup.
  - The LLM is called at most once per analysis_interval_s.
  - If a previous LLM call is still in progress, the next cycle is skipped
    rather than queued, preventing cascading latency.
"""

import asyncio
import dataclasses
import json
import logging
from typing import Literal, cast

from vision_agents.core.llm import LLM
from vision_agents.core.processors import Processor

from ._aggregator import MetricsAggregator
from ._baseline import BaselineTracker
from ._prompt import SYSTEM_PROMPT
from ._scorer import build_precomputed
from ._types import (
    AnalysisResult,
    BehavioralPayload,
    DominanceAnalysis,
    PrecomputedScores,
)
from .events import BehavioralAnalysisEvent

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


def _parse_result(raw: str) -> AnalysisResult | None:
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


def _risk_flag(stress: float | None, aggression: float | None) -> str:
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
    pre-scoring, then calls the LLM for holistic analysis and coaching output.
    Emits BehavioralAnalysisEvent on the agent's event bus after each cycle.

    Args:
        llm: Dedicated LLM instance for analysis (must not be the agent's LLM).
        analysis_interval_s: Seconds between analysis cycles (default 5).

    Raises:
        ValueError: If llm is None.
    """

    name = "behavioral_analyzer"

    def __init__(self, llm: LLM, analysis_interval_s: float = 5.0) -> None:
        if llm is None:
            raise ValueError("llm must not be None")
        self._llm = llm
        self._analysis_interval_s = analysis_interval_s
        self._aggregator = MetricsAggregator(window_size_s=analysis_interval_s)
        self._baseline = BaselineTracker()
        self._analysis_task: asyncio.Task | None = None
        self._analysis_lock = asyncio.Lock()
        self._running = False
        self._agent = None

    def attach_agent(self, agent) -> None:
        """Register the BehavioralAnalysisEvent on the agent's event bus."""
        self._agent = agent
        agent.events.register(BehavioralAnalysisEvent)

    async def start(self) -> None:
        """Start the background analysis loop and configure the LLM."""
        self._llm.set_instructions(SYSTEM_PROMPT)
        self._running = True
        self._analysis_task = asyncio.create_task(
            self._analysis_loop(), name="behavioral_analysis_loop"
        )
        logger.info("BehavioralAnalyzer started (interval=%.1fs)", self._analysis_interval_s)

    async def stop(self) -> None:
        """Stop the background loop and wait for it to exit cleanly."""
        self._running = False
        if self._analysis_task and not self._analysis_task.done():
            self._analysis_task.cancel()
            try:
                await self._analysis_task
            except asyncio.CancelledError:
                pass
        logger.info("BehavioralAnalyzer stopped")

    async def close(self) -> None:
        await self.stop()

    async def ingest_audio_event(self, **kwargs) -> None:
        """Feed an audio observation into the current window buffer.

        Keyword args mirror MetricsAggregator.ingest_audio_event().
        """
        await self._aggregator.ingest_audio_event(**kwargs)

    async def ingest_video_event(self, **kwargs) -> None:
        """Feed a video/pose observation into the current window buffer.

        Keyword args mirror MetricsAggregator.ingest_video_event().
        """
        await self._aggregator.ingest_video_event(**kwargs)

    async def ingest_turn_event(self, **kwargs) -> None:
        """Feed a conversation-turn observation into the current window buffer.

        Keyword args mirror MetricsAggregator.ingest_turn_event().
        """
        await self._aggregator.ingest_turn_event(**kwargs)

    def record_turn_completed(self) -> None:
        """Notify the baseline tracker that a full conversation turn has completed."""
        self._baseline.record_turn()

    async def _analysis_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self._analysis_interval_s)

            if self._analysis_lock.locked():
                logger.debug("Skipping cycle: previous analysis still in progress")
                continue

            async with self._analysis_lock:
                try:
                    await self._run_analysis()
                except asyncio.CancelledError:
                    raise
                except (ValueError, RuntimeError, OSError):
                    logger.exception("Behavioral analysis cycle failed")

    async def _run_analysis(self) -> None:
        payload = await self._aggregator.flush_window()

        self._baseline.record_window(payload.audio, payload.video)

        pre = build_precomputed(
            payload.audio,
            payload.video,
            payload.conversation,
            self._baseline,
        )

        payload.history.baseline_established = self._baseline.established
        payload.history.rolling_avg_stress_estimate = pre.stress_score
        payload.history.rolling_avg_aggression_estimate = pre.aggression_score

        self._aggregator.update_rolling_estimates(pre.stress_score, pre.aggression_score)

        if pre.stress_confidence < _MIN_CONFIDENCE_FOR_LLM and pre.aggression_confidence < _MIN_CONFIDENCE_FOR_LLM:
            logger.debug(
                "Skipping LLM call: low confidence (stress=%.2f, agg=%.2f)",
                pre.stress_confidence,
                pre.aggression_confidence,
            )
            return

        result = await self._call_llm_with_retry(payload, pre)
        if result is None:
            result = _deterministic_fallback(pre)

        self._emit_result(result, payload.window_id)

        logger.info(
            "Analysis window=%s stress=%.2f(conf=%.2f) agg=%.2f risk=%s trend=%s",
            payload.window_id,
            result.candidate_stress or 0.0,
            result.stress_confidence,
            result.interviewer_aggression or 0.0,
            result.risk_flag,
            result.pressure_trend,
        )

    async def _call_llm_with_retry(
        self, payload: BehavioralPayload, pre: PrecomputedScores
    ) -> AnalysisResult | None:
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

                logger.warning(
                    "LLM output unparseable on attempt %d — retrying", attempt + 1
                )

            except (RuntimeError, TimeoutError, OSError, ConnectionError):
                logger.exception("LLM call failed on attempt %d", attempt + 1)
                return None

        return None

    def _emit_result(self, result: AnalysisResult, window_id: str) -> None:
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
        )
        self._agent.events.send(event)

    async def __aenter__(self) -> "BehavioralAnalyzer":
        await self.start()
        return self

    async def __aexit__(self, *_) -> None:
        await self.stop()
