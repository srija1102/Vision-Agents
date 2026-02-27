"""InterviewBridge — routes STT and turn events into BehavioralAnalyzer metrics.

Subscribes to agent events in attach_agent() and derives the following:
- Words per minute, filler word count, speech rate change (from STT transcripts)
- Response delay (time from interviewer turn end → candidate turn start)
- Silence before response (from TurnEndedEvent.trailing_silence_ms)
- Interruption detection (interviewer starts while candidate is still in turn)
- Speaking time ratios (accumulated per turn)
"""

import logging
import time
from collections import deque
from typing import TYPE_CHECKING

from vision_agents.core.processors.base_processor import Processor
from vision_agents.core.stt.events import STTTranscriptEvent
from vision_agents.core.turn_detection.events import TurnEndedEvent, TurnStartedEvent

from .analyzer import BehavioralAnalyzer

if TYPE_CHECKING:
    from vision_agents.core.agents import Agent

logger = logging.getLogger(__name__)

_FILLER_SINGLE = frozenset({
    "um", "uh", "erm", "hmm", "ah",
    "like", "basically", "literally", "actually", "honestly",
    "right", "okay", "so", "well", "anyway",
})
_FILLER_PHRASES = ("you know", "i mean", "kind of", "sort of")
_WPM_HISTORY_SIZE = 20


def _count_fillers(text: str) -> int:
    lowered = text.lower()
    words = lowered.split()
    count = sum(1 for w in words if w in _FILLER_SINGLE)
    for phrase in _FILLER_PHRASES:
        count += lowered.count(phrase)
    return count


class InterviewBridge(Processor):
    """Bridges STT and turn events into BehavioralAnalyzer metric feeds.

    Attaches to the agent event bus and translates real-time transcript and
    turn events into structured metrics for behavioral analysis.

    Args:
        analyzer: The BehavioralAnalyzer instance to feed metrics into.
        candidate_user_id: The user_id of the candidate participant. If None,
            the first non-agent participant encountered is treated as the candidate.

    Raises:
        ValueError: If analyzer is None.
    """

    name = "interview_bridge"

    def __init__(
        self,
        analyzer: BehavioralAnalyzer,
        candidate_user_id: str | None = None,
    ) -> None:
        if analyzer is None:
            raise ValueError("analyzer must not be None")
        self._analyzer = analyzer
        self._candidate_user_id = candidate_user_id
        self._agent: "Agent | None" = None

        self._candidate_in_turn: bool = False
        self._candidate_speaking_ms: float = 0.0
        self._interviewer_speaking_ms: float = 0.0
        self._last_interviewer_turn_end: float | None = None
        self._wpm_history: deque[float] = deque(maxlen=_WPM_HISTORY_SIZE)

    @property
    def _agent_user_id(self) -> str | None:
        if self._agent is None:
            return None
        return self._agent.agent_user.id

    def _is_candidate(self, user_id: str | None) -> bool:
        if user_id is None:
            return False
        if self._candidate_user_id is not None:
            return user_id == self._candidate_user_id
        return user_id != self._agent_user_id

    def _is_interviewer(self, user_id: str | None) -> bool:
        if user_id is None:
            return False
        return not self._is_candidate(user_id)

    def attach_agent(self, agent: "Agent") -> None:
        """Subscribe to STT transcript and turn events on the agent event bus."""
        self._agent = agent

        @agent.events.subscribe
        async def on_transcript(event: STTTranscriptEvent):
            await self._handle_transcript(event)

        @agent.events.subscribe
        async def on_turn_started(event: TurnStartedEvent):
            await self._handle_turn_started(event)

        @agent.events.subscribe
        async def on_turn_ended(event: TurnEndedEvent):
            await self._handle_turn_ended(event)

        logger.info("InterviewBridge attached (candidate_user_id=%s)", self._candidate_user_id)

    async def _handle_transcript(self, event: STTTranscriptEvent) -> None:
        user_id = event.participant.user_id if event.participant else None
        if not self._is_candidate(user_id):
            return

        text = event.text.strip()
        if not text:
            return

        word_count = len(text.split())
        audio_ms = event.response.audio_duration_ms
        filler_count = _count_fillers(text)

        wpm: float | None = None
        if audio_ms and audio_ms > 0:
            wpm = word_count / (audio_ms / 60_000.0)

        speech_rate_change: float | None = None
        if wpm is not None and len(self._wpm_history) >= 3:
            baseline = sum(self._wpm_history) / len(self._wpm_history)
            if baseline > 0:
                speech_rate_change = ((wpm - baseline) / baseline) * 100.0
        if wpm is not None:
            self._wpm_history.append(wpm)

        await self._analyzer.ingest_audio_event(
            words_this_window=word_count,
            filler_count=filler_count,
            speech_rate_change_pct=speech_rate_change,
        )
        self._analyzer.record_turn_completed()

        logger.debug(
            "transcript: words=%d wpm=%s fillers=%d rate_delta=%s",
            word_count,
            f"{wpm:.0f}" if wpm else "n/a",
            filler_count,
            f"{speech_rate_change:+.1f}%" if speech_rate_change else "n/a",
        )

    async def _handle_turn_started(self, event: TurnStartedEvent) -> None:
        user_id = event.participant.user_id if event.participant else None

        if self._is_candidate(user_id):
            self._candidate_in_turn = True

            if self._last_interviewer_turn_end is not None:
                delay_ms = (time.monotonic() - self._last_interviewer_turn_end) * 1000.0
                self._last_interviewer_turn_end = None
                await self._analyzer.ingest_audio_event(response_delay_ms=delay_ms)
                logger.debug("response_delay=%.0fms", delay_ms)

        elif self._is_interviewer(user_id) and self._candidate_in_turn:
            await self._analyzer.ingest_audio_event(interrupted=True)
            logger.debug("interruption detected: interviewer started while candidate in turn")

    async def _handle_turn_ended(self, event: TurnEndedEvent) -> None:
        user_id = event.participant.user_id if event.participant else None
        duration_ms = event.duration_ms or 0.0

        if self._is_candidate(user_id):
            self._candidate_in_turn = False
            self._candidate_speaking_ms += duration_ms

            if event.trailing_silence_ms:
                await self._analyzer.ingest_audio_event(silence_ms=event.trailing_silence_ms)

        elif self._is_interviewer(user_id):
            self._interviewer_speaking_ms += duration_ms
            self._last_interviewer_turn_end = time.monotonic()

            if self._candidate_in_turn:
                await self._analyzer.ingest_turn_event(interviewer_interrupted=True)

        total_ms = self._candidate_speaking_ms + self._interviewer_speaking_ms
        if total_ms > 0:
            await self._analyzer.ingest_turn_event(
                candidate_speaking_ms=self._candidate_speaking_ms,
                interviewer_speaking_ms=self._interviewer_speaking_ms,
            )
            self._candidate_speaking_ms = 0.0
            self._interviewer_speaking_ms = 0.0

    async def close(self) -> None:
        pass
