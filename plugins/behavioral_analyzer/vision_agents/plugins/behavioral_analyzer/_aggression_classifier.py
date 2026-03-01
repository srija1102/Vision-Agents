"""Aggression style classifier for interviewer behavioral signals.

Classifies interviewer aggression into one of five categories using
deterministic signal thresholds. The classification is computed from
existing pre-scored signals — no additional inference required.
"""

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AggressionClass(str, Enum):
    """Interviewer aggression style categories.

    Ordered from least to most severe.
    """

    NEUTRAL = "neutral"
    CHALLENGING = "challenging"
    RAPID_FIRE = "rapid_fire"
    INTERRUPTIVE = "interruptive"
    HOSTILE = "hostile"


@dataclass
class AggressionClassification:
    """Result of aggression style classification.

    Attributes:
        classification: Detected aggression class.
        confidence: Classification confidence [0, 1].
        primary_signal: The dominant signal that drove the classification.
        aggression_score: Underlying numeric score from the pre-scorer.
    """

    classification: AggressionClass
    confidence: float
    primary_signal: str
    aggression_score: float


class AggressionClassifier:
    """Classifies interviewer aggression style from pre-computed behavioral signals.

    Evaluation proceeds in priority order from most severe to least.
    The first matching rule wins; all thresholds are configurable at the
    class level for easy tuning without subclassing.

    Classification rules (priority order):
        1. HOSTILE    — high aggression score AND multiple volume spikes
        2. INTERRUPTIVE — frequent interruptions per window
        3. RAPID_FIRE  — high question rate per minute
        4. CHALLENGING  — elevated aggression score or dominance imbalance
        5. NEUTRAL     — none of the above

    Args: none. Construct once and call classify() per analysis window.
    """

    HOSTILE_SCORE_THRESHOLD: float = 0.75
    HOSTILE_VOLUME_SPIKES: int = 2

    INTERRUPTIVE_THRESHOLD: float = 3.0  # interruptions per 5s window
    RAPID_FIRE_THRESHOLD: float = 2.0    # questions per minute
    CHALLENGING_SCORE_THRESHOLD: float = 0.40
    DOMINANCE_IMBALANCE_THRESHOLD: float = 0.65

    def classify(
        self,
        aggression_score: float,
        interruptions_per_window: float,
        question_rate_per_min: float,
        interviewer_dominance: float,
        volume_spikes: float,
    ) -> AggressionClassification:
        """Classify aggression type from behavioral signals.

        Args:
            aggression_score: Pre-computed aggression score [0, 1].
            interruptions_per_window: Interruption count in current 5s window.
            question_rate_per_min: Questions asked per minute by interviewer.
            interviewer_dominance: Interviewer speaking ratio [0, 1].
            volume_spikes: Volume spike count in current window.

        Returns:
            AggressionClassification with class, confidence, and primary signal.
        """
        if (
            aggression_score >= self.HOSTILE_SCORE_THRESHOLD
            and volume_spikes >= self.HOSTILE_VOLUME_SPIKES
        ):
            return AggressionClassification(
                classification=AggressionClass.HOSTILE,
                confidence=min(aggression_score + 0.10, 1.0),
                primary_signal="high_aggression_score_with_volume_spikes",
                aggression_score=aggression_score,
            )

        if interruptions_per_window >= self.INTERRUPTIVE_THRESHOLD:
            confidence = min(interruptions_per_window / 5.0, 1.0)
            return AggressionClassification(
                classification=AggressionClass.INTERRUPTIVE,
                confidence=round(confidence, 4),
                primary_signal="high_interruption_count",
                aggression_score=aggression_score,
            )

        if question_rate_per_min >= self.RAPID_FIRE_THRESHOLD:
            confidence = min(question_rate_per_min / 4.0, 1.0)
            return AggressionClassification(
                classification=AggressionClass.RAPID_FIRE,
                confidence=round(confidence, 4),
                primary_signal="high_question_rate",
                aggression_score=aggression_score,
            )

        challenging_by_score = aggression_score >= self.CHALLENGING_SCORE_THRESHOLD
        challenging_by_dominance = interviewer_dominance >= self.DOMINANCE_IMBALANCE_THRESHOLD
        if challenging_by_score or challenging_by_dominance:
            primary = (
                "aggression_score" if challenging_by_score else "dominance_imbalance"
            )
            confidence = max(aggression_score, interviewer_dominance * 0.80)
            return AggressionClassification(
                classification=AggressionClass.CHALLENGING,
                confidence=round(confidence, 4),
                primary_signal=primary,
                aggression_score=aggression_score,
            )

        return AggressionClassification(
            classification=AggressionClass.NEUTRAL,
            confidence=round(1.0 - aggression_score, 4),
            primary_signal="no_dominant_signal",
            aggression_score=aggression_score,
        )
