"""Question difficulty analysis and stress correlation.

Classifies question complexity using deterministic NLP (no model inference):
  - token length
  - technical keyword density
  - syntactic clause depth

Correlates observed complexity with candidate stress using Pearson r.
All question analysis is cached with lru_cache to avoid redundant computation.
"""

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

_TECHNICAL_KEYWORDS: frozenset[str] = frozenset(
    {
        # Software engineering
        "algorithm",
        "complexity",
        "runtime",
        "async",
        "thread",
        "mutex",
        "deadlock",
        "distributed",
        "latency",
        "throughput",
        "concurrency",
        "cache",
        "index",
        "schema",
        "normalization",
        "transaction",
        "idempotent",
        "eventual",
        "consensus",
        "microservice",
        "kubernetes",
        "docker",
        "container",
        "orchestration",
        # Machine learning
        "regression",
        "gradient",
        "tensor",
        "embedding",
        "inference",
        "overfit",
        "hyperparameter",
        "precision",
        "recall",
        "f1",
        "auc",
        "roc",
        "transformer",
        "attention",
        "fine-tuning",
        # Business/strategy
        "stakeholder",
        "tradeoff",
        "prioritize",
        "negotiate",
        "escalate",
        "roadmap",
        "kpi",
        "roi",
        "runway",
        # Finance
        "valuation",
        "amortize",
        "leverage",
        "liquidity",
        "volatility",
        "arbitrage",
        "derivative",
        "equity",
    }
)

# Patterns that indicate nested or compound clause structure
_CLAUSE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(if|when|assuming|suppose|given that|in the case|provided that)\b", re.I),
    re.compile(r"\b(and|but|however|whereas|although|despite|even though)\b", re.I),
    re.compile(r"\b(because|since|therefore|thus|hence|as a result)\b", re.I),
    re.compile(r"\b(first|then|finally|additionally|furthermore|moreover)\b", re.I),
]


@dataclass
class QuestionComplexity:
    """Complexity analysis for a single question.

    Attributes:
        question: Original question text.
        token_count: Number of whitespace-delimited tokens.
        technical_keyword_count: Count of domain-specific terms.
        syntactic_depth: Number of distinct clause-pattern types matched.
        complexity_score: Weighted composite score [0, 1].
    """

    question: str
    token_count: int
    technical_keyword_count: int
    syntactic_depth: int
    complexity_score: float


@dataclass
class CorrelationResult:
    """Pearson correlation between question complexity and stress.

    Attributes:
        pearson_r: Pearson correlation coefficient [-1, 1].
        stress_by_complexity_bucket: Mean stress per complexity bucket
            (low / medium / high).
        observation_count: Number of (complexity, stress) pairs used.
    """

    pearson_r: float
    stress_by_complexity_bucket: dict[str, float]
    observation_count: int


@lru_cache(maxsize=512)
def _analyze_question_cached(question: str) -> QuestionComplexity:
    """Cached deterministic NLP analysis of a question.

    Args:
        question: Raw question text.

    Returns:
        QuestionComplexity for the question.
    """
    tokens = question.lower().split()
    token_count = len(tokens)
    cleaned = [t.strip(".,?!;:'\"") for t in tokens]
    tech_count = sum(1 for t in cleaned if t in _TECHNICAL_KEYWORDS)
    tech_density = tech_count / max(token_count, 1)
    depth = sum(1 for p in _CLAUSE_PATTERNS if p.search(question))

    # Each dimension normalized to [0, 1]
    length_score = min(token_count / 50.0, 1.0)        # 50 tokens → full score
    tech_score = min(tech_density * 5.0, 1.0)           # 20% keyword density → full
    depth_score = min(depth / len(_CLAUSE_PATTERNS), 1.0)

    complexity_score = 0.40 * length_score + 0.35 * tech_score + 0.25 * depth_score

    return QuestionComplexity(
        question=question,
        token_count=token_count,
        technical_keyword_count=tech_count,
        syntactic_depth=depth,
        complexity_score=round(complexity_score, 4),
    )


class QuestionDifficultyAnalyzer:
    """Correlates question complexity with candidate stress responses.

    Complexity is computed deterministically — no model inference.
    Results are cached per unique question text.

    Usage::

        analyzer = QuestionDifficultyAnalyzer()
        analyzer.record("Tell me about a time you failed.", stress=0.42)
        analyzer.record("Design a distributed rate limiter.", stress=0.71)
        result = analyzer.correlate()  # pearson_r, bucket breakdown
    """

    def __init__(self) -> None:
        self._observations: list[tuple[float, float]] = []  # (complexity, stress)

    def record(self, question: str, stress_at_response: float) -> QuestionComplexity:
        """Record a question and the stress observed during the candidate's response.

        Args:
            question: Raw question text from transcript.
            stress_at_response: Stress score [0, 1] during response window.

        Returns:
            QuestionComplexity for the question.
        """
        complexity = _analyze_question_cached(question)
        self._observations.append((complexity.complexity_score, stress_at_response))
        return complexity

    def correlate(self) -> Optional[CorrelationResult]:
        """Compute Pearson r between question complexity and stress.

        Returns:
            CorrelationResult if >= 3 observations exist, else None.
        """
        n = len(self._observations)
        if n < 3:
            return None

        xs = [o[0] for o in self._observations]
        ys = [o[1] for o in self._observations]

        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        cov = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n)) / n
        std_x = (sum((x - mean_x) ** 2 for x in xs) / n) ** 0.5
        std_y = (sum((y - mean_y) ** 2 for y in ys) / n) ** 0.5

        if std_x < 1e-9 or std_y < 1e-9:
            pearson_r = 0.0
        else:
            pearson_r = cov / (std_x * std_y)
            pearson_r = max(-1.0, min(1.0, pearson_r))

        buckets: dict[str, list[float]] = {"low": [], "medium": [], "high": []}
        for complexity, stress in self._observations:
            if complexity < 0.33:
                buckets["low"].append(stress)
            elif complexity < 0.67:
                buckets["medium"].append(stress)
            else:
                buckets["high"].append(stress)

        stress_by_bucket = {
            k: round(sum(v) / len(v), 4) if v else 0.0 for k, v in buckets.items()
        }

        return CorrelationResult(
            pearson_r=round(pearson_r, 4),
            stress_by_complexity_bucket=stress_by_bucket,
            observation_count=n,
        )

    def observation_count(self) -> int:
        """Return number of recorded (question, stress) pairs."""
        return len(self._observations)
