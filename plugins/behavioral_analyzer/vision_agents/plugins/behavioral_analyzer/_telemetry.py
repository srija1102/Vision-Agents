"""Optional OpenTelemetry latency instrumentation for the behavioral analyzer.

Tracks per-stage latency with configurable budget thresholds and logs
warnings when budgets are exceeded. Degrades gracefully if the
opentelemetry package is not installed.

Stage budgets (ms):
    pose_inference       50
    stt_processing      200
    window_flush          5
    deterministic_scoring 2
    llm_call           1500
    event_emission        1
    end_to_end         2000
"""

import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

# Per-stage latency budgets in milliseconds.
LATENCY_BUDGETS_MS: dict[str, float] = {
    "pose_inference": 50.0,
    "stt_processing": 200.0,
    "window_flush": 5.0,
    "deterministic_scoring": 2.0,
    "llm_call": 1500.0,
    "event_emission": 1.0,
    "end_to_end": 2000.0,
}

try:
    from opentelemetry import metrics as otel_metrics

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False
    logger.debug("opentelemetry not installed — telemetry falls back to in-memory stats")


class LatencyTracker:
    """Measures and records per-stage latency with optional OTEL export.

    Maintains an in-memory list of timing samples per stage and exports
    to an OTEL histogram when a provider is configured. Budget violations
    are logged at WARNING level.

    Usage::

        tracker = LatencyTracker()

        with tracker.measure("llm_call"):
            result = await llm.simple_response(msg)

        stats = tracker.get_stats()
        # {"llm_call": {"mean_ms": 820.3, "p95_ms": 1340.1, "max_ms": 1490.0, "count": 12.0}}

    Args:
        service_name: OTEL service name for histogram labelling.
    """

    def __init__(self, service_name: str = "behavioral_analyzer") -> None:
        self._service_name = service_name
        self._samples: dict[str, list[float]] = defaultdict(list)
        self._histogram: Optional[object] = None

        if _OTEL_AVAILABLE:
            try:
                meter = otel_metrics.get_meter(service_name)
                self._histogram = meter.create_histogram(
                    name="behavioral_analyzer.stage_latency_ms",
                    description="Per-stage pipeline latency in milliseconds",
                    unit="ms",
                )
            except RuntimeError:
                # No OTEL MeterProvider configured — silent degradation
                pass

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        """Context manager that times a code block and records the result.

        Args:
            stage: Stage identifier (should match a key in LATENCY_BUDGETS_MS
                   for budget enforcement, but any string is accepted).

        Example::

            with tracker.measure("deterministic_scoring"):
                pre = build_precomputed(audio, video, conversation, baseline)
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._record(stage, elapsed_ms)

    def _record(self, stage: str, elapsed_ms: float) -> None:
        self._samples[stage].append(elapsed_ms)

        if self._histogram is not None:
            try:
                self._histogram.record(elapsed_ms, {"stage": stage})
            except RuntimeError:
                pass

        budget = LATENCY_BUDGETS_MS.get(stage)
        if budget is not None and elapsed_ms > budget:
            logger.warning(
                "Latency budget exceeded: stage=%s elapsed=%.1fms budget=%.1fms",
                stage,
                elapsed_ms,
                budget,
            )

    def get_stats(self) -> dict[str, dict[str, float]]:
        """Return per-stage latency statistics.

        Returns:
            Dict keyed by stage name, each containing:
                mean_ms, max_ms, p95_ms, count.
        """
        result: dict[str, dict[str, float]] = {}
        for stage, samples in self._samples.items():
            if not samples:
                continue
            sorted_s = sorted(samples)
            n = len(sorted_s)
            p95_idx = max(0, int(n * 0.95) - 1)
            result[stage] = {
                "mean_ms": round(sum(samples) / n, 2),
                "max_ms": round(max(samples), 2),
                "p95_ms": round(sorted_s[p95_idx], 2),
                "count": float(n),
            }
        return result

    def prometheus_text(self) -> str:
        """Format latency stats as Prometheus text exposition format.

        Returns:
            Multi-line string in Prometheus text format.
        """
        lines: list[str] = []
        stats = self.get_stats()
        for stage, s in stats.items():
            base = f'behavioral_analyzer_stage_latency_ms{{stage="{stage}"}}'
            lines += [
                f"# HELP {base} Stage latency in ms",
                f"# TYPE {base} gauge",
                f"{base}_mean {s['mean_ms']}",
                f"{base}_max {s['max_ms']}",
                f"{base}_p95 {s['p95_ms']}",
                f"{base}_count {s['count']}",
            ]
        return "\n".join(lines)
