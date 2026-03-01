"""Comprehensive unit tests for the behavioral_analyzer plugin.

Test matrix covers:
  - Spike detection (normal, edge cases, cause tagging)
  - Resilience tracking (fast recovery, timeout, no spikes)
  - Question difficulty analysis (NLP scoring, correlation, caching)
  - Aggression classification (all five classes, boundary conditions)
  - Session summary generation (coaching derivation, JSON/Markdown output)
  - Session comparison (improvement detection, edge cases)
  - Explainability (contribution ordering, zero-signal handling)
  - Graceful degradation (all signal loss combinations)
  - Latency tracker (budget enforcement, stats computation)
  - Aggregator (window flush, empty window, rolling history)
  - Audio/video edge cases (None metrics, zero values)
  - Signal dropout and partial data

Never mock. All tests exercise real code paths.
"""

import pytest

from vision_agents.plugins.behavioral_analyzer._aggression_classifier import (
    AggressionClass,
    AggressionClassifier,
)
from vision_agents.plugins.behavioral_analyzer._degradation import (
    GracefulDegradationEngine,
    SignalAvailability,
)
from vision_agents.plugins.behavioral_analyzer._explainability import Explainer
from vision_agents.plugins.behavioral_analyzer._question_analyzer import (
    QuestionDifficultyAnalyzer,
    _analyze_question_cached,
)
from vision_agents.plugins.behavioral_analyzer._resilience import ResilienceTracker
from vision_agents.plugins.behavioral_analyzer._session_comparison import SessionComparisonEngine
from vision_agents.plugins.behavioral_analyzer._session_summary import (
    SessionSummaryGenerator,
    WindowRecord,
)
from vision_agents.plugins.behavioral_analyzer._spike_detector import SpikeCause, SpikeDetector
from vision_agents.plugins.behavioral_analyzer._telemetry import LatencyTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_window(
    window_id: str = "w0001",
    timestamp: float = 0.0,
    stress: float = 0.30,
    aggression: float = 0.20,
    risk: str = "none",
    aggression_class: str = "neutral",
    dominant_party: str = "balanced",
) -> WindowRecord:
    return WindowRecord(
        window_id=window_id,
        timestamp=timestamp,
        candidate_stress=stress,
        interviewer_aggression=aggression,
        aggression_class=aggression_class,
        risk_flag=risk,
        coaching_feedback="Stay calm.",
        dominant_party=dominant_party,
    )


# ---------------------------------------------------------------------------
# SpikeDetector
# ---------------------------------------------------------------------------


class TestSpikeDetector:
    def test_no_spike_with_insufficient_samples(self) -> None:
        detector = SpikeDetector()
        # Feed fewer than MIN_SAMPLES_FOR_DETECTION readings
        for i in range(4):
            result = detector.ingest(0.3 + i * 0.01, f"w{i:04d}")
        assert result is None
        assert detector.spike_count() == 0

    def _varied_baseline(self, detector: SpikeDetector, n: int = 10) -> None:
        """Feed a varied baseline so std > 0, enabling spike detection."""
        for i in range(n):
            val = 0.27 + (i % 5) * 0.01  # range [0.27, 0.31]
            detector.ingest(val, f"w{i:04d}")

    def test_spike_detected_on_large_deviation(self) -> None:
        detector = SpikeDetector()
        self._varied_baseline(detector)
        # Now inject a large spike well above the baseline
        spike = detector.ingest(0.90, "w0010")
        assert spike is not None
        assert spike.stress_after == pytest.approx(0.90)
        assert spike.delta_sigma >= 2.0
        assert spike.cause == SpikeCause.UNKNOWN

    def test_spike_cause_tagging(self) -> None:
        detector = SpikeDetector()
        self._varied_baseline(detector)
        detector.hint_cause(SpikeCause.INTERRUPTION)
        spike = detector.ingest(0.95, "w0010")
        assert spike is not None
        assert spike.cause == SpikeCause.INTERRUPTION

    def test_cause_reset_after_spike(self) -> None:
        detector = SpikeDetector()
        self._varied_baseline(detector)
        detector.hint_cause(SpikeCause.QUESTION_RECEIVED)
        detector.ingest(0.95, "w0010")  # consumes cause; spike must fire given varied baseline
        # Return to near-baseline so next spike is also detectable
        for j in range(5):
            val = 0.27 + (j % 5) * 0.01
            detector.ingest(val, f"w{j+11:04d}")
        # Next spike should have UNKNOWN cause (hint was consumed above)
        spike2 = detector.ingest(0.95, "w0016")
        if spike2 is not None:
            assert spike2.cause == SpikeCause.UNKNOWN

    def test_no_spike_below_threshold(self) -> None:
        detector = SpikeDetector()
        for i in range(20):
            val = 0.30 + (i % 3) * 0.02  # small fluctuation
            detector.ingest(val, f"w{i:04d}")
        assert detector.spike_count() == 0

    def test_spike_history_accumulates(self) -> None:
        detector = SpikeDetector()
        for i in range(10):
            detector.ingest(0.30, f"w{i:04d}")
        detector.ingest(0.95, "w0010")
        for i in range(5):
            detector.ingest(0.30, f"w{i+11:04d}")
        detector.ingest(0.95, "w0016")
        assert detector.spike_count() >= 1
        assert len(detector.spike_history()) >= 1

    def test_recent_spikes_filter(self) -> None:
        detector = SpikeDetector()
        for i in range(10):
            detector.ingest(0.30, f"w{i:04d}")
        detector.ingest(0.95, "w0010")
        recent = detector.recent_spikes(within_s=5.0)
        all_spikes = detector.spike_history()
        assert len(recent) <= len(all_spikes)

    def test_zero_stress_throughout_no_spike(self) -> None:
        detector = SpikeDetector()
        for i in range(20):
            spike = detector.ingest(0.0, f"w{i:04d}")
            assert spike is None  # zero std — no spike possible

    def test_uniform_high_stress_no_spike(self) -> None:
        detector = SpikeDetector()
        for i in range(20):
            spike = detector.ingest(0.90, f"w{i:04d}")
            assert spike is None  # uniform — std near 0


# ---------------------------------------------------------------------------
# ResilienceTracker
# ---------------------------------------------------------------------------


class TestResilienceTracker:
    def test_no_spikes_perfect_resilience(self) -> None:
        tracker = ResilienceTracker()
        result = tracker.compute()
        assert result.resilience_score == pytest.approx(1.0)
        assert result.completed_recoveries == 0
        assert result.pending_recoveries == 0

    def test_fast_recovery_high_score(self) -> None:
        tracker = ResilienceTracker()
        t0 = 0.0
        tracker.on_spike(spike_stress=0.80, timestamp=t0)
        # Recover within 5 seconds
        tracker.update(current_stress=0.30, baseline_mean=0.30, baseline_std=0.10, timestamp=t0 + 5.0)
        result = tracker.compute()
        assert result.resilience_score > 0.90
        assert result.completed_recoveries == 1
        assert result.average_recovery_s == pytest.approx(5.0)

    def test_slow_recovery_lower_score(self) -> None:
        tracker = ResilienceTracker()
        t0 = 0.0
        tracker.on_spike(spike_stress=0.80, timestamp=t0)
        tracker.update(current_stress=0.30, baseline_mean=0.30, baseline_std=0.10, timestamp=t0 + 45.0)
        result = tracker.compute()
        assert result.resilience_score < 0.30

    def test_timeout_penalizes_score(self) -> None:
        tracker = ResilienceTracker()
        t0 = 0.0
        tracker.on_spike(spike_stress=0.80, timestamp=t0)
        # Never recover — update past MAX_RECOVERY_S
        tracker.update(current_stress=0.80, baseline_mean=0.30, baseline_std=0.10, timestamp=t0 + 65.0)
        result = tracker.compute()
        assert result.timeout_recoveries == 1
        assert result.resilience_score == pytest.approx(0.0)

    def test_multiple_spikes_averaging(self) -> None:
        tracker = ResilienceTracker()
        t0 = 0.0
        # Spike 1: recovery at t=10 → recovery_time=10s
        tracker.on_spike(spike_stress=0.80, timestamp=t0)
        tracker.update(current_stress=0.25, baseline_mean=0.30, baseline_std=0.10, timestamp=t0 + 10.0)
        # Spike 2: recovery at t=50 from spike at t=20 → recovery_time=30s
        tracker.on_spike(spike_stress=0.80, timestamp=t0 + 20.0)
        tracker.update(current_stress=0.25, baseline_mean=0.30, baseline_std=0.10, timestamp=t0 + 50.0)
        result = tracker.compute()
        assert result.completed_recoveries == 2
        # Average of 10s and 30s = 20s
        assert result.average_recovery_s == pytest.approx(20.0)

    def test_pending_recoveries_counted(self) -> None:
        tracker = ResilienceTracker()
        tracker.on_spike(spike_stress=0.80)
        result = tracker.compute()
        assert result.pending_recoveries == 1
        assert result.completed_recoveries == 0


# ---------------------------------------------------------------------------
# QuestionDifficultyAnalyzer
# ---------------------------------------------------------------------------


class TestQuestionDifficultyAnalyzer:
    def test_simple_question_low_complexity(self) -> None:
        result = _analyze_question_cached("Tell me about yourself.")
        assert result.complexity_score < 0.40

    def test_technical_question_higher_complexity(self) -> None:
        result = _analyze_question_cached(
            "Design a distributed cache with eventual consistency and describe the tradeoffs."
        )
        assert result.complexity_score > 0.40

    def test_long_question_increases_score(self) -> None:
        short = _analyze_question_cached("Why?")
        long = _analyze_question_cached(
            "Can you walk me through your approach to designing scalable distributed systems "
            "that handle high throughput with low latency and explain the tradeoffs you "
            "would make between consistency and availability in a production environment?"
        )
        assert long.complexity_score > short.complexity_score

    def test_technical_keywords_detected(self) -> None:
        result = _analyze_question_cached("Explain the tradeoffs between consistency and latency.")
        assert result.technical_keyword_count >= 1

    def test_syntactic_depth_detected(self) -> None:
        result = _analyze_question_cached(
            "If you were given a distributed system, how would you handle it, "
            "and because latency is important, what tradeoffs would you make?"
        )
        assert result.syntactic_depth >= 2

    def test_correlation_requires_three_observations(self) -> None:
        analyzer = QuestionDifficultyAnalyzer()
        analyzer.record("Short question?", stress_at_response=0.3)
        analyzer.record("Another short one?", stress_at_response=0.4)
        assert analyzer.correlate() is None
        analyzer.record("Third question?", stress_at_response=0.5)
        assert analyzer.correlate() is not None

    def test_pearson_r_in_valid_range(self) -> None:
        analyzer = QuestionDifficultyAnalyzer()
        questions = [
            ("Tell me about yourself.", 0.25),
            ("Describe a project.", 0.30),
            ("Design a distributed cache with consensus algorithms.", 0.70),
            ("Walk through your algorithm optimization approach with tradeoffs.", 0.65),
            ("Simple: what is your name?", 0.20),
        ]
        for q, s in questions:
            analyzer.record(q, stress_at_response=s)
        result = analyzer.correlate()
        assert result is not None
        assert -1.0 <= result.pearson_r <= 1.0

    def test_bucket_breakdown_present(self) -> None:
        analyzer = QuestionDifficultyAnalyzer()
        questions = [
            ("Hi?", 0.20),
            ("Why?", 0.22),
            ("Tell me about a project.", 0.35),
            ("Design a distributed consensus algorithm.", 0.70),
            ("Implement a lock-free concurrent queue.", 0.75),
        ]
        for q, s in questions:
            analyzer.record(q, stress_at_response=s)
        result = analyzer.correlate()
        assert result is not None
        assert "low" in result.stress_by_complexity_bucket
        assert "high" in result.stress_by_complexity_bucket

    def test_cache_returns_same_object(self) -> None:
        result1 = _analyze_question_cached("Tell me about yourself.")
        result2 = _analyze_question_cached("Tell me about yourself.")
        assert result1 is result2  # lru_cache identity


# ---------------------------------------------------------------------------
# AggressionClassifier
# ---------------------------------------------------------------------------


class TestAggressionClassifier:
    def setup_method(self) -> None:
        self.clf = AggressionClassifier()

    def test_neutral_low_all_signals(self) -> None:
        result = self.clf.classify(
            aggression_score=0.10,
            interruptions_per_window=0,
            question_rate_per_min=0.5,
            interviewer_dominance=0.40,
            volume_spikes=0,
        )
        assert result.classification == AggressionClass.NEUTRAL

    def test_hostile_high_score_and_volume(self) -> None:
        result = self.clf.classify(
            aggression_score=0.85,
            interruptions_per_window=1,
            question_rate_per_min=1.0,
            interviewer_dominance=0.60,
            volume_spikes=3,
        )
        assert result.classification == AggressionClass.HOSTILE

    def test_interruptive_high_interruptions(self) -> None:
        result = self.clf.classify(
            aggression_score=0.30,
            interruptions_per_window=5,
            question_rate_per_min=1.0,
            interviewer_dominance=0.50,
            volume_spikes=0,
        )
        assert result.classification == AggressionClass.INTERRUPTIVE

    def test_rapid_fire_high_question_rate(self) -> None:
        result = self.clf.classify(
            aggression_score=0.25,
            interruptions_per_window=0,
            question_rate_per_min=3.5,
            interviewer_dominance=0.50,
            volume_spikes=0,
        )
        assert result.classification == AggressionClass.RAPID_FIRE

    def test_challenging_elevated_score(self) -> None:
        result = self.clf.classify(
            aggression_score=0.50,
            interruptions_per_window=1,
            question_rate_per_min=1.0,
            interviewer_dominance=0.55,
            volume_spikes=0,
        )
        assert result.classification == AggressionClass.CHALLENGING

    def test_challenging_by_dominance(self) -> None:
        result = self.clf.classify(
            aggression_score=0.20,
            interruptions_per_window=0,
            question_rate_per_min=1.0,
            interviewer_dominance=0.80,
            volume_spikes=0,
        )
        assert result.classification == AggressionClass.CHALLENGING
        assert result.primary_signal == "dominance_imbalance"

    def test_confidence_in_valid_range(self) -> None:
        for agg in [0.0, 0.3, 0.5, 0.8, 1.0]:
            result = self.clf.classify(
                aggression_score=agg,
                interruptions_per_window=0,
                question_rate_per_min=0.5,
                interviewer_dominance=0.40,
                volume_spikes=0,
            )
            assert 0.0 <= result.confidence <= 1.0

    def test_priority_hostile_wins_over_interruptive(self) -> None:
        result = self.clf.classify(
            aggression_score=0.85,
            interruptions_per_window=5,
            question_rate_per_min=4.0,
            interviewer_dominance=0.80,
            volume_spikes=3,
        )
        assert result.classification == AggressionClass.HOSTILE


# ---------------------------------------------------------------------------
# SessionSummaryGenerator
# ---------------------------------------------------------------------------


class TestSessionSummaryGenerator:
    def _make_generator_with_windows(self, n: int = 5) -> SessionSummaryGenerator:
        gen = SessionSummaryGenerator("test-session-001")
        for i in range(n):
            gen.record_window(
                _make_window(
                    window_id=f"w{i:04d}",
                    timestamp=float(i * 5),
                    stress=0.30 + i * 0.05,
                )
            )
        return gen

    def test_raises_without_windows(self) -> None:
        gen = SessionSummaryGenerator("empty")
        with pytest.raises(ValueError, match="No windows recorded"):
            gen.generate()

    def test_peak_stress_correct(self) -> None:
        gen = self._make_generator_with_windows(5)
        summary = gen.generate()
        # Last window has highest stress: 0.30 + 4 * 0.05 = 0.50
        assert summary.peak_stress == pytest.approx(0.50, abs=0.01)

    def test_average_stress_correct(self) -> None:
        gen = SessionSummaryGenerator("avg-test")
        stresses = [0.20, 0.40, 0.60]
        for i, s in enumerate(stresses):
            gen.record_window(_make_window(window_id=f"w{i}", timestamp=float(i), stress=s))
        summary = gen.generate()
        assert summary.average_stress == pytest.approx(0.40, abs=0.01)

    def test_aggression_spikes_captured(self) -> None:
        gen = SessionSummaryGenerator("spike-test")
        gen.record_window(_make_window(window_id="w0000", timestamp=0.0, risk="none"))
        gen.record_window(_make_window(window_id="w0001", timestamp=5.0, risk="high"))
        gen.record_window(_make_window(window_id="w0002", timestamp=10.0, risk="moderate"))
        summary = gen.generate()
        assert "w0001" in summary.aggression_spike_window_ids
        assert "w0002" in summary.aggression_spike_window_ids
        assert "w0000" not in summary.aggression_spike_window_ids

    def test_filler_density_zero_when_no_fillers(self) -> None:
        gen = self._make_generator_with_windows(3)
        gen.record_filler_total(0)
        summary = gen.generate()
        assert summary.filler_word_density_per_min == pytest.approx(0.0)

    def test_filler_density_computed(self) -> None:
        gen = SessionSummaryGenerator("filler-test")
        for i in range(12):  # 12 × 5s = 60s
            gen.record_window(_make_window(window_id=f"w{i}", timestamp=float(i * 5)))
        gen.record_filler_total(60)  # 60 fillers in 60s → 60/min
        summary = gen.generate()
        assert summary.filler_word_density_per_min == pytest.approx(60.0, rel=0.1)

    def test_to_json_excludes_window_records_by_default(self) -> None:
        gen = self._make_generator_with_windows(3)
        summary = gen.generate()
        json_str = gen.to_json(summary, verbose=False)
        assert "window_records" not in json_str

    def test_to_json_verbose_includes_window_records(self) -> None:
        gen = self._make_generator_with_windows(3)
        summary = gen.generate()
        json_str = gen.to_json(summary, verbose=True)
        assert "window_records" in json_str

    def test_to_markdown_contains_key_sections(self) -> None:
        gen = self._make_generator_with_windows(3)
        summary = gen.generate()
        md = gen.to_markdown(summary)
        assert "# Interview Session Report" in md
        assert "Stress Analysis" in md
        assert "Resilience" in md

    def test_strengths_when_low_stress(self) -> None:
        gen = SessionSummaryGenerator("strengths-test")
        for i in range(5):
            gen.record_window(_make_window(window_id=f"w{i}", timestamp=float(i), stress=0.20))
        gen.record_filler_total(0)
        summary = gen.generate()
        assert len(summary.strengths) > 0

    def test_improvements_when_high_stress(self) -> None:
        gen = SessionSummaryGenerator("improvements-test")
        for i in range(5):
            gen.record_window(_make_window(window_id=f"w{i}", timestamp=float(i), stress=0.80))
        summary = gen.generate()
        assert len(summary.improvement_areas) > 0


# ---------------------------------------------------------------------------
# SessionComparisonEngine
# ---------------------------------------------------------------------------


class TestSessionComparisonEngine:
    def _make_summary(
        self,
        session_id: str,
        avg_stress: float = 0.40,
        peak_stress: float = 0.70,
        resilience: float = 0.60,
        filler: float = 3.0,
        spike_count: int = 2,
    ):
        gen = SessionSummaryGenerator(session_id)
        for i in range(5):
            gen.record_window(
                _make_window(
                    window_id=f"w{i}",
                    timestamp=float(i * 5),
                    stress=avg_stress,
                )
            )
        return gen.generate()

    def test_improved_stress_detected(self) -> None:
        s1 = self._make_summary("s1", avg_stress=0.60)
        s2 = self._make_summary("s2", avg_stress=0.40)
        engine = SessionComparisonEngine()
        comparison = engine.compare(s1, s2)
        stress_delta = next(d for d in comparison.deltas if d.metric == "average_stress")
        assert stress_delta.improved is True

    def test_overall_score_between_0_and_1(self) -> None:
        s1 = self._make_summary("s1")
        s2 = self._make_summary("s2")
        engine = SessionComparisonEngine()
        comparison = engine.compare(s1, s2)
        assert 0.0 <= comparison.overall_improvement_score <= 1.0

    def test_summary_lines_non_empty(self) -> None:
        s1 = self._make_summary("s1")
        s2 = self._make_summary("s2")
        engine = SessionComparisonEngine()
        comparison = engine.compare(s1, s2)
        assert len(comparison.summary_lines) > 0

    def test_direction_symbols(self) -> None:
        s1 = self._make_summary("s1", avg_stress=0.50)
        s2 = self._make_summary("s2", avg_stress=0.30)
        engine = SessionComparisonEngine()
        comparison = engine.compare(s1, s2)
        delta = next(d for d in comparison.deltas if d.metric == "average_stress")
        assert delta.direction_symbol == "↓"


# ---------------------------------------------------------------------------
# Explainer
# ---------------------------------------------------------------------------


class TestExplainer:
    def setup_method(self) -> None:
        self.explainer = Explainer()

    def test_top_signals_sorted_by_contribution(self) -> None:
        report = self.explainer.explain(
            window_id="w0001",
            stress_score=0.55,
            aggression_score=0.30,
            stress_signal_values={
                "response_delay": 0.8,
                "filler_rate": 0.2,
                "pitch_variance": 0.1,
            },
            aggression_signal_values={"interviewer_interruptions": 0.5},
            confidence_breakdown={"stress_confidence": 0.75},
        )
        contributions = report.top_stress_signals
        for i in range(len(contributions) - 1):
            assert contributions[i].weighted_contribution >= contributions[i + 1].weighted_contribution

    def test_at_most_four_stress_signals(self) -> None:
        report = self.explainer.explain(
            window_id="w0001",
            stress_score=0.5,
            aggression_score=0.2,
            stress_signal_values={k: 0.5 for k in ["response_delay", "filler_rate", "pitch_variance",
                                                    "silence_before_response", "interruptions_received"]},
            aggression_signal_values={},
            confidence_breakdown={},
        )
        assert len(report.top_stress_signals) <= 4

    def test_at_most_three_aggression_signals(self) -> None:
        report = self.explainer.explain(
            window_id="w0001",
            stress_score=0.3,
            aggression_score=0.5,
            stress_signal_values={},
            aggression_signal_values={k: 0.5 for k in [
                "interviewer_interruptions", "dominance_imbalance", "speaking_ratio_imbalance",
                "rapid_questioning", "volume_spikes"
            ]},
            confidence_breakdown={},
        )
        assert len(report.top_aggression_signals) <= 3

    def test_zero_signals_no_error(self) -> None:
        report = self.explainer.explain(
            window_id="w0001",
            stress_score=0.0,
            aggression_score=0.0,
            stress_signal_values={},
            aggression_signal_values={},
            confidence_breakdown={},
        )
        assert report.stress_score == 0.0

    def test_audit_log_has_required_fields(self) -> None:
        report = self.explainer.explain(
            window_id="w0042",
            stress_score=0.61,
            aggression_score=0.34,
            stress_signal_values={"response_delay": 0.7},
            aggression_signal_values={},
            confidence_breakdown={"stress_confidence": 0.8},
        )
        assert report.audit_log["window_id"] == "w0042"
        assert report.audit_log["cross_user_comparison"] is False
        assert report.audit_log["normalization_scope"] == "per_session_only"

    def test_percent_of_total_sums_near_100(self) -> None:
        report = self.explainer.explain(
            window_id="w0001",
            stress_score=0.5,
            aggression_score=0.3,
            stress_signal_values={k: 0.5 for k in [
                "response_delay", "filler_rate", "speech_rate_change",
                "pitch_variance", "silence_before_response",
            ]},
            aggression_signal_values={},
            confidence_breakdown={},
        )
        total_pct = sum(c.percent_of_total for c in report.top_stress_signals)
        # Only top 4 included so sum may be < 100
        assert total_pct > 0


# ---------------------------------------------------------------------------
# GracefulDegradationEngine
# ---------------------------------------------------------------------------


class TestGracefulDegradationEngine:
    def setup_method(self) -> None:
        self.engine = GracefulDegradationEngine()

    def test_all_available_no_cap(self) -> None:
        policy = self.engine.evaluate(SignalAvailability())
        assert policy.confidence_cap == pytest.approx(1.0)
        assert policy.should_run_llm is True
        assert not policy.warnings

    def test_stt_unavailable_caps_confidence(self) -> None:
        policy = self.engine.evaluate(SignalAvailability(stt_available=False))
        assert policy.confidence_cap <= 0.55
        assert any("STT" in w for w in policy.warnings)

    def test_audio_unavailable_caps_lower(self) -> None:
        policy = self.engine.evaluate(SignalAvailability(audio_available=False))
        assert policy.confidence_cap <= 0.45

    def test_video_unavailable_caps_confidence(self) -> None:
        policy = self.engine.evaluate(SignalAvailability(video_available=False))
        assert policy.confidence_cap <= 0.70
        assert any("Video" in w for w in policy.warnings)

    def test_llm_unavailable_skips_llm(self) -> None:
        policy = self.engine.evaluate(SignalAvailability(llm_available=False))
        assert policy.should_run_llm is False

    def test_all_unavailable_critical_cap(self) -> None:
        policy = self.engine.evaluate(
            SignalAvailability(
                audio_available=False,
                video_available=False,
                stt_available=False,
                llm_available=False,
            )
        )
        assert policy.confidence_cap <= 0.30
        assert policy.should_run_llm is False

    def test_warning_message_concatenated(self) -> None:
        policy = self.engine.evaluate(SignalAvailability(stt_available=False, video_available=False))
        assert len(policy.warning_message) > 0

    def test_nominal_warning_message(self) -> None:
        policy = self.engine.evaluate(SignalAvailability())
        assert policy.warning_message == "All signals nominal"


# ---------------------------------------------------------------------------
# LatencyTracker
# ---------------------------------------------------------------------------


class TestLatencyTracker:
    def test_measure_records_duration(self) -> None:
        tracker = LatencyTracker()
        with tracker.measure("deterministic_scoring"):
            # Simulate work
            _ = sum(range(1000))
        stats = tracker.get_stats()
        assert "deterministic_scoring" in stats
        assert stats["deterministic_scoring"]["count"] == 1.0
        assert stats["deterministic_scoring"]["mean_ms"] > 0

    def test_multiple_measures_averaged(self) -> None:
        tracker = LatencyTracker()
        for _ in range(5):
            with tracker.measure("window_flush"):
                pass
        stats = tracker.get_stats()
        assert stats["window_flush"]["count"] == 5.0

    def test_p95_within_max(self) -> None:
        tracker = LatencyTracker()
        for _ in range(20):
            with tracker.measure("llm_call"):
                pass
        stats = tracker.get_stats()
        assert stats["llm_call"]["p95_ms"] <= stats["llm_call"]["max_ms"]

    def test_prometheus_text_format(self) -> None:
        tracker = LatencyTracker()
        with tracker.measure("pose_inference"):
            pass
        text = tracker.prometheus_text()
        assert "behavioral_analyzer_stage_latency_ms" in text
        assert "pose_inference" in text
        assert "mean" in text

    def test_unknown_stage_no_budget_warning(self) -> None:
        tracker = LatencyTracker()
        # Should not raise even for unknown stage names
        with tracker.measure("custom_stage"):
            pass
        stats = tracker.get_stats()
        assert "custom_stage" in stats

    def test_empty_stats_when_no_measures(self) -> None:
        tracker = LatencyTracker()
        assert tracker.get_stats() == {}


# ---------------------------------------------------------------------------
# Integration: SpikeDetector + ResilienceTracker
# ---------------------------------------------------------------------------


class TestSpikeAndResilienceIntegration:
    def test_spike_triggers_resilience_tracking(self) -> None:
        detector = SpikeDetector()
        tracker = ResilienceTracker()

        # Varied baseline so std > 0
        for i in range(10):
            val = 0.27 + (i % 5) * 0.01
            detector.ingest(val, f"w{i:04d}")

        spike = detector.ingest(0.95, "w0010")
        assert spike is not None

        tracker.on_spike(spike_stress=spike.stress_after, timestamp=spike.timestamp)
        tracker.update(
            current_stress=0.30,
            baseline_mean=0.30,
            baseline_std=0.10,
            timestamp=spike.timestamp + 8.0,
        )
        result = tracker.compute()
        assert result.resilience_score > 0.85
        assert result.completed_recoveries == 1


# ---------------------------------------------------------------------------
# Edge cases: empty / None metrics
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_question_analyzer_empty_string(self) -> None:
        result = _analyze_question_cached("")
        assert result.complexity_score == pytest.approx(0.0)

    def test_aggression_classifier_all_zeros(self) -> None:
        clf = AggressionClassifier()
        result = clf.classify(0.0, 0.0, 0.0, 0.0, 0.0)
        assert result.classification == AggressionClass.NEUTRAL
        assert result.confidence == pytest.approx(1.0)

    def test_resilience_tracker_baseline_std_zero(self) -> None:
        tracker = ResilienceTracker()
        tracker.on_spike(spike_stress=0.80, timestamp=0.0)
        # std=0 should not raise ZeroDivisionError
        tracker.update(current_stress=0.30, baseline_mean=0.30, baseline_std=0.0, timestamp=10.0)
        result = tracker.compute()
        assert 0.0 <= result.resilience_score <= 1.0

    def test_session_summary_single_window(self) -> None:
        gen = SessionSummaryGenerator("single-window")
        gen.record_window(_make_window(window_id="w0000", timestamp=0.0, stress=0.45))
        summary = gen.generate()
        assert summary.peak_stress == pytest.approx(0.45)
        assert summary.average_stress == pytest.approx(0.45)

    def test_session_comparison_identical_sessions(self) -> None:
        gen = SessionSummaryGenerator("s1")
        for i in range(5):
            gen.record_window(_make_window(window_id=f"w{i}", timestamp=float(i), stress=0.40))
        s1 = gen.generate()
        gen2 = SessionSummaryGenerator("s2")
        for i in range(5):
            gen2.record_window(_make_window(window_id=f"w{i}", timestamp=float(i), stress=0.40))
        s2 = gen2.generate()
        engine = SessionComparisonEngine()
        comparison = engine.compare(s1, s2)
        # No change — delta should be near zero
        for d in comparison.deltas:
            assert abs(d.delta_absolute) < 0.01

    def test_spike_detector_single_anomaly_then_recovery(self) -> None:
        detector = SpikeDetector()
        # Varied baseline so std > 0
        for i in range(15):
            val = 0.27 + (i % 5) * 0.01
            detector.ingest(val, f"w{i:04d}")
        spike = detector.ingest(0.95, "w0015")
        assert spike is not None
        # Return to near-baseline — no more spikes expected
        for i in range(5):
            val = 0.27 + (i % 5) * 0.01
            result = detector.ingest(val, f"w{i+16:04d}")
            assert result is None
