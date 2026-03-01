# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Python monorepo managed with **uv workspaces**.
The core framework lives in `agents-core/` and plugins live in `plugins/` (37+ packages).
Python >= 3.10, 3.12 recommended.

## Setup

```bash
uv venv --python 3.12.11
uv sync --all-extras --dev
pre-commit install
cp env.example .env  # fill in secrets
```

## Commands

All commands use `uv`. Never use `python -m`. If you run into dependency issues, stop and ask.

```bash
# Full check (ruff + mypy + unit tests)
uv run dev.py check

# Unit tests only
uv run py.test -m "not integration" -n auto

# Run a single test file
uv run py.test tests/test_agents.py -n auto

# Integration tests (needs .env secrets)
uv run py.test -m "integration"

# Lint & format
uv run ruff check .
uv run ruff format .

# Type check (core and plugins separately)
uv run mypy --install-types --non-interactive -p vision_agents
uv run mypy --install-types --non-interactive --exclude 'plugins/.*/tests/.*' plugins

# Run an example agent
uv run examples/01_simple_agent_example/simple_agent_example.py run

# Serve as HTTP API
uv run <path-to-example> serve --host=<host> --port=<port>

# Use a video file instead of a live stream
uv run <path-to-example> run --video-track-override <path-to-video>
```

## Architecture

### Agent wiring

`Agent` is the top-level object. Constructor params:
- **`edge`** – WebRTC transport (typically `getstream.Edge()`)
- **`llm`** – language/speech model (`gemini.Realtime()`, `openai.Realtime()`, `gemini.LLM()`, …)
- **`stt`** / **`tts`** – speech-to-text / text-to-speech for non-realtime LLMs
- **`turn_detection`** – when to trigger agent responses
- **`processors`** – list of `Processor` instances for audio/video side-processing
- **`instructions`** – system prompt string or `@file.md` reference

### STT/TTS flow (non-realtime LLMs)

1. Agent receives `AudioReceivedEvent` → forwards PCM to STT.
2. STT fires `STTPartialTranscriptEvent` / `STTTranscriptEvent`.
3. Agent calls `llm.simple_response(text)` → returns `LLMResponseEvent` (`.text`, `.exception`).
4. Agent calls `tts.send(llm_response.text)`.

### Realtime STS flow

- Agent pipes `AudioReceivedEvent` → `llm.simple_audio_response(pcm_data)`.
- For video: `llm.watch_video_track(track)` streams frames over WebSocket/WebRTC.
- Audio replies come back via `llm.audio_track`.

### Processors

Implement one of the ABCs from `vision_agents.core.processors`:

| ABC | Role |
|-----|------|
| `VideoProcessor` | Consume incoming video (e.g. YOLO) |
| `VideoPublisher` | Produce outgoing video (e.g. avatar) |
| `AudioProcessor` | Consume incoming audio |
| `AudioPublisher` | Produce outgoing audio |
| `VideoProcessorPublisher` | Consume + publish video |
| `AudioProcessorPublisher` | Consume + publish audio |

Required overrides: `name` (property) and `close()`. Optional: `attach_agent(agent)` to access the event bus.

### Events

- Plugin events extend `PluginBaseEvent`; core events extend `BaseEvent`.
- Define `type: str = field(default="plugin.name.event", init=False)`.
- `EventManager`: `.register(EventClass)`, `.send(event)`, `@manager.subscribe(EventClass)`.

### Audio — `PcmData`

All audio is `PcmData` (never raw bytes). WebRTC uses Opus 48 kHz stereo; models vary.

```python
from getstream.video.rtc import PcmData
pcm = PcmData.from_bytes(audio_bytes, sample_rate=16000, channels=1, format="s16")
resampled = pcm.resample(48_000, target_channels=2)
wav_bytes = pcm.to_wav_bytes()
```

Use `AudioQueue` for buffering; `AudioTrack.write()` / `.flush()` for playback.

### Warmup (lazy resource loading)

Plugins that load models implement `Warmable[T]`:

```python
class MySTT(STT, Warmable[WhisperModel]):
    async def on_warmup(self) -> WhisperModel: ...       # runs once, result shared globally
    def on_warmed_up(self, model: WhisperModel) -> None: ...  # called per agent instance
```

## Testing

- Framework: pytest. Never mock.
- `@pytest.mark.asyncio` is not needed (asyncio_mode = auto).
- Integration tests use `@pytest.mark.integration`.
- Never adjust `sys.path`.

### Text-only testing with `TestSession`

Use `vision_agents.testing` to test agent logic without audio/video infrastructure:

```python
from vision_agents.testing import TestSession, LLMJudge

async def test_greeting():
    judge = LLMJudge(gemini.LLM(MODEL))
    async with TestSession(llm=llm, instructions="Be friendly") as session:
        response = await session.simple_response("Hello")
        verdict = await judge.evaluate(response.chat_messages[0], intent="Friendly greeting")
        assert verdict.success, verdict.reason

async def test_tool_call():
    async with TestSession(llm=llm, instructions="...") as session:
        response = await session.simple_response("Weather in Tokyo?")
        response.assert_function_called("get_weather", arguments={"location": "Tokyo"})
```

## Creating a new plugin

1. Copy `plugins/sample_plugin/` as a template.
2. Set `pyproject.toml` name to `vision-agents-plugins-<name>` (dashes, not underscores).
3. Add `[tool.hatch.build.targets.wheel] packages = [".", "vision_agents"]`.
4. Register in root `pyproject.toml` under both `[tool.uv.sources]` and `[tool.uv.workspace] members`.
5. Package path convention: `plugins/<name>/vision_agents/plugins/<name>/`.
6. Every plugin needs an integration test.

## Interview Pressure Analyzer

The `plugins/behavioral_analyzer/` plugin is a production-grade multimodal behavioral analysis system. Key architectural constraints:

### Design invariants
- **Deterministic anchor, LLM refiner**: `_scorer.py` computes the authoritative pre-scores; the LLM only adjusts within bounds and generates coaching text. Never let LLM be the sole score source.
- **Dedicated LLM**: The `BehavioralAnalyzer` requires its own LLM instance separate from the agent's conversation LLM.
- **Skip-not-queue**: If the previous analysis cycle is still running, the next one is skipped (no backlog buildup).
- **No cross-user normalization**: Baseline z-scores are per-session only. Never compare across candidates.
- **<2s total latency budget**: All per-stage budgets are enforced via `LatencyTracker`.

### Module map

| Module | Responsibility |
|--------|---------------|
| `_types.py` | All shared dataclasses |
| `_baseline.py` | 60s calibration window, z-score normalization |
| `_scorer.py` | Deterministic weighted signal fusion (11 stress + 5 aggression signals) |
| `_aggregator.py` | 5s window buffering and metric aggregation |
| `_prompt.py` | LLM system prompt (anchor-adjustment rules) |
| `_spike_detector.py` | 1s sliding window, detects >2σ stress spikes, tags cause |
| `_resilience.py` | Tracks post-spike recovery, computes resilience score |
| `_question_analyzer.py` | Deterministic NLP question complexity + stress correlation |
| `_aggression_classifier.py` | Classifies aggression style: neutral/challenging/rapid_fire/interruptive/hostile |
| `_session_summary.py` | End-of-session summary (JSON + Markdown), coaching plan |
| `_session_comparison.py` | Compares two sessions, per-metric delta percentages |
| `_explainability.py` | Signal contribution breakdown, audit log |
| `_degradation.py` | Graceful degradation policy when STT/video/LLM unavailable |
| `_telemetry.py` | Optional OpenTelemetry latency tracking with budget alerts |
| `analyzer.py` | Main `BehavioralAnalyzer` processor — wires all modules |
| `interview_bridge.py` | Routes STT + turn events → analyzer |
| `pose_bridge.py` | Routes YOLO pose keypoints → analyzer |
| `events.py` | All emitted events (`BehavioralAnalysisEvent`, `SpikeDetectedEvent`, `SessionSummaryEvent`) |

### Signal weights (do not change without updating `_explainability.py` mirror)

Stress signals sum to 1.0 across 11 inputs. Aggression signals sum to 1.0 across 5 inputs. Multi-signal gate: <2 active signals → score capped at 50% (stress) or 60% (aggression).

### Adding a new signal
1. Add the raw field to `AudioMetrics` / `VideoMetrics` in `_types.py`.
2. Add ingestion in `_aggregator.py`.
3. Add weight in `_scorer.py` (rebalance existing weights to sum 1.0).
4. Mirror the weight in `_explainability.py` (`_STRESS_WEIGHTS` / `_AGGRESSION_WEIGHTS`).
5. Update the LLM prompt in `_prompt.py` if the signal needs LLM awareness.

## Python rules

- Never use `from __future__ import annotations`.
- Never write `except Exception as e`. Catch specific exceptions.
- Avoid `getattr`, `hasattr`, `delattr`, `setattr`; prefer normal attribute access.
- Docstrings: Google style, keep them short.
- Do not use section comments like `# -- some section --`
- Prefer `logger.exception()` when logging an error with a traceback instead of `logger.error("Error: {exc}")`
- Do not use local imports, import at the top of the module

## Code style

**Imports**:

- ordered as: stdlib, third-party, local package, relative. Use `TYPE_CHECKING` guard for imports only needed by type annotations.
- Never import from private modules (`_foo`) outside of the package's own `__init__.py`. Use the public re-export (e.g. `from vision_agents.testing import TestResponse`, not `from vision_agents.testing._run_result import TestResponse`).

**Naming**:

- private attributes and methods use a leading underscore (`_sessions`, `_warmup_agent`). Public API is plain snake_case.

**Type annotations**:

- use them everywhere. Modern syntax: `X | Y` unions, `dict[str, T]` generics, full `Callable` signatures, `Optional` for nullable params.

**Logging**:
module-level `logger = logging.getLogger(__name__)`. Use `debug` for lifecycle, `info` for notable events, `error` for failures without a traceback,
`exception` for errors with traceback.

**Constructor validation**:

- raise `ValueError` with a descriptive message for invalid args. Prefer custom domain exceptions over generic ones.

**Async patterns**:

- async-first lifecycle methods (`start`/`stop`). Support `__aenter__`/`__aexit__` for context manager usage.
- Use `asyncio.Lock`, `asyncio.Task`, `asyncio.gather` for concurrency.
- Clean up resources in `finally` blocks.

**Method order**:

- `__init__`, public lifecycle methods, properties, public feature methods, private helpers, dunder methods.
