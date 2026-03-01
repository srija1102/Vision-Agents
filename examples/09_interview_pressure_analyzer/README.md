# Interview Pressure Analyzer

A real-time multimodal AI agent that watches and listens to job interviews, detecting candidate stress, interviewer aggression, and behavioral patterns — with live coaching feedback delivered in under 2 seconds.

Built on [Vision Agents](https://github.com/GetStream/Vision-Agents) for the Vision Agents Hackathon.

---

## What It Does

The agent joins a live video call and analyzes both parties simultaneously:

- **Candidate**: stress level, vocal patterns, filler words, posture, recovery after pressure spikes
- **Interviewer**: aggression style (neutral / challenging / rapid-fire / interruptive / hostile), speaking dominance
- **Together**: conversation imbalance, pressure trend, risk flag, live coaching tips

Every 5 seconds the agent emits a behavioral analysis window. A live browser dashboard displays real-time charts, spike alerts, and actionable coaching feedback.

At the end of the session, a full summary is generated — and if this is a repeat session, the agent compares it against the previous one with per-metric delta percentages.

---

## Architecture

```
Candidate video  → YOLO pose (15 fps) → PoseBridge → BehavioralAnalyzer
Candidate audio  → Deepgram STT       → InterviewBridge → BehavioralAnalyzer
Turn events      → InterviewBridge    → BehavioralAnalyzer

Every 5s:
  BehavioralAnalyzer
    → deterministic scorer (11 stress + 5 aggression signals)
    → baseline z-score normalizer (60s calibration window)
    → Claude Haiku (anchored refiner — adjusts within bounds, generates coaching)
    → BehavioralAnalysisEvent
    → WebSocket → Browser dashboard (live chart + coaching + spike log)
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Deterministic anchor + LLM refiner | LLM never sole score source — prevents hallucinated scores |
| Skip-not-queue on cycle overlap | No backlog of stale analysis if LLM is slow |
| Per-session z-score baseline | No cross-candidate comparison; fair to each individual |
| <2s total latency budget | Enforced per-stage via `LatencyTracker` |
| Graceful degradation | Falls back to deterministic result if STT/video/LLM unavailable |

---

## Prerequisites

- Python 3.12
- `uv` package manager
- A [GetStream](https://getstream.io) account (for WebRTC edge)
- API keys for: OpenRouter, Deepgram, ElevenLabs

---

## Setup

```bash
# From repo root
uv venv --python 3.12
uv sync --all-extras --dev
cp env.example .env
# Fill in your credentials (see below)
```

### Required Environment Variables

```env
STREAM_API_KEY=...
STREAM_API_SECRET=...
OPENROUTER_API_KEY=...
DEEPGRAM_API_KEY=...
ELEVENLABS_API_KEY=...
```

### Optional Model Overrides

```env
INTERVIEWER_MODEL=anthropic/claude-haiku-4-5   # LLM voice for the AI interviewer
ANALYSIS_MODEL=anthropic/claude-haiku-4-5      # LLM for behavioral analysis
```

---

## Running

### Live Interview Mode

```bash
uv run python examples/09_interview_pressure_analyzer/interview_analyzer.py \
    --call-id <your-getstream-call-id>
```

The agent joins the call, opens the dashboard at `http://localhost:8080`, and starts a browser tab at `/join` so the candidate can connect via WebRTC.

Optional flags:
```
--candidate-user-id   User ID of the candidate (auto-detected if omitted)
--port                Dashboard port (default: 8080)
```

### Dashboard Preview (no live call needed)

Replays a synthetic 90-second adversarial interview through the real dashboard:

```bash
uv run python examples/09_interview_pressure_analyzer/preview_dashboard.py
```

Then open `http://localhost:8080`.

---

## Dashboard

| Panel | Shows |
|-------|-------|
| **Candidate Stress** | 0–100% stress score + confidence |
| **Interviewer Aggression** | 0–100% aggression score + confidence |
| **Risk Level** | None / Moderate / High (with animated border) |
| **Resilience** | Post-spike recovery score |
| **Chart** | Rolling 30-window stress & aggression time series |
| **Coaching Feedback** | Live actionable tip from Claude Haiku |
| **Spike Log** | Real-time stress spikes with cause and σ magnitude |
| **Session Metrics** | Interviewer style badge, dominant party, imbalance score |
| **Session Summary** | Overlay on session end: peak stress, strengths, areas to improve, comparison vs previous session |

---

## Scoring

### Stress Signals (11, sum to 1.0)
Speech rate deviation, filler word density, pitch variance, pause frequency, answer latency, syllable stress, topic complexity, turn interruptions, posture confidence, facial tension, response length deviation.

### Aggression Signals (5, sum to 1.0)
Interruption rate, rapid-fire question density, challenging question ratio, speaking dominance, tone escalation.

### Multi-signal Gate
- Stress capped at 50% if fewer than 2 signals are active
- Aggression capped at 60% if fewer than 2 signals are active

Prevents single-feature false positives (e.g. one unusually long pause triggering a high stress score).

---

## Hackathon Alignment

| Requirement | How this satisfies it |
|-------------|-----------------------|
| Multi-modal agents that watch, listen, understand video | YOLO pose (video) + Deepgram STT (audio) + LLM reasoning — all in real-time |
| <30ms audio/video latency | Stream WebRTC edge network |
| 500ms join latency | Stream edge |
| Combine YOLO + LLM in real-time | YOLOv11 pose → BehavioralAnalyzer → Claude Haiku, every 5s |
| Native LLM APIs | OpenRouter with Claude Haiku via native Anthropic API |
| Cross-platform SDKs | Candidate joins via Stream's React/iOS/Android/Flutter/RN SDK |

---

## Running Tests

```bash
uv run py.test plugins/behavioral_analyzer/tests/ -q
# Expected: 74 passed
```
