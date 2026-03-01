"""Real-time AI Interview Pressure Analyzer with live browser dashboard.

Full pipeline:
  Candidate video  → PoseBridge (YOLO pose) → BehavioralAnalyzer
  Candidate audio  → Deepgram STT → InterviewBridge → BehavioralAnalyzer
  Turn events      → InterviewBridge → BehavioralAnalyzer
  Every 5 seconds  → BehavioralAnalyzer → Claude Haiku → BehavioralAnalysisEvent
  All events       → WebSocket → Browser (live chart + coaching + spike log)

Run:
    STREAM_API_KEY=... STREAM_API_SECRET=... \\
    OPENROUTER_API_KEY=... DEEPGRAM_API_KEY=... ELEVENLABS_API_KEY=... \\
    uv run python examples/09_interview_pressure_analyzer/interview_analyzer.py \\
        --call-id <your-call-id>

Dashboard: http://localhost:8080

Model selection (via env vars):
    INTERVIEWER_MODEL   LLM for the AI interviewer voice  (default: anthropic/claude-haiku-4-5)
    ANALYSIS_MODEL      LLM for behavioral analysis       (default: anthropic/claude-haiku-4-5)
"""

import argparse
import asyncio
import logging
import os
from typing import Optional

import webbrowser
from urllib.parse import urlencode

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse
from getstream import Stream

import vision_agents.plugins.deepgram as deepgram
import vision_agents.plugins.elevenlabs as elevenlabs
import vision_agents.plugins.getstream as getstream
import vision_agents.plugins.openrouter as openrouter
from vision_agents.plugins.getstream.sfu_events import ParticipantLeftEvent
from vision_agents.core.agents import Agent
from vision_agents.core.edge.types import User
from vision_agents.plugins.behavioral_analyzer import (
    BehavioralAnalysisEvent,
    BehavioralAnalyzer,
    InterviewBridge,
    SessionComparisonEngine,
    SessionSummaryEvent,
    SpikeDetectedEvent,
)
from vision_agents.plugins.behavioral_analyzer._session_summary import SessionSummary

try:
    from vision_agents.plugins.behavioral_analyzer import PoseBridge
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

_RISK_ICONS = {"none": "✅", "moderate": "⚠️", "high": "🚨"}
_TREND_ICONS = {"increasing": "📈", "stable": "➡️", "decreasing": "📉"}

INTERVIEWER_INSTRUCTIONS = """\
You are a professional technical interviewer conducting a software engineering interview.
Ask challenging but fair questions. Probe depth of knowledge with follow-ups.
Keep responses concise. Do not reveal you are an AI unless directly asked.
"""

# ---------------------------------------------------------------------------
# Dashboard HTML (embedded so deployment is a single file)
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Interview Pressure Analyzer</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e;
    --green: #3fb950; --yellow: #d29922; --red: #f85149;
    --blue: #58a6ff; --purple: #bc8cff; --orange: #ffa657;
    --stress-color: #f85149; --agg-color: #ffa657;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", monospace; }

  header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 24px; background: var(--surface); border-bottom: 1px solid var(--border);
    position: sticky; top: 0; z-index: 10;
  }
  header h1 { font-size: 17px; font-weight: 600; letter-spacing: .3px; }
  header h1 span { color: var(--blue); }
  .session-badge {
    font-size: 12px; color: var(--muted); background: var(--bg);
    border: 1px solid var(--border); border-radius: 20px; padding: 3px 12px;
  }
  .status-dot {
    display: flex; align-items: center; gap: 7px; font-size: 13px;
  }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--muted); }
  .dot.live { background: var(--green); box-shadow: 0 0 6px var(--green); animation: pulse 1.5s infinite; }
  .dot.connecting { background: var(--yellow); }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

  .main { display: grid; grid-template-columns: 1fr 320px; gap: 16px; padding: 16px 24px; }

  .cards { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 16px; }
  .card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px; text-align: center;
    transition: border-color .3s;
  }
  .card .label { font-size: 10px; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); margin-bottom: 8px; }
  .card .value { font-size: 28px; font-weight: 700; font-variant-numeric: tabular-nums; }
  .card .sub { font-size: 11px; color: var(--muted); margin-top: 4px; }
  .card.stress .value  { color: var(--stress-color); }
  .card.agg .value     { color: var(--orange); }
  .card.risk .value    { font-size: 22px; }
  .card.resilience .value { color: var(--green); }
  .card.risk.none      { border-color: var(--green); }
  .card.risk.moderate  { border-color: var(--yellow); }
  .card.risk.high      { border-color: var(--red); animation: flash-border .5s ease; }
  @keyframes flash-border { 0%{border-color:var(--red)} 50%{border-color:#ff9090} 100%{border-color:var(--red)} }

  .chart-panel {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 18px; grid-column: 1;
  }
  .chart-title { font-size: 13px; font-weight: 600; color: var(--muted); margin-bottom: 14px; display: flex; justify-content: space-between; }
  .legend { display: flex; gap: 16px; }
  .legend-item { display: flex; align-items: center; gap: 5px; font-size: 11px; color: var(--muted); }
  .legend-dot { width: 10px; height: 10px; border-radius: 2px; }

  .right-panel { display: flex; flex-direction: column; gap: 12px; }

  .coaching-panel, .spike-panel, .style-panel {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px;
  }
  .panel-title { font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); margin-bottom: 10px; }

  .coaching-text {
    font-size: 13px; line-height: 1.6; color: var(--text);
    border-left: 3px solid var(--blue); padding-left: 10px;
    min-height: 40px;
    transition: opacity .4s;
  }
  .coaching-text.updating { opacity: .4; }

  .spike-list { list-style: none; max-height: 160px; overflow-y: auto; }
  .spike-list::-webkit-scrollbar { width: 4px; }
  .spike-list::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
  .spike-item {
    display: flex; align-items: flex-start; gap: 8px;
    padding: 6px 0; border-bottom: 1px solid var(--border);
    font-size: 12px; animation: slide-in .3s ease;
  }
  .spike-item:last-child { border-bottom: none; }
  @keyframes slide-in { from{transform:translateX(10px);opacity:0} to{transform:none;opacity:1} }
  .spike-icon { color: var(--red); font-size: 14px; flex-shrink: 0; margin-top: 1px; }
  .spike-sigma { font-weight: 700; color: var(--red); }
  .spike-cause { color: var(--muted); }
  .spike-time { color: var(--muted); font-size: 10px; margin-left: auto; white-space: nowrap; }
  .no-spikes { color: var(--muted); font-size: 12px; font-style: italic; }

  .style-row { display: flex; justify-content: space-between; align-items: center; font-size: 13px; padding: 4px 0; }
  .style-row .label { color: var(--muted); font-size: 11px; }
  .style-badge {
    padding: 2px 10px; border-radius: 20px; font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: .5px;
  }
  .style-badge.neutral    { background: #21262d; color: var(--muted); }
  .style-badge.challenging{ background: #1c2128; color: var(--blue); }
  .style-badge.rapid_fire { background: #2a1f00; color: var(--orange); }
  .style-badge.interruptive{ background: #2d1f00; color: #ffab40; }
  .style-badge.hostile    { background: #2d0f0f; color: var(--red); }

  /* Summary overlay */
  #summary-overlay {
    display: none; position: fixed; inset: 0; background: rgba(13,17,23,.92);
    z-index: 100; align-items: center; justify-content: center;
  }
  #summary-overlay.show { display: flex; }
  .summary-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 14px; padding: 32px; width: 560px; max-height: 80vh; overflow-y: auto;
    animation: pop-in .3s ease;
  }
  @keyframes pop-in { from{transform:scale(.95);opacity:0} to{transform:none;opacity:1} }
  .summary-card h2 { font-size: 18px; margin-bottom: 6px; }
  .summary-card .session-id { color: var(--muted); font-size: 12px; margin-bottom: 20px; }
  .summary-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 20px; }
  .summary-metric { background: var(--bg); border-radius: 8px; padding: 12px; }
  .summary-metric .m-label { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; }
  .summary-metric .m-value { font-size: 22px; font-weight: 700; margin-top: 4px; }
  .comparison-section { margin-top: 16px; }
  .comparison-section h3 { font-size: 13px; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; }
  .delta-row { display: flex; align-items: center; justify-content: space-between; font-size: 13px; padding: 5px 0; border-bottom: 1px solid var(--border); }
  .delta-row:last-child { border-bottom: none; }
  .delta-improved { color: var(--green); }
  .delta-regressed { color: var(--red); }
  .strengths h3, .improvements h3 { font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); margin: 14px 0 6px; }
  .strengths li, .improvements li { font-size: 13px; padding: 3px 0; list-style: none; }
  .strengths li::before { content: "✓ "; color: var(--green); }
  .improvements li::before { content: "→ "; color: var(--yellow); }
  .close-btn {
    margin-top: 20px; width: 100%; padding: 10px; background: var(--blue);
    border: none; border-radius: 8px; color: white; font-size: 14px; font-weight: 600;
    cursor: pointer; transition: opacity .2s;
  }
  .close-btn:hover { opacity: .8; }

  .window-id { font-size: 11px; color: var(--muted); }
</style>
</head>
<body>

<header>
  <h1>🎙 <span>Interview</span> Pressure Analyzer</h1>
  <span class="session-badge" id="session-label">No session</span>
  <div class="status-dot">
    <div class="dot connecting" id="dot"></div>
    <span id="status-text">Connecting…</span>
  </div>
</header>

<div style="padding: 16px 24px 0">
  <div class="cards">
    <div class="card stress">
      <div class="label">Candidate Stress</div>
      <div class="value" id="stress-val">—</div>
      <div class="sub" id="stress-conf">conf —</div>
    </div>
    <div class="card agg">
      <div class="label">Interviewer Aggression</div>
      <div class="value" id="agg-val">—</div>
      <div class="sub" id="agg-conf">conf —</div>
    </div>
    <div class="card risk none" id="risk-card">
      <div class="label">Risk Level</div>
      <div class="value" id="risk-val">—</div>
      <div class="sub" id="trend-val">trend —</div>
    </div>
    <div class="card resilience">
      <div class="label">Resilience</div>
      <div class="value" id="resilience-val">—</div>
      <div class="sub" id="window-id">window —</div>
    </div>
  </div>
</div>

<div class="main">
  <div>
    <div class="chart-panel">
      <div class="chart-title">
        <span>Stress &amp; Aggression Over Time</span>
        <div class="legend">
          <div class="legend-item"><div class="legend-dot" style="background:#f85149"></div>Stress</div>
          <div class="legend-item"><div class="legend-dot" style="background:#ffa657"></div>Aggression</div>
        </div>
      </div>
      <div style="position:relative;height:220px;">
        <canvas id="chart"></canvas>
      </div>
    </div>
  </div>

  <div class="right-panel">
    <div class="coaching-panel">
      <div class="panel-title">💡 Coaching Feedback</div>
      <div class="coaching-text" id="coaching">Waiting for analysis…</div>
    </div>

    <div class="spike-panel">
      <div class="panel-title">⚡ Spike Log</div>
      <ul class="spike-list" id="spike-list">
        <li class="no-spikes">No spikes detected yet</li>
      </ul>
    </div>

    <div class="style-panel">
      <div class="panel-title">📊 Session Metrics</div>
      <div class="style-row">
        <span class="label">Interviewer Style</span>
        <span class="style-badge neutral" id="style-badge">—</span>
      </div>
      <div class="style-row" style="margin-top:8px">
        <span class="label">Dominant Party</span>
        <span id="dominant-val" style="font-size:13px">—</span>
      </div>
      <div class="style-row">
        <span class="label">Imbalance Score</span>
        <span id="imbalance-val" style="font-size:13px">—</span>
      </div>
    </div>
  </div>
</div>

<!-- Session Summary Overlay -->
<div id="summary-overlay">
  <div class="summary-card">
    <h2>📋 Session Complete</h2>
    <div class="session-id" id="sum-session-id"></div>
    <div class="summary-grid" id="sum-grid"></div>
    <div class="comparison-section" id="sum-comparison" style="display:none">
      <h3>vs Previous Session</h3>
      <div id="sum-deltas"></div>
    </div>
    <div class="strengths"><h3>Strengths</h3><ul id="sum-strengths"></ul></div>
    <div class="improvements"><h3>Areas to Improve</h3><ul id="sum-improvements"></ul></div>
    <button class="close-btn" onclick="closeSummary()">Close & Continue</button>
  </div>
</div>

<script>
const MAX_POINTS = 30;
const labels = [], stressData = [], aggData = [];
let spikeCount = 0;

// Chart setup
const ctx = document.getElementById('chart').getContext('2d');
const chart = new Chart(ctx, {
  type: 'line',
  data: {
    labels,
    datasets: [
      {
        label: 'Stress', data: stressData,
        borderColor: '#f85149', backgroundColor: 'rgba(248,81,73,.1)',
        borderWidth: 2, pointRadius: 3, pointHoverRadius: 5,
        tension: 0.4, fill: true,
      },
      {
        label: 'Aggression', data: aggData,
        borderColor: '#ffa657', backgroundColor: 'rgba(255,166,87,.07)',
        borderWidth: 2, pointRadius: 3, pointHoverRadius: 5,
        tension: 0.4, fill: true,
      }
    ]
  },
  options: {
    animation: { duration: 400 },
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: { grid: { color: '#21262d' }, ticks: { color: '#8b949e', font: { size: 10 } } },
      y: {
        min: 0, max: 1,
        grid: { color: '#21262d' },
        ticks: { color: '#8b949e', font: { size: 10 }, stepSize: 0.25 }
      }
    },
    plugins: { legend: { display: false } }
  }
});

function push(label, stress, agg) {
  if (labels.length >= MAX_POINTS) { labels.shift(); stressData.shift(); aggData.shift(); }
  labels.push(label); stressData.push(stress); aggData.push(agg);
  chart.update();
}

// WebSocket
function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => {
    document.getElementById('dot').className = 'dot live';
    document.getElementById('status-text').textContent = 'Live';
  };
  ws.onclose = () => {
    document.getElementById('dot').className = 'dot connecting';
    document.getElementById('status-text').textContent = 'Reconnecting…';
    setTimeout(connect, 2000);
  };

  ws.onmessage = ({ data }) => {
    const msg = JSON.parse(data);
    if (msg.type === 'analysis') onAnalysis(msg);
    else if (msg.type === 'spike') onSpike(msg);
    else if (msg.type === 'summary') onSummary(msg);
    else if (msg.type === 'session_start') {
      document.getElementById('session-label').textContent = 'Session: ' + msg.session_id;
    }
  };
}

function fmt(v) { return v != null ? v.toFixed(2) : '—'; }
function fmtPct(v) { return v != null ? (v * 100).toFixed(0) + '%' : '—'; }

function onAnalysis(e) {
  // Cards
  document.getElementById('stress-val').textContent = fmtPct(e.candidate_stress);
  document.getElementById('stress-conf').textContent = 'conf ' + fmt(e.stress_confidence);
  document.getElementById('agg-val').textContent = fmtPct(e.interviewer_aggression);
  document.getElementById('agg-conf').textContent = 'conf ' + fmt(e.aggression_confidence);
  document.getElementById('window-id').textContent = e.window_id;

  // Risk card
  const icons = { none: '✅ None', moderate: '⚠️ Moderate', high: '🚨 High' };
  const trends = { increasing: '📈 Rising', stable: '➡️ Stable', decreasing: '📉 Falling' };
  document.getElementById('risk-val').textContent = icons[e.risk_flag] || e.risk_flag;
  document.getElementById('trend-val').textContent = trends[e.pressure_trend] || e.pressure_trend;
  const riskCard = document.getElementById('risk-card');
  riskCard.className = 'card risk ' + (e.risk_flag || 'none');

  // Resilience
  if (e.resilience_score != null)
    document.getElementById('resilience-val').textContent = fmtPct(e.resilience_score);

  // Coaching
  const coachEl = document.getElementById('coaching');
  coachEl.classList.add('updating');
  setTimeout(() => { coachEl.textContent = e.coaching_feedback || '—'; coachEl.classList.remove('updating'); }, 200);

  // Style
  const badge = document.getElementById('style-badge');
  badge.textContent = (e.aggression_class || 'neutral').replace('_', ' ').toUpperCase();
  badge.className = 'style-badge ' + (e.aggression_class || 'neutral');
  document.getElementById('dominant-val').textContent = e.dominant_party || '—';
  document.getElementById('imbalance-val').textContent = e.imbalance_score != null ? fmt(e.imbalance_score) : '—';

  // Chart
  push(e.window_id, e.candidate_stress, e.interviewer_aggression);
}

function onSpike(e) {
  const list = document.getElementById('spike-list');
  const noSpikes = list.querySelector('.no-spikes');
  if (noSpikes) noSpikes.remove();

  spikeCount++;
  const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  const li = document.createElement('li');
  li.className = 'spike-item';
  li.innerHTML = `
    <span class="spike-icon">⚡</span>
    <span>
      <span class="spike-sigma">${e.delta_sigma.toFixed(1)}σ</span>
      <span class="spike-cause"> ${e.cause.replace('_', ' ')}</span>
      <div style="font-size:10px;color:var(--muted)">
        ${fmt(e.stress_before)} → ${fmt(e.stress_after)}
      </div>
    </span>
    <span class="spike-time">${now}</span>`;
  list.insertBefore(li, list.firstChild);
}

function onSummary(e) {
  const s = e.summary;
  document.getElementById('sum-session-id').textContent = s.session_id + ' · ' + s.duration_s + 's';

  const grid = document.getElementById('sum-grid');
  grid.innerHTML = [
    ['Peak Stress', fmtPct(s.peak_stress), 'var(--stress-color)'],
    ['Avg Stress', fmtPct(s.average_stress), 'var(--orange)'],
    ['Resilience', fmtPct(s.resilience_score), 'var(--green)'],
    ['Spikes', s.spike_count, 'var(--red)'],
    ['Filler Rate', (s.filler_word_density_per_min || 0).toFixed(1) + '/min', 'var(--muted)'],
    ['Style', (s.dominant_aggression_class || '—').replace('_', ' '), 'var(--purple)'],
  ].map(([label, val, color]) => `
    <div class="summary-metric">
      <div class="m-label">${label}</div>
      <div class="m-value" style="color:${color}">${val}</div>
    </div>`).join('');

  // Comparison
  if (e.comparison && e.comparison.deltas && e.comparison.deltas.length) {
    document.getElementById('sum-comparison').style.display = 'block';
    document.getElementById('sum-deltas').innerHTML = e.comparison.deltas.map(d =>
      `<div class="delta-row">
        <span>${d.metric.replace(/_/g, ' ')}</span>
        <span class="${d.improved ? 'delta-improved' : 'delta-regressed'}">
          ${d.direction_symbol} ${d.delta_percent != null ? Math.abs(d.delta_percent).toFixed(1) + '%' : ''}
        </span>
      </div>`
    ).join('');
  }

  // Strengths / improvements
  document.getElementById('sum-strengths').innerHTML =
    (s.strengths || []).map(t => `<li>${t}</li>`).join('') || '<li>—</li>';
  document.getElementById('sum-improvements').innerHTML =
    (s.improvement_areas || []).map(t => `<li>${t}</li>`).join('') || '<li>—</li>';

  document.getElementById('summary-overlay').classList.add('show');
}

function closeSummary() {
  document.getElementById('summary-overlay').classList.remove('show');
}

connect();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# WebSocket broadcast hub
# ---------------------------------------------------------------------------

class _Dashboard:
    """Manages connected WebSocket clients and broadcasts events as JSON."""

    def __init__(self, call_id: str, stream_api_key: str, stream_api_secret: str) -> None:
        self._clients: set[WebSocket] = set()
        self._call_id = call_id
        self._stream_client = Stream(api_key=stream_api_key, api_secret=stream_api_secret)
        self.app = FastAPI(title="Interview Pressure Analyzer")
        self._register_routes()

    def _register_routes(self) -> None:
        @self.app.get("/", response_class=HTMLResponse)
        async def index() -> str:
            return _DASHBOARD_HTML

        @self.app.get("/join")
        async def join(user_name: str = "Candidate") -> RedirectResponse:
            user_id = user_name.lower().replace(" ", "_")
            token = self._stream_client.create_token(user_id, expiration=3600)
            params = {
                "api_key": self._stream_client.api_key,
                "token": token,
                "skip_lobby": "true",
                "user_name": user_name,
                "video_encoder": "h264",
            }
            url = f"https://getstream.io/video/demos/join/{self._call_id}?{urlencode(params)}"
            return RedirectResponse(url)

        @self.app.websocket("/ws")
        async def ws_endpoint(websocket: WebSocket) -> None:
            await websocket.accept()
            self._clients.add(websocket)
            logger.info("Dashboard client connected (%d total)", len(self._clients))
            try:
                while True:
                    await websocket.receive_text()
            except WebSocketDisconnect:
                self._clients.discard(websocket)
                logger.info("Dashboard client disconnected (%d total)", len(self._clients))

    async def broadcast(self, payload: dict) -> None:
        dead: set[WebSocket] = set()
        for ws in self._clients:
            try:
                await ws.send_json(payload)
            except (RuntimeError, WebSocketDisconnect):
                dead.add(ws)
        self._clients -= dead


# ---------------------------------------------------------------------------
# Terminal formatting helpers
# ---------------------------------------------------------------------------

def _format_analysis(event: BehavioralAnalysisEvent) -> str:
    risk = _RISK_ICONS.get(event.risk_flag, "")
    trend = _TREND_ICONS.get(event.pressure_trend, "")
    stress = f"{event.candidate_stress:.2f}" if event.candidate_stress is not None else "n/a"
    agg = f"{event.interviewer_aggression:.2f}" if event.interviewer_aggression is not None else "n/a"

    lines = [
        f"\n{'─'*60}",
        f"  {risk} Window {event.window_id}  {trend} {event.pressure_trend}",
        f"{'─'*60}",
        f"  stress={stress} (conf={event.stress_confidence:.2f})  "
        f"aggression={agg} (conf={event.aggression_confidence:.2f})",
        f"  style={event.aggression_class}  "
        f"dominant={event.dominant_party}  imbalance={event.imbalance_score:.2f}",
        f"  risk={event.risk_flag}  signals={event.key_signals}",
        f"  coach → \"{event.coaching_feedback}\"",
        f"{'─'*60}",
    ]
    return "\n".join(lines)


def _format_spike(event: SpikeDetectedEvent) -> str:
    return (
        f"\n⚡ SPIKE {event.delta_sigma:.1f}σ  cause={event.cause}  "
        f"{event.stress_before:.2f} → {event.stress_after:.2f}  "
        f"window={event.window_id}"
    )


# ---------------------------------------------------------------------------
# Core interview runner
# ---------------------------------------------------------------------------

async def run_interview(
    call_id: str,
    candidate_user_id: Optional[str],
    dashboard: _Dashboard,
    previous_summary: Optional[SessionSummary],
    comparison_engine: SessionComparisonEngine,
) -> Optional[SessionSummary]:
    """Run one interview session and return the session summary."""
    analysis_model = os.environ.get("ANALYSIS_MODEL", "anthropic/claude-haiku-4-5")
    interviewer_model = os.environ.get("INTERVIEWER_MODEL", "anthropic/claude-haiku-4-5")
    session_id = call_id
    logger.info(
        "Starting analyzer (analysis_model=%s, interviewer_model=%s, session=%s)",
        analysis_model, interviewer_model, session_id,
    )

    await dashboard.broadcast({"type": "session_start", "session_id": session_id})

    analysis_llm = openrouter.LLM(model=analysis_model, max_tokens=400, temperature=0.2)
    analyzer = BehavioralAnalyzer(llm=analysis_llm, analysis_interval_s=5.0)
    analyzer.start_session(session_id=session_id)

    bridge = InterviewBridge(analyzer=analyzer, candidate_user_id=candidate_user_id)

    processors = [bridge, analyzer]

    if _YOLO_AVAILABLE:
        pose = PoseBridge(
            analyzer=analyzer,
            model_path="yolo11n-pose.pt",
            fps=15,
            enable_hand_tracking=False,
        )
        processors.insert(0, pose)
        logger.info("PoseBridge enabled")
    else:
        logger.info("PoseBridge skipped (ultralytics not installed)")

    interviewer_llm = openrouter.LLM(model=interviewer_model, max_tokens=400, temperature=0.2)

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="AI Interviewer", id="ai_interviewer"),
        instructions=INTERVIEWER_INSTRUCTIONS,
        llm=interviewer_llm,
        stt=deepgram.STT(eager_turn_detection=True),
        tts=elevenlabs.TTS(model_id="eleven_flash_v2_5"),
        processors=processors,
    )

    @agent.events.subscribe
    async def on_analysis(event: BehavioralAnalysisEvent):
        print(_format_analysis(event))
        await dashboard.broadcast({
            "type": "analysis",
            "window_id": event.window_id,
            "candidate_stress": event.candidate_stress,
            "stress_confidence": event.stress_confidence,
            "interviewer_aggression": event.interviewer_aggression,
            "aggression_confidence": event.aggression_confidence,
            "pressure_trend": event.pressure_trend,
            "dominant_party": event.dominant_party,
            "imbalance_score": event.imbalance_score,
            "risk_flag": event.risk_flag,
            "key_signals": event.key_signals,
            "coaching_feedback": event.coaching_feedback,
            "aggression_class": event.aggression_class,
            "aggression_class_confidence": event.aggression_class_confidence,
            "resilience_score": getattr(event, "resilience_score", None),
        })

        if event.risk_flag == "high":
            logger.warning(
                "HIGH RISK — stress=%.2f aggression=%.2f",
                event.candidate_stress or 0.0,
                event.interviewer_aggression or 0.0,
            )

    @agent.events.subscribe
    async def on_spike(event: SpikeDetectedEvent):
        print(_format_spike(event))
        await dashboard.broadcast({
            "type": "spike",
            "window_id": event.window_id,
            "stress_before": event.stress_before,
            "stress_after": event.stress_after,
            "delta_sigma": event.delta_sigma,
            "cause": event.cause,
        })

    captured_summary: list[SessionSummary] = []
    _summary_sent = False
    _stop_requested: asyncio.Event = asyncio.Event()

    async def _send_summary_to_dashboard() -> None:
        nonlocal _summary_sent
        if _summary_sent:
            return
        _summary_sent = True
        summary = analyzer.generate_summary()
        if summary is None:
            return
        captured_summary.append(summary)

        comparison_payload = None
        if previous_summary is not None:
            comparison = comparison_engine.compare(previous_summary, summary)
            comparison_payload = {
                "overall_improvement_score": comparison.overall_improvement_score,
                "deltas": [
                    {
                        "metric": d.metric,
                        "session_1": d.session_1,
                        "session_2": d.session_2,
                        "delta_percent": d.delta_percent,
                        "improved": d.improved,
                        "direction_symbol": d.direction_symbol,
                    }
                    for d in comparison.deltas
                ],
                "summary_lines": comparison.summary_lines,
            }
            print("\n📊 Session Comparison:")
            for line in comparison.summary_lines:
                print(f"  {line}")

        await dashboard.broadcast({
            "type": "summary",
            "summary": {
                "session_id": summary.session_id,
                "duration_s": summary.duration_s,
                "peak_stress": summary.peak_stress,
                "average_stress": summary.average_stress,
                "resilience_score": summary.resilience_score,
                "spike_count": summary.spike_count,
                "filler_word_density_per_min": summary.filler_word_density_per_min,
                "dominant_aggression_class": summary.dominant_aggression_class,
                "dominant_party_summary": summary.dominant_party_summary,
                "strengths": summary.strengths,
                "improvement_areas": summary.improvement_areas,
            },
            "comparison": comparison_payload,
        })

    @agent.events.subscribe
    async def on_session_summary(event: SessionSummaryEvent):
        await _send_summary_to_dashboard()

    @agent.events.subscribe
    async def on_participant_left(event: ParticipantLeftEvent):
        participant = event.participant
        if participant is None or participant.user_id == "ai_interviewer":
            return
        if candidate_user_id and participant.user_id != candidate_user_id:
            return
        logger.info("Candidate left — sending early session summary")
        await _send_summary_to_dashboard()
        _stop_requested.set()

    await agent.create_user()
    call = await agent.create_call("default", call_id)

    async with agent.join(call, participant_wait_timeout=None):
        await analyzer.start()
        logger.info("Agent joined call. Dashboard: http://localhost:8080")

        await agent.simple_response(
            "Hello! I'm ready to begin. Could you start by telling me "
            "a bit about your background and what you're looking for?"
        )

        finish_task = asyncio.create_task(agent.finish())
        stop_task = asyncio.create_task(_stop_requested.wait())
        done, pending = await asyncio.wait(
            {finish_task, stop_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
        await analyzer.stop()

    logger.info("Interview session ended.")
    return captured_summary[0] if captured_summary else None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def _main(call_id: str, candidate_user_id: Optional[str], port: int) -> None:
    dashboard = _Dashboard(
        call_id=call_id,
        stream_api_key=os.environ["STREAM_API_KEY"],
        stream_api_secret=os.environ["STREAM_API_SECRET"],
    )
    comparison_engine = SessionComparisonEngine()
    previous_summary: Optional[SessionSummary] = None

    server_config = uvicorn.Config(
        dashboard.app,
        host="0.0.0.0",
        port=port,
        log_level="warning",
    )
    server = uvicorn.Server(server_config)

    logger.info("Dashboard:  http://localhost:%d", port)
    logger.info("Join call:  http://localhost:%d/join", port)

    # Open both tabs automatically after a brief delay
    async def _open_browser() -> None:
        await asyncio.sleep(2.5)
        await asyncio.to_thread(webbrowser.open, f"http://localhost:{port}")
        await asyncio.sleep(0.5)
        await asyncio.to_thread(webbrowser.open, f"http://localhost:{port}/join")

    asyncio.ensure_future(_open_browser())

    async def _run_interview_loop() -> None:
        nonlocal previous_summary
        summary = await run_interview(
            call_id, candidate_user_id, dashboard, previous_summary, comparison_engine
        )
        if summary is not None:
            previous_summary = summary
        server.should_exit = True

    await asyncio.gather(
        server.serve(),
        _run_interview_loop(),
    )


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="AI Interview Pressure Analyzer")
    parser.add_argument("--call-id", required=True, help="GetStream call ID to join")
    parser.add_argument(
        "--candidate-user-id",
        default=None,
        help="user_id of the candidate (auto-detected if omitted)",
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Dashboard port (default: 8080)"
    )
    args = parser.parse_args()

    asyncio.run(_main(args.call_id, args.candidate_user_id, args.port))


if __name__ == "__main__":
    main()
