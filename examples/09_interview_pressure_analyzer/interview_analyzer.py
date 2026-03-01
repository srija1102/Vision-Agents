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
import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import webbrowser
from urllib.parse import urlencode

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse
from getstream import Stream

import vision_agents.plugins.deepgram as deepgram
import vision_agents.plugins.elevenlabs as elevenlabs
import vision_agents.plugins.getstream as getstream
import vision_agents.plugins.openrouter as openrouter
from vision_agents.plugins.getstream.sfu_events import ParticipantLeftEvent
from vision_agents.core.stt.events import STTTranscriptEvent
from vision_agents.core.llm.events import LLMResponseCompletedEvent
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

  .transcript-panel {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px;
  }
  .transcript-list { max-height: 260px; overflow-y: auto; }
  .transcript-list::-webkit-scrollbar { width: 4px; }
  .transcript-list::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
  .transcript-entry {
    display: flex; gap: 10px; padding: 8px 0;
    border-bottom: 1px solid var(--border); font-size: 13px;
    animation: slide-in .25s ease;
  }
  .transcript-entry:last-child { border-bottom: none; }
  .t-speaker {
    font-size: 10px; font-weight: 700; letter-spacing: .5px; text-transform: uppercase;
    white-space: nowrap; min-width: 76px; padding-top: 2px;
  }
  .t-speaker.candidate { color: var(--blue); }
  .t-speaker.interviewer { color: var(--purple); }
  .t-text { color: var(--text); line-height: 1.5; flex: 1; }
  .t-time { color: var(--muted); font-size: 10px; white-space: nowrap; padding-top: 2px; }
  .no-transcript { color: var(--muted); font-size: 12px; font-style: italic; }

  #ended-banner {
    display: none; background: #161b22; border-bottom: 1px solid var(--border);
    padding: 10px 24px; text-align: center; font-size: 13px; color: var(--muted);
    position: sticky; top: 53px; z-index: 9;
  }
  #ended-banner.show { display: block; }

  .history-btn {
    background: var(--surface); border: 1px solid var(--border); color: var(--text);
    border-radius: 8px; padding: 5px 14px; font-size: 12px; cursor: pointer;
    transition: border-color .2s;
  }
  .history-btn:hover { border-color: var(--blue); }

  #sessions-modal {
    display: none; position: fixed; inset: 0; background: rgba(13,17,23,.92);
    z-index: 200; align-items: center; justify-content: center;
  }
  #sessions-modal.show { display: flex; }
  .sessions-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 14px; padding: 28px; width: 620px; max-height: 80vh;
    display: flex; flex-direction: column; animation: pop-in .25s ease;
  }
  .sessions-card h2 { font-size: 17px; margin-bottom: 18px; }
  .sessions-list { flex: 1; overflow-y: auto; }
  .sessions-list::-webkit-scrollbar { width: 4px; }
  .sessions-list::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
  .session-row {
    display: flex; align-items: center; gap: 12px; padding: 10px 0;
    border-bottom: 1px solid var(--border); font-size: 13px;
  }
  .session-row:last-child { border-bottom: none; }
  .session-id-col { font-weight: 600; flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .session-meta { color: var(--muted); font-size: 11px; white-space: nowrap; }
  .session-stress { font-size: 11px; padding: 2px 8px; border-radius: 20px; background: var(--bg); white-space: nowrap; }
  .load-btn {
    background: var(--blue); color: white; border: none; border-radius: 6px;
    padding: 4px 12px; font-size: 12px; cursor: pointer; white-space: nowrap;
  }
  .load-btn:hover { opacity: .85; }
  .no-sessions { color: var(--muted); font-size: 13px; font-style: italic; padding: 16px 0; }
</style>
</head>
<body>

<header>
  <h1>🎙 <span>Interview</span> Pressure Analyzer</h1>
  <span class="session-badge" id="session-label">No session</span>
  <div style="display:flex;align-items:center;gap:12px;">
    <button class="history-btn" onclick="openSessions()">📁 History</button>
    <div class="status-dot">
      <div class="dot connecting" id="dot"></div>
      <span id="status-text">Connecting…</span>
    </div>
  </div>
</header>

<!-- Sessions History Modal -->
<div id="sessions-modal">
  <div class="sessions-card">
    <h2>📁 Session History</h2>
    <div class="sessions-list" id="sessions-list"><div class="no-sessions">Loading…</div></div>
    <button class="close-btn" style="margin-top:16px" onclick="closeSessions()">Close</button>
  </div>
</div>

<div id="ended-banner">
  Session ended — review your performance below &nbsp;·&nbsp; Ctrl-C in terminal to quit
</div>

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

<div style="padding: 0 24px 24px">
  <div class="transcript-panel">
    <div class="panel-title">💬 Conversation Transcript</div>
    <div class="transcript-list" id="transcript-list">
      <div class="no-transcript">Transcript will appear here when the session starts…</div>
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
let sessionEnded = false;

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
    if (sessionEnded) {
      document.getElementById('dot').className = 'dot';
      document.getElementById('status-text').textContent = 'Session Ended';
      return;
    }
    document.getElementById('dot').className = 'dot connecting';
    document.getElementById('status-text').textContent = 'Reconnecting…';
    setTimeout(connect, 2000);
  };

  ws.onmessage = ({ data }) => {
    const msg = JSON.parse(data);
    if (msg.type === 'analysis') onAnalysis(msg);
    else if (msg.type === 'spike') onSpike(msg);
    else if (msg.type === 'summary') onSummary(msg);
    else if (msg.type === 'transcript') onTranscript(msg);
    else if (msg.type === 'session_start') {
      document.getElementById('session-label').textContent = 'Session: ' + msg.session_id;
    } else if (msg.type === 'session_ended') {
      sessionEnded = true;
      document.getElementById('ended-banner').classList.add('show');
      document.getElementById('dot').className = 'dot';
      document.getElementById('status-text').textContent = 'Session Ended';
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

function onTranscript(e) {
  const list = document.getElementById('transcript-list');
  const placeholder = list.querySelector('.no-transcript');
  if (placeholder) placeholder.remove();
  const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  const label = e.speaker === 'candidate' ? '🎤 You' : '🤖 AI';
  const div = document.createElement('div');
  div.className = 'transcript-entry';
  div.innerHTML = `
    <span class="t-speaker ${e.speaker}">${label}</span>
    <span class="t-text">${e.text}</span>
    <span class="t-time">${now}</span>`;
  list.appendChild(div);
  list.scrollTop = list.scrollHeight;
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

// ── Sessions history ──────────────────────────────────────────────────────

async function openSessions() {
  document.getElementById('sessions-modal').classList.add('show');
  const list = document.getElementById('sessions-list');
  list.innerHTML = '<div class="no-sessions">Loading…</div>';
  try {
    const sessions = await fetch('/api/sessions').then(r => r.json());
    if (!sessions.length) {
      list.innerHTML = '<div class="no-sessions">No sessions recorded yet.</div>';
      return;
    }
    list.innerHTML = sessions.map(s => {
      const date = s.started_at ? new Date(s.started_at).toLocaleString() : '—';
      const dur  = s.duration_s  ? Math.round(s.duration_s) + 's' : '—';
      const stress = s.peak_stress != null ? (s.peak_stress * 100).toFixed(0) + '% stress' : '—';
      const style  = (s.dominant_agg_class || 'neutral').replace('_', ' ');
      return `<div class="session-row">
        <span class="session-id-col" title="${s.session_id}">${s.session_id}</span>
        <span class="session-meta">${date}</span>
        <span class="session-meta">${dur}</span>
        <span class="session-stress">${stress} · ${style}</span>
        <button class="load-btn" onclick="loadSession('${s.session_id}')">Load →</button>
      </div>`;
    }).join('');
  } catch {
    list.innerHTML = '<div class="no-sessions">Failed to load sessions.</div>';
  }
}

function closeSessions() {
  document.getElementById('sessions-modal').classList.remove('show');
}

function resetDashboard() {
  labels.length = 0; stressData.length = 0; aggData.length = 0;
  chart.update();
  document.getElementById('transcript-list').innerHTML =
    '<div class="no-transcript">Transcript will appear here when the session starts…</div>';
  document.getElementById('spike-list').innerHTML =
    '<li class="no-spikes">No spikes detected yet</li>';
  document.getElementById('summary-overlay').classList.remove('show');
  spikeCount = 0;
}

async function loadSession(sessionId) {
  closeSessions();
  resetDashboard();
  try {
    const data = await fetch(`/api/sessions/${sessionId}`).then(r => r.json());
    document.getElementById('session-label').textContent = 'Session: ' + sessionId;

    for (const w of (data.windows || [])) {
      onAnalysis({
        window_id: w.window_id,
        candidate_stress: w.stress,
        stress_confidence: w.stress_conf,
        interviewer_aggression: w.aggression,
        aggression_confidence: w.agg_conf,
        risk_flag: w.risk,
        pressure_trend: w.trend,
        dominant_party: w.dominant,
        imbalance_score: w.imbalance,
        coaching_feedback: w.coaching,
        aggression_class: w.agg_class,
        resilience_score: w.resilience,
      });
    }

    for (const s of (data.spikes || [])) {
      onSpike({ window_id: s.window_id, stress_before: s.stress_before,
                stress_after: s.stress_after, delta_sigma: s.delta_sigma, cause: s.cause });
    }

    for (const t of (data.transcript || [])) {
      onTranscript({ speaker: t.speaker, text: t.text });
    }

    if (data.summary) {
      const s = data.summary;
      onSummary({
        summary: {
          session_id: sessionId,
          duration_s: (data.session || {}).duration_s || 0,
          peak_stress: s.peak_stress,
          average_stress: s.average_stress,
          resilience_score: s.resilience_score,
          spike_count: s.spike_count,
          filler_word_density_per_min: s.filler_density,
          dominant_aggression_class: s.dominant_agg_class,
          dominant_party_summary: s.dominant_party,
          strengths: JSON.parse(s.strengths || '[]'),
          improvement_areas: JSON.parse(s.improvement_areas || '[]'),
        },
        comparison: null,
      });
    }
  } catch (err) {
    alert('Failed to load session: ' + err.message);
  }
}

connect();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# SQLite session database
# ---------------------------------------------------------------------------

_DB_PATH = Path("sessions/interviews.db")


class SessionDB:
    """Persistent SQLite store for all session data."""

    def __init__(self) -> None:
        _DB_PATH.parent.mkdir(exist_ok=True)
        self._path = str(_DB_PATH)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    call_id    TEXT,
                    started_at TEXT,
                    ended_at   TEXT,
                    duration_s REAL
                );
                CREATE TABLE IF NOT EXISTS windows (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id  TEXT,
                    window_id   TEXT,
                    t           REAL,
                    stress      REAL,
                    stress_conf REAL,
                    aggression  REAL,
                    agg_conf    REAL,
                    risk        TEXT,
                    trend       TEXT,
                    dominant    TEXT,
                    imbalance   REAL,
                    coaching    TEXT,
                    agg_class   TEXT,
                    resilience  REAL
                );
                CREATE TABLE IF NOT EXISTS spikes (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id   TEXT,
                    window_id    TEXT,
                    t            REAL,
                    stress_before REAL,
                    stress_after  REAL,
                    delta_sigma  REAL,
                    cause        TEXT
                );
                CREATE TABLE IF NOT EXISTS transcripts (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    speaker    TEXT,
                    text       TEXT,
                    t          REAL
                );
                CREATE TABLE IF NOT EXISTS summaries (
                    session_id        TEXT PRIMARY KEY,
                    peak_stress       REAL,
                    average_stress    REAL,
                    resilience_score  REAL,
                    spike_count       INTEGER,
                    filler_density    REAL,
                    dominant_agg_class TEXT,
                    dominant_party    TEXT,
                    strengths         TEXT,
                    improvement_areas TEXT
                );
            """)

    def upsert_session(self, session_id: str, call_id: str, started_at: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO sessions (session_id, call_id, started_at) VALUES (?,?,?)",
                (session_id, call_id, started_at),
            )

    def update_session_end(self, session_id: str, ended_at: str, duration_s: float) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET ended_at=?, duration_s=? WHERE session_id=?",
                (ended_at, duration_s, session_id),
            )

    def insert_window(self, session_id: str, msg: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO windows
                   (session_id,window_id,t,stress,stress_conf,aggression,agg_conf,
                    risk,trend,dominant,imbalance,coaching,agg_class,resilience)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    session_id, msg.get("window_id"), time.time(),
                    msg.get("candidate_stress"), msg.get("stress_confidence"),
                    msg.get("interviewer_aggression"), msg.get("aggression_confidence"),
                    msg.get("risk_flag"), msg.get("pressure_trend"),
                    msg.get("dominant_party"), msg.get("imbalance_score"),
                    msg.get("coaching_feedback"), msg.get("aggression_class"),
                    msg.get("resilience_score"),
                ),
            )

    def insert_spike(self, session_id: str, msg: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO spikes
                   (session_id,window_id,t,stress_before,stress_after,delta_sigma,cause)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    session_id, msg.get("window_id"), time.time(),
                    msg.get("stress_before"), msg.get("stress_after"),
                    msg.get("delta_sigma"), msg.get("cause"),
                ),
            )

    def insert_transcript(self, session_id: str, msg: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO transcripts (session_id,speaker,text,t) VALUES (?,?,?,?)",
                (session_id, msg.get("speaker"), msg.get("text"), time.time()),
            )

    def upsert_summary(self, session_id: str, s: dict, duration_s: float) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO summaries
                   (session_id,peak_stress,average_stress,resilience_score,spike_count,
                    filler_density,dominant_agg_class,dominant_party,strengths,improvement_areas)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    session_id,
                    s.get("peak_stress"), s.get("average_stress"),
                    s.get("resilience_score"), s.get("spike_count"),
                    s.get("filler_word_density_per_min"),
                    s.get("dominant_aggression_class"), s.get("dominant_party_summary"),
                    json.dumps(s.get("strengths") or []),
                    json.dumps(s.get("improvement_areas") or []),
                ),
            )
            conn.execute(
                "UPDATE sessions SET duration_s=?, ended_at=? WHERE session_id=?",
                (duration_s, datetime.now(timezone.utc).isoformat(), session_id),
            )

    def list_sessions(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT s.session_id, s.call_id, s.started_at, s.ended_at, s.duration_s,
                          sm.peak_stress, sm.dominant_agg_class
                   FROM sessions s
                   LEFT JOIN summaries sm ON s.session_id = sm.session_id
                   ORDER BY s.started_at DESC"""
            ).fetchall()
        return [dict(r) for r in rows]

    def get_session(self, session_id: str) -> dict | None:
        with self._connect() as conn:
            sess = conn.execute(
                "SELECT * FROM sessions WHERE session_id=?", (session_id,)
            ).fetchone()
            if not sess:
                return None
            windows = conn.execute(
                "SELECT * FROM windows WHERE session_id=? ORDER BY t", (session_id,)
            ).fetchall()
            spikes = conn.execute(
                "SELECT * FROM spikes WHERE session_id=? ORDER BY t", (session_id,)
            ).fetchall()
            transcript = conn.execute(
                "SELECT * FROM transcripts WHERE session_id=? ORDER BY t", (session_id,)
            ).fetchall()
            summary = conn.execute(
                "SELECT * FROM summaries WHERE session_id=?", (session_id,)
            ).fetchone()
        return {
            "session": dict(sess),
            "windows": [dict(r) for r in windows],
            "spikes": [dict(r) for r in spikes],
            "transcript": [dict(r) for r in transcript],
            "summary": dict(summary) if summary else None,
        }


# ---------------------------------------------------------------------------
# WebSocket broadcast hub
# ---------------------------------------------------------------------------

class _Dashboard:
    """Manages connected WebSocket clients and broadcasts events as JSON."""

    def __init__(self, call_id: str, stream_api_key: str, stream_api_secret: str) -> None:
        self._clients: set[WebSocket] = set()
        self._call_id = call_id
        self._stream_client = Stream(api_key=stream_api_key, api_secret=stream_api_secret)
        self._buffer: list[dict] = []
        self._current_session_id: str = ""
        self._db = SessionDB()
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

        @self.app.get("/api/sessions")
        async def api_sessions() -> list:
            return await asyncio.to_thread(self._db.list_sessions)

        @self.app.get("/api/sessions/{session_id}")
        async def api_session(session_id: str) -> dict:
            data = await asyncio.to_thread(self._db.get_session, session_id)
            if data is None:
                raise HTTPException(status_code=404, detail="Session not found")
            return data

        @self.app.websocket("/ws")
        async def ws_endpoint(websocket: WebSocket) -> None:
            await websocket.accept()
            for msg in list(self._buffer):
                try:
                    await websocket.send_json(msg)
                except (RuntimeError, WebSocketDisconnect):
                    return
            self._clients.add(websocket)
            logger.info("Dashboard client connected (%d total)", len(self._clients))
            try:
                while True:
                    await websocket.receive_text()
            except WebSocketDisconnect:
                self._clients.discard(websocket)
                logger.info("Dashboard client disconnected (%d total)", len(self._clients))

    async def broadcast(self, payload: dict) -> None:
        msg_type = payload.get("type")
        if msg_type == "session_start":
            self._buffer = [payload]
            self._current_session_id = payload.get("session_id", "")
            await asyncio.to_thread(
                self._db.upsert_session,
                self._current_session_id,
                self._call_id,
                datetime.now(timezone.utc).isoformat(),
            )
        else:
            self._buffer.append(payload)

        sid = self._current_session_id
        if sid:
            if msg_type == "analysis":
                await asyncio.to_thread(self._db.insert_window, sid, payload)
            elif msg_type == "spike":
                await asyncio.to_thread(self._db.insert_spike, sid, payload)
            elif msg_type == "transcript":
                await asyncio.to_thread(self._db.insert_transcript, sid, payload)
            elif msg_type == "summary":
                await asyncio.to_thread(
                    self._db.upsert_summary, sid,
                    payload.get("summary", {}),
                    payload.get("summary", {}).get("duration_s", 0),
                )

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
# Session persistence
# ---------------------------------------------------------------------------

def _save_session(
    session_id: str,
    summary: SessionSummary,
    transcript: list[dict],
    start_time: float,
) -> None:
    sessions_dir = Path("sessions")
    sessions_dir.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = sessions_dir / f"{session_id}_{ts}.json"
    data = {
        "session_id": session_id,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "duration_s": summary.duration_s,
        "summary": {
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
        "transcript": transcript,
    }
    path.write_text(json.dumps(data, indent=2))
    logger.info("Session saved → %s", path)


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
        tts=deepgram.TTS(),
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
    transcript_log: list[dict] = []
    session_start_time = time.time()
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
        _save_session(session_id, summary, transcript_log, session_start_time)

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

    @agent.events.subscribe
    async def on_candidate_transcript(event: STTTranscriptEvent):
        entry = {"speaker": "candidate", "text": event.text, "t": round(time.time() - session_start_time, 1)}
        transcript_log.append(entry)
        await dashboard.broadcast({"type": "transcript", "speaker": "candidate", "text": event.text})

    @agent.events.subscribe
    async def on_interviewer_response(event: LLMResponseCompletedEvent):
        if not event.text:
            return
        entry = {"speaker": "interviewer", "text": event.text, "t": round(time.time() - session_start_time, 1)}
        transcript_log.append(entry)
        await dashboard.broadcast({"type": "transcript", "speaker": "interviewer", "text": event.text})

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

    # Open both tabs automatically on local runs only (skipped when deployed)
    if not os.environ.get("RAILWAY_ENVIRONMENT") and not os.environ.get("NO_BROWSER"):
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
        await dashboard.broadcast({"type": "session_ended"})
        logger.info(
            "Session complete — dashboard open at http://localhost:%d  (Ctrl-C to quit)", port
        )

    await asyncio.gather(
        server.serve(),
        _run_interview_loop(),
    )


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="AI Interview Pressure Analyzer")
    parser.add_argument(
        "--call-id",
        default=os.environ.get("CALL_ID"),
        help="GetStream call ID to join (or set CALL_ID env var)",
    )
    parser.add_argument(
        "--candidate-user-id",
        default=None,
        help="user_id of the candidate (auto-detected if omitted)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", 8080)),
        help="Dashboard port (default: 8080, or PORT env var)",
    )
    args = parser.parse_args()

    if not args.call_id:
        parser.error("--call-id is required (or set the CALL_ID environment variable)")

    asyncio.run(_main(args.call_id, args.candidate_user_id, args.port))


if __name__ == "__main__":
    main()
