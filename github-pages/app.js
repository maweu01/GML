/**
 * Guardian ML — Static Frontend (Mode B / GitHub Pages)
 * Self-contained simulation using data/config.json
 * NO backend required. All paths are relative.
 */

"use strict";

// ── State ──────────────────────────────────────────────────────────
const State = {
  config: null,
  filtered: [],
  threshold: 0.5,
  analysisRun: false,
};

// ── Plotly dark layout defaults ────────────────────────────────────
const PLOT_LAYOUT = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor:  "rgba(0,0,0,0)",
  font: { color: "#8b949e", family: "Segoe UI, system-ui, sans-serif", size: 12 },
  margin: { t: 20, r: 20, b: 40, l: 40 },
  xaxis: { gridcolor: "#30363d", linecolor: "#30363d", zerolinecolor: "#30363d" },
  yaxis: { gridcolor: "#30363d", linecolor: "#30363d", zerolinecolor: "#30363d" },
  legend: { bgcolor: "rgba(0,0,0,0)", bordercolor: "#30363d" },
};
const PLOT_CONFIG = { displayModeBar: false, responsive: true };

// ── Bootstrap ──────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
  await loadConfig();
  bindEvents();
  populateDashboardCards();
  document.getElementById("model-label").textContent =
    `Model: ${State.config.best_model.replace(/_/g," ")}`;
});

// ── Load config.json (relative path — works on GH Pages + file://) ─
async function loadConfig() {
  try {
    const res = await fetch("data/config.json");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    State.config = await res.json();
  } catch (e) {
    console.error("Failed to load config.json:", e);
    showError("Could not load configuration. Ensure data/config.json exists.");
  }
}

// ── Populate summary dashboard cards ──────────────────────────────
function populateDashboardCards() {
  const stats = State.config.system_stats;
  const metrics = State.config.model_metrics[State.config.best_model];

  setText("dc-total",   stats.total_analyzed.toLocaleString());
  setText("dc-high",    stats.high_risk_count.toLocaleString());
  setText("dc-medium",  stats.medium_risk_count.toLocaleString());
  setText("dc-low",     stats.low_risk_count.toLocaleString());
  setText("dc-f1",      `${(metrics.f1_score * 100).toFixed(1)}%`);
  setText("dc-trained", stats.last_trained);
}

// ── Event Bindings ─────────────────────────────────────────────────
function bindEvents() {
  document.getElementById("btn-run").addEventListener("click", runAnalysis);
  document.getElementById("btn-refresh").addEventListener("click", refreshData);

  document.getElementById("profile-select").addEventListener("change", () => {
    if (State.analysisRun) applyFilter();
  });

  const slider = document.getElementById("threshold-slider");
  slider.addEventListener("input", () => {
    const val = parseInt(slider.value, 10);
    State.threshold = val / 100;
    document.getElementById("threshold-val").textContent = `${val}%`;
    if (State.analysisRun) applyFilter();
  });
}

// ── Run Analysis ───────────────────────────────────────────────────
async function runAnalysis() {
  showSpinner("Running ML analysis...");
  await sleep(900);  // Simulated processing time

  const cfg = State.config;

  // 1. Filter profiles by threshold
  State.filtered = cfg.risk_profiles.filter(p => p.risk_score >= State.threshold);
  applyProfileFilter();

  // 2. Render model metrics
  renderModelMetrics();

  // 3. Charts
  renderTrendChart();
  renderFeatureImportanceChart();
  renderMetricsComparisonChart();

  // 4. Insights + Recommendations
  renderInsights();
  renderRecommendations();

  // 5. Show all panels
  showPanels(["card-profiles","card-trend","card-fi","card-metrics",
              "card-recs","model-card","insights-card"]);

  State.analysisRun = true;
  hideSpinner();
  document.getElementById("status-badge").textContent = "● Analysis Complete";
}

// ── Refresh ────────────────────────────────────────────────────────
async function refreshData() {
  showSpinner("Refreshing data...");
  await sleep(600);
  await loadConfig();
  populateDashboardCards();
  if (State.analysisRun) await runAnalysis();
  else hideSpinner();
}

// ── Filter logic ───────────────────────────────────────────────────
function applyFilter() {
  const filterLabel = document.getElementById("profile-select").value;
  State.filtered = State.config.risk_profiles.filter(p => {
    const aboveThreshold = p.risk_score >= State.threshold;
    const matchesLabel = filterLabel === "all" || p.risk_label === filterLabel;
    return aboveThreshold && matchesLabel;
  });
  renderProfiles(State.filtered);
}

function applyProfileFilter() {
  const filterLabel = document.getElementById("profile-select").value;
  State.filtered = State.config.risk_profiles.filter(p => {
    const aboveThreshold = p.risk_score >= State.threshold;
    const matchesLabel = filterLabel === "all" || p.risk_label === filterLabel;
    return aboveThreshold && matchesLabel;
  });
  renderProfiles(State.filtered);
}

// ── Render Risk Profile Rows ───────────────────────────────────────
function renderProfiles(profiles) {
  const container = document.getElementById("profiles-container");
  const countEl   = document.getElementById("profile-count");

  countEl.textContent = `${profiles.length} result${profiles.length !== 1 ? "s" : ""}`;

  if (profiles.length === 0) {
    container.innerHTML = `<div style="color:var(--text-muted);font-size:0.85rem;padding:1rem 0">
      No profiles match current threshold / filter.</div>`;
    return;
  }

  container.innerHTML = profiles.map(p => {
    const cls   = p.risk_label.toLowerCase();
    const pct   = Math.round(p.risk_score * 100);
    const color = cls === "high" ? "#ef4444" : cls === "medium" ? "#f59e0b" : "#22c55e";
    const circ  = buildScoreRing(pct, color);
    const alerts = p.alerts.length
      ? `<div class="profile-alerts">${p.alerts.map(a => `<span class="alert-chip">⚠ ${a}</span>`).join("")}</div>`
      : "";

    return `
    <div class="profile-row ${cls}">
      ${circ}
      <div>
        <div class="profile-name">${p.entity}</div>
        <div class="profile-id">${p.id}</div>
        ${alerts}
        <div class="profile-rec">→ ${p.recommendation}</div>
      </div>
      <span class="profile-badge ${cls}">${p.risk_label}</span>
    </div>`;
  }).join("");
}

function buildScoreRing(pct, color) {
  const r = 22, cx = 28, cy = 28;
  const circ = 2 * Math.PI * r;
  const dash = (pct / 100) * circ;
  return `
  <div class="score-ring">
    <svg width="56" height="56" viewBox="0 0 56 56">
      <circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="#30363d" stroke-width="4"/>
      <circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="${color}" stroke-width="4"
        stroke-dasharray="${dash.toFixed(1)} ${circ.toFixed(1)}" stroke-linecap="round"/>
    </svg>
    <span class="score-text">${pct}%</span>
  </div>`;
}

// ── Render Model Metrics Mini Table ───────────────────────────────
function renderModelMetrics() {
  const body = document.getElementById("model-card-body");
  const metrics = State.config.model_metrics;
  const best    = State.config.best_model;

  const rows = Object.entries(metrics).map(([name, m]) => {
    const isBest = name === best;
    const label  = name.replace(/_/g," ").replace(/\b\w/g, c => c.toUpperCase());
    return `
    <tr class="${isBest ? "best" : ""}">
      <td>${isBest ? "★ " : ""}${label}</td>
      <td>${(m.accuracy  * 100).toFixed(1)}%</td>
      <td>
        <div class="bar-cell">
          <div class="bar-track"><div class="bar-fill" style="width:${(m.f1_score*100).toFixed(1)}%"></div></div>
          <span>${(m.f1_score*100).toFixed(1)}%</span>
        </div>
      </td>
    </tr>`;
  }).join("");

  body.innerHTML = `
    <table class="mini-table">
      <thead><tr><th>Model</th><th>Accuracy</th><th>F1 Score</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

// ── Render Trend Chart ─────────────────────────────────────────────
function renderTrendChart() {
  const td = State.config.trend_data;
  const traces = [
    { x: td.labels, y: td.high_risk,   name: "High Risk",   mode: "lines+markers", line: { color: "#ef4444", width: 2 }, marker: { size: 5 } },
    { x: td.labels, y: td.medium_risk, name: "Medium Risk", mode: "lines+markers", line: { color: "#f59e0b", width: 2 }, marker: { size: 5 } },
    { x: td.labels, y: td.low_risk,    name: "Low Risk",    mode: "lines+markers", line: { color: "#22c55e", width: 2 }, marker: { size: 5 } },
  ];
  Plotly.newPlot("chart-trend", traces, {
    ...PLOT_LAYOUT,
    yaxis: { ...PLOT_LAYOUT.yaxis, title: "Count" },
  }, PLOT_CONFIG);
}

// ── Render Feature Importance Chart ───────────────────────────────
function renderFeatureImportanceChart() {
  const fi = State.config.feature_importance;
  const sorted = Object.entries(fi).sort((a, b) => a[1] - b[1]);
  const labels = sorted.map(e => e[0]);
  const values = sorted.map(e => e[1]);

  Plotly.newPlot("chart-fi", [{
    type: "bar",
    orientation: "h",
    x: values,
    y: labels,
    marker: {
      color: values.map((v, i) =>
        `rgba(59,130,246,${0.4 + 0.6 * (i / values.length)})`
      ),
    },
  }], {
    ...PLOT_LAYOUT,
    xaxis: { ...PLOT_LAYOUT.xaxis, title: "Importance" },
    margin: { ...PLOT_LAYOUT.margin, l: 130 },
  }, PLOT_CONFIG);
}

// ── Render Metrics Comparison Chart ───────────────────────────────
function renderMetricsComparisonChart() {
  const metrics = State.config.model_metrics;
  const metricKeys  = ["accuracy", "precision", "recall", "f1_score"];
  const metricLabels = ["Accuracy", "Precision", "Recall", "F1 Score"];
  const colors = ["#3b82f6", "#22c55e"];

  const traces = Object.entries(metrics).map(([name, m], i) => ({
    type:  "bar",
    name:  name.replace(/_/g," ").replace(/\b\w/g, c => c.toUpperCase()),
    x:     metricLabels,
    y:     metricKeys.map(k => parseFloat((m[k] * 100).toFixed(2))),
    marker: { color: colors[i % colors.length], opacity: 0.85 },
  }));

  Plotly.newPlot("chart-metrics", traces, {
    ...PLOT_LAYOUT,
    barmode: "group",
    yaxis: { ...PLOT_LAYOUT.yaxis, title: "Score (%)", range: [0, 105] },
  }, PLOT_CONFIG);
}

// ── Render Insights ───────────────────────────────────────────────
function renderInsights() {
  document.getElementById("insights-body").innerHTML =
    State.config.insights
      .map(i => `<div class="insight-item">💡 ${i}</div>`)
      .join("");
}

// ── Render Recommendations ─────────────────────────────────────────
function renderRecommendations() {
  document.getElementById("recs-body").innerHTML =
    State.config.recommendations
      .map(r => `<div class="rec-item">✓ ${r}</div>`)
      .join("");
}

// ── Utility Helpers ────────────────────────────────────────────────
function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

function showPanels(ids) {
  ids.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.style.display = "";
  });
}

function showSpinner(msg = "Processing...") {
  document.getElementById("spinner").style.display = "flex";
  document.getElementById("spinner-msg").textContent = msg;
}

function hideSpinner() {
  document.getElementById("spinner").style.display = "none";
}

function showError(msg) {
  hideSpinner();
  alert(`Guardian ML Error:\n${msg}`);
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}
