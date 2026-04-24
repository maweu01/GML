/**
 * Guardian ML — Frontend App (Mode A)
 * Connects to FastAPI backend at /api/v1
 */

const API_BASE = "http://localhost:8000/api/v1";

// ── State ──────────────────────────────────────────────────────────
const state = {
  step: 0,        // 0=none, 1=uploaded, 2=processed, 3=trained
  columns: [],
  features: [],
};

// ── DOM refs ───────────────────────────────────────────────────────
const els = {
  fileInput:      () => document.getElementById("file-input"),
  btnUpload:      () => document.getElementById("btn-upload"),
  btnProcess:     () => document.getElementById("btn-process"),
  btnTrain:       () => document.getElementById("btn-train"),
  btnPredict:     () => document.getElementById("btn-predict"),
  btnViz:         () => document.getElementById("btn-viz"),
  btnClearLog:    () => document.getElementById("btn-clear-log"),
  targetCol:      () => document.getElementById("target-col"),
  predictInput:   () => document.getElementById("predict-input"),
  modelSelect:    () => document.getElementById("model-select"),
  dropZone:       () => document.getElementById("drop-zone"),
  logBox:         () => document.getElementById("log-box"),
  statusBadge:    () => document.getElementById("status-badge"),
  spinner:        () => document.getElementById("spinner"),
  spinnerMsg:     () => document.getElementById("spinner-msg"),
  dcStatusVal:    () => document.getElementById("dc-status-val"),
  dcSamplesVal:   () => document.getElementById("dc-samples-val"),
  dcFeaturesVal:  () => document.getElementById("dc-features-val"),
  dcRiskVal:      () => document.getElementById("dc-risk-val"),
};

// ── Logging ────────────────────────────────────────────────────────
function log(msg, type = "info") {
  const box = els.logBox();
  const now = new Date().toLocaleTimeString();
  const entry = document.createElement("div");
  entry.className = `log-entry log-${type}`;
  entry.innerHTML = `<span class="log-time">${now}</span><span class="log-msg">${msg}</span>`;
  box.appendChild(entry);
  box.scrollTop = box.scrollHeight;
}

// ── Spinner ────────────────────────────────────────────────────────
function showSpinner(msg = "Processing...") {
  els.spinner().style.display = "flex";
  els.spinnerMsg().textContent = msg;
}
function hideSpinner() {
  els.spinner().style.display = "none";
}

// ── Step tracker ───────────────────────────────────────────────────
function setStep(n) {
  state.step = n;
  for (let i = 1; i <= 5; i++) {
    const el = document.getElementById(`step-${["upload","process","train","evaluate","predict"][i-1]}`);
    if (!el) continue;
    el.classList.remove("active", "done");
    if (i < n) el.classList.add("done");
    else if (i === n) el.classList.add("active");
  }
}

// ── API helper ─────────────────────────────────────────────────────
async function apiFetch(endpoint, opts = {}) {
  const res = await fetch(`${API_BASE}${endpoint}`, opts);
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
  return data;
}

// ── Health check ───────────────────────────────────────────────────
async function checkHealth() {
  try {
    await fetch("http://localhost:8000/health");
    const badge = els.statusBadge();
    badge.textContent = "● Backend Online";
    badge.className = "badge badge--live";
    els.dcStatusVal().textContent = "Online";
    log("Backend connected at localhost:8000", "success");
  } catch {
    const badge = els.statusBadge();
    badge.textContent = "● Backend Offline";
    badge.className = "badge badge--offline";
    els.dcStatusVal().textContent = "Offline";
    log("Backend not reachable. Start the FastAPI server.", "error");
  }
}

// ── File selection ─────────────────────────────────────────────────
function initFileInput() {
  const input = els.fileInput();
  const drop = els.dropZone();

  input.addEventListener("change", () => {
    if (input.files.length) {
      drop.querySelector(".drop-text").innerHTML =
        `<strong>${input.files[0].name}</strong><br/><small>${(input.files[0].size/1024).toFixed(1)} KB</small>`;
      els.btnUpload().disabled = false;
      log(`File selected: ${input.files[0].name}`);
    }
  });

  // Drag & Drop
  drop.addEventListener("dragover", e => { e.preventDefault(); drop.classList.add("dragover"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("dragover"));
  drop.addEventListener("drop", e => {
    e.preventDefault();
    drop.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file) {
      const dt = new DataTransfer();
      dt.items.add(file);
      input.files = dt.files;
      input.dispatchEvent(new Event("change"));
    }
  });
}

// ── Step 1: Upload ─────────────────────────────────────────────────
document.getElementById("btn-upload").addEventListener("click", async () => {
  const file = els.fileInput().files[0];
  if (!file) return;

  const fd = new FormData();
  fd.append("file", file);

  showSpinner("Uploading dataset...");
  log("Uploading file...", "info");

  try {
    const data = await apiFetch("/upload", { method: "POST", body: fd });
    hideSpinner();
    state.columns = data.columns;
    els.dcSamplesVal().textContent = data.n_samples.toLocaleString();
    els.dcFeaturesVal().textContent = data.n_features;
    log(`Uploaded: ${data.filename} — ${data.n_samples} samples, ${data.n_features} features`, "success");
    log(`Columns: ${data.columns.slice(0, 8).join(", ")}${data.columns.length > 8 ? "..." : ""}`, "info");
    els.btnProcess().disabled = false;
    setStep(2);
  } catch (e) {
    hideSpinner();
    log(`Upload failed: ${e.message}`, "error");
  }
});

// ── Step 2: Process ────────────────────────────────────────────────
document.getElementById("btn-process").addEventListener("click", async () => {
  const target = els.targetCol().value.trim() || null;
  showSpinner("Preprocessing data...");
  log(`Preprocessing... target_col="${target || "none"}"`, "info");

  try {
    const data = await apiFetch("/process", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ target_col: target }),
    });
    hideSpinner();
    state.features = data.feature_names;
    log(`Preprocessed: ${data.train_samples} train / ${data.test_samples} test samples`, "success");
    log(`Features: ${data.n_features} — supervised: ${data.supervised}`, "info");
    els.btnTrain().disabled = false;
    setStep(3);

    // Auto-fill predict textarea with feature template
    if (state.features.length) {
      const template = {};
      state.features.slice(0, 6).forEach(f => template[f] = 0);
      els.predictInput().value = JSON.stringify(template, null, 2);
    }
  } catch (e) {
    hideSpinner();
    log(`Preprocessing failed: ${e.message}`, "error");
  }
});

// ── Step 3: Train ──────────────────────────────────────────────────
document.getElementById("btn-train").addEventListener("click", async () => {
  showSpinner("Training ML models...");
  log("Training Logistic Regression + Random Forest...", "info");

  try {
    const data = await apiFetch("/train", { method: "POST" });
    hideSpinner();
    log("Training complete!", "success");

    // Show metrics
    renderMetricsTable(data.results);

    // Fetch evaluation
    const report = await apiFetch("/evaluate");
    renderInsights(report);

    els.btnPredict().disabled = false;
    document.getElementById("card-metrics").style.display = "";
    document.getElementById("card-viz").style.display = "";
    document.getElementById("card-insights").style.display = "";
    setStep(4);

    // Log per model
    for (const [name, res] of Object.entries(data.results)) {
      const m = res.metrics;
      if (m.accuracy !== undefined) {
        log(`[${name}] Acc=${m.accuracy} | P=${m.precision} | R=${m.recall} | F1=${m.f1_score}`, "success");
      }
    }
    if (report.best_model) {
      log(`Best model: ${report.best_model} (F1=${report.best_f1_score})`, "info");
    }
  } catch (e) {
    hideSpinner();
    log(`Training failed: ${e.message}`, "error");
  }
});

// ── Step 4: Predict ────────────────────────────────────────────────
document.getElementById("btn-predict").addEventListener("click", async () => {
  let inputData;
  try {
    inputData = JSON.parse(els.predictInput().value);
  } catch {
    log("Invalid JSON in prediction input.", "error");
    return;
  }

  const model = els.modelSelect().value;
  showSpinner("Running prediction...");
  log(`Predicting with model: ${model}`, "info");

  try {
    const result = await apiFetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data: inputData, model_name: model }),
    });
    hideSpinner();
    renderPrediction(result);
    setStep(5);
    log(`Risk: ${result.risk_label} | Score: ${result.risk_score} | Prediction: ${result.prediction}`, "success");
  } catch (e) {
    hideSpinner();
    log(`Prediction failed: ${e.message}`, "error");
  }
});

// ── Load Charts ────────────────────────────────────────────────────
document.getElementById("btn-viz").addEventListener("click", async () => {
  showSpinner("Generating charts...");
  try {
    const data = await apiFetch("/visualize");
    hideSpinner();
    if (data.charts.metrics_comparison) {
      document.getElementById("viz-metrics").innerHTML =
        `<img src="data:image/png;base64,${data.charts.metrics_comparison}" alt="Metrics Comparison"/>`;
    }
    if (data.charts.feature_importance) {
      document.getElementById("viz-fi").innerHTML =
        `<img src="data:image/png;base64,${data.charts.feature_importance}" alt="Feature Importance"/>`;
    }
    log("Charts loaded.", "success");
  } catch (e) {
    hideSpinner();
    log(`Visualization error: ${e.message}`, "error");
  }
});

// ── Clear log ──────────────────────────────────────────────────────
document.getElementById("btn-clear-log").addEventListener("click", () => {
  els.logBox().innerHTML = "";
});

// ── Render helpers ─────────────────────────────────────────────────
function renderMetricsTable(results) {
  const container = document.getElementById("metrics-table-container");
  const rows = Object.entries(results).map(([name, res]) => {
    const m = res.metrics;
    return `<tr class="${name === Object.keys(results)[0] ? "best-model" : ""}">
      <td>${name.replace(/_/g," ").replace(/\b\w/g,c=>c.toUpperCase())}</td>
      <td>${(m.accuracy*100||0).toFixed(1)}%</td>
      <td>${(m.precision*100||0).toFixed(1)}%</td>
      <td>${(m.recall*100||0).toFixed(1)}%</td>
      <td>${(m.f1_score*100||0).toFixed(1)}%</td>
      <td>${res.n_train}</td>
    </tr>`;
  }).join("");

  container.innerHTML = `
    <table class="metrics-table">
      <thead><tr>
        <th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1 Score</th><th>Train N</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

function renderPrediction(result) {
  const label = result.risk_label.toLowerCase();
  const card = document.getElementById("card-result");
  card.style.display = "";
  document.getElementById("prediction-result").innerHTML = `
    <div class="risk-result ${label}">
      <div class="risk-label">${result.risk_label} RISK</div>
      <div class="risk-score">Score: ${(result.risk_score * 100).toFixed(1)}% &nbsp;|&nbsp; Model: ${result.model_used.replace(/_/g," ")}</div>
    </div>`;
  els.dcRiskVal().textContent = `${(result.risk_score * 100).toFixed(0)}%`;
}

function renderInsights(report) {
  const body = document.getElementById("insights-body");
  const insights = (report.risk_insights || []).map(i =>
    `<div class="insight-item">💡 ${i}</div>`).join("");
  const recs = (report.recommendations || []).map(r =>
    `<div class="rec-item">✓ ${r}</div>`).join("");
  body.innerHTML = insights + (recs ? `<hr style="border-color:var(--border);margin:0.5rem 0"/>` + recs : "");
}

// ── Init ───────────────────────────────────────────────────────────
initFileInput();
checkHealth();
setStep(1);
log("Guardian ML frontend initialized.", "info");
