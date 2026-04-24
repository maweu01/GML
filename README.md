# ⚡ Guardian ML
### Intelligent Risk & Decision Support Platform

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?style=flat-square)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange?style=flat-square)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)](#)

Guardian ML is a production-ready, dual-mode machine learning platform for automated risk detection, pattern analysis, and decision support from structured data.

---

## Overview

| Mode | Description | Requirements |
|------|-------------|--------------|
| **Mode A** — Full Stack | FastAPI backend + ML pipeline + connected UI | Python 3.9+ |
| **Mode B** — Static | Self-contained GitHub Pages deployment | None (browser only) |

Both modes share the same UI design language and deliver identical analytical outputs — Mode B simulates the ML results using a pre-configured JSON dataset.

---

## Scientific Motivation

Risk stratification from tabular data is a fundamental challenge across finance, healthcare, operations, and cybersecurity. Traditional rule-based systems fail to capture non-linear feature interactions and require constant manual tuning. Guardian ML applies supervised machine learning (Logistic Regression + Random Forest) with rigorous preprocessing, calibrated probability outputs, and interpretable feature importance — enabling data-driven risk classification with quantifiable performance guarantees.

---

## Features

- **Full ML Pipeline**: Upload → Preprocess → Train → Evaluate → Predict
- **Dual Model Training**: Logistic Regression + Random Forest with comparative evaluation
- **Risk Scoring**: Calibrated probability → LOW / MEDIUM / HIGH risk labels
- **Metrics Dashboard**: Accuracy, Precision, Recall, F1 Score per model
- **Visualizations**: Performance comparison charts, feature importance (Plotly.js + Matplotlib)
- **Insights Engine**: Automated narrative insights and deployment recommendations
- **Static Mode**: Full dashboard simulation deployable with zero infrastructure
- **Reproducible**: Fixed random seeds, deterministic pipeline

---

## Repository Structure

```
Guardian-ML/
├── backend/                  # Mode A — Python/FastAPI ML system
│   ├── main.py               # Application entry point
│   ├── requirements.txt      # Python dependencies
│   ├── config.yaml           # All tunable parameters
│   ├── core/
│   │   ├── pipeline.py       # Pipeline orchestrator
│   │   └── preprocessor.py   # Data cleaning, encoding, scaling
│   ├── models/
│   │   ├── trainer.py        # LR + RF training, model persistence
│   │   └── evaluator.py      # Metrics, insights, recommendations
│   ├── api/
│   │   └── routes.py         # FastAPI endpoints
│   ├── utils/
│   │   └── logger.py         # Structured logging
│   └── data/                 # Runtime: uploads, models, plots
│
├── frontend/                 # Mode A frontend
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── github-pages/             # Mode B — static deployment
│   ├── index.html            # <base> tag for GH Pages
│   ├── style.css
│   ├── app.js
│   ├── data/
│   │   └── config.json       # Simulation data
│   └── README.md
│
├── docs/
│   ├── system_architecture.md
│   └── methodology.md
│
├── tests/
│   └── test_api.py           # pytest suite for all endpoints
│
├── .gitignore
├── README.md
└── run.sh                    # Unified launcher
```

---

## Installation (Mode A)

### Prerequisites

- Python 3.9 or higher
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/Guardian-ML.git
cd Guardian-ML

# 2. Install dependencies
./run.sh install
# or manually:
cd backend && pip install -r requirements.txt

# 3. Start the backend
./run.sh backend
# Backend available at: http://localhost:8000
# API docs at:          http://localhost:8000/docs

# 4. Serve the frontend (new terminal)
./run.sh frontend
# UI available at: http://localhost:3000
```

---

## Usage Guide

### Mode A — Full Pipeline

1. **Upload**: Drag & drop a CSV or JSON file onto the upload zone.
2. **Process**: Enter your target column name (e.g., `label`, `class`, `risk`). Leave blank for unsupervised mode.
3. **Train**: Click "Train Models" — both Logistic Regression and Random Forest will be trained and evaluated.
4. **Evaluate**: Metrics, feature importance, and automated insights are displayed.
5. **Predict**: Enter a JSON object with feature values and click "Run Prediction" to get a risk score.

### Mode B — Static Dashboard

1. Open `github-pages/index.html` in a browser **or** visit the deployed GitHub Pages URL.
2. Click **▶ Run Analysis** to simulate the full ML pipeline.
3. Adjust the risk threshold slider and profile filter to explore results.
4. All charts, tables, and insights render from `data/config.json`.

---

## API Documentation

Base URL: `http://localhost:8000/api/v1`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload` | Upload CSV or JSON dataset |
| `POST` | `/process` | Preprocess data (with optional target column) |
| `POST` | `/train` | Train all configured ML models |
| `POST` | `/predict` | Get risk prediction for a data point |
| `GET`  | `/evaluate` | Full evaluation report with insights |
| `GET`  | `/visualize` | Get base64-encoded chart images |
| `GET`  | `/status` | Pipeline state (loaded/preprocessed/trained) |
| `GET`  | `/health` | Health check |

**Interactive docs** (Swagger UI): `http://localhost:8000/docs`  
**Alternative docs** (ReDoc): `http://localhost:8000/redoc`

### Example: Predict Request

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": {"feature_a": 2.5, "feature_b": 1.2, "feature_c": 0.4},
    "model_name": "random_forest"
  }'
```

Response:
```json
{
  "prediction": 1,
  "risk_score": 0.7840,
  "risk_label": "HIGH",
  "probabilities": [0.216, 0.784],
  "model_used": "random_forest"
}
```

---

## Example Workflows

### Workflow 1: Financial Transaction Risk

```
1. Prepare CSV with columns: amount, frequency, velocity, merchant_type, label
2. Upload via /upload
3. POST /process with target_col="label"
4. POST /train
5. GET /evaluate — review F1 score and feature importance
6. POST /predict for each new transaction
```

### Workflow 2: Operational Anomaly Detection

```
1. Export sensor readings as CSV (no label column)
2. Upload → Process (no target_col)
3. Train runs in unsupervised mode
4. Use /predict to score new readings
```

### Workflow 3: Static Dashboard for Reporting

```
1. Edit github-pages/data/config.json with your metrics
2. Open index.html in browser
3. Click "Run Analysis"
4. Screenshot or share the URL for stakeholder reporting
```

---

## Running Tests

```bash
# From the project root
./run.sh test

# Or directly:
cd backend
python -m pytest ../tests/test_api.py -v
```

Tests cover: health checks, upload, preprocessing, training, prediction, evaluation, visualization, and PNG chart validation.

---

## Deployment Guide

### GitHub Pages (Mode B)

1. Edit `github-pages/index.html` line 9:
   ```html
   <base href="/YOUR-REPO-NAME/" />
   ```
2. Push the `github-pages/` folder (or whole repo) to GitHub.
3. Enable Pages: `Settings → Pages → Source: main → /github-pages`
4. Access at: `https://YOUR_USERNAME.github.io/YOUR-REPO-NAME/`

### Docker (Mode A)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY backend/ .
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["python", "main.py"]
```

```bash
docker build -t guardian-ml .
docker run -p 8000:8000 guardian-ml
```

### Production Considerations

- Replace in-memory pipeline state with Redis or a database
- Add authentication (OAuth2/JWT) to API endpoints
- Use object storage (S3/GCS) for model and data persistence
- Deploy behind a reverse proxy (nginx) with TLS
- Add Prometheus metrics endpoint for monitoring

---

## Configuration Reference

All tunable parameters are in `backend/config.yaml`:

```yaml
ml:
  random_seed: 42       # Reproducibility seed
  test_size: 0.2        # 20% held-out test set
  models:               # Which models to train
    - logistic_regression
    - random_forest

risk:
  thresholds:
    low:    0.33        # Score ≤ 0.33 → LOW
    medium: 0.66        # Score ≤ 0.66 → MEDIUM
    high:   1.0         # Score > 0.66 → HIGH
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Architecture & Methodology

- [System Architecture](docs/system_architecture.md)
- [ML Methodology](docs/methodology.md)
