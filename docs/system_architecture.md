# Guardian ML — System Architecture

## Overview

Guardian ML is a dual-mode intelligent risk and decision support platform. It operates as a full-stack ML system with a Python/FastAPI backend (Mode A) and as a fully self-contained static web application suitable for GitHub Pages deployment (Mode B).

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                     GUARDIAN ML                         │
│              Risk & Decision Support Platform           │
└─────────────────────────────────────────────────────────┘

MODE A — Full Stack                MODE B — Static (GitHub Pages)
────────────────────               ──────────────────────────────
                                   
  Browser (frontend/)               Browser (github-pages/)
      │                                   │
      │  HTTP/REST                        │  fetch()
      ▼                                   ▼
  FastAPI Backend                   data/config.json
  (backend/main.py)                 (local simulation)
      │
  ┌───┴────────────────────┐
  │    API Routes Layer    │
  │  /upload /process      │
  │  /train /predict       │
  │  /visualize /evaluate  │
  └───┬────────────────────┘
      │
  ┌───┴────────────────────┐
  │    Pipeline Core       │
  │  GuardianPipeline      │
  └───┬────────────────────┘
      │
  ┌───┴──────────┬──────────────┐
  │              │              │
  ▼              ▼              ▼
DataPreprocessor  ModelTrainer  ModelEvaluator
(core/)           (models/)     (models/)
      │              │
      ▼              ▼
  cleaned X,y    LR + RF models
                 (joblib saved)
```

---

## Component Breakdown

### Backend (Mode A)

| Component | File | Responsibility |
|-----------|------|----------------|
| Entry Point | `main.py` | FastAPI app, CORS, lifespan, config loading |
| Routes | `api/routes.py` | All HTTP endpoints, request validation |
| Pipeline | `core/pipeline.py` | Orchestrates all ML stages |
| Preprocessor | `core/preprocessor.py` | Data loading, cleaning, encoding, scaling |
| Trainer | `models/trainer.py` | LR + RF training, joblib persistence |
| Evaluator | `models/evaluator.py` | Metrics, insights, recommendations |
| Logger | `utils/logger.py` | Structured console + file logging |
| Config | `config.yaml` | All tuneable parameters |

### Frontend Mode A

| File | Responsibility |
|------|----------------|
| `frontend/index.html` | 5-step pipeline UI, dashboard cards |
| `frontend/style.css` | Dark engineering theme |
| `frontend/app.js` | API calls, state management, chart rendering |

### Frontend Mode B (Static)

| File | Responsibility |
|------|----------------|
| `github-pages/index.html` | Full dashboard, base tag for GH Pages |
| `github-pages/style.css` | Identical dark theme, standalone |
| `github-pages/app.js` | Simulation engine, Plotly charts, filters |
| `github-pages/data/config.json` | All data: metrics, profiles, trends |

---

## Data Flow

### Mode A (Full Pipeline)

```
User uploads CSV/JSON
    │
    ▼
/upload → DataPreprocessor.load()
    │       → validate schema
    │       → compute feature summary
    ▼
/process → DataPreprocessor.fit_transform()
    │        → clean nulls
    │        → encode categoricals (LabelEncoder)
    │        → scale numerics (StandardScaler)
    │        → train/test split
    ▼
/train → ModelTrainer.train_all()
    │     → LogisticRegression.fit()
    │     → RandomForestClassifier.fit()
    │     → compute accuracy/precision/recall/F1
    │     → save models via joblib
    ▼
/evaluate → ModelEvaluator.generate_report()
    │         → model comparison table
    │         → insights
    │         → recommendations
    ▼
/predict → GuardianPipeline.predict()
    │        → transform new input
    │        → model.predict() + predict_proba()
    │        → map score → risk label (LOW/MEDIUM/HIGH)
    ▼
/visualize → matplotlib bar charts
              → base64 PNG returned to frontend
```

### Mode B (Static Simulation)

```
Browser loads index.html
    │
    ▼
app.js: fetch('data/config.json')
    │
    ▼
Populate dashboard cards from system_stats
    │
    ▼
User clicks "Run Analysis"
    │
    ├── Filter risk_profiles by threshold + label
    ├── Render SVG score rings per profile
    ├── Build Plotly trend chart (risk over 12 months)
    ├── Build Plotly feature importance bar chart
    ├── Build Plotly model comparison bar chart
    ├── Render insights + recommendations from JSON
    └── Render model metrics mini-table
```

---

## API Reference

### Base URL
`http://localhost:8000/api/v1`

### Endpoints

#### `POST /upload`
- **Input**: multipart/form-data with `file` (.csv or .json)
- **Output**: `{ status, filename, n_samples, n_features, columns, preview }`

#### `POST /process`
- **Input**: `{ "target_col": "label" }` (optional)
- **Output**: `{ status, train_samples, test_samples, n_features, feature_names, supervised }`

#### `POST /train`
- **Input**: none (uses processed state)
- **Output**: `{ status, results: { model: { metrics, feature_importance, n_train } } }`

#### `POST /predict`
- **Input**: `{ "data": { "feat1": val }, "model_name": "random_forest" }`
- **Output**: `{ prediction, risk_score, risk_label, probabilities, model_used }`

#### `GET /evaluate`
- **Output**: `{ model_comparison, best_model, feature_importance, risk_insights, recommendations }`

#### `GET /visualize`
- **Output**: `{ charts: { metrics_comparison: "<base64>", feature_importance: "<base64>" } }`

#### `GET /status`
- **Output**: `{ data_loaded, preprocessed, trained, models_available }`

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Backend framework | FastAPI 0.111 |
| ML | scikit-learn 1.5 (LR, RF) |
| Data handling | pandas, numpy |
| Model persistence | joblib |
| Visualization (backend) | matplotlib, seaborn |
| Visualization (frontend) | Plotly.js 2.32 |
| Config | PyYAML |
| Testing | pytest, httpx |
| Frontend | Vanilla HTML5 / CSS3 / ES2022 |
| Static deployment | GitHub Pages |

---

## Security Considerations

- CORS is set to `*` for development. Restrict in production to known origins.
- File uploads are validated by extension (.csv, .json only).
- Model files are stored locally; use object storage (S3/GCS) in production.
- API has no authentication layer — add OAuth2/JWT for production.
