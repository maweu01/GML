# Guardian ML — Methodology

## 1. Problem Framing

Guardian ML addresses the challenge of automated risk stratification from structured tabular data. The system ingests arbitrary CSV/JSON datasets, learns decision boundaries from labeled examples, and outputs calibrated risk scores in the range [0, 1] mapped to three actionable classes: LOW, MEDIUM, HIGH.

This is a supervised binary/multi-class classification problem, with unsupervised anomaly profiling available when labels are absent.

---

## 2. Data Preprocessing Pipeline

### 2.1 Data Ingestion

Input formats: CSV, JSON. The preprocessor infers schema automatically.

Validation steps:
- Drop rows/columns that are entirely null
- Detect numeric vs. categorical features via dtype inspection
- Log feature count, sample count, null rate per column

### 2.2 Missing Value Strategy

| Feature Type | Imputation Strategy |
|---|---|
| Numeric | Median imputation (robust to outliers) |
| Categorical | Mode imputation |

Median is preferred over mean for numeric features because it is resistant to extreme values, which are common in risk and anomaly datasets.

### 2.3 Categorical Encoding

`sklearn.preprocessing.LabelEncoder` is applied per categorical column. The encoder is fitted on training data only and applied to test/inference data to prevent data leakage.

Unknown categories at inference time are mapped to the first known class — a conservative fallback that avoids runtime errors while signaling possible distribution shift.

### 2.4 Feature Scaling

`sklearn.preprocessing.StandardScaler` normalizes all numeric features to zero mean and unit variance. This is essential for Logistic Regression, which is sensitive to feature scale, and beneficial for Random Forest in high-dimensional settings.

The scaler is fitted on training data only (fit_transform on X_train, transform on X_test and future inference data).

### 2.5 Train/Test Split

Default: 80% training, 20% test, stratified not enforced by default but configurable. Random seed is fixed (default: 42) for full reproducibility.

---

## 3. Machine Learning Models

### 3.1 Logistic Regression

**Rationale**: Provides a linear decision boundary, interpretable coefficients (via `coef_`), and calibrated probability estimates. Serves as a strong baseline and is deployable in latency-sensitive environments.

**Configuration**:
- Solver: `lbfgs` (efficient for small/medium datasets)
- Max iterations: 1000 (ensures convergence)
- Multi-class: `auto` (OvR for binary, softmax for multi-class)
- Random seed: 42

**Strengths**: Fast training, low memory, interpretable, well-calibrated probabilities  
**Weaknesses**: Cannot capture non-linear relationships

### 3.2 Random Forest Classifier

**Rationale**: An ensemble of decision trees using bootstrap aggregation (bagging). Robust to outliers, handles mixed feature types, provides native feature importance via mean impurity decrease (Gini importance).

**Configuration**:
- Estimators: 100 trees
- Max depth: 10 (prevents overfitting)
- Parallelism: n_jobs=-1 (uses all CPU cores)
- Random seed: 42

**Strengths**: Handles non-linearities, robust to noise, provides feature importance  
**Weaknesses**: Less interpretable per-prediction, slower inference than LR at scale

---

## 4. Evaluation Metrics

For each trained model, the following metrics are computed on held-out test data:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Accuracy | (TP + TN) / N | Overall correctness |
| Precision | TP / (TP + FP) | Of flagged risks, how many are real |
| Recall | TP / (TP + FN) | Of real risks, how many were caught |
| F1 Score | 2·P·R / (P+R) | Harmonic mean — primary selection criterion |

**Average strategy**: `binary` for two-class problems, `weighted` for multi-class (weights by class support to handle imbalance).

**Best model selection**: The model with the highest F1 score on the test set is designated the primary deployment model. F1 is preferred over accuracy because it penalizes both false positives and false negatives, which is critical in risk applications.

---

## 5. Risk Scoring

After prediction, the raw class probability (from `predict_proba`) is mapped to a risk label:

| Score Range | Label | Action |
|-------------|-------|--------|
| 0.00 – 0.33 | LOW | Standard monitoring |
| 0.34 – 0.66 | MEDIUM | Secondary review |
| 0.67 – 1.00 | HIGH | Immediate escalation |

Thresholds are configurable in `config.yaml` under `risk.thresholds`.

---

## 6. Feature Importance

For Random Forest, native Gini importance is extracted from `feature_importances_`. For Logistic Regression, the absolute value of coefficients (`|coef_|`) is used as a proxy for feature influence.

The top 10 features are surfaced in the visualization layer to aid interpretability.

---

## 7. Model Persistence

Trained models are serialized with `joblib` to `data/models/<model_name>.joblib`. This preserves both the fitted model weights and internal state (e.g., fitted label encoders in the pipeline are stored separately in the preprocessor object).

For production: models should be versioned with a timestamp and stored in object storage (S3/GCS) with a model registry.

---

## 8. Reproducibility

All randomness is controlled by a fixed seed (`random_seed: 42` in config.yaml), covering:
- Train/test split (scikit-learn)
- Logistic Regression initialization
- Random Forest tree construction and bootstrap sampling

This ensures that identical data produces identical results across runs.

---

## 9. Extensibility

The pipeline is designed for incremental enhancement:

| Enhancement | Implementation Path |
|---|---|
| Add XGBoost/LightGBM | Extend `ModelTrainer._build_*` methods |
| SHAP explainability | Add `shap` to requirements, call in evaluator |
| Class imbalance | Add `SMOTE` (imbalanced-learn) in preprocessor |
| Hyperparameter tuning | Wrap trainer in GridSearchCV/Optuna |
| Online learning | Replace batch fit with `partial_fit` (SGD) |
| Multi-output risk | Extend predict to return per-dimension scores |
