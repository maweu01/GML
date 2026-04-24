"""
Guardian ML — Model Trainer
Trains Logistic Regression and Random Forest classifiers/regressors.
Persists models with joblib.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils.logger import get_logger

logger = get_logger("trainer")


class ModelTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.seed = config.get("ml", {}).get("random_seed", 42)
        self.model_dir = config.get("ml", {}).get("model_dir", "data/models")
        os.makedirs(self.model_dir, exist_ok=True)
        self._models: dict = {}

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def train_all(
        self,
        X_train: np.ndarray,
        y_train: Optional[np.ndarray],
        X_test: np.ndarray,
        y_test: Optional[np.ndarray],
        feature_names: list[str],
    ) -> dict:
        """Train all configured models. Returns dict of results per model."""
        supervised = y_train is not None and y_test is not None
        results = {}

        model_keys = self.config.get("ml", {}).get("models", ["logistic_regression", "random_forest"])

        for key in model_keys:
            if key == "logistic_regression":
                model = self._build_lr(supervised)
            elif key == "random_forest":
                model = self._build_rf(supervised)
            else:
                logger.warning(f"Unknown model key: {key}. Skipping.")
                continue

            if supervised:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                metrics = self._compute_metrics(y_test, y_pred)
            else:
                # Unsupervised fallback — fit on all data
                model.fit(X_train)
                metrics = {"note": "Unsupervised mode — no classification metrics"}

            self._models[key] = model
            self._save_model(model, key)

            fi = self._feature_importance(model, feature_names)

            results[key] = {
                "model": key,
                "metrics": metrics,
                "feature_importance": fi,
                "n_train": int(X_train.shape[0]),
                "n_test": int(X_test.shape[0]),
            }
            logger.info(f"Trained [{key}] — metrics: {metrics}")

        return results

    def get_model(self, name: str):
        """Retrieve a trained model by name."""
        return self._models.get(name)

    def load_model(self, name: str):
        """Load a persisted model from disk."""
        path = os.path.join(self.model_dir, f"{name}.joblib")
        if os.path.exists(path):
            self._models[name] = joblib.load(path)
            logger.info(f"Loaded model from disk: {path}")
            return self._models[name]
        return None

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _build_lr(self, supervised: bool) -> LogisticRegression:
        return LogisticRegression(
            max_iter=1000,
            random_state=self.seed,
            solver="lbfgs",
            multi_class="auto",
        )

    def _build_rf(self, supervised: bool) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.seed,
            n_jobs=-1,
        )

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        average = "weighted" if len(np.unique(y_true)) > 2 else "binary"
        return {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision": round(float(precision_score(y_true, y_pred, average=average, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, average=average, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_true, y_pred, average=average, zero_division=0)), 4),
        }

    def _feature_importance(self, model, feature_names: list[str]) -> dict:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_[0])
        else:
            return {}
        return dict(sorted(
            zip(feature_names, importances.tolist()),
            key=lambda x: x[1],
            reverse=True,
        ))

    def _save_model(self, model, name: str):
        path = os.path.join(self.model_dir, f"{name}.joblib")
        joblib.dump(model, path)
        logger.info(f"Model saved: {path}")
