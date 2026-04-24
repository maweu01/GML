"""
Guardian ML — Core Pipeline Orchestrator
Coordinates preprocessing → training → evaluation → prediction.
"""

from __future__ import annotations

import os
import json
from typing import Optional

import numpy as np
import pandas as pd

from core.preprocessor import DataPreprocessor
from models.trainer import ModelTrainer
from models.evaluator import ModelEvaluator
from utils.logger import get_logger

logger = get_logger("pipeline")


class GuardianPipeline:
    """
    End-to-end ML pipeline for risk detection and decision support.

    Lifecycle:
        load_data() → preprocess() → train() → evaluate() → predict()
    """

    def __init__(self, config: dict):
        self.config = config
        self.preprocessor = DataPreprocessor(config)
        self.trainer = ModelTrainer(config)
        self.evaluator = ModelEvaluator(config)

        self._raw_df: Optional[pd.DataFrame] = None
        self._X_train: Optional[np.ndarray] = None
        self._X_test: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._y_test: Optional[np.ndarray] = None
        self._feature_names: list[str] = []
        self._results: dict = {}

    # ------------------------------------------------------------------
    # Stage 1 — Data Loading
    # ------------------------------------------------------------------

    def load_data(self, path: str) -> dict:
        self._raw_df = self.preprocessor.load(path)
        summary = self.preprocessor.feature_summary(self._raw_df)
        logger.info(f"Data loaded: {summary['n_samples']} samples, {summary['n_features']} features")
        return summary

    # ------------------------------------------------------------------
    # Stage 2 — Preprocessing
    # ------------------------------------------------------------------

    def preprocess(self, target_col: Optional[str] = None) -> dict:
        if self._raw_df is None:
            raise RuntimeError("Call load_data() first.")

        from sklearn.model_selection import train_test_split

        X, y = self.preprocessor.fit_transform(self._raw_df, target_col)
        self._feature_names = self.preprocessor.feature_names

        if y is not None:
            seed = self.config.get("ml", {}).get("random_seed", 42)
            test_size = self.config.get("ml", {}).get("test_size", 0.2)
            self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
                X, y, test_size=test_size, random_state=seed
            )
        else:
            # Unsupervised mode — no split needed
            self._X_train = X
            self._X_test = X
            self._y_train = None
            self._y_test = None

        return {
            "status": "preprocessed",
            "train_samples": int(self._X_train.shape[0]) if self._X_train is not None else 0,
            "test_samples": int(self._X_test.shape[0]) if self._X_test is not None else 0,
            "n_features": len(self._feature_names),
            "feature_names": self._feature_names,
            "supervised": y is not None,
        }

    # ------------------------------------------------------------------
    # Stage 3 — Training
    # ------------------------------------------------------------------

    def train(self) -> dict:
        if self._X_train is None:
            raise RuntimeError("Call preprocess() first.")

        results = self.trainer.train_all(
            self._X_train, self._y_train,
            self._X_test, self._y_test,
            self._feature_names,
        )
        self._results = results
        logger.info(f"Training complete. Models trained: {list(results.keys())}")
        return results

    # ------------------------------------------------------------------
    # Stage 4 — Evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> dict:
        if not self._results:
            raise RuntimeError("Call train() first.")
        report = self.evaluator.generate_report(self._results)
        return report

    # ------------------------------------------------------------------
    # Stage 5 — Prediction
    # ------------------------------------------------------------------

    def predict(self, data: dict | pd.DataFrame, model_name: str = "random_forest") -> dict:
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()

        X = self.preprocessor.transform(df)
        model = self.trainer.get_model(model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' not found. Train first.")

        pred = model.predict(X)
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X).tolist()

        risk_score = float(proba[0][1]) if proba and len(proba[0]) > 1 else float(pred[0])
        risk_label = self._score_to_label(risk_score)

        return {
            "prediction": int(pred[0]),
            "risk_score": round(risk_score, 4),
            "risk_label": risk_label,
            "probabilities": proba[0] if proba else None,
            "model_used": model_name,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _score_to_label(self, score: float) -> str:
        thresholds = self.config.get("risk", {}).get("thresholds", {"low": 0.33, "medium": 0.66, "high": 1.0})
        if score <= thresholds["low"]:
            return "LOW"
        elif score <= thresholds["medium"]:
            return "MEDIUM"
        return "HIGH"

    def get_status(self) -> dict:
        return {
            "data_loaded": self._raw_df is not None,
            "preprocessed": self._X_train is not None,
            "trained": bool(self._results),
            "models_available": list(self._results.keys()) if self._results else [],
        }
