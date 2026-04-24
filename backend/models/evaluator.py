"""
Guardian ML — Model Evaluator
Generates evaluation reports, comparison tables, and risk insights.
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from utils.logger import get_logger

logger = get_logger("evaluator")


class ModelEvaluator:
    def __init__(self, config: dict):
        self.config = config

    def generate_report(self, results: dict) -> dict:
        """
        Generate a structured evaluation report from training results.
        """
        if not results:
            return {"error": "No results to evaluate."}

        comparison = {}
        best_model = None
        best_f1 = -1.0

        for model_name, data in results.items():
            metrics = data.get("metrics", {})
            comparison[model_name] = metrics
            f1 = metrics.get("f1_score", -1)
            if f1 > best_f1:
                best_f1 = f1
                best_model = model_name

        report = {
            "model_comparison": comparison,
            "best_model": best_model,
            "best_f1_score": best_f1,
            "feature_importance": {
                model: data.get("feature_importance", {})
                for model, data in results.items()
            },
            "risk_insights": self._generate_insights(comparison, best_model),
            "recommendations": self._generate_recommendations(comparison, best_model),
        }

        logger.info(f"Evaluation report generated. Best model: {best_model} (F1={best_f1})")
        return report

    def _generate_insights(self, comparison: dict, best_model: Optional[str]) -> list[str]:
        insights = []
        if not comparison:
            return insights

        if best_model:
            metrics = comparison[best_model]
            f1 = metrics.get("f1_score", 0)
            acc = metrics.get("accuracy", 0)

            if f1 >= 0.90:
                insights.append(f"High predictive performance — {best_model} achieves F1={f1:.2%}.")
            elif f1 >= 0.75:
                insights.append(f"Moderate predictive performance — {best_model} achieves F1={f1:.2%}. Consider feature engineering.")
            else:
                insights.append(f"Low predictive performance (F1={f1:.2%}). Review data quality and class balance.")

            if acc >= 0.85:
                insights.append("Model accuracy is satisfactory for production deployment.")
            else:
                insights.append("Accuracy below 85% — evaluate class imbalance and data volume.")

        # Cross-model comparison
        if len(comparison) > 1:
            f1_scores = {m: v.get("f1_score", 0) for m, v in comparison.items()}
            delta = max(f1_scores.values()) - min(f1_scores.values())
            if delta > 0.1:
                insights.append(f"Significant performance gap between models (ΔF1={delta:.2%}). Best model strongly preferred.")
            else:
                insights.append(f"Models perform comparably (ΔF1={delta:.2%}). Ensemble is viable.")

        return insights

    def _generate_recommendations(self, comparison: dict, best_model: Optional[str]) -> list[str]:
        recs = []
        if best_model:
            recs.append(f"Deploy '{best_model}' as the primary risk scoring model.")
        recs.append("Monitor model drift on live data with a rolling evaluation window.")
        recs.append("Retrain monthly or when F1 drops below 0.75 on production data.")
        recs.append("Use SHAP values for individual prediction explainability.")
        return recs
