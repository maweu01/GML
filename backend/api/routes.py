"""
Guardian ML — API Routes
Endpoints: /upload /process /train /predict /visualize /status
"""

from __future__ import annotations

import io
import os
import base64
import tempfile
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.pipeline import GuardianPipeline
from utils.logger import get_logger

logger = get_logger("routes")
router = APIRouter()

# Global pipeline instance (stateful for demo; use DB/cache in production)
_pipeline: Optional[GuardianPipeline] = None
_last_upload_path: Optional[str] = None


def _get_pipeline(request: Request) -> GuardianPipeline:
    global _pipeline
    if _pipeline is None:
        config = getattr(request.app.state, "config", {})
        _pipeline = GuardianPipeline(config)
    return _pipeline


# ------------------------------------------------------------------
# /upload — Accept CSV or JSON file
# ------------------------------------------------------------------

class UploadResponse(BaseModel):
    status: str
    filename: str
    n_samples: int
    n_features: int
    columns: list[str]
    preview: list[dict]


@router.post("/upload", response_model=UploadResponse)
async def upload_data(request: Request, file: UploadFile = File(...)):
    global _last_upload_path, _pipeline

    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in (".csv", ".json"):
        raise HTTPException(status_code=400, detail="Only CSV and JSON files are supported.")

    config = getattr(request.app.state, "config", {})
    upload_dir = config.get("ml", {}).get("data_dir", "data/uploads")
    os.makedirs(upload_dir, exist_ok=True)

    save_path = os.path.join(upload_dir, file.filename)
    contents = await file.read()
    with open(save_path, "wb") as f:
        f.write(contents)

    # Reset pipeline on new upload
    _pipeline = GuardianPipeline(config)
    pipeline = _pipeline
    _last_upload_path = save_path

    summary = pipeline.load_data(save_path)
    df = pipeline._raw_df

    return UploadResponse(
        status="uploaded",
        filename=file.filename,
        n_samples=summary["n_samples"],
        n_features=summary["n_features"],
        columns=list(df.columns),
        preview=df.head(5).fillna("").to_dict(orient="records"),
    )


# ------------------------------------------------------------------
# /process — Preprocess loaded data
# ------------------------------------------------------------------

class ProcessRequest(BaseModel):
    target_col: Optional[str] = None


@router.post("/process")
async def process_data(request: Request, body: ProcessRequest):
    pipeline = _get_pipeline(request)
    if pipeline._raw_df is None:
        raise HTTPException(status_code=400, detail="No data loaded. Call /upload first.")
    try:
        result = pipeline.preprocess(target_col=body.target_col)
        return result
    except Exception as e:
        logger.error(f"/process error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------
# /train — Train ML models
# ------------------------------------------------------------------

@router.post("/train")
async def train_models(request: Request):
    pipeline = _get_pipeline(request)
    if pipeline._X_train is None:
        raise HTTPException(status_code=400, detail="Data not preprocessed. Call /process first.")
    try:
        results = pipeline.train()
        # Strip non-serializable model objects
        clean = {
            k: {
                "metrics": v["metrics"],
                "feature_importance": dict(list(v["feature_importance"].items())[:10]),
                "n_train": v["n_train"],
                "n_test": v["n_test"],
            }
            for k, v in results.items()
        }
        return {"status": "trained", "results": clean}
    except Exception as e:
        logger.error(f"/train error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------
# /predict — Risk prediction on new data
# ------------------------------------------------------------------

class PredictRequest(BaseModel):
    data: dict
    model_name: str = "random_forest"


@router.post("/predict")
async def predict(request: Request, body: PredictRequest):
    pipeline = _get_pipeline(request)
    if not pipeline._results:
        raise HTTPException(status_code=400, detail="Models not trained. Call /train first.")
    try:
        result = pipeline.predict(body.data, body.model_name)
        return result
    except Exception as e:
        logger.error(f"/predict error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------
# /visualize — Generate charts as base64 PNG
# ------------------------------------------------------------------

@router.get("/visualize")
async def visualize(request: Request):
    pipeline = _get_pipeline(request)
    if not pipeline._results:
        raise HTTPException(status_code=400, detail="No training results. Train models first.")

    charts = {}

    # 1. Metrics comparison bar chart
    model_names = list(pipeline._results.keys())
    metrics_names = ["accuracy", "precision", "recall", "f1_score"]
    x = np.arange(len(metrics_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, model in enumerate(model_names):
        vals = [pipeline._results[model]["metrics"].get(m, 0) for m in metrics_names]
        ax.bar(x + i * width, vals, width, label=model.replace("_", " ").title())
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics_names])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    charts["metrics_comparison"] = _fig_to_b64(fig)
    plt.close(fig)

    # 2. Feature importance (Random Forest)
    rf_result = pipeline._results.get("random_forest", {})
    fi = rf_result.get("feature_importance", {})
    if fi:
        top_n = dict(list(fi.items())[:10])
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        ax2.barh(list(top_n.keys()), list(top_n.values()), color="#3b82f6")
        ax2.set_xlabel("Importance")
        ax2.set_title("Top 10 Feature Importances (Random Forest)")
        ax2.invert_yaxis()
        ax2.grid(axis="x", alpha=0.3)
        charts["feature_importance"] = _fig_to_b64(fig2)
        plt.close(fig2)

    return {"charts": charts, "model_count": len(model_names)}


# ------------------------------------------------------------------
# /status — Pipeline status
# ------------------------------------------------------------------

@router.get("/status")
async def status(request: Request):
    pipeline = _get_pipeline(request)
    return pipeline.get_status()


# ------------------------------------------------------------------
# /evaluate — Full evaluation report
# ------------------------------------------------------------------

@router.get("/evaluate")
async def evaluate(request: Request):
    pipeline = _get_pipeline(request)
    if not pipeline._results:
        raise HTTPException(status_code=400, detail="No training results. Train models first.")
    try:
        report = pipeline.evaluate()
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
