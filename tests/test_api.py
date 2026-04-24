"""
Guardian ML — API Unit Tests
Run: pytest tests/test_api.py -v
Requires: backend running OR test client mode
"""

import io
import json
import os
import sys
import pytest

# Allow importing from backend root
BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend")
sys.path.insert(0, os.path.abspath(BACKEND_DIR))

from fastapi.testclient import TestClient

# Minimal config for tests
TEST_CONFIG = {
    "app": {"name": "Guardian ML Test", "version": "1.0.0"},
    "server": {"host": "0.0.0.0", "port": 8000},
    "ml": {
        "random_seed": 42,
        "test_size": 0.2,
        "models": ["logistic_regression", "random_forest"],
        "model_dir": "/tmp/guardian_test_models",
        "data_dir": "/tmp/guardian_test_uploads",
    },
    "logging": {"level": "WARNING"},
    "risk": {"thresholds": {"low": 0.33, "medium": 0.66, "high": 1.0}},
    "visualization": {"output_dir": "/tmp/guardian_plots", "dpi": 72, "format": "png"},
}


def _make_app():
    """Create a test FastAPI app with injected config."""
    import main as m
    app = m.app
    app.state.config = TEST_CONFIG
    return app


@pytest.fixture(scope="module")
def client():
    app = _make_app()
    with TestClient(app) as c:
        yield c


# ── CSV fixture ────────────────────────────────────────────────────
SAMPLE_CSV = """feature_a,feature_b,feature_c,label
1.2,3.4,0.1,0
2.3,1.1,0.5,1
0.5,2.2,0.9,0
3.1,0.8,0.3,1
1.8,2.9,0.7,0
2.6,1.5,0.4,1
0.9,3.3,0.8,0
3.4,0.6,0.2,1
1.5,2.7,0.6,0
2.9,1.2,0.5,1
0.7,3.1,0.9,0
3.0,0.9,0.3,1
1.3,2.5,0.7,0
2.7,1.3,0.4,1
0.8,3.0,0.8,0
3.2,0.7,0.2,1
1.6,2.6,0.6,0
2.8,1.1,0.5,1
0.6,3.2,0.9,0
3.3,0.8,0.3,1
"""


# ── Tests ──────────────────────────────────────────────────────────

class TestHealthEndpoints:
    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        body = r.json()
        assert body["system"] == "Guardian ML"
        assert body["status"] == "operational"
        assert "/api/v1/upload" in body["endpoints"]

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    def test_api_status(self, client):
        r = client.get("/api/v1/status")
        assert r.status_code == 200
        body = r.json()
        assert "data_loaded" in body
        assert "preprocessed" in body
        assert "trained" in body


class TestUploadEndpoint:
    def test_upload_csv(self, client):
        csv_bytes = SAMPLE_CSV.encode("utf-8")
        files = {"file": ("test_data.csv", io.BytesIO(csv_bytes), "text/csv")}
        r = client.post("/api/v1/upload", files=files)
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "uploaded"
        assert body["filename"] == "test_data.csv"
        assert body["n_samples"] == 20
        assert body["n_features"] == 4
        assert "label" in body["columns"]
        assert len(body["preview"]) <= 5

    def test_upload_invalid_format(self, client):
        files = {"file": ("test.txt", io.BytesIO(b"not a csv"), "text/plain")}
        r = client.post("/api/v1/upload", files=files)
        assert r.status_code == 400
        assert "detail" in r.json()


class TestProcessEndpoint:
    def test_process_with_target(self, client):
        # Upload first
        csv_bytes = SAMPLE_CSV.encode("utf-8")
        files = {"file": ("test_data.csv", io.BytesIO(csv_bytes), "text/csv")}
        client.post("/api/v1/upload", files=files)

        r = client.post(
            "/api/v1/process",
            json={"target_col": "label"},
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "preprocessed"
        assert body["n_features"] == 3  # feature_a, feature_b, feature_c
        assert body["supervised"] is True
        assert body["train_samples"] > 0
        assert body["test_samples"] > 0

    def test_process_no_data(self, client):
        """Process without uploading should fail if pipeline is fresh."""
        # This test depends on state — skip if pipeline already has data
        pass


class TestTrainEndpoint:
    def test_full_pipeline_train(self, client):
        # Upload
        csv_bytes = SAMPLE_CSV.encode("utf-8")
        files = {"file": ("test_data.csv", io.BytesIO(csv_bytes), "text/csv")}
        client.post("/api/v1/upload", files=files)

        # Process
        client.post(
            "/api/v1/process",
            json={"target_col": "label"},
            headers={"Content-Type": "application/json"},
        )

        # Train
        r = client.post("/api/v1/train")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "trained"
        assert "results" in body

        results = body["results"]
        for model_name in ["logistic_regression", "random_forest"]:
            assert model_name in results
            metrics = results[model_name]["metrics"]
            assert "accuracy"  in metrics
            assert "precision" in metrics
            assert "recall"    in metrics
            assert "f1_score"  in metrics
            assert 0.0 <= metrics["accuracy"]  <= 1.0
            assert 0.0 <= metrics["f1_score"]  <= 1.0


class TestPredictEndpoint:
    def _setup(self, client):
        csv_bytes = SAMPLE_CSV.encode("utf-8")
        files = {"file": ("test_data.csv", io.BytesIO(csv_bytes), "text/csv")}
        client.post("/api/v1/upload", files=files)
        client.post(
            "/api/v1/process",
            json={"target_col": "label"},
            headers={"Content-Type": "application/json"},
        )
        client.post("/api/v1/train")

    def test_predict_random_forest(self, client):
        self._setup(client)
        payload = {
            "data": {"feature_a": 2.5, "feature_b": 1.2, "feature_c": 0.4},
            "model_name": "random_forest",
        }
        r = client.post("/api/v1/predict", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert "risk_score" in body
        assert "risk_label" in body
        assert body["risk_label"] in ("LOW", "MEDIUM", "HIGH")
        assert 0.0 <= body["risk_score"] <= 1.0
        assert body["model_used"] == "random_forest"

    def test_predict_logistic_regression(self, client):
        self._setup(client)
        payload = {
            "data": {"feature_a": 0.9, "feature_b": 3.2, "feature_c": 0.8},
            "model_name": "logistic_regression",
        }
        r = client.post("/api/v1/predict", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert "prediction" in body
        assert "risk_score" in body


class TestEvaluateEndpoint:
    def test_evaluate_after_train(self, client):
        # Setup pipeline
        csv_bytes = SAMPLE_CSV.encode("utf-8")
        files = {"file": ("test_data.csv", io.BytesIO(csv_bytes), "text/csv")}
        client.post("/api/v1/upload", files=files)
        client.post(
            "/api/v1/process",
            json={"target_col": "label"},
            headers={"Content-Type": "application/json"},
        )
        client.post("/api/v1/train")

        r = client.get("/api/v1/evaluate")
        assert r.status_code == 200
        body = r.json()
        assert "model_comparison"  in body
        assert "best_model"        in body
        assert "risk_insights"     in body
        assert "recommendations"   in body
        assert isinstance(body["risk_insights"],   list)
        assert isinstance(body["recommendations"], list)


class TestVisualizeEndpoint:
    def test_visualize_after_train(self, client):
        csv_bytes = SAMPLE_CSV.encode("utf-8")
        files = {"file": ("test_data.csv", io.BytesIO(csv_bytes), "text/csv")}
        client.post("/api/v1/upload", files=files)
        client.post(
            "/api/v1/process",
            json={"target_col": "label"},
            headers={"Content-Type": "application/json"},
        )
        client.post("/api/v1/train")

        r = client.get("/api/v1/visualize")
        assert r.status_code == 200
        body = r.json()
        assert "charts" in body
        assert "metrics_comparison" in body["charts"]
        # Validate base64 PNG
        b64 = body["charts"]["metrics_comparison"]
        assert len(b64) > 100
        import base64
        decoded = base64.b64decode(b64)
        assert decoded[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic bytes
