"""
GUARDIAN ML — Intelligent Risk & Decision Support Platform
Backend Entry Point — FastAPI Application
"""

import os
import uvicorn
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from api.routes import router
from utils.logger import get_logger

logger = get_logger("main")


def load_config(path: str = None) -> dict:
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Guardian ML backend starting up...")
    config = load_config()
    app.state.config = config
    logger.info(f"Loaded config: {config.get('app', {}).get('name', 'Guardian ML')}")
    yield
    logger.info("Guardian ML backend shutting down.")


app = FastAPI(
    title="Guardian ML",
    description="Intelligent Risk & Decision Support Platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "system": "Guardian ML",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": ["/api/v1/upload", "/api/v1/process", "/api/v1/train", "/api/v1/predict", "/api/v1/visualize"],
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    config = load_config()
    host = config.get("server", {}).get("host", "0.0.0.0")
    port = config.get("server", {}).get("port", 8000)
    uvicorn.run("main:app", host=host, port=port, reload=True, log_level="info")
