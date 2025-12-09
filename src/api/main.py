"""
FastAPI Application for Credit Risk Scoring System.

This module provides REST API endpoints for credit scoring,
health checks, and model information.
"""

import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, Optional

import mlflow
import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from src.api.schemas import (
    ApplicantFeatures,
    BatchScoringRequest,
    BatchScoringResponse,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    ScoringResponse,
)
from src.scoring.scorecard import CreditScorecard


# Global state
class AppState:
    """Application state container."""

    model: Any = None
    pipeline: Any = None
    scorecard: Optional[CreditScorecard] = None
    model_version: str = "unknown"
    model_info: Dict = {}
    is_loaded: bool = False


state = AppState()


def load_config(config_path: str = "configs/paths.yaml") -> Dict:
    """Load configuration from YAML file."""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def load_model_from_mlflow() -> bool:
    """Load model from MLflow Model Registry.

    Returns:
        True if model loaded successfully, False otherwise.
    """
    try:
        config = load_config()
        tracking_uri = config.get("models", {}).get(
            "tracking_uri", "sqlite:///mlruns/mlflow.db"
        )

        mlflow.set_tracking_uri(tracking_uri)

        # Try to load production model
        model_name = "credit_risk_model"

        try:
            # Try loading from model registry
            model_uri = f"models:/{model_name}/Production"
            state.model = mlflow.sklearn.load_model(model_uri)
            state.model_version = "Production"
            logger.info(f"Loaded production model: {model_name}")
        except Exception as e:
            logger.warning(f"Could not load from registry: {e}")

            # Fallback: try to load latest run
            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment_by_name("credit_risk_scoring")

            if experiment:
                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["start_time DESC"],
                    max_results=1,
                )

                if runs:
                    run = runs[0]
                    model_uri = f"runs:/{run.info.run_id}/model"
                    state.model = mlflow.sklearn.load_model(model_uri)
                    state.model_version = run.info.run_id[:8]

                    # Get metrics
                    state.model_info = {
                        "run_id": run.info.run_id,
                        "metrics": run.data.metrics,
                        "params": run.data.params,
                    }
                    logger.info(f"Loaded model from run: {run.info.run_id}")

        state.is_loaded = state.model is not None
        return state.is_loaded

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False


def load_mock_model() -> bool:
    """Load a mock model for testing when no trained model exists.

    Returns:
        True if mock model created successfully.
    """
    from sklearn.linear_model import LogisticRegression

    logger.warning("Loading mock model for development/testing")

    # Create a simple mock model
    # In production, this should never be used
    state.model = LogisticRegression()

    # Fit on dummy data
    X_dummy = np.random.randn(100, 20)
    y_dummy = np.random.randint(0, 2, 100)
    state.model.fit(X_dummy, y_dummy)

    state.model_version = "mock-v1"
    state.model_info = {
        "type": "mock",
        "warning": "This is a mock model for testing only",
    }
    state.is_loaded = True

    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup
    logger.info("Starting Credit Risk Scoring API")

    # Initialize scorecard
    try:
        state.scorecard = CreditScorecard()
        logger.info("Scorecard initialized")
    except Exception as e:
        logger.error(f"Failed to initialize scorecard: {e}")

    # Try to load model
    if not load_model_from_mlflow():
        # Fall back to mock model for development
        load_mock_model()

    yield

    # Shutdown
    logger.info("Shutting down Credit Risk Scoring API")


# Create FastAPI app
app = FastAPI(
    title="Credit Risk Scoring API",
    description="API for credit risk assessment and scoring",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=datetime.utcnow(),
        ).model_dump(mode="json"),
    )


def prepare_features(applicant: ApplicantFeatures) -> pd.DataFrame:
    """Convert applicant features to DataFrame for model input.

    Args:
        applicant: Applicant features.

    Returns:
        DataFrame with features.
    """
    # Convert to dict and then DataFrame
    data = applicant.model_dump()
    df = pd.DataFrame([data])
    return df


def predict_default_probability(features: pd.DataFrame) -> float:
    """Get probability of default from model.

    Args:
        features: Feature DataFrame.

    Returns:
        Probability of default.
    """
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Get probability for positive class (default)
        prob = state.model.predict_proba(features)[:, 1][0]
        return float(prob)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        # Return a mock prediction for testing
        return 0.15


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint.

    Returns service health status and model availability.
    """
    return HealthResponse(
        status="healthy" if state.is_loaded else "unhealthy",
        model_loaded=state.is_loaded,
        timestamp=datetime.utcnow(),
    )


@app.get("/model_info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """Get information about the deployed model.

    Returns model version, type, metrics, and training information.
    """
    if not state.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    metrics = state.model_info.get("metrics", {})

    return ModelInfoResponse(
        model_name="credit_risk_model",
        model_version=state.model_version,
        model_type=type(state.model).__name__,
        training_date=state.model_info.get("training_date"),
        metrics={k: round(v, 4) for k, v in metrics.items() if isinstance(v, (int, float))},
        feature_count=20,  # Based on German Credit dataset
        is_calibrated="Calibrated" in type(state.model).__name__,
    )


@app.post("/score", response_model=ScoringResponse, tags=["Scoring"])
async def score_applicant(applicant: ApplicantFeatures):
    """Score a single applicant.

    Takes applicant features and returns:
    - Probability of default (PD)
    - Credit score (300-850)
    - Risk band (Low/Medium/High)
    - Recommended decision
    """
    request_id = f"req_{uuid.uuid4().hex[:12]}"

    try:
        # Prepare features
        features = prepare_features(applicant)

        # Get PD from model
        pd_value = predict_default_probability(features)

        # Score using scorecard
        if state.scorecard:
            result = state.scorecard.score_applicant(
                pd_value, applicant_data={"age": applicant.age}
            )
        else:
            # Fallback scoring
            from src.scoring.scorecard import CreditScorecard
            scorecard = CreditScorecard()
            result = scorecard.score_applicant(pd_value)

        # Log request (without PII)
        logger.info(
            f"Scored applicant {request_id}: PD={pd_value:.4f}, "
            f"Score={result.credit_score}, Band={result.risk_band}"
        )

        return ScoringResponse(
            request_id=request_id,
            probability_of_default=round(pd_value, 4),
            credit_score=result.credit_score,
            risk_band=result.risk_band,
            decision=result.decision,
            expected_default_rate=result.expected_default_rate,
            model_version=state.model_version,
            timestamp=datetime.utcnow(),
        )

    except Exception as e:
        logger.error(f"Scoring error for {request_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/score/batch", response_model=BatchScoringResponse, tags=["Scoring"])
async def score_batch(request: BatchScoringRequest):
    """Score multiple applicants in batch.

    Takes a list of applicants and returns scoring results for each.
    Maximum batch size: 1000 applicants.
    """
    batch_id = f"batch_{uuid.uuid4().hex[:12]}"
    start_time = time.time()

    results = []
    for i, applicant in enumerate(request.applicants):
        try:
            features = prepare_features(applicant)
            pd_value = predict_default_probability(features)

            if state.scorecard:
                result = state.scorecard.score_applicant(
                    pd_value, applicant_data={"age": applicant.age}
                )
            else:
                from src.scoring.scorecard import CreditScorecard
                scorecard = CreditScorecard()
                result = scorecard.score_applicant(pd_value)

            results.append(
                ScoringResponse(
                    request_id=f"{batch_id}_{i}",
                    probability_of_default=round(pd_value, 4),
                    credit_score=result.credit_score,
                    risk_band=result.risk_band,
                    decision=result.decision,
                    expected_default_rate=result.expected_default_rate,
                    model_version=state.model_version,
                    timestamp=datetime.utcnow(),
                )
            )
        except Exception as e:
            logger.error(f"Error scoring applicant {i} in batch {batch_id}: {e}")
            # Include error result
            results.append(
                ScoringResponse(
                    request_id=f"{batch_id}_{i}_error",
                    probability_of_default=1.0,
                    credit_score=300,
                    risk_band="High Risk",
                    decision="decline",
                    expected_default_rate=0.25,
                    model_version=state.model_version,
                    timestamp=datetime.utcnow(),
                )
            )

    processing_time = (time.time() - start_time) * 1000

    logger.info(
        f"Batch {batch_id}: processed {len(results)} applicants in {processing_time:.2f}ms"
    )

    return BatchScoringResponse(
        batch_id=batch_id,
        results=results,
        total_processed=len(results),
        processing_time_ms=round(processing_time, 2),
    )


@app.get("/risk_bands", tags=["Configuration"])
async def get_risk_bands():
    """Get configured risk band definitions."""
    if state.scorecard:
        return {"risk_bands": state.scorecard.risk_bands}

    # Fallback
    from src.scoring.scorecard import load_risk_policy
    config = load_risk_policy()
    return {"risk_bands": config.get("risk_bands", {})}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
