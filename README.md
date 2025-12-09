# Credit Risk Scoring System

A production-grade credit risk scoring system for predicting probability of default (PD) and generating credit scores for retail loan and BNPL customers.

## Overview

This system implements an end-to-end ML pipeline for credit risk assessment:

- **Data Pipeline**: Ingestion, cleaning, and feature engineering
- **Model Training**: LightGBM/XGBoost with Optuna hyperparameter tuning
- **Experiment Tracking**: MLflow for experiment logging and model registry
- **Scoring Engine**: PD to credit score conversion with risk bands
- **API Service**: FastAPI REST API for real-time scoring
- **Monitoring**: Data drift and model performance monitoring
- **Explainability**: SHAP-based feature importance and individual explanations
- **Deployment**: Docker containerization with docker-compose

## Quick Start

### Prerequisites

- Python 3.10+
- Docker and Docker Compose (optional)
- Make (optional, for convenience commands)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Credit-Risk-Scoring-System

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Or use Make:

```bash
make setup
source .venv/bin/activate
```

### Run the Pipeline

```bash
# 1. Download and preprocess data
make download_data
make preprocess

# 2. Train model with hyperparameter tuning
make tune

# 3. Start the API server
make run_api

# 4. Access the API
# - API: http://localhost:8000
# - Docs: http://localhost:8000/docs
# - Health: http://localhost:8000/health
```

### Using Docker

```bash
# Build and run containers
make docker_build
make docker_run

# View logs
make docker_logs

# Stop containers
make docker_stop
```

## Project Structure

```
Credit-Risk-Scoring-System/
├── configs/                    # Configuration files
│   ├── paths.yaml             # Data and model paths
│   ├── hyperparameters.yaml   # Model hyperparameters
│   ├── risk_policy.yaml       # Risk bands and thresholds
│   └── monitoring.yaml        # Monitoring settings
├── data/
│   ├── raw/                   # Raw dataset
│   ├── processed/             # Processed train/val/test splits
│   └── external/              # External data sources
├── docs/
│   ├── business_requirements.md
│   └── model_card.md
├── mlruns/                    # MLflow tracking
├── notebooks/                 # Exploration notebooks
├── reports/
│   ├── drift/                 # Drift monitoring reports
│   ├── training/              # Training reports
│   └── explainability/        # SHAP reports
├── src/
│   ├── api/                   # FastAPI application
│   │   ├── main.py           # API endpoints
│   │   └── schemas.py        # Pydantic models
│   ├── data/                  # Data processing
│   │   ├── ingestion.py      # Data loading
│   │   ├── cleaning.py       # Data cleaning
│   │   └── feature_engineering.py
│   ├── models/                # Model training
│   │   └── trainer.py        # Training with Optuna + MLflow
│   ├── scoring/               # Credit scoring
│   │   └── scorecard.py      # PD to score conversion
│   ├── monitoring/            # Model monitoring
│   │   ├── drift_monitor.py
│   │   └── performance_monitor.py
│   ├── explainability/        # SHAP explanations
│   │   ├── shap_explainer.py
│   │   └── shap_reports.py
│   └── pipelines/             # Orchestration
│       ├── training_pipeline.py
│       ├── retraining_pipeline.py
│       └── promote_model_if_better.py
├── tests/                     # Unit tests
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
└── pyproject.toml
```

## API Usage

### Score an Applicant

```bash
curl -X POST "http://localhost:8000/score" \
  -H "Content-Type: application/json" \
  -d '{
    "checking_account_status": "0_to_200",
    "duration_months": 24,
    "credit_history": "existing_paid_duly",
    "purpose": "car_new",
    "credit_amount": 5000,
    "savings_account": "100_to_500",
    "employment_duration": "1_to_4_years",
    "installment_rate": 3,
    "age": 35,
    "housing": "own",
    "existing_credits": 1,
    "num_dependents": 1
  }'
```

### Response

```json
{
  "request_id": "req_abc123",
  "probability_of_default": 0.08,
  "credit_score": 680,
  "risk_band": "Medium Risk",
  "decision": "manual_review",
  "expected_default_rate": 0.08,
  "model_version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Risk Bands

| Risk Band | Score Range | PD Range | Decision |
|-----------|-------------|----------|----------|
| Low Risk | 700-850 | ≤5% | Auto-Approve |
| Medium Risk | 600-699 | 5%-15% | Manual Review |
| High Risk | 300-599 | >15% | Decline |

## Key Commands

```bash
# Data Pipeline
make download_data     # Download raw dataset
make preprocess        # Run preprocessing pipeline

# Model Training
make train             # Train with default settings
make tune              # Train with Optuna tuning
make train_baseline    # Train baseline models

# API
make run_api           # Start API server
make run_api_prod      # Start API in production mode

# Monitoring
make monitoring        # Run drift and performance monitoring
make retrain           # Run retraining pipeline
make promote           # Check and promote model if better

# Reports
make reports           # Generate SHAP reports

# Development
make test              # Run unit tests
make lint              # Run linting
make format            # Format code

# Docker
make docker_build      # Build Docker image
make docker_run        # Start containers
make docker_stop       # Stop containers

# MLflow
make mlflow_ui         # Start MLflow UI at http://localhost:5000
```

## Configuration

### Risk Policy (`configs/risk_policy.yaml`)

```yaml
score_conversion:
  base_score: 600
  base_odds: 50
  pdo: 20  # Points to Double the Odds
  min_score: 300
  max_score: 850

risk_bands:
  low_risk:
    min_score: 700
    max_score: 850
    decision: "auto_approve"
```

### Monitoring Thresholds (`configs/monitoring.yaml`)

```yaml
data_drift:
  psi:
    warning_threshold: 0.1
    critical_threshold: 0.25

performance:
  auc:
    warning_threshold: 0.72
    critical_threshold: 0.68
```

## Monitoring & Retraining

### Automated Monitoring

The system monitors:
- **Data Drift**: PSI, KS statistic for feature distributions
- **Model Performance**: AUC, KS, Brier score trends
- **Risk Band Stability**: Actual vs expected default rates

### Retraining Triggers

Automatic retraining is triggered when:
- PSI exceeds 0.25 for major features
- AUC drops below 0.68
- Default rates deviate >10% from expected

### Model Promotion

```bash
# Check if staging model should be promoted
python -m src.pipelines.promote_model_if_better

# Force promote (use with caution)
python -m src.pipelines.promote_model_if_better --force-promote
```

## Testing

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_scoring.py -v

# Run with coverage
pytest --cov=src --cov-report=html
```

## Documentation

- [Business Requirements](docs/business_requirements.md) - Problem definition and requirements
- [Model Card](docs/model_card.md) - Model documentation and limitations
- [API Documentation](http://localhost:8000/docs) - Interactive API docs (when running)

## Technology Stack

- **ML Framework**: scikit-learn, LightGBM, XGBoost
- **Hyperparameter Tuning**: Optuna
- **Experiment Tracking**: MLflow
- **API Framework**: FastAPI
- **Containerization**: Docker, Docker Compose
- **Explainability**: SHAP
- **Monitoring**: Evidently AI (optional)
- **Testing**: pytest

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and add tests
4. Run `make quality` to ensure code quality
5. Submit a pull request
