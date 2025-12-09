.PHONY: help setup install clean test lint format train tune run_api \
        docker_build docker_run docker_stop monitoring retrain \
        download_data preprocess mlflow_ui reports

# Default target
help:
	@echo "Credit Risk Scoring System - Available Commands"
	@echo "================================================"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make setup          - Create virtual environment and install dependencies"
	@echo "  make install        - Install dependencies only"
	@echo "  make clean          - Remove build artifacts and caches"
	@echo ""
	@echo "Data Pipeline:"
	@echo "  make download_data  - Download raw dataset"
	@echo "  make preprocess     - Run data preprocessing and feature engineering"
	@echo ""
	@echo "Model Training:"
	@echo "  make train          - Train model with default configuration"
	@echo "  make tune           - Run hyperparameter tuning with Optuna"
	@echo "  make train_baseline - Train baseline models only"
	@echo ""
	@echo "API & Deployment:"
	@echo "  make run_api        - Run FastAPI server locally"
	@echo "  make docker_build   - Build Docker image"
	@echo "  make docker_run     - Run Docker containers"
	@echo "  make docker_stop    - Stop Docker containers"
	@echo ""
	@echo "Monitoring & Maintenance:"
	@echo "  make monitoring     - Run drift and performance monitoring"
	@echo "  make retrain        - Run retraining pipeline"
	@echo "  make reports        - Generate all reports"
	@echo ""
	@echo "Development:"
	@echo "  make test           - Run unit tests"
	@echo "  make lint           - Run linting checks"
	@echo "  make format         - Format code with black and isort"
	@echo "  make mlflow_ui      - Start MLflow UI"

# Python and environment settings
PYTHON := python3
VENV := .venv
PIP := $(VENV)/bin/pip
PYTHON_VENV := $(VENV)/bin/python

# Setup virtual environment and install dependencies
setup:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Setup complete. Activate with: source $(VENV)/bin/activate"

# Install dependencies only (assumes venv is active)
install:
	pip install -r requirements.txt

# Clean build artifacts
clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .coverage htmlcov
	rm -rf build dist *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned build artifacts"

# Download raw dataset
download_data:
	$(PYTHON) -m src.data.ingestion download
	@echo "Data downloaded successfully"

# Run data preprocessing
preprocess:
	$(PYTHON) -m src.data.ingestion preprocess
	@echo "Preprocessing complete"

# Train model with default configuration
train:
	$(PYTHON) -m src.pipelines.training_pipeline --model lightgbm
	@echo "Training complete"

# Train baseline models
train_baseline:
	$(PYTHON) -m src.pipelines.training_pipeline --model logistic_regression --baseline
	$(PYTHON) -m src.pipelines.training_pipeline --model xgboost --baseline
	@echo "Baseline training complete"

# Run hyperparameter tuning
tune:
	$(PYTHON) -m src.pipelines.training_pipeline --model lightgbm --tune
	@echo "Tuning complete"

# Run FastAPI server locally
run_api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Run API without reload (production mode)
run_api_prod:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4

# Docker commands
docker_build:
	docker build -t credit-risk-api:latest .

docker_run:
	docker-compose up -d

docker_stop:
	docker-compose down

docker_logs:
	docker-compose logs -f

# Run monitoring scripts
monitoring:
	$(PYTHON) -m src.monitoring.drift_monitor
	$(PYTHON) -m src.monitoring.performance_monitor
	@echo "Monitoring complete"

# Run retraining pipeline
retrain:
	$(PYTHON) -m src.pipelines.retraining_pipeline
	@echo "Retraining pipeline complete"

# Promote model if better
promote:
	$(PYTHON) -m src.pipelines.promote_model_if_better
	@echo "Model promotion check complete"

# Generate reports
reports:
	$(PYTHON) -m src.explainability.shap_reports
	@echo "Reports generated"

# Start MLflow UI
mlflow_ui:
	mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --host 0.0.0.0 --port 5000

# Run unit tests
test:
	pytest tests/ -v --cov=src --cov-report=term-missing

# Run specific test file
test_file:
	pytest $(FILE) -v

# Run linting
lint:
	flake8 src/ tests/ --max-line-length=100 --ignore=E501,W503
	mypy src/ --ignore-missing-imports

# Format code
format:
	black src/ tests/
	isort src/ tests/

# Check formatting without applying
format_check:
	black --check src/ tests/
	isort --check-only src/ tests/

# Run all quality checks
quality: format_check lint test

# Generate documentation
docs:
	@echo "Documentation generation not yet implemented"

# Full pipeline: download, preprocess, train
full_pipeline: download_data preprocess tune
	@echo "Full pipeline complete"

# Development setup with pre-commit hooks
dev_setup: setup
	$(PIP) install pre-commit
	pre-commit install
	@echo "Development setup complete"
