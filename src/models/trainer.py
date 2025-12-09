"""
Model Training Module for Credit Risk Scoring System.

This module handles model training, evaluation, and integration
with Optuna for hyperparameter tuning and MLflow for experiment tracking.
"""

import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import yaml
from loguru import logger
from optuna.integration.mlflow import MLflowCallback
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None


def load_hyperparameter_config(config_path: str = "configs/hyperparameters.yaml") -> Dict:
    """Load hyperparameter configuration from YAML file.

    Args:
        config_path: Path to configuration file.

    Returns:
        Dictionary containing hyperparameter configuration.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Kolmogorov-Smirnov statistic.

    The KS statistic measures the maximum separation between the
    cumulative distributions of positive and negative classes.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities for positive class.

    Returns:
        KS statistic value.
    """
    # Separate probabilities by class
    pos_probs = y_prob[y_true == 1]
    neg_probs = y_prob[y_true == 0]

    # Create bins for cumulative distribution
    all_probs = np.concatenate([pos_probs, neg_probs])
    bins = np.linspace(0, 1, 100)

    # Compute cumulative distributions
    pos_cdf = np.array([np.mean(pos_probs <= b) for b in bins])
    neg_cdf = np.array([np.mean(neg_probs <= b) for b in bins])

    # KS is maximum difference
    ks_stat = np.max(np.abs(pos_cdf - neg_cdf))

    return ks_stat


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics.

    Args:
        y_true: True binary labels.
        y_pred: Predicted binary labels.
        y_prob: Predicted probabilities for positive class.

    Returns:
        Dictionary of evaluation metrics.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob),
        "ks_statistic": compute_ks_statistic(y_true, y_prob),
    }

    # Compute confusion matrix metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["true_positives"] = int(tp)
    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0

    return metrics


def get_calibration_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> Dict[str, Any]:
    """Compute calibration curve and metrics.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.
        n_bins: Number of bins for calibration curve.

    Returns:
        Dictionary with calibration metrics and curves.
    """
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )

    # Expected Calibration Error (ECE)
    bin_counts = np.histogram(y_prob, bins=n_bins, range=(0, 1))[0]
    ece = np.sum(
        bin_counts / len(y_prob) * np.abs(fraction_of_positives - mean_predicted_value)
    )

    return {
        "fraction_of_positives": fraction_of_positives.tolist(),
        "mean_predicted_value": mean_predicted_value.tolist(),
        "expected_calibration_error": ece,
    }


def create_model(
    model_type: str, params: Optional[Dict] = None, random_state: int = 42
) -> Any:
    """Create a model instance based on type.

    Args:
        model_type: Type of model ('logistic_regression', 'xgboost', 'lightgbm').
        params: Model parameters.
        random_state: Random seed.

    Returns:
        Model instance.
    """
    params = params or {}

    if model_type == "logistic_regression":
        default_params = {
            "random_state": random_state,
            "max_iter": 1000,
            "solver": "saga",
        }
        default_params.update(params)
        return LogisticRegression(**default_params)

    elif model_type == "xgboost":
        if xgb is None:
            raise ImportError("XGBoost not installed")
        default_params = {
            "random_state": random_state,
            "eval_metric": "logloss",
            "use_label_encoder": False,
        }
        default_params.update(params)
        return xgb.XGBClassifier(**default_params)

    elif model_type == "lightgbm":
        if lgb is None:
            raise ImportError("LightGBM not installed")
        default_params = {
            "random_state": random_state,
            "verbose": -1,
        }
        default_params.update(params)
        return lgb.LGBMClassifier(**default_params)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def sample_hyperparameters(
    trial: optuna.Trial, model_type: str, config: Dict
) -> Dict[str, Any]:
    """Sample hyperparameters for Optuna trial.

    Args:
        trial: Optuna trial object.
        model_type: Type of model.
        config: Hyperparameter configuration.

    Returns:
        Dictionary of sampled hyperparameters.
    """
    params = {}
    param_config = config.get(model_type, {})

    for param_name, param_spec in param_config.items():
        param_type = param_spec.get("type")

        if param_type == "int":
            params[param_name] = trial.suggest_int(
                param_name, param_spec["low"], param_spec["high"]
            )
        elif param_type == "float":
            log_scale = param_spec.get("log", False)
            params[param_name] = trial.suggest_float(
                param_name, param_spec["low"], param_spec["high"], log=log_scale
            )
        elif param_type == "categorical":
            choices = param_spec["choices"]
            # Handle None values
            choices = [c if c is not None else "balanced" for c in choices]
            params[param_name] = trial.suggest_categorical(param_name, choices)

    return params


def objective_factory(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str,
    config: Dict,
    cv_folds: int = 5,
    metric: str = "roc_auc",
) -> Callable:
    """Create objective function for Optuna optimization.

    Args:
        X_train: Training features.
        y_train: Training labels.
        model_type: Type of model to optimize.
        config: Hyperparameter configuration.
        cv_folds: Number of cross-validation folds.
        metric: Optimization metric.

    Returns:
        Objective function for Optuna.
    """

    def objective(trial: optuna.Trial) -> float:
        # Sample hyperparameters
        params = sample_hyperparameters(trial, model_type, config)

        # Create model
        model = create_model(model_type, params)

        # Cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=metric)

        return scores.mean()

    return objective


class ModelTrainer:
    """Handles model training with Optuna and MLflow integration."""

    def __init__(
        self,
        model_type: str = "lightgbm",
        experiment_name: str = "credit_risk_scoring",
        tracking_uri: Optional[str] = None,
    ):
        """Initialize the trainer.

        Args:
            model_type: Type of model to train.
            experiment_name: MLflow experiment name.
            tracking_uri: MLflow tracking URI.
        """
        self.model_type = model_type
        self.experiment_name = experiment_name
        self.config = load_hyperparameter_config()

        # Setup MLflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # Use local SQLite database
            os.makedirs("mlruns", exist_ok=True)
            mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")

        mlflow.set_experiment(experiment_name)

        self.model = None
        self.calibrated_model = None
        self.best_params = None
        self.metrics = None

    def train_baseline(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Train a baseline model with default parameters.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features.
            y_val: Validation labels.

        Returns:
            Dictionary with model and metrics.
        """
        logger.info(f"Training baseline {self.model_type} model")

        with mlflow.start_run(run_name=f"baseline_{self.model_type}"):
            # Create model with defaults
            self.model = create_model(self.model_type)

            # Log model type
            mlflow.log_param("model_type", self.model_type)
            mlflow.log_param("is_baseline", True)

            # Train
            self.model.fit(X_train, y_train)

            # Evaluate on training set
            y_train_pred = self.model.predict(X_train)
            y_train_prob = self.model.predict_proba(X_train)[:, 1]
            train_metrics = evaluate_model(y_train, y_train_pred, y_train_prob)

            for name, value in train_metrics.items():
                mlflow.log_metric(f"train_{name}", value)

            # Evaluate on validation set if provided
            if X_val is not None and y_val is not None:
                y_val_pred = self.model.predict(X_val)
                y_val_prob = self.model.predict_proba(X_val)[:, 1]
                val_metrics = evaluate_model(y_val, y_val_pred, y_val_prob)

                for name, value in val_metrics.items():
                    mlflow.log_metric(f"val_{name}", value)

                self.metrics = val_metrics
            else:
                self.metrics = train_metrics

            # Log model
            mlflow.sklearn.log_model(self.model, "model")

            logger.info(f"Baseline model trained. ROC-AUC: {self.metrics['roc_auc']:.4f}")

        return {"model": self.model, "metrics": self.metrics}

    def tune_with_optuna(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        n_trials: int = 50,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Tune hyperparameters using Optuna with MLflow logging.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features.
            y_val: Validation labels.
            n_trials: Number of Optuna trials.
            timeout: Maximum time in seconds for optimization.

        Returns:
            Dictionary with best model and metrics.
        """
        logger.info(f"Starting Optuna optimization for {self.model_type}")

        optuna_config = self.config.get("optuna", {})
        metric = optuna_config.get("metric", "roc_auc")
        cv_folds = self.config.get("cross_validation", {}).get("n_folds", 5)

        # Create objective function
        objective = objective_factory(
            X_train, y_train, self.model_type, self.config, cv_folds, metric
        )

        # Create study
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        )

        # MLflow callback
        mlflow_callback = MLflowCallback(
            tracking_uri=mlflow.get_tracking_uri(),
            metric_name=metric,
            create_experiment=False,
        )

        # Optimize
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=[mlflow_callback],
            show_progress_bar=True,
        )

        # Get best parameters
        self.best_params = study.best_params
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best CV score: {study.best_value:.4f}")

        # Train final model with best parameters
        with mlflow.start_run(run_name=f"best_{self.model_type}"):
            self.model = create_model(self.model_type, self.best_params)
            self.model.fit(X_train, y_train)

            # Log parameters
            mlflow.log_param("model_type", self.model_type)
            mlflow.log_param("is_tuned", True)
            mlflow.log_params(self.best_params)

            # Evaluate
            y_train_pred = self.model.predict(X_train)
            y_train_prob = self.model.predict_proba(X_train)[:, 1]
            train_metrics = evaluate_model(y_train, y_train_pred, y_train_prob)

            for name, value in train_metrics.items():
                mlflow.log_metric(f"train_{name}", value)

            if X_val is not None and y_val is not None:
                y_val_pred = self.model.predict(X_val)
                y_val_prob = self.model.predict_proba(X_val)[:, 1]
                val_metrics = evaluate_model(y_val, y_val_pred, y_val_prob)

                for name, value in val_metrics.items():
                    mlflow.log_metric(f"val_{name}", value)

                self.metrics = val_metrics
            else:
                self.metrics = train_metrics

            # Log model
            mlflow.sklearn.log_model(self.model, "model")

            logger.info(f"Tuned model trained. ROC-AUC: {self.metrics['roc_auc']:.4f}")

        return {
            "model": self.model,
            "best_params": self.best_params,
            "metrics": self.metrics,
            "study": study,
        }

    def calibrate_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        method: str = "isotonic",
        cv: int = 5,
    ) -> CalibratedClassifierCV:
        """Calibrate the model for better probability estimates.

        Args:
            X_train: Training features.
            y_train: Training labels.
            method: Calibration method ('isotonic' or 'sigmoid').
            cv: Number of cross-validation folds.

        Returns:
            Calibrated model.
        """
        if self.model is None:
            raise ValueError("Model must be trained before calibration")

        logger.info(f"Calibrating model using {method} method")

        self.calibrated_model = CalibratedClassifierCV(
            self.model, method=method, cv=cv
        )
        self.calibrated_model.fit(X_train, y_train)

        logger.info("Model calibration complete")

        return self.calibrated_model

    def evaluate_on_test(
        self, X_test: np.ndarray, y_test: np.ndarray, use_calibrated: bool = True
    ) -> Dict[str, Any]:
        """Evaluate model on test set.

        Args:
            X_test: Test features.
            y_test: Test labels.
            use_calibrated: Whether to use calibrated model.

        Returns:
            Dictionary with test metrics.
        """
        model = self.calibrated_model if use_calibrated and self.calibrated_model else self.model

        if model is None:
            raise ValueError("No trained model available")

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        test_metrics = evaluate_model(y_test, y_pred, y_prob)
        calibration_metrics = get_calibration_metrics(y_test, y_prob)

        logger.info(f"Test metrics:")
        logger.info(f"  ROC-AUC: {test_metrics['roc_auc']:.4f}")
        logger.info(f"  KS Statistic: {test_metrics['ks_statistic']:.4f}")
        logger.info(f"  Brier Score: {test_metrics['brier_score']:.4f}")
        logger.info(f"  ECE: {calibration_metrics['expected_calibration_error']:.4f}")

        return {
            "metrics": test_metrics,
            "calibration": calibration_metrics,
        }

    def register_model(
        self,
        model_name: str = "credit_risk_model",
        stage: str = "Staging",
        pipeline: Optional[Pipeline] = None,
    ) -> str:
        """Register model in MLflow Model Registry.

        Args:
            model_name: Name for registered model.
            stage: Model stage ('Staging' or 'Production').
            pipeline: Optional preprocessing pipeline to include.

        Returns:
            Model version.
        """
        model_to_register = (
            self.calibrated_model if self.calibrated_model else self.model
        )

        if model_to_register is None:
            raise ValueError("No trained model available to register")

        with mlflow.start_run(run_name=f"register_{model_name}"):
            # Log metrics
            if self.metrics:
                for name, value in self.metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(name, value)

            # Log model
            model_info = mlflow.sklearn.log_model(
                model_to_register, "model", registered_model_name=model_name
            )

            # If pipeline provided, log it too
            if pipeline is not None:
                mlflow.sklearn.log_model(pipeline, "preprocessing_pipeline")

            logger.info(f"Model registered as {model_name}")

        return model_info.model_uri

    def get_model(self, calibrated: bool = True) -> Any:
        """Get the trained model.

        Args:
            calibrated: Whether to return calibrated model.

        Returns:
            Trained model.
        """
        if calibrated and self.calibrated_model:
            return self.calibrated_model
        return self.model
