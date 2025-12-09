"""
Training Pipeline for Credit Risk Scoring System.

This module orchestrates the end-to-end training process including
data loading, preprocessing, model training, tuning, and registration.
"""

import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

import mlflow
import pandas as pd
import yaml
from loguru import logger

from src.data.cleaning import run_cleaning_pipeline
from src.data.feature_engineering import get_feature_lists, run_feature_engineering
from src.data.ingestion import load_raw_data, run_ingestion_pipeline
from src.models.trainer import ModelTrainer
from src.scoring.scorecard import CreditScorecard


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_training_pipeline(
    model_type: str = "lightgbm",
    tune: bool = True,
    n_trials: int = 50,
    register: bool = True,
    experiment_name: str = "credit_risk_scoring",
) -> Dict[str, Any]:
    """Run the complete training pipeline.

    Args:
        model_type: Type of model to train.
        tune: Whether to run hyperparameter tuning.
        n_trials: Number of Optuna trials.
        register: Whether to register the model.
        experiment_name: MLflow experiment name.

    Returns:
        Dictionary with training results.
    """
    logger.info(f"Starting training pipeline for {model_type}")
    logger.info(f"Tuning: {tune}, Trials: {n_trials}")

    results = {
        "model_type": model_type,
        "timestamp": datetime.utcnow().isoformat(),
        "tuned": tune,
    }

    # Load configuration
    paths_config = load_config("configs/paths.yaml")
    hyperparams_config = load_config("configs/hyperparameters.yaml")

    # Step 1: Data ingestion
    logger.info("Step 1: Data ingestion")
    train_path = paths_config["data"]["train_data"]
    val_path = paths_config["data"]["validation_data"]
    test_path = paths_config["data"]["test_data"]

    # Check if data exists, if not run ingestion
    if not os.path.exists(train_path):
        logger.info("Processed data not found, running ingestion pipeline...")
        run_ingestion_pipeline()

    # Load data
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    logger.info(f"Loaded train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    # Step 2: Data cleaning
    logger.info("Step 2: Data cleaning")
    train_df, cleaning_metadata = run_cleaning_pipeline(train_df)
    val_df, _ = run_cleaning_pipeline(val_df, validate=False)
    test_df, _ = run_cleaning_pipeline(test_df, validate=False)

    # Step 3: Feature engineering
    logger.info("Step 3: Feature engineering")
    fe_results = run_feature_engineering(
        train_df=train_df,
        validation_df=val_df,
        test_df=test_df,
        target_col="target",
    )

    X_train = fe_results["X_train"]
    y_train = fe_results["y_train"]
    X_val = fe_results["X_val"]
    y_val = fe_results["y_val"]
    X_test = fe_results["X_test"]
    y_test = fe_results["y_test"]
    pipeline = fe_results["pipeline"]

    logger.info(f"Features shape: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    # Step 4: Model training
    logger.info("Step 4: Model training")
    trainer = ModelTrainer(
        model_type=model_type,
        experiment_name=experiment_name,
    )

    if tune:
        # Run hyperparameter tuning
        train_results = trainer.tune_with_optuna(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            n_trials=n_trials,
        )
        results["best_params"] = train_results["best_params"]
    else:
        # Train baseline model
        train_results = trainer.train_baseline(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
        )

    results["training_metrics"] = train_results["metrics"]

    # Step 5: Model calibration
    logger.info("Step 5: Model calibration")
    calibration_config = hyperparams_config.get("calibration", {})
    trainer.calibrate_model(
        X_train=X_train,
        y_train=y_train,
        method=calibration_config.get("method", "isotonic"),
        cv=calibration_config.get("cv", 5),
    )

    # Step 6: Evaluate on test set
    logger.info("Step 6: Evaluating on test set")
    test_results = trainer.evaluate_on_test(X_test, y_test, use_calibrated=True)
    results["test_metrics"] = test_results["metrics"]
    results["calibration_metrics"] = test_results["calibration"]

    logger.info(f"Test AUC: {test_results['metrics']['roc_auc']:.4f}")
    logger.info(f"Test KS: {test_results['metrics']['ks_statistic']:.4f}")

    # Step 7: Score distribution analysis
    logger.info("Step 7: Score distribution analysis")
    scorecard = CreditScorecard()
    model = trainer.get_model(calibrated=True)
    y_prob_test = model.predict_proba(X_test)[:, 1]
    score_stats = scorecard.get_score_distribution_stats(y_prob_test)
    results["score_distribution"] = score_stats

    logger.info(f"Approval rate: {score_stats['band_rates'].get('low_risk', 0):.2%}")
    logger.info(f"Decline rate: {score_stats['band_rates'].get('high_risk', 0):.2%}")

    # Step 8: Register model
    if register:
        logger.info("Step 8: Registering model")
        model_uri = trainer.register_model(
            model_name="credit_risk_model",
            stage="Staging",
            pipeline=pipeline,
        )
        results["model_uri"] = model_uri

    # Save training report
    report_dir = "reports/training"
    os.makedirs(report_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(report_dir, f"training_report_{timestamp}.json")

    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Training report saved to {report_path}")
    logger.info("Training pipeline completed successfully")

    return results


def main():
    """Main entry point for training pipeline."""
    parser = argparse.ArgumentParser(description="Credit Risk Model Training Pipeline")
    parser.add_argument(
        "--model",
        type=str,
        default="lightgbm",
        choices=["logistic_regression", "xgboost", "lightgbm"],
        help="Model type to train",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run hyperparameter tuning with Optuna",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Train baseline model without tuning",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials",
    )
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="Don't register the model",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="credit_risk_scoring",
        help="MLflow experiment name",
    )

    args = parser.parse_args()

    tune = args.tune or not args.baseline

    run_training_pipeline(
        model_type=args.model,
        tune=tune,
        n_trials=args.n_trials,
        register=not args.no_register,
        experiment_name=args.experiment,
    )


if __name__ == "__main__":
    main()
