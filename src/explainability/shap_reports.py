"""
SHAP Report Generation Script for Credit Risk Scoring System.

This module generates periodic SHAP reports for model explainability.
"""

import argparse
import os
from datetime import datetime

import mlflow
import numpy as np
import pandas as pd
import yaml
from loguru import logger

from src.data.feature_engineering import get_feature_lists, run_feature_engineering
from src.explainability.shap_explainer import (
    generate_explainability_report,
    generate_regulator_report,
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model_from_mlflow(model_name: str = "credit_risk_model", stage: str = "Production"):
    """Load model from MLflow registry.

    Args:
        model_name: Name of registered model.
        stage: Model stage to load.

    Returns:
        Loaded model or None.
    """
    try:
        os.makedirs("mlruns", exist_ok=True)
        mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")

        model_uri = f"models:/{model_name}/{stage}"
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Loaded model from {model_uri}")
        return model

    except Exception as e:
        logger.warning(f"Could not load {stage} model: {e}")

        # Try loading from latest run
        try:
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
                    model = mlflow.sklearn.load_model(model_uri)
                    logger.info(f"Loaded model from run {run.info.run_id}")
                    return model

        except Exception as e2:
            logger.error(f"Could not load model from runs: {e2}")

    return None


def run_shap_reports(
    output_dir: str = "reports/explainability",
    generate_regulator: bool = True,
) -> dict:
    """Generate SHAP explainability reports.

    Args:
        output_dir: Directory for output reports.
        generate_regulator: Whether to generate regulator-style report.

    Returns:
        Dictionary with report paths and info.
    """
    logger.info("Starting SHAP report generation")

    paths_config = load_config("configs/paths.yaml")
    results = {"timestamp": datetime.utcnow().isoformat()}

    # Load data
    train_path = paths_config["data"]["train_data"]
    test_path = paths_config["data"]["test_data"]

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        logger.error("Training/test data not found. Run data pipeline first.")
        return {"error": "Data not found"}

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Run feature engineering
    fe_results = run_feature_engineering(
        train_df=train_df,
        test_df=test_df,
        target_col="target",
    )

    X_train = fe_results["X_train"]
    X_test = fe_results["X_test"]
    y_test = fe_results["y_test"]

    # Get feature names (approximation based on original features)
    feature_lists = get_feature_lists(train_df.drop(columns=["target"], errors="ignore"))
    feature_names = feature_lists["numeric"] + feature_lists["categorical"]

    # Load model
    model = load_model_from_mlflow()

    if model is None:
        logger.warning("No model available, creating demo model for report generation")
        from sklearn.ensemble import GradientBoostingClassifier

        model = GradientBoostingClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, fe_results["y_train"])

    # Generate explainability report
    try:
        report = generate_explainability_report(
            model=model,
            X_train=X_train,
            X_test=X_test,
            feature_names=None,  # Will use generic names
            output_dir=output_dir,
            n_samples_explain=100,
        )
        results["explainability_report"] = report

    except Exception as e:
        logger.error(f"Error generating explainability report: {e}")
        results["explainability_error"] = str(e)

    # Generate regulator report
    if generate_regulator:
        try:
            regulator_path = generate_regulator_report(
                model=model,
                X_train=X_train,
                X_test=X_test,
                y_test=y_test.values if hasattr(y_test, "values") else y_test,
                feature_names=None,
                output_dir=output_dir,
                model_description="Credit risk scoring model for predicting probability of default on retail loans.",
            )
            results["regulator_report"] = regulator_path

        except Exception as e:
            logger.error(f"Error generating regulator report: {e}")
            results["regulator_error"] = str(e)

    logger.info("SHAP report generation completed")
    return results


def main():
    """Main entry point for SHAP report generation."""
    parser = argparse.ArgumentParser(description="Generate SHAP Explainability Reports")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/explainability",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--no-regulator",
        action="store_true",
        help="Skip regulator report generation",
    )

    args = parser.parse_args()

    run_shap_reports(
        output_dir=args.output_dir,
        generate_regulator=not args.no_regulator,
    )


if __name__ == "__main__":
    main()
