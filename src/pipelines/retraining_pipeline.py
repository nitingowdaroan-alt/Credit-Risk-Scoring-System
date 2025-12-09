"""
Retraining Pipeline for Credit Risk Scoring System.

This module handles automated retraining when drift or performance
degradation is detected.
"""

import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

import mlflow
import yaml
from loguru import logger

from src.pipelines.training_pipeline import run_training_pipeline


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def check_retraining_trigger() -> bool:
    """Check if retraining has been triggered.

    Returns:
        True if retraining trigger file exists.
    """
    trigger_file = "data/processed/retrain_trigger.flag"
    return os.path.exists(trigger_file)


def clear_retraining_trigger() -> None:
    """Remove the retraining trigger file."""
    trigger_file = "data/processed/retrain_trigger.flag"
    if os.path.exists(trigger_file):
        os.remove(trigger_file)
        logger.info("Retraining trigger cleared")


def get_production_model_metrics() -> Optional[Dict[str, float]]:
    """Get metrics of current production model.

    Returns:
        Dictionary of metrics or None if no production model.
    """
    try:
        client = mlflow.tracking.MlflowClient()

        # Get production model version
        model_name = "credit_risk_model"

        try:
            versions = client.get_latest_versions(model_name, stages=["Production"])
            if not versions:
                return None

            version = versions[0]
            run_id = version.run_id

            # Get run metrics
            run = client.get_run(run_id)
            return run.data.metrics

        except mlflow.exceptions.MlflowException:
            return None

    except Exception as e:
        logger.error(f"Error getting production model metrics: {e}")
        return None


def compare_models(
    candidate_metrics: Dict[str, float],
    production_metrics: Dict[str, float],
    config: Dict,
) -> Dict[str, Any]:
    """Compare candidate model with production model.

    Args:
        candidate_metrics: Metrics of candidate model.
        production_metrics: Metrics of production model.
        config: Monitoring configuration.

    Returns:
        Comparison results with recommendation.
    """
    governance_config = load_config("configs/risk_policy.yaml").get("governance", {})

    comparison = {
        "candidate_metrics": candidate_metrics,
        "production_metrics": production_metrics,
        "improvements": {},
        "degradations": {},
    }

    # Compare key metrics
    key_metrics = ["roc_auc", "val_roc_auc", "ks_statistic", "brier_score"]

    for metric in key_metrics:
        candidate_val = candidate_metrics.get(metric) or candidate_metrics.get(f"test_{metric}")
        production_val = production_metrics.get(metric) or production_metrics.get(f"test_{metric}")

        if candidate_val is None or production_val is None:
            continue

        diff = candidate_val - production_val

        # For AUC and KS, higher is better
        # For brier_score, lower is better
        if metric in ["brier_score", "log_loss"]:
            diff = -diff  # Invert so positive = improvement

        if diff > 0:
            comparison["improvements"][metric] = {
                "candidate": candidate_val,
                "production": production_val,
                "improvement": abs(diff),
            }
        elif diff < 0:
            comparison["degradations"][metric] = {
                "candidate": candidate_val,
                "production": production_val,
                "degradation": abs(diff),
            }

    # Make recommendation
    min_improvement_margin = 0.005  # 0.5% improvement required
    max_degradation_tolerance = 0.01  # 1% degradation tolerated

    auc_improvement = comparison["improvements"].get("roc_auc", {}).get("improvement", 0)
    auc_degradation = comparison["degradations"].get("roc_auc", {}).get("degradation", 0)

    # Check minimum AUC requirement
    candidate_auc = candidate_metrics.get("roc_auc") or candidate_metrics.get("val_roc_auc", 0)
    min_auc = governance_config.get("min_auc", 0.70)

    if candidate_auc < min_auc:
        comparison["recommendation"] = "reject"
        comparison["reason"] = f"Candidate AUC ({candidate_auc:.4f}) below minimum ({min_auc})"
    elif auc_degradation > max_degradation_tolerance:
        comparison["recommendation"] = "reject"
        comparison["reason"] = f"AUC degradation ({auc_degradation:.4f}) exceeds tolerance"
    elif auc_improvement >= min_improvement_margin:
        comparison["recommendation"] = "promote"
        comparison["reason"] = f"AUC improved by {auc_improvement:.4f}"
    else:
        comparison["recommendation"] = "review"
        comparison["reason"] = "Marginal changes, manual review recommended"

    return comparison


def run_retraining_pipeline(
    force: bool = False,
    model_type: str = "lightgbm",
    n_trials: int = 50,
) -> Dict[str, Any]:
    """Run the retraining pipeline.

    Args:
        force: Force retraining even without trigger.
        model_type: Type of model to train.
        n_trials: Number of Optuna trials.

    Returns:
        Retraining results.
    """
    logger.info("Starting retraining pipeline")

    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "trigger_detected": check_retraining_trigger(),
        "forced": force,
    }

    # Check if retraining is needed
    if not force and not check_retraining_trigger():
        logger.info("No retraining trigger detected and force=False, skipping")
        results["status"] = "skipped"
        results["reason"] = "No trigger detected"
        return results

    # Setup MLflow
    os.makedirs("mlruns", exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")

    # Get current production model metrics
    production_metrics = get_production_model_metrics()
    results["production_metrics"] = production_metrics

    if production_metrics:
        logger.info(f"Current production model AUC: {production_metrics.get('roc_auc', 'N/A')}")

    # Run training pipeline
    logger.info("Running training pipeline...")
    training_results = run_training_pipeline(
        model_type=model_type,
        tune=True,
        n_trials=n_trials,
        register=True,
        experiment_name="credit_risk_retraining",
    )

    results["training_results"] = training_results

    # Compare with production model if exists
    if production_metrics:
        candidate_metrics = training_results.get("test_metrics", {})
        comparison = compare_models(
            candidate_metrics=candidate_metrics,
            production_metrics=production_metrics,
            config=load_config("configs/monitoring.yaml"),
        )
        results["model_comparison"] = comparison

        logger.info(f"Model comparison recommendation: {comparison['recommendation']}")
        logger.info(f"Reason: {comparison['reason']}")
    else:
        logger.info("No production model to compare, will promote candidate")
        results["model_comparison"] = {
            "recommendation": "promote",
            "reason": "No existing production model",
        }

    # Clear retraining trigger
    if check_retraining_trigger():
        clear_retraining_trigger()

    # Save results
    report_dir = "reports/training"
    os.makedirs(report_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(report_dir, f"retraining_report_{timestamp}.json")

    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Retraining report saved to {report_path}")

    results["status"] = "completed"
    return results


def main():
    """Main entry point for retraining pipeline."""
    parser = argparse.ArgumentParser(description="Credit Risk Model Retraining Pipeline")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining even without trigger",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lightgbm",
        choices=["logistic_regression", "xgboost", "lightgbm"],
        help="Model type to train",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials",
    )

    args = parser.parse_args()

    run_retraining_pipeline(
        force=args.force,
        model_type=args.model,
        n_trials=args.n_trials,
    )


if __name__ == "__main__":
    main()
