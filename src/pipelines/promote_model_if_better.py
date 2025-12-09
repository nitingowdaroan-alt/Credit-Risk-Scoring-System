"""
Model Promotion Script for Credit Risk Scoring System.

This module handles automated model promotion from Staging to Production
based on performance comparison and governance rules.
"""

import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import mlflow
import yaml
from loguru import logger


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class ModelPromoter:
    """Handles model promotion logic and governance checks."""

    def __init__(self):
        """Initialize the model promoter."""
        self.config = load_config("configs/monitoring.yaml")
        self.risk_policy = load_config("configs/risk_policy.yaml")
        self.governance = self.risk_policy.get("governance", {})

        # Setup MLflow
        os.makedirs("mlruns", exist_ok=True)
        mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
        self.client = mlflow.tracking.MlflowClient()

    def get_staging_model(self, model_name: str) -> Optional[Any]:
        """Get the current staging model.

        Args:
            model_name: Name of the registered model.

        Returns:
            Model version info or None.
        """
        try:
            versions = self.client.get_latest_versions(model_name, stages=["Staging"])
            return versions[0] if versions else None
        except mlflow.exceptions.MlflowException as e:
            logger.error(f"Error getting staging model: {e}")
            return None

    def get_production_model(self, model_name: str) -> Optional[Any]:
        """Get the current production model.

        Args:
            model_name: Name of the registered model.

        Returns:
            Model version info or None.
        """
        try:
            versions = self.client.get_latest_versions(model_name, stages=["Production"])
            return versions[0] if versions else None
        except mlflow.exceptions.MlflowException as e:
            logger.debug(f"No production model found: {e}")
            return None

    def get_model_metrics(self, version: Any) -> Dict[str, float]:
        """Get metrics for a model version.

        Args:
            version: Model version object.

        Returns:
            Dictionary of metrics.
        """
        try:
            run = self.client.get_run(version.run_id)
            return run.data.metrics
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return {}

    def check_governance_requirements(
        self, metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Check if model meets governance requirements.

        Args:
            metrics: Model metrics.

        Returns:
            Dictionary with check results.
        """
        checks = []
        passed = True

        # Check minimum AUC
        min_auc = self.governance.get("min_auc", 0.70)
        auc = metrics.get("roc_auc") or metrics.get("val_roc_auc") or metrics.get("test_roc_auc", 0)

        auc_check = {
            "check": "minimum_auc",
            "threshold": min_auc,
            "value": auc,
            "passed": auc >= min_auc,
        }
        checks.append(auc_check)
        if not auc_check["passed"]:
            passed = False

        # Check minimum KS statistic
        min_ks = self.governance.get("min_ks_statistic", 0.30)
        ks = metrics.get("ks_statistic") or metrics.get("val_ks_statistic") or metrics.get("test_ks_statistic", 0)

        ks_check = {
            "check": "minimum_ks_statistic",
            "threshold": min_ks,
            "value": ks,
            "passed": ks >= min_ks,
        }
        checks.append(ks_check)
        if not ks_check["passed"]:
            passed = False

        # Check maximum Brier score
        max_brier = self.governance.get("max_brier_score", 0.25)
        brier = metrics.get("brier_score") or metrics.get("val_brier_score") or metrics.get("test_brier_score", 1.0)

        brier_check = {
            "check": "maximum_brier_score",
            "threshold": max_brier,
            "value": brier,
            "passed": brier <= max_brier,
        }
        checks.append(brier_check)
        if not brier_check["passed"]:
            passed = False

        return {"passed": passed, "checks": checks}

    def compare_with_production(
        self,
        staging_metrics: Dict[str, float],
        production_metrics: Dict[str, float],
    ) -> Dict[str, Any]:
        """Compare staging model with production model.

        Args:
            staging_metrics: Staging model metrics.
            production_metrics: Production model metrics.

        Returns:
            Comparison results.
        """
        comparison = {"improvements": {}, "degradations": {}, "unchanged": {}}

        # Metrics where higher is better
        higher_better = ["roc_auc", "val_roc_auc", "test_roc_auc", "ks_statistic", "precision", "recall"]
        # Metrics where lower is better
        lower_better = ["brier_score", "log_loss", "expected_calibration_error"]

        all_metrics = set(staging_metrics.keys()) | set(production_metrics.keys())

        for metric in all_metrics:
            staging_val = staging_metrics.get(metric)
            production_val = production_metrics.get(metric)

            if staging_val is None or production_val is None:
                continue

            diff = staging_val - production_val

            # Determine if this is an improvement
            is_improvement = False
            if any(m in metric for m in ["auc", "ks", "precision", "recall", "f1"]):
                is_improvement = diff > 0
            elif any(m in metric for m in ["brier", "log_loss", "calibration"]):
                is_improvement = diff < 0

            if abs(diff) < 0.001:  # Effectively unchanged
                comparison["unchanged"][metric] = {
                    "staging": staging_val,
                    "production": production_val,
                }
            elif is_improvement:
                comparison["improvements"][metric] = {
                    "staging": staging_val,
                    "production": production_val,
                    "change": diff,
                }
            else:
                comparison["degradations"][metric] = {
                    "staging": staging_val,
                    "production": production_val,
                    "change": diff,
                }

        # Determine overall recommendation
        has_auc_improvement = any("auc" in k for k in comparison["improvements"])
        has_auc_degradation = any("auc" in k for k in comparison["degradations"])

        # Get the main AUC values
        staging_auc = staging_metrics.get("roc_auc") or staging_metrics.get("val_roc_auc", 0)
        production_auc = production_metrics.get("roc_auc") or production_metrics.get("val_roc_auc", 0)
        auc_diff = staging_auc - production_auc

        if auc_diff > 0.005:  # 0.5% improvement
            comparison["recommendation"] = "promote"
            comparison["reason"] = f"AUC improved by {auc_diff:.4f}"
        elif auc_diff < -0.01:  # 1% degradation
            comparison["recommendation"] = "reject"
            comparison["reason"] = f"AUC degraded by {abs(auc_diff):.4f}"
        else:
            comparison["recommendation"] = "review"
            comparison["reason"] = "Changes are within tolerance, manual review recommended"

        return comparison

    def promote_to_production(
        self, model_name: str, version: int, archive_previous: bool = True
    ) -> bool:
        """Promote a model version to production.

        Args:
            model_name: Name of the registered model.
            version: Version number to promote.
            archive_previous: Whether to archive the previous production model.

        Returns:
            True if promotion successful.
        """
        try:
            # Archive previous production model
            if archive_previous:
                current_prod = self.get_production_model(model_name)
                if current_prod:
                    logger.info(f"Archiving previous production model (version {current_prod.version})")
                    self.client.transition_model_version_stage(
                        name=model_name,
                        version=current_prod.version,
                        stage="Archived",
                    )

            # Promote new version to production
            logger.info(f"Promoting model version {version} to Production")
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production",
            )

            return True

        except Exception as e:
            logger.error(f"Error promoting model: {e}")
            return False

    def evaluate_and_promote(
        self,
        model_name: str = "credit_risk_model",
        auto_promote: bool = False,
        require_approval: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate staging model and potentially promote to production.

        Args:
            model_name: Name of the registered model.
            auto_promote: Automatically promote if criteria met.
            require_approval: Require human approval even if criteria met.

        Returns:
            Evaluation and promotion results.
        """
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "model_name": model_name,
            "auto_promote": auto_promote,
        }

        # Get staging model
        staging_model = self.get_staging_model(model_name)
        if not staging_model:
            logger.warning("No staging model found")
            results["status"] = "no_staging_model"
            return results

        staging_metrics = self.get_model_metrics(staging_model)
        results["staging_version"] = staging_model.version
        results["staging_metrics"] = staging_metrics

        # Check governance requirements
        governance_check = self.check_governance_requirements(staging_metrics)
        results["governance_check"] = governance_check

        if not governance_check["passed"]:
            logger.warning("Model does not meet governance requirements")
            results["status"] = "governance_failed"
            results["recommendation"] = "reject"
            return results

        # Get production model for comparison
        production_model = self.get_production_model(model_name)

        if production_model:
            production_metrics = self.get_model_metrics(production_model)
            results["production_version"] = production_model.version
            results["production_metrics"] = production_metrics

            # Compare models
            comparison = self.compare_with_production(staging_metrics, production_metrics)
            results["comparison"] = comparison
            results["recommendation"] = comparison["recommendation"]

            logger.info(f"Comparison recommendation: {comparison['recommendation']}")
            logger.info(f"Reason: {comparison['reason']}")

        else:
            # No production model, promote if governance passed
            results["recommendation"] = "promote"
            results["reason"] = "No existing production model, promoting first model"
            logger.info("No production model exists, recommending promotion")

        # Perform promotion if appropriate
        should_promote = (
            results["recommendation"] == "promote"
            and auto_promote
            and not require_approval
        )

        if should_promote:
            success = self.promote_to_production(
                model_name=model_name,
                version=staging_model.version,
            )
            results["promoted"] = success
            results["status"] = "promoted" if success else "promotion_failed"
        else:
            results["promoted"] = False
            if require_approval and results["recommendation"] == "promote":
                results["status"] = "awaiting_approval"
                logger.info("Model ready for promotion, awaiting human approval")
            else:
                results["status"] = results["recommendation"]

        # Save results
        report_dir = "reports/training"
        os.makedirs(report_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(report_dir, f"promotion_report_{timestamp}.json")

        with open(report_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Promotion report saved to {report_path}")

        return results


def main():
    """Main entry point for model promotion."""
    parser = argparse.ArgumentParser(description="Model Promotion for Credit Risk System")
    parser.add_argument(
        "--model-name",
        type=str,
        default="credit_risk_model",
        help="Name of the registered model",
    )
    parser.add_argument(
        "--auto-promote",
        action="store_true",
        help="Automatically promote if criteria met",
    )
    parser.add_argument(
        "--no-approval",
        action="store_true",
        help="Skip human approval requirement",
    )
    parser.add_argument(
        "--force-promote",
        action="store_true",
        help="Force promotion regardless of comparison",
    )

    args = parser.parse_args()

    promoter = ModelPromoter()

    if args.force_promote:
        staging = promoter.get_staging_model(args.model_name)
        if staging:
            success = promoter.promote_to_production(args.model_name, staging.version)
            print(f"Force promotion: {'success' if success else 'failed'}")
        else:
            print("No staging model to promote")
    else:
        results = promoter.evaluate_and_promote(
            model_name=args.model_name,
            auto_promote=args.auto_promote,
            require_approval=not args.no_approval,
        )

        print(f"\nPromotion Status: {results.get('status')}")
        print(f"Recommendation: {results.get('recommendation')}")

        if "comparison" in results:
            print(f"Reason: {results['comparison'].get('reason')}")


if __name__ == "__main__":
    main()
