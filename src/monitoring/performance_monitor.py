"""
Performance Monitoring Module for Credit Risk Scoring System.

This module monitors model performance over time, tracks metric trends,
and detects performance degradation.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def load_monitoring_config(config_path: str = "configs/monitoring.yaml") -> Dict:
    """Load monitoring configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Kolmogorov-Smirnov statistic."""
    pos_probs = y_prob[y_true == 1]
    neg_probs = y_prob[y_true == 0]

    if len(pos_probs) == 0 or len(neg_probs) == 0:
        return 0.0

    bins = np.linspace(0, 1, 100)
    pos_cdf = np.array([np.mean(pos_probs <= b) for b in bins])
    neg_cdf = np.array([np.mean(neg_probs <= b) for b in bins])

    return float(np.max(np.abs(pos_cdf - neg_cdf)))


def compute_expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error (ECE)."""
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )

    bin_counts = np.histogram(y_prob, bins=n_bins, range=(0, 1))[0]
    ece = np.sum(
        bin_counts / len(y_prob) * np.abs(fraction_of_positives - mean_predicted_value)
    )

    return float(ece)


class PerformanceMonitor:
    """Monitor model performance over time."""

    def __init__(self, config_path: str = "configs/monitoring.yaml"):
        """Initialize the performance monitor.

        Args:
            config_path: Path to configuration file.
        """
        self.config = load_monitoring_config(config_path)
        self.performance_config = self.config.get("performance", {})
        self.metrics_history: List[Dict] = []
        self.alerts: List[Dict] = []

    def compute_metrics(
        self, y_true: np.ndarray, y_prob: np.ndarray, y_pred: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Compute comprehensive performance metrics.

        Args:
            y_true: True labels.
            y_prob: Predicted probabilities.
            y_pred: Predicted labels (optional).

        Returns:
            Dictionary of metrics.
        """
        if y_pred is None:
            y_pred = (y_prob >= 0.5).astype(int)

        metrics = {
            "roc_auc": roc_auc_score(y_true, y_prob),
            "log_loss": log_loss(y_true, y_prob),
            "brier_score": brier_score_loss(y_true, y_prob),
            "ks_statistic": compute_ks_statistic(y_true, y_prob),
            "expected_calibration_error": compute_expected_calibration_error(y_true, y_prob),
            "default_rate": float(np.mean(y_true)),
            "predicted_default_rate": float(np.mean(y_prob)),
            "n_samples": len(y_true),
            "n_defaults": int(np.sum(y_true)),
        }

        return metrics

    def check_thresholds(self, metrics: Dict[str, float]) -> List[Dict]:
        """Check if metrics exceed configured thresholds.

        Args:
            metrics: Dictionary of computed metrics.

        Returns:
            List of threshold violations.
        """
        violations = []

        # Check each configured threshold
        threshold_checks = [
            ("roc_auc", "auc", "less_than"),
            ("log_loss", "log_loss", "greater_than"),
            ("brier_score", "brier_score", "greater_than"),
            ("ks_statistic", "ks_statistic", "less_than"),
            ("expected_calibration_error", "calibration_error", "greater_than"),
        ]

        for metric_name, config_key, comparison in threshold_checks:
            config = self.performance_config.get(config_key, {})
            warning_threshold = config.get("warning_threshold")
            critical_threshold = config.get("critical_threshold")

            value = metrics.get(metric_name)
            if value is None:
                continue

            if comparison == "less_than":
                if critical_threshold and value < critical_threshold:
                    violations.append(
                        {
                            "metric": metric_name,
                            "level": "critical",
                            "value": value,
                            "threshold": critical_threshold,
                            "message": f"{metric_name} ({value:.4f}) below critical threshold ({critical_threshold})",
                        }
                    )
                elif warning_threshold and value < warning_threshold:
                    violations.append(
                        {
                            "metric": metric_name,
                            "level": "warning",
                            "value": value,
                            "threshold": warning_threshold,
                            "message": f"{metric_name} ({value:.4f}) below warning threshold ({warning_threshold})",
                        }
                    )
            else:  # greater_than
                if critical_threshold and value > critical_threshold:
                    violations.append(
                        {
                            "metric": metric_name,
                            "level": "critical",
                            "value": value,
                            "threshold": critical_threshold,
                            "message": f"{metric_name} ({value:.4f}) exceeds critical threshold ({critical_threshold})",
                        }
                    )
                elif warning_threshold and value > warning_threshold:
                    violations.append(
                        {
                            "metric": metric_name,
                            "level": "warning",
                            "value": value,
                            "threshold": warning_threshold,
                            "message": f"{metric_name} ({value:.4f}) exceeds warning threshold ({warning_threshold})",
                        }
                    )

        return violations

    def monitor_risk_bands(
        self,
        y_true: np.ndarray,
        risk_bands: np.ndarray,
    ) -> Dict[str, Any]:
        """Monitor default rates by risk band.

        Args:
            y_true: True labels.
            risk_bands: Risk band assignments.

        Returns:
            Dictionary with risk band metrics.
        """
        band_config = self.config.get("risk_band_monitoring", {}).get("expected_rates", {})

        results = {}
        violations = []

        for band in ["low_risk", "medium_risk", "high_risk"]:
            mask = risk_bands == band
            if not np.any(mask):
                continue

            actual_rate = float(np.mean(y_true[mask]))
            expected_config = band_config.get(band, {})
            expected_rate = expected_config.get("expected", 0.1)
            tolerance = expected_config.get("tolerance", 0.05)

            deviation = abs(actual_rate - expected_rate)

            results[band] = {
                "actual_default_rate": actual_rate,
                "expected_default_rate": expected_rate,
                "deviation": deviation,
                "n_samples": int(np.sum(mask)),
                "n_defaults": int(np.sum(y_true[mask])),
            }

            if deviation > tolerance:
                violations.append(
                    {
                        "band": band,
                        "actual": actual_rate,
                        "expected": expected_rate,
                        "deviation": deviation,
                        "message": f"{band} default rate ({actual_rate:.2%}) deviates from expected ({expected_rate:.2%})",
                    }
                )

        results["violations"] = violations

        return results

    def update_history(
        self,
        metrics: Dict[str, float],
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Add metrics to history.

        Args:
            metrics: Computed metrics.
            timestamp: Timestamp for metrics.
            metadata: Additional metadata.
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        entry = {
            "timestamp": timestamp.isoformat(),
            "metrics": metrics,
            "metadata": metadata or {},
        }

        self.metrics_history.append(entry)

    def get_trend_analysis(
        self, metric_name: str, window_size: int = 7
    ) -> Dict[str, Any]:
        """Analyze trend for a specific metric.

        Args:
            metric_name: Name of metric to analyze.
            window_size: Number of recent observations.

        Returns:
            Dictionary with trend analysis.
        """
        if len(self.metrics_history) < 2:
            return {"status": "insufficient_data"}

        # Get recent values
        values = []
        timestamps = []

        for entry in self.metrics_history[-window_size:]:
            if metric_name in entry.get("metrics", {}):
                values.append(entry["metrics"][metric_name])
                timestamps.append(entry["timestamp"])

        if len(values) < 2:
            return {"status": "insufficient_data"}

        values = np.array(values)

        # Compute trend
        slope = np.polyfit(range(len(values)), values, 1)[0]

        return {
            "metric": metric_name,
            "current_value": float(values[-1]),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "trend_slope": float(slope),
            "trend_direction": "improving" if slope > 0 else "degrading",
            "n_observations": len(values),
        }

    def save_metrics(self, output_path: str = "reports/training/performance_history.json") -> None:
        """Save metrics history to file.

        Args:
            output_path: Path to save metrics.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self.metrics_history, f, indent=2)

        logger.info(f"Metrics history saved to {output_path}")

    def load_metrics(self, input_path: str = "reports/training/performance_history.json") -> None:
        """Load metrics history from file.

        Args:
            input_path: Path to load metrics from.
        """
        if os.path.exists(input_path):
            with open(input_path, "r") as f:
                self.metrics_history = json.load(f)
            logger.info(f"Loaded {len(self.metrics_history)} historical records")

    def generate_report(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Generate a comprehensive performance report.

        Args:
            metrics: Current metrics.

        Returns:
            Dictionary with full performance report.
        """
        violations = self.check_thresholds(metrics)

        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "current_metrics": metrics,
            "threshold_violations": violations,
            "has_critical_violations": any(v["level"] == "critical" for v in violations),
            "has_warnings": any(v["level"] == "warning" for v in violations),
        }

        # Add trend analysis for key metrics
        key_metrics = ["roc_auc", "ks_statistic", "brier_score"]
        report["trends"] = {}
        for metric in key_metrics:
            report["trends"][metric] = self.get_trend_analysis(metric)

        return report


def run_performance_monitoring(
    predictions_path: str = "data/processed/predictions_with_outcomes.csv",
    output_dir: str = "reports/training",
) -> Dict[str, Any]:
    """Run the performance monitoring pipeline.

    Args:
        predictions_path: Path to predictions with realized outcomes.
        output_dir: Directory for output reports.

    Returns:
        Performance monitoring results.
    """
    logger.info("Starting performance monitoring pipeline")

    # Initialize monitor
    monitor = PerformanceMonitor()

    # Load historical metrics
    history_path = os.path.join(output_dir, "performance_history.json")
    monitor.load_metrics(history_path)

    # Load predictions with outcomes
    if os.path.exists(predictions_path):
        df = pd.read_csv(predictions_path)
    else:
        # Use test data as proxy
        test_path = "data/processed/test.csv"
        if not os.path.exists(test_path):
            logger.error("No data available for performance monitoring")
            return {"error": "No data available"}

        df = pd.read_csv(test_path)

        # Simulate predictions (in production, these would be stored)
        logger.warning("Using test data for performance monitoring demo")

    # Check required columns
    if "target" not in df.columns:
        logger.error("Target column not found in data")
        return {"error": "Target column not found"}

    y_true = df["target"].values

    # Use predicted probabilities if available, otherwise simulate
    if "predicted_probability" in df.columns:
        y_prob = df["predicted_probability"].values
    else:
        # Simulate probabilities for demo
        np.random.seed(42)
        y_prob = np.clip(y_true * 0.6 + np.random.beta(2, 5, len(y_true)), 0, 1)

    # Compute metrics
    metrics = monitor.compute_metrics(y_true, y_prob)
    logger.info(f"Computed metrics: AUC={metrics['roc_auc']:.4f}, KS={metrics['ks_statistic']:.4f}")

    # Update history
    monitor.update_history(metrics)

    # Generate report
    report = monitor.generate_report(metrics)

    # Check for violations
    if report["has_critical_violations"]:
        logger.error("Critical performance violations detected!")
        for v in report["threshold_violations"]:
            if v["level"] == "critical":
                logger.error(f"  {v['message']}")

    elif report["has_warnings"]:
        logger.warning("Performance warnings detected:")
        for v in report["threshold_violations"]:
            logger.warning(f"  {v['message']}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"performance_report_{timestamp}.json")

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Save updated history
    monitor.save_metrics(history_path)

    logger.info(f"Performance report saved to {report_path}")

    return report


if __name__ == "__main__":
    run_performance_monitoring()
