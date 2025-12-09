"""
Data Drift Monitoring Module for Credit Risk Scoring System.

This module monitors data drift between training and production data
using PSI, KS statistics, and Evidently AI.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from scipy import stats

try:
    from evidently import ColumnMapping
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    logger.warning("Evidently not available. Using custom drift detection.")


def load_monitoring_config(config_path: str = "configs/monitoring.yaml") -> Dict:
    """Load monitoring configuration.

    Args:
        config_path: Path to configuration file.

    Returns:
        Dictionary with monitoring configuration.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """Compute Population Stability Index (PSI).

    PSI measures the shift in distribution between two datasets.
    - PSI < 0.1: No significant change
    - 0.1 <= PSI < 0.25: Moderate change
    - PSI >= 0.25: Significant change

    Args:
        expected: Expected (training) distribution.
        actual: Actual (production) distribution.
        n_bins: Number of bins for discretization.
        eps: Small value to avoid division by zero.

    Returns:
        PSI value.
    """
    # Create bins based on expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)

    # Compute bin frequencies
    expected_bins = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_bins = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    # Add epsilon to avoid log(0)
    expected_bins = np.clip(expected_bins, eps, 1)
    actual_bins = np.clip(actual_bins, eps, 1)

    # Compute PSI
    psi = np.sum((actual_bins - expected_bins) * np.log(actual_bins / expected_bins))

    return psi


def compute_ks_statistic(
    reference: np.ndarray, current: np.ndarray
) -> Tuple[float, float]:
    """Compute Kolmogorov-Smirnov statistic and p-value.

    Args:
        reference: Reference (training) distribution.
        current: Current (production) distribution.

    Returns:
        Tuple of (KS statistic, p-value).
    """
    ks_stat, p_value = stats.ks_2samp(reference, current)
    return ks_stat, p_value


def compute_js_divergence(
    p: np.ndarray, q: np.ndarray, n_bins: int = 10
) -> float:
    """Compute Jensen-Shannon divergence.

    Args:
        p: First distribution.
        q: Second distribution.
        n_bins: Number of bins.

    Returns:
        JS divergence value.
    """
    # Create common bins
    all_data = np.concatenate([p, q])
    bins = np.histogram_bin_edges(all_data, bins=n_bins)

    # Compute histograms
    p_hist = np.histogram(p, bins=bins)[0] / len(p)
    q_hist = np.histogram(q, bins=bins)[0] / len(q)

    # Add small epsilon
    eps = 1e-10
    p_hist = p_hist + eps
    q_hist = q_hist + eps

    # Normalize
    p_hist = p_hist / p_hist.sum()
    q_hist = q_hist / q_hist.sum()

    # Compute JS divergence
    m = 0.5 * (p_hist + q_hist)
    js = 0.5 * stats.entropy(p_hist, m) + 0.5 * stats.entropy(q_hist, m)

    return js


class DriftMonitor:
    """Monitor data drift between reference and current data."""

    def __init__(self, config_path: str = "configs/monitoring.yaml"):
        """Initialize the drift monitor.

        Args:
            config_path: Path to configuration file.
        """
        self.config = load_monitoring_config(config_path)
        self.drift_config = self.config.get("data_drift", {})
        self.reference_data: Optional[pd.DataFrame] = None
        self.drift_results: Dict[str, Any] = {}

    def set_reference_data(
        self, data: pd.DataFrame, sample_size: Optional[int] = None
    ) -> None:
        """Set reference data for drift comparison.

        Args:
            data: Reference DataFrame (typically training data).
            sample_size: Optional sample size limit.
        """
        if sample_size and len(data) > sample_size:
            self.reference_data = data.sample(n=sample_size, random_state=42)
        else:
            self.reference_data = data.copy()

        logger.info(f"Reference data set with {len(self.reference_data)} samples")

    def compute_feature_drift(
        self, current_data: pd.DataFrame, feature: str
    ) -> Dict[str, Any]:
        """Compute drift metrics for a single feature.

        Args:
            current_data: Current production data.
            feature: Feature name.

        Returns:
            Dictionary with drift metrics.
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set")

        ref_values = self.reference_data[feature].dropna().values
        cur_values = current_data[feature].dropna().values

        # Skip if insufficient data
        if len(ref_values) < 10 or len(cur_values) < 10:
            return {"status": "insufficient_data"}

        # Numeric feature
        if np.issubdtype(self.reference_data[feature].dtype, np.number):
            psi = compute_psi(ref_values, cur_values)
            ks_stat, ks_pvalue = compute_ks_statistic(ref_values, cur_values)
            js_div = compute_js_divergence(ref_values, cur_values)

            # Determine drift status
            psi_threshold = self.drift_config.get("psi", {}).get("critical_threshold", 0.25)
            ks_threshold = self.drift_config.get("ks_statistic", {}).get(
                "critical_threshold", 0.2
            )

            is_drifted = psi > psi_threshold or ks_stat > ks_threshold

            return {
                "feature_type": "numeric",
                "psi": psi,
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_pvalue,
                "js_divergence": js_div,
                "is_drifted": is_drifted,
                "reference_mean": float(np.mean(ref_values)),
                "current_mean": float(np.mean(cur_values)),
                "reference_std": float(np.std(ref_values)),
                "current_std": float(np.std(cur_values)),
            }

        # Categorical feature
        else:
            ref_counts = pd.Series(ref_values).value_counts(normalize=True)
            cur_counts = pd.Series(cur_values).value_counts(normalize=True)

            # Align categories
            all_categories = set(ref_counts.index) | set(cur_counts.index)
            ref_aligned = pd.Series(
                [ref_counts.get(c, 0) for c in all_categories], index=all_categories
            )
            cur_aligned = pd.Series(
                [cur_counts.get(c, 0) for c in all_categories], index=all_categories
            )

            # Compute chi-square
            chi2, pvalue = stats.chisquare(
                cur_aligned.values + 1e-10, ref_aligned.values + 1e-10
            )

            return {
                "feature_type": "categorical",
                "chi2_statistic": chi2,
                "chi2_pvalue": pvalue,
                "is_drifted": pvalue < 0.05,
                "reference_distribution": ref_counts.to_dict(),
                "current_distribution": cur_counts.to_dict(),
            }

    def detect_drift(
        self,
        current_data: pd.DataFrame,
        features: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Detect drift across all features.

        Args:
            current_data: Current production data.
            features: List of features to check (all if None).

        Returns:
            Dictionary with drift detection results.
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set")

        if features is None:
            # Get common columns, excluding target
            features = [
                c
                for c in self.reference_data.columns
                if c in current_data.columns and c != "target"
            ]

        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "reference_samples": len(self.reference_data),
            "current_samples": len(current_data),
            "features_checked": len(features),
            "feature_drift": {},
        }

        drifted_features = []

        for feature in features:
            if feature in current_data.columns:
                drift_result = self.compute_feature_drift(current_data, feature)
                results["feature_drift"][feature] = drift_result

                if drift_result.get("is_drifted", False):
                    drifted_features.append(feature)

        # Overall drift assessment
        drift_ratio = len(drifted_features) / len(features) if features else 0

        results["summary"] = {
            "drifted_features": drifted_features,
            "drift_ratio": drift_ratio,
            "overall_drift_detected": drift_ratio > 0.2,  # >20% features drifted
        }

        self.drift_results = results

        logger.info(
            f"Drift detection complete: {len(drifted_features)}/{len(features)} "
            f"features drifted ({drift_ratio:.1%})"
        )

        return results

    def generate_evidently_report(
        self,
        current_data: pd.DataFrame,
        output_path: str = "reports/drift/drift_report.html",
    ) -> Optional[str]:
        """Generate drift report using Evidently AI.

        Args:
            current_data: Current production data.
            output_path: Path to save HTML report.

        Returns:
            Path to generated report or None if unavailable.
        """
        if not EVIDENTLY_AVAILABLE:
            logger.warning("Evidently not available, skipping report generation")
            return None

        if self.reference_data is None:
            raise ValueError("Reference data not set")

        # Create column mapping
        numeric_cols = self.reference_data.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        categorical_cols = self.reference_data.select_dtypes(
            exclude=[np.number]
        ).columns.tolist()

        # Remove target
        numeric_cols = [c for c in numeric_cols if c != "target"]
        categorical_cols = [c for c in categorical_cols if c != "target"]

        column_mapping = ColumnMapping(
            numerical_features=numeric_cols, categorical_features=categorical_cols
        )

        # Create report
        report = Report(metrics=[DataDriftPreset()])

        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=column_mapping,
        )

        # Save report
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        report.save_html(output_path)

        logger.info(f"Evidently drift report saved to {output_path}")

        return output_path

    def save_results(self, output_path: str = "reports/drift/drift_results.json") -> None:
        """Save drift detection results to JSON.

        Args:
            output_path: Path to save results.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self.drift_results, f, indent=2, default=str)

        logger.info(f"Drift results saved to {output_path}")

    def check_retraining_trigger(self) -> bool:
        """Check if drift exceeds retraining thresholds.

        Returns:
            True if retraining should be triggered.
        """
        if not self.drift_results:
            return False

        summary = self.drift_results.get("summary", {})

        # Check if overall drift detected
        if summary.get("overall_drift_detected", False):
            logger.warning("Drift detected - retraining may be needed")
            return True

        # Check individual thresholds
        retraining_config = self.config.get("retraining", {})
        triggers = retraining_config.get("triggers", [])

        for trigger in triggers:
            metric = trigger.get("metric")
            threshold = trigger.get("threshold")

            if metric == "psi":
                # Check average PSI
                psi_values = [
                    r.get("psi", 0)
                    for r in self.drift_results.get("feature_drift", {}).values()
                    if "psi" in r
                ]
                if psi_values and np.mean(psi_values) > threshold:
                    logger.warning(f"PSI threshold exceeded: {np.mean(psi_values):.3f}")
                    return True

        return False


def run_drift_monitoring(
    reference_path: str = "data/processed/train.csv",
    current_path: str = "data/processed/production_recent.csv",
    output_dir: str = "reports/drift",
) -> Dict[str, Any]:
    """Run the drift monitoring pipeline.

    Args:
        reference_path: Path to reference data.
        current_path: Path to current production data.
        output_dir: Directory for output reports.

    Returns:
        Drift detection results.
    """
    logger.info("Starting drift monitoring pipeline")

    # Load data
    if os.path.exists(reference_path):
        reference_data = pd.read_csv(reference_path)
    else:
        logger.error(f"Reference data not found: {reference_path}")
        return {"error": "Reference data not found"}

    if os.path.exists(current_path):
        current_data = pd.read_csv(current_path)
    else:
        # Use test data as proxy for current production
        test_path = "data/processed/test.csv"
        if os.path.exists(test_path):
            current_data = pd.read_csv(test_path)
            logger.warning(f"Using test data as proxy for production data")
        else:
            logger.error(f"No current data available")
            return {"error": "Current data not found"}

    # Initialize monitor
    monitor = DriftMonitor()
    monitor.set_reference_data(reference_data)

    # Detect drift
    results = monitor.detect_drift(current_data)

    # Generate reports
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON results
    json_path = os.path.join(output_dir, f"drift_results_{timestamp}.json")
    monitor.save_results(json_path)

    # Generate Evidently report if available
    if EVIDENTLY_AVAILABLE:
        html_path = os.path.join(output_dir, f"drift_report_{timestamp}.html")
        monitor.generate_evidently_report(current_data, html_path)

    # Check retraining trigger
    if monitor.check_retraining_trigger():
        trigger_file = "data/processed/retrain_trigger.flag"
        os.makedirs(os.path.dirname(trigger_file), exist_ok=True)
        with open(trigger_file, "w") as f:
            f.write(f"Drift detected at {timestamp}\n")
        logger.warning(f"Retraining trigger file created: {trigger_file}")

    logger.info("Drift monitoring completed")

    return results


if __name__ == "__main__":
    run_drift_monitoring()
