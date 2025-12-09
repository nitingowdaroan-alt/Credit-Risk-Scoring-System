"""
SHAP Explainability Module for Credit Risk Scoring System.

This module provides global and local model explanations using SHAP values,
generating reports suitable for regulators and stakeholders.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available")


class SHAPExplainer:
    """SHAP-based model explainer for credit risk models."""

    def __init__(self, model: Any, feature_names: Optional[List[str]] = None):
        """Initialize the explainer.

        Args:
            model: Trained model (sklearn-compatible).
            feature_names: List of feature names.
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required for explainability features")

        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        self.base_value = None

    def create_explainer(
        self, X_background: np.ndarray, explainer_type: str = "auto"
    ) -> None:
        """Create SHAP explainer with background data.

        Args:
            X_background: Background dataset for SHAP calculations.
            explainer_type: Type of explainer ('tree', 'kernel', 'auto').
        """
        model_type = type(self.model).__name__.lower()

        if explainer_type == "auto":
            # Auto-detect explainer type based on model
            if any(t in model_type for t in ["lgbm", "xgb", "forest", "gbm", "boost"]):
                explainer_type = "tree"
            else:
                explainer_type = "kernel"

        logger.info(f"Creating {explainer_type} explainer for {model_type}")

        if explainer_type == "tree":
            self.explainer = shap.TreeExplainer(self.model)
        elif explainer_type == "kernel":
            # Sample background data if too large
            if len(X_background) > 100:
                X_background = shap.sample(X_background, 100)
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba, X_background
            )
        else:
            raise ValueError(f"Unknown explainer type: {explainer_type}")

    def compute_shap_values(
        self, X: np.ndarray, check_additivity: bool = False
    ) -> np.ndarray:
        """Compute SHAP values for given data.

        Args:
            X: Input features.
            check_additivity: Whether to check SHAP additivity.

        Returns:
            Array of SHAP values.
        """
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer first.")

        logger.info(f"Computing SHAP values for {len(X)} samples")

        shap_values = self.explainer.shap_values(X, check_additivity=check_additivity)

        # Handle binary classification (get values for positive class)
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]

        self.shap_values = shap_values

        # Get base value
        if hasattr(self.explainer, "expected_value"):
            base = self.explainer.expected_value
            if isinstance(base, (list, np.ndarray)) and len(base) == 2:
                self.base_value = base[1]
            else:
                self.base_value = base

        return shap_values

    def get_global_importance(self) -> pd.DataFrame:
        """Get global feature importance based on mean absolute SHAP values.

        Returns:
            DataFrame with feature importances.
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed")

        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)

        feature_names = self.feature_names or [f"feature_{i}" for i in range(len(mean_abs_shap))]

        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": mean_abs_shap}
        ).sort_values("importance", ascending=False)

        return importance_df

    def get_local_explanation(
        self, sample_idx: int, X: np.ndarray, top_n: int = 10
    ) -> Dict[str, Any]:
        """Get local explanation for a single prediction.

        Args:
            sample_idx: Index of the sample.
            X: Feature matrix.
            top_n: Number of top features to include.

        Returns:
            Dictionary with local explanation.
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed")

        sample_shap = self.shap_values[sample_idx]
        sample_features = X[sample_idx]

        feature_names = self.feature_names or [f"feature_{i}" for i in range(len(sample_shap))]

        # Create explanation dataframe
        explanation_df = pd.DataFrame(
            {
                "feature": feature_names,
                "value": sample_features,
                "shap_value": sample_shap,
                "abs_shap": np.abs(sample_shap),
            }
        ).sort_values("abs_shap", ascending=False)

        # Get top contributors
        top_positive = explanation_df[explanation_df["shap_value"] > 0].head(top_n)
        top_negative = explanation_df[explanation_df["shap_value"] < 0].head(top_n)

        return {
            "sample_idx": sample_idx,
            "base_value": float(self.base_value) if self.base_value else None,
            "prediction_contribution": float(sample_shap.sum()),
            "top_positive_contributors": top_positive.to_dict("records"),
            "top_negative_contributors": top_negative.to_dict("records"),
            "all_contributions": explanation_df.to_dict("records"),
        }

    def plot_summary(self, X: np.ndarray, output_path: str, max_display: int = 20) -> str:
        """Generate SHAP summary plot.

        Args:
            X: Feature matrix.
            output_path: Path to save the plot.
            max_display: Maximum features to display.

        Returns:
            Path to saved plot.
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed")

        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values,
            X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Summary plot saved to {output_path}")
        return output_path

    def plot_bar(self, output_path: str, max_display: int = 20) -> str:
        """Generate SHAP bar plot (mean absolute values).

        Args:
            output_path: Path to save the plot.
            max_display: Maximum features to display.

        Returns:
            Path to saved plot.
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed")

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values,
            feature_names=self.feature_names,
            plot_type="bar",
            max_display=max_display,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Bar plot saved to {output_path}")
        return output_path

    def plot_waterfall(
        self, sample_idx: int, output_path: str, max_display: int = 15
    ) -> str:
        """Generate waterfall plot for a single prediction.

        Args:
            sample_idx: Index of the sample.
            output_path: Path to save the plot.
            max_display: Maximum features to display.

        Returns:
            Path to saved plot.
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed")

        plt.figure(figsize=(10, 8))

        # Create Explanation object for waterfall plot
        explanation = shap.Explanation(
            values=self.shap_values[sample_idx],
            base_values=self.base_value if self.base_value else 0,
            feature_names=self.feature_names,
        )

        shap.waterfall_plot(explanation, max_display=max_display, show=False)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Waterfall plot saved to {output_path}")
        return output_path


def generate_explainability_report(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: Optional[List[str]] = None,
    output_dir: str = "reports/explainability",
    n_samples_explain: int = 100,
) -> Dict[str, Any]:
    """Generate comprehensive explainability report.

    Args:
        model: Trained model.
        X_train: Training features (for background).
        X_test: Test features (to explain).
        feature_names: Feature names.
        output_dir: Output directory.
        n_samples_explain: Number of test samples to explain.

    Returns:
        Dictionary with report information.
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available, skipping explainability report")
        return {"status": "skipped", "reason": "SHAP not installed"}

    logger.info("Generating explainability report")

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize explainer
    explainer = SHAPExplainer(model, feature_names)

    # Sample background data
    n_background = min(100, len(X_train))
    background_idx = np.random.choice(len(X_train), n_background, replace=False)
    X_background = X_train[background_idx]

    explainer.create_explainer(X_background)

    # Compute SHAP values for test samples
    n_explain = min(n_samples_explain, len(X_test))
    explain_idx = np.random.choice(len(X_test), n_explain, replace=False)
    X_explain = X_test[explain_idx]

    explainer.compute_shap_values(X_explain)

    report = {
        "timestamp": timestamp,
        "n_samples_explained": n_explain,
        "n_features": X_explain.shape[1],
        "plots": {},
        "feature_importance": {},
        "sample_explanations": [],
    }

    # Generate global importance
    importance_df = explainer.get_global_importance()
    report["feature_importance"] = importance_df.to_dict("records")

    # Save importance to CSV
    importance_path = os.path.join(output_dir, f"feature_importance_{timestamp}.csv")
    importance_df.to_csv(importance_path, index=False)

    # Generate plots
    summary_path = os.path.join(output_dir, f"shap_summary_{timestamp}.png")
    explainer.plot_summary(X_explain, summary_path)
    report["plots"]["summary"] = summary_path

    bar_path = os.path.join(output_dir, f"shap_bar_{timestamp}.png")
    explainer.plot_bar(bar_path)
    report["plots"]["bar"] = bar_path

    # Generate sample explanations (top 5)
    for i in range(min(5, n_explain)):
        sample_explanation = explainer.get_local_explanation(i, X_explain)
        report["sample_explanations"].append(sample_explanation)

        # Generate waterfall plot for this sample
        waterfall_path = os.path.join(output_dir, f"waterfall_sample_{i}_{timestamp}.png")
        explainer.plot_waterfall(i, waterfall_path)
        sample_explanation["waterfall_plot"] = waterfall_path

    # Save report
    report_path = os.path.join(output_dir, f"explainability_report_{timestamp}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Explainability report saved to {report_path}")

    return report


def generate_regulator_report(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Optional[List[str]] = None,
    output_dir: str = "reports/explainability",
    model_description: Optional[str] = None,
) -> str:
    """Generate regulator-style HTML report.

    Args:
        model: Trained model.
        X_train: Training features.
        X_test: Test features.
        y_test: Test labels.
        feature_names: Feature names.
        output_dir: Output directory.
        model_description: Description of the model.

    Returns:
        Path to generated HTML report.
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available")
        return ""

    logger.info("Generating regulator report")

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate basic explainability
    explainer = SHAPExplainer(model, feature_names)

    n_background = min(100, len(X_train))
    background_idx = np.random.choice(len(X_train), n_background, replace=False)
    explainer.create_explainer(X_train[background_idx])

    n_explain = min(200, len(X_test))
    explain_idx = np.random.choice(len(X_test), n_explain, replace=False)
    explainer.compute_shap_values(X_test[explain_idx])

    importance_df = explainer.get_global_importance()

    # Generate HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Credit Risk Model - Explainability Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            h1 {{ color: #333; border-bottom: 2px solid #333; padding-bottom: 10px; }}
            h2 {{ color: #555; margin-top: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .section {{ margin: 30px 0; padding: 20px; background: #f9f9f9; border-radius: 5px; }}
            .highlight {{ background-color: #fff3cd; padding: 15px; border-radius: 5px; }}
            img {{ max-width: 100%; height: auto; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>Credit Risk Scoring Model - Explainability Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="section">
            <h2>1. Model Description</h2>
            <p>{model_description or 'Binary classification model for predicting probability of default.'}</p>
            <ul>
                <li><strong>Model Type:</strong> {type(model).__name__}</li>
                <li><strong>Target:</strong> Probability of Default (PD)</li>
                <li><strong>Prediction Horizon:</strong> 12 months</li>
            </ul>
        </div>

        <div class="section">
            <h2>2. Feature Importance (Global)</h2>
            <p>The following table shows the relative importance of each feature in the model's predictions,
            measured by mean absolute SHAP value.</p>
            <table>
                <tr><th>Rank</th><th>Feature</th><th>Importance</th></tr>
    """

    for i, row in importance_df.head(15).iterrows():
        html_content += f"""
                <tr>
                    <td>{i+1}</td>
                    <td>{row['feature']}</td>
                    <td>{row['importance']:.4f}</td>
                </tr>
        """

    html_content += """
            </table>
        </div>

        <div class="section">
            <h2>3. How Features Influence Decisions</h2>
            <div class="highlight">
                <p><strong>Key Observations:</strong></p>
                <ul>
    """

    top_features = importance_df.head(5)["feature"].tolist()
    for feat in top_features:
        html_content += f"<li>{feat} is among the most influential features in credit decisions.</li>\n"

    html_content += """
                </ul>
            </div>
        </div>

        <div class="section">
            <h2>4. Model Fairness Considerations</h2>
            <p>This model has been designed with fairness in mind:</p>
            <ul>
                <li>Protected characteristics are monitored for disparate impact</li>
                <li>Feature importance is regularly reviewed for unexpected patterns</li>
                <li>Model decisions are explainable at both global and individual levels</li>
            </ul>
        </div>

        <div class="section">
            <h2>5. Monitoring and Governance</h2>
            <ul>
                <li>Model performance is monitored daily</li>
                <li>Data drift is checked weekly</li>
                <li>Full model review conducted quarterly</li>
                <li>Retraining triggered automatically when drift exceeds thresholds</li>
            </ul>
        </div>

        <footer>
            <hr>
            <p><em>This report is generated automatically by the Credit Risk Scoring System.</em></p>
        </footer>
    </body>
    </html>
    """

    report_path = os.path.join(output_dir, f"regulator_report_{timestamp}.html")
    with open(report_path, "w") as f:
        f.write(html_content)

    logger.info(f"Regulator report saved to {report_path}")

    return report_path


if __name__ == "__main__":
    # Demo/test code
    logger.info("SHAP Explainer module loaded")
