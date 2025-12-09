"""
Feature Engineering Module for Credit Risk Scoring System.

This module handles feature creation, encoding, and transformation
using scikit-learn pipelines for reproducibility.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler


class DomainFeatureCreator(BaseEstimator, TransformerMixin):
    """Create domain-specific features for credit risk assessment."""

    def __init__(self):
        """Initialize the feature creator."""
        self.feature_names_out_ = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DomainFeatureCreator":
        """Fit the transformer (computes statistics needed for transform).

        Args:
            X: Input features DataFrame.
            y: Target variable (unused).

        Returns:
            Self.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create domain-specific features.

        Args:
            X: Input features DataFrame.

        Returns:
            DataFrame with additional domain features.
        """
        X = X.copy()

        # Credit utilization ratio (credit amount relative to duration)
        # Higher values indicate taking larger amounts for shorter periods
        if "credit_amount" in X.columns and "duration_months" in X.columns:
            X["credit_per_month"] = X["credit_amount"] / X["duration_months"].replace(0, 1)

        # Installment burden (installment rate as percentage)
        # Higher installment rate indicates higher monthly burden
        if "installment_rate" in X.columns:
            X["high_installment_burden"] = (X["installment_rate"] >= 4).astype(int)

        # Credit to age ratio (how much credit relative to age)
        if "credit_amount" in X.columns and "age" in X.columns:
            X["credit_to_age_ratio"] = X["credit_amount"] / X["age"].replace(0, 1)

        # Age risk groups
        if "age" in X.columns:
            X["age_group_young"] = (X["age"] < 25).astype(int)
            X["age_group_senior"] = (X["age"] >= 60).astype(int)

        # Employment stability indicator
        if "employment_duration" in X.columns:
            employment_stability_map = {
                "unemployed": 0,
                "lt_1_year": 1,
                "1_to_4_years": 2,
                "4_to_7_years": 3,
                "gte_7_years": 4,
            }
            X["employment_stability_score"] = X["employment_duration"].map(employment_stability_map)
            X["employment_stability_score"] = X["employment_stability_score"].fillna(0)

        # Savings security indicator
        if "savings_account" in X.columns:
            savings_score_map = {
                "lt_100": 0,
                "100_to_500": 1,
                "500_to_1000": 2,
                "gte_1000": 3,
                "unknown": 0,
            }
            X["savings_security_score"] = X["savings_account"].map(savings_score_map)
            X["savings_security_score"] = X["savings_security_score"].fillna(0)

        # Checking account risk indicator
        if "checking_account_status" in X.columns:
            checking_risk_map = {
                "negative_balance": 3,
                "0_to_200": 2,
                "200_plus": 1,
                "no_checking": 2,  # No checking can indicate risk
            }
            X["checking_risk_score"] = X["checking_account_status"].map(checking_risk_map)
            X["checking_risk_score"] = X["checking_risk_score"].fillna(2)

        # Credit history score
        if "credit_history" in X.columns:
            history_score_map = {
                "no_credits_taken": 2,
                "all_paid_duly": 0,
                "existing_paid_duly": 0,
                "delay_in_past": 2,
                "critical_account": 3,
            }
            X["credit_history_score"] = X["credit_history"].map(history_score_map)
            X["credit_history_score"] = X["credit_history_score"].fillna(2)

        # Multiple obligations indicator
        if "existing_credits" in X.columns and "other_installment_plans" in X.columns:
            has_other_plans = (X["other_installment_plans"] != "none").astype(int)
            X["multiple_obligations"] = (X["existing_credits"] > 1).astype(int) + has_other_plans

        # Housing stability
        if "housing" in X.columns:
            housing_stability_map = {"rent": 0, "for_free": 1, "own": 2}
            X["housing_stability_score"] = X["housing"].map(housing_stability_map)
            X["housing_stability_score"] = X["housing_stability_score"].fillna(0)

        # Dependency burden
        if "num_dependents" in X.columns:
            X["has_dependents"] = (X["num_dependents"] > 0).astype(int)

        # Foreign worker flag (binary)
        if "foreign_worker" in X.columns:
            X["is_foreign_worker"] = (X["foreign_worker"] == "yes").astype(int)

        # Composite risk score (sum of risk indicators)
        risk_cols = [
            "checking_risk_score",
            "credit_history_score",
            "high_installment_burden",
            "age_group_young",
        ]
        existing_risk_cols = [c for c in risk_cols if c in X.columns]
        if existing_risk_cols:
            X["composite_risk_score"] = X[existing_risk_cols].sum(axis=1)

        # Composite stability score
        stability_cols = [
            "employment_stability_score",
            "savings_security_score",
            "housing_stability_score",
        ]
        existing_stability_cols = [c for c in stability_cols if c in X.columns]
        if existing_stability_cols:
            X["composite_stability_score"] = X[existing_stability_cols].sum(axis=1)

        return X

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """Get output feature names.

        Args:
            input_features: Input feature names.

        Returns:
            List of output feature names.
        """
        return input_features


def get_feature_lists(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Identify numeric and categorical columns.

    Args:
        df: Input DataFrame.

    Returns:
        Dictionary with 'numeric' and 'categorical' column lists.
    """
    # Exclude target and any ID columns
    exclude_cols = ["target", "id", "index"]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c not in exclude_cols]

    return {"numeric": numeric_cols, "categorical": categorical_cols}


def create_preprocessing_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    scaler_type: str = "robust",
    handle_unknown: str = "ignore",
) -> ColumnTransformer:
    """Create a preprocessing pipeline using sklearn ColumnTransformer.

    Args:
        numeric_features: List of numeric column names.
        categorical_features: List of categorical column names.
        scaler_type: Type of scaler ('standard', 'robust').
        handle_unknown: How to handle unknown categories in OneHotEncoder.

    Returns:
        ColumnTransformer with preprocessing steps.
    """
    # Select scaler
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")

    # Create transformers
    numeric_transformer = Pipeline(steps=[("scaler", scaler)])

    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(handle_unknown=handle_unknown, sparse_output=False, drop="first"),
            )
        ]
    )

    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="passthrough",  # Keep any other columns
    )

    return preprocessor


def create_full_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    scaler_type: str = "robust",
) -> Pipeline:
    """Create a full feature engineering pipeline.

    Args:
        numeric_features: List of numeric column names.
        categorical_features: List of categorical column names.
        scaler_type: Type of scaler to use.

    Returns:
        Complete preprocessing Pipeline.
    """
    preprocessor = create_preprocessing_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        scaler_type=scaler_type,
    )

    pipeline = Pipeline(
        steps=[
            ("domain_features", DomainFeatureCreator()),
            ("preprocessor", preprocessor),
        ]
    )

    return pipeline


def get_feature_names_from_pipeline(
    pipeline: Pipeline, numeric_features: List[str], categorical_features: List[str]
) -> List[str]:
    """Extract feature names from a fitted pipeline.

    Args:
        pipeline: Fitted sklearn Pipeline.
        numeric_features: Original numeric feature names.
        categorical_features: Original categorical feature names.

    Returns:
        List of transformed feature names.
    """
    try:
        # Try to get feature names from the preprocessor
        preprocessor = pipeline.named_steps.get("preprocessor")
        if preprocessor is not None:
            return preprocessor.get_feature_names_out().tolist()
    except Exception:
        pass

    # Fallback: construct names manually
    feature_names = numeric_features.copy()

    # OneHot encoded features
    for cat_col in categorical_features:
        feature_names.append(f"{cat_col}_encoded")

    return feature_names


def save_pipeline(pipeline: Pipeline, filepath: str) -> None:
    """Save a fitted pipeline to disk.

    Args:
        pipeline: Fitted sklearn Pipeline.
        filepath: Path to save the pipeline.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(pipeline, filepath)
    logger.info(f"Pipeline saved to {filepath}")


def load_pipeline(filepath: str) -> Pipeline:
    """Load a pipeline from disk.

    Args:
        filepath: Path to the saved pipeline.

    Returns:
        Loaded sklearn Pipeline.
    """
    pipeline = joblib.load(filepath)
    logger.info(f"Pipeline loaded from {filepath}")
    return pipeline


def prepare_features(
    df: pd.DataFrame, target_col: str = "target"
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Separate features and target from DataFrame.

    Args:
        df: Input DataFrame.
        target_col: Name of target column.

    Returns:
        Tuple of (features DataFrame, target Series or None).
    """
    if target_col in df.columns:
        X = df.drop(columns=[target_col])
        y = df[target_col]
    else:
        X = df
        y = None

    return X, y


def run_feature_engineering(
    train_df: pd.DataFrame,
    validation_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    target_col: str = "target",
    scaler_type: str = "robust",
) -> Dict[str, Any]:
    """Run the feature engineering pipeline on train/validation/test sets.

    Args:
        train_df: Training DataFrame.
        validation_df: Validation DataFrame (optional).
        test_df: Test DataFrame (optional).
        target_col: Name of target column.
        scaler_type: Type of scaler to use.

    Returns:
        Dictionary containing transformed data and pipeline.
    """
    logger.info("Starting feature engineering pipeline")

    # Separate features and target
    X_train, y_train = prepare_features(train_df, target_col)

    # Apply domain feature creation first to get all columns
    domain_creator = DomainFeatureCreator()
    X_train_with_domain = domain_creator.fit_transform(X_train)

    # Get feature lists from the enhanced dataframe
    feature_lists = get_feature_lists(X_train_with_domain)
    numeric_features = feature_lists["numeric"]
    categorical_features = feature_lists["categorical"]

    logger.info(f"Numeric features ({len(numeric_features)}): {numeric_features}")
    logger.info(f"Categorical features ({len(categorical_features)}): {categorical_features}")

    # Create and fit pipeline
    pipeline = create_full_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        scaler_type=scaler_type,
    )

    # Fit on training data
    X_train_transformed = pipeline.fit_transform(X_train)
    logger.info(f"Training data shape after transformation: {X_train_transformed.shape}")

    result = {
        "pipeline": pipeline,
        "X_train": X_train_transformed,
        "y_train": y_train,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }

    # Transform validation data
    if validation_df is not None:
        X_val, y_val = prepare_features(validation_df, target_col)
        X_val_transformed = pipeline.transform(X_val)
        result["X_val"] = X_val_transformed
        result["y_val"] = y_val
        logger.info(f"Validation data shape: {X_val_transformed.shape}")

    # Transform test data
    if test_df is not None:
        X_test, y_test = prepare_features(test_df, target_col)
        X_test_transformed = pipeline.transform(X_test)
        result["X_test"] = X_test_transformed
        result["y_test"] = y_test
        logger.info(f"Test data shape: {X_test_transformed.shape}")

    logger.info("Feature engineering completed")

    return result
