"""
Data Cleaning Module for Credit Risk Scoring System.

This module handles missing value imputation, outlier treatment,
and data validation.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Check and report missing values in the dataset.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame summarizing missing values per column.
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100

    missing_df = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    missing_df = missing_df[missing_df["missing_count"] > 0].sort_values(
        "missing_pct", ascending=False
    )

    if len(missing_df) > 0:
        logger.warning(f"Found {len(missing_df)} columns with missing values")
        logger.info(f"\n{missing_df}")
    else:
        logger.info("No missing values found")

    return missing_df


def impute_missing_values(
    df: pd.DataFrame,
    numeric_strategy: str = "median",
    categorical_strategy: str = "mode",
    numeric_columns: Optional[List[str]] = None,
    categorical_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Impute missing values with domain-sensible strategies.

    Args:
        df: Input DataFrame.
        numeric_strategy: Strategy for numeric columns ('median', 'mean', 'zero').
        categorical_strategy: Strategy for categorical columns ('mode', 'missing').
        numeric_columns: List of numeric columns (auto-detected if None).
        categorical_columns: List of categorical columns (auto-detected if None).

    Returns:
        Tuple of (imputed DataFrame, imputation values dict for future use).
    """
    df = df.copy()
    imputation_values = {"numeric": {}, "categorical": {}}

    # Auto-detect column types if not provided
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if categorical_columns is None:
        categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Impute numeric columns
    for col in numeric_columns:
        if df[col].isnull().any():
            if numeric_strategy == "median":
                fill_value = df[col].median()
            elif numeric_strategy == "mean":
                fill_value = df[col].mean()
            elif numeric_strategy == "zero":
                fill_value = 0
            else:
                raise ValueError(f"Unknown numeric strategy: {numeric_strategy}")

            df[col] = df[col].fillna(fill_value)
            imputation_values["numeric"][col] = fill_value
            logger.debug(f"Imputed {col} with {numeric_strategy}: {fill_value}")

    # Impute categorical columns
    for col in categorical_columns:
        if col in df.columns and df[col].isnull().any():
            if categorical_strategy == "mode":
                fill_value = df[col].mode().iloc[0] if not df[col].mode().empty else "missing"
            elif categorical_strategy == "missing":
                fill_value = "missing"
            else:
                raise ValueError(f"Unknown categorical strategy: {categorical_strategy}")

            df[col] = df[col].fillna(fill_value)
            imputation_values["categorical"][col] = fill_value
            logger.debug(f"Imputed {col} with {categorical_strategy}: {fill_value}")

    return df, imputation_values


def detect_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "iqr",
    threshold: float = 1.5,
) -> pd.DataFrame:
    """Detect outliers in numeric columns.

    Args:
        df: Input DataFrame.
        columns: Columns to check (auto-detect numeric if None).
        method: Detection method ('iqr', 'zscore').
        threshold: Threshold for outlier detection.

    Returns:
        DataFrame with outlier flags for each column.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    outlier_flags = pd.DataFrame(index=df.index)

    for col in columns:
        if col == "target":  # Skip target column
            continue

        if method == "iqr":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outlier_flags[f"{col}_outlier"] = (df[col] < lower_bound) | (df[col] > upper_bound)

        elif method == "zscore":
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outlier_flags[f"{col}_outlier"] = z_scores > threshold

    n_outliers = outlier_flags.any(axis=1).sum()
    logger.info(f"Detected {n_outliers} rows with outliers ({n_outliers/len(df)*100:.1f}%)")

    return outlier_flags


def treat_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "winsorize",
    lower_percentile: float = 0.01,
    upper_percentile: float = 0.99,
) -> Tuple[pd.DataFrame, Dict]:
    """Treat outliers using winsorization or capping.

    Args:
        df: Input DataFrame.
        columns: Columns to treat (auto-detect numeric if None).
        method: Treatment method ('winsorize', 'cap', 'remove').
        lower_percentile: Lower percentile for capping.
        upper_percentile: Upper percentile for capping.

    Returns:
        Tuple of (treated DataFrame, capping bounds dict).
    """
    df = df.copy()
    capping_bounds = {}

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove target from columns
        columns = [c for c in columns if c != "target"]

    for col in columns:
        lower_bound = df[col].quantile(lower_percentile)
        upper_bound = df[col].quantile(upper_percentile)

        if method in ["winsorize", "cap"]:
            original_min = df[col].min()
            original_max = df[col].max()

            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

            capping_bounds[col] = {"lower": lower_bound, "upper": upper_bound}

            if original_min < lower_bound or original_max > upper_bound:
                logger.debug(
                    f"Capped {col}: [{original_min:.2f}, {original_max:.2f}] "
                    f"-> [{lower_bound:.2f}, {upper_bound:.2f}]"
                )

        elif method == "remove":
            mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            df = df[mask]

    logger.info(f"Outlier treatment applied to {len(columns)} columns using {method}")

    return df, capping_bounds


def validate_data(
    df: pd.DataFrame, required_columns: Optional[List[str]] = None
) -> Tuple[bool, List[str]]:
    """Validate data quality and completeness.

    Args:
        df: Input DataFrame.
        required_columns: List of required columns.

    Returns:
        Tuple of (is_valid, list of issues).
    """
    issues = []

    # Check for empty DataFrame
    if df.empty:
        issues.append("DataFrame is empty")
        return False, issues

    # Check required columns
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")

    # Check for duplicate rows
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        issues.append(f"Found {n_duplicates} duplicate rows")

    # Check target column if exists
    if "target" in df.columns:
        unique_targets = df["target"].unique()
        if not set(unique_targets).issubset({0, 1}):
            issues.append(f"Target column has unexpected values: {unique_targets}")

        # Check class imbalance
        target_ratio = df["target"].mean()
        if target_ratio < 0.05 or target_ratio > 0.95:
            issues.append(f"Severe class imbalance: {target_ratio:.2%} positive class")

    # Check for remaining missing values
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        issues.append(f"Columns with missing values: {missing_cols}")

    is_valid = len(issues) == 0

    if is_valid:
        logger.info("Data validation passed")
    else:
        logger.warning(f"Data validation found {len(issues)} issues")
        for issue in issues:
            logger.warning(f"  - {issue}")

    return is_valid, issues


def standardize_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize common encoding patterns.

    Converts various representations to consistent formats:
    - Y/N, Yes/No -> 1/0
    - True/False -> 1/0

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with standardized encodings.
    """
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == "object":
            # Check for Y/N pattern
            unique_vals = set(df[col].dropna().unique())

            if unique_vals.issubset({"Y", "N", "y", "n"}):
                df[col] = df[col].map({"Y": 1, "N": 0, "y": 1, "n": 0})
                logger.debug(f"Converted {col} from Y/N to 1/0")

            elif unique_vals.issubset({"Yes", "No", "yes", "no", "YES", "NO"}):
                df[col] = df[col].str.lower().map({"yes": 1, "no": 0})
                logger.debug(f"Converted {col} from Yes/No to 1/0")

            elif unique_vals.issubset({"True", "False", "true", "false", "TRUE", "FALSE"}):
                df[col] = df[col].str.lower().map({"true": 1, "false": 0})
                logger.debug(f"Converted {col} from True/False to 1/0")

    return df


def run_cleaning_pipeline(
    df: pd.DataFrame,
    impute: bool = True,
    treat_outliers_flag: bool = True,
    validate: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """Run the complete data cleaning pipeline.

    Args:
        df: Input DataFrame.
        impute: Whether to impute missing values.
        treat_outliers_flag: Whether to treat outliers.
        validate: Whether to validate data.

    Returns:
        Tuple of (cleaned DataFrame, cleaning metadata).
    """
    logger.info("Starting data cleaning pipeline")
    metadata = {}

    # Check initial state
    check_missing_values(df)

    # Standardize encodings
    df = standardize_encodings(df)

    # Impute missing values
    if impute:
        df, imputation_values = impute_missing_values(df)
        metadata["imputation_values"] = imputation_values

    # Treat outliers
    if treat_outliers_flag:
        df, capping_bounds = treat_outliers(df)
        metadata["capping_bounds"] = capping_bounds

    # Validate
    if validate:
        is_valid, issues = validate_data(df)
        metadata["validation"] = {"is_valid": is_valid, "issues": issues}

    logger.info("Data cleaning pipeline completed")

    return df, metadata
