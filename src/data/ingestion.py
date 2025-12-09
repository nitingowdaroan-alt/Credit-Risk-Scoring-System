"""
Data Ingestion Module for Credit Risk Scoring System.

This module handles downloading, loading, and initial preprocessing of raw data.
Uses the German Credit dataset from UCI Machine Learning Repository.
"""

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yaml
from loguru import logger
from sklearn.model_selection import train_test_split


def load_config(config_path: str = "configs/paths.yaml") -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Dictionary containing configuration.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def download_german_credit_data(output_path: str) -> pd.DataFrame:
    """Download German Credit dataset from UCI repository.

    The German Credit dataset contains 1000 instances with 20 attributes,
    classifying people as good or bad credit risks.

    Source: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)

    Args:
        output_path: Path to save the downloaded data.

    Returns:
        DataFrame containing the raw data.
    """
    # UCI repository URL for German Credit data
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

    logger.info(f"Downloading German Credit dataset from {url}")

    # Column names based on german.doc
    column_names = [
        "checking_account_status",
        "duration_months",
        "credit_history",
        "purpose",
        "credit_amount",
        "savings_account",
        "employment_duration",
        "installment_rate",
        "personal_status_sex",
        "other_debtors",
        "residence_duration",
        "property",
        "age",
        "other_installment_plans",
        "housing",
        "existing_credits",
        "job",
        "num_dependents",
        "telephone",
        "foreign_worker",
        "target",
    ]

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Parse the data (space-separated)
        lines = response.text.strip().split("\n")
        data = [line.split() for line in lines]
        df = pd.DataFrame(data, columns=column_names)

        # Convert numeric columns
        numeric_cols = [
            "duration_months",
            "credit_amount",
            "installment_rate",
            "residence_duration",
            "age",
            "existing_credits",
            "num_dependents",
            "target",
        ]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col])

        # Convert target: 1 = Good, 2 = Bad -> 0 = Good (no default), 1 = Bad (default)
        df["target"] = df["target"].map({1: 0, 2: 1})

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Data saved to {output_path}")
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Target distribution:\n{df['target'].value_counts(normalize=True)}")

        return df

    except requests.RequestException as e:
        logger.error(f"Failed to download data: {e}")
        raise


def load_raw_data(file_path: str) -> pd.DataFrame:
    """Load raw data from CSV file.

    Args:
        file_path: Path to the CSV file.

    Returns:
        DataFrame containing the raw data.
    """
    logger.info(f"Loading data from {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")

    return df


def decode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Decode categorical features from codes to meaningful labels.

    Args:
        df: DataFrame with coded categorical values.

    Returns:
        DataFrame with decoded categorical values.
    """
    df = df.copy()

    # Checking account status
    checking_map = {
        "A11": "negative_balance",
        "A12": "0_to_200",
        "A13": "200_plus",
        "A14": "no_checking",
    }
    df["checking_account_status"] = df["checking_account_status"].map(checking_map)

    # Credit history
    credit_history_map = {
        "A30": "no_credits_taken",
        "A31": "all_paid_duly",
        "A32": "existing_paid_duly",
        "A33": "delay_in_past",
        "A34": "critical_account",
    }
    df["credit_history"] = df["credit_history"].map(credit_history_map)

    # Purpose
    purpose_map = {
        "A40": "car_new",
        "A41": "car_used",
        "A42": "furniture",
        "A43": "radio_tv",
        "A44": "appliances",
        "A45": "repairs",
        "A46": "education",
        "A47": "vacation",
        "A48": "retraining",
        "A49": "business",
        "A410": "others",
    }
    df["purpose"] = df["purpose"].map(purpose_map)

    # Savings account
    savings_map = {
        "A61": "lt_100",
        "A62": "100_to_500",
        "A63": "500_to_1000",
        "A64": "gte_1000",
        "A65": "unknown",
    }
    df["savings_account"] = df["savings_account"].map(savings_map)

    # Employment duration
    employment_map = {
        "A71": "unemployed",
        "A72": "lt_1_year",
        "A73": "1_to_4_years",
        "A74": "4_to_7_years",
        "A75": "gte_7_years",
    }
    df["employment_duration"] = df["employment_duration"].map(employment_map)

    # Personal status and sex
    personal_map = {
        "A91": "male_divorced_separated",
        "A92": "female_divorced_separated_married",
        "A93": "male_single",
        "A94": "male_married_widowed",
        "A95": "female_single",
    }
    df["personal_status_sex"] = df["personal_status_sex"].map(personal_map)

    # Other debtors/guarantors
    debtors_map = {"A101": "none", "A102": "co_applicant", "A103": "guarantor"}
    df["other_debtors"] = df["other_debtors"].map(debtors_map)

    # Property
    property_map = {
        "A121": "real_estate",
        "A122": "savings_insurance",
        "A123": "car_other",
        "A124": "unknown_none",
    }
    df["property"] = df["property"].map(property_map)

    # Other installment plans
    installment_map = {"A141": "bank", "A142": "stores", "A143": "none"}
    df["other_installment_plans"] = df["other_installment_plans"].map(installment_map)

    # Housing
    housing_map = {"A151": "rent", "A152": "own", "A153": "for_free"}
    df["housing"] = df["housing"].map(housing_map)

    # Job
    job_map = {
        "A171": "unemployed_unskilled_non_resident",
        "A172": "unskilled_resident",
        "A173": "skilled_employee",
        "A174": "management_self_employed",
    }
    df["job"] = df["job"].map(job_map)

    # Telephone
    telephone_map = {"A191": "none", "A192": "yes"}
    df["telephone"] = df["telephone"].map(telephone_map)

    # Foreign worker
    foreign_map = {"A201": "yes", "A202": "no"}
    df["foreign_worker"] = df["foreign_worker"].map(foreign_map)

    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    random_state: int = 42,
    stratify_col: str = "target",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, validation, and test sets.

    Args:
        df: Input DataFrame.
        test_size: Proportion of data for test set.
        validation_size: Proportion of training data for validation.
        random_state: Random seed for reproducibility.
        stratify_col: Column to stratify splits on.

    Returns:
        Tuple of (train, validation, test) DataFrames.
    """
    logger.info("Splitting data into train/validation/test sets")

    # First split: separate test set
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[stratify_col]
    )

    # Second split: separate validation from training
    train, validation = train_test_split(
        train_val,
        test_size=validation_size,
        random_state=random_state,
        stratify=train_val[stratify_col],
    )

    logger.info(f"Train set: {len(train)} samples")
    logger.info(f"Validation set: {len(validation)} samples")
    logger.info(f"Test set: {len(test)} samples")

    return train, validation, test


def save_splits(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: str,
) -> None:
    """Save data splits to CSV files.

    Args:
        train: Training DataFrame.
        validation: Validation DataFrame.
        test: Test DataFrame.
        output_dir: Directory to save files.
    """
    os.makedirs(output_dir, exist_ok=True)

    train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    validation.to_csv(os.path.join(output_dir, "validation.csv"), index=False)
    test.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    logger.info(f"Data splits saved to {output_dir}")


def run_ingestion_pipeline(config_path: str = "configs/paths.yaml") -> None:
    """Run the complete data ingestion pipeline.

    Args:
        config_path: Path to configuration file.
    """
    config = load_config(config_path)

    # Download data if not exists
    raw_path = config["data"]["raw_data"]
    if not os.path.exists(raw_path):
        logger.info("Raw data not found, downloading...")
        df = download_german_credit_data(raw_path)
    else:
        logger.info("Loading existing raw data...")
        df = load_raw_data(raw_path)

    # Decode categorical features
    df = decode_categorical_features(df)

    # Split data
    train, validation, test = split_data(df)

    # Save splits
    save_splits(train, validation, test, config["data"]["processed_dir"])

    logger.info("Data ingestion pipeline completed successfully")


def main():
    """Main entry point for data ingestion."""
    parser = argparse.ArgumentParser(description="Data Ingestion for Credit Risk System")
    parser.add_argument(
        "action",
        choices=["download", "preprocess", "all"],
        help="Action to perform",
    )
    parser.add_argument(
        "--config",
        default="configs/paths.yaml",
        help="Path to configuration file",
    )

    args = parser.parse_args()

    if args.action == "download":
        config = load_config(args.config)
        download_german_credit_data(config["data"]["raw_data"])
    elif args.action == "preprocess":
        run_ingestion_pipeline(args.config)
    elif args.action == "all":
        run_ingestion_pipeline(args.config)


if __name__ == "__main__":
    main()
