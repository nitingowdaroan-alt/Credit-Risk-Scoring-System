"""
Unit tests for data pipeline modules.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.cleaning import (
    check_missing_values,
    impute_missing_values,
    run_cleaning_pipeline,
    standardize_encodings,
    treat_outliers,
    validate_data,
)
from src.data.feature_engineering import (
    DomainFeatureCreator,
    create_preprocessing_pipeline,
    get_feature_lists,
    prepare_features,
)


class TestDataCleaning:
    """Tests for data cleaning module."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame(
            {
                "numeric1": [1.0, 2.0, np.nan, 4.0, 5.0],
                "numeric2": [10, 20, 30, 40, 50],
                "categorical1": ["A", "B", None, "A", "B"],
                "categorical2": ["X", "Y", "X", "Y", "X"],
                "target": [0, 1, 0, 1, 0],
            }
        )

    def test_check_missing_values(self, sample_data):
        """Test missing value detection."""
        missing_df = check_missing_values(sample_data)
        assert "numeric1" in missing_df.index
        assert "categorical1" in missing_df.index

    def test_impute_missing_values_numeric(self, sample_data):
        """Test numeric imputation."""
        df, metadata = impute_missing_values(
            sample_data, numeric_strategy="median", categorical_strategy="mode"
        )

        # Check no missing values remain
        assert df["numeric1"].isnull().sum() == 0
        assert df["categorical1"].isnull().sum() == 0

        # Check imputation values are recorded
        assert "numeric" in metadata
        assert "categorical" in metadata

    def test_treat_outliers(self):
        """Test outlier treatment."""
        df = pd.DataFrame(
            {"values": [1, 2, 3, 4, 5, 100, 200], "target": [0, 1, 0, 1, 0, 1, 0]}
        )

        treated_df, bounds = treat_outliers(
            df, columns=["values"], method="winsorize"
        )

        # Check bounds were recorded
        assert "values" in bounds
        assert "lower" in bounds["values"]
        assert "upper" in bounds["values"]

        # Check extreme values are capped
        assert treated_df["values"].max() <= bounds["values"]["upper"]

    def test_validate_data_valid(self):
        """Test data validation with valid data."""
        df = pd.DataFrame(
            {"feature1": [1, 2, 3], "feature2": [4, 5, 6], "target": [0, 1, 0]}
        )

        is_valid, issues = validate_data(df)
        assert is_valid is True
        assert len(issues) == 0

    def test_validate_data_invalid_target(self):
        """Test data validation with invalid target values."""
        df = pd.DataFrame(
            {"feature1": [1, 2, 3], "target": [0, 1, 2]}  # Invalid: should be 0 or 1
        )

        is_valid, issues = validate_data(df)
        assert is_valid is False
        assert any("unexpected values" in issue for issue in issues)

    def test_standardize_encodings(self):
        """Test standardization of Y/N encodings."""
        df = pd.DataFrame({"yn_col": ["Y", "N", "Y", "N"], "other": ["A", "B", "A", "B"]})

        standardized = standardize_encodings(df)

        assert standardized["yn_col"].tolist() == [1, 0, 1, 0]
        assert standardized["other"].tolist() == ["A", "B", "A", "B"]

    def test_run_cleaning_pipeline(self, sample_data):
        """Test full cleaning pipeline."""
        cleaned_df, metadata = run_cleaning_pipeline(sample_data, validate=False)

        # Check no missing values
        assert cleaned_df.isnull().sum().sum() == 0

        # Check metadata structure
        assert "imputation_values" in metadata


class TestFeatureEngineering:
    """Tests for feature engineering module."""

    @pytest.fixture
    def sample_features(self):
        """Create sample features for testing."""
        return pd.DataFrame(
            {
                "credit_amount": [1000, 2000, 3000, 4000, 5000],
                "duration_months": [12, 24, 36, 12, 24],
                "age": [25, 35, 45, 55, 65],
                "installment_rate": [1, 2, 3, 4, 4],
                "employment_duration": [
                    "lt_1_year",
                    "1_to_4_years",
                    "4_to_7_years",
                    "gte_7_years",
                    "unemployed",
                ],
                "savings_account": ["lt_100", "100_to_500", "500_to_1000", "gte_1000", "unknown"],
                "checking_account_status": [
                    "negative_balance",
                    "0_to_200",
                    "200_plus",
                    "no_checking",
                    "0_to_200",
                ],
                "credit_history": [
                    "all_paid_duly",
                    "existing_paid_duly",
                    "delay_in_past",
                    "critical_account",
                    "no_credits_taken",
                ],
                "housing": ["rent", "own", "for_free", "own", "rent"],
                "other_installment_plans": ["bank", "stores", "none", "none", "bank"],
                "existing_credits": [1, 2, 1, 3, 2],
                "target": [0, 1, 0, 1, 0],
            }
        )

    def test_domain_feature_creator(self, sample_features):
        """Test domain feature creation."""
        creator = DomainFeatureCreator()
        transformed = creator.fit_transform(sample_features)

        # Check new features are created
        assert "credit_per_month" in transformed.columns
        assert "high_installment_burden" in transformed.columns
        assert "credit_to_age_ratio" in transformed.columns

    def test_domain_feature_values(self, sample_features):
        """Test domain feature calculations."""
        creator = DomainFeatureCreator()
        transformed = creator.fit_transform(sample_features)

        # Check credit_per_month calculation
        expected = sample_features["credit_amount"] / sample_features["duration_months"]
        np.testing.assert_array_almost_equal(
            transformed["credit_per_month"].values, expected.values
        )

    def test_get_feature_lists(self, sample_features):
        """Test feature list detection."""
        feature_lists = get_feature_lists(sample_features)

        assert "numeric" in feature_lists
        assert "categorical" in feature_lists

        # Check specific features are categorized correctly
        assert "credit_amount" in feature_lists["numeric"]
        assert "employment_duration" in feature_lists["categorical"]

    def test_prepare_features(self, sample_features):
        """Test feature/target separation."""
        X, y = prepare_features(sample_features, target_col="target")

        assert "target" not in X.columns
        assert len(y) == len(sample_features)
        assert list(y) == [0, 1, 0, 1, 0]

    def test_create_preprocessing_pipeline(self, sample_features):
        """Test preprocessing pipeline creation."""
        numeric = ["credit_amount", "duration_months", "age"]
        categorical = ["employment_duration", "housing"]

        pipeline = create_preprocessing_pipeline(numeric, categorical)

        # Pipeline should have transformers
        assert pipeline is not None
        assert hasattr(pipeline, "fit_transform")


class TestIntegration:
    """Integration tests for data pipeline."""

    @pytest.fixture
    def full_sample_data(self):
        """Create full sample data for integration testing."""
        np.random.seed(42)
        n = 100

        return pd.DataFrame(
            {
                "checking_account_status": np.random.choice(
                    ["negative_balance", "0_to_200", "200_plus", "no_checking"], n
                ),
                "duration_months": np.random.randint(6, 60, n),
                "credit_history": np.random.choice(
                    ["all_paid_duly", "existing_paid_duly", "delay_in_past", "critical_account"], n
                ),
                "credit_amount": np.random.randint(500, 10000, n),
                "savings_account": np.random.choice(
                    ["lt_100", "100_to_500", "500_to_1000", "gte_1000"], n
                ),
                "employment_duration": np.random.choice(
                    ["unemployed", "lt_1_year", "1_to_4_years", "4_to_7_years", "gte_7_years"], n
                ),
                "installment_rate": np.random.randint(1, 5, n),
                "age": np.random.randint(18, 75, n),
                "housing": np.random.choice(["rent", "own", "for_free"], n),
                "existing_credits": np.random.randint(1, 5, n),
                "target": np.random.randint(0, 2, n),
            }
        )

    def test_full_pipeline(self, full_sample_data):
        """Test complete data pipeline."""
        # Clean data
        cleaned_df, _ = run_cleaning_pipeline(full_sample_data, validate=False)

        # Separate features and target
        X, y = prepare_features(cleaned_df, target_col="target")

        # Create domain features
        creator = DomainFeatureCreator()
        X_enhanced = creator.fit_transform(X)

        # Check no critical columns dropped
        assert "credit_amount" in X_enhanced.columns
        assert "age" in X_enhanced.columns
        assert len(X_enhanced) == len(full_sample_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
