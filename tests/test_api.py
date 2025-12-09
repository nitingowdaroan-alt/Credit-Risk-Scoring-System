"""
Unit tests for FastAPI application.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.schemas import ApplicantFeatures


@pytest.fixture
def client():
    """Create test client for API."""
    return TestClient(app)


@pytest.fixture
def sample_applicant():
    """Create sample applicant data for testing."""
    return {
        "checking_account_status": "0_to_200",
        "duration_months": 24,
        "credit_history": "existing_paid_duly",
        "purpose": "car_new",
        "credit_amount": 5000.0,
        "savings_account": "100_to_500",
        "employment_duration": "1_to_4_years",
        "installment_rate": 3,
        "personal_status_sex": "male_single",
        "other_debtors": "none",
        "residence_duration": 2,
        "property": "car_other",
        "age": 35,
        "other_installment_plans": "none",
        "housing": "own",
        "existing_credits": 1,
        "job": "skilled_employee",
        "num_dependents": 1,
        "telephone": "yes",
        "foreign_worker": "no",
    }


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self, client):
        """Test health endpoint returns correct structure."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "timestamp" in data

    def test_health_check_status_values(self, client):
        """Test health status has valid values."""
        response = client.get("/health")
        data = response.json()

        assert data["status"] in ["healthy", "unhealthy"]
        assert isinstance(data["model_loaded"], bool)


class TestModelInfoEndpoint:
    """Tests for model info endpoint."""

    def test_model_info_structure(self, client):
        """Test model info endpoint returns correct structure."""
        response = client.get("/model_info")

        # Should return 200 or 503 if model not loaded
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "model_name" in data
            assert "model_version" in data
            assert "model_type" in data


class TestScoringEndpoint:
    """Tests for scoring endpoint."""

    def test_score_valid_applicant(self, client, sample_applicant):
        """Test scoring with valid applicant data."""
        response = client.post("/score", json=sample_applicant)
        assert response.status_code == 200

        data = response.json()
        assert "request_id" in data
        assert "probability_of_default" in data
        assert "credit_score" in data
        assert "risk_band" in data
        assert "decision" in data
        assert "model_version" in data

    def test_score_response_values(self, client, sample_applicant):
        """Test scoring response has valid values."""
        response = client.post("/score", json=sample_applicant)
        data = response.json()

        # Check PD is in valid range
        assert 0 <= data["probability_of_default"] <= 1

        # Check score is in valid range
        assert 300 <= data["credit_score"] <= 850

        # Check risk band is valid
        assert data["risk_band"] in ["Low Risk", "Medium Risk", "High Risk"]

        # Check decision is valid
        assert data["decision"] in ["auto_approve", "manual_review", "decline"]

    def test_score_invalid_age(self, client, sample_applicant):
        """Test scoring with invalid age."""
        sample_applicant["age"] = 10  # Too young
        response = client.post("/score", json=sample_applicant)

        # Should return 422 (validation error) or 200 with decline
        if response.status_code == 200:
            data = response.json()
            assert data["decision"] == "decline" or data["risk_band"] == "High Risk"
        else:
            assert response.status_code == 422

    def test_score_invalid_credit_amount(self, client, sample_applicant):
        """Test scoring with invalid credit amount."""
        sample_applicant["credit_amount"] = -1000  # Negative
        response = client.post("/score", json=sample_applicant)

        # Should return 422 (validation error)
        assert response.status_code == 422

    def test_score_missing_required_field(self, client):
        """Test scoring with missing required field."""
        incomplete_data = {
            "duration_months": 24,
            # Missing credit_amount and other required fields
        }
        response = client.post("/score", json=incomplete_data)
        assert response.status_code == 422


class TestBatchScoringEndpoint:
    """Tests for batch scoring endpoint."""

    def test_batch_score(self, client, sample_applicant):
        """Test batch scoring with multiple applicants."""
        batch_request = {"applicants": [sample_applicant, sample_applicant]}

        response = client.post("/score/batch", json=batch_request)
        assert response.status_code == 200

        data = response.json()
        assert "batch_id" in data
        assert "results" in data
        assert "total_processed" in data
        assert "processing_time_ms" in data

        assert data["total_processed"] == 2
        assert len(data["results"]) == 2

    def test_batch_score_single(self, client, sample_applicant):
        """Test batch scoring with single applicant."""
        batch_request = {"applicants": [sample_applicant]}

        response = client.post("/score/batch", json=batch_request)
        assert response.status_code == 200

        data = response.json()
        assert data["total_processed"] == 1

    def test_batch_score_empty(self, client):
        """Test batch scoring with empty list."""
        batch_request = {"applicants": []}

        response = client.post("/score/batch", json=batch_request)
        # Should return 422 because min_length=1
        assert response.status_code == 422


class TestRiskBandsEndpoint:
    """Tests for risk bands configuration endpoint."""

    def test_get_risk_bands(self, client):
        """Test retrieving risk band configuration."""
        response = client.get("/risk_bands")
        assert response.status_code == 200

        data = response.json()
        assert "risk_bands" in data

        risk_bands = data["risk_bands"]
        assert "low_risk" in risk_bands or len(risk_bands) > 0


class TestSchemaValidation:
    """Tests for Pydantic schema validation."""

    def test_applicant_features_valid(self):
        """Test valid applicant features."""
        data = {
            "duration_months": 24,
            "credit_amount": 5000.0,
            "installment_rate": 3,
            "residence_duration": 2,
            "age": 35,
            "existing_credits": 1,
            "num_dependents": 1,
        }

        applicant = ApplicantFeatures(**data)
        assert applicant.age == 35
        assert applicant.credit_amount == 5000.0

    def test_applicant_features_age_validation(self):
        """Test age validation in schema."""
        data = {
            "duration_months": 24,
            "credit_amount": 5000.0,
            "installment_rate": 3,
            "residence_duration": 2,
            "age": 150,  # Invalid age
            "existing_credits": 1,
            "num_dependents": 1,
        }

        with pytest.raises(Exception):  # Pydantic validation error
            ApplicantFeatures(**data)

    def test_applicant_features_optional_fields(self):
        """Test optional fields can be None."""
        data = {
            "duration_months": 24,
            "credit_amount": 5000.0,
            "installment_rate": 3,
            "residence_duration": 2,
            "age": 35,
            "existing_credits": 1,
            "num_dependents": 1,
            "checking_account_status": None,  # Optional
        }

        applicant = ApplicantFeatures(**data)
        assert applicant.checking_account_status is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
