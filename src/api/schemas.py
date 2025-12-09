"""
Pydantic schemas for API request/response validation.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ApplicantFeatures(BaseModel):
    """Input features for credit scoring."""

    # Account and credit features
    checking_account_status: Optional[str] = Field(
        None,
        description="Status of checking account",
        examples=["negative_balance", "0_to_200", "200_plus", "no_checking"],
    )
    duration_months: int = Field(
        ..., ge=1, le=120, description="Duration of credit in months"
    )
    credit_history: Optional[str] = Field(
        None,
        description="Credit history status",
        examples=["all_paid_duly", "existing_paid_duly", "delay_in_past", "critical_account"],
    )
    purpose: Optional[str] = Field(
        None,
        description="Purpose of the credit",
        examples=["car_new", "car_used", "furniture", "radio_tv", "education", "business"],
    )
    credit_amount: float = Field(..., gt=0, description="Credit amount in currency units")

    # Savings and employment
    savings_account: Optional[str] = Field(
        None,
        description="Savings account status",
        examples=["lt_100", "100_to_500", "500_to_1000", "gte_1000", "unknown"],
    )
    employment_duration: Optional[str] = Field(
        None,
        description="Employment duration",
        examples=["unemployed", "lt_1_year", "1_to_4_years", "4_to_7_years", "gte_7_years"],
    )

    # Personal details
    installment_rate: int = Field(
        ..., ge=1, le=4, description="Installment rate in percentage of disposable income"
    )
    personal_status_sex: Optional[str] = Field(
        None, description="Personal status and sex"
    )
    other_debtors: Optional[str] = Field(
        None,
        description="Other debtors/guarantors",
        examples=["none", "co_applicant", "guarantor"],
    )
    residence_duration: int = Field(
        ..., ge=1, le=4, description="Years at current residence"
    )
    property: Optional[str] = Field(
        None,
        description="Property ownership",
        examples=["real_estate", "savings_insurance", "car_other", "unknown_none"],
    )
    age: int = Field(..., ge=18, le=100, description="Age in years")
    other_installment_plans: Optional[str] = Field(
        None,
        description="Other installment plans",
        examples=["bank", "stores", "none"],
    )
    housing: Optional[str] = Field(
        None, description="Housing status", examples=["rent", "own", "for_free"]
    )
    existing_credits: int = Field(
        ..., ge=1, le=4, description="Number of existing credits at this bank"
    )
    job: Optional[str] = Field(
        None,
        description="Job type",
        examples=["unemployed_unskilled_non_resident", "unskilled_resident",
                  "skilled_employee", "management_self_employed"],
    )
    num_dependents: int = Field(
        ..., ge=0, le=10, description="Number of dependents"
    )
    telephone: Optional[str] = Field(
        None, description="Has telephone registered", examples=["none", "yes"]
    )
    foreign_worker: Optional[str] = Field(
        None, description="Is foreign worker", examples=["yes", "no"]
    )

    class Config:
        json_schema_extra = {
            "example": {
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
        }


class ScoringResponse(BaseModel):
    """Response from credit scoring endpoint."""

    request_id: str = Field(..., description="Unique request identifier")
    probability_of_default: float = Field(
        ..., ge=0, le=1, description="Probability of default (0-1)"
    )
    credit_score: int = Field(..., ge=300, le=850, description="Credit score (300-850)")
    risk_band: str = Field(
        ..., description="Risk band classification", examples=["Low Risk", "Medium Risk", "High Risk"]
    )
    decision: str = Field(
        ..., description="Recommended decision", examples=["auto_approve", "manual_review", "decline"]
    )
    expected_default_rate: float = Field(
        ..., description="Expected default rate for this risk band"
    )
    model_version: str = Field(..., description="Model version used for scoring")
    timestamp: datetime = Field(..., description="Scoring timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "req_abc123",
                "probability_of_default": 0.08,
                "credit_score": 680,
                "risk_band": "Medium Risk",
                "decision": "manual_review",
                "expected_default_rate": 0.08,
                "model_version": "1.0.0",
                "timestamp": "2024-01-15T10:30:00Z",
            }
        }


class BatchScoringRequest(BaseModel):
    """Request for batch scoring multiple applicants."""

    applicants: List[ApplicantFeatures] = Field(
        ..., min_length=1, max_length=1000, description="List of applicants to score"
    )


class BatchScoringResponse(BaseModel):
    """Response from batch scoring endpoint."""

    batch_id: str = Field(..., description="Unique batch identifier")
    results: List[ScoringResponse] = Field(..., description="Scoring results for each applicant")
    total_processed: int = Field(..., description="Total applicants processed")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status", examples=["healthy", "unhealthy"])
    model_loaded: bool = Field(..., description="Whether model is loaded")
    timestamp: datetime = Field(..., description="Health check timestamp")


class ModelInfoResponse(BaseModel):
    """Model information response."""

    model_name: str = Field(..., description="Model name")
    model_version: str = Field(..., description="Model version")
    model_type: str = Field(..., description="Type of model")
    training_date: Optional[str] = Field(None, description="Date model was trained")
    metrics: Dict[str, float] = Field(..., description="Model performance metrics")
    feature_count: int = Field(..., description="Number of input features")
    is_calibrated: bool = Field(..., description="Whether model is calibrated")


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error details")
    request_id: Optional[str] = Field(None, description="Request ID if available")
    timestamp: datetime = Field(..., description="Error timestamp")
