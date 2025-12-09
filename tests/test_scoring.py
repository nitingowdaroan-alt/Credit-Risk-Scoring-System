"""
Unit tests for credit scoring module.
"""

import numpy as np
import pytest

from src.scoring.scorecard import CreditScorecard, ScoreResult


class TestCreditScorecard:
    """Tests for credit scorecard functionality."""

    @pytest.fixture
    def scorecard(self):
        """Create scorecard instance for testing."""
        return CreditScorecard()

    def test_pd_to_score_low_pd(self, scorecard):
        """Test score conversion for low probability of default."""
        # Low PD should result in high score
        score = scorecard.pd_to_score(0.01)
        assert score > 700  # Low risk

    def test_pd_to_score_high_pd(self, scorecard):
        """Test score conversion for high probability of default."""
        # High PD should result in low score
        score = scorecard.pd_to_score(0.50)
        assert score < 500  # High risk

    def test_pd_to_score_range(self, scorecard):
        """Test that scores fall within expected range."""
        for pd in [0.01, 0.05, 0.10, 0.20, 0.50, 0.80]:
            score = scorecard.pd_to_score(pd)
            assert scorecard.min_score <= score <= scorecard.max_score

    def test_pd_to_score_monotonic(self, scorecard):
        """Test that scores decrease as PD increases."""
        pds = [0.01, 0.05, 0.10, 0.20, 0.50]
        scores = [scorecard.pd_to_score(pd) for pd in pds]

        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], f"Score should decrease as PD increases"

    def test_score_to_pd_inverse(self, scorecard):
        """Test that score_to_pd is inverse of pd_to_score."""
        original_pd = 0.15
        score = scorecard.pd_to_score(original_pd)
        recovered_pd = scorecard.score_to_pd(score)

        # Allow small numerical error due to rounding
        assert abs(original_pd - recovered_pd) < 0.01

    def test_pd_edge_cases(self, scorecard):
        """Test edge cases for PD values."""
        # PD = 0 should give max score
        assert scorecard.pd_to_score(0.0) == scorecard.max_score

        # PD = 1 should give min score
        assert scorecard.pd_to_score(1.0) == scorecard.min_score

    def test_get_risk_band_low(self, scorecard):
        """Test risk band assignment for low risk."""
        band_name, band_config = scorecard.get_risk_band(750)
        assert band_name == "low_risk"
        assert band_config["decision"] == "auto_approve"

    def test_get_risk_band_medium(self, scorecard):
        """Test risk band assignment for medium risk."""
        band_name, band_config = scorecard.get_risk_band(650)
        assert band_name == "medium_risk"
        assert band_config["decision"] == "manual_review"

    def test_get_risk_band_high(self, scorecard):
        """Test risk band assignment for high risk."""
        band_name, band_config = scorecard.get_risk_band(450)
        assert band_name == "high_risk"
        assert band_config["decision"] == "decline"

    def test_score_applicant(self, scorecard):
        """Test complete applicant scoring."""
        result = scorecard.score_applicant(0.08)

        assert isinstance(result, ScoreResult)
        assert result.probability_of_default == 0.08
        assert scorecard.min_score <= result.credit_score <= scorecard.max_score
        assert result.risk_band in ["Low Risk", "Medium Risk", "High Risk"]
        assert result.decision in ["auto_approve", "manual_review", "decline"]

    def test_score_applicant_with_age(self, scorecard):
        """Test applicant scoring with age validation."""
        # Valid age
        result = scorecard.score_applicant(0.10, applicant_data={"age": 35})
        assert result.decision in ["auto_approve", "manual_review", "decline"]

        # Invalid age (too young)
        result = scorecard.score_applicant(0.10, applicant_data={"age": 16})
        assert result.decision == "decline"

    def test_batch_score(self, scorecard):
        """Test batch scoring."""
        pds = np.array([0.02, 0.08, 0.25, 0.50])
        results = scorecard.batch_score(pds)

        assert len(results) == 4
        assert all(isinstance(r, ScoreResult) for r in results)

    def test_score_distribution_stats(self, scorecard):
        """Test score distribution statistics."""
        pds = np.array([0.02, 0.05, 0.08, 0.12, 0.20, 0.35, 0.50])
        stats = scorecard.get_score_distribution_stats(pds)

        assert "mean_score" in stats
        assert "median_score" in stats
        assert "band_counts" in stats
        assert "band_rates" in stats

        # Band rates should sum to 1
        total_rate = sum(stats["band_rates"].values())
        assert abs(total_rate - 1.0) < 0.001

    def test_cutoff_analysis(self, scorecard):
        """Test cutoff analysis."""
        y_true = np.array([0, 0, 0, 1, 1, 0, 1])
        pds = np.array([0.02, 0.05, 0.08, 0.20, 0.35, 0.10, 0.50])

        analysis = scorecard.get_cutoff_analysis(y_true, pds)

        assert "cutoff_analysis" in analysis
        assert len(analysis["cutoff_analysis"]) > 0

        for cutoff_result in analysis["cutoff_analysis"]:
            assert "cutoff_score" in cutoff_result
            assert "approval_rate" in cutoff_result
            assert "default_rate" in cutoff_result


class TestScoreResultDataclass:
    """Tests for ScoreResult dataclass."""

    def test_score_result_creation(self):
        """Test ScoreResult dataclass creation."""
        result = ScoreResult(
            probability_of_default=0.10,
            credit_score=680,
            risk_band="Medium Risk",
            decision="manual_review",
            expected_default_rate=0.08,
        )

        assert result.probability_of_default == 0.10
        assert result.credit_score == 680
        assert result.risk_band == "Medium Risk"
        assert result.decision == "manual_review"
        assert result.expected_default_rate == 0.08

    def test_score_result_with_breakdown(self):
        """Test ScoreResult with score breakdown."""
        result = ScoreResult(
            probability_of_default=0.10,
            credit_score=680,
            risk_band="Medium Risk",
            decision="manual_review",
            expected_default_rate=0.08,
            score_breakdown={"feature1": 50, "feature2": -30},
        )

        assert result.score_breakdown is not None
        assert "feature1" in result.score_breakdown


class TestScoreConsistency:
    """Tests for scoring consistency and edge cases."""

    @pytest.fixture
    def scorecard(self):
        return CreditScorecard()

    def test_consistent_scoring(self, scorecard):
        """Test that same PD always produces same score."""
        pd = 0.15

        scores = [scorecard.pd_to_score(pd) for _ in range(10)]

        assert all(s == scores[0] for s in scores)

    def test_risk_band_boundaries(self, scorecard):
        """Test risk band assignment at boundaries."""
        # Test at exact boundaries
        low_boundary = 700
        medium_low_boundary = 600

        _, low_band = scorecard.get_risk_band(low_boundary)
        assert low_band["name"] == "Low Risk"

        _, medium_band = scorecard.get_risk_band(medium_low_boundary)
        assert medium_band["name"] == "Medium Risk"

        _, high_band = scorecard.get_risk_band(medium_low_boundary - 1)
        assert high_band["name"] == "High Risk"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
