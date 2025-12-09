"""
Credit Scorecard Module for Credit Risk Scoring System.

This module handles the conversion of probability of default (PD)
to credit scores and risk band classification.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml
from loguru import logger


@dataclass
class ScoreResult:
    """Result of credit scoring for an applicant."""

    probability_of_default: float
    credit_score: int
    risk_band: str
    decision: str
    expected_default_rate: float
    score_breakdown: Optional[Dict[str, Any]] = None


def load_risk_policy(config_path: str = "configs/risk_policy.yaml") -> Dict:
    """Load risk policy configuration.

    Args:
        config_path: Path to risk policy configuration file.

    Returns:
        Dictionary containing risk policy configuration.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class CreditScorecard:
    """Credit scorecard for converting PD to scores and risk bands."""

    def __init__(self, config_path: str = "configs/risk_policy.yaml"):
        """Initialize the scorecard.

        Args:
            config_path: Path to risk policy configuration.
        """
        self.config = load_risk_policy(config_path)
        self._setup_score_conversion()
        self._setup_risk_bands()

    def _setup_score_conversion(self) -> None:
        """Set up score conversion parameters using PDO methodology."""
        score_config = self.config["score_conversion"]

        self.base_score = score_config["base_score"]
        self.base_odds = score_config["base_odds"]
        self.pdo = score_config["pdo"]
        self.min_score = score_config["min_score"]
        self.max_score = score_config["max_score"]

        # Calculate factor and offset
        # Score = Offset - Factor * ln(odds)
        # where odds = (1 - PD) / PD
        self.factor = self.pdo / np.log(2)
        self.offset = self.base_score + self.factor * np.log(self.base_odds)

        logger.debug(f"Score conversion: factor={self.factor:.2f}, offset={self.offset:.2f}")

    def _setup_risk_bands(self) -> None:
        """Set up risk band definitions."""
        self.risk_bands = self.config["risk_bands"]
        self.pd_thresholds = self.config.get("pd_thresholds", {})

    def pd_to_score(self, pd: float) -> int:
        """Convert probability of default to credit score.

        Uses the Points to Double the Odds (PDO) methodology:
        Score = Offset - Factor * ln(odds)
        where odds = (1 - PD) / PD

        Args:
            pd: Probability of default (0 to 1).

        Returns:
            Credit score (clamped to min_score, max_score range).
        """
        # Handle edge cases
        if pd <= 0:
            return self.max_score
        if pd >= 1:
            return self.min_score

        # Calculate odds
        odds = (1 - pd) / pd

        # Calculate score
        score = self.offset - self.factor * np.log(odds)

        # Clamp to range
        score = int(round(np.clip(score, self.min_score, self.max_score)))

        return score

    def score_to_pd(self, score: int) -> float:
        """Convert credit score back to probability of default.

        Args:
            score: Credit score.

        Returns:
            Probability of default.
        """
        # Calculate log odds from score
        log_odds = (self.offset - score) / self.factor

        # Convert to probability
        odds = np.exp(log_odds)
        pd = 1 / (1 + odds)

        return pd

    def get_risk_band(self, score: int) -> Tuple[str, Dict]:
        """Determine risk band based on credit score.

        Args:
            score: Credit score.

        Returns:
            Tuple of (risk band name, band configuration).
        """
        for band_name, band_config in self.risk_bands.items():
            if band_config["min_score"] <= score <= band_config["max_score"]:
                return band_name, band_config

        # Default to high risk if no band matches
        return "high_risk", self.risk_bands["high_risk"]

    def get_risk_band_by_pd(self, pd: float) -> Tuple[str, Dict]:
        """Determine risk band based on probability of default.

        Args:
            pd: Probability of default.

        Returns:
            Tuple of (risk band name, band configuration).
        """
        low_max = self.pd_thresholds.get("low_risk_max_pd", 0.05)
        medium_max = self.pd_thresholds.get("medium_risk_max_pd", 0.15)

        if pd <= low_max:
            return "low_risk", self.risk_bands["low_risk"]
        elif pd <= medium_max:
            return "medium_risk", self.risk_bands["medium_risk"]
        else:
            return "high_risk", self.risk_bands["high_risk"]

    def score_applicant(
        self,
        pd: float,
        applicant_data: Optional[Dict] = None,
        use_pd_thresholds: bool = False,
    ) -> ScoreResult:
        """Score an applicant and determine risk band.

        Args:
            pd: Probability of default from model.
            applicant_data: Optional applicant data for additional checks.
            use_pd_thresholds: Use PD thresholds instead of score thresholds.

        Returns:
            ScoreResult with score, risk band, and decision.
        """
        # Convert PD to score
        score = self.pd_to_score(pd)

        # Determine risk band
        if use_pd_thresholds:
            band_name, band_config = self.get_risk_band_by_pd(pd)
        else:
            band_name, band_config = self.get_risk_band(score)

        # Apply business rules if applicant data provided
        decision = band_config["decision"]
        if applicant_data:
            decision = self._apply_business_rules(decision, applicant_data)

        return ScoreResult(
            probability_of_default=pd,
            credit_score=score,
            risk_band=band_config["name"],
            decision=decision,
            expected_default_rate=band_config["expected_default_rate"],
        )

    def _apply_business_rules(self, initial_decision: str, applicant_data: Dict) -> str:
        """Apply business rules that may override the model decision.

        Args:
            initial_decision: Decision based on score.
            applicant_data: Applicant information.

        Returns:
            Final decision after applying business rules.
        """
        business_rules = self.config.get("business_rules", {})

        # Check hard decline rules
        hard_declines = business_rules.get("hard_decline", [])
        for rule in hard_declines:
            if applicant_data.get(rule, False):
                logger.info(f"Hard decline triggered by: {rule}")
                return "decline"

        # Check minimum requirements
        min_reqs = business_rules.get("min_requirements", {})

        if "age" in applicant_data:
            min_age = min_reqs.get("min_age", 18)
            max_age = min_reqs.get("max_age", 100)
            if applicant_data["age"] < min_age or applicant_data["age"] > max_age:
                logger.info(f"Age outside acceptable range: {applicant_data['age']}")
                return "decline"

        return initial_decision

    def batch_score(
        self, pd_array: np.ndarray, use_pd_thresholds: bool = False
    ) -> List[ScoreResult]:
        """Score multiple applicants.

        Args:
            pd_array: Array of probabilities of default.
            use_pd_thresholds: Use PD thresholds instead of score thresholds.

        Returns:
            List of ScoreResult objects.
        """
        results = []
        for pd in pd_array:
            results.append(self.score_applicant(pd, use_pd_thresholds=use_pd_thresholds))
        return results

    def get_score_distribution_stats(self, pd_array: np.ndarray) -> Dict[str, Any]:
        """Calculate score distribution statistics.

        Args:
            pd_array: Array of probabilities of default.

        Returns:
            Dictionary with distribution statistics.
        """
        scores = np.array([self.pd_to_score(pd) for pd in pd_array])

        # Count by risk band
        band_counts = {"low_risk": 0, "medium_risk": 0, "high_risk": 0}
        for score in scores:
            band_name, _ = self.get_risk_band(score)
            band_counts[band_name] += 1

        total = len(scores)
        band_rates = {k: v / total for k, v in band_counts.items()}

        return {
            "mean_score": float(np.mean(scores)),
            "median_score": float(np.median(scores)),
            "std_score": float(np.std(scores)),
            "min_score": int(np.min(scores)),
            "max_score": int(np.max(scores)),
            "band_counts": band_counts,
            "band_rates": band_rates,
            "approval_rate": band_rates.get("low_risk", 0),
            "decline_rate": band_rates.get("high_risk", 0),
        }

    def get_cutoff_analysis(
        self, y_true: np.ndarray, pd_array: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze performance at different score cutoffs.

        Args:
            y_true: True default labels.
            pd_array: Predicted probabilities of default.

        Returns:
            Dictionary with cutoff analysis results.
        """
        scores = np.array([self.pd_to_score(pd) for pd in pd_array])

        results = []
        cutoffs = range(self.min_score, self.max_score + 1, 50)

        for cutoff in cutoffs:
            approved = scores >= cutoff
            n_approved = np.sum(approved)
            n_total = len(scores)

            if n_approved > 0:
                default_rate_approved = np.mean(y_true[approved])
            else:
                default_rate_approved = 0

            results.append(
                {
                    "cutoff_score": cutoff,
                    "approval_rate": n_approved / n_total,
                    "n_approved": int(n_approved),
                    "default_rate": default_rate_approved,
                }
            )

        return {"cutoff_analysis": results}


def create_scorecard(config_path: str = "configs/risk_policy.yaml") -> CreditScorecard:
    """Factory function to create a scorecard instance.

    Args:
        config_path: Path to risk policy configuration.

    Returns:
        CreditScorecard instance.
    """
    return CreditScorecard(config_path)
