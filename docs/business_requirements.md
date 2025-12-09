# Business Requirements Document

## Credit Risk Scoring System

### 1. Executive Summary

This document outlines the business requirements for a production-grade Credit Risk Scoring System designed for retail loan and BNPL (Buy Now Pay Later) customers. The system predicts the probability of default (PD) and converts it into a human-friendly credit score for automated decision-making.

### 2. Problem Statement

**Objective:** Develop a binary classification model to predict whether a customer will default on their loan obligation within a 12-month horizon.

**Business Need:**
- Automate credit decisions to reduce manual review overhead
- Maintain consistent risk assessment across all applications
- Provide explainable decisions for regulatory compliance
- Enable risk-based pricing strategies

### 3. Target Definition

**Target Variable:** Default (binary: 0 = Non-default, 1 = Default)

**Definition of Default:**
- Payment is 90+ days past due
- Account is charged off or sent to collections
- Borrower files for bankruptcy affecting the loan

**Prediction Horizon:** 12 months from loan origination

**Observation Window:** Historical performance data from the past 24 months

### 4. Dataset

**Primary Dataset:** German Credit Dataset (UCI Machine Learning Repository)

**Dataset Characteristics:**
- 1,000 loan applications
- 20 input features
- Binary outcome (Good/Bad credit risk)
- Class distribution: ~70% Good, ~30% Bad

**Mapping to Real-World Lending:**

| Dataset Feature | Real-World Equivalent |
|-----------------|----------------------|
| checking_account_status | Current account balance tier |
| duration_months | Loan term |
| credit_history | Bureau credit history |
| purpose | Loan purpose |
| credit_amount | Requested loan amount |
| savings_account | Savings balance tier |
| employment_duration | Employment tenure |
| installment_rate | Debt service ratio indicator |
| age | Applicant age |
| housing | Housing status |

**Limitations:**
- Dataset is relatively small (1,000 records)
- Data is from a different era; feature distributions may differ from modern applicants
- Limited demographic diversity

### 5. Model Outputs

#### 5.1 Probability of Default (PD)

- **Range:** 0.0 to 1.0
- **Interpretation:** Likelihood that the applicant will default within 12 months
- **Calibration:** Model probabilities are calibrated using isotonic regression to ensure reliability

#### 5.2 Credit Score

- **Range:** 300 to 850
- **Methodology:** Points-to-Double-the-Odds (PDO) approach
- **Parameters:**
  - Base Score: 600
  - Base Odds: 50:1 (at base score)
  - PDO: 20 points

**Score Interpretation:**
- Higher scores indicate lower risk
- Every 20-point increase represents halving of default odds

### 6. Risk Bands and Decision Thresholds

| Risk Band | Score Range | PD Range | Decision | Expected Default Rate |
|-----------|-------------|----------|----------|----------------------|
| Low Risk | 700-850 | ≤5% | Auto-Approve | ~2% |
| Medium Risk | 600-699 | 5%-15% | Manual Review | ~8% |
| High Risk | 300-599 | >15% | Decline | ~25% |

#### 6.1 Decision Actions

**Auto-Approve (Low Risk):**
- Automatic loan approval
- Standard terms and pricing
- No manual intervention required

**Manual Review (Medium Risk):**
- Flag for underwriter review
- May require additional documentation
- Potential for conditional approval with modified terms

**Decline (High Risk):**
- Automatic decline
- Adverse action notice with reason codes
- Option to appeal with additional information

### 7. Business Rules and Overrides

#### 7.1 Hard Decline Rules (Regardless of Score)

- Active bankruptcy
- Fraud alert on file
- Identity not verified
- Age below 18 or above 75

#### 7.2 Risk Modifiers

- Existing customer status (potential upgrade)
- Debt-to-income ratio > 50% (potential downgrade)
- Recent delinquency patterns

### 8. Key Performance Indicators (KPIs)

#### 8.1 Model Performance

| Metric | Minimum Acceptable | Target |
|--------|-------------------|--------|
| ROC-AUC | 0.70 | 0.75+ |
| KS Statistic | 0.30 | 0.40+ |
| Gini Coefficient | 0.40 | 0.50+ |
| Brier Score | <0.25 | <0.20 |

#### 8.2 Business Metrics

- Approval Rate: 55-65%
- Default Rate (Approved Portfolio): <5%
- Manual Review Rate: 15-25%

### 9. Regulatory Considerations

#### 9.1 Fair Lending Requirements

- Model must not use prohibited factors (race, religion, national origin)
- Disparate impact analysis required before deployment
- Adverse action reasons must be provided to declined applicants

#### 9.2 Explainability Requirements

- Global feature importance documentation
- Individual prediction explanations available on request
- Model documentation for regulatory examination

#### 9.3 Model Governance

- Quarterly model performance reviews
- Annual full model validation
- Documentation of all model changes

### 10. Assumptions and Constraints

#### 10.1 Assumptions

1. Historical data is representative of future applicants
2. Economic conditions remain relatively stable
3. Target definition is consistent over time
4. Data quality is maintained in production

#### 10.2 Constraints

1. Model must make decisions in <500ms
2. System must handle 1,000+ applications per day
3. Model must be explainable (no pure black-box)
4. Retraining must be possible without service interruption

### 11. Success Criteria

1. Model deployed to production within defined timeline
2. All KPIs meet minimum acceptable thresholds
3. System passes regulatory compliance review
4. Stakeholder approval of risk band calibration
5. Successful integration with existing loan origination system

### 12. Stakeholders

| Role | Responsibility |
|------|----------------|
| Credit Risk | Define risk appetite and thresholds |
| Data Science | Model development and monitoring |
| Engineering | System integration and deployment |
| Compliance | Regulatory requirements and fair lending |
| Operations | Manual review process and escalations |

### 13. Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2024-01-01 | Credit Risk Team | Initial document |
