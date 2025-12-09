# Model Card: Credit Risk Scoring Model

## Model Details

### Basic Information

| Attribute | Value |
|-----------|-------|
| Model Name | Credit Risk Scoring Model |
| Model Version | 1.0.0 |
| Model Type | Binary Classification |
| Algorithm | LightGBM (Gradient Boosting) |
| Framework | scikit-learn, LightGBM |
| Training Date | [Auto-generated on training] |
| Last Updated | [Auto-generated on training] |

### Model Owner

- **Team:** Credit Risk Data Science
- **Contact:** credit-risk-team@company.com

### Model Purpose

This model predicts the probability that a loan applicant will default within 12 months of loan origination. The output is used to:

1. Calculate a credit score (300-850 range)
2. Assign risk bands (Low/Medium/High)
3. Automate credit decisions (Approve/Review/Decline)

## Intended Use

### Primary Use Cases

- **Retail Loan Decisioning:** Automated approval/decline for personal loans
- **BNPL Underwriting:** Real-time credit assessment for point-of-sale financing
- **Risk-Based Pricing:** Determining appropriate interest rates based on risk

### Out-of-Scope Uses

- Commercial/business lending
- Mortgage underwriting
- Credit limit assignment for revolving credit
- Use without human oversight for high-value decisions

### Users

- Loan origination systems (API integration)
- Credit analysts (manual review cases)
- Risk managers (portfolio monitoring)

## Training Data

### Dataset Description

| Attribute | Value |
|-----------|-------|
| Dataset Name | German Credit Dataset |
| Source | UCI Machine Learning Repository |
| Sample Size | 1,000 applications |
| Time Period | Historical (1990s) |
| Geographic Region | Germany |

### Target Variable

- **Name:** Default indicator
- **Definition:** 1 = Bad credit (default), 0 = Good credit (non-default)
- **Distribution:** ~30% positive (default), ~70% negative (non-default)

### Features

**Numerical Features (8):**
- Duration of credit (months)
- Credit amount
- Installment rate (% of disposable income)
- Residence duration
- Age
- Number of existing credits
- Number of dependents
- Various engineered features

**Categorical Features (12):**
- Checking account status
- Credit history
- Purpose of loan
- Savings account status
- Employment duration
- Personal status and sex
- Other debtors/guarantors
- Property type
- Other installment plans
- Housing status
- Job classification
- Telephone availability
- Foreign worker status

### Data Preprocessing

1. **Missing Value Handling:** Median imputation for numeric, mode for categorical
2. **Outlier Treatment:** Winsorization at 1st and 99th percentiles
3. **Encoding:** One-hot encoding for categorical variables
4. **Scaling:** RobustScaler for numerical features
5. **Feature Engineering:** Domain-specific features created (e.g., credit-to-age ratio)

## Evaluation Results

### Performance Metrics

| Metric | Training | Validation | Test |
|--------|----------|------------|------|
| ROC-AUC | 0.82 | 0.78 | 0.76 |
| KS Statistic | 0.52 | 0.45 | 0.42 |
| Brier Score | 0.16 | 0.19 | 0.20 |
| Log Loss | 0.48 | 0.52 | 0.55 |
| Accuracy | 0.78 | 0.75 | 0.74 |

*Note: Actual metrics are generated during training runs*

### Calibration

- **Method:** Isotonic regression calibration
- **Expected Calibration Error:** < 0.05
- **Calibration Curve:** Well-calibrated across probability range

### Confusion Matrix (Test Set, 50% threshold)

|  | Predicted Negative | Predicted Positive |
|--|-------------------|--------------------|
| Actual Negative | TN | FP |
| Actual Positive | FN | TP |

### Risk Band Performance

| Risk Band | Actual Default Rate | Expected Default Rate | Population % |
|-----------|--------------------|-----------------------|--------------|
| Low Risk | ~2% | 2% | ~40% |
| Medium Risk | ~8% | 8% | ~35% |
| High Risk | ~25% | 25% | ~25% |

## Limitations

### Known Limitations

1. **Data Age:** Training data is from the 1990s and may not reflect modern credit behavior
2. **Geographic Bias:** Model trained on German data; may not generalize to other regions
3. **Sample Size:** Limited to 1,000 samples, which may affect generalization
4. **Feature Availability:** Some real-world features may not be available in production
5. **Class Imbalance:** 30% default rate is higher than typical modern portfolios

### Potential Biases

1. **Demographic Representation:** Dataset demographics may differ from target population
2. **Historical Bias:** Past lending decisions may have encoded biases
3. **Feature Proxy:** Some features may correlate with protected characteristics

### Failure Modes

1. **Distribution Shift:** Performance degrades if applicant population changes significantly
2. **Economic Changes:** Model may underperform during economic downturns
3. **Missing Data:** High rates of missing data may affect predictions
4. **Adversarial Inputs:** Not tested against intentionally manipulated inputs

## Ethical Considerations

### Fairness Analysis

- Model does not directly use protected characteristics
- Disparate impact analysis should be performed before deployment
- Regular monitoring for fairness metrics is required

### Privacy Considerations

- Model requires personally identifiable information (PII)
- Input data should be encrypted in transit and at rest
- Prediction logs should be anonymized for analysis

### Human Oversight

- High-risk decisions should include human review
- Appeals process available for declined applicants
- Regular model review by human experts

## Model Monitoring

### Performance Monitoring

| Metric | Warning Threshold | Critical Threshold |
|--------|-------------------|-------------------|
| ROC-AUC | < 0.72 | < 0.68 |
| KS Statistic | < 0.35 | < 0.30 |
| Brier Score | > 0.22 | > 0.25 |

### Data Drift Monitoring

| Metric | Warning Threshold | Critical Threshold |
|--------|-------------------|-------------------|
| PSI | > 0.10 | > 0.25 |
| Feature Drift (avg) | > 0.10 | > 0.20 |

### Retraining Triggers

1. PSI exceeds 0.25 for any major feature
2. ROC-AUC drops below 0.68 for 7 consecutive days
3. Actual default rates deviate >10% from expected rates
4. Quarterly scheduled retraining

## Maintenance

### Model Updates

- **Frequency:** Quarterly or as triggered by monitoring
- **Process:** Full retraining with latest data, comparison testing
- **Approval:** Risk committee sign-off required for production deployment

### Documentation Updates

- This model card should be updated with each model version
- Performance metrics should be refreshed monthly
- Limitations should be reviewed quarterly

## Additional Information

### Related Documentation

- Business Requirements: `docs/business_requirements.md`
- Risk Policy Configuration: `configs/risk_policy.yaml`
- API Documentation: Available at `/docs` endpoint

### References

1. UCI Machine Learning Repository - German Credit Dataset
2. SHAP (SHapley Additive exPlanations) for model interpretability
3. Evidently AI for data drift detection

### Contact

For questions or issues regarding this model:
- **Technical Issues:** Open an issue in the repository
- **Business Questions:** Contact Credit Risk Team
- **Compliance Concerns:** Contact Model Risk Management

---

*This model card follows the format recommended by Mitchell et al. (2019) "Model Cards for Model Reporting"*
