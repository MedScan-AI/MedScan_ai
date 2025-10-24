## MedScan-AI: Data Pipeline

### Overview

The MedScan-AI Data Pipeline provides comprehensive data validation, quality monitoring, and bias detection for medical imaging datasets. It uses Great Expectations for schema validation, drift detection, and MLflow for experiment tracking.

### Features

1. **Schema Validation & Statistics**
   - Automated schema inference with domain constraints
   - Statistical analysis of numerical and categorical features
   - Anomaly detection (missing values, out-of-range, invalid categories)

2. **Drift Detection**
   - Detects statistical drift between baseline and new data
   - Uses KS test for numerical features and Chi-square for categorical
   - Automated alerts and reporting

3. **Exploratory Data Analysis (EDA)**
   - Comprehensive analysis of distributions, correlations, outliers
   - Interactive HTML reports
   - Missing data analysis

4. **⭐ Data Bias Detection (NEW)**
   - **Data Slicing**: Analyzes data across demographic features (age, gender, diagnosis)
   - **Fairness Metrics**: Computes demographic parity, disparate impact, statistical parity
   - **Bias Detection**: Uses Chi-square tests, Cohen's d effect size, and disparate impact ratios
   - **Automated Mitigation**: Resampling, class weights, stratified splits
   - **Comprehensive Documentation**: Detailed HTML reports with recommendations

### Bias Detection & Mitigation

The pipeline now includes sophisticated bias detection using data slicing techniques to ensure fairness and equity in medical AI models:

#### **Data Slicing Features**
- **Age Groups**: Young Adult (18-35), Middle Age (36-55), Senior (56-75), Elderly (76+)
- **Gender**: Male, Female, Other
- **Diagnosis Class**: Disease categories specific to each dataset
- **Urgency Level**: Routine, Urgent, Emergent, STAT
- **Body Region**: Anatomical regions being examined

#### **Statistical Tests**
- **Chi-square test**: Detects unequal distribution across slices
- **Cohen's d**: Measures effect size for numeric features between groups
- **Disparate Impact**: 80% rule for detecting discrimination
- **Independence test**: Checks if diagnosis depends on sensitive features

#### **Bias Mitigation Strategies**
1. **Resampling**: Over-sample underrepresented groups to achieve target ratio
2. **Class Weights**: Compute balanced weights for model training
3. **Stratified Splits**: Ensure proportional representation in train/val/test sets

#### **Fairness Metrics**
- Demographic Parity: Equal representation across groups
- Equal Opportunity: Equal distribution of positive outcomes
- Statistical Parity: Independence between features and outcomes
- Disparate Impact: Ratio of selection rates (80% rule)

### Configuration

All features are configured in `config/metadata.yml`:

```yaml
# Bias Detection Configuration
bias_detection:
  enable: true
  slicing_features:
    - "Gender"
    - "Age_Years"
    - "Diagnosis_Class"
  
  statistical_tests:
    chi_square_threshold: 0.05
    effect_size_threshold: 0.3
    min_slice_size: 30
  
  mitigation:
    enable: true
    strategies:
      - "resample_underrepresented"
      - "class_weights"
      - "stratified_split"
```

### Usage

Run the complete pipeline:

```bash
cd DataPipeline
python scripts/data_preprocessing/schema_statistics.py --config config/metadata.yml
```

### Output Structure

```
data/ge_outputs/
├── bias_analysis/          # NEW: Bias detection reports
│   └── YYYY/MM/DD/
│       ├── *_bias_analysis.json
│       ├── *_bias_mitigation.json
│       ├── *_bias_report.html
│       └── *_mitigated.csv
├── baseline/               # Baseline statistics
├── drift/                  # Drift detection results
├── eda/                    # Exploratory data analysis
├── reports/                # HTML visualization reports
├── schemas/                # Schema definitions
└── validations/            # Validation results
```

### Bias Analysis Report

The bias analysis HTML report includes:

1. **Overall Status**: Total samples, significant biases detected, features analyzed
2. **Data Slicing Analysis**: Distribution and statistical tests per feature
3. **Significant Biases**: Detailed list of detected biases with descriptions
4. **Recommendations**: Actionable steps for bias mitigation
5. **Fairness Metrics**: Comprehensive fairness evaluation
6. **Mitigation Results**: Effectiveness of applied strategies

### Example: Bias Detection Process

1. **Detection**: Pipeline automatically analyzes data across demographic slices
   ```
   Analyzing slices for feature: Gender
   Chi-square test: p=0.0234 (significant)
   ⚠️ Bias detected: Unequal distribution across Gender slices
   ```

2. **Mitigation**: If bias detected, applies configured strategies
   ```
   Applying resampling to balance underrepresented groups
   Resampled 2 groups in Gender
   Bias mitigation effectiveness: 67.3% reduction
   ```

3. **Documentation**: Generates comprehensive reports
   - JSON reports for programmatic access
   - HTML reports for human review
   - MLflow tracking for experiment management

### MLflow Tracking

All bias detection metrics are tracked in MLflow:

- `{dataset}_bias_detected`: Binary flag (1=bias detected)
- `{dataset}_num_significant_biases`: Count of biases
- `{dataset}_bias_reduction_pct`: Mitigation effectiveness
- `{dataset}_samples_added`: Resampling count

View in MLflow UI:
```bash
cd DataPipeline
mlflow ui --backend-store-uri file:///$(pwd)/data/mlflow_store/mlruns
```

### Best Practices

1. **Monitor Regularly**: Run bias detection with each new data batch
2. **Review Reports**: Carefully examine HTML reports for bias patterns
3. **Track Over Time**: Use MLflow to monitor bias metrics across runs
4. **Stratified Sampling**: Use recommendations for fair train/test splits
5. **Model Fairness**: Apply class weights during model training
6. **Slice-Specific Evaluation**: Test model performance on each demographic slice

### References

The bias detection implementation follows best practices from:
- **Fairlearn**: Microsoft's fairness assessment toolkit
- **AIF360**: IBM's AI Fairness 360 toolkit
- **TensorFlow Model Analysis (TFMA)**: Google's model analysis tool
- **80% Rule**: EEOC's disparate impact standard