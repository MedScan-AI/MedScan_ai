# Schema Statistics Documentation

## Overview

`schema_statistics.py` provides automated data validation and metadata tracking for medical imaging datasets using **Great Expectations** for validation and **MLflow** for experiment tracking. It performs schema inference, statistics generation, anomaly detection, and drift monitoring.

## Purpose

Ensures data quality through:
- Automated statistics generation from CSV datasets
- Schema inference with medical domain constraints
- Data validation and anomaly detection
- Drift detection between baseline and new data
- HTML visualization reports
- Complete experiment tracking in MLflow

## Usage

### Basic Command

```bash
cd Data-Pipeline
python scripts/data_preprocessing/schema_statistics.py
```

### View Results

```bash
mlflow ui
```

Then open browser to `http://localhost:5000`

## Configuration

All parameters are defined in `config/metadata.yml`:

```yaml
# Datasets to process
datasets:
  brain_tumor:
    name: "brain_tumor_mri_patients"
    path: "Data-Pipeline/data/synthetic_metadata/brain_tumor_mri_patients.csv"

# Medical domain constraints
schema_constraints:
  numerical_features:
    Age_Years: {min: 0, max: 120, required: true}
    Weight_KG: {min: 2, max: 300, required: true}
    Height_CM: {min: 40, max: 250, required: true}
  
  categorical_features:
    Gender: {allowed_values: ["Male", "Female", "Other"], required: true}
    Examination_Type: {allowed_values: ["X-ray", "CT", "MRI", "Ultrasound", "PET", "Mammography"]}

# Execution control
execution:
  process_datasets: ["brain_tumor", "lung_cancer"]
  operations:
    generate_statistics: true
    infer_schema: true
    validate_data: true
    detect_drift: true
    generate_reports: true
```

## Functionality

### Core Components

**1. SchemaStatisticsManager**
Main orchestrator class that manages the entire validation pipeline.

**2. MLflow Integration**
- Tracks all validation runs as MLflow experiments
- Logs parameters (dataset info, feature counts)
- Logs metrics (anomalies, drift indicators)
- Stores all artifacts (statistics, schemas, reports)

**3. Great Expectations**
- Generates feature statistics
- Infers data schemas
- Creates expectations for validation
- Detects anomalies

### Pipeline Operations

**1. Statistics Generation**
```python
# Computes for each dataset:
- Numerical stats: count, mean, std, min, max, missing
- Categorical stats: unique values, top frequencies, missing
```

**Output:** `{dataset}_stats.json`

**2. Schema Inference**
```python
# Creates expectations based on config:
- expect_column_values_to_be_between (Age: 0-120)
- expect_column_values_to_be_in_set (Gender: Male/Female/Other)
- expect_column_values_to_not_be_null (required fields)
- expect_column_values_to_be_unique (Patient_ID)
```

**Output:** `{dataset}_schema.json`

**3. Data Validation**
```python
# Detects anomalies:
- Out-of-range values (Age > 120)
- Invalid categories (Gender = "Unknown")
- Missing required values
- Duplicate unique values
```

**Output:** `{dataset}_validation.json`

**4. Drift Detection**
```python
# Statistical tests:
- Kolmogorov-Smirnov test (numerical features)
- Chi-square test (categorical features)
- Compares baseline (70%) vs new data (30%)
```

**Output:** `{dataset}_drift.json`

**5. HTML Reports**
```python
# Generates visualizations:
- Statistics comparison report
- Validation anomalies report
- Drift detection report
```

**Output:** `{dataset}_*.html`

### Execution Flow

1. **Initialize MLflow**: Sets up tracking URI and experiment
2. **Start Run**: Creates MLflow run for each dataset
3. **Load Data**: Reads CSV and splits into baseline/new
4. **Generate Statistics**: Computes features stats and logs to MLflow
5. **Infer Schema**: Creates expectations from domain constraints
6. **Validate Data**: Checks data against schema, logs anomalies
7. **Detect Drift**: Runs statistical tests, logs drift metrics
8. **Generate Reports**: Creates HTML visualizations
9. **End Run**: Finalizes MLflow tracking

## Domain Constraints

| Feature | Constraint | Type |
|---------|-----------|------|
| Age_Years | 0-120 | Range |
| Weight_KG | 2-300 | Range |
| Height_CM | 40-250 | Range |
| Gender | Male/Female/Other | Categorical |
| Examination_Type | 6 types (X-ray, CT, MRI, etc.) | Categorical |
| Body_Region | 7 regions (Head, Chest, etc.) | Categorical |
| Urgency_Level | Routine/Urgent/Emergent/STAT | Categorical |
| Patient_ID | Unique identifier | Unique |

## Output Structure

```
Data-Pipeline/data/
├── ge_outputs/
│   ├── baseline/              # Baseline statistics (JSON)
│   ├── new_data/              # New data statistics (JSON)
│   ├── schemas/               # Inferred schemas (JSON)
│   ├── validations/           # Validation results (JSON)
│   ├── drift/                 # Drift reports (JSON)
│   └── reports/               # HTML visualizations
└── mlmd_store/
    └── mlruns/                # MLflow tracking data
```

## MLflow Tracking

### Parameters Logged
- Dataset path and name
- Number of rows and columns
- Number of expectations/features

### Metrics Logged
- Missing values count
- Validation status (is_valid: 0 or 1)
- Number of anomalies
- Drift status (has_drift: 0 or 1)
- Number of drifted features

### Artifacts Logged
- Statistics files (JSON)
- Schema files (JSON)
- Validation results (JSON)
- Drift reports (JSON)
- HTML visualization reports

## Example Output

**Console:**
```
================================================================================
Great Expectations + MLflow Pipeline - MedScan AI
================================================================================

Processing dataset: brain_tumor
Loaded 7024 rows and 14 columns
Generating statistics for brain_tumor_mri_patients_baseline
Inferring schema for brain_tumor_mri_patients
Validating data for brain_tumor_mri_patients
No anomalies found in brain_tumor_mri_patients
Detecting drift for brain_tumor_mri_patients
No drift detected in brain_tumor_mri_patients

================================================================================
MLFLOW TRACKING SUMMARY
================================================================================
Total Runs: 2
Experiment ID: 1
Tracking URI: file:///C:/path/to/mlruns

Run: brain_tumor_mri_patients_validation
  Run ID: abc123...
  Status: FINISHED
  Metrics: 6 recorded
    brain_tumor_mri_patients_baseline_missing_values: 0.0
    brain_tumor_mri_patients_baseline_is_valid: 1.0
  Artifacts: 12 files
================================================================================
```

**MLflow UI:**
- Navigate to `http://localhost:5000`
- View all validation runs
- Compare metrics across datasets
- Download artifacts
- Analyze trends over time

## Key Features

✅ **Automated Validation**: No manual checks required  
✅ **Medical Constraints**: Domain-specific validation rules  
✅ **Anomaly Detection**: Missing values, out-of-range, invalid categories  
✅ **Drift Monitoring**: Statistical tests for distribution changes  
✅ **MLflow Tracking**: Complete experiment lineage and versioning  
✅ **HTML Reports**: Interactive visualizations for stakeholders  
✅ **Reproducible**: All parameters in configuration file  

## Anomaly Types Detected

1. **Missing Values**: Required fields with null values
2. **Out-of-Range**: Age = 150, Weight = 500
3. **Invalid Categories**: Gender = "Unknown", Examination_Type = "Laser"
4. **Type Mismatches**: Non-numeric values in Age field
5. **Uniqueness Violations**: Duplicate Patient_IDs

## Drift Detection

**Thresholds:**
```yaml
drift_detection:
  statistical_test_threshold: 0.05  # P-value threshold
```

**Interpretation:**
- **p-value < 0.05**: Significant drift detected
- **p-value >= 0.05**: No significant drift

**Tests Used:**
- **K-S Test**: Compares cumulative distributions (numerical)
- **Chi-square**: Tests independence (categorical)

## ⭐ Bias Detection and Mitigation (NEW)

### Overview

The pipeline now includes comprehensive bias detection using **data slicing** techniques to identify and mitigate unfairness across demographic subgroups. This ensures equitable model performance for all patient populations.

### Motivation

Medical AI systems must provide fair and unbiased predictions across all demographic groups. Systematic biases in training data can lead to:
- **Disparate Impact**: Different outcomes for different groups
- **Underrepresentation**: Poor performance on minority groups
- **Health Inequity**: Reduced care quality for underserved populations

### Data Slicing Approach

The pipeline analyzes data by slicing across demographic and clinical features:

#### Slicing Dimensions
1. **Age Groups**:
   - Young Adult (18-35)
   - Middle Age (36-55)
   - Senior (56-75)
   - Elderly (76-120)

2. **Gender**: Male, Female, Other

3. **Diagnosis Class**: Disease categories (TB, Cancer types, Normal)

4. **Urgency Level**: Routine, Urgent, Emergent, STAT

5. **Body Region**: Head, Chest, Abdomen, Pelvis, etc.

### Bias Detection Methods

#### 1. Demographic Parity Test
**Purpose**: Detect unequal representation across groups

**Test**: Chi-square goodness-of-fit test
```python
H0: All groups equally represented
Ha: Significant imbalance exists
Threshold: p < 0.05
```

**Example Detection**:
```
Gender distribution:
  Male: 2800 (70%)
  Female: 1000 (25%)
  Other: 200 (5%)

Chi-square: 450.32, p=0.0001
⚠️ BIAS DETECTED: Unequal gender distribution
```

#### 2. Disparate Impact Analysis
**Purpose**: Detect discrimination using 80% rule

**Rule**: Minority selection rate ≥ 80% of majority rate
```python
Disparate Impact Ratio = Minority Rate / Majority Rate
Threshold: < 0.8 indicates bias
```

**Example Detection**:
```
Diagnosis distribution by gender:
  Male TB rate: 40%
  Female TB rate: 25%
  Ratio: 0.625 < 0.8
  
⚠️ BIAS DETECTED: Disparate impact for Female vs Male
```

#### 3. Statistical Independence Test
**Purpose**: Check if diagnosis depends on sensitive features

**Test**: Chi-square test of independence
```python
H0: Diagnosis independent of demographic feature
Ha: Significant dependence exists
Threshold: p < 0.05
```

**Example Detection**:
```
Contingency Table: Age_Group x Diagnosis_Class
Chi-square: 87.45, p=0.0023

⚠️ BIAS DETECTED: Diagnosis depends on age group
```

#### 4. Effect Size Analysis
**Purpose**: Measure magnitude of differences across groups

**Metric**: Cohen's d
```python
d = (mean1 - mean2) / pooled_std
Small: d = 0.2
Medium: d = 0.5
Large: d = 0.8
Threshold: d > 0.3
```

**Example Detection**:
```
Age difference between diagnosis groups:
  Normal: mean=45, std=12
  TB: mean=58, std=14
  Cohen's d: 0.89 (large effect)
  
⚠️ BIAS DETECTED: Large age difference between groups
```

### Configuration

Enable and configure bias detection in `config/metadata.yml`:

```yaml
bias_detection:
  # Enable bias detection
  enable: true
  
  # Output directory
  output_dir: "data/ge_outputs/bias_analysis"
  
  # Features to slice by
  slicing_features:
    - "Gender"
    - "Age_Years"  # Auto-converted to Age_Group
    - "Diagnosis_Class"
    - "Urgency_Level"
    - "Body_Region"
  
  # Age grouping bins
  age_bins:
    - name: "Young Adult"
      min: 18
      max: 35
    - name: "Middle Age"
      min: 36
      max: 55
    - name: "Senior"
      min: 56
      max: 75
    - name: "Elderly"
      min: 76
      max: 120
  
  # Statistical test thresholds
  statistical_tests:
    chi_square_threshold: 0.05      # P-value significance
    effect_size_threshold: 0.3      # Cohen's d threshold
    min_slice_size: 30              # Minimum samples per slice
  
  # Bias mitigation strategies
  mitigation:
    enable: true
    
    strategies:
      - "resample_underrepresented"  # Oversample minority groups
      - "class_weights"              # Compute balanced weights
      - "stratified_split"           # Proportional splits
    
    resampling:
      target_ratio: 0.8   # Target minority:majority ratio
      method: "oversample"  # oversample, undersample, or smote
    
    generate_report: true
  
  # Fairness metrics to compute
  fairness_metrics:
    - "demographic_parity"    # Equal representation
    - "equal_opportunity"     # Equal positive outcomes
    - "statistical_parity"    # Independence test
    - "disparate_impact"      # 80% rule
```

### Pipeline Integration

Add to execution operations:

```yaml
execution:
  operations:
    generate_statistics: true
    infer_schema: true
    perform_eda: true
    validate_data: true
    detect_drift: true
    detect_bias: true        # NEW: Enable bias detection
    generate_reports: true
```

### Bias Detection Workflow

```
1. Load Data
   └─> Apply to latest partition (or baseline if only one)

2. Create Age Groups
   └─> Convert Age_Years to categorical Age_Group

3. Slice Data
   ├─> By Gender (Male, Female, Other)
   ├─> By Age_Group (Young Adult, Middle Age, Senior, Elderly)
   ├─> By Diagnosis_Class (Normal, TB, Cancer types)
   ├─> By Urgency_Level (Routine, Urgent, Emergent, STAT)
   └─> By Body_Region (Head, Chest, Abdomen, etc.)

4. Statistical Tests (per slice)
   ├─> Chi-square: Test for equal distribution
   ├─> Disparate Impact: Check 80% rule
   ├─> Independence: Test diagnosis vs feature
   └─> Cohen's d: Measure effect sizes

5. Detect Biases
   ├─> Unequal distribution (chi-square p<0.05)
   ├─> Disparate impact (ratio<0.8)
   ├─> Diagnosis dependence (independence p<0.05)
   └─> Numeric differences (Cohen's d>0.3)

6. Compute Fairness Metrics
   ├─> Demographic parity deviation
   ├─> Statistical parity chi-square
   └─> Independence tests

7. Generate Recommendations
   └─> Actionable steps for bias mitigation

8. Save Results
   ├─> JSON: {dataset}_bias_analysis.json
   └─> HTML: {dataset}_bias_report.html
```

### Bias Mitigation Strategies

#### 1. Resample Underrepresented Groups

**Method**: Oversample minority groups to achieve target ratio

```python
# Example: Gender imbalance
Original:
  Male: 2800 (70%)
  Female: 1000 (25%)
  Other: 200 (5%)

After Resampling (target_ratio=0.8):
  Male: 2800 (baseline)
  Female: 2240 (80% of Male)
  Other: 2240 (80% of Male)
  
Total: 2800 + 2240 + 2240 = 7280
Added: 1240 + 2040 = 3280 samples
```

**Output**: `{dataset}_mitigated.csv` with balanced data

#### 2. Compute Class Weights

**Method**: Calculate inverse frequency weights for model training

```python
# Balanced class weights
class_weights = {
    'Normal': total / (n_classes * count_normal),
    'TB': total / (n_classes * count_tb),
    ...
}

# Example output:
{
    'Normal': 0.714,
    'TB': 1.429
}
```

**Usage**: Pass weights to model training:
```python
model.fit(X, y, class_weight=class_weights)
```

#### 3. Stratified Split Recommendations

**Method**: Identify features requiring stratification

```python
# Recommendation example:
stratified_split_recommendations:
  stratify_by: ["Gender", "Age_Group", "Diagnosis_Class"]
  description: "Use these features for stratified train/val/test splits"
```

**Usage**: Apply in train/test split:
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    stratify=df[['Gender', 'Age_Group', 'Diagnosis_Class']],
    random_state=42
)
```

### Mitigation Effectiveness Evaluation

After applying mitigation strategies, the pipeline re-runs bias detection:

```python
# Effectiveness metrics
{
    'original_bias_count': 8,
    'mitigated_bias_count': 3,
    'reduction': 5,
    'reduction_percentage': 62.5
}
```

**Interpretation**:
- **>50% reduction**: Effective mitigation
- **25-50% reduction**: Moderate improvement
- **<25% reduction**: Consider additional strategies

### Output Files

```
data/ge_outputs/bias_analysis/YYYY/MM/DD/
├── {dataset}_bias_analysis.json      # Detailed bias analysis
├── {dataset}_bias_mitigation.json    # Mitigation report
├── {dataset}_bias_report.html        # Interactive HTML report
├── {dataset}_mitigated.csv           # Resampled dataset
└── {dataset}_post_mitigation_bias.json  # Re-evaluation
```

### HTML Bias Report

The comprehensive HTML report includes:

#### Section 1: Overall Status
- Total samples analyzed
- Number of significant biases detected
- Features analyzed
- Overall bias status (✓ or ⚠️)

#### Section 2: Data Slicing Analysis
Per feature:
- Distribution table (slice counts and proportions)
- Statistical test results (chi-square, independence)
- Bias status indicator

#### Section 3: Significant Biases
Table of all detected biases:
- Feature name
- Bias type (unequal_distribution, disparate_impact, etc.)
- Detailed description
- Statistical evidence

#### Section 4: Recommendations
Actionable steps:
- Resampling strategies
- Fairness constraints
- Monitoring guidelines
- Model training recommendations

#### Section 5: Mitigation Results (if applied)
- Strategies applied
- Sample counts (before/after)
- Resampling modifications
- Effectiveness metrics

#### Section 6: Fairness Metrics
- Demographic parity scores
- Statistical parity tests
- Disparate impact ratios
- Overall fairness assessment

### MLflow Tracking

Bias detection metrics logged to MLflow:

**Metrics**:
```python
{
    '{dataset}_bias_detected': 1.0 or 0.0,
    '{dataset}_num_significant_biases': int,
    '{dataset}_num_slices_analyzed': int,
    '{dataset}_bias_reduction_pct': float,
    '{dataset}_samples_added': int
}
```

**Parameters**:
```python
{
    'slicing_features': ['Gender', 'Age_Group', ...],
    'chi_square_threshold': 0.05,
    'effect_size_threshold': 0.3,
    'mitigation_enabled': True
}
```

**Artifacts**:
- Bias analysis JSON
- Mitigation report JSON
- HTML report
- Mitigated CSV (if resampling applied)

### Best Practices

#### 1. Regular Monitoring
```bash
# Run bias detection with each new data batch
python scripts/data_preprocessing/schema_statistics.py --config config/metadata.yml
```

#### 2. Review HTML Reports
- Examine distribution disparities
- Check statistical test results
- Read bias descriptions carefully
- Follow mitigation recommendations

#### 3. Track Over Time
```python
# Query MLflow for bias trends
runs = mlflow.search_runs(
    experiment_ids=['1'],
    filter_string="params.dataset='tb_patients'"
)
bias_trend = runs['metrics.tb_patients_num_significant_biases']
```

#### 4. Stratified Sampling
```python
# Always use stratified splits
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in skf.split(X, stratify_by):
    # Train with balanced folds
```

#### 5. Apply Class Weights
```python
# Use computed weights in training
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(
    class_weight=class_weights,  # From mitigation report
    random_state=42
)
```

#### 6. Slice-Specific Evaluation
```python
# Test model on each demographic slice
for gender in ['Male', 'Female', 'Other']:
    slice_data = test_data[test_data['Gender'] == gender]
    accuracy = model.score(slice_data[features], slice_data['target'])
    print(f"{gender} accuracy: {accuracy:.3f}")
```

### Example Console Output

```
================================================================================
Performing bias detection via data slicing for tb_patients
================================================================================
Analyzing slices for feature: Gender
  Chi-square test: statistic=127.43, p=0.0001 (significant)
  ⚠️ Unequal distribution across Gender slices
  Disparate impact: Female_vs_Male ratio=0.714 < 0.8
  ⚠️ Disparate impact detected

Analyzing slices for feature: Age_Group
  Chi-square test: statistic=45.67, p=0.0342 (significant)
  Independence test: chi2=89.23, p=0.0012 (significant)
  ⚠️ Diagnosis depends on Age_Group
  
Analyzing slices for feature: Diagnosis_Class
  No bias detected (p=0.234)

⚠️ Bias detected in tb_patients
  - Unequal distribution across Gender slices (p=0.0001)
  - Disparate impact: Female_vs_Male (ratio=0.71, <0.8)
  - Diagnosis depends on Age_Group (p=0.0012)

Applying bias mitigation strategies for tb_patients
  Resampled 2 groups in Gender
    Gender_Female: 1000 → 2240 (added 1240)
    Gender_Other: 200 → 2240 (added 2040)
  Computed class weights: {'Normal': 0.85, 'TB': 1.15}

Evaluating mitigation effectiveness...
  Original biases: 8
  Remaining biases: 3
  Reduction: 62.5%
  ✓ Effective bias mitigation

Bias analysis saved to: data/ge_outputs/bias_analysis/2025/10/24/tb_patients_bias_analysis.json
Bias mitigation report saved to: data/ge_outputs/bias_analysis/2025/10/24/tb_patients_bias_mitigation.json
Bias analysis HTML report saved to: data/ge_outputs/bias_analysis/2025/10/24/tb_patients_bias_report.html
```

### References & Standards

The bias detection implementation follows industry best practices:

#### Tools & Frameworks
- **Fairlearn** (Microsoft): Fairness assessment and mitigation
- **AIF360** (IBM): AI Fairness 360 toolkit
- **TensorFlow Model Analysis (TFMA)**: Model fairness evaluation
- **What-If Tool**: Interactive bias exploration

#### Standards & Guidelines
- **80% Rule**: EEOC's disparate impact standard
- **GDPR**: Right to explanation and non-discrimination
- **IEEE P7003**: Algorithmic bias considerations
- **FDA**: Medical device bias guidance

#### Academic References
- Mehrabi et al. (2021): "A Survey on Bias and Fairness in Machine Learning"
- Chouldechova & Roth (2020): "A snapshot of the frontiers of fairness in machine learning"
- Obermeyer et al. (2019): "Dissecting racial bias in healthcare algorithms"

## Use Cases

1. **Data Quality Assurance**: Validate new data batches before processing
2. **Schema Enforcement**: Ensure data meets medical domain requirements
3. **Drift Monitoring**: Detect distribution changes for model retraining
4. **Bias Detection & Mitigation**: Ensure fairness across demographic groups ⭐
5. **Compliance**: Document data validation for regulatory requirements
6. **Debugging**: Identify data issues early in pipeline

## Prerequisites

- CSV files in `data/synthetic_metadata/`
- Configuration: `config/metadata.yml`
- Required packages: `great-expectations`, `mlflow`, `scipy`, `pandas`, `numpy`

## Integration

Part of the data preprocessing pipeline:
1. **Data Acquisition** → Images downloaded
2. **Preprocessing** → Images standardized
3. **Synthetic Metadata** → Patient data generated
4. **Schema Validation** ← **This module**
5. **Bias Detection** ← **NEW**
6. **Model Training** → Fair, clean data used

---

**Version:** 2.0  
**Dependencies:** Great Expectations, MLflow, SciPy, Pandas, NumPy  
**Author:** MedScan AI Team

