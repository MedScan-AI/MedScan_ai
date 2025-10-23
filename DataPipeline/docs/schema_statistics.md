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

## Use Cases

1. **Data Quality Assurance**: Validate new data batches before processing
2. **Schema Enforcement**: Ensure data meets medical domain requirements
3. **Drift Monitoring**: Detect distribution changes for model retraining
4. **Compliance**: Document data validation for regulatory requirements
5. **Debugging**: Identify data issues early in pipeline

## Prerequisites

- CSV files in `data/synthetic_metadata/`
- Configuration: `config/metadata.yml`
- Required packages: `great-expectations`, `mlflow`, `scipy`, `pandas`

## Integration

Part of the data preprocessing pipeline:
1. **Data Acquisition** → Images downloaded
2. **Preprocessing** → Images standardized
3. **Synthetic Metadata** → Patient data generated
4. **Schema Validation** ← **This module**
5. **Model Training** → Clean data used

---

**Version:** 1.0  
**Dependencies:** Great Expectations, MLflow, SciPy, Pandas  
**Author:** MedScan AI Team

