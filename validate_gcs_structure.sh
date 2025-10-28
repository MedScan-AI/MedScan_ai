#!/bin/bash
echo "=========================================="
echo "MedScan Vision Pipeline - Output Validation"
echo "=========================================="
echo ""

BUCKET="gs://medscan-pipeline-medscanai-476500/vision"
PARTITION="2025/10/28"

# Function to check and count files
check_path() {
    local path=$1
    local description=$2
    echo -n "Checking $description... "
    
    if gsutil ls $path &>/dev/null; then
        count=$(gsutil ls $path | wc -l)
        echo "✅ ($count items)"
        return 0
    else
        echo "❌ NOT FOUND"
        return 1
    fi
}

echo "=== 1. RAW DATA (Downloaded from Kaggle) ==="
check_path "$BUCKET/raw/tb/$PARTITION/**" "TB raw images"
check_path "$BUCKET/raw/lung_cancer/$PARTITION/**" "Lung Cancer raw images"

echo ""
echo "=== 2. PREPROCESSED DATA (224x224 JPEG) ==="
check_path "$BUCKET/preprocessed/tb/$PARTITION/**" "TB preprocessed images"
check_path "$BUCKET/preprocessed/lung_cancer/$PARTITION/**" "Lung Cancer preprocessed"

echo ""
echo "=== 3. METADATA (Patient CSVs) ==="
check_path "$BUCKET/metadata/tb/$PARTITION/tb_patients.csv" "TB metadata CSV"
check_path "$BUCKET/metadata/lung_cancer/$PARTITION/*.csv" "Lung Cancer metadata CSV"

echo ""
echo "=== 4. GREAT EXPECTATIONS OUTPUTS ==="
check_path "$BUCKET/ge_outputs/baseline/$PARTITION/**" "Baseline statistics"
check_path "$BUCKET/ge_outputs/schemas/$PARTITION/**" "Schema definitions"
check_path "$BUCKET/ge_outputs/validations/$PARTITION/**" "Validation results"
check_path "$BUCKET/ge_outputs/drift/$PARTITION/**" "Drift detection"
check_path "$BUCKET/ge_outputs/bias_analysis/$PARTITION/**" "Bias analysis"
check_path "$BUCKET/ge_outputs/eda/$PARTITION/**" "EDA reports"
check_path "$BUCKET/ge_outputs/reports/$PARTITION/**" "HTML reports"

echo ""
echo "=== 5. BIAS MITIGATION ==="
check_path "$BUCKET/metadata_mitigated/$PARTITION/**" "Mitigated metadata"

echo ""
echo "=== 6. MLFLOW ARTIFACTS ==="
check_path "$BUCKET/mlflow/$PARTITION/**" "MLflow tracking"

echo ""
echo "=========================================="
echo "Validation Complete!"
echo "=========================================="