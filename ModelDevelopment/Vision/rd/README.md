# ResNet18 Training (Simplified)

## Overview

This directory contains a simplified training script for ResNet18 model. The script trains ResNet18 models for medical image classification:
- **Tuberculosis (TB) Detection**: Binary classification (Normal vs Tuberculosis)
- **Lung Cancer Detection**: Multi-class classification (6 classes)

## Features

- **Simplified**: No MLflow, bias detection, hyperparameter tuning, or early stopping
- **Basic Training**: Only training and model saving
- **Configuration-Based**: All parameters configurable via YAML config file
- **Automatic Latest Partition Detection**: Automatically finds and uses the latest data partition (YYYY/MM/DD structure)
- **Native Keras Format**: Models are saved in the modern `.keras` format

## Requirements

See `requirements.txt` for all dependencies. Key packages include:
- TensorFlow
- NumPy, Pandas, Pillow, PyYAML

Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

All training parameters are configured via `config.yml`. The config file includes:

- **Paths**: Data and output paths
- **Training Parameters**: Epochs, batch size, image size, validation split
- **Model Configuration**: Learning rate (0.01)

You can override any config value using command-line arguments (see Usage section).

## Usage

### Local Training

#### Windows (PowerShell)

```powershell
# Train ResNet18 with default config
python resnet.py

# Train with custom config file
python resnet.py --config C:\path\to\custom_config.yml

# Override specific parameters
python resnet.py --epochs 30 --batch_size 16

# Train specific dataset
python resnet.py --datasets tb

# Load latest model and continue training
python resnet.py --load_latest

# Load latest model for specific dataset
python resnet.py --datasets lung_cancer_ct_scan --load_latest

# Full example with overrides
python resnet.py `
    --config config.yml `
    --data_path ..\..\DataPipeline\data\preprocessed `
    --output_path ..\..\data `
    --epochs 8 `
    --batch_size 32 `
    --image_size 224 224 `
    --datasets tb lung_cancer_ct_scan
```

#### Windows (CMD)

```cmd
REM Train ResNet18 with default config
python resnet.py

REM Train with custom config file
python resnet.py --config C:\path\to\custom_config.yml

REM Override specific parameters
python resnet.py --epochs 30 --batch_size 16

REM Train specific dataset
python resnet.py --datasets tb

REM Load latest model and continue training
python resnet.py --load_latest

REM Load latest model for specific dataset
python resnet.py --datasets lung_cancer_ct_scan --load_latest

REM Full example with overrides (use ^ for line continuation in CMD)
python resnet.py ^
    --config config.yml ^
    --data_path ..\..\DataPipeline\data\preprocessed ^
    --output_path ..\..\data ^
    --epochs 8 ^
    --batch_size 32 ^
    --image_size 224 224 ^
    --datasets tb lung_cancer_ct_scan
```

#### Linux/Mac

```bash
# Train ResNet18 with default config
python resnet.py

# Train with custom config file
python resnet.py --config /path/to/custom_config.yml

# Override specific parameters
python resnet.py --epochs 30 --batch_size 16

# Train specific dataset
python resnet.py --datasets tb

# Load latest model and continue training
python resnet.py --load_latest

# Load latest model for specific dataset
python resnet.py --datasets lung_cancer_ct_scan --load_latest

# Full example with overrides
python resnet.py \
    --config config.yml \
    --data_path ../../DataPipeline/data/preprocessed \
    --output_path ../../data \
    --epochs 8 \
    --batch_size 32 \
    --image_size 224 224 \
    --datasets tb lung_cancer_ct_scan
```

### Docker Training

#### Build the Docker image:

**Windows (PowerShell):**
```powershell
cd ModelDevelopment\Vision\rd
docker build -t resnet18-training:latest .
```

**Windows (CMD):**
```cmd
cd ModelDevelopment\Vision\rd
docker build -t resnet18-training:latest .
```

**Linux/Mac:**
```bash
cd ModelDevelopment/Vision/rd
docker build -t resnet18-training:latest .
```

**Note**: After making changes to training scripts, you must rebuild the Docker image for changes to take effect.

#### Run training in Docker:

**Windows (PowerShell):**
```powershell
# Get project root directory (navigate to project root first)
cd ..\..\..
$PROJECT_ROOT = $PWD.Path

# Train ResNet18 - Mount the preprocessed data directory and output directory
# Override data_path and output_path to use Docker container paths
docker run --rm --cpus="4" `
    -v "${PROJECT_ROOT}\DataPipeline\data\preprocessed:/app/data/preprocessed" `
    -v "${PROJECT_ROOT}\ModelDevelopment\data:/app/data" `
    resnet18-training:latest `
    python resnet.py --config /app/config.yml --data_path /app/data/preprocessed --output_path /app/data

# With custom parameters (always override data_path and output_path for Docker)
docker run --rm --cpus="1" `
    -v "${PROJECT_ROOT}\DataPipeline\data\preprocessed:/app/data/preprocessed" `
    -v "${PROJECT_ROOT}\ModelDevelopment\data:/app/data" `
    resnet18-training:latest `
    python resnet.py --config /app/config.yml --data_path /app/data/preprocessed --output_path /app/data --epochs 10 --batch_size 16

# Train specific dataset
docker run --rm --cpus="4" `
    -v "${PROJECT_ROOT}\DataPipeline\data\preprocessed:/app/data/preprocessed" `
    -v "${PROJECT_ROOT}\ModelDevelopment\data:/app/data" `
    resnet18-training:latest `
    python resnet.py --config /app/config.yml --data_path /app/data/preprocessed --output_path /app/data --datasets tb

# Load latest model and continue training
docker run --rm --cpus="4" `
    -v "${PROJECT_ROOT}\DataPipeline\data\preprocessed:/app/data/preprocessed" `
    -v "${PROJECT_ROOT}\ModelDevelopment\data:/app/data" `
    resnet18-training:latest `
    python resnet.py --config /app/config.yml --data_path /app/data/preprocessed --output_path /app/data --load_latest

# Load latest model for specific dataset
docker run --rm --cpus="4" `
    -v "${PROJECT_ROOT}\DataPipeline\data\preprocessed:/app/data/preprocessed" `
    -v "${PROJECT_ROOT}\ModelDevelopment\data:/app/data" `
    resnet18-training:latest `
    python resnet.py --config /app/config.yml --data_path /app/data/preprocessed --output_path /app/data --datasets lung_cancer_ct_scan --load_latest
```

**Windows (CMD):**
```cmd
REM Get project root directory (navigate to project root first)
cd ..\..\..
set PROJECT_ROOT=%CD%

REM Train ResNet18 - Mount the preprocessed data directory and output directory
REM Override data_path and output_path to use Docker container paths
docker run --rm --cpus="4" ^
    -v "%PROJECT_ROOT%\DataPipeline\data\preprocessed:/app/data/preprocessed" ^
    -v "%PROJECT_ROOT%\ModelDevelopment\data:/app/data" ^
    resnet18-training:latest ^
    python resnet.py --config /app/config.yml --data_path /app/data/preprocessed --output_path /app/data

REM With custom parameters (always override data_path and output_path for Docker)
docker run --rm --cpus="1" ^
    -v "%PROJECT_ROOT%\DataPipeline\data\preprocessed:/app/data/preprocessed" ^
    -v "%PROJECT_ROOT%\ModelDevelopment\data:/app/data" ^
    resnet18-training:latest ^
    python resnet.py --config /app/config.yml --data_path /app/data/preprocessed --output_path /app/data --epochs 10 --batch_size 16

REM Train specific dataset
docker run --rm --cpus="4" ^
    -v "%PROJECT_ROOT%\DataPipeline\data\preprocessed:/app/data/preprocessed" ^
    -v "%PROJECT_ROOT%\ModelDevelopment\data:/app/data" ^
    resnet18-training:latest ^
    python resnet.py --config /app/config.yml --data_path /app/data/preprocessed --output_path /app/data --datasets tb

REM Load latest model and continue training
docker run --rm --cpus="4" ^
    -v "%PROJECT_ROOT%\DataPipeline\data\preprocessed:/app/data/preprocessed" ^
    -v "%PROJECT_ROOT%\ModelDevelopment\data:/app/data" ^
    resnet18-training:latest ^
    python resnet.py --config /app/config.yml --data_path /app/data/preprocessed --output_path /app/data --load_latest

REM Load latest model for specific dataset
docker run --rm --cpus="4" ^
    -v "%PROJECT_ROOT%\DataPipeline\data\preprocessed:/app/data/preprocessed" ^
    -v "%PROJECT_ROOT%\ModelDevelopment\data:/app/data" ^
    resnet18-training:latest ^
    python resnet.py --config /app/config.yml --data_path /app/data/preprocessed --output_path /app/data --datasets lung_cancer_ct_scan --load_latest
```

**Linux/Mac:**
```bash
# Get project root directory (navigate to project root first)
cd ../../..
PROJECT_ROOT=$(pwd)

# Train ResNet18 - Mount the preprocessed data directory and output directory
# Override data_path and output_path to use Docker container paths
docker run --rm --cpus="4" \
    -v "${PROJECT_ROOT}/DataPipeline/data/preprocessed:/app/data/preprocessed" \
    -v "${PROJECT_ROOT}/ModelDevelopment/data:/app/data" \
    resnet18-training:latest \
    python resnet.py --config /app/config.yml --data_path /app/data/preprocessed --output_path /app/data

# With custom parameters (always override data_path and output_path for Docker)
docker run --rm --cpus="1" \
    -v "${PROJECT_ROOT}/DataPipeline/data/preprocessed:/app/data/preprocessed" \
    -v "${PROJECT_ROOT}/ModelDevelopment/data:/app/data" \
    resnet18-training:latest \
    python resnet.py --config /app/config.yml --data_path /app/data/preprocessed --output_path /app/data --epochs 10 --batch_size 16

# Train specific dataset
docker run --rm --cpus="4" \
    -v "${PROJECT_ROOT}/DataPipeline/data/preprocessed:/app/data/preprocessed" \
    -v "${PROJECT_ROOT}/ModelDevelopment/data:/app/data" \
    resnet18-training:latest \
    python resnet.py --config /app/config.yml --data_path /app/data/preprocessed --output_path /app/data --datasets tb

# Load latest model and continue training
docker run --rm --cpus="4" \
    -v "${PROJECT_ROOT}/DataPipeline/data/preprocessed:/app/data/preprocessed" \
    -v "${PROJECT_ROOT}/ModelDevelopment/data:/app/data" \
    resnet18-training:latest \
    python resnet.py --config /app/config.yml --data_path /app/data/preprocessed --output_path /app/data --load_latest

# Load latest model for specific dataset
docker run --rm --cpus="4" \
    -v "${PROJECT_ROOT}/DataPipeline/data/preprocessed:/app/data/preprocessed" \
    -v "${PROJECT_ROOT}/ModelDevelopment/data:/app/data" \
    resnet18-training:latest \
    python resnet.py --config /app/config.yml --data_path /app/data/preprocessed --output_path /app/data --datasets lung_cancer_ct_scan --load_latest
```

**Important Notes for Docker:**
- **Config file**: The config file is copied into the image at `/app/config.yml`
- **Data paths**: Always override `--data_path` and `--output_path` in Docker commands to use container paths (`/app/data/preprocessed` and `/app/data`) instead of relative paths from config
- **Windows Docker paths**: Use forward slashes (`/`) in Docker volume paths even on Windows (the `-v` flag handles this automatically)
- **Project root**: Commands assume you start from `ModelDevelopment/Vision/rd` directory and navigate to project root (`cd ../../..`) before running Docker commands
- **Explicit paths**: Always specify `--config`, `--data_path`, and `--output_path` explicitly in Docker to avoid path resolution issues
- **Path variables**: The examples use `$PROJECT_ROOT` (PowerShell), `%PROJECT_ROOT%` (CMD), or `$PROJECT_ROOT` (bash) to reference the project root directory

### Command Line Arguments

All arguments are optional and will use config file values if not provided:

- `--config`: Path to config YAML file (default: `config.yml` in script directory)
- `--data_path`: Path to preprocessed data directory (overrides config)
- `--output_path`: Base path for outputs (overrides config, default: `../../data`)
- `--epochs`: Number of training epochs (overrides config)
- `--batch_size`: Batch size for training (overrides config)
- `--image_size`: Image size as height width (overrides config)
- `--datasets`: List of datasets to train (overrides config, default: `tb lung_cancer_ct_scan`)
- `--load_latest`: Load the latest saved model and continue training

## Output Structure

After training, the following structure is created under `ModelDevelopment/data/`:

```
ModelDevelopment/data/
└── models/
    └── YYYY/
        └── MM/
            └── DD/
                └── HHMMSS/
                    └── {dataset_name}_CNN_ResNet18/
                        └── CNN_ResNet18_best.keras
```

## Data Requirements

The training script expects the following directory structure:

```
DataPipeline/data/preprocessed/
├── tb/
│   └── YYYY/
│       └── MM/
│           └── DD/
│               ├── train/
│               │   ├── Normal/
│               │   └── Tuberculosis/
│               └── test/
│                   ├── Normal/
│                   └── Tuberculosis/
└── lung_cancer_ct_scan/
    └── YYYY/
        └── MM/
            └── DD/
                ├── train/
                │   ├── adenocarcinoma/
                │   ├── benign/
                │   ├── large_cell_carcinoma/
                │   ├── malignant/
                │   ├── normal/
                │   └── squamous_cell_carcinoma/
                ├── test/
                │   └── ...
                └── valid/ (optional)
                    └── ...
```

## Configuration File

The configuration file (`config.yml`) allows you to customize:

- **Training parameters**: epochs, batch_size, image_size, validation_split
- **Model configuration**: learning_rate (0.01)

Example: To change the number of epochs, edit `training.epochs` in the config file, or use `--epochs` command-line argument.

## Notes

- The script automatically finds the latest partition (most recent YYYY/MM/DD)
- If a separate `valid` directory exists, it's used for validation; otherwise, validation_split from config is used
- Images are expected to be preprocessed to the size specified in config (default: 224x224)
- No data augmentation is applied (only pixel value rescaling)
- Models are saved in Keras format (`.keras`)
- All outputs are organized by dataset and timestamp for easy tracking
- Configuration file values can be overridden via command-line arguments

## Troubleshooting

### Out of Memory Errors
- Reduce `batch_size` in config or via `--batch_size` argument
- Reduce `image_size` in config or via `--image_size` argument
- Train one dataset at a time using `--datasets tb`

### Missing Data
- Ensure preprocessed data exists in the expected location
- Check that the latest partition contains train/test directories
- Verify class directories match expected names
- Check the `data_path` in config file or `--data_path` argument

### Configuration Issues
- Verify config file exists at `ModelDevelopment/Vision/rd/config.yml`
- Check YAML syntax (indentation, quotes, etc.)
- Use `--config` argument to specify custom config file path

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (Python 3.9+ recommended)
- Verify TensorFlow installation: `python -c "import tensorflow; print(tensorflow.__version__)"`
