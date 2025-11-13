# Vision Model Development

## Overview

This directory contains the training pipeline for medical image classification models. The system trains separate models for:
- **Tuberculosis (TB) Detection**: Binary classification (Normal vs Tuberculosis)
- **Lung Cancer Detection**: Multi-class classification (6 classes: adenocarcinoma, benign, large_cell_carcinoma, malignant, normal, squamous_cell_carcinoma)

## Model Architectures

The training script implements and compares three different model architectures with similar parameter counts (~22-25M parameters):

1. **CNN_ResNet50**: ResNet50 architecture trained from scratch (no pre-trained weights, ~25M parameters) with custom classification head
2. **CNN_Custom**: Custom convolutional neural network with 4 convolutional blocks
3. **ViT**: Vision Transformer using ViT-Small/16 configuration (~22M parameters) - standard architecture matching ResNet50's size

All models use standard, well-known architectures (no custom builds) and have similar parameter counts for fair comparison. The best model is automatically selected based on test accuracy.

## Features

- **Configuration-Based**: All parameters are configurable via YAML config file
- **Automatic Latest Partition Detection**: Automatically finds and uses the latest data partition (YYYY/MM/DD structure)
- **Hyperparameter Tuning**: Optional automated hyperparameter tuning using KerasTuner (RandomSearch, BayesianOptimization, Hyperband)
- **MLflow Integration**: Tracks experiments, metrics, and model artifacts
- **Model Selection**: Automatically selects the best model based on performance metrics
- **Data Augmentation**: Applies augmentation to training data for better generalization
- **Early Stopping**: Prevents overfitting with early stopping and learning rate reduction
- **Dry Run Mode**: Quick testing mode using only 64 images (use `--dry_run` flag) - limits to 2 epochs for fast validation
- **Native Keras Format**: Models are saved in the modern `.keras` format (not legacy HDF5 `.h5` format)

## Requirements

See `requirements.txt` for all dependencies. Key packages include:
- TensorFlow
- KerasTuner (for hyperparameter tuning)
- MLflow
- NumPy, Pandas, Pillow, PyYAML

Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

All training parameters are configured via `ModelDevelopment/config/vision_training.yml`. The config file includes:

- **Paths**: Data and output paths
- **Training Parameters**: Epochs, batch size, image size, validation split
- **Data Augmentation**: Rotation, shifts, flips, zoom settings
- **Model Architectures**: Configuration for ResNet50, CNN Custom, and ViT models
- **Callbacks**: Early stopping, learning rate reduction, model checkpoint settings
- **Model Selection**: Metric used for selecting the best model
- **MLflow**: Tracking configuration

You can override any config value using command-line arguments (see Usage section).

## Training Scripts

The training pipeline has been separated into three individual scripts, one for each model architecture:

1. **`train_resnet.py`** - Trains ResNet50 model
2. **`train_vit.py`** - Trains Vision Transformer (ViT) model
3. **`train_custom_cnn.py`** - Trains Custom CNN model

Each script can be run independently and supports the same command-line arguments. This allows you to:
- Train models separately for easier debugging and monitoring
- Run different models in parallel on different machines
- Focus on specific architectures without loading all models

## Usage

### Local Training

#### Windows (PowerShell)

```powershell
# Train ResNet50 with default config
python train_resnet.py

# Train ViT with default config
python train_vit.py

# Train Custom CNN with default config
python train_custom_cnn.py

# Dry run mode (quick test with only 64 images)
python train_resnet.py --dry_run

# Train with custom config file
python train_resnet.py --config C:\path\to\custom_config.yml

# Override specific parameters
python train_resnet.py --epochs 30 --batch_size 16

# Train specific dataset
python train_resnet.py --datasets tb

# Full example with overrides
python train_resnet.py `
    --config ..\config\vision_training.yml `
    --data_path ..\..\DataPipeline\data\preprocessed `
    --output_path ..\data `
    --epochs 50 `
    --batch_size 32 `
    --image_size 224 224 `
    --datasets tb lung_cancer_ct_scan
```

#### Windows (CMD)

```cmd
REM Train ResNet50 with default config
python train_resnet.py

REM Train ViT with default config
python train_vit.py

REM Train Custom CNN with default config
python train_custom_cnn.py

REM Dry run mode (quick test with only 64 images)
python train_resnet.py --dry_run

REM Train with custom config file
python train_resnet.py --config C:\path\to\custom_config.yml

REM Override specific parameters
python train_resnet.py --epochs 30 --batch_size 16

REM Train specific dataset
python train_resnet.py --datasets tb

REM Full example with overrides (use ^ for line continuation in CMD)
python train_resnet.py ^
    --config ..\config\vision_training.yml ^
    --data_path ..\..\DataPipeline\data\preprocessed ^
    --output_path ..\data ^
    --epochs 50 ^
    --batch_size 32 ^
    --image_size 224 224 ^
    --datasets tb lung_cancer_ct_scan
```

#### Linux/Mac

```bash
# Train ResNet50 with default config
python train_resnet.py

# Train ViT with default config
python train_vit.py

# Train Custom CNN with default config
python train_custom_cnn.py

# Dry run mode (quick test with only 64 images)
python train_resnet.py --dry_run

# Train with custom config file
python train_resnet.py --config /path/to/custom_config.yml

# Override specific parameters
python train_resnet.py --epochs 30 --batch_size 16

# Train specific dataset
python train_resnet.py --datasets tb

# Full example with overrides
python train_resnet.py \
    --config ../config/vision_training.yml \
    --data_path ../../DataPipeline/data/preprocessed \
    --output_path ../data \
    --epochs 50 \
    --batch_size 32 \
    --image_size 224 224 \
    --datasets tb lung_cancer_ct_scan
```

### Docker Training

#### Build the Docker image:

**Windows (PowerShell):**
```powershell
cd ModelDevelopment\Vision
docker build -t medscan-vision-training:latest .
```

**Windows (CMD):**
```cmd
cd ModelDevelopment\Vision
docker build -t medscan-vision-training:latest .
```

**Linux/Mac:**
```bash
cd ModelDevelopment/Vision
docker build -t medscan-vision-training:latest .
```

**Note**: After making changes to training scripts, you must rebuild the Docker image for changes to take effect.

#### Run training in Docker:

**Note on CPU Limits:** To limit CPU usage to 50% of one CPU core, add `--cpus="0.5"` to the docker run command. For example:
```powershell
docker run --rm --cpus="0.5" ...
```

**Windows (PowerShell):**
```powershell
# Get project root directory (navigate to project root first)
cd ..\..
$PROJECT_ROOT = $PWD.Path

# Train ResNet50 - Mount the preprocessed data directory, output directory, and config directory
# Override data_path and output_path to use Docker container paths
# Add --cpus="0.5" to limit CPU usage to 50% of one CPU core
docker run --rm --cpus="0.5" `
    -v "${PROJECT_ROOT}\DataPipeline\data\preprocessed:/app/data/preprocessed" `
    -v "${PROJECT_ROOT}\ModelDevelopment\data:/app/data" `
    -v "${PROJECT_ROOT}\ModelDevelopment\config:/app/config" `
    medscan-vision-training:latest `
    python train_resnet.py --config /app/config/vision_training.yml --data_path /app/data/preprocessed --output_path /app/data

# Train ViT
docker run --rm --cpus="0.5" `
    -v "${PROJECT_ROOT}\DataPipeline\data\preprocessed:/app/data/preprocessed" `
    -v "${PROJECT_ROOT}\ModelDevelopment\data:/app/data" `
    -v "${PROJECT_ROOT}\ModelDevelopment\config:/app/config" `
    medscan-vision-training:latest `
    python train_vit.py --config /app/config/vision_training.yml --data_path /app/data/preprocessed --output_path /app/data

# Train Custom CNN
docker run --rm --cpus="0.5" `
    -v "${PROJECT_ROOT}\DataPipeline\data\preprocessed:/app/data/preprocessed" `
    -v "${PROJECT_ROOT}\ModelDevelopment\data:/app/data" `
    -v "${PROJECT_ROOT}\ModelDevelopment\config:/app/config" `
    medscan-vision-training:latest `
    python train_custom_cnn.py --config /app/config/vision_training.yml --data_path /app/data/preprocessed --output_path /app/data

# With custom parameters (always override data_path and output_path for Docker)
docker run --rm --cpus="0.5" `
    -v "${PROJECT_ROOT}\DataPipeline\data\preprocessed:/app/data/preprocessed" `
    -v "${PROJECT_ROOT}\ModelDevelopment\data:/app/data" `
    -v "${PROJECT_ROOT}\ModelDevelopment\config:/app/config" `
    medscan-vision-training:latest `
    python train_resnet.py --config /app/config/vision_training.yml --data_path /app/data/preprocessed --output_path /app/data --epochs 30 --batch_size 16

# Dry run mode (quick test with only 64 images)
docker run --rm --cpus="0.5" `
    -v "${PROJECT_ROOT}\DataPipeline\data\preprocessed:/app/data/preprocessed" `
    -v "${PROJECT_ROOT}\ModelDevelopment\data:/app/data" `
    -v "${PROJECT_ROOT}\ModelDevelopment\config:/app/config" `
    medscan-vision-training:latest `
    python train_resnet.py --config /app/config/vision_training.yml --data_path /app/data/preprocessed --output_path /app/data --dry_run

# Dry run with ViT
docker run --rm --cpus="8.0" `
    -v "${PROJECT_ROOT}\DataPipeline\data\preprocessed:/app/data/preprocessed" `
    -v "${PROJECT_ROOT}\ModelDevelopment\data:/app/data" `
    -v "${PROJECT_ROOT}\ModelDevelopment\config:/app/config" `
    medscan-vision-training:latest `
    python train_vit.py --config /app/config/vision_training.yml --data_path /app/data/preprocessed --output_path /app/data --dry_run

# Dry run with Custom CNN
docker run --rm --cpus="0.5" `
    -v "${PROJECT_ROOT}\DataPipeline\data\preprocessed:/app/data/preprocessed" `
    -v "${PROJECT_ROOT}\ModelDevelopment\data:/app/data" `
    -v "${PROJECT_ROOT}\ModelDevelopment\config:/app/config" `
    medscan-vision-training:latest `
    python train_custom_cnn.py --config /app/config/vision_training.yml --data_path /app/data/preprocessed --output_path /app/data --dry_run

# Piping output to a file manually
docker run --rm --cpus="0.5" `
    -v "${PROJECT_ROOT}\DataPipeline\data\preprocessed:/app/data/preprocessed" `
    -v "${PROJECT_ROOT}\ModelDevelopment\data:/app/data" `
    -v "${PROJECT_ROOT}\ModelDevelopment\config:/app/config" `
    medscan-vision-training:latest `
    python train_vit.py --config /app/config/vision_training.yml --data_path /app/data/preprocessed --output_path /app/data --dry_run |
    Tee-Object -FilePath "${PROJECT_ROOT}\ModelDevelopment\data\logs\training_log.txt"
```

**Windows (CMD):**
```cmd
REM Get project root directory (navigate to project root first)
cd ..\..
set PROJECT_ROOT=%CD%

REM Train ResNet50 - Mount the preprocessed data directory, output directory, and config directory
REM Override data_path and output_path to use Docker container paths
REM Add --cpus="0.5" to limit CPU usage to 50% of one CPU core
docker run --rm --cpus="0.5" ^
    -v "%PROJECT_ROOT%\DataPipeline\data\preprocessed:/app/data/preprocessed" ^
    -v "%PROJECT_ROOT%\ModelDevelopment\data:/app/data" ^
    -v "%PROJECT_ROOT%\ModelDevelopment\config:/app/config" ^
    medscan-vision-training:latest ^
    python train_resnet.py --config /app/config/vision_training.yml --data_path /app/data/preprocessed --output_path /app/data

REM Train ViT
docker run --rm --cpus="0.5" ^
    -v "%PROJECT_ROOT%\DataPipeline\data\preprocessed:/app/data/preprocessed" ^
    -v "%PROJECT_ROOT%\ModelDevelopment\data:/app/data" ^
    -v "%PROJECT_ROOT%\ModelDevelopment\config:/app/config" ^
    medscan-vision-training:latest ^
    python train_vit.py --config /app/config/vision_training.yml --data_path /app/data/preprocessed --output_path /app/data

REM Train Custom CNN
docker run --rm --cpus="0.5" ^
    -v "%PROJECT_ROOT%\DataPipeline\data\preprocessed:/app/data/preprocessed" ^
    -v "%PROJECT_ROOT%\ModelDevelopment\data:/app/data" ^
    -v "%PROJECT_ROOT%\ModelDevelopment\config:/app/config" ^
    medscan-vision-training:latest ^
    python train_custom_cnn.py --config /app/config/vision_training.yml --data_path /app/data/preprocessed --output_path /app/data

REM With custom parameters (always override data_path and output_path for Docker)
docker run --rm --cpus="0.5" ^
    -v "%PROJECT_ROOT%\DataPipeline\data\preprocessed:/app/data/preprocessed" ^
    -v "%PROJECT_ROOT%\ModelDevelopment\data:/app/data" ^
    -v "%PROJECT_ROOT%\ModelDevelopment\config:/app/config" ^
    medscan-vision-training:latest ^
    python train_resnet.py --config /app/config/vision_training.yml --data_path /app/data/preprocessed --output_path /app/data --epochs 30 --batch_size 16

REM Dry run mode (quick test with only 64 images)
docker run --rm --cpus="0.5" ^
    -v "%PROJECT_ROOT%\DataPipeline\data\preprocessed:/app/data/preprocessed" ^
    -v "%PROJECT_ROOT%\ModelDevelopment\data:/app/data" ^
    -v "%PROJECT_ROOT%\ModelDevelopment\config:/app/config" ^
    medscan-vision-training:latest ^
    python train_resnet.py --config /app/config/vision_training.yml --data_path /app/data/preprocessed --output_path /app/data --dry_run

REM Dry run with ViT
docker run --rm --cpus="0.5" ^
    -v "%PROJECT_ROOT%\DataPipeline\data\preprocessed:/app/data/preprocessed" ^
    -v "%PROJECT_ROOT%\ModelDevelopment\data:/app/data" ^
    -v "%PROJECT_ROOT%\ModelDevelopment\config:/app/config" ^
    medscan-vision-training:latest ^
    python train_vit.py --config /app/config/vision_training.yml --data_path /app/data/preprocessed --output_path /app/data --dry_run

REM Dry run with Custom CNN
docker run --rm --cpus="0.5" ^
    -v "%PROJECT_ROOT%\DataPipeline\data\preprocessed:/app/data/preprocessed" ^
    -v "%PROJECT_ROOT%\ModelDevelopment\data:/app/data" ^
    -v "%PROJECT_ROOT%\ModelDevelopment\config:/app/config" ^
    medscan-vision-training:latest ^
    python train_custom_cnn.py --config /app/config/vision_training.yml --data_path /app/data/preprocessed --output_path /app/data --dry_run
```

**Linux/Mac:**
```bash
# Get project root directory (navigate to project root first)
cd ../..
PROJECT_ROOT=$(pwd)

# Train ResNet50 - Mount the preprocessed data directory, output directory, and config directory
# Override data_path and output_path to use Docker container paths
# Add --cpus="0.5" to limit CPU usage to 50% of one CPU core
docker run --rm --cpus="0.5" \
    -v "${PROJECT_ROOT}/DataPipeline/data/preprocessed:/app/data/preprocessed" \
    -v "${PROJECT_ROOT}/ModelDevelopment/data:/app/data" \
    -v "${PROJECT_ROOT}/ModelDevelopment/config:/app/config" \
    medscan-vision-training:latest \
    python train_resnet.py --config /app/config/vision_training.yml --data_path /app/data/preprocessed --output_path /app/data

# Train ViT
docker run --rm --cpus="0.5" \
    -v "${PROJECT_ROOT}/DataPipeline/data/preprocessed:/app/data/preprocessed" \
    -v "${PROJECT_ROOT}/ModelDevelopment/data:/app/data" \
    -v "${PROJECT_ROOT}/ModelDevelopment/config:/app/config" \
    medscan-vision-training:latest \
    python train_vit.py --config /app/config/vision_training.yml --data_path /app/data/preprocessed --output_path /app/data

# Train Custom CNN
docker run --rm --cpus="0.5" \
    -v "${PROJECT_ROOT}/DataPipeline/data/preprocessed:/app/data/preprocessed" \
    -v "${PROJECT_ROOT}/ModelDevelopment/data:/app/data" \
    -v "${PROJECT_ROOT}/ModelDevelopment/config:/app/config" \
    medscan-vision-training:latest \
    python train_custom_cnn.py --config /app/config/vision_training.yml --data_path /app/data/preprocessed --output_path /app/data

# With custom parameters (always override data_path and output_path for Docker)
docker run --rm --cpus="0.5" \
    -v "${PROJECT_ROOT}/DataPipeline/data/preprocessed:/app/data/preprocessed" \
    -v "${PROJECT_ROOT}/ModelDevelopment/data:/app/data" \
    -v "${PROJECT_ROOT}/ModelDevelopment/config:/app/config" \
    medscan-vision-training:latest \
    python train_resnet.py --config /app/config/vision_training.yml --data_path /app/data/preprocessed --output_path /app/data --epochs 30 --batch_size 16

# Dry run mode (quick test with only 64 images)
docker run --rm --cpus="0.5" \
    -v "${PROJECT_ROOT}/DataPipeline/data/preprocessed:/app/data/preprocessed" \
    -v "${PROJECT_ROOT}/ModelDevelopment/data:/app/data" \
    -v "${PROJECT_ROOT}/ModelDevelopment/config:/app/config" \
    medscan-vision-training:latest \
    python train_resnet.py --config /app/config/vision_training.yml --data_path /app/data/preprocessed --output_path /app/data --dry_run

# Dry run with ViT
docker run --rm --cpus="0.5" \
    -v "${PROJECT_ROOT}/DataPipeline/data/preprocessed:/app/data/preprocessed" \
    -v "${PROJECT_ROOT}/ModelDevelopment/data:/app/data" \
    -v "${PROJECT_ROOT}/ModelDevelopment/config:/app/config" \
    medscan-vision-training:latest \
    python train_vit.py --config /app/config/vision_training.yml --data_path /app/data/preprocessed --output_path /app/data --dry_run

# Dry run with Custom CNN
docker run --rm --cpus="0.5" \
    -v "${PROJECT_ROOT}/DataPipeline/data/preprocessed:/app/data/preprocessed" \
    -v "${PROJECT_ROOT}/ModelDevelopment/data:/app/data" \
    -v "${PROJECT_ROOT}/ModelDevelopment/config:/app/config" \
    medscan-vision-training:latest \
    python train_custom_cnn.py --config /app/config/vision_training.yml --data_path /app/data/preprocessed --output_path /app/data --dry_run
```

**Important Notes for Docker:**
- **Config file**: The config directory must be mounted as a volume. The config file should be at `/app/config/vision_training.yml` inside the container.
- **Data paths**: Always override `--data_path` and `--output_path` in Docker commands to use container paths (`/app/data/preprocessed` and `/app/data`) instead of relative paths from config
- **Windows Docker paths**: Use forward slashes (`/`) in Docker volume paths even on Windows (the `-v` flag handles this automatically)
- **Project root**: Commands assume you start from `ModelDevelopment/Vision` directory and navigate to project root (`cd ../..`) before running Docker commands
- **Explicit paths**: Always specify `--config`, `--data_path`, and `--output_path` explicitly in Docker to avoid path resolution issues
- **Path variables**: The examples use `$PROJECT_ROOT` (PowerShell), `%PROJECT_ROOT%` (CMD), or `$PROJECT_ROOT` (bash) to reference the project root directory

### Command Line Arguments

All arguments are optional and will use config file values if not provided:

- `--config`: Path to config YAML file (default: `ModelDevelopment/config/vision_training.yml`)
- `--data_path`: Path to preprocessed data directory (overrides config)
- `--output_path`: Base path for outputs (overrides config, default: `../data` which is `ModelDevelopment/data`)
- `--epochs`: Number of training epochs (overrides config)
- `--batch_size`: Batch size for training (overrides config)
- `--image_size`: Image size as height width (overrides config)
- `--datasets`: List of datasets to train (overrides config, default: `tb lung_cancer_ct_scan`)
- `--dry_run`: **Dry run mode** - Use only 64 images for quick testing (limits to 2 epochs)

### Dry Run Mode

Dry run mode is useful for:
- Quick validation of the training pipeline
- Testing configuration changes
- Debugging issues without waiting for full training

When `--dry_run` is enabled:
- Only 64 images are used (32 for training, 16 for validation, 16 for test)
- Training is limited to 2 epochs
- All other functionality remains the same (model saving, MLflow tracking, etc.)

Example:
```bash
# Quick test with dry run
python train_resnet.py --dry_run --datasets tb
```

## Output Structure

After training, the following structure is created under `ModelDevelopment/data/`:

```
ModelDevelopment/data/
├── models/
│   ├── tb/
│   │   └── YYYYMMDD_HHMMSS/
│   │       ├── CNN_ResNet50_best.keras
│   │       ├── CNN_ResNet50_final.keras
│   │       ├── CNN_Custom_best.keras
│   │       ├── CNN_Custom_final.keras
│   │       ├── ViT_best.keras
│   │       └── ViT_final.keras
│   └── lung_cancer_ct_scan/
│       └── YYYYMMDD_HHMMSS/
│           └── ...
├── logs/
│   ├── tb/
│   │   └── YYYYMMDD_HHMMSS/
│   │       ├── CNN_ResNet50_training.log
│   │       ├── CNN_Custom_training.log
│   │       └── ViT_training.log
│   └── lung_cancer_ct_scan/
│       └── YYYYMMDD_HHMMSS/
│           └── ...
├── tb/
│   └── YYYYMMDD_HHMMSS/
│       ├── training_metadata.json
│       └── training_summary.json
├── lung_cancer_ct_scan/
│   └── YYYYMMDD_HHMMSS/
│       └── ...
└── mlruns/
    └── (MLflow experiment tracking data)
```

### Metadata Files

**training_metadata.json**: Contains comprehensive training information:
```json
{
  "best_model": "CNN_ResNet50",
  "metrics": {
    "test_loss": 0.2345,
    "test_accuracy": 0.9234,
    "test_top_k_accuracy": 0.9876
  },
  "all_model_results": {
    "CNN_ResNet50": {...},
    "CNN_Custom": {...},
    "ViT": {...}
  },
  "timestamp": "2025-10-24T12:34:56.789012",
  "dataset": "tb",
  "num_classes": 2,
  "epochs": 50,
  "batch_size": 32,
  "image_size": [224, 224],
  "train_samples": 3500,
  "val_samples": 875,
  "test_samples": 1000
}
```

**training_summary.json**: Includes all metadata plus best model path information.

## MLflow Tracking

All training runs are logged to MLflow. The tracking URI is automatically set to `ModelDevelopment/data/mlruns/`.

To view results:

**Windows (PowerShell):**
```powershell
# Start MLflow UI (from project root)
cd ModelDevelopment
$uri = "file:///" + (Resolve-Path .).Path.Replace('\', '/') + "/data/mlruns"
mlflow ui --backend-store-uri $uri

# Or navigate to the data directory
cd data
mlflow ui
```

**Windows (CMD):**
```cmd
REM Start MLflow UI (from project root)
cd ModelDevelopment
cd data
mlflow ui
```

**Linux/Mac:**
```bash
# Start MLflow UI (from project root)
cd ModelDevelopment
mlflow ui --backend-store-uri file:///$(pwd)/data/mlruns

# Or navigate to the data directory
cd data
mlflow ui
```

**Access MLflow UI:**
- Open your browser and navigate to: `http://localhost:5000`

Experiments are organized by dataset:
- `tb_model_training`
- `lung_cancer_ct_scan_model_training`

Each run includes:
- Model parameters (architecture, hyperparameters)
- Training metrics (loss, accuracy per epoch)
- Test metrics (final evaluation)
- Model artifacts (saved models)

## Model Selection

The best model is selected based on **test accuracy** by default (configurable via `model_selection.metric` in config file). The selection logic can be customized to use different metrics (e.g., `test_loss`, `test_top_k_accuracy`).

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
│               ├── test/
│               │   ├── Normal/
│               │   └── Tuberculosis/
│               └── image_metadata.csv (optional)
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
                ├── valid/ (optional)
                │   └── ...
                └── image_metadata.csv (optional)
```

## Configuration File

The configuration file (`ModelDevelopment/config/vision_training.yml`) allows you to customize:

- **Training parameters**: epochs, batch_size, image_size, validation_split
- **Data augmentation**: rotation_range, width_shift_range, height_shift_range, horizontal_flip, zoom_range
- **Model architectures**: 
  - ResNet50: weights, head configuration, learning_rate
  - CNN Custom: conv_blocks, dense_layers, learning_rate
  - ViT: patch_size, projection_dim, num_heads, transformer_layers, mlp_head_units, learning_rate
- **Callbacks**: early_stopping patience, reduce_lr settings, model_checkpoint monitor
- **Model selection**: metric for best model selection
- **Hyperparameter tuning**: Enable/disable tuning, tuner type, search space, max trials

Example: To change the number of epochs, edit `training.epochs` in the config file, or use `--epochs` command-line argument.

## Hyperparameter Tuning

The training script supports automated hyperparameter tuning using KerasTuner. To enable it:

1. **Enable in config**: Set `hyperparameter_tuning.enabled: true` in the config file
2. **Choose tuner type**: Options are:
   - `random`: Random search (fast, good for exploration)
   - `bayesian`: Bayesian optimization (efficient, learns from previous trials)
   - `hyperband`: Hyperband algorithm (adaptive resource allocation)
3. **Configure search space**: Define hyperparameter ranges in `hyperparameter_tuning.search_space`
4. **Set max trials**: Configure `max_trials` to control how many hyperparameter combinations to try

### Example: Enable Hyperparameter Tuning

```yaml
hyperparameter_tuning:
  enabled: true
  tuner_type: "bayesian"  # or "random", "hyperband"
  max_trials: 20
  objective: "val_accuracy"
  direction: "max"
```

### Tunable Hyperparameters

Each model architecture has specific hyperparameters that can be tuned:

**ResNet50:**
- Learning rate
- Dense layer units (256, 512, 1024)
- Dropout rates (0.3-0.6)

**CNN Custom:**
- Learning rate
- Convolutional filter multiplier (1.0, 1.5, 2.0)
- Dense layer units
- Dropout rates

**ViT:**
- Learning rate
- Projection dimension (256, 384, 512)
- Number of attention heads (4, 6, 8)
- Transformer layers (8, 12, 16)
- MLP head units

When tuning is enabled, the script will:
1. Search for optimal hyperparameters for each model
2. Save best hyperparameters to `{model_name}_best_hyperparameters.json`
3. Use the best hyperparameters to train the final models
4. Save tuning results to `ModelDevelopment/data/hyperparameter_tuning/`

**Note**: Hyperparameter tuning significantly increases training time. Start with a small `max_trials` (e.g., 5-10) for testing.

## Notes

- The script automatically finds the latest partition (most recent YYYY/MM/DD)
- If a separate `valid` directory exists, it's used for validation; otherwise, validation_split from config is used
- Images are expected to be preprocessed to the size specified in config (default: 224x224)
- Data augmentation is applied only to training data
- Models are saved in H5 format and also logged to MLflow
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
- Verify config file exists at `ModelDevelopment/config/vision_training.yml`
- Check YAML syntax (indentation, quotes, etc.)
- Use `--config` argument to specify custom config file path

### MLflow Issues
- Ensure MLflow tracking URI is accessible (default: `ModelDevelopment/data/mlruns/`)
- Check disk space for artifact storage
- Verify write permissions in output directory
- MLflow UI can be started from the `ModelDevelopment/data` directory

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (Python 3.9+ recommended)
- Verify TensorFlow installation: `python -c "import tensorflow; print(tensorflow.__version__)"`
