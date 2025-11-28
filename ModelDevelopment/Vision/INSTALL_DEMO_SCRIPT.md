# Installation Instructions for Demo Script (test2.py)

## Required Packages

The `test2.py` script requires the following packages:

### Core Dependencies (usually already installed):
- `numpy`
- `matplotlib`
- `Pillow` (PIL)
- `pandas`

### Additional Dependencies (may need installation):

1. **scikit-image** (required for LIME boundary visualization):
   ```bash
   pip install scikit-image
   ```

2. **opencv-python** (optional, for better image smoothing):
   ```bash
   pip install opencv-python
   ```

## Quick Installation

### Option 1: Install from requirements.txt (recommended)
```bash
cd ModelDevelopment/Vision
pip install -r requirements.txt
```

### Option 2: Install only demo script dependencies
```bash
pip install scikit-image opencv-python
```

### Option 3: Install in Docker (if using Docker)
The Docker image should already include these packages. If not, rebuild the image:
```bash
docker build -t medscan-vision-training:latest -f ModelDevelopment/Vision/Dockerfile ModelDevelopment/Vision/
```

## Verify Installation

Test if packages are installed:
```bash
python -c "import skimage; import cv2; print('All packages installed successfully!')"
```

## Running the Script

Once installed, run the script from the project root:

### Correct Data Path:
The preprocessed data is located in `DataPipeline/data/preprocessed/`, not `ModelDevelopment/data/preprocessed/`.

```bash
# From project root (MedScan_ai/)
python ModelDevelopment/Vision/test2.py \
    --data_path DataPipeline/data/preprocessed \
    --output_path ModelDevelopment/data \
    --datasets tb lung_cancer_ct_scan \
    --num_shap 5 \
    --num_lime 3
```

### Alternative: If running from ModelDevelopment/Vision/ directory:
```bash
cd ModelDevelopment/Vision
python test2.py \
    --data_path ../../DataPipeline/data/preprocessed \
    --output_path ../data \
    --datasets tb lung_cancer_ct_scan \
    --num_shap 5 \
    --num_lime 3
```

**Note**: The script will automatically try to find data in common alternative locations if the specified path doesn't exist.

## Note

The script will work even if `opencv-python` is not installed (it will use simpler smoothing). However, `scikit-image` is recommended for better LIME visualizations.

