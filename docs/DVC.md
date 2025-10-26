## Data Version Control (DVC)

We use DVC to version control our data pipeline outputs (1.2 GB, 17k files) without storing them in Git.

### Quick Start
```bash
# Clone the repository
git clone <your-repo-url>
cd MedScan_ai

# Pull the data
dvc pull

# All data is now available in DataPipeline/data/
```

### What's Tracked by DVC

**Total**: 1.2 GB | 17,305 files

- `data/raw/` - Kaggle datasets (TB X-rays, Lung Cancer CT scans)
- `data/preprocessed/` - Processed 224x224 images
- `data/synthetic_metadata/` - Generated patient CSVs
- `data/synthetic_metadata_mitigated/` - Bias-corrected datasets
- `data/ge_outputs/` - Validation reports, bias analysis, EDA
- `data/mlflow_store/` - MLflow experiment tracking

### DVC Workflow

**Getting latest data:**
```bash
git pull
dvc pull
```

**After regenerating data:**
```bash
dvc add DataPipeline/data/
git add DataPipeline/data.dvc
git commit -m "Update pipeline data"
dvc push
git push
```

### Storage

- **Development**: Local at `~/dvc-storage/medscan-ai`
- **Production**: Can be configured for GCS/S3

### Why DVC?

✅ Git stays small (only 110-byte metadata file)  
✅ Data versioning and reproducibility  
✅ Team collaboration without Git bloat  
✅ Proper MLOps practice
