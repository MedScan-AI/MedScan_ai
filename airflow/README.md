# MedScan AI - Airflow Pipeline Setup

Complete instructions to run the data pipeline orchestration with Airflow.

## Prerequisites

- Docker Desktop installed and running
- Git (to clone the repo)
- DVC installed: `pip install dvc`
- At least 4GB RAM allocated to Docker
- ~2GB free disk space for data

## Setup Instructions (First Time Only)

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd MedScan_ai
```

### 2. Checkout the Airflow Branch
```bash
git checkout vision_pipeline
```

### 3. Pull Data with DVC
```bash
# Pull the data (1.2 GB)
dvc pull

# Verify data exists
ls -la DataPipeline/data/
```

### 4. Configure Email Alerts (Optional)

If you want to receive email alerts:
```bash
cd airflow
nano .env
```

Update these values:
```bash
SMTP_USER=your-gmail@gmail.com
SMTP_PASSWORD=your-app-password
ALERT_EMAIL_RECIPIENTS=your-email@gmail.com,teammate@gmail.com
```

**To get Gmail App Password:**
1. Go to Google Account → Security
2. Enable 2-Factor Authentication
3. Generate App Password for "Mail"
4. Use that 16-character password in `.env`

### 5. Start Airflow
```bash
cd airflow

# Start all services (first time takes 2-3 minutes)
docker compose up -d

# Wait for services to initialize
sleep 60

# Check if services are running
docker compose ps
```

You should see:
```
NAME                STATUS
airflow-init        exited (0)
airflow-postgres    running
airflow-scheduler   running
airflow-webserver   running
```

### 6. Access Airflow UI

Open browser: http://localhost:8080

**Login:**
- Username: `admin`
- Password: `admin`

## Running the Pipeline

### Step 1: Enable the DAG

1. In Airflow UI, find `medscan_vision_pipeline`
2. Toggle the switch on the left to **ON** (blue)

### Step 2: Trigger the Pipeline

1. Click the **Play button** ▶️ on the right
2. Click **"Trigger DAG"**
3. Click on the DAG name to watch progress

### Step 3: Monitor Execution

Switch to **Graph View** to see visual progress:

- ⚪ Gray = Queued
- 🟡 Yellow = Running
- 🟢 Green = Success
- 🔴 Red = Failed

**Expected Runtime:** ~8-10 minutes (with cached data)

### Step 4: Check Results

After completion:
- Check `DataPipeline/data/ge_outputs/reports/` for HTML reports
- Check email for alerts (if configured)
- View logs in Airflow UI by clicking task boxes

## Pipeline Stages
```
start
  ↓
test_data_acquisition (pytest)
  ↓
download_kaggle_data (fetch TB & Lung Cancer datasets)
  ↓
test_preprocessing
  ↓
[process_tb || process_lung_cancer] (parallel processing)
  ↓
test_synthetic_data
  ↓
generate_patient_metadata (Faker data)
  ↓
test_validation
  ↓
validate_data (Great Expectations + Bias Detection)
  ↓
check_validation_results → check_drift_results
  ↓
check_if_alert_needed (conditional gate)
  ↓
generate_alert_email → send_alert_email (if issues found)
  ↓
complete
```

## Troubleshooting

### Pipeline Fails on First Run

**Problem:** "No module named 'great_expectations'" or similar

**Solution:** Rebuild Docker image
```bash
cd airflow
docker compose down
docker compose build --no-cache
docker compose up -d
```

### MLflow Errors in Logs

**Problem:** "OSError: Resource deadlock avoided"

**Solution:** Already fixed! Using `/tmp` storage in Docker.
If still occurs, check `DataPipeline/config/metadata.yml`:
```yaml
mlmd:
  store:
    database_path: "/tmp/mlflow/metadata.db"  # Should be /tmp, not data/
```

### Email Alerts Not Working

**Problem:** No emails received

**Check:**
1. `.env` file has correct SMTP credentials
2. Gmail App Password (not regular password)
3. Check Airflow logs: `docker compose logs webserver | grep -i smtp`

### Container Out of Memory

**Problem:** Docker containers crash

**Solution:** Increase Docker memory
1. Docker Desktop → Settings → Resources
2. Set Memory to at least 4GB
3. Restart Docker Desktop

### Data Not Found

**Problem:** "FileNotFoundError: data/raw/..."

**Solution:** Pull data with DVC
```bash
cd ~/Documents/MedScan_ai
dvc pull
ls DataPipeline/data/raw/  # Should show folders
```

## Viewing Logs
```bash
cd airflow

# View all logs
docker compose logs

# View specific service
docker compose logs scheduler
docker compose logs webserver

# Follow logs in real-time
docker compose logs -f scheduler

# View logs for specific task in Airflow UI:
# Click task box → Click "Log" button
```

## Stopping Airflow
```bash
cd airflow

# Stop services (keeps data)
docker compose down

# Stop and remove all data
docker compose down -v
```

## Restarting After Changes

If you modify the DAG or code:
```bash
cd airflow

# Restart scheduler to pick up changes
docker compose restart scheduler

# Or restart everything
docker compose restart
```

## Project Structure
```
airflow/
├── dags/
│   └── medscan_vision_pipeline.py    # Main DAG definition
├── docker-compose.yml                 # Docker services config
├── Dockerfile                         # Custom Airflow image
├── .env                              # Configuration (SMTP, emails)
└── README.md                         # This file
```

## Data Pipeline Outputs

After successful run:
```
DataPipeline/data/
├── raw/                              # Downloaded Kaggle data
├── preprocessed/                     # Processed 224x224 images  
├── synthetic_metadata/               # Generated patient CSVs
├── synthetic_metadata_mitigated/     # Bias-corrected data
├── ge_outputs/                       # Validation reports
│   ├── reports/                      # HTML visualizations
│   ├── bias_analysis/                # Bias detection results
│   ├── eda/                          # Exploratory data analysis
│   └── drift/                        # Drift detection results
└── mlflow_store/                     # Experiment tracking (in /tmp in Docker)
```

## Testing Individual Scripts (Without Airflow)

To test scripts locally:
```bash
cd DataPipeline

# Test data acquisition
python scripts/data_acquisition/fetch_data.py --config config/vision_pipeline.yml

# Test preprocessing
python scripts/data_preprocessing/process_tb.py --config config/vision_pipeline.yml

# Test validation
python scripts/data_preprocessing/schema_statistics.py --config config/metadata.yml

# Run all tests
pytest tests/ -v
```

## macOS Specific Notes

If you're on macOS and encounter file locking issues:
- ✅ Already handled! Using `/tmp` storage fixes Docker Desktop osxfs issues
- Only affects MLflow, rest of pipeline unaffected

## Team Collaboration

**Before making changes:**
```bash
git pull origin vision_pipeline
dvc pull
```

**After making changes:**
```bash
# If you regenerated data
dvc add DataPipeline/data/
dvc push

# Commit code
git add .
git commit -m "Your changes"
git push origin vision_pipeline
```

## Help & Support

- **Airflow Docs**: https://airflow.apache.org/docs/
- **DVC Docs**: https://dvc.org/doc
- **Issues**: Create GitHub issue or contact team

## Quick Reference

| Command | Purpose |
|---------|---------|
| `docker compose up -d` | Start Airflow |
| `docker compose down` | Stop Airflow |
| `docker compose logs -f` | View logs |
| `docker compose ps` | Check status |
| `docker compose restart` | Restart services |
| `dvc pull` | Get latest data |
| `dvc push` | Upload data changes |

---

**Ready to run!** If you encounter issues, check the Troubleshooting section above.
