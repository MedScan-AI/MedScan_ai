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

### 2. Checkout the Correct Branch
```bash
# For latest development version
git checkout vision_pipeline

# OR for stable version
git checkout main
```

### 3. Pull Data with DVC
```bash
# Install DVC if not already installed
pip install dvc

# Pull the data (1.2 GB - takes 2-5 minutes)
dvc pull

# Verify data exists
ls -la DataPipeline/data/
# Should show: raw/, preprocessed/, synthetic_metadata/, etc.
```

### 4. Create Environment Configuration File

**CRITICAL:** Airflow requires a `.env` file with security keys. This file is not in Git for security reasons.
```bash
cd airflow

# Generate security keys
FERNET_KEY=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")

# Create .env file
cat > .env << EOF
# Airflow UID
AIRFLOW_UID=50000

# Database Connection
AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres/airflow

# Security Keys (REQUIRED - DO NOT USE THESE DUMMY VALUES)
AIRFLOW__CORE__FERNET_KEY=${FERNET_KEY}
AIRFLOW__WEBSERVER__SECRET_KEY=${SECRET_KEY}

# Admin User
AIRFLOW_USERNAME=admin
AIRFLOW_PASSWORD=admin
AIRFLOW_FIRSTNAME=Admin
AIRFLOW_LASTNAME=User
AIRFLOW_EMAIL=admin@example.com

# SMTP Configuration (Leave empty to disable email alerts)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=
SMTP_PASSWORD=
SMTP_MAIL_FROM=

# MLflow
MLFLOW_TRACKING_URI=file:///tmp/mlflow
EOF

echo "✅ .env file created successfully!"
```

**What each variable does:**
- `FERNET_KEY`: Encrypts passwords in Airflow database (REQUIRED)
- `SECRET_KEY`: Secures web sessions (REQUIRED)
- `SQL_ALCHEMY_CONN`: Database connection string (REQUIRED)
- `SMTP_*`: Email configuration (OPTIONAL - only needed for alerts)

### 5. Configure Email Alerts (Optional)

If you want to receive email alerts for anomalies/drift:
```bash
nano .env
```

Update these lines with your Gmail credentials:
```bash
SMTP_USER=your-gmail@gmail.com
SMTP_PASSWORD=your-16-char-app-password
SMTP_MAIL_FROM=your-gmail@gmail.com
```

**How to get Gmail App Password:**
1. Go to https://myaccount.google.com/security
2. Enable **2-Step Verification** (if not already enabled)
3. Search for **"App passwords"**
4. Select **"Mail"** and your device
5. Copy the **16-character password** (format: `xxxx xxxx xxxx xxxx`)
6. Paste it as `SMTP_PASSWORD` in `.env` (**remove spaces**)

**Important:** Use App Password, NOT your regular Gmail password!

### 6. Start Airflow
```bash
cd airflow

# Start all services (first time takes 2-3 minutes)
docker compose up -d

# Wait for services to initialize
sleep 60

# Check if services are running
docker compose ps
```

**Expected output:**
```
NAME                  STATUS
airflow-postgres-1    Up (healthy)
airflow-init-1        Exited (0)
airflow-scheduler-1   Up
airflow-webserver-1   Up (healthy)
```

**Note:** `airflow-init-1` should show `Exited (0)` - this is normal. It runs once to initialize the database.

### 7. Access Airflow UI

Open browser: http://localhost:8080

**Login credentials:**
- Username: `admin`
- Password: `admin`

**First login takes 10-20 seconds to load.**

---

## Troubleshooting

### Problem: "Could not parse SQLAlchemy URL from string ''"

**Cause:** `.env` file is missing or not being read.

**Solution:**
```bash
cd airflow
# Check if .env exists
ls -la .env

# If missing, recreate using Step 4 above
# If exists, verify it has content:
cat .env | grep FERNET_KEY
# Should show a long random string, not empty
```

### Problem: Webserver shows "unhealthy"

**Cause:** Missing security keys in `.env`.

**Solution:**
```bash
cd airflow
# Regenerate keys and recreate .env (see Step 4)
docker compose down -v
docker compose up -d
```

### Problem: "Permission denied" or "Cannot connect to Docker daemon"

**Cause:** Docker Desktop not running.

**Solution:**
1. Open Docker Desktop application
2. Wait for it to fully start (icon in menu bar should be steady)
3. Try again: `docker compose up -d`

### Problem: Port 8080 already in use

**Cause:** Another service using port 8080.

**Solution:**
```bash
# Find what's using port 8080
lsof -i :8080

# Kill the process or change Airflow port in docker-compose.yml:
# Change "8080:8080" to "8081:8080"
# Then access at http://localhost:8081
```

### Problem: Email alerts not working

**Check:**
1. `.env` has valid Gmail credentials
2. Using App Password (not regular password)
3. Check Airflow logs: `docker compose logs webserver | grep -i smtp`

---

## Important Notes

### Security
- **Never commit `.env` to Git** - it contains sensitive credentials
- The `.env` file is in `.gitignore` for security
- Each team member must create their own `.env` file

### Data Storage
- Data pulled via DVC is stored in `DataPipeline/data/`
- MLflow data stored in `/tmp/mlflow` inside Docker (lost on container restart)
- Postgres data persists in Docker volume `postgres-db-volume`

### Clean Restart
If you need to start completely fresh:
```bash
cd airflow
docker compose down -v  # -v removes volumes (deletes database)
# Then follow setup steps again from Step 4
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
