"""
DVC Initialization Script for Docker Container
Initializes Git and DVC with GCS remote storage
"""
import os
import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def run_command(cmd, cwd=None, check=False):
    """Run shell command and return result."""
    try:
        result = subprocess.run(
            cmd if isinstance(cmd, list) else cmd.split(),
            cwd=cwd,
            capture_output=True,
            text=True,
            check=check
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.CalledProcessError as e:
        return e.stdout, e.stderr, e.returncode


def init_git():
    """Initialize Git repository if not already initialized."""
    data_pipeline_dir = Path("/opt/airflow/DataPipeline")
    
    if (data_pipeline_dir / ".git").exists():
        logger.info("Git already initialized")
        return True
    
    logger.info("Initializing Git repository")
    
    # Init git
    stdout, stderr, code = run_command("git init", cwd=str(data_pipeline_dir))
    if code != 0:
        logger.error(f"Git init failed: {stderr}")
        return False
    
    # Configure git
    run_command("git config user.name 'MedScan AI'", cwd=str(data_pipeline_dir))
    run_command("git config user.email 'admin@medscan.ai'", cwd=str(data_pipeline_dir))
    
    # Create initial commit
    run_command("git add .", cwd=str(data_pipeline_dir))
    run_command("git commit -m 'Initial commit' --allow-empty", cwd=str(data_pipeline_dir))
    
    logger.info("Git initialized")
    return True


def init_dvc():
    """Initialize DVC in the DataPipeline directory."""
    data_pipeline_dir = Path("/opt/airflow/DataPipeline")
    
    if (data_pipeline_dir / ".dvc").exists():
        logger.info("DVC already initialized")
        return True
    
    logger.info("Initializing DVC")
    stdout, stderr, code = run_command("dvc init", cwd=str(data_pipeline_dir))
    
    if code != 0:
        logger.error(f"DVC init failed: {stderr}")
        return False
    
    # Commit DVC initialization
    run_command("git add .dvc .dvcignore", cwd=str(data_pipeline_dir))
    run_command("git commit -m 'Initialize DVC'", cwd=str(data_pipeline_dir))
    
    logger.info("DVC initialized")
    return True


def configure_gcs_remotes():
    """Configure GCS remotes for both pipelines."""
    data_pipeline_dir = Path("/opt/airflow/DataPipeline")
    bucket_name = os.getenv("GCS_BUCKET_NAME", "medscan-data")
    project_id = os.getenv("GCP_PROJECT_ID", "medscanai-476203")
    creds_path = "/opt/airflow/gcp-service-account.json"
    
    remotes = {
        'vision': f"gs://{bucket_name}/dvc-storage/vision",
        'rag': f"gs://{bucket_name}/dvc-storage/rag"
    }
    
    for remote_name, remote_url in remotes.items():
        logger.info(f"Configuring {remote_name} remote")
        
        # Add remote
        stdout, stderr, code = run_command(
            ['dvc', 'remote', 'add', remote_name, remote_url],
            cwd=str(data_pipeline_dir)
        )
        
        if code != 0 and "already exists" not in stderr.lower():
            logger.error(f"Failed to add {remote_name} remote: {stderr}")
            continue
        
        # Set as default for vision
        if remote_name == 'vision':
            run_command(
                ['dvc', 'remote', 'default', 'vision'],
                cwd=str(data_pipeline_dir)
            )
        
        # Configure project
        run_command(
            ['dvc', 'remote', 'modify', remote_name, 'projectname', project_id],
            cwd=str(data_pipeline_dir)
        )
        
        # Set credentials
        if os.path.exists(creds_path):
            run_command(
                ['dvc', 'remote', 'modify', remote_name, 'credentialpath', creds_path],
                cwd=str(data_pipeline_dir)
            )
    
    logger.info("GCS remotes configured")
    return True


def configure_dvc_settings():
    """Configure DVC settings."""
    data_pipeline_dir = Path("/opt/airflow/DataPipeline")
    
    # Disable analytics
    run_command(
        ['dvc', 'config', 'core.analytics', 'false'],
        cwd=str(data_pipeline_dir)
    )
    
    # Set autostage
    run_command(
        ['dvc', 'config', 'core.autostage', 'true'],
        cwd=str(data_pipeline_dir)
    )
    
    return True


def create_dvcignore():
    """Create .dvcignore file."""
    data_pipeline_dir = Path("/opt/airflow/DataPipeline")
    dvcignore = data_pipeline_dir / ".dvcignore"
    
    patterns = [
        "# Temporary files",
        "temp/",
        "logs/",
        "*.log",
        "",
        "# MLflow",
        "data/mlflow_store/",
        "",
        "# Great Expectations outputs",
        "data/ge_outputs/",
        "",
        "# Partition metadata",
        "data/partition_metadata.json",
    ]
    
    with open(dvcignore, 'w') as f:
        f.write('\n'.join(patterns) + '\n')
    
    logger.info(".dvcignore created")
    return True


def main():
    """Main initialization function."""
    logger.info("DVC Initialization for MedScan AI")
    
    try:
        # Step 1: Initialize Git
        if not init_git():
            logger.error("Failed to initialize Git")
            sys.exit(1)
        
        # Step 2: Initialize DVC
        if not init_dvc():
            logger.error("Failed to initialize DVC")
            sys.exit(1)
        
        # Step 3: Configure GCS remotes
        if not configure_gcs_remotes():
            logger.error("Failed to configure GCS remotes")
            sys.exit(1)
        
        # Step 4: Configure DVC settings
        configure_dvc_settings()
        
        # Step 5: Create .dvcignore
        create_dvcignore()
        
        logger.info("DVC initialization completed")
        return True
        
    except Exception as e:
        logger.error(f"Initialization failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()