"""DVC Helper Functions for Airflow DAGs - Best Practices"""
import subprocess
import logging
from pathlib import Path
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)


class DVCManager:
    """Manage DVC operations following best practices."""
    
    def __init__(self, work_dir: str = "/opt/airflow/DataPipeline"):
        self.work_dir = Path(work_dir)
        self._verify_initialized()
    
    def _verify_initialized(self):
        """Verify DVC is initialized."""
        if not (self.work_dir / ".dvc").exists():
            logger.warning("DVC not initialized")
    
    def _run_cmd(self, cmd: List[str], timeout: int = 600) -> tuple:
        """Run DVC command with automatic lock recovery."""
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.work_dir),
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            # If lock error, try to recover once
            if result.returncode != 0 and "unable to acquire lock" in result.stderr.lower():
                logger.warning("DVC lock detected, attempting recovery")
                
                # Remove stale lock
                lock_dir = self.work_dir / ".dvc" / "tmp"
                if lock_dir.exists():
                    import shutil
                    shutil.rmtree(lock_dir, ignore_errors=True)
                    lock_dir.mkdir(parents=True, exist_ok=True)
                    logger.info("Cleared stale locks")
                
                # Retry once
                import time
                time.sleep(2)
                result = subprocess.run(
                    cmd,
                    cwd=str(self.work_dir),
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
            
            return result.returncode, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            logger.error(f"Command timeout: {' '.join(cmd)}")
            return 1, "", "Timeout"
        except Exception as e:
            logger.error(f"Command failed: {e}")
            return 1, "", str(e)
    
    def add_batch(self, paths: List[str]) -> Dict[str, bool]:
        """
        Add multiple paths to DVC in batch.
        Best practice: Use single dvc add command with multiple targets.
        
        Args:
            paths: List of paths to track
            
        Returns:
            Dict mapping paths to success status
        """
        results = {}
        
        if not paths:
            return results
        
        logger.info(f"Adding {len(paths)} paths to DVC")
        
        # Add all paths in one command
        cmd = ["dvc", "add"] + paths
        code, stdout, stderr = self._run_cmd(cmd)
        
        if code == 0:
            for path in paths:
                results[path] = True
            logger.info(f"Successfully added {len(paths)} paths")
        else:
            # If batch fails, try individually
            logger.warning(f"Batch add failed, trying individually: {stderr}")
            for path in paths:
                results[path] = self.add(path)
        
        return results
    
    def add(self, path: str) -> bool:
        """
        Add single path to DVC tracking.
        
        Args:
            path: Path to track
            
        Returns:
            True if successful
        """
        logger.info(f"Adding to DVC: {path}")
        code, stdout, stderr = self._run_cmd(["dvc", "add", path])
        
        if code != 0:
            if "is already tracked" in stderr or "didn't change" in stderr:
                logger.info(f"Already tracked: {path}")
                return True
            if "is git-ignored" in stderr:
                logger.error(f"Path is git-ignored: {path}")
                logger.error("Fix .gitignore - it should NOT ignore *.dvc files")
                return False
            
            logger.error(f"DVC add failed for {path}: {stderr}")
            return False
        
        logger.info(f"Added to DVC: {path}")
        return True
    
    def push(self, remote: str = "vision", targets: Optional[List[str]] = None, 
             jobs: int = 4) -> bool:
        """
        Push data to DVC remote.
        
        Args:
            remote: Remote name
            targets: Optional specific targets
            jobs: Number of parallel jobs
            
        Returns:
            True if successful
        """
        cmd = ["dvc", "push", "-r", remote, "--jobs", str(jobs)]
        if targets:
            cmd.extend(targets)
        
        logger.info(f"Pushing to DVC remote: {remote}")
        code, stdout, stderr = self._run_cmd(cmd, timeout=900)
        
        if code != 0:
            if "already exists" in stderr or "up to date" in stderr or "everything is up to date" in stdout:
                logger.info(f"Remote up to date: {remote}")
                return True
            
            logger.error(f"DVC push failed: {stderr}")
            return False
        
        logger.info(f"Pushed to remote: {remote}")
        return True
    
    def pull(self, remote: str = "vision", targets: Optional[List[str]] = None,
             jobs: int = 4) -> bool:
        """
        Pull data from DVC remote.
        
        Args:
            remote: Remote name
            targets: Optional specific targets  
            jobs: Number of parallel jobs
            
        Returns:
            True if successful
        """
        cmd = ["dvc", "pull", "-r", remote, "--jobs", str(jobs)]
        if targets:
            cmd.extend(targets)
        
        logger.info(f"Pulling from DVC remote: {remote}")
        code, stdout, stderr = self._run_cmd(cmd, timeout=900)
        
        if code != 0:
            if "does not exist" in stderr or "no outputs to download" in stderr:
                logger.info(f"No data in remote yet: {remote}")
                return True
            
            logger.warning(f"DVC pull warning: {stderr}")
            return False
        
        logger.info(f"Pulled from remote: {remote}")
        return True
    
    def status(self, remote: Optional[str] = None) -> Dict:
        """Get DVC status."""
        cmd = ["dvc", "status"]
        if remote:
            cmd.extend(["-r", remote])
        
        code, stdout, stderr = self._run_cmd(cmd)
        
        return {
            "success": code == 0,
            "output": stdout if code == 0 else stderr
        }
    
    def commit_dvc_files(self, message: str) -> bool:
        """
        Commit .dvc files to git.
        
        Args:
            message: Commit message
            
        Returns:
            True if successful
        """
        try:
            # Stage .dvc files
            result = subprocess.run(
                ["git", "add", "*.dvc", ".dvcignore", ".gitignore"],
                cwd=str(self.work_dir),
                capture_output=True,
                text=True
            )
            
            # Commit
            result = subprocess.run(
                ["git", "commit", "-m", message, "--allow-empty"],
                cwd=str(self.work_dir),
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"Committed: {message}")
                return True
            elif "nothing to commit" in result.stdout or "nothing added to commit" in result.stdout:
                logger.info("No changes to commit")
                return True
            else:
                logger.warning(f"Commit warning: {result.stderr}")
                return True
                
        except Exception as e:
            logger.error(f"Git commit failed: {e}")
            return False