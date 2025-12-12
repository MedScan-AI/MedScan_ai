"""
retraining_trigger.py - Trigger Vision model retraining via GitHub Actions
Can be called by web portal when override threshold is exceeded
"""
import os
import subprocess
import logging
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def trigger_vision_retraining(
    reason: str,
    github_token: Optional[str] = None,
    github_repo: Optional[str] = None
) -> dict:
    """
    Trigger Vision model retraining via GitHub Actions workflow.
    
    Args:
        reason: Reason for retraining (e.g., "Override threshold exceeded: 15 overrides")
        github_token: GitHub personal access token (or set GITHUB_TOKEN env var)
        github_repo: GitHub repository (format: owner/repo, or set GITHUB_REPOSITORY env var)
    
    Returns:
        Dictionary with success status and message
    """
    github_token = github_token or os.getenv('GITHUB_TOKEN')
    github_repo = github_repo or os.getenv('GITHUB_REPOSITORY', 'rjaditya-2702/MedScan_ai')
    
    if not github_token:
        error_msg = "GITHUB_TOKEN not set. Cannot trigger retraining workflow."
        logger.error(error_msg)
        return {
            'success': False,
            'error': error_msg
        }
    
    try:
        logger.info(f"🔄 Triggering Vision model retraining: {reason}")
        
        # Use GitHub CLI to trigger workflow
        cmd = [
            'gh', 'workflow', 'run', 'vision-training.yaml',
            '--ref', 'main',
            '--field', f'triggered_by=web_portal',
            '--field', f'reason={reason}'
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env={**os.environ, 'GITHUB_TOKEN': github_token}
        )
        
        if result.returncode == 0:
            logger.info("✅ Retraining workflow triggered successfully")
            return {
                'success': True,
                'message': 'Retraining workflow triggered successfully',
                'reason': reason,
                'timestamp': datetime.utcnow().isoformat()
            }
        else:
            error_msg = f"Failed to trigger workflow: {result.stderr}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'stderr': result.stderr
            }
            
    except FileNotFoundError:
        error_msg = "GitHub CLI (gh) not found. Install it or use GitHub API directly."
        logger.error(error_msg)
        return {
            'success': False,
            'error': error_msg
        }
    except Exception as e:
        error_msg = f"Error triggering retraining: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            'success': False,
            'error': error_msg
        }


def trigger_via_github_api(
    reason: str,
    github_token: Optional[str] = None,
    github_repo: Optional[str] = None
) -> dict:
    """
    Alternative: Trigger via GitHub REST API (if gh CLI not available).
    
    Args:
        reason: Reason for retraining
        github_token: GitHub personal access token
        github_repo: GitHub repository (format: owner/repo)
    
    Returns:
        Dictionary with success status
    """
    try:
        import requests
    except ImportError:
        return {
            'success': False,
            'error': 'requests library not installed. Install with: pip install requests'
        }
    
    github_token = github_token or os.getenv('GITHUB_TOKEN')
    github_repo = github_repo or os.getenv('GITHUB_REPOSITORY', 'rjaditya-2702/MedScan_ai')
    
    if not github_token:
        return {
            'success': False,
            'error': 'GITHUB_TOKEN not set'
        }
    
    owner, repo = github_repo.split('/')
    
    url = f"https://api.github.com/repos/{owner}/{repo}/actions/workflows/vision-training.yaml/dispatches"
    
    headers = {
        'Authorization': f'token {github_token}',
        'Accept': 'application/vnd.github.v3+json'
    }
    
    data = {
        'ref': 'main',
        'inputs': {
            'triggered_by': 'web_portal',
            'reason': reason
        }
    }
    
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        
        logger.info("✅ Retraining workflow triggered via GitHub API")
        return {
            'success': True,
            'message': 'Retraining workflow triggered successfully',
            'reason': reason,
            'timestamp': datetime.utcnow().isoformat()
        }
    except Exception as e:
        error_msg = f"Failed to trigger via GitHub API: {str(e)}"
        logger.error(error_msg)
        return {
            'success': False,
            'error': error_msg
        }

