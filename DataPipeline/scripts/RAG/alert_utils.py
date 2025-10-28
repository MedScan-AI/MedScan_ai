"""Simple alert utilities for RAG pipeline using Airflow's email."""
import logging
from typing import Dict, Any, Optional
from airflow.utils.email import send_email

logger = logging.getLogger(__name__)


def send_failure_alert(
    task_name: str,
    error_message: str,
    context: Dict[str, Any],
    recipients: list
):
    """Send email alert when task fails.
    
    Args:
        task_name: Name of the failed task
        error_message: Error message
        context: Airflow context dictionary
        recipients: List of email addresses
    """
    if not recipients:
        logger.warning("No email recipients configured for alerts")
        return
    
    execution_date = context.get('execution_date', 'Unknown')
    dag_id = context.get('dag', {}).dag_id if 'dag' in context else 'Unknown'
    run_id = context.get('run_id', 'Unknown')
    
    subject = f"RAG Pipeline Task Failed: {task_name}"
    
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            .header {{ background-color: #e74c3c; color: white; padding: 20px; }}
            .content {{ padding: 20px; }}
            .details {{ background-color: #f5f5f5; padding: 15px; margin: 15px 0; }}
            .error {{ background-color: #ffebee; padding: 15px; margin: 15px 0; border-left: 4px solid #e74c3c; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2>RAG Pipeline Task Failed</h2>
        </div>
        <div class="content">
            <h3>Task: {task_name}</h3>
            
            <div class="details">
                <p><strong>DAG:</strong> {dag_id}</p>
                <p><strong>Run ID:</strong> {run_id}</p>
                <p><strong>Execution Date:</strong> {execution_date}</p>
            </div>
            
            <div class="error">
                <h4>Error:</h4>
                <pre>{error_message}</pre>
            </div>
            
            <p>Please check the Airflow logs for more details.</p>
            <p><em>This is an automated alert from the MedScan RAG Pipeline.</em></p>
        </div>
    </body>
    </html>
    """
    
    try:
        send_email(
            to=recipients,
            subject=subject,
            html_content=html_content
        )
        logger.info(f"Failure alert sent to {len(recipients)} recipients")
    except Exception as e:
        logger.error(f"Failed to send email alert: {e}")


def send_threshold_alert(
    task_name: str,
    threshold_name: str,
    actual_value: float,
    threshold_value: float,
    context: Dict[str, Any],
    recipients: list,
    additional_info: Optional[Dict] = None
):
    """Send email alert when threshold is exceeded.
    
    Args:
        task_name: Name of the task
        threshold_name: Name of the threshold metric
        actual_value: Actual measured value
        threshold_value: Threshold value that was exceeded
        context: Airflow context dictionary
        recipients: List of email addresses
        additional_info: Optional additional context
    """
    if not recipients:
        logger.warning("No email recipients configured for alerts")
        return
    
    execution_date = context.get('execution_date', 'Unknown')
    dag_id = context.get('dag', {}).dag_id if 'dag' in context else 'Unknown'
    run_id = context.get('run_id', 'Unknown')
    
    subject = f"⚠️ RAG Pipeline Threshold Exceeded: {threshold_name}"
    
    # Build additional info section
    additional_info_html = ""
    if additional_info:
        additional_info_html = "<h4>Additional Information:</h4><ul>"
        for key, value in additional_info.items():
            additional_info_html += f"<li><strong>{key}:</strong> {value}</li>"
        additional_info_html += "</ul>"
    
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            .header {{ background-color: #f39c12; color: white; padding: 20px; }}
            .content {{ padding: 20px; }}
            .details {{ background-color: #f5f5f5; padding: 15px; margin: 15px 0; }}
            .warning {{ background-color: #fff3cd; padding: 15px; margin: 15px 0; border-left: 4px solid #f39c12; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2>RAG Pipeline Threshold Exceeded</h2>
        </div>
        <div class="content">
            <h3>Task: {task_name}</h3>
            
            <div class="details">
                <p><strong>DAG:</strong> {dag_id}</p>
                <p><strong>Run ID:</strong> {run_id}</p>
                <p><strong>Execution Date:</strong> {execution_date}</p>
            </div>
            
            <div class="warning">
                <h4>Threshold Exceeded:</h4>
                <p><strong>Metric:</strong> {threshold_name}</p>
                <p><strong>Actual Value:</strong> {actual_value:.2f}</p>
                <p><strong>Threshold:</strong> {threshold_value:.2f}</p>
                {additional_info_html}
            </div>
            
            <p>The pipeline continued running, but this metric exceeded the configured threshold.</p>
            <p><em>This is an automated alert from the MedScan RAG Pipeline.</em></p>
        </div>
    </body>
    </html>
    """
    
    try:
        send_email(
            to=recipients,
            subject=subject,
            html_content=html_content
        )
        logger.info(f"Threshold alert sent to {len(recipients)} recipients")
    except Exception as e:
        logger.error(f"Failed to send email alert: {e}")