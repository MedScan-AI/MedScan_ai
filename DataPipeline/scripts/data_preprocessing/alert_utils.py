"""
Alert utilities for Vision Pipeline
Enhanced with bias detection support and improved email formatting
"""
import logging
import smtplib
import traceback
import json
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "config"))
import vision_gcp_config

logger = logging.getLogger(__name__)


def send_email_alert(subject: str, body: str, html_body: str = None, recipients: list = None):
    """
    Send email alert using SMTP configuration.
    
    Args:
        subject: Email subject
        body: Plain text body
        html_body: HTML formatted body (optional, preferred)
        recipients: List of email addresses (uses config default if None)
    """
    alert_config = vision_gcp_config.get_alert_config()
    
    if not alert_config.get('enabled', True):
        logger.info("Email alerts disabled, skipping")
        return
    
    recipients = recipients or alert_config.get('email_recipients', [])
    smtp_settings = alert_config.get('smtp_settings', {})
    
    if not recipients:
        logger.warning("No email recipients configured")
        return
    
    try:
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"[MedScan Vision] {subject}"
        msg['From'] = smtp_settings.get('username', 'noreply@medscan.ai')
        msg['To'] = ', '.join(recipients)
        msg['Date'] = datetime.utcnow().strftime('%a, %d %b %Y %H:%M:%S +0000')
        
        # Attach plain text version
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach HTML version if provided
        if html_body:
            msg.attach(MIMEText(html_body, 'html'))
        
        # Send email
        with smtplib.SMTP(smtp_settings.get('host', 'smtp.gmail.com'), 
                         smtp_settings.get('port', 587)) as server:
            if smtp_settings.get('use_tls', True):
                server.starttls()
            
            username = smtp_settings.get('username')
            password = smtp_settings.get('password')
            
            if username and password:
                server.login(username, password)
            
            server.send_message(msg)
        
        logger.info(f"✓ Email alert sent: {subject} → {recipients}")
        
    except Exception as e:
        logger.error(f"Failed to send email alert: {e}", exc_info=True)


def check_validation_results(validation_dir: str = '/opt/airflow/DataPipeline/data/ge_outputs/validations') -> dict:
    """
    Check validation results and return anomaly details.
    
    Returns:
        {
            'alert_needed': bool,
            'total_anomalies': int,
            'anomalies': [{'dataset': str, 'num_anomalies': int, 'details': []}]
        }
    """
    logger.info("="*80)
    logger.info("Checking validation results for anomalies...")
    logger.info("="*80)
    
    validation_files = []
    if os.path.exists(validation_dir):
        for root, dirs, files in os.walk(validation_dir):
            for file in files:
                if file.endswith('_validation.json'):
                    validation_files.append(os.path.join(root, file))
    
    if not validation_files:
        logger.info("No validation files found. Skipping anomaly check.")
        return {'alert_needed': False, 'total_anomalies': 0, 'anomalies': []}
    
    anomalies_found = []
    total_anomalies = 0
    
    for val_file in validation_files:
        try:
            with open(val_file, 'r') as f:
                results = json.load(f)
            
            dataset_name = results.get('dataset_name', 'Unknown')
            is_valid = results.get('is_valid', True)
            num_anomalies = results.get('num_anomalies', 0)
            anomaly_list = results.get('anomalies', [])
            
            logger.info(f"\nDataset: {dataset_name}")
            logger.info(f"  Valid: {is_valid}")
            logger.info(f"  Anomalies: {num_anomalies}")
            
            if not is_valid and num_anomalies > 0:
                anomalies_found.append({
                    'dataset': dataset_name,
                    'num_anomalies': num_anomalies,
                    'details': anomaly_list[:5]
                })
                total_anomalies += num_anomalies
        
        except Exception as e:
            logger.error(f"Error reading validation file {val_file}: {e}")
    
    if anomalies_found:
        logger.warning("="*80)
        logger.warning(f"⚠️  ALERT: {total_anomalies} total anomalies found!")
        logger.warning("="*80)
        return {
            'alert_needed': True,
            'total_anomalies': total_anomalies,
            'anomalies': anomalies_found
        }
    else:
        logger.info("="*80)
        logger.info("✓ No anomalies detected")
        logger.info("="*80)
        return {'alert_needed': False, 'total_anomalies': 0, 'anomalies': []}


def check_drift_results(drift_dir: str = '/opt/airflow/DataPipeline/data/ge_outputs/drift') -> dict:
    """
    Check drift detection results.
    
    Returns:
        {
            'drift_detected': bool,
            'total_drifted_features': int,
            'drift_details': [{'dataset': str, 'num_drifted': int, 'features': []}]
        }
    """
    logger.info("="*80)
    logger.info("Checking drift detection results...")
    logger.info("="*80)
    
    drift_files = []
    if os.path.exists(drift_dir):
        for root, dirs, files in os.walk(drift_dir):
            for file in files:
                if file.endswith('_drift.json'):
                    drift_files.append(os.path.join(root, file))
    
    if not drift_files:
        logger.info("No drift files found. Skipping drift check.")
        return {'drift_detected': False, 'total_drifted_features': 0, 'drift_details': []}
    
    drift_found = []
    total_drifted_features = 0
    
    for drift_file in drift_files:
        try:
            with open(drift_file, 'r') as f:
                results = json.load(f)
            
            dataset_name = results.get('dataset_name', 'Unknown')
            has_drift = results.get('has_drift', False)
            num_drifted = results.get('num_drifted_features', 0)
            drifted_features = results.get('drifted_features', [])
            
            logger.info(f"\nDataset: {dataset_name}")
            logger.info(f"  Drift Detected: {has_drift}")
            logger.info(f"  Drifted Features: {num_drifted}")
            
            if has_drift and num_drifted > 0:
                drift_found.append({
                    'dataset': dataset_name,
                    'num_drifted': num_drifted,
                    'features': [f['feature'] if isinstance(f, dict) else f for f in drifted_features[:5]]
                })
                total_drifted_features += num_drifted
        
        except Exception as e:
            logger.error(f"Error reading drift file {drift_file}: {e}")
    
    if drift_found:
        logger.warning("="*80)
        logger.warning(f"⚠️  DRIFT ALERT: {total_drifted_features} features drifted!")
        logger.warning("="*80)
        return {
            'drift_detected': True,
            'total_drifted_features': total_drifted_features,
            'drift_details': drift_found
        }
    else:
        logger.info("="*80)
        logger.info("✓ No drift detected")
        logger.info("="*80)
        return {'drift_detected': False, 'total_drifted_features': 0, 'drift_details': []}


def generate_alert_email_content(anomalies: list, total_anomalies: int, 
                                 drift_details: list, total_drifted: int,
                                 bias_details: list, total_biases: int,
                                 dag_run_id: str, execution_date: str,
                                 partition: str) -> tuple:
    """
    Generate both plain text and HTML email content.
    NOW INCLUDES BIAS DETECTION!
    
    Returns:
        (plain_text_body, html_body)
    """
    # Plain text version
    email_lines = [
        "MedScan AI Data Pipeline Alert",
        "=" * 60,
        f"Pipeline Run: {dag_run_id}",
        f"Execution Date: {execution_date}",
        f"Partition: {partition}",
        ""
    ]
    
    if anomalies:
        email_lines.extend([
            "🚨 DATA VALIDATION ALERTS",
            "=" * 60,
            f"Total Anomalies Detected: {total_anomalies}",
            ""
        ])
        
        for anom in anomalies:
            email_lines.append(f"Dataset: {anom['dataset']}")
            email_lines.append(f"  Anomalies: {anom['num_anomalies']}")
            if anom.get('details'):
                email_lines.append("  Details:")
                for detail in anom['details']:
                    if isinstance(detail, dict):
                        col = detail.get('column', 'Unknown')
                        desc = detail.get('description', 'No description')
                        email_lines.append(f"    - {col}: {desc}")
                    else:
                        email_lines.append(f"    - {detail}")
            email_lines.append("")
    
    if drift_details:
        email_lines.extend([
            "⚠️  DATA DRIFT ALERTS",
            "=" * 60,
            f"Total Drifted Features: {total_drifted}",
            ""
        ])
        
        for drift in drift_details:
            email_lines.append(f"Dataset: {drift['dataset']}")
            email_lines.append(f"  Drifted Features: {drift['num_drifted']}")
            email_lines.append(f"  Features: {', '.join(drift['features'])}")
            email_lines.append("")
    
    # ADD BIAS SECTION
    if bias_details:
        email_lines.extend([
            "⚠️  BIAS DETECTION ALERTS",
            "=" * 60,
            f"Total Biases Detected: {total_biases}",
            ""
        ])
        
        for bias in bias_details:
            email_lines.append(f"Dataset: {bias['dataset']}")
            email_lines.append(f"  Significant Biases: {bias['num_biases']}")
            if bias.get('details'):
                email_lines.append("  Problematic Slices:")
                for detail in bias['details'][:5]:
                    if isinstance(detail, dict):
                        slice_name = detail.get('slice', 'Unknown')
                        significance = detail.get('significance', 'N/A')
                        email_lines.append(f"    - {slice_name} (p={significance})")
                    else:
                        email_lines.append(f"    - {detail}")
            email_lines.append("")
    
    email_lines.extend([
        "RECOMMENDED ACTIONS",
        "=" * 60,
        "1. Review validation reports in GCS: gs://your-bucket/vision/ge_outputs/",
        "2. Check bias analysis HTML reports for detailed slicing analysis",
        "3. Review drift analysis for feature-level details",
        "4. Consider model retraining if significant drift detected",
        "5. Apply bias mitigation strategies before training models",
        "",
        f"View full logs in Airflow UI: {vision_gcp_config.AIRFLOW_URL}",
        f"GCS Console: {vision_gcp_config.get_gcs_console_url(f'vision/ge_outputs/{partition}')}",
    ])
    
    plain_text = "\n".join(email_lines)
    
    # HTML version
    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 0; }}
            .header {{ background-color: #ff4444; color: white; padding: 20px; text-align: center; }}
            .content {{ padding: 20px; background-color: #f9f9f9; }}
            .section {{ background: white; margin: 15px 0; padding: 15px; border-left: 4px solid #ff4444; }}
            .section h3 {{ margin-top: 0; color: #ff4444; }}
            .anomaly-item, .drift-item, .bias-item {{ margin: 10px 0; padding: 10px; background: #fff3cd; border-left: 3px solid #ffc107; }}
            .details {{ margin-left: 20px; font-size: 14px; color: #666; }}
            .actions {{ background: #e8f5e9; padding: 15px; border-left: 4px solid #4caf50; }}
            .footer {{ padding: 10px; text-align: center; font-size: 12px; color: #666; background: #333; color: white; }}
            ul {{ margin: 10px 0; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🚨 MedScan AI Pipeline Alert</h1>
            <p>Issues Detected in Data Pipeline</p>
        </div>
        <div class="content">
            <div class="section">
                <h3>Pipeline Information</h3>
                <p><strong>Run ID:</strong> {dag_run_id}</p>
                <p><strong>Execution Date:</strong> {execution_date}</p>
                <p><strong>Partition:</strong> {partition}</p>
            </div>
    """
    
    if anomalies:
        html_body += f"""
            <div class="section">
                <h3>🚨 DATA VALIDATION ALERTS</h3>
                <p><strong>Total Anomalies Detected:</strong> {total_anomalies}</p>
        """
        for anom in anomalies:
            html_body += f"""
                <div class="anomaly-item">
                    <strong>Dataset:</strong> {anom['dataset']}<br>
                    <strong>Anomalies:</strong> {anom['num_anomalies']}<br>
            """
            if anom.get('details'):
                html_body += "<div class='details'><strong>Details:</strong><ul>"
                for detail in anom['details']:
                    if isinstance(detail, dict):
                        col = detail.get('column', 'Unknown')
                        desc = detail.get('description', 'No description')
                        html_body += f"<li><strong>{col}:</strong> {desc}</li>"
                    else:
                        html_body += f"<li>{detail}</li>"
                html_body += "</ul></div>"
            html_body += "</div>"
        html_body += "</div>"
    
    if drift_details:
        html_body += f"""
            <div class="section">
                <h3>⚠️ DATA DRIFT ALERTS</h3>
                <p><strong>Total Drifted Features:</strong> {total_drifted}</p>
        """
        for drift in drift_details:
            html_body += f"""
                <div class="drift-item">
                    <strong>Dataset:</strong> {drift['dataset']}<br>
                    <strong>Drifted Features:</strong> {drift['num_drifted']}<br>
                    <strong>Features:</strong> {', '.join(drift['features'])}
                </div>
            """
        html_body += "</div>"
    
    # ADD BIAS SECTION TO HTML
    if bias_details:
        html_body += f"""
            <div class="section">
                <h3>⚠️ BIAS DETECTION ALERTS</h3>
                <p><strong>Total Biases Detected:</strong> {total_biases}</p>
                <p style="color: #d32f2f;">Bias detected in data slicing analysis. Review bias reports before model training.</p>
        """
        for bias in bias_details:
            html_body += f"""
                <div class="bias-item">
                    <strong>Dataset:</strong> {bias['dataset']}<br>
                    <strong>Significant Biases:</strong> {bias['num_biases']}<br>
            """
            if bias.get('details'):
                html_body += "<div class='details'><strong>Problematic Slices:</strong><ul>"
                for detail in bias['details'][:5]:
                    if isinstance(detail, dict):
                        slice_name = detail.get('slice', 'Unknown')
                        sig = detail.get('significance', 'N/A')
                        html_body += f"<li><strong>{slice_name}</strong> (p={sig})</li>"
                    else:
                        html_body += f"<li>{detail}</li>"
                html_body += "</ul></div>"
            html_body += "</div>"
        html_body += "</div>"
    
    html_body += f"""
            <div class="actions">
                <h3>📋 RECOMMENDED ACTIONS</h3>
                <ol>
                    <li>Review validation reports in GCS: <code>gs://{vision_gcp_config.GCS_BUCKET_NAME}/vision/ge_outputs/{partition}/</code></li>
                    <li><strong>Review bias analysis HTML reports</strong> for detailed slicing insights</li>
                    <li>Check drift analysis for feature-level details</li>
                    <li>Consider data augmentation or resampling to address bias</li>
                    <li>Apply bias mitigation strategies before model training</li>
                    <li>Consider retraining if significant drift detected</li>
                </ol>
                <p>
                    <a href="{vision_gcp_config.AIRFLOW_URL}" style="color: #4caf50; font-weight: bold;">View Airflow Logs →</a> | 
                    <a href="{vision_gcp_config.get_gcs_console_url(f'vision/ge_outputs/{partition}')}" style="color: #2196F3; font-weight: bold;">View GCS Reports →</a>
                </p>
            </div>
        </div>
        <div class="footer">
            <p>MedScan AI Automated Data Pipeline - Vision Module</p>
            <p>GCS Bucket: {vision_gcp_config.GCS_BUCKET_NAME} | Partition: {partition}</p>
        </div>
    </body>
    </html>
    """
    
    return plain_text, html_body