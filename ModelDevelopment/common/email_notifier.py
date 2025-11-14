"""
email_notifier.py - Standalone email notification helper
Can be used in training scripts or Cloud Build steps
"""
import smtplib
import os
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def send_training_notification(
    status: str,
    model_name: str,
    dataset: str,
    accuracy: Optional[float] = None,
    build_id: Optional[str] = None,
    error_message: Optional[str] = None,
    smtp_user: str = None,
    smtp_password: str = None,
    recipients: List[str] = None
):
    """
    Send training status notification email.
    
    Args:
        status: 'success' or 'failure'
        model_name: Name of the model
        dataset: Dataset name
        accuracy: Test accuracy (if success)
        build_id: Cloud Build ID
        error_message: Error message (if failure)
        smtp_user: SMTP username (defaults to env var)
        smtp_password: SMTP password (defaults to env var)
        recipients: Email addresses (defaults to config)
    """
    # Get SMTP credentials
    if smtp_user is None:
        smtp_user = os.getenv('SMTP_USER')
    if smtp_password is None:
        smtp_password = os.getenv('SMTP_PASSWORD')
    
    if not smtp_user or not smtp_password:
        logger.warning("SMTP credentials not provided - skipping email")
        return
    
    # Default recipients
    if recipients is None:
        recipients = [
            'harshitha8.shekar@gmail.com',
            'kothari.sau@northeastern.edu'
        ]
    
    # Create message
    msg = MIMEMultipart('alternative')
    msg['From'] = f'MedScan AI <{smtp_user}>'
    msg['To'] = ', '.join(recipients)
    
    # Subject based on status
    if status == 'success':
        emoji = ''
        msg['Subject'] = f'{emoji} MedScan AI - {model_name} Training Complete'
    else:
        emoji = ''
        msg['Subject'] = f'{emoji} MedScan AI - {model_name} Training Failed'
    
    # Plain text body
    if status == 'success':
        plain_text = f"""Model training completed successfully!

Model Details:
- Model: {model_name}
- Dataset: {dataset}
- Test Accuracy: {accuracy:.2%}
- Build ID: {build_id or 'N/A'}
- Completed: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC

Validation Results:
- Accuracy: {accuracy:.2%} (threshold: 70%)
- Status: PASSED

View build logs:
https://console.cloud.google.com/cloud-build/builds/{build_id}?project=medscanai-476203

Model artifacts:
gs://medscan-data/vision/trained_models/{build_id}/

---
MedScan AI - Automated Training System
"""
    else:
        plain_text = f"""Model training failed.

Model Details:
- Model: {model_name}
- Dataset: {dataset}
- Build ID: {build_id or 'N/A'}
- Failed: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC

Error:
{error_message or 'Unknown error - check build logs'}

Action Required:
- Review build logs for details
- Check data availability
- Verify configuration

View error logs:
https://console.cloud.google.com/cloud-build/builds/{build_id}?project=medscanai-476203

---
MedScan AI - Automated Training System
"""
    
    # HTML body
    if status == 'success':
        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <div style="background-color: #4CAF50; color: white; padding: 20px;">
                <h2>Model Training Complete</h2>
            </div>
            <div style="padding: 20px;">
                <h3>Model Details</h3>
                <table style="border-collapse: collapse; width: 100%;">
                    <tr><td style="padding: 8px;"><strong>Model:</strong></td><td>{model_name}</td></tr>
                    <tr style="background-color: #f9f9f9;"><td style="padding: 8px;"><strong>Dataset:</strong></td><td>{dataset}</td></tr>
                    <tr><td style="padding: 8px;"><strong>Test Accuracy:</strong></td><td style="color: #4CAF50; font-weight: bold;">{accuracy:.2%}</td></tr>
                    <tr style="background-color: #f9f9f9;"><td style="padding: 8px;"><strong>Build ID:</strong></td><td>{build_id or 'N/A'}</td></tr>
                    <tr><td style="padding: 8px;"><strong>Completed:</strong></td><td>{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</td></tr>
                </table>
                
                <h3 style="color: #4CAF50;">Validation Results</h3>
                <ul>
                    <li> Accuracy: {accuracy:.2%} (threshold: 70%)</li>
                    <li> Status: PASSED</li>
                </ul>
                
                <p><a href="https://console.cloud.google.com/cloud-build/builds/{build_id}?project=medscanai-476203" 
                   style="background-color: #2196F3; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px;">
                   View Build Logs →
                </a></p>
            </div>
            <div style="background-color: #f5f5f5; padding: 15px; text-align: center; margin-top: 20px;">
                <p style="margin: 0;">MedScan AI - Automated Training System</p>
            </div>
        </body>
        </html>
        """
    else:
        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <div style="background-color: #f44336; color: white; padding: 20px;">
                <h2> Model Training Failed</h2>
            </div>
            <div style="padding: 20px;">
                <h3>Model Details</h3>
                <table style="border-collapse: collapse; width: 100%;">
                    <tr><td style="padding: 8px;"><strong>Model:</strong></td><td>{model_name}</td></tr>
                    <tr style="background-color: #f9f9f9;"><td style="padding: 8px;"><strong>Dataset:</strong></td><td>{dataset}</td></tr>
                    <tr><td style="padding: 8px;"><strong>Build ID:</strong></td><td>{build_id or 'N/A'}</td></tr>
                    <tr style="background-color: #f9f9f9;"><td style="padding: 8px;"><strong>Failed:</strong></td><td>{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</td></tr>
                </table>
                
                <h3 style="color: #f44336;">Error Details</h3>
                <div style="background-color: #ffebee; padding: 15px; border-left: 4px solid #f44336;">
                    <pre>{error_message or 'Unknown error - check build logs'}</pre>
                </div>
                
                <h3>Action Required</h3>
                <ul>
                    <li>Review build logs for details</li>
                    <li>Check data availability in GCS</li>
                    <li>Verify training configuration</li>
                </ul>
                
                <p><a href="https://console.cloud.google.com/cloud-build/builds/{build_id}?project=medscanai-476203" 
                   style="background-color: #2196F3; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px;">
                   View Error Logs →
                </a></p>
            </div>
            <div style="background-color: #f5f5f5; padding: 15px; text-align: center; margin-top: 20px;">
                <p style="margin: 0;">MedScan AI - Automated Training System</p>
            </div>
        </body>
        </html>
        """
    
    msg.attach(MIMEText(plain_text, 'plain'))
    msg.attach(MIMEText(html, 'html'))
    
    # Send email
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        
        logger.info(f" Email notification sent ({status}) to {len(recipients)} recipients")
        
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        raise


def main():
    """CLI interface for sending notifications"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Send training notification email')
    parser.add_argument('--status', required=True, choices=['success', 'failure'])
    parser.add_argument('--model', required=True, help='Model name')
    parser.add_argument('--dataset', required=True, help='Dataset name')
    parser.add_argument('--accuracy', type=float, help='Test accuracy (for success)')
    parser.add_argument('--build-id', help='Cloud Build ID')
    parser.add_argument('--error', help='Error message (for failure)')
    
    args = parser.parse_args()
    
    send_training_notification(
        status=args.status,
        model_name=args.model,
        dataset=args.dataset,
        accuracy=args.accuracy,
        build_id=args.build_id,
        error_message=args.error
    )


if __name__ == "__main__":
    main()