"""
send_notification.py - Email notification utility for RAG pipeline
Supports: training completion, validation failures, bias check failures, pipeline failures
"""
import smtplib
import os
import sys
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Dict, Any, List
from datetime import datetime

# Default recipients
DEFAULT_RECIPIENTS = [
    'harshitha8.shekar@gmail.com',
    'chandrashekar.h@northeastern.edu'
]


def get_smtp_credentials() -> tuple:
    """Get SMTP credentials from Secret Manager or environment"""
    smtp_user = os.getenv('SMTP_USER')
    smtp_password = os.getenv('SMTP_PASSWORD')
    
    if not smtp_user or not smtp_password:
        # Try to get from Secret Manager if gcloud is available
        try:
            import subprocess
            project_id = os.getenv('GCP_PROJECT_ID', 'medscanai-476500')
            
            smtp_user_cmd = subprocess.run(
                ['gcloud', 'secrets', 'versions', 'access', 'latest', 
                 '--secret=smtp-username', f'--project={project_id}'],
                capture_output=True, text=True, check=False
            )
            smtp_password_cmd = subprocess.run(
                ['gcloud', 'secrets', 'versions', 'access', 'latest',
                 '--secret=smtp-password', f'--project={project_id}'],
                capture_output=True, text=True, check=False
            )
            
            if smtp_user_cmd.returncode == 0:
                smtp_user = smtp_user_cmd.stdout.strip()
            if smtp_password_cmd.returncode == 0:
                smtp_password = smtp_password_cmd.stdout.strip()
        except Exception:
            pass
    
    return smtp_user, smtp_password


def send_email(
    subject: str,
    plain_text: str,
    html: str,
    recipients: Optional[List[str]] = None
) -> bool:
    """
    Send email notification
    
    Args:
        subject: Email subject
        plain_text: Plain text body
        html: HTML body
        recipients: List of recipient emails
        
    Returns:
        True if sent successfully, False otherwise
    """
    smtp_user, smtp_password = get_smtp_credentials()
    
    if not smtp_user or not smtp_password:
        print("SMTP credentials not available - skipping email")
        return False
    
    if recipients is None:
        recipients = DEFAULT_RECIPIENTS
    
    msg = MIMEMultipart('alternative')
    msg['From'] = f'MedScan AI <{smtp_user}>'
    msg['To'] = ', '.join(recipients)
    msg['Subject'] = subject
    
    msg.attach(MIMEText(plain_text, 'plain'))
    msg.attach(MIMEText(html, 'html'))
    
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        print(f"Email sent successfully to {len(recipients)} recipients")
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False


def send_training_completion_email(
    build_id: str,
    model_name: str,
    composite_score: float,
    validation_passed: bool,
    bias_check_passed: bool,
    bias_details: Optional[Dict[str, Any]] = None,
    project_id: Optional[str] = None,
    bucket_name: Optional[str] = None
):
    """Send email notification for training completion"""
    project_id = project_id or os.getenv('GCP_PROJECT_ID', 'medscanai-476500')
    bucket_name = bucket_name or os.getenv('GCS_BUCKET_NAME', 'medscan-pipeline-medscanai-476500')
    
    # Determine overall status
    if validation_passed and bias_check_passed:
        status_color = "#4CAF50"
        status_text = "SUCCESS"
        status_emoji = "✅"
    else:
        status_color = "#FF9800"
        status_text = "COMPLETED WITH WARNINGS"
        status_emoji = "⚠️"
    
    subject = f"{status_emoji} RAG Model Training Complete - {build_id}"
    
    # Build validation section
    validation_section = ""
    if validation_passed:
        validation_section = """
        <h3 style="color: #4CAF50;">✅ Validation: PASSED</h3>
        <p>Model meets minimum performance threshold.</p>
        """
    else:
        validation_section = """
        <h3 style="color: #f44336;">❌ Validation: FAILED</h3>
        <p>Model does not meet minimum performance threshold.</p>
        """
    
    # Build bias check section
    bias_section = ""
    if bias_check_passed:
        bias_section = """
        <h3 style="color: #4CAF50;">✅ Bias Check: PASSED</h3>
        <p>No significant bias detected in model performance.</p>
        """
    else:
        bias_section = """
        <h3 style="color: #f44336;">❌ Bias Check: FAILED</h3>
        <p>Bias detected in model performance across data slices.</p>
        """
        if bias_details:
            violations = bias_details.get('violations', [])
            if violations:
                bias_section += "<ul>"
                for v in violations[:5]:  # Show first 5
                    bias_section += f"<li>{v}</li>"
                bias_section += "</ul>"
    
    plain_text = f"""RAG Model Training Completed

Model Details:
- Model: {model_name}
- Composite Score: {composite_score:.4f}
- Build ID: {build_id}
- Completed: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC

Validation: {'PASSED' if validation_passed else 'FAILED'}
Bias Check: {'PASSED' if bias_check_passed else 'FAILED'}

View build:
https://console.cloud.google.com/cloud-build/builds/{build_id}?project={project_id}

Model artifacts:
gs://{bucket_name}/RAG/models/{build_id}/

MedScan AI - Automated Training System
"""
    
    html = f"""
    <html>
    <body style="font-family: Arial, sans-serif;">
        <div style="background-color: {status_color}; color: white; padding: 20px;">
            <h2>{status_emoji} RAG Model Training Complete</h2>
        </div>
        <div style="padding: 20px;">
            <h3>Model Details</h3>
            <table style="border-collapse: collapse; width: 100%;">
                <tr><td style="padding: 8px;"><strong>Model:</strong></td><td>{model_name}</td></tr>
                <tr style="background-color: #f9f9f9;"><td style="padding: 8px;"><strong>Composite Score:</strong></td><td style="font-weight: bold;">{composite_score:.4f}</td></tr>
                <tr><td style="padding: 8px;"><strong>Build ID:</strong></td><td>{build_id}</td></tr>
                <tr style="background-color: #f9f9f9;"><td style="padding: 8px;"><strong>Completed:</strong></td><td>{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</td></tr>
            </table>
            
            {validation_section}
            {bias_section}
            
            <p><a href="https://console.cloud.google.com/cloud-build/builds/{build_id}?project={project_id}" 
               style="background-color: #2196F3; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px;">
               View Build Logs →
            </a></p>
        </div>
        <div style="background-color: #f5f5f5; padding: 15px; text-align: center;">
            <p>MedScan AI - Automated Training System</p>
        </div>
    </body>
    </html>
    """
    
    send_email(subject, plain_text, html)


def send_validation_failure_email(
    build_id: str,
    model_name: str,
    composite_score: float,
    threshold: float,
    project_id: Optional[str] = None
):
    """Send email notification for validation failure"""
    project_id = project_id or os.getenv('GCP_PROJECT_ID', 'medscanai-476500')
    
    subject = "❌ RAG Model Validation Failed"
    
    plain_text = f"""RAG Model Validation Failed

Model Details:
- Model: {model_name}
- Composite Score: {composite_score:.4f}
- Required Threshold: {threshold:.4f}
- Build ID: {build_id}
- Failed: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC

The model's composite score ({composite_score:.4f}) is below the minimum threshold ({threshold:.4f}).

Action Required:
- Review model selection criteria
- Check training data quality
- Consider adjusting hyperparameters
- Model will not be deployed

View build logs:
https://console.cloud.google.com/cloud-build/builds/{build_id}?project={project_id}

MedScan AI - Automated Training System
"""
    
    html = f"""
    <html>
    <body style="font-family: Arial, sans-serif;">
        <div style="background-color: #f44336; color: white; padding: 20px;">
            <h2>❌ RAG Model Validation Failed</h2>
        </div>
        <div style="padding: 20px;">
            <h3>Model Details</h3>
            <table style="border-collapse: collapse; width: 100%;">
                <tr><td style="padding: 8px;"><strong>Model:</strong></td><td>{model_name}</td></tr>
                <tr style="background-color: #f9f9f9;"><td style="padding: 8px;"><strong>Composite Score:</strong></td><td style="color: #f44336; font-weight: bold;">{composite_score:.4f}</td></tr>
                <tr><td style="padding: 8px;"><strong>Required Threshold:</strong></td><td>{threshold:.4f}</td></tr>
                <tr style="background-color: #f9f9f9;"><td style="padding: 8px;"><strong>Build ID:</strong></td><td>{build_id}</td></tr>
            </table>
            
            <div style="background-color: #ffebee; padding: 15px; border-left: 4px solid #f44336; margin: 20px 0;">
                <p><strong>Validation Failed:</strong> The model's composite score ({composite_score:.4f}) is below the minimum threshold ({threshold:.4f}).</p>
            </div>
            
            <h3>Action Required</h3>
            <ul>
                <li>Review model selection criteria</li>
                <li>Check training data quality</li>
                <li>Consider adjusting hyperparameters</li>
                <li><strong>Model will not be deployed</strong></li>
            </ul>
            
            <p><a href="https://console.cloud.google.com/cloud-build/builds/{build_id}?project={project_id}" 
               style="background-color: #2196F3; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px;">
               View Build Logs →
            </a></p>
        </div>
        <div style="background-color: #f5f5f5; padding: 15px; text-align: center;">
            <p>MedScan AI - Automated Training System</p>
        </div>
    </body>
    </html>
    """
    
    send_email(subject, plain_text, html)


def send_bias_failure_email(
    build_id: str,
    model_name: str,
    bias_details: Dict[str, Any],
    project_id: Optional[str] = None
):
    """Send email notification for bias check failure"""
    project_id = project_id or os.getenv('GCP_PROJECT_ID', 'medscanai-476500')
    
    subject = "⚠️ RAG Model Bias Check Failed"
    
    violations = bias_details.get('violations', [])
    num_violations = bias_details.get('num_violations', 0)
    recommendations = bias_details.get('recommendations', [])
    
    violations_text = "\n".join([f"  - {v}" for v in violations[:10]])
    recommendations_text = "\n".join([f"  - {r}" for r in recommendations[:5]])
    
    plain_text = f"""RAG Model Bias Check Failed

Model Details:
- Model: {model_name}
- Build ID: {build_id}
- Failed: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC

Bias Detection Results:
- Bias Detected: YES
- Number of Violations: {num_violations}

Violations:
{violations_text}

Recommendations:
{recommendations_text}

Action Required:
- Review bias analysis results
- Consider applying mitigation strategies
- Model deployment may be affected

View build logs:
https://console.cloud.google.com/cloud-build/builds/{build_id}?project={project_id}

MedScan AI - Automated Training System
"""
    
    violations_html = "<ul>"
    for v in violations[:10]:
        violations_html += f"<li>{v}</li>"
    violations_html += "</ul>"
    
    recommendations_html = "<ul>"
    for r in recommendations[:5]:
        recommendations_html += f"<li>{r}</li>"
    recommendations_html += "</ul>"
    
    html = f"""
    <html>
    <body style="font-family: Arial, sans-serif;">
        <div style="background-color: #FF9800; color: white; padding: 20px;">
            <h2>⚠️ RAG Model Bias Check Failed</h2>
        </div>
        <div style="padding: 20px;">
            <h3>Model Details</h3>
            <table style="border-collapse: collapse; width: 100%;">
                <tr><td style="padding: 8px;"><strong>Model:</strong></td><td>{model_name}</td></tr>
                <tr style="background-color: #f9f9f9;"><td style="padding: 8px;"><strong>Build ID:</strong></td><td>{build_id}</td></tr>
            </table>
            
            <h3 style="color: #FF9800;">Bias Detection Results</h3>
            <ul>
                <li><strong>Bias Detected:</strong> YES</li>
                <li><strong>Number of Violations:</strong> {num_violations}</li>
            </ul>
            
            <h4>Violations:</h4>
            <div style="background-color: #fff3e0; padding: 15px; border-left: 4px solid #FF9800;">
                {violations_html}
            </div>
            
            <h4>Recommendations:</h4>
            <div style="background-color: #e3f2fd; padding: 15px; border-left: 4px solid #2196F3;">
                {recommendations_html}
            </div>
            
            <h3>Action Required</h3>
            <ul>
                <li>Review bias analysis results</li>
                <li>Consider applying mitigation strategies</li>
                <li>Model deployment may be affected</li>
            </ul>
            
            <p><a href="https://console.cloud.google.com/cloud-build/builds/{build_id}?project={project_id}" 
               style="background-color: #2196F3; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px;">
               View Build Logs →
            </a></p>
        </div>
        <div style="background-color: #f5f5f5; padding: 15px; text-align: center;">
            <p>MedScan AI - Automated Training System</p>
        </div>
    </body>
    </html>
    """
    
    send_email(subject, plain_text, html)


def send_pipeline_failure_email(
    build_id: str,
    failed_step: str,
    error_message: str,
    project_id: Optional[str] = None
):
    """Send email notification for pipeline failure"""
    project_id = project_id or os.getenv('GCP_PROJECT_ID', 'medscanai-476500')
    
    subject = "❌ RAG Pipeline Failed"
    
    plain_text = f"""RAG Pipeline Failed

Build Details:
- Build ID: {build_id}
- Failed Step: {failed_step}
- Failed: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC

Error:
{error_message[:500]}

Action Required:
- Review build logs for details
- Check data availability in GCS
- Verify configuration
- Fix issues and retry

View error logs:
https://console.cloud.google.com/cloud-build/builds/{build_id}?project={project_id}

MedScan AI - Automated Training System
"""
    
    html = f"""
    <html>
    <body style="font-family: Arial, sans-serif;">
        <div style="background-color: #f44336; color: white; padding: 20px;">
            <h2>❌ RAG Pipeline Failed</h2>
        </div>
        <div style="padding: 20px;">
            <h3>Build Details</h3>
            <table style="border-collapse: collapse; width: 100%;">
                <tr><td style="padding: 8px;"><strong>Build ID:</strong></td><td>{build_id}</td></tr>
                <tr style="background-color: #f9f9f9;"><td style="padding: 8px;"><strong>Failed Step:</strong></td><td style="color: #f44336; font-weight: bold;">{failed_step}</td></tr>
                <tr><td style="padding: 8px;"><strong>Failed:</strong></td><td>{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</td></tr>
            </table>
            
            <h3 style="color: #f44336;">Error Details</h3>
            <div style="background-color: #ffebee; padding: 15px; border-left: 4px solid #f44336;">
                <pre style="white-space: pre-wrap; word-wrap: break-word;">{error_message[:1000]}</pre>
            </div>
            
            <h3>Action Required</h3>
            <ul>
                <li>Review build logs for details</li>
                <li>Check data availability in GCS</li>
                <li>Verify configuration</li>
                <li>Fix issues and retry</li>
            </ul>
            
            <p><a href="https://console.cloud.google.com/cloud-build/builds/{build_id}?project={project_id}" 
               style="background-color: #2196F3; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px;">
               View Error Logs →
            </a></p>
        </div>
        <div style="background-color: #f5f5f5; padding: 15px; text-align: center;">
            <p>MedScan AI - Automated Training System</p>
        </div>
    </body>
    </html>
    """
    
    send_email(subject, plain_text, html)


if __name__ == "__main__":
    # CLI interface
    import argparse
    
    parser = argparse.ArgumentParser(description='Send RAG pipeline notification email')
    parser.add_argument('--type', required=True, 
                       choices=['completion', 'validation-failure', 'bias-failure', 'pipeline-failure'])
    parser.add_argument('--build-id', required=True)
    parser.add_argument('--model-name', help='Model name')
    parser.add_argument('--composite-score', type=float, help='Composite score')
    parser.add_argument('--threshold', type=float, help='Validation threshold')
    parser.add_argument('--validation-passed', type=bool, help='Validation passed')
    parser.add_argument('--bias-check-passed', type=bool, help='Bias check passed')
    parser.add_argument('--bias-details', help='Bias details JSON file path')
    parser.add_argument('--failed-step', help='Failed step name')
    parser.add_argument('--error-message', help='Error message')
    
    args = parser.parse_args()
    
    if args.type == 'completion':
        bias_details = None
        if args.bias_details and os.path.exists(args.bias_details):
            with open(args.bias_details, 'r') as f:
                bias_details = json.load(f)
        
        # Parse boolean strings
        validation_passed = str(args.validation_passed).lower() in ('true', '1', 'yes') if args.validation_passed else False
        bias_check_passed = str(args.bias_check_passed).lower() in ('true', '1', 'yes') if args.bias_check_passed else False
        
        send_training_completion_email(
            build_id=args.build_id,
            model_name=args.model_name or 'Unknown',
            composite_score=args.composite_score or 0.0,
            validation_passed=validation_passed,
            bias_check_passed=bias_check_passed,
            bias_details=bias_details
        )
    elif args.type == 'validation-failure':
        send_validation_failure_email(
            build_id=args.build_id,
            model_name=args.model_name or 'Unknown',
            composite_score=args.composite_score or 0.0,
            threshold=args.threshold or 0.5
        )
    elif args.type == 'bias-failure':
        bias_details = {}
        if args.bias_details and os.path.exists(args.bias_details):
            with open(args.bias_details, 'r') as f:
                bias_details = json.load(f)
        
        send_bias_failure_email(
            build_id=args.build_id,
            model_name=args.model_name or 'Unknown',
            bias_details=bias_details
        )
    elif args.type == 'pipeline-failure':
        send_pipeline_failure_email(
            build_id=args.build_id,
            failed_step=args.failed_step or 'Unknown',
            error_message=args.error_message or 'Unknown error'
        )

