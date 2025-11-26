"""
send_notification.py - Email notification utility for Vision pipeline
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
    'kothari.sau@northeastern.edu'
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
    dataset: str,
    test_accuracy: float,
    validation_passed: bool,
    bias_check_passed: bool,
    bias_details: Optional[Dict[str, Any]] = None,
    project_id: Optional[str] = None,
    bucket_name: Optional[str] = None
):
    """Send training completion email"""
    project_id = project_id or os.getenv('GCP_PROJECT_ID', 'medscanai-476500')
    bucket_name = bucket_name or os.getenv('GCS_BUCKET_NAME', 'medscan-pipeline-medscanai-476500')
    
    validation_status = "PASSED" if validation_passed else "FAILED"
    bias_status = "PASSED" if bias_check_passed else "FAILED"
    
    subject = f"Vision Model Training Complete - {model_name} ({dataset})"
    
    plain_text = f"""Vision Model Training Status

Model Details:
- Model: {model_name}
- Dataset: {dataset}
- Test Accuracy: {test_accuracy:.2f}%
- Build ID: {build_id}

Results:
- Validation: {validation_status}
- Bias Check: {bias_status}
- Deployment: {'SUCCESS' if validation_passed and bias_check_passed else 'SKIPPED/FAILED'}

View build:
https://console.cloud.google.com/cloud-build/builds/{build_id}?project={project_id}

Model artifacts:
gs://{bucket_name}/vision/trained_models/{build_id}/

---
MedScan AI - Automated Training System
"""
    if not bias_check_passed:
        bias_gcs_path = f"vision/validation/{build_id}/bias_results.json"
        bias_console_url = f"https://console.cloud.google.com/storage/browser/{bucket_name}/vision/validation/{build_id}?project={project_id}"
        plain_text += f"""
Bias report:
- gs://{bucket_name}/{bias_gcs_path}
- Console: {bias_console_url}
"""
    
    validation_color = "green" if validation_passed else "red"
    bias_color = "green" if bias_check_passed else "red"
    deployment_color = "green" if validation_passed and bias_check_passed else "red"
    
    html = f"""
    <html>
    <body style="font-family: Arial, sans-serif;">
        <div style="background-color: {'#4CAF50' if validation_passed and bias_check_passed else '#f44336'}; color: white; padding: 20px;">
            <h2>Vision Model Training {'SUCCESS' if validation_passed and bias_check_passed else 'FAILURE'}</h2>
        </div>
        <div style="padding: 20px;">
            <h3>Model Details</h3>
            <ul>
                <li><strong>Model:</strong> {model_name}</li>
                <li><strong>Dataset:</strong> {dataset}</li>
                <li><strong>Test Accuracy:</strong> {test_accuracy:.2f}%</li>
                <li><strong>Build ID:</strong> {build_id}</li>
            </ul>
            
            <h3>Results</h3>
            <ul>
                <li style="color: {validation_color};">Validation: {validation_status}</li>
                <li style="color: {bias_color};">Bias Check: {bias_status}</li>
                <li style="color: {deployment_color};">Deployment: {'SUCCESS' if validation_passed and bias_check_passed else 'SKIPPED/FAILED'}</li>
            </ul>
            {"<p><a href=\"https://console.cloud.google.com/storage/browser/"
             + f"{bucket_name}/vision/validation/{build_id}?project={project_id}\">View bias report →</a></p>"
             if not bias_check_passed else ""}
            
            <p><a href="https://console.cloud.google.com/cloud-build/builds/{build_id}?project={project_id}">View Build Logs →</a></p>
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
    dataset: str,
    test_accuracy: float,
    threshold: float,
    project_id: Optional[str] = None
):
    """Send validation failure email"""
    project_id = project_id or os.getenv('GCP_PROJECT_ID', 'medscanai-476500')
    
    subject = f"Vision Model Validation FAILED - {model_name} ({dataset})"
    
    plain_text = f"""Vision Model Validation Failed

Model Details:
- Model: {model_name}
- Dataset: {dataset}
- Test Accuracy: {test_accuracy:.2f}%
- Required Threshold: {threshold:.2f}%
- Build ID: {build_id}

The model did not meet the minimum performance threshold and will NOT be deployed.

View build:
https://console.cloud.google.com/cloud-build/builds/{build_id}?project={project_id}

---
MedScan AI - Automated Training System
"""
    
    html = f"""
    <html>
    <body style="font-family: Arial, sans-serif;">
        <div style="background-color: #f44336; color: white; padding: 20px;">
            <h2>Vision Model Validation FAILED</h2>
        </div>
        <div style="padding: 20px;">
            <h3>Model Details</h3>
            <ul>
                <li><strong>Model:</strong> {model_name}</li>
                <li><strong>Dataset:</strong> {dataset}</li>
                <li><strong>Test Accuracy:</strong> {test_accuracy:.2f}%</li>
                <li><strong>Required Threshold:</strong> {threshold:.2f}%</li>
                <li><strong>Build ID:</strong> {build_id}</li>
            </ul>
            
            <p style="color: red; font-weight: bold;">
                The model did not meet the minimum performance threshold and will NOT be deployed.
            </p>
            
            <p><a href="https://console.cloud.google.com/cloud-build/builds/{build_id}?project={project_id}">View Build Logs →</a></p>
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
    dataset: str,
    bias_details: Dict[str, Any],
    project_id: Optional[str] = None
):
    """Send bias failure email"""
    project_id = project_id or os.getenv('GCP_PROJECT_ID', 'medscanai-476500')
    
    num_violations = bias_details.get('num_violations', 0)
    violations = bias_details.get('violations', [])
    
    subject = f"Vision Model Bias Detected - {model_name} ({dataset})"
    
    plain_text = f"""Vision Model Bias Detection Alert

Model Details:
- Model: {model_name}
- Dataset: {dataset}
- Build ID: {build_id}

Bias Detection Results:
- Bias Detected: YES
- Number of Violations: {num_violations}

"""
    
    if violations:
        plain_text += "Violations:\n"
        for i, violation in enumerate(violations[:5], 1):  # Limit to first 5
            plain_text += f"  {i}. {violation.get('description', 'Unknown violation')}\n"
        if len(violations) > 5:
            plain_text += f"  ... and {len(violations) - 5} more violations\n"
    
    plain_text += f"""
View build:
https://console.cloud.google.com/cloud-build/builds/{build_id}?project={project_id}

---
MedScan AI - Automated Training System
"""
    
    violations_html = ""
    if violations:
        violations_html = "<ul>"
        for violation in violations[:5]:
            violations_html += f"<li>{violation.get('description', 'Unknown violation')}</li>"
        if len(violations) > 5:
            violations_html += f"<li>... and {len(violations) - 5} more violations</li>"
        violations_html += "</ul>"
    
    html = f"""
    <html>
    <body style="font-family: Arial, sans-serif;">
        <div style="background-color: #ff9800; color: white; padding: 20px;">
            <h2>Vision Model Bias Detected</h2>
        </div>
        <div style="padding: 20px;">
            <h3>Model Details</h3>
            <ul>
                <li><strong>Model:</strong> {model_name}</li>
                <li><strong>Dataset:</strong> {dataset}</li>
                <li><strong>Build ID:</strong> {build_id}</li>
            </ul>
            
            <h3>Bias Detection Results</h3>
            <ul>
                <li style="color: red;"><strong>Bias Detected:</strong> YES</li>
                <li><strong>Number of Violations:</strong> {num_violations}</li>
            </ul>
            
            {violations_html}
            
            <p><a href="https://console.cloud.google.com/cloud-build/builds/{build_id}?project={project_id}">View Build Logs →</a></p>
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
    """Send pipeline failure email"""
    project_id = project_id or os.getenv('GCP_PROJECT_ID', 'medscanai-476500')
    
    subject = f"Vision Pipeline FAILED - Step: {failed_step}"
    
    plain_text = f"""Vision Model Training Pipeline Failed

Build Details:
- Build ID: {build_id}
- Failed Step: {failed_step}

Error Message:
{error_message}

View build:
https://console.cloud.google.com/cloud-build/builds/{build_id}?project={project_id}

---
MedScan AI - Automated Training System
"""
    
    html = f"""
    <html>
    <body style="font-family: Arial, sans-serif;">
        <div style="background-color: #f44336; color: white; padding: 20px;">
            <h2>Vision Pipeline FAILED</h2>
        </div>
        <div style="padding: 20px;">
            <h3>Build Details</h3>
            <ul>
                <li><strong>Build ID:</strong> {build_id}</li>
                <li><strong>Failed Step:</strong> {failed_step}</li>
            </ul>
            
            <h3>Error Message</h3>
            <pre style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto;">{error_message[:500]}</pre>
            
            <p><a href="https://console.cloud.google.com/cloud-build/builds/{build_id}?project={project_id}">View Build Logs →</a></p>
        </div>
        <div style="background-color: #f5f5f5; padding: 15px; text-align: center;">
            <p>MedScan AI - Automated Training System</p>
        </div>
    </body>
    </html>
    """
    
    send_email(subject, plain_text, html)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Send Vision pipeline email notifications')
    parser.add_argument('--type', required=True, 
                       choices=['completion', 'validation-failure', 'bias-failure', 'pipeline-failure'],
                       help='Email type')
    parser.add_argument('--build-id', required=True, help='Cloud Build ID')
    parser.add_argument('--model-name', help='Model name')
    parser.add_argument('--dataset', help='Dataset name')
    parser.add_argument('--test-accuracy', type=float, help='Test accuracy')
    parser.add_argument('--validation-passed', help='Validation passed (true/false)')
    parser.add_argument('--bias-check-passed', help='Bias check passed (true/false)')
    parser.add_argument('--threshold', type=float, help='Validation threshold')
    parser.add_argument('--bias-details', help='Path to bias details JSON file')
    parser.add_argument('--bias-report-path', help='GCS path to bias report')
    parser.add_argument('--failed-step', help='Failed step name')
    parser.add_argument('--error-message', help='Error message')
    
    args = parser.parse_args()
    
    if args.type == 'completion':
        bias_details = None
        if args.bias_details:
            with open(args.bias_details, 'r') as f:
                bias_details = json.load(f)
        if args.bias_report_path:
            if bias_details is None:
                bias_details = {}
            bias_details['gcs_path'] = args.bias_report_path
        
        send_training_completion_email(
            build_id=args.build_id,
            model_name=args.model_name or 'Unknown',
            dataset=args.dataset or 'unknown',
            test_accuracy=args.test_accuracy or 0.0,
            validation_passed=args.validation_passed.lower() == 'true' if args.validation_passed else True,
            bias_check_passed=args.bias_check_passed.lower() == 'true' if args.bias_check_passed else True,
            bias_details=bias_details
        )
    elif args.type == 'validation-failure':
        send_validation_failure_email(
            build_id=args.build_id,
            model_name=args.model_name or 'Unknown',
            dataset=args.dataset or 'unknown',
            test_accuracy=args.test_accuracy or 0.0,
            threshold=args.threshold or 0.5
        )
    elif args.type == 'bias-failure':
        bias_details = {}
        if args.bias_details:
            with open(args.bias_details, 'r') as f:
                bias_details = json.load(f)
        
        send_bias_failure_email(
            build_id=args.build_id,
            model_name=args.model_name or 'Unknown',
            dataset=args.dataset or 'unknown',
            bias_details=bias_details
        )
    elif args.type == 'pipeline-failure':
        send_pipeline_failure_email(
            build_id=args.build_id,
            failed_step=args.failed_step or 'Unknown',
            error_message=args.error_message or 'Unknown error'
        )


if __name__ == '__main__':
    main()

