"""
monitoring.py - Monitoring utilities with Email notifications
"""
import logging
from google.cloud import monitoring_v3
from google.cloud import pubsub_v1
import json
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmailNotifier:
    """Send email notifications via Gmail SMTP"""
    
    def __init__(
        self,
        smtp_user: str,
        smtp_password: str,
        smtp_host: str = "smtp.gmail.com",
        smtp_port: int = 587
    ):
        """
        Initialize email notifier.
        
        Args:
            smtp_user: Gmail address
            smtp_password: Gmail app password (not regular password)
            smtp_host: SMTP server host
            smtp_port: SMTP server port
        """
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
    
    def send_email(
        self,
        to_addresses: List[str],
        subject: str,
        body: str,
        html_body: Optional[str] = None
    ):
        """
        Send email notification.
        
        Args:
            to_addresses: List of recipient email addresses
            subject: Email subject
            body: Plain text body
            html_body: Optional HTML body
        """
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['From'] = f"MedScan AI <{self.smtp_user}>"
            msg['To'] = ', '.join(to_addresses)
            msg['Subject'] = subject
            
            # Add plain text part
            msg.attach(MIMEText(body, 'plain'))
            
            # Add HTML part if provided
            if html_body:
                msg.attach(MIMEText(html_body, 'html'))
            
            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            
            logger.info(f"Email sent to {len(to_addresses)} recipients")
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            raise


class ModelMonitor:
    """Monitor deployed models and send email alerts"""
    
    def __init__(
        self,
        project_id: str = "medscanai-476203",
        smtp_user: str = None,
        smtp_password: str = None
    ):
        self.project_id = project_id
        self.monitoring_client = monitoring_v3.MetricServiceClient()
        self.publisher = pubsub_v1.PublisherClient()
        self.project_name = f"projects/{project_id}"
        
        # Initialize email notifier if credentials provided
        if smtp_user and smtp_password:
            self.email_notifier = EmailNotifier(smtp_user, smtp_password)
        else:
            self.email_notifier = None
            logger.warning("Email notifier not initialized - no SMTP credentials")
    
    def create_custom_metric(
        self,
        metric_type: str,
        value: float,
        labels: dict
    ):
        """
        Write custom metric to Cloud Monitoring.
        
        Args:
            metric_type: Metric type (e.g., 'vision/model_accuracy')
            value: Metric value
            labels: Labels dict
        """
        series = monitoring_v3.TimeSeries()
        series.metric.type = f"custom.googleapis.com/{metric_type}"
        series.resource.type = "global"
        
        # Add labels
        for key, val in labels.items():
            series.metric.labels[key] = str(val)
        
        # Create data point
        now = datetime.utcnow()
        point = monitoring_v3.Point()
        point.value.double_value = value
        point.interval.end_time.seconds = int(now.timestamp())
        
        series.points = [point]
        
        # Write metric
        self.monitoring_client.create_time_series(
            name=self.project_name,
            time_series=[series]
        )
        
        logger.info(f"Metric written: {metric_type} = {value}")
    
    def send_alert_email(
        self,
        alert_type: str,
        message: str,
        severity: str = "WARNING",
        metadata: dict = None,
        recipients: List[str] = None
    ):
        """
        Send alert via email.
        
        Args:
            alert_type: Type of alert
            message: Alert message
            severity: Alert severity (INFO, WARNING, HIGH, CRITICAL)
            metadata: Additional metadata
            recipients: Email addresses (defaults to config)
        """
        if not self.email_notifier:
            logger.warning("Cannot send email - notifier not configured")
            return
        
        if recipients is None:
            recipients = [
                "harshitha8.shekar@gmail.com",
                "kothari.sau@northeastern.edu"
            ]
        
        # Format email subject based on severity
        severity_emoji = {
            "INFO": "",
            "WARNING": "",
            "HIGH": "",
            "CRITICAL": ""
        }
        
        emoji = severity_emoji.get(severity, "")
        subject = f"{emoji} MedScan AI Alert - {alert_type}"
        
        # Format plain text body
        body = f"""MedScan AI Model Alert

Alert Type: {alert_type}
Severity: {severity}
Timestamp: {datetime.utcnow().isoformat()}

Message:
{message}
"""
        
        if metadata:
            body += "\n\nAdditional Details:\n"
            for key, value in metadata.items():
                body += f"  - {key}: {value}\n"
        
        body += """
---
MedScan AI - Automated Monitoring System
Dashboard: https://console.cloud.google.com/monitoring?project=medscanai-476203
"""
        
        # Format HTML body
        html_body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; color: #333; }}
                .header {{ background-color: #f44336; color: white; padding: 20px; }}
                .header.warning {{ background-color: #ff9800; }}
                .header.info {{ background-color: #2196F3; }}
                .content {{ padding: 20px; }}
                .details {{ background-color: #f5f5f5; padding: 15px; margin: 15px 0; border-left: 4px solid #f44336; }}
                .metadata {{ background-color: #e3f2fd; padding: 15px; margin: 15px 0; }}
                .footer {{ background-color: #f5f5f5; padding: 15px; margin-top: 20px; text-align: center; }}
            </style>
        </head>
        <body>
            <div class="header {severity.lower()}">
                <h2>{emoji} MedScan AI Alert</h2>
                <h3>{alert_type}</h3>
            </div>
            <div class="content">
                <div class="details">
                    <p><strong>Severity:</strong> {severity}</p>
                    <p><strong>Timestamp:</strong> {datetime.utcnow().isoformat()}</p>
                    <p><strong>Message:</strong></p>
                    <p>{message}</p>
                </div>
"""
        
        if metadata:
            html_body += '<div class="metadata"><h4>Additional Details:</h4><ul>'
            for key, value in metadata.items():
                html_body += f'<li><strong>{key}:</strong> {value}</li>'
            html_body += '</ul></div>'
        
        html_body += """
                <p><a href="https://console.cloud.google.com/monitoring?project=medscanai-476203">View Monitoring Dashboard →</a></p>
            </div>
            <div class="footer">
                <p>MedScan AI - Automated Monitoring System</p>
            </div>
        </body>
        </html>
        """
        
        # Send email
        self.email_notifier.send_email(
            to_addresses=recipients,
            subject=subject,
            body=body,
            html_body=html_body
        )
        
        logger.info(f"Alert email sent: {alert_type}")
    
    def send_alert(
        self,
        topic_name: str,
        alert_type: str,
        message: str,
        severity: str = "WARNING",
        metadata: dict = None
    ):
        """
        Send alert via Pub/Sub (for internal triggers).
        
        Args:
            topic_name: Pub/Sub topic name
            alert_type: Type of alert
            message: Alert message
            severity: Alert severity
            metadata: Additional metadata
        """
        topic_path = self.publisher.topic_path(self.project_id, topic_name)
        
        alert_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "alert_type": alert_type,
            "severity": severity,
            "message": message,
            "metadata": metadata or {}
        }
        
        # Publish message
        data = json.dumps(alert_data).encode('utf-8')
        future = self.publisher.publish(topic_path, data)
        
        logger.info(f" Alert published to Pub/Sub: {alert_type}")
        return future.result()


def send_training_complete_email(
    model_name: str,
    dataset: str,
    accuracy: float,
    build_id: str,
    smtp_user: str,
    smtp_password: str,
    recipients: List[str] = None
):
    """
    Send training completion email.
    
    Args:
        model_name: Name of trained model
        dataset: Dataset name
        accuracy: Test accuracy
        build_id: Cloud Build ID
        smtp_user: SMTP username
        smtp_password: SMTP password
        recipients: Email addresses
    """
    if recipients is None:
        recipients = [
            "harshitha8.shekar@gmail.com",
            "kothari.sau@northeastern.edu"
        ]
    
    notifier = EmailNotifier(smtp_user, smtp_password)
    
    subject = f" MedScan AI - {model_name} Training Complete"
    
    body = f"""Model training completed successfully!

Model Details:
- Model: {model_name}
- Dataset: {dataset}
- Test Accuracy: {accuracy:.2%}
- Build ID: {build_id}

View Results:
https://console.cloud.google.com/cloud-build/builds/{build_id}?project=medscanai-476203

Next Steps:
- Review validation metrics
- Check bias analysis results
- Deploy to production if all checks pass

---
MedScan AI - Automated Training System
"""
    
    html_body = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            .header {{ background-color: #4CAF50; color: white; padding: 20px; }}
            .content {{ padding: 20px; }}
            .metrics {{ background-color: #e8f5e9; padding: 15px; margin: 15px 0; }}
            .footer {{ background-color: #f5f5f5; padding: 15px; text-align: center; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2> Model Training Complete</h2>
        </div>
        <div class="content">
            <h3>Model Details</h3>
            <div class="metrics">
                <p><strong>Model:</strong> {model_name}</p>
                <p><strong>Dataset:</strong> {dataset}</p>
                <p><strong>Test Accuracy:</strong> {accuracy:.2%}</p>
                <p><strong>Build ID:</strong> {build_id}</p>
            </div>
            
            <p><a href="https://console.cloud.google.com/cloud-build/builds/{build_id}?project=medscanai-476203">View Build Logs →</a></p>
            
            <h3>Next Steps</h3>
            <ul>
                <li>Review validation metrics</li>
                <li>Check bias analysis results</li>
                <li>Deploy to production if all checks pass</li>
            </ul>
        </div>
        <div class="footer">
            <p>MedScan AI - Automated Training System</p>
        </div>
    </body>
    </html>
    """
    
    notifier.send_email(recipients, subject, body, html_body)


def main():
    """Example usage"""
    import os
    
    # Get SMTP credentials from environment
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")
    
    if not smtp_user or not smtp_password:
        logger.error("SMTP credentials not set")
        return
    
    monitor = ModelMonitor(
        smtp_user=smtp_user,
        smtp_password=smtp_password
    )
    
    # Example: Track model accuracy
    monitor.create_custom_metric(
        metric_type="vision/model_accuracy",
        value=0.92,
        labels={
            "dataset": "tb",
            "model": "resnet50"
        }
    )
    
    # Example: Send alert email
    monitor.send_alert_email(
        alert_type="Accuracy Drop Detected",
        message="Model accuracy dropped below 70% threshold",
        severity="HIGH",
        metadata={
            "current_accuracy": "65%",
            "threshold": "70%",
            "model": "tb_resnet50"
        }
    )
    
    # Example: Send training complete email
    send_training_complete_email(
        model_name="CNN_ResNet50",
        dataset="tb",
        accuracy=0.89,
        build_id="abc123",
        smtp_user=smtp_user,
        smtp_password=smtp_password
    )


if __name__ == "__main__":
    main()