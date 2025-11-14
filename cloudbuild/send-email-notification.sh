#!/bin/bash
# send-email-notification.sh - Send email notification from Cloud Build
# Usage: ./send-email-notification.sh "subject" "body" "status"

SUBJECT="$1"
BODY="$2"
STATUS="$3"  # SUCCESS or FAILURE

# Email recipients
TO_EMAILS="harshitha8.shekar@gmail.com,kothari.sau@northeastern.edu"

# Get SMTP credentials from Secret Manager
SMTP_USER=$(gcloud secrets versions access latest --secret="smtp-username" --project=medscanai-476203 2>/dev/null || echo "")
SMTP_PASSWORD=$(gcloud secrets versions access latest --secret="smtp-password" --project=medscanai-476203 2>/dev/null || echo "")

if [ -z "$SMTP_USER" ] || [ -z "$SMTP_PASSWORD" ]; then
    echo " SMTP credentials not found in Secret Manager"
    echo "Skipping email notification"
    exit 0
fi

# Install mail utilities
pip install --quiet secure-smtplib 2>/dev/null || apt-get install -y mailutils 2>/dev/null || true

# Create email content
cat > /tmp/email.txt << EOF
From: MedScan AI <${SMTP_USER}>
To: ${TO_EMAILS}
Subject: ${SUBJECT}
Content-Type: text/plain; charset=UTF-8

${BODY}

---
MedScan AI - Automated Training System
Build ID: ${BUILD_ID}
Project: medscanai-476203
EOF

# Send email using Python (more reliable than mailutils)
python3 << PYTHON_EOF
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

smtp_user = "${SMTP_USER}"
smtp_password = "${SMTP_PASSWORD}"
to_emails = "${TO_EMAILS}".split(',')

msg = MIMEMultipart()
msg['From'] = f"MedScan AI <{smtp_user}>"
msg['To'] = ', '.join(to_emails)
msg['Subject'] = "${SUBJECT}"

body = """${BODY}

---
MedScan AI - Automated Training System
Build ID: ${BUILD_ID}
Project: medscanai-476203
"""

msg.attach(MIMEText(body, 'plain'))

try:
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)
    print("Email sent successfully")
except Exception as e:
    print(f"Failed to send email: {e}")
    exit(1)
PYTHON_EOF

echo "Email notification sent to ${TO_EMAILS}"