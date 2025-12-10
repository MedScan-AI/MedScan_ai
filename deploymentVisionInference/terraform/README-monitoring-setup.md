# Vision Inference Monitoring Setup Guide

## Why You're Not Receiving Email Alerts

If alert policies are triggering but you're not receiving emails, follow these steps:

## Step 1: Verify Email Notification Channel

Email notification channels in Google Cloud Monitoring require verification before they can send emails.

### Check Notification Channel Status

1. Go to [Google Cloud Console - Notification Channels](https://console.cloud.google.com/monitoring/alerting/notifications?project=medscanai-476500)

2. Look for "Vision Inference API - Email Alerts"

3. Check the status:
   - **Unverified** ❌ - Email won't be sent
   - **Verified** ✅ - Email should be sent

### Verify Email Address

If the channel is unverified:

1. Click on the notification channel
2. You should see a "Verify" button or link
3. Google will send a verification email to `sriharsha.py@gmail.com`
4. Click the verification link in that email
5. Return to the Cloud Console and confirm it's now verified

## Step 2: Check Alert Policy Notification Channels

Verify that alert policies are linked to the notification channel:

1. Go to [Alert Policies](https://console.cloud.google.com/monitoring/alerting/policies?project=medscanai-476500)

2. Click on "Vision Inference API - Low Confidence Streak" (or any other policy)

3. Under "Notifications", you should see:
   - **Notification channels**: Vision Inference API - Email Alerts (sriharsha.py@gmail.com)

4. If no notification channel is listed:
   - Click "Edit"
   - Under "Notifications", click "Add Notification Channel"
   - Select "Vision Inference API - Email Alerts"
   - Save

## Step 3: Enable the Low Confidence Metric

The metric has now been enabled by default (`create_low_conf_metric = true`).

To apply this change:

```bash
cd deploymentVisionInference/terraform
terraform plan   # Review changes
terraform apply  # Apply changes
```

This will create:
- ✅ Log-based metric: `low_confidence_predictions`
- ✅ Alert policy: "Vision Inference API - Low Confidence Streak"

**Note**: After creating the metric, wait ~10 minutes before the alert policy can use it.

## Step 4: Test the Alert

### Option A: Generate Low Confidence Predictions

Make predictions that will return low confidence scores (< threshold).

### Option B: Test Notification Channel Directly

1. Go to [Notification Channels](https://console.cloud.google.com/monitoring/alerting/notifications?project=medscanai-476500)
2. Click on "Vision Inference API - Email Alerts"
3. Click "Send Test Notification"
4. Check your email at `sriharsha.py@gmail.com`

## Step 5: Check Gmail Settings

If you still don't receive emails after verification:

1. **Check Spam/Junk folder** - Google Cloud alerts might be filtered
2. **Add to Safe Senders**: Add `noreply@google.com` and `monitoring-noreply@google.com` to your contacts
3. **Check Gmail filters** - Ensure no filters are blocking Google Cloud emails

## Alternative: Daily Monitoring Workflow

Even if the real-time alert policy doesn't send emails, you have a backup:

The workflow `.github/workflows/vision-inference-retrain-decoy.yaml`:
- ✅ Runs every 24 hours automatically
- ✅ Checks low confidence predictions via direct log query
- ✅ Sends email when threshold exceeded
- ✅ Works independently of Cloud Monitoring alert policies

This workflow uses GitHub Actions email notifications, which are more reliable.

## Troubleshooting

### Check if Metric is Receiving Data

```bash
# List all user-defined metrics
gcloud logging metrics list --project=medscanai-476500

# Check if low_confidence_predictions metric exists
gcloud logging metrics describe low_confidence_predictions --project=medscanai-476500

# View recent log entries that should match the metric
gcloud logging read 'resource.type="cloud_run_revision" 
  resource.labels.service_name="vision-inference-api" 
  textPayload:"[low_confidence]"' \
  --project=medscanai-476500 \
  --limit=10
```

### Check Alert Policy Status

```bash
# List all alert policies
gcloud alpha monitoring policies list --project=medscanai-476500

# Get details of specific policy
gcloud alpha monitoring policies describe POLICY_ID --project=medscanai-476500
```

### Manual Notification Channel Verification via CLI

```bash
# List notification channels
gcloud alpha monitoring channels list --project=medscanai-476500

# Describe specific channel
gcloud alpha monitoring channels describe CHANNEL_ID --project=medscanai-476500
```

## Summary Checklist

- [ ] Email notification channel verified in Cloud Console
- [ ] Alert policies linked to notification channel
- [ ] `create_low_conf_metric = true` in terraform
- [ ] Terraform applied successfully
- [ ] Waited 10+ minutes after metric creation
- [ ] Test notification sent successfully
- [ ] Checked spam/junk folder
- [ ] Daily workflow enabled as backup

## Need Help?

If issues persist:

1. Check Cloud Logging for the actual log entries:
   https://console.cloud.google.com/logs/query;query=resource.type%3D%22cloud_run_revision%22%0Aresource.labels.service_name%3D%22vision-inference-api%22%0AtextPayload%3A%22%5Blow_confidence%5D%22?project=medscanai-476500

2. Check monitoring metrics:
   https://console.cloud.google.com/monitoring/metrics-explorer?project=medscanai-476500

3. Review alert policy history:
   https://console.cloud.google.com/monitoring/alerting/incidents?project=medscanai-476500

