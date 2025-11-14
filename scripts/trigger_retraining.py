"""
trigger_retraining.py - Trigger model retraining based on performance
"""
import logging
from google.cloud import monitoring_v3
from google.cloud import build_v1
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrainingTrigger:
    """Trigger retraining based on performance metrics"""
    
    def __init__(self, project_id: str = "medscanai-476203"):
        self.project_id = project_id
        self.monitoring_client = monitoring_v3.MetricServiceClient()
        self.build_client = build_v1.CloudBuildClient()
    
    def check_metric_threshold(
        self,
        metric_type: str,
        threshold: float,
        comparison: str = "LESS_THAN",
        lookback_hours: int = 24
    ) -> bool:
        """
        Check if metric breaches threshold.
        
        Returns:
            True if threshold breached
        """
        # Query metric
        now = datetime.utcnow()
        start_time = now - timedelta(hours=lookback_hours)
        
        interval = monitoring_v3.TimeInterval()
        interval.end_time.seconds = int(now.timestamp())
        interval.start_time.seconds = int(start_time.timestamp())
        
        results = self.monitoring_client.list_time_series(
            request={
                "name": f"projects/{self.project_id}",
                "filter": f'metric.type = "{metric_type}"',
                "interval": interval
            }
        )
        
        # Check if any values breach threshold
        for result in results:
            for point in result.points:
                value = point.value.double_value
                
                if comparison == "LESS_THAN" and value < threshold:
                    logger.warning(f"Metric {metric_type} = {value} < {threshold}")
                    return True
                elif comparison == "GREATER_THAN" and value > threshold:
                    logger.warning(f"Metric {metric_type} = {value} > {threshold}")
                    return True
        
        return False
    
    def trigger_cloud_build(
        self,
        config_file: str,
        substitutions: dict = None
    ):
        """
        Trigger Cloud Build for retraining.
        
        Args:
            config_file: Path to cloudbuild.yaml
            substitutions: Substitution variables
        """
        logger.info(f"Triggering Cloud Build: {config_file}")
        
        build = build_v1.Build()
        build.steps = []  # Will be loaded from config
        
        if substitutions:
            build.substitutions = substitutions
        
        operation = self.build_client.create_build(
            project_id=self.project_id,
            build=build
        )
        
        logger.info(f"Build triggered: {operation.metadata.build.id}")
        return operation
    
    def check_and_trigger_retraining(self):
        """Main function to check metrics and trigger retraining"""
        logger.info("Checking metrics for retraining triggers...")
        
        # Check Vision model accuracy
        vision_breach = self.check_metric_threshold(
            metric_type="custom.googleapis.com/vision/model_accuracy",
            threshold=0.70,
            comparison="LESS_THAN"
        )
        
        if vision_breach:
            logger.info("Vision model accuracy below threshold - triggering retraining")
            self.trigger_cloud_build(
                config_file="cloudbuild/vision-training.yaml",
                substitutions={"_TRIGGER_REASON": "accuracy_drop"}
            )
        
        # Check RAG model performance
        rag_breach = self.check_metric_threshold(
            metric_type="custom.googleapis.com/rag/composite_score",
            threshold=0.60,
            comparison="LESS_THAN"
        )
        
        if rag_breach:
            logger.info("RAG model performance degraded - triggering retraining")
            self.trigger_cloud_build(
                config_file="cloudbuild/rag-training.yaml",
                substitutions={"_TRIGGER_REASON": "performance_degradation"}
            )
        
        # Check bias metrics
        bias_breach = self.check_metric_threshold(
            metric_type="custom.googleapis.com/model/bias_disparity",
            threshold=0.10,
            comparison="GREATER_THAN"
        )
        
        if bias_breach:
            logger.warning("Bias detected - triggering retraining with bias mitigation")
            # Trigger both pipelines
            self.trigger_cloud_build(config_file="cloudbuild/vision-training.yaml")
            self.trigger_cloud_build(config_file="cloudbuild/rag-training.yaml")


def main():
    """Run retraining check"""
    trigger = RetrainingTrigger()
    trigger.check_and_trigger_retraining()


if __name__ == "__main__":
    main()