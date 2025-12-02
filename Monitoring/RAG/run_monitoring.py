"""
run_monitoring.py - Main monitoring script
"""
import argparse
import logging
from rag_monitor import RAGMonitor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-id', default='medscanai-476500')
    parser.add_argument('--bucket', default='medscan-pipeline-medscanai-476500')
    parser.add_argument('--hours', type=int, default=24)
    parser.add_argument('--trigger-retrain', action='store_true')
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("RAG MODEL MONITORING")
    logger.info("="*70)
    
    monitor = RAGMonitor(args.project_id, args.bucket)
    
    # Step 1: Collect logs
    logger.info("\nStep 1: Collecting prediction logs...")
    logs = monitor.collect_prediction_logs(hours=args.hours)
    
    if not logs:
        logger.warning("No logs found")
        return
    
    # Step 2: Calculate operational metrics
    logger.info("Step 2: Calculating performance metrics...")
    metrics = monitor.calculate_performance_metrics(logs)
    
    print("\n" + "="*70)
    print("📊 OPERATIONAL METRICS")
    print("="*70)
    print(f"Total Predictions:  {metrics['total_predictions']}")
    print(f"Error Rate:         {metrics['error_rate']:.2%}")
    print(f"Avg Latency:        {metrics['avg_latency']:.2f}s")
    print(f"P95 Latency:        {metrics['p95_latency']:.2f}s")
    print(f"Avg Relevance:      {metrics['avg_relevance']:.2f}")
    
    # Step 3: Get model quality metrics
    logger.info("\nStep 3: Checking model quality...")
    model_metrics = monitor.get_current_model_metrics()
    
    print("\n" + "="*70)
    print("🎯 MODEL QUALITY METRICS")
    print("="*70)
    print(f"Current Model:      {model_metrics['model_name']}")
    print(f"Semantic Score:     {model_metrics['semantic_score']:.2f}")
    print(f"Hallucination:      {model_metrics['hallucination_score']:.2f}")
    print(f"Retrieval Score:    {model_metrics['retrieval_score']:.2f}")
    
    # Step 4: Detect drift
    logger.info("\nStep 4: Detecting data drift...")
    drift_info = monitor.detect_data_drift(logs)
    
    print("\n" + "="*70)
    print("🔍 DATA DRIFT ANALYSIS")
    print("="*70)
    print(f"Has Drift:          {drift_info.get('has_drift', False)}")
    
    if drift_info.get('drift_details'):
        for feature, details in drift_info['drift_details'].items():
            if details.get('has_drift'):
                print(f"  ⚠️  {feature}: DRIFT DETECTED")
    
    # Step 5: Determine retraining strategy
    logger.info("\nStep 5: Determining retraining strategy...")
    decision = monitor.determine_retraining_strategy(
        metrics, drift_info, model_metrics
    )
    
    print("\n" + "="*70)
    print("⚙️  RETRAINING DECISION")
    print("="*70)
    print(f"Needs Retraining:   {decision['needs_retraining']}")
    print(f"Strategy:           {decision['strategy'].upper()}")
    
    if decision.get('blocked'):
        print(f"Blocked:            {decision['blocked']}")
    
    if decision.get('reasons'):
        print("\nReasons:")
        for reason in decision['reasons']:
            print(f"  • {reason}")
    
    # Explain strategy
    if decision['strategy'] == 'full':
        print("\n📋 FULL RETRAINING will:")
        print("  1. Re-chunk medical documents")
        print("  2. Re-generate embeddings")
        print("  3. Rebuild FAISS index")
        print("  4. Run HPO with new data")
        print("  5. Select best model")
        print("  6. Deploy to Vertex AI")
    elif decision['strategy'] == 'model_only':
        print("\nMODEL SELECTION will:")
        print("  1. Use EXISTING embeddings/FAISS")
        print("  2. Run HPO experiments")
        print("  3. Select best model")
        print("  4. Deploy to Vertex AI")
    
    # Step 6: Save report
    logger.info("\nStep 6: Saving monitoring report...")
    report = monitor.save_monitoring_report(metrics, drift_info, decision)
    
    # Step 7: Trigger retraining if needed
    if decision['needs_retraining']:
        if args.trigger_retrain:
            logger.info("\nStep 7: Triggering retraining...")
            reason = "; ".join(decision['reasons'])
            success = monitor.trigger_retraining(decision['strategy'], reason)
            
            if success:
                logger.info("Retraining pipeline triggered")
            else:
                logger.error("Failed to trigger retraining")
        else:
            logger.info("\nStep 7: Retraining recommended but not triggered")
            print("\nTo trigger retraining:")
            print(f"   python {__file__} --trigger-retrain")
    else:
        logger.info("\nStep 7: No retraining needed")
    
    logger.info("MONITORING COMPLETE")


if __name__ == '__main__':
    main()