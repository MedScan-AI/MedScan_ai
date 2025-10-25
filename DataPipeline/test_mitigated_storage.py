#!/usr/bin/env python3
"""
Test script to verify that mitigated data storage works with disease-specific structure.
"""

import os
import sys
import logging
from datetime import datetime

# Add the scripts directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.data_preprocessing.schema_statistics import SchemaStatisticsManager

def test_mitigated_storage():
    """Test that mitigated data storage creates disease-specific directory structure."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize the manager
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'metadata.yml')
        manager = SchemaStatisticsManager(config_path)
        
        logger.info("✓ SchemaStatisticsManager initialized successfully")
        
        # Test the new mitigated data partition path method
        base_dir = "data/synthetic_metadata_mitigated"
        dataset_key = "tb"
        timestamp = "2025-10-24T00:00:00"
        
        # Test TB dataset
        tb_path = manager._get_mitigated_data_partition_path(base_dir, dataset_key, timestamp)
        expected_tb_path = os.path.join(base_dir, "tb", "2025", "10", "24")
        
        logger.info(f"✓ TB mitigated path: {tb_path}")
        logger.info(f"✓ Expected TB path: {expected_tb_path}")
        
        if tb_path == expected_tb_path:
            logger.info("✓ TB mitigated path structure is correct")
        else:
            logger.error("✗ TB mitigated path structure is incorrect")
            return False
        
        # Test lung cancer dataset
        dataset_key = "lung_cancer"
        lc_path = manager._get_mitigated_data_partition_path(base_dir, dataset_key, timestamp)
        expected_lc_path = os.path.join(base_dir, "lung_cancer", "2025", "10", "24")
        
        logger.info(f"✓ Lung cancer mitigated path: {lc_path}")
        logger.info(f"✓ Expected lung cancer path: {expected_lc_path}")
        
        if lc_path == expected_lc_path:
            logger.info("✓ Lung cancer mitigated path structure is correct")
        else:
            logger.error("✗ Lung cancer mitigated path structure is incorrect")
            return False
        
        # Test without timestamp (should use current date)
        current_tb_path = manager._get_mitigated_data_partition_path(base_dir, "tb")
        logger.info(f"✓ Current timestamp TB path: {current_tb_path}")
        
        logger.info("✓ All mitigated storage tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mitigated_storage()
    sys.exit(0 if success else 1)
