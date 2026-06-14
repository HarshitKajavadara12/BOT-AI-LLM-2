"""
QUANTUM-FORGE Pipeline Launcher
===============================
Launches the end-to-end pipeline defined in core/pipeline.py.
"""

import time
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.pipeline import QuantumForgePipeline

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
    )
    logger = logging.getLogger("PipelineLauncher")
    
    logger.info("Starting QUANTUM-FORGE Pipeline...")
    
    try:
        pipeline = QuantumForgePipeline()
        pipeline.start()
        
        logger.info("Pipeline is running. Press Ctrl+C to stop.")
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Stopping pipeline...")
        if 'pipeline' in locals():
            pipeline.stop()
        logger.info("Pipeline stopped.")
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        if 'pipeline' in locals():
            pipeline.stop()
        sys.exit(1)

if __name__ == "__main__":
    main()
