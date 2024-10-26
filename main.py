# main.py

import logging
import sys
import threading
import time
from pathlib import Path
from git import Repo
import multiprocessing

from dashboard.src.app import start_dashboard
from src.training.code_processor import process_code_data
from src.training.model_trainer import ModelTrainer
from src.training.training_callback import TrainingCallback

def setup_logging():
    """Configure logging for the application."""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler("logs/main.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def run_dashboard():
    """Run dashboard in a separate process."""
    start_dashboard(port=5002, debug=False)

def main():
    """Main orchestration function."""
    logger = setup_logging()
    logger.info("Starting application")
    
    # Start Dashboard in a separate process instead of thread
    dashboard_process = multiprocessing.Process(target=run_dashboard)
    dashboard_process.start()
    logger.info("Dashboard started at http://localhost:5002")

    try:
        dataset_path = "path/to/your/real/dataset"  # <-- We'll update this
        logger.info(f"Using dataset from: {dataset_path}")
        
        # Initialize model trainer with callback
        logger.info("Initializing model trainer")
        trainer = ModelTrainer(config_path="config/model_config.yaml")
        
        # Create training callback for dashboard updates
        training_callback = TrainingCallback(logger=logger)
        trainer.add_callback(training_callback)
        
        # Start training
        logger.info("Starting model training")
        trainer.train()
        
        # Keep the application running
        logger.info("Training complete. Press Ctrl+C to exit.")
        dashboard_process.join()
            
    except KeyboardInterrupt:
        logger.info("Shutting down application gracefully")
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
    finally:
        dashboard_process.terminate()
        sys.exit(0)

if __name__ == "__main__":
    multiprocessing.freeze_support()  # Required for Windows
    main()