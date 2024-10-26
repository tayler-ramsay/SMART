# cli.py

import argparse
import logging
import sys
import threading
import subprocess
import time
from pathlib import Path

from src.training.code_processor import process_code_data
from src.training.model_trainer import ModelTrainer
from dashboard.src.app import start_dashboard

def setup_logging():
    """Configure logging for the CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/cli.log")
        ]
    )

def clone_repository(git_url: str, destination: str = "data"):
    """Clone the Git repository to the specified destination."""
    try:
        logging.info(f"Cloning repository from {git_url} to {destination}...")
        subprocess.run(['git', 'clone', git_url, destination], check=True)
        logging.info("Repository cloned successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to clone repository: {e}")
        sys.exit(1)

def run_dashboard():
    """Start the Flask dashboard."""
    logging.info("Starting the dashboard...")
    start_dashboard(port=5002)

def run_training(model_name: str, config_path: str = "config/model_config.yaml"):
    """Start the model training process."""
    logging.info(f"Starting model training for model: {model_name}")
    trainer = ModelTrainer(config_path=config_path)
    trainer.train()

def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description="CLI to start the LLM fine-tuning process with dashboard monitoring."
    )
    parser.add_argument(
        '--git_url',
        type=str,
        help='Git repository URL of the codebase to fine-tune the LLM on.',
        required=True
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-neo',
        help='Name of the model to fine-tune. Default is "gpt-neo".'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/model_config.yaml',
        help='Path to the model configuration YAML file.'
    )

    args = parser.parse_args()

    # Step 1: Clone the Git repository
    clone_repository(args.git_url)

    # Step 2: Process the code data
    logging.info("Starting data processing...")
    process_code_data(input_dir="data", output_dir="processed_code")
    logging.info("Data processing completed.")

    # Step 3: Start the dashboard in a separate thread
    dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
    dashboard_thread.start()
    logging.info("Dashboard is running on http://localhost:5002")

    # Give the dashboard some time to start
    time.sleep(5)

    # Step 4: Start model training
    run_training(model_name=args.model, config_path=args.config)

    # Keep the CLI running to maintain dashboard availability
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Shutting down the CLI.")
        sys.exit(0)

if __name__ == "__main__":
    main()