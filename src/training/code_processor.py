# src/training/code_processor.py

import os
from pathlib import Path
from datasets import Dataset
import logging

def collect_code_files(directory):
    """Collect all code files from directory"""
    code_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.ts', '.js', '.tsx', '.jsx')):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        code_files.append({
                            'content': content,
                            'path': file_path,
                            'language': 'typescript' if file.endswith(('.ts', '.tsx')) else 'javascript'
                        })
                except Exception as e:
                    logging.error(f"Error reading {file_path}: {str(e)}")
    return code_files

def process_code_data(input_dir="data", output_dir="processed_code"):
    """Process code files into a dataset"""
    logging.info("Processing code data...")
    
    # Ensure input directory exists
    if not os.path.exists(input_dir):
        logging.warning(f"Input directory {input_dir} not found, using current directory")
        input_dir = "."
    
    # Collect code files
    code_files = collect_code_files(input_dir)
    logging.info(f"Collected {len(code_files)} code files")
    
    if not code_files:
        raise ValueError("No code files found to process")
    
    # Create dataset
    dataset = Dataset.from_dict({
        'text': [file['content'] for file in code_files],
        'path': [file['path'] for file in code_files],
        'language': [file['language'] for file in code_files]
    })
    
    # Save dataset
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    dataset.save_to_disk(str(output_path))
    logging.info(f"Dataset saved to {output_path}")
    
    return dataset