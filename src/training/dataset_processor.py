import json
from pathlib import Path
import logging
from datasets import Dataset, DatasetDict
from typing import List, Dict, Any

class DatasetProcessor:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.logger = logging.getLogger(__name__)
    
    def load_data(self, filename: str = "collected_code.json") -> List[Dict[str, Any]]:
        """Load collected code data"""
        input_file = self.input_path / filename
        self.logger.info(f"Loading data from {input_file}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def process(self):
        """Process collected code into a training dataset with splits."""
        try:
            # Load collected data
            data = self.load_data()
            self.logger.info(f"Loaded {len(data)} files")
            
            # Create the base dataset
            dataset = Dataset.from_dict({
                'text': [item['content'] for item in data],
                'language': [item['language'] for item in data],
                'path': [item['path'] for item in data]
            })
            
            # Create train-validation split (80% train, 20% validation)
            train_test_split = dataset.train_test_split(test_size=0.2)
            split_dataset = DatasetDict({
                'train': train_test_split['train'],
                'validation': train_test_split['test']
            })
            
            # Save processed dataset
            self.output_path.mkdir(exist_ok=True, parents=True)
            split_dataset.save_to_disk(str(self.output_path))
            self.logger.info(f"Saved dataset with splits to {self.output_path}")
        
        except Exception as e:
            self.logger.error("Dataset processing failed", exc_info=True)
            raise

