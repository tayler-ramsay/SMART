import torch
import logging
import time
import psutil
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk, DatasetDict
import logging
from pathlib import Path
import yaml
from datetime import datetime
from typing import Dict, Any
import gc
from transformers import AutoTokenizer
from datasets import load_dataset

class ModelTrainer:
    def __init__(self, config_path: str = "config/model_config.yaml"):
        self.logger = self._setup_logging()
        self.config = self._load_config(config_path)
        self.cache_dir = Path("model_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.output_dir = Path("models") / f"{self.model_name.split('/')[-1]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device_info = self.check_device()
        self.device = self.device_info["device_type"]
        self.dataset = self._load_dataset()  # Load and split the dataset here
        self.setup_model()
    # Load your dataset, replace "my_dataset" with your actual dataset or file path
    # raw_datasets = load_dataset("my_dataset")

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Python-hf")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def log_model_info(self):
        # Log basic model information for debugging and verification
        self.logger.info(f"Model type: {self.model_type}")
        self.logger.info(f"Model name: {self.model_name}")
        self.logger.info(f"Model structure:\n{self.model}")  # This will print the model architecture
    def _load_dataset(self):
        """Load the processed dataset and create a train-validation split."""
        try:
            # Load dataset from disk
            dataset = load_from_disk("processed_code")  # Adjust this path if needed

            # Tokenize the dataset
            def tokenize_function(examples):
                return self.tokenizer(examples['text'], truncation=True, padding=True, return_tensors="pt")

            tokenized_datasets = dataset.map(tokenize_function, batched=True)

            # Check if splits already exist
            if isinstance(tokenized_datasets, DatasetDict) and 'train' in tokenized_datasets and 'validation' in tokenized_datasets:
                self.logger.info("Dataset with splits loaded successfully.")
                return tokenized_datasets
            else:
                # Perform an 80/20 train-validation split if no splits are found
                split_dataset = tokenized_datasets.train_test_split(test_size=0.2)
                self.logger.info("Dataset split into training and validation sets.")
                return DatasetDict({
                    'train': split_dataset['train'],
                    'validation': split_dataset['test']
                })
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            raise
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load model configuration"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Retrieve the selected model type from default_model
        model_type = config['model_config']['default_model']
        self.model_type = model_type
        self.model_name = config['model_config'][model_type]['name']
        self.trust_remote_code = config['model_config'][model_type].get('trust_remote_code', False)
        
        
        # Store training and LoRA configs
        self.training_config = config['training_config']
        self.lora_config = config['lora_config']
     
        
        return config
    
    def _setup_logging(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Create logs directory with date subdirectory
        date_str = datetime.now().strftime('%Y%m%d')
        log_dir = Path("logs") / date_str
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Define log file name with timestamp
        log_file_name = log_dir / f"training_{datetime.now().strftime('%H%M%S')}.log"
        
        # File handler
        file_handler = logging.FileHandler(log_file_name)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def setup_model(self):
        self.logger.info(f"Setting up {self.model_type} model: {self.model_name}")
              # Initialize self.model here
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=str(self.cache_dir),
            trust_remote_code=self.trust_remote_code
        )
           # Print out all module names within the model
        for name, module in self.model.named_modules():
            print(f"{name}: {module}")  # Adjusted to show 'name' and 'module' separately

        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=str(self.cache_dir),
            trust_remote_code=self.trust_remote_code
        )
        
        model_kwargs = {
            "cache_dir": str(self.cache_dir),
            "trust_remote_code": self.trust_remote_code
        }
        
        if self.device == "mps":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                **model_kwargs
            ).to("mps")
        else:
            model_kwargs["device_map"] = "auto"
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
        
        if self.model_type in self.lora_config['target_modules']:
            self.setup_lora()
        else:
            self.logger.info(f"LoRA not configured for {self.model_type}")
        
        self.log_model_info()
    
    def setup_lora(self):
        self.logger.info("Applying LoRA configuration...")
        
        lora_config = LoraConfig(
            r=self.lora_config['r'],
            lora_alpha = self.lora_config['lora_alpha'],
            target_modules=self.lora_config['target_modules'][self.model_type],
            lora_dropout=self.lora_config['dropout'],
            bias="none",
            task_type="CAUSAL_LM"
            
        )
        self.model = get_peft_model(self.model, lora_config)
    
    def check_device(self):
        if torch.backends.mps.is_available():
            return {"device_type": "mps", "description": "Apple Silicon GPU"}
        elif torch.cuda.is_available():
            return {
                "device_type": "cuda",
                "description": torch.cuda.get_device_name(0)
            }
        return {"device_type": "cpu", "description": "CPU Only"}
    
    def train(self):
        try:
            dataset = load_from_disk("processed_code")
            self.logger.info(f"Dataset loaded with {len(dataset)} examples")

            if isinstance(dataset, DatasetDict):
                train_dataset = dataset['train']
                eval_dataset = dataset['validation']
            else:
                split = dataset.train_test_split(test_size=0.2)
                train_dataset = split['train']
                eval_dataset = split['test']
            
            training_args = TrainingArguments(
                output_dir=str(self.output_dir),
                num_train_epochs=self.training_config['num_train_epochs'],
                per_device_train_batch_size=self.training_config['batch_size'],
                gradient_accumulation_steps=self.training_config['gradient_accumulation_steps'],
                learning_rate=float(self.training_config['learning_rate']),
                logging_steps=10,
                save_steps=100,
                remove_unused_columns=False,
                use_mps_device=(self.device == "mps"),
                optim="adamw_torch"
            )
            
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=DataCollatorForLanguageModeling(
                    tokenizer=self.tokenizer,
                    mlm=False
                )
            )
            
            self.logger.info("Starting training...")
            logging.info(f"Training dataset columns: {train_dataset.column_names}")
            logging.info(f"Model's forward method signature: {self.model.forward.__annotations__}")
            trainer.train()
            
            final_model_path = self.output_dir / "final_model"
            trainer.save_model(final_model_path)
            self.logger.info(f"Model saved to {final_model_path}")
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            self.logger.error("Error during training", exc_info=True)
            raise
    def log_memory_usage():
        process = psutil.Process()
        memory_info = process.memory_info()
        logging.info(f"Memory usage: RSS = {memory_info.rss / (1024 ** 2):.2f} MB")

    def load_shard(shard):
        start_time = time.time()
        logging.info(f"Loading shard: {shard}")
        # Include the actual shard loading code here.
        end_time = time.time()
        logging.info(f"Loaded shard {shard} in {end_time - start_time:.2f} seconds")
    log_memory_usage()
def main():
    trainer = ModelTrainer()
    trainer.train()

if __name__ == "__main__":
    main()
