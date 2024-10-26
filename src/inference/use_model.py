import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import logging
from pathlib import Path
from typing import Optional

class ModelInference:
    def __init__(self, model_path: Optional[str] = None):
        self.setup_logging()
        self.model_path = model_path or "models"
        self.model_name = "codellama/CodeLlama-7b-Python-hf"
        self.setup_model()
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("ModelInference")
    
    def check_device(self):
        """Determine the best available device"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def setup_model(self):
        """Load the base model and trained adapter"""
        try:
            self.logger.info("Loading tokenizer and model...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Load base model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=self.check_device(),
                low_cpu_mem_usage=True
            )
            
            # Load trained LoRA adapter if available
            adapter_path = Path(self.model_path)
            if adapter_path.exists():
                self.logger.info(f"Loading trained adapter from {adapter_path}")
                self.model = PeftModel.from_pretrained(
                    self.model,
                    adapter_path,
                    torch_dtype=torch.float16
                )
            else:
                self.logger.warning(f"No trained adapter found at {adapter_path}")
            
            self.model.eval()
            self.logger.info("Model setup complete")
            
        except Exception as e:
            self.logger.error("Error setting up model", exc_info=True)
            raise
    
    def generate_code(self, prompt: str, 
                     max_length: int = 2048,
                     temperature: float = 0.7,
                     top_p: float = 0.9,
                     top_k: int = 50) -> str:
        """Generate code based on prompt"""
        try:
            # Prepare input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True
                )
            
            # Decode and clean up output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return self.clean_generated_text(generated_text, prompt)
            
        except Exception as e:
            self.logger.error("Error during code generation", exc_info=True)
            raise
    
    def clean_generated_text(self, text: str, prompt: str) -> str:
        """Clean up generated text"""
        if text.startswith(prompt):
            text = text[len(prompt):]
        return text.strip()

def main():
    # Example usage
    try:
        inference = ModelInference()
        
        # Example prompts
        prompts = [
            "Write a TypeScript function that sorts an array of numbers:",
            "Create a React component that displays a loading spinner:",
        ]
        
        for prompt in prompts:
            print(f"\nPrompt: {prompt}\n")
            generated_code = inference.generate_code(prompt)
            print("Generated Code:")
            print(generated_code)
            print("-" * 80)
            
    except Exception as e:
        logging.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()