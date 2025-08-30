"""
Inference and evaluation script for fine-tuned models
"""

import os
import torch
import argparse
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel
import pandas as pd
from typing import List, Dict, Any
from data.data_loader import DatasetLoader

class ModelInference:
    """Inference class for fine-tuned models"""
    
    def __init__(self, model_path: str, base_model_name: str = None):
        self.model_path = model_path
        self.base_model_name = base_model_name
        self.model = None
        self.tokenizer = None
        
        # Load training config if available
        config_path = os.path.join(model_path, "training_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.training_config = json.load(f)
                if not base_model_name:
                    self.base_model_name = self.training_config.get('model_name')
        else:
            self.training_config = {}
    
    def load_model(self):
        """Load the fine-tuned model"""
        print(f"Loading model from: {self.model_path}")
        
        if not self.base_model_name:
            raise ValueError("Base model name must be provided")
        
        # Quantization config for inference
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # Load LoRA weights
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        print("Model loaded successfully!")
    
    def generate_response(self, instruction: str, input_text: str = "", max_length: int = 512, temperature: float = 0.7):
        """Generate response for given instruction"""
        if self.model is None:
            self.load_model()
        
        # Format prompt
        if input_text.strip():
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part
        response = response[len(prompt):].strip()
        
        return response
    
    def batch_inference(self, instructions: List[str], inputs: List[str] = None, **kwargs) -> List[str]:
        """Run batch inference"""
        if inputs is None:
            inputs = [""] * len(instructions)
        
        responses = []
        for instruction, input_text in zip(instructions, inputs):
            response = self.generate_response(instruction, input_text, **kwargs)
            responses.append(response)
        
        return responses
    
    def evaluate_on_dataset(self, dataset_name: str = "all", num_samples: int = 50):
        """Evaluate model on test dataset"""
        print(f"Evaluating on {dataset_name} dataset...")
        
        # Load dataset
        data_loader = DatasetLoader()
        
        if dataset_name == "all":
            dataset = data_loader.load_all_datasets()
        elif dataset_name == "california":
            dataset = data_loader.load_california_housing()
        elif dataset_name == "adult":
            dataset = data_loader.load_adult_dataset()
        elif dataset_name == "insurance":
            dataset = data_loader.load_insurance_dataset()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Sample from test set
        test_data = dataset['test'].shuffle(seed=42).select(range(min(num_samples, len(dataset['test']))))
        
        results = []
        
        for i, example in enumerate(test_data):
            print(f"Processing sample {i+1}/{len(test_data)}")
            
            # Generate prediction
            prediction = self.generate_response(
                example['instruction'],
                example.get('input', ''),
                temperature=0.1  # Lower temperature for more consistent evaluation
            )
            
            result = {
                'instruction': example['instruction'],
                'input': example.get('input', ''),
                'expected_response': example['response'],
                'predicted_response': prediction,
                'sample_id': i
            }
            
            results.append(result)
        
        return results
    
    def save_evaluation_results(self, results: List[Dict], output_file: str):
        """Save evaluation results to file"""
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"Evaluation results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned model")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to fine-tuned model")
    parser.add_argument("--base_model", type=str,
                       help="Base model name (if not in config)")
    parser.add_argument("--mode", type=str, choices=["interactive", "evaluate", "single"],
                       default="interactive", help="Inference mode")
    parser.add_argument("--instruction", type=str,
                       help="Single instruction for single mode")
    parser.add_argument("--input", type=str, default="",
                       help="Input text for single mode")
    parser.add_argument("--dataset", type=str, 
                       choices=["california", "adult", "insurance", "all"],
                       default="all", help="Dataset for evaluation")
    parser.add_argument("--num_samples", type=int, default=50,
                       help="Number of samples for evaluation")
    parser.add_argument("--output_file", type=str, default="evaluation_results.csv",
                       help="Output file for evaluation results")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Generation temperature")
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = ModelInference(args.model_path, args.base_model)
    
    if args.mode == "single":
        if not args.instruction:
            print("Error: --instruction required for single mode")
            return
        
        response = inference.generate_response(
            args.instruction,
            args.input,
            temperature=args.temperature
        )
        print(f"\nInstruction: {args.instruction}")
        if args.input:
            print(f"Input: {args.input}")
        print(f"Response: {response}")
    
    elif args.mode == "evaluate":
        results = inference.evaluate_on_dataset(args.dataset, args.num_samples)
        inference.save_evaluation_results(results, args.output_file)
        
        # Print some sample results
        print("\n=== Sample Results ===")
        for i, result in enumerate(results[:3]):
            print(f"\nSample {i+1}:")
            print(f"Instruction: {result['instruction'][:100]}...")
            print(f"Expected: {result['expected_response'][:100]}...")
            print(f"Predicted: {result['predicted_response'][:100]}...")
    
    elif args.mode == "interactive":
        print("Interactive mode - Enter instructions (type 'quit' to exit)")
        
        while True:
            instruction = input("\nInstruction: ").strip()
            if instruction.lower() in ['quit', 'exit', 'q']:
                break
            
            input_text = input("Input (optional): ").strip()
            
            response = inference.generate_response(
                instruction,
                input_text,
                temperature=args.temperature
            )
            
            print(f"Response: {response}")

if __name__ == "__main__":
    main()
