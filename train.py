"""
Main training script for fine-tuning LLMs with QLoRA on tabular datasets
"""

import os
import torch
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from datasets import DatasetDict
import wandb
from typing import Optional
import json
from huggingface_hub import login
from dotenv import load_dotenv

from config.model_configs import get_model_config, MODEL_CONFIGS
from data.data_loader import DatasetLoader, format_instruction_dataset

class QLoRATrainer:
    """QLoRA trainer for LLM fine-tuning"""
    
    def __init__(self, model_key: str, output_dir: str = "outputs", use_wandb: bool = False):
        self.model_key = model_key
        self.config = get_model_config(model_key)
        self.output_dir = output_dir
        self.use_wandb = use_wandb
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize wandb if requested
        if use_wandb:
            wandb.init(
                project="llm-tabular-finetuning",
                name=f"{model_key}-qlora",
                config=self.config.__dict__
            )
    
    def _setup_hf_authentication(self):
        """Setup Hugging Face authentication using .env file or environment variables"""
        self.hf_token = None
        
        # Load .env file if it exists
        load_dotenv()
        
        # Check if this model requires authentication
        if not self.config.requires_auth:
            print(f"✅ Model {self.config.model_name} does not require authentication")
            return
        
        print(f"🔐 Model {self.config.model_name} requires authentication")
        
        if self.config.is_gated:
            print("   📋 This is a gated model - you need to request access first")
            
        if self.config.license_info:
            print(f"   📄 {self.config.license_info}")
        
        # Try to get token from environment variable (loaded from .env or set directly)
        self.hf_token = os.environ.get('HF_TOKEN')
        
        if not self.hf_token:
            print("⚠️  HF_TOKEN not found in environment variables or .env file")
            print("   Please set your Hugging Face token:")
            print("   1. Create a .env file with: HF_TOKEN=hf_your_token_here")
            print("   2. Or set environment variable: export HF_TOKEN=hf_your_token_here")
            print("   3. Or login using: huggingface-cli login")
            
            # Try to use cached token from huggingface-cli login
            try:
                from huggingface_hub import HfFolder
                self.hf_token = HfFolder.get_token()
                if self.hf_token:
                    print("✅ Using cached Hugging Face token from CLI login")
                else:
                    print("❌ No cached token found. Please authenticate first.")
                    print(f"   Steps to fix:")
                    print(f"   1. Get access to {self.config.model_name} on Hugging Face")
                    print(f"   2. Get your token from https://huggingface.co/settings/tokens")
                    print(f"   3. Create .env file or set HF_TOKEN environment variable")
                    raise ValueError("Hugging Face authentication required")
            except Exception as e:
                print(f"❌ Authentication failed: {e}")
                raise
        else:
            print("✅ Using HF_TOKEN from environment (.env file or system variable)")
            
        # Login to Hugging Face Hub
        try:
            login(token=self.hf_token, add_to_git_credential=False)
            print("✅ Successfully authenticated with Hugging Face")
        except Exception as e:
            print(f"❌ Failed to authenticate with Hugging Face: {e}")
            print("   Make sure you have access to the gated model")
            raise
    
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer with QLoRA configuration"""
        print(f"Loading model: {self.config.model_name}")
        
        # Setup Hugging Face authentication
        self._setup_hf_authentication()
        
        # Quantization configuration
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.load_in_4bit,
            bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
        )
        
        # Load model with authentication token if required
        model_kwargs = {
            "quantization_config": bnb_config,
            "device_map": "auto",
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
        }
        
        if self.config.requires_auth and self.hf_token:
            model_kwargs["token"] = self.hf_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )
        
        # Load tokenizer with authentication token if required
        tokenizer_kwargs = {
            "trust_remote_code": True,
        }
        
        if self.config.requires_auth and self.hf_token:
            tokenizer_kwargs["token"] = self.hf_token
            
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            **tokenizer_kwargs
        )
        
        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # LoRA configuration
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        
        print(f"Model loaded with {self.model.num_parameters()} parameters")
        print(f"Trainable parameters: {self.model.num_parameters(only_trainable=True)}")
        
    def prepare_dataset(self, dataset: DatasetDict):
        """Prepare dataset for training"""
        print("Preparing dataset...")
        
        # Format for instruction tuning
        dataset = dataset.map(format_instruction_dataset)
        
        # Tokenize
        def tokenize_function(examples):
            # Tokenize the text
            tokenized = self.tokenizer(
                examples['text'],
                truncation=True,
                padding=False,
                max_length=self.config.max_length,
                return_tensors=None,
            )
            
            # Set labels for causal language modeling
            tokenized['labels'] = tokenized['input_ids'].copy()
            
            return tokenized
        
        # Apply tokenization
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset['train'].column_names,
            desc="Tokenizing dataset"
        )
        
        return tokenized_dataset
    
    def train(self, dataset: DatasetDict):
        """Train the model"""
        print("Starting training...")
        
        # Prepare dataset
        tokenized_dataset = self.prepare_dataset(dataset)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            max_steps=self.config.max_steps,
            learning_rate=self.config.learning_rate,
            fp16=self.config.fp16,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="wandb" if self.use_wandb else None,
            run_name=f"{self.model_key}-qlora" if self.use_wandb else None,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset['train'],
            eval_dataset=tokenized_dataset['test'],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Train
        trainer.train()
        
        # Save final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Save training config
        config_path = os.path.join(self.output_dir, "training_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        print(f"Training completed! Model saved to {self.output_dir}")
        
        return trainer

def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLM with QLoRA on tabular datasets")
    parser.add_argument("--model", type=str, required=True, 
                       choices=list(MODEL_CONFIGS.keys()),
                       help="Model to fine-tune")
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Output directory for saved model")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    parser.add_argument("--dataset", type=str,
                       choices=["california", "adult", "insurance"],
                       default="california",
                       help="Dataset to use for training (single dataset per model)")
    
    args = parser.parse_args()
    
    print(f"Starting fine-tuning with model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Available models: {list(MODEL_CONFIGS.keys())}")
    
    # Initialize trainer
    trainer = QLoRATrainer(
        model_key=args.model,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb
    )
    
    # Setup model and tokenizer
    trainer.setup_model_and_tokenizer()
    
    # Load specific dataset
    data_loader = DatasetLoader()
    dataset = data_loader.load_dataset_by_name(args.dataset)
    
    print(f"Dataset '{args.dataset}' loaded: {len(dataset['train'])} train samples, {len(dataset['test'])} test samples")
    
    # Train model
    trained_model = trainer.train(dataset)
    
    print("Fine-tuning completed successfully!")

if __name__ == "__main__":
    main()
