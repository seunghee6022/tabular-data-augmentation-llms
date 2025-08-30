#!/bin/bash

# Training script for Mistral 7B model on individual datasets
# Make sure you have the required dependencies installed

echo "Starting Mistral 7B fine-tuning with QLoRA..."

# Train separate models for each dataset
DATASETS=("california" "adult" "insurance")

for dataset in "${DATASETS[@]}"; do
    echo "Training on $dataset dataset..."
    
    python train.py \
        --model mistral_7b \
        --dataset $dataset \
        --output_dir outputs/mistral_7b_${dataset}_qlora \
        --use_wandb
    
    echo "Completed training on $dataset dataset"
done

echo "All training completed!"
