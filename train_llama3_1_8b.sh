#!/bin/bash

# Training script for llama3_1_8b model on individual datasets
# Make sure you have the required dependencies installed

echo "Starting llama3_1_8b fine-tuning with QLoRA..."

# Train separate models for each dataset
DATASETS=("california" "adult" "insurance")

for dataset in "${DATASETS[@]}"; do
    echo "Training on $dataset dataset..."
    
    python train.py \
        --model llama3_1_8b \
        --dataset $dataset \
        --output_dir outputs/llama3_1_8b_${dataset}_qlora \
        --use_wandb
    
    echo "Completed training on $dataset dataset"
done

echo "All training completed!"
