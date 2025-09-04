#!/bin/bash

# Script to train all top HuggingFace leaderboard models for comparison
# This will take significant time and GPU resources

echo "=== Training Top LLM Models for Tabular Data Generation ==="

# Define models to train (adjust based on your GPU memory)
declare -A MODELS=(
    ["llama3_1_8b"]="meta-llama/Meta-Llama-3.1-8B"
    ["qwen2_5_7b"]="Qwen/Qwen2.5-7B"
    ["mistral_nemo_12b"]="mistralai/Mistral-Nemo-Base-2407"
    ["gemma2_9b"]="google/gemma-2-9b"
    ["phi3_medium_14b"]="microsoft/Phi-3-medium-4k-instruct"
)

# Check GPU memory and recommend models
echo "Checking system resources..."
if command -v nvidia-smi &> /dev/null; then
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    echo "GPU Memory: ${GPU_MEMORY}MB"
    
    if [ "$GPU_MEMORY" -lt 12000 ]; then
        echo "⚠️  Warning: Less than 12GB GPU memory detected."
        echo "   Consider training smaller models first (7B-9B parameters)"
    fi
else
    echo "⚠️  nvidia-smi not found. Make sure you have CUDA-capable GPU."
fi

echo ""
echo "Models to train:"
for model in "${!MODELS[@]}"; do
    echo "  - $model (${MODELS[$model]})"
done

echo ""
read -p "Continue with training? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

# Train each model
for model in "${!MODELS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Training: $model"
    echo "Base model: ${MODELS[$model]}"
    echo "=========================================="
    
    # Check if already trained
    if [ -d "outputs/${model}_qlora" ]; then
        echo "⚠️  Model already exists: outputs/${model}_qlora"
        read -p "Overwrite? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Skipping $model"
            continue
        fi
    fi
    
    # Start training
    START_TIME=$(date +%s)
    
    # Train on each dataset separately
    DATASETS=("california" "adult" "insurance")
    
    for dataset in "${DATASETS[@]}"; do
        echo "Training $model on $dataset dataset..."
        
        python train.py \
            --model "$model" \
            --dataset "$dataset" \
            --output_dir "outputs/${model}_${dataset}_qlora" \
            --use_wandb
        
        if [ $? -eq 0 ]; then
            echo "✅ Successfully trained $model on $dataset"
        else
            echo "❌ Failed to train $model on $dataset"
        fi
    done
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    if [ $? -eq 0 ]; then
        echo "✅ Successfully trained $model in ${DURATION}s"
    else
        echo "❌ Failed to train $model"
    fi
    
    echo ""
done

echo ""
echo "=== Training Complete ==="
echo "Trained models are available in outputs/ directory"
echo ""
echo "Next steps:"
echo "1. Run model comparison:"
echo "   bash scripts/compare_top_models.sh"
echo ""
echo "2. Generate synthetic data:"
echo "   python generate_synthetic_data.py --model_path outputs/<model>_qlora --dataset <dataset>"
