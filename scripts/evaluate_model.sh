#!/bin/bash

# Evaluation script for fine-tuned models
# Usage: ./evaluate_model.sh <model_path> <base_model_name> <dataset>

MODEL_PATH=${1:-"outputs/llama2_7b_qlora"}
BASE_MODEL=${2:-"meta-llama/Llama-2-7b-hf"}
DATASET=${3:-"all"}

echo "Evaluating model: $MODEL_PATH"
echo "Base model: $BASE_MODEL"
echo "Dataset: $DATASET"

python inference.py \
    --model_path "$MODEL_PATH" \
    --base_model "$BASE_MODEL" \
    --mode evaluate \
    --dataset "$DATASET" \
    --num_samples 100 \
    --output_file "evaluation_results_$(basename $MODEL_PATH).csv"

echo "Evaluation completed!"
