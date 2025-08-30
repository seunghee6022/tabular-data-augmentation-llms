#!/bin/bash

# Script to compare top HuggingFace leaderboard models for tabular data generation
# Make sure you have trained the models first using the training scripts

echo "=== Comparing Top LLM Models for Tabular Data Generation ==="

# Define the top models to compare (adjust based on your trained models)
MODELS=(
    "llama3_1_8b"
    "qwen2_5_7b" 
    "mistral_nemo_12b"
    "gemma2_9b"
    "llama2_7b"
    "mistral_7b"
)

# Check which models are available
echo "Checking available trained models..."
AVAILABLE_MODELS=()

for model in "${MODELS[@]}"; do
    if [ -d "outputs/${model}_qlora" ]; then
        echo "✅ Found: $model"
        AVAILABLE_MODELS+=("$model")
    else
        echo "❌ Missing: $model (run training first)"
    fi
done

if [ ${#AVAILABLE_MODELS[@]} -eq 0 ]; then
    echo "No trained models found. Please train models first using:"
    echo "  python train.py --model <model_name> --output_dir outputs/<model_name>_qlora"
    exit 1
fi

echo ""
echo "Comparing ${#AVAILABLE_MODELS[@]} available models: ${AVAILABLE_MODELS[*]}"
echo ""

# Run the comparison
python compare_models.py \
    --models "${AVAILABLE_MODELS[@]}" \
    --datasets california adult insurance \
    --num_samples 300 \
    --output_dir model_comparison_results

echo ""
echo "=== Comparison Complete ==="
echo "Results saved to: model_comparison_results/"
echo ""
echo "Key files generated:"
echo "  - comparison_results.json (detailed results)"
echo "  - model_comparison_summary.csv (summary table)"
echo "  - overall_scores.png (performance chart)"
echo "  - quality_heatmap.png (quality by dataset)"
echo "  - speed_vs_quality.png (efficiency analysis)"
