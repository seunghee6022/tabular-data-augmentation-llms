#!/bin/bash

# Training script for text datasets
# This demonstrates training on rich text data that leverages LLM capabilities

echo "🚀 Training LLMs on Text Datasets"
echo "================================="

# Configuration
MODEL="qwen2_5_7b"  # Top performing model
TEXT_DATASETS=("imdb_reviews" "amazon_reviews" "news_articles" "customer_support" "job_postings")

echo "📝 Text datasets provide richer context for LLM training:"
echo "  ✅ Natural language patterns"
echo "  ✅ Semantic understanding"
echo "  ✅ Creative generation capabilities"
echo "  ✅ Better generalization"
echo ""

# Train on each text dataset
for dataset in "${TEXT_DATASETS[@]}"; do
    echo "🎯 Training on $dataset dataset..."
    
    output_dir="outputs/${MODEL}_${dataset}_qlora"
    
    python train.py \
        --model $MODEL \
        --dataset $dataset \
        --output_dir $output_dir \
        --use_wandb
    
    if [ $? -eq 0 ]; then
        echo "✅ Successfully trained model on $dataset"
        echo "   Model saved to: $output_dir"
    else
        echo "❌ Failed to train model on $dataset"
    fi
    echo ""
done

echo "🎉 Text dataset training completed!"
echo ""
echo "📊 Next steps:"
echo "  1. Generate synthetic text data:"
echo "     python generate_synthetic_data.py --model_path outputs/${MODEL}_imdb_reviews_qlora --dataset imdb_reviews"
echo ""
echo "  2. Compare text vs tabular generation:"
echo "     python examples/text_vs_tabular_comparison.py"
echo ""
echo "  3. Evaluate generation quality:"
echo "     python compare_models.py --models $MODEL --datasets imdb_reviews amazon_reviews"
