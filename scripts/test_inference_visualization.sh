#!/bin/bash
# 推理可视化测试脚本示例
# 在运行前请确保：
# 1. 有可用的实验目录和模型权重
# 2. 已激活conda rec环境

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rec

# 示例1: 快速测试（只处理5个批次）
echo "🧪 测试1: 快速功能测试"
python scripts/inference_visualization.py \
    --experiment_dir exp/your_experiment_dir \
    --state_dict_path exp/your_experiment_dir/your_model.pth \
    --output_dir exp/test_inference_viz \
    --max_batches 5 \
    --batch_size 128 \
    --device cpu \
    --journal_style custom

# 示例2: 完整推理可视化（高质量）
echo "📊 测试2: 完整推理可视化"
python scripts/inference_visualization.py \
    --experiment_dir exp/your_experiment_dir \
    --state_dict_path exp/your_experiment_dir/your_model.pth \
    --dataset_type test \
    --output_dir exp/inference_visualization_full \
    --journal_style nature \
    --viz_dpi 300 \
    --viz_format pdf \
    --save_publication_figs true \
    --include_fourier true \
    --tsne_sample_size 2000 \
    --batch_size 256 \
    --device cuda

# 示例3: 验证集可视化
echo "🔍 测试3: 验证集可视化"
python scripts/inference_visualization.py \
    --experiment_dir exp/your_experiment_dir \
    --state_dict_path exp/your_experiment_dir/your_model.pth \
    --dataset_type valid \
    --output_dir exp/inference_viz_valid \
    --max_batches 10 \
    --journal_style high_quality

echo "✅ 测试完成！请检查输出目录中的可视化文件。"