# 推理时可视化脚本使用说明

## 功能说明

`scripts/inference_visualization.py` 是一个专门用于在推理时生成专家热力图和t-SNE图的脚本，与main.py训练时的可视化完全一致。

## 主要功能

1. **专家路由热力图** - 显示不同领域对专家的使用模式
2. **t-SNE专家专业化图** - 展示专家嵌入空间的专业化分布  
3. **Fourier频率分析**（可选） - 多领域频率模式比较
4. **期刊级别图表质量** - 支持Nature、Science等期刊样式

## 使用方法

### 基本用法
```bash
python scripts/inference_visualization.py \
    --experiment_dir exp/your_experiment \
    --state_dict_path exp/your_experiment/model.pth \
    --output_dir exp/inference_viz
```

### 完整参数示例
```bash
python scripts/inference_visualization.py \
    --experiment_dir exp/beauty_games_ml-1m_test \
    --state_dict_path exp/beauty_games_ml-1m_test/SASRec.epoch=50.lr=0.001.layer=2.head=2.hidden=64.maxlen=100.pth \
    --dataset_type test \
    --output_dir exp/inference_visualization \
    --journal_style nature \
    --viz_dpi 300 \
    --viz_format pdf \
    --save_publication_figs true \
    --batch_size 256 \
    --max_batches 20 \
    --include_fourier true \
    --seed 42 \
    --device cuda
```

## 参数说明

### 必需参数
- `--experiment_dir`: 训练实验目录路径
- `--state_dict_path`: 训练好的模型权重文件路径

### 数据参数
- `--dataset_type`: 推理数据集类型 (valid/test，默认test)
- `--use_datasets`: 使用的数据集列表（默认从实验args.txt读取）
- `--batch_size`: 推理批次大小（默认256）
- `--max_batches`: 最大处理批次数（用于快速测试，默认无限制）

### 可视化参数
- `--journal_style`: 期刊样式 (nature/science/cell/high_quality/custom，默认custom)
- `--viz_dpi`: 图片DPI（默认300）
- `--viz_format`: 图片格式 (pdf/png/svg/eps，默认png)
- `--save_publication_figs`: 是否保存发表质量图片（默认true）
- `--tsne_sample_size`: t-SNE采样大小（默认1000）
- `--include_fourier`: 是否包含Fourier分析（默认false）

### 输出参数
- `--output_dir`: 输出目录（默认exp/inference_visualization）
- `--device`: 计算设备 (cuda/cpu，默认cuda)

### 可重复性参数
- `--seed`: 随机种子，确保结果可重复（默认42）

## 输出文件

脚本会在输出目录生成以下可视化文件：

1. **专家路由热力图**
   - `expert_routing_heatmap_inference.png/pdf`
   - 显示各领域对不同专家的使用强度

2. **t-SNE专家专业化图**
   - `tsne_specialization_inference.png/pdf`
   - 展示专家嵌入空间的聚类和专业化模式

3. **Fourier频率分析**（如果启用）
   - `multi_domain_fourier_layer_X_inference.png/pdf`
   - 每个transformer层的多领域频率模式比较

## 与训练时可视化的对比

- **数据来源**: 推理脚本使用验证/测试集，训练时使用训练集
- **可视化函数**: 完全相同的enhanced visualization模块
- **图表样式**: 完全相同的期刊级别样式
- **专家分析**: 相同的专家路由和专业化分析方法

## 使用场景

1. **模型分析**: 分析训练好的模型在测试数据上的专家使用模式
2. **性能调试**: 检查模型推理时的专家专业化程度
3. **论文发表**: 生成发表质量的专家分析图表
4. **模型比较**: 比较不同checkpoint的专家行为差异

## 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 使用更小的batch_size或切换到CPU
   --batch_size 64 --device cpu
   ```

2. **数据集路径问题**
   ```bash
   # 确保实验目录包含args.txt文件
   ls exp/your_experiment/args.txt
   ```

3. **快速测试**
   ```bash
   # 限制处理批次数量进行快速测试
   --max_batches 5
   ```

### 依赖检查

确保已安装以下依赖：
- torch >= 1.8.0
- matplotlib >= 3.3.0
- seaborn >= 0.11.0
- scikit-learn >= 0.24.0
- tqdm >= 4.60.0

## 性能提示

1. **内存优化**: 使用`--max_batches`参数限制处理的批次数量
2. **速度优化**: 如果只需要特定可视化，可以修改脚本跳过不需要的部分
3. **质量平衡**: 高DPI设置会增加生成时间，根据需要调整`--viz_dpi`

## 扩展功能

脚本支持以下扩展：
- 自定义采样策略
- 额外的专家分析指标
- 批量处理多个模型
- 与SwanLab等平台的集成