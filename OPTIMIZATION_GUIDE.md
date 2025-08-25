# 专家专业化优化功能指南

本文档描述了新实现的专家专业化优化功能，用于改善MoE模型中领域-专家对应关系。

## 🚀 新功能

### 1. 温度调节门控 (Temperature-Adjusted Gating)
- **作用**: 随着时间推移使专家选择更加尖锐，鼓励专业化
- **参数**:
  - `--gate_temperature`: 初始softmax温度 (默认: 1.0)
  - `--min_gate_temperature`: 最小温度 (默认: 0.1)
  - `--temperature_decay`: 每步衰减率 (默认: 0.995)

### 2. 专业化损失 (Specialization Loss)
- **作用**: 鼓励领域i主要使用专家i
- **参数**:
  - `--use_specialization_loss`: 启用/禁用 (默认: False)
  - `--specialization_weight`: 损失权重 (默认: 0.1)

### 3. 对比学习 (Contrastive Learning)
- **作用**: 使相同领域的表示更相似，不同领域的表示更不同
- **参数**:
  - `--use_contrastive_loss`: 启用/禁用 (默认: False)
  - `--contrastive_weight`: 损失权重 (默认: 0.05)

### 4. 自适应负载均衡 (Adaptive Load Balancing)
- **作用**: 随着专业化程度增加，减少负载均衡权重
- **参数**:
  - `--use_adaptive_balance`: 启用/禁用 (默认: False)

## 🔧 使用方法

### 基础用法 (启用所有优化)
```bash
python main.py \
  --train_dir your_experiment \
  --gate_temperature 2.0 \
  --use_specialization_loss true \
  --use_contrastive_loss true \
  --use_adaptive_balance true
```

### 保守用法 (仅从温度开始)
```bash
python main.py \
  --train_dir your_experiment \
  --gate_temperature 1.5 \
  --min_gate_temperature 0.2 \
  --temperature_decay 0.998
```

### 激进专业化
```bash
python main.py \
  --train_dir your_experiment \
  --gate_temperature 3.0 \
  --min_gate_temperature 0.05 \
  --use_specialization_loss true \
  --specialization_weight 0.2 \
  --use_contrastive_loss true \
  --contrastive_weight 0.1 \
  --use_adaptive_balance true
```

## 📊 预期效果

使用这些优化后，您应该观察到：

1. **更尖锐的专家使用**: 热图应显示更清晰的对角线模式
2. **更好的t-SNE聚类**: 专家着色的t-SNE应显示更明显的聚类
3. **改善的专业化**: 每个领域应主要使用其对应的专家
4. **保持的性能**: 推荐指标应保持稳定或有所改善

## ⚠️ 重要注意事项

1. **向后兼容性**: 所有优化默认禁用，现有模型不受影响
2. **渐进引入**: 从温度门控开始，然后逐步添加其他优化
3. **超参数调优**: 根据您的具体数据集和需求调整权重
4. **监控**: 注意过度激进的专业化可能会损害泛化能力

## 🔬 技术细节

- **温度门控**: 应用于'shared_base'和'vanilla'策略中的softmax
- **专业化损失**: 使用当前分布与理想(独热)分布之间的KL散度
- **对比损失**: 最大化不同领域中心之间的余弦距离
- **自适应均衡**: 根据门控熵调整负载均衡权重

## 🐛 故障排除

- 如果损失变成NaN: 减少损失权重或增加min_temperature
- 如果专业化过于激进: 增加min_temperature或减少specialization_weight
- 如果性能下降: 尝试启用adaptive_balance来保持一些专家多样性
