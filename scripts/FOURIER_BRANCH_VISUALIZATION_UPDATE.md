# 🎨 Fourier分支分离可视化 - 使用指南

## ✅ 修改完成

现在的 `plot_multi_domain_fourier_comparison_journal()` 函数已经完全替换为**详细分支展示模式**！

## 🆕 新的可视化布局

### **3行 x N列布局**
```
┌─────────────────────────────────────────────────────┐
│ 第1行: High-Frequency Branch (红色调)                │
│ [Beauty]    [Games]    [MovieLens]                  │
│   🔥          🔥          🔥                         │
├─────────────────────────────────────────────────────┤
│ 第2行: Low-Frequency Branch (蓝色调)                 │
│ [Beauty]    [Games]    [MovieLens]                  │
│   🌊          🌊          🌊                         │
├─────────────────────────────────────────────────────┤
│ 第3行: Adaptive Weights (发散色调)                   │
│ [Beauty]    [Games]    [MovieLens]                  │
│   ⚖️          ⚖️          ⚖️                         │
└─────────────────────────────────────────────────────┘
```

### **差异化配色方案**
- **High-Freq Branch**: 红色调 (Reds colormap) - 突出短期、高频变化
- **Low-Freq Branch**: 蓝色调 (Blues colormap) - 体现长期、低频趋势  
- **Adaptive Weights**: 发散色调 (RdYlBu_r) - 显示权重分布平衡

## 🔍 现在你能看到什么

### 1. **高频分支注意力模式**
- 捕捉短期用户行为变化
- 显示近期评分的注意力焦点
- 红色调热图，突出局部模式

### 2. **低频分支注意力模式** 
- 体现长期用户偏好趋势
- 显示历史行为的全局影响
- 蓝色调热图，展现平滑模式

### 3. **自适应融合权重**
- 显示三个组件的动态平衡：
  - `Original`: 原始rating嵌入权重
  - `Long-term`: 低频增强特征权重
  - `Short-term`: 高频增强特征权重
- 横轴为序列位置，纵轴为权重类型

## 🎯 分析能力

现在你可以直观地分析：

### **频域对比分析**
- 高频 vs 低频注意力的空间分布差异
- 不同领域的频域偏好特征
- 序列不同位置的频率响应模式

### **领域特异性分析**
- Beauty, Games, MovieLens 在频域上的差异
- 各领域的短期/长期偏好倾向
- 自适应权重的领域特定分布

### **动态权重分析**
- 序列中不同位置的组件重要性
- 权重分布随位置的变化趋势
- 三个组件的协调与竞争关系

## 🚀 使用方式

### **训练时自动生成**
训练过程中会自动调用新的可视化，显示详细的分支分析：

```python
# 在main.py中自动触发
if 'fourier_rating_attention_detailed' in viz_data:
    fourier_detailed = viz_data['fourier_rating_attention_detailed']
    for layer_idx, fourier_attn_data in enumerate(fourier_detailed):
        if fourier_attn_data is not None and len(fourier_attn_data) > 1:
            multi_fig, multi_saved_files = plot_multi_domain_fourier_comparison_journal(
                fourier_attn_data, domain_map, layer_idx, epoch,
                enhanced_viz_config, save_plots=args.save_publication_figs
            )
```

### **手动调用分析**
```python
from visualization import plot_multi_domain_fourier_comparison_journal

# 使用模型生成的数据
fig, saved_files = plot_multi_domain_fourier_comparison_journal(
    fourier_attn_data=fourier_data,  # 来自模型的分支数据
    domain_map={0: 'Beauty', 1: 'Games', 2: 'MovieLens'},
    layer_idx=0,
    epoch=10,
    config=None,  # 可选：可视化配置
    save_plots=True  # 是否保存图片
)
```

## 💡 预期洞察

这种详细分支展示将帮助你：

1. **验证Fourier模块设计** - 确认高频/低频分离是否如预期工作
2. **分析领域差异** - 发现不同推荐场景对频域的不同需求
3. **优化权重机制** - 观察自适应融合的实际表现
4. **指导模型改进** - 基于可视化结果调整架构设计

## 🎊 完成状态

✅ **合并显示** → **分支分离展示** 转换完成
✅ **红蓝配色区分** 高频/低频分支  
✅ **权重可视化** 展示自适应融合过程
✅ **期刊级质量** 符合学术发表标准
✅ **测试验证** 功能正常工作

现在你的Fourier注意力可视化提供了最详细和最有洞察力的分析视图！🚀