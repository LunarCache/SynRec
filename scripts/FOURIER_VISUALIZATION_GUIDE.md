# Fourier注意力可视化与消融实验指南

## 🎯 概述

这份指南展示了完整的Fourier注意力可视化和频域消融实验系统的使用方法。

## ✅ 已修复的问题

### 1. **数据传递问题修复**
- ✅ 修复了`fourier_rating_attention_detailed`数据生成和传递
- ✅ 确保与`plot_multi_domain_fourier_comparison_journal`的数据格式兼容
- ✅ 解决了matplotlib colormap注册的版本兼容性问题

### 2. **消融实验支持**
- ✅ 添加了`AblationMode`枚举：`FULL`, `LOW_FREQ_ONLY`, `HIGH_FREQ_ONLY`
- ✅ 支持运行时动态切换消融模式
- ✅ 提供完整的自动化消融实验框架

## 🚀 功能特性

### 🔍 **Fourier注意力可视化**

#### 基础多领域对比（详细分支分析）
```python
from visualization import plot_multi_domain_fourier_comparison_journal

# 在训练过程中自动调用，显示各领域的Fourier注意力分支分析
# 3行布局：高频分支、低频分支、自适应权重
fig, saved_files = plot_multi_domain_fourier_comparison_journal(
    fourier_attn_data=viz_data['fourier_rating_attention_detailed'][layer_idx],
    domain_map={0: 'Beauty', 1: 'Games', 2: 'MovieLens'},
    layer_idx=0,
    epoch=1,
    config=None,  # 使用默认配置
    save_plots=True
)
```

#### 增强版多维度分析
```python
from visualization import plot_multi_domain_fourier_comparison_enhanced_journal

# 更详细的分析，包含分支对比和能量分布
fig, saved_files = plot_multi_domain_fourier_comparison_enhanced_journal(
    fourier_attn_data=fourier_data,
    domain_map=domain_map,
    layer_idx=0,
    epoch=1,
    show_frequency_analysis=True,  # 显示频率能量分布
    show_branch_comparison=True    # 显示高频/低频分支对比
)
```

### 🎯 **多领域FFT幅度谱比较** ⭐ **NEW**

#### 代表用户频谱对比分析
```bash
# 生成多领域代表用户的3×3对比分析（含FFT幅度谱）
python scripts/multi_domain_user_comparison.py \
    --experiment_dir exp/your_experiment \
    --state_dict_path exp/your_experiment/best_model.pth \
    --output_dir exp/multi_domain_fft_comparison \
    --selection_strategy median \
    --journal_style nature
```

#### 3×3分析布局设计
```
第1行: 时频分析       │ Beauty时域信号   │ Games时域信号    │ MovieLens时域信号 │
第2行: 分解分析       │ 原始/低/高频     │ 原始/低/高频      │ 原始/低/高频      │  
第3行: FFT幅度谱     │ 频谱+峰值分析    │ 频谱+峰值分析     │ 频谱+峰值分析     │
```

#### FFT幅度谱核心功能
- **专业频谱可视化**: 基于cli.py的plot_time_and_spectrum标准设计
- **双截止线显示**: 80%能量截止 + 模型学习的自适应截止  
- **峰值周期标注**: 自动检测并标注主导频率PEAK_K和对应周期PERIOD_T
- **统一配色方案**: 使用matplotlib prop_cycle确保视觉一致性
- **能量分布信息**: 右上角显示低频/高频能量比例
- **归一化频率**: 顶部辅助坐标轴显示k/N标准化频率

#### 频域对比分析价值
- **领域特异性识别**: 直观比较不同领域的频域行为模式
- **周期性模式发现**: 揭示Beauty(短周期)、Games(中周期)、MovieLens(长周期)的差异
- **模型验证**: 验证领域特异性编码器学习到的频域特征
- **研究洞察**: 为推荐系统的领域适应性提供科学依据

### 🧪 **消融实验系统**

#### 手动消融模式切换
```python
from keys.temporal_rating_modules import AblationMode

# 切换到仅低频模式
model.enhanced_rating_module.set_ablation_mode(AblationMode.LOW_FREQ_ONLY)

# 切换到仅高频模式
model.enhanced_rating_module.set_ablation_mode(AblationMode.HIGH_FREQ_ONLY)

# 切换回完整模式
model.enhanced_rating_module.set_ablation_mode(AblationMode.FULL)
```

#### 自动化消融实验
```bash
# 运行完整的消融实验
python scripts/run_fourier_ablation_fixed.py \
    --model_path exp/best_model.pth \
    --output_dir exp/fourier_ablation \
    --generate_plots
```

### 📊 **消融结果可视化**
```python
from visualization import plot_frequency_ablation_comparison_journal

# 可视化消融实验结果
ablation_results = {
    'full': {'ndcg@10': 0.4523, 'hit@10': 0.6234, 'recall@10': 0.3456},
    'low_only': {'ndcg@10': 0.3876, 'hit@10': 0.5423, 'recall@10': 0.2987},
    'high_only': {'ndcg@10': 0.3123, 'hit@10': 0.4876, 'recall@10': 0.2234}
}

fig, saved_files = plot_frequency_ablation_comparison_journal(
    ablation_results=ablation_results,
    domain_map={0: 'Beauty', 1: 'Games', 2: 'MovieLens'},
    epoch=50
)
```

## 📈 **数据流架构**

### 训练时的自动可视化
```
1. TemporalEnhancedRatingModule.forward()
   ├── OptimizedFourierRatingEncoder (per domain)
   │   ├── FFT分解 → 低频/高频分支
   │   ├── 双分支注意力 → branch1_attn, branch2_attn
   │   └── 收集visualization_data
   ├── 按domain_id组织数据
   └── 返回frequency_analysis

2. HAGMRec.log2feats()
   ├── 接收frequency_analysis
   ├── 生成fourier_rating_attention_detailed
   └── 按layer和domain组织数据

3. main.py训练循环
   ├── 检查fourier_rating_attention_detailed
   └── 调用plot_multi_domain_fourier_comparison_journal
```

### 多领域FFT幅度谱比较数据流 ⭐ **NEW**
```
1. multi_domain_user_comparison.py
   ├── select_representative_users() → 每领域选择代表用户
   ├── analyze_representative_user() → 对每个代表用户进行分析
   │   ├── compute_scalar_fft_decomposition() → FFT分解
   │   ├── find_top_peaks() → 峰值检测 → PEAK_K  
   │   ├── 计算PERIOD_T = L / PEAK_K
   │   ├── auto_cutoff_80() → 80%能量截止
   │   └── energy_ratio() → 低频/高频能量比例
   └── generate_multi_domain_comparison_plot() → 3×3可视化
       ├── _plot_independent_time_frequency()
       ├── _plot_independent_decomposition() 
       └── _plot_independent_fft_spectrum() → 新增FFT幅度谱

2. 输出数据结构
   ├── representative_users.json → 代表用户信息
   ├── comparison_metrics.json → 包含PEAK_K和PERIOD_T
   └── multi_domain_comparison.png → 3×3对比图
```

### 数据结构
```python
viz_data = {
    'fourier_rating_attention_detailed': [
        # Layer 0
        {
            0: {  # domain_id 0
                'branch1': tensor,  # 高频分支注意力 (short-term)
                'branch2': tensor,  # 低频分支注意力 (long-term)  
                'adaptive_weights': tensor  # 自适应融合权重
            },
            1: {...},  # domain_id 1
            2: {...}   # domain_id 2
        },
        # Layer 1
        {...}
    ]
}

# 实际可视化函数签名
plot_multi_domain_fourier_comparison_journal(
    fourier_attn_data: Dict[int, Dict[str, torch.Tensor]],  # 单层数据
    domain_map: Dict[int, str],
    layer_idx: int,
    epoch: int,
    config: Optional[VisualizationConfig] = None,
    save_plots: bool = True
) -> Tuple[plt.Figure, List[str]]
```

## 🔬 **消融实验分析能力**

### 可分析的研究问题

1. **频域贡献分析**
   - 低频成分（长期趋势）对推荐的贡献度
   - 高频成分（短期变化）对推荐的贡献度
   - 两个成分的协同作用效果

2. **领域特异性分析**
   - 不同领域对频率成分的依赖差异
   - Beauty vs Games vs MovieLens的频域偏好

3. **性能退化分析**
   - 移除某个频域成分后的性能损失
   - 各频域成分的重要性排序

### 自动生成的分析报告

消融实验会自动生成：
- **JSON格式结果**：详细的数值指标
- **可视化图表**：性能对比柱状图
- **分析摘要**：最佳模式识别和性能退化分析

## 📋 **使用清单**

### ✅ 验证系统工作状态
1. 运行训练，观察是否出现 "🎨 Triggering visualization" 信息
2. 检查不再有matplotlib colormap警告
3. 确认生成了Fourier注意力可视化图表

### ✅ 进行消融实验
1. 确保有训练好的模型检查点
2. 运行`scripts/run_fourier_ablation_fixed.py`
3. 查看生成的结果报告和可视化

### ✅ 深度分析
1. 使用增强版可视化查看频率能量分布
2. 分析不同领域的频域偏好差异
3. 基于消融结果优化模型设计

## 🎨 **可视化效果**

### 标准多领域对比
- 显示各领域的综合Fourier注意力热图
- 使用期刊级配色和布局

### 详细分支分析（基础版）
- **第1行**：高频分支注意力（红色调热图）
- **第2行**：低频分支注意力（蓝色调热图）
- **第3行**：自适应融合权重（发散色调热图）

### 增强版分析
- **可选3行布局**：根据参数动态调整显示内容
- **分支对比模式**：高频/低频分支并排对比
- **频率分析模式**：频率能量分布饼图

### 消融实验结果
- 多指标对比柱状图
- 性能退化百分比显示
- 各模式的相对表现

## 🔧 **技术细节**

### 支持的消融模式
- `FULL`: 完整模型（原始+低频+高频融合）
- `LOW_FREQ_ONLY`: 仅使用低频增强特征
- `HIGH_FREQ_ONLY`: 仅使用高频增强特征

### 频域处理流程
1. **FFT分解**：将rating嵌入变换到频域
2. **频率分离**：基于learnable cutoff分离低/高频
3. **IFFT重构**：重构时域信号
4. **双分支注意力**：分别处理低频和高频信号
5. **自适应融合**：学习最优的融合权重

### 可视化技术规格
- **分辨率**：300 DPI，支持PDF/PNG/SVG
- **配色**：符合Nature/Science期刊标准
- **兼容性**：支持matplotlib 3.x各版本

## 🎯 **预期应用**

这个系统让你能够：

1. **验证Fourier模块工作**：确认注意力数据正确生成和可视化
2. **深入理解频域机制**：分析低频/高频成分的作用机制  
3. **开展消融实验**：定量分析各频域成分的贡献度
4. **优化模型设计**：基于消融结果调整模型架构
5. **撰写技术论文**：生成期刊级质量的可视化图表
6. **多领域FFT分析** ⭐ **NEW**：比较不同领域的频域行为差异
7. **用户行为周期性研究**：通过PEAK_K和PERIOD_T量化用户行为模式
8. **领域适应性验证**：验证模型在不同领域的频域特征学习效果

### 🎨 **新增可视化效果**

#### 多领域FFT幅度谱对比
- **3×3专业布局**：时频分析 + 分解分析 + FFT幅度谱
- **领域特异性展示**：Beauty、Games、MovieLens的频域差异
- **周期性模式识别**：主导频率和对应时间周期的直观展示
- **能量分布对比**：低频/高频能量比例的跨领域比较

#### 研究洞察支持
- **用户行为模式分类**：基于PEAK_K和PERIOD_T的用户分群
- **推荐时机优化**：根据用户行为周期优化推荐频率
- **领域特异性设计**：为不同领域设计针对性的推荐策略
- **模型解释性增强**：通过频域分析解释模型决策过程

系统现在已经完全就绪，可以支持你的Fourier频域推荐系统研究！🚀