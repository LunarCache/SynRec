# Fourier消融实验统计分析工具

本目录包含带显著性检验的Fourier消融实验分析工具。

## 文件说明

### 主要脚本
- `run_fourier_ablation_statistical.py`: 带统计分析的消融实验主脚本
- `run_fourier_ablation_fixed.py`: 原始消融实验脚本（单次运行）
- `example_statistical_ablation.py`: 使用示例和演示脚本

### 主要改进

#### 统计功能增强
- ✅ 多次独立运行（默认30次）
- ✅ 配对t检验和Wilcoxon符号秩检验
- ✅ Cohen's d效应量计算
- ✅ 多重比较校正（Bonferroni, Holm, FDR）
- ✅ Bootstrap置信区间
- ✅ 详细统计报告生成

#### 可视化增强
- ✅ 带误差棒和置信区间的性能比较图
- ✅ 显著性检验热力图
- ✅ 效应量比较图
- ✅ 分布比较箱线图

#### 输出格式
- ✅ JSON格式完整结果
- ✅ Markdown格式可读报告
- ✅ CSV格式数据表
- ✅ 高质量PNG图表

## 快速开始

### 1. 基本使用
```bash
python run_fourier_ablation_statistical.py \
    --model_path path/to/model.pth \
    --output_dir results/statistical_ablation \
    --generate_plots
```

### 2. 运行示例
```bash
python example_statistical_ablation.py
```

### 3. 自定义参数
```bash
python run_fourier_ablation_statistical.py \
    --model_path path/to/model.pth \
    --output_dir results/statistical_ablation \
    --num_runs 50 \
    --alpha 0.01 \
    --effect_size_threshold 0.8 \
    --correction_method fdr_bh \
    --generate_plots
```

## 统计方法

### 实验设计
- **对照组**: FULL（完整模型）
- **实验组**: LOW_FREQ_ONLY（仅低频）、HIGH_FREQ_ONLY（仅高频）
- **重复**: 每种条件运行多次（默认30次）
- **随机化**: 每次运行使用不同随机种子

### 统计检验
1. **描述性统计**: 均值、标准差、中位数、置信区间
2. **显著性检验**: 配对t检验（主要）+ Wilcoxon检验（备用）
3. **效应量**: Cohen's d衡量实际差异大小
4. **多重比较校正**: FDR控制假发现率

### 结果解读
- **p值**: 差异的统计显著性（p < 0.05为显著）
- **效应量**: 差异的实际意义（|d| ≥ 0.5为中等以上效应）
- **置信区间**: 效应量的可能范围
- **校正后p值**: 控制多重比较错误率

## 最佳实践

### 运行次数建议
- **快速验证**: 10-15次
- **常规分析**: 30次（推荐）
- **严格验证**: 50次或更多

### 参数设置建议
- **显著性水平**: α = 0.05（常规）或α = 0.01（严格）
- **效应量阈值**: 0.5（中等效应）或0.8（大效应）
- **校正方法**: fdr_bh（推荐）或bonferroni（保守）

### 硬件要求
- **GPU内存**: 建议8GB以上
- **系统内存**: 建议16GB以上
- **存储空间**: 每次分析约需要100-500MB

## 输出文件详解

### 主要结果
1. `statistical_ablation_results.json`: 完整的统计分析结果
2. `statistical_report.md`: 人类可读的详细报告
3. `descriptive_statistics.csv`: 所有模式的描述性统计
4. `significance_tests.csv`: 成对比较的显著性检验结果

### 可视化图表
1. `performance_comparison_with_ci.png`: 性能对比（带置信区间）
2. `significance_heatmap.png`: 显著性检验热力图
3. `effect_size_comparison.png`: 效应量对比图
4. `distribution_comparison.png`: 数据分布箱线图

## 故障排除

### 常见问题
1. **内存不足**: 减少`--num_runs`或使用`--device cpu`
2. **运行时间长**: 正常现象，可减少运行次数
3. **统计错误**: 检查scipy和pandas版本
4. **模型加载失败**: 检查模型路径和兼容性

### 依赖要求
- Python >= 3.8
- PyTorch >= 1.8.0
- SciPy >= 1.7.0
- Pandas >= 1.3.0
- Matplotlib >= 3.3.0
- Seaborn >= 0.11.0

## 与原版本的对比

| 功能 | 原版本 | 统计版本 |
|------|--------|----------|
| 运行次数 | 1次 | 30次（可配置） |
| 统计检验 | 无 | t检验、Wilcoxon |
| 效应量 | 无 | Cohen's d |
| 多重比较校正 | 无 | Bonferroni、FDR |
| 置信区间 | 无 | Bootstrap CI |
| 报告格式 | 简单 | 详细报告+可视化 |
| 结果可靠性 | 低 | 高 |

## 学术使用

此工具适用于学术研究和论文发表，提供了严格的统计验证：

- **假设检验**: 正确的配对检验设计
- **效应量报告**: 符合APA标准
- **多重比较控制**: 避免假阳性结果
- **可重现性**: 详细的参数记录和随机种子

建议在论文中报告：
1. 实验设计（运行次数、随机种子）
2. 统计方法（检验类型、校正方法）
3. 效应量和置信区间
4. 校正后的p值

## 致谢

基于原始的 `run_fourier_ablation_fixed.py` 开发，增加了完整的统计分析功能。