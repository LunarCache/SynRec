
# Multi-Domain Frequency Analysis Scripts

该目录包含了用于多领域频域分析和可视化的脚本，支持HAGMRec模型的领域特异性分析。

## 📁 文件概览

- **`cli.py`**: 单用户频域分析工具，支持领域对比
- **`batch_viz_stats.py`**: 批量用户统计分析，支持领域感知处理
- **`multi_domain_user_comparison.py`**: 多领域代表用户比较分析，包含FFT幅度谱对比 ⭐ **NEW**
- **`run_fourier_ablation_fixed.py`**: Fourier频域消融实验脚本，支持自动化消融研究 🧪 **NEW**
- **`viz_lib.py`**: 核心可视化和分析库
- **`README.md`**: 本说明文档

## 🚀 快速开始

### 环境准备
```bash
# 激活conda环境
conda activate rec
cd /path/to/CMREC
```

## 1️⃣ 单用户频域分析 (cli.py)

### 基础用法
```bash
# 基础分析：自动检测用户领域
python scripts/cli.py \
    --experiment_dir exp/your_experiment \
    --user_id 123 \
    --output_dir exp/user_analysis

# 指定领域分析
python scripts/cli.py \
    --experiment_dir exp/your_experiment \
    --user_id 123 \
    --domain_id 1 \
    --output_dir exp/user_analysis
```

### 🔥 多领域对比模式
```bash
# 生成跨领域对比分析
python scripts/cli.py \
    --experiment_dir exp/your_experiment \
    --user_id 123 \
    --compare_domains true \
    --output_dir exp/domain_comparison \
    --journal_style high_quality
```

### 核心参数
- `--domain_id`: 指定使用的领域ID (0, 1, 2)
- `--compare_domains`: 启用多领域对比模式
- `--auto_domain`: 自动检测用户领域 (默认: true)
- `--cutoff_policy`: 频谱区域策略 (model, energy80, fixed_k)

## 2️⃣ 批量统计分析 (batch_viz_stats.py)

### 🌟 领域感知批量分析 (平衡采样)
```bash
# 平衡采样 + 高质量用户(longest策略)：确保每个领域用户数量相等，且选择序列最长的用户
python scripts/batch_viz_stats.py \
    --experiment_dir exp/your_experiment \
    --n_users 300 \
    --sample_strategy longest \
    --domain_aware true \
    --compare_domains true \
    --output_dir exp/balanced_longest_analysis

# 平衡采样 + 随机用户(random策略)：确保每个领域用户数量相等，随机选择用户
python scripts/batch_viz_stats.py \
    --experiment_dir exp/your_experiment \
    --n_users 300 \
    --sample_strategy random \
    --domain_aware true \
    --compare_domains true \
    --output_dir exp/balanced_random_analysis
```

### 特定领域分析
```bash
# 只分析特定领域的用户
python scripts/batch_viz_stats.py \
    --experiment_dir exp/your_experiment \
    --analyze_domain 1 \
    --n_users 100 \
    --output_dir exp/domain_1_analysis
```

### 🔧 核心参数
- `--domain_aware`: 启用领域感知处理 (默认: true)
- `--analyze_domain`: 只分析指定领域的用户
- `--compare_domains`: 生成领域对比图表 (默认: true)
- `--n_users`: 总采样用户数，自动按领域平衡分配
- `--sample_strategy`: 采样策略选择
  - `longest`: 选择序列最长的高质量用户
  - `random`: 随机无偏采样

## 📊 输出文件说明

### 单用户分析输出
```
exp/user_analysis/
├── fig1_time_vs_frequency_domain_X.png     # 时频域分析图
├── fig2_decomposition_overlay_domain_X.png # 分解叠加图
├── fig3_tsne_embeddings_domain_X.png       # t-SNE嵌入图
├── analysis_metrics_domain_X.json          # 分析指标
└── comparison_*.png                         # 领域对比图 (if --compare_domains)
```

### 批量分析输出
```
exp/batch_stats/
├── metrics_user_level.csv                  # 用户级指标
├── summary_stats.json                      # 总体统计
├── domain_comparison_stats.json            # 领域对比统计 (NEW)
├── group_lowfrac_hist.png                  # 低频能量分布
├── domain_comparison_*.png                 # 领域对比图表 (NEW)
└── user_*/                                  # 用户级图表 (可选)
```

### 多领域代表用户比较输出 ⭐ **NEW**
```
exp/multi_domain_comparison/
├── multi_domain_comparison.png             # 3×3主要对比图 (含FFT幅度谱)
│   ├── Row 1: 时频分析 (Beauty|Games|MovieLens)
│   ├── Row 2: 分解分析 (原始/低频/高频成分)  
│   └── Row 3: FFT幅度谱 (频谱+峰值+截止线)
├── representative_users.json               # 代表用户信息
│   ├── 用户ID、序列长度、领域信息
├── comparison_metrics.json                 # FFT分析指标
│   ├── PEAK_K (主导频率索引)
│   ├── PERIOD_T (主导周期)
│   ├── 低频/高频能量比例
│   └── R²分解质量得分
└── analysis_summary.txt                    # 分析摘要 (可选)
```

### Fourier频域消融实验输出 🧪 **NEW**
```
exp/fourier_ablation/
├── ablation_results.json                   # 完整消融实验结果
│   ├── full: 完整模型性能 (NDCG@10, Hit@10)
│   ├── low_only: 仅低频成分性能
│   ├── high_only: 仅高频成分性能
│   └── 按领域分解的详细指标
├── intermediate_results.json               # 实验过程中间结果
├── domain_frequency_trend_comparison.png   # 消融对比可视化图表
│   ├── 双指标并排对比 (NDCG@10 & Hit@10)
│   ├── 三领域趋势线 (Beauty|Games|MovieLens)
│   ├── 精确数值标注
│   └── 期刊级300DPI质量
└── figures/                                # 可视化输出目录
    └── domain_frequency_trend_comparison_epoch_0.png
```

## 🎯 典型使用场景

### 场景1: 探索用户的领域特异性
```bash
# 对比同一用户在不同领域编码器下的表现
python scripts/cli.py \
    --experiment_dir exp/hagmrec_model \
    --user_id 42 \
    --compare_domains true \
    --output_dir exp/user42_domains
```

### 场景2: 领域级批量统计 (平衡对比)
```bash
# 使用longest策略：每个领域选择高质量用户，确保公平对比
python scripts/batch_viz_stats.py \
    --experiment_dir exp/hagmrec_model \
    --n_users 600 \
    --sample_strategy longest \
    --domain_aware true \
    --compare_domains true \
    --journal_style nature \
    --output_dir exp/balanced_longest_comparison

# 使用random策略：每个领域随机采样，避免选择偏差
python scripts/batch_viz_stats.py \
    --experiment_dir exp/hagmrec_model \
    --n_users 600 \
    --sample_strategy random \
    --domain_aware true \
    --compare_domains true \
    --journal_style nature \
    --output_dir exp/balanced_random_comparison
```

### 场景3: 特定领域深度分析
```bash
# 专门分析beauty领域(domain 0)的用户
python scripts/batch_viz_stats.py \
    --experiment_dir exp/hagmrec_model \
    --analyze_domain 0 \
    --n_users 300 \
    --save_user_figs true \
    --output_dir exp/beauty_domain_deep_dive
```

### 场景4: 多领域代表用户FFT频谱对比 ⭐ **NEW**
```bash
# 标准代表用户分析（推荐）
python scripts/multi_domain_user_comparison.py \
    --experiment_dir exp/hagmrec_model \
    --state_dict_path exp/hagmrec_model/best_model.pth \
    --output_dir exp/multi_domain_fft_analysis \
    --selection_strategy median \
    --journal_style nature

# 高质量用户FFT对比（序列最长）
python scripts/multi_domain_user_comparison.py \
    --experiment_dir exp/hagmrec_model \
    --state_dict_path exp/hagmrec_model/best_model.pth \
    --output_dir exp/high_quality_user_fft \
    --selection_strategy longest \
    --min_sequence_length 50 \
    --journal_style high_quality
```

### 场景5: Fourier频域消融实验 🧪 **NEW**
```bash
# 基础消融实验（评估频域成分贡献）
python scripts/run_fourier_ablation_fixed.py \
    --model_path exp/hagmrec_model/best_model.pth \
    --output_dir exp/fourier_ablation_basic \
    --device auto

# 完整消融实验（包含可视化）
python scripts/run_fourier_ablation_fixed.py \
    --model_path exp/hagmrec_model/best_model.pth \
    --output_dir exp/fourier_ablation_complete \
    --generate_plots \
    --device cuda

# 论文级消融实验（高质量可视化）
python scripts/run_fourier_ablation_fixed.py \
    --model_path exp/hagmrec_model/best_model.pth \
    --output_dir exp/fourier_ablation_journal \
    --generate_plots \
    --device cuda
```

## ⚙️ 高级配置

### 可视化风格
- `--journal_style`: nature, science, cell, high_quality

### 采样策略
- 🎯 **平衡采样**: 自动为每个领域分配相等的用户数量，确保公平对比
- `--sample_strategy longest`: 在每个领域内选择序列最长的高质量用户
- `--sample_strategy random`: 在每个领域内随机选择用户，避免偏差
- `--n_users`: 总分析用户数量，自动平均分配到各个领域

### 频谱分析策略
- `--cutoff_policies`: model,energy80,fixed_k
- `--fixed_k`: 固定截止频率索引

### t-SNE参数
- `--tsne_perplexity`: 困惑度参数
- `--tsne_seed`: 随机种子

## 3️⃣ 多领域代表用户比较 (multi_domain_user_comparison.py) ⭐ **NEW**

### 🌟 核心功能
生成统一的多领域代表用户对比分析，包含全新的FFT幅度谱比较功能，采用专业的3×3网格布局展示不同领域的频域特征差异。

### 基础用法
```bash
# 标准代表用户比较分析
python scripts/multi_domain_user_comparison.py \
    --experiment_dir exp/your_experiment \
    --state_dict_path exp/your_experiment/best_model.pth \
    --output_dir exp/multi_domain_comparison \
    --selection_strategy median

# 选择序列最长的代表用户
python scripts/multi_domain_user_comparison.py \
    --experiment_dir exp/your_experiment \
    --state_dict_path exp/your_experiment/best_model.pth \
    --output_dir exp/multi_domain_comparison_longest \
    --selection_strategy longest \
    --journal_style nature
```

### 🎯 3×3 分析布局
```
第1行: 时频分析       │ Beauty      │ Games        │ MovieLens    │
第2行: 分解分析       │ 原始/低/高频  │ 原始/低/高频   │ 原始/低/高频  │  
第3行: FFT幅度谱 (NEW)│ 频谱+峰值    │ 频谱+峰值     │ 频谱+峰值    │
```

### 📊 FFT幅度谱新功能特点
- **专业频谱可视化**: 基于cli.py的plot_time_and_spectrum设计
- **统一配色方案**: 使用matplotlib prop_cycle确保一致性
- **双截止线标记**: 显示80%能量截止和模型学习的截止
- **峰值周期分析**: 自动标注主导频率和对应的时间周期
- **归一化频率轴**: 提供k/N标准化频率显示
- **能量比例框**: 右上角显示低频/高频能量分布

### 🔧 核心参数
- `--selection_strategy`: 代表用户选择策略
  - `longest`: 选择序列最长的用户（高质量用户）
  - `median`: 选择序列长度中位数的用户（典型用户）
  - `random`: 随机选择用户
- `--min_sequence_length`: 最小序列长度要求 (默认: 20)
- `--cutoff_policy`: 频谱截止策略 (model, energy80, fixed_k)
- `--journal_style`: 可视化风格 (nature, science, high_quality)

### 📈 输出文件
```
exp/multi_domain_comparison/
├── multi_domain_comparison.png           # 主要3×3对比图 (含FFT幅度谱)
├── representative_users.json             # 选中的代表用户信息
├── comparison_metrics.json               # 对比指标和FFT峰值数据
└── analysis_summary.txt                  # 分析摘要 (可选)
```

### 🎯 分析指标说明
#### PEAK_K (主导频率索引)
- **定义**: FFT幅度谱中最高峰值对应的频率索引
- **含义**: 用户评分行为的主导频率成分
- **范围**: 1 到 L//2 (L为序列长度)

#### PERIOD_T (主导周期)  
- **定义**: PEAK_K对应的时间域周期长度
- **计算**: T = L / PEAK_K
- **含义**: 用户评分行为的典型重复周期
- **单位**: 时间步长

#### 领域特征示例
- **Beauty**: 短周期，频繁购买 (PEAK_K高, PERIOD_T小)
- **Games**: 中等周期，定期游戏 (PEAK_K中等, PERIOD_T中等)  
- **MovieLens**: 长周期，偶尔观影 (PEAK_K低, PERIOD_T大)

## 4️⃣ Fourier频域消融实验 (run_fourier_ablation_fixed.py) 🧪 **NEW**

### 🔬 核心功能
自动化的Fourier频域消融实验框架，支持完整的消融研究流程，包括模型加载、实验执行、结果分析和可视化生成。

### 消融模式支持
- **FULL**: 完整模型（原始+低频+高频融合）
- **LOW_FREQ_ONLY**: 仅使用低频增强特征（长期趋势）
- **HIGH_FREQ_ONLY**: 仅使用高频增强特征（短期变化）

### 基础用法
```bash
# 基础消融实验
python scripts/run_fourier_ablation_fixed.py \
    --model_path exp/your_experiment/best_model.pth \
    --output_dir exp/fourier_ablation \
    --device auto

# 包含可视化的完整消融实验
python scripts/run_fourier_ablation_fixed.py \
    --model_path exp/your_experiment/best_model.pth \
    --output_dir exp/fourier_ablation_with_plots \
    --generate_plots \
    --device cuda
```

### 🎯 实验流程
1. **自动参数加载**: 从模型目录的`args.txt`自动加载训练参数
2. **模型和数据初始化**: 加载训练好的模型和多领域数据集
3. **三模式消融实验**: 依次运行FULL、LOW_FREQ_ONLY、HIGH_FREQ_ONLY模式
4. **性能评估**: 使用标准评估指标（NDCG@10、Hit@10）
5. **结果保存**: JSON格式保存详细结果和性能对比
6. **可视化生成**: 生成期刊级质量的消融对比图表

### 🔧 核心参数
- `--model_path`: 训练好的模型检查点路径（必需）
- `--output_dir`: 实验结果输出目录（必需）
- `--device`: 计算设备选择 (auto, cuda, cpu)
- `--generate_plots`: 是否生成可视化图表

### 📊 输出文件
```
exp/fourier_ablation/
├── ablation_results.json                   # 完整消融实验结果
├── intermediate_results.json               # 中间结果（实验过程保存）
└── domain_frequency_trend_comparison.png   # 可视化图表 (if --generate_plots)
```

### 📈 结果分析内容
#### JSON结果结构
```json
{
  "full": {
    "ndcg@10": 0.4523,
    "hit@10": 0.6234,
    "beauty_ndcg@10": 0.4321,
    "games_ndcg@10": 0.4567,
    "movielens_ndcg@10": 0.4681
  },
  "low_only": {
    "ndcg@10": 0.3876,
    "hit@10": 0.5423,
    "..."
  },
  "high_only": {
    "ndcg@10": 0.3123,
    "hit@10": 0.4876,
    "..."
  }
}
```

#### 自动生成的分析摘要
- **整体性能对比**: 三种模式的综合指标对比
- **相对性能计算**: 相对于完整模型的性能变化百分比
- **领域特异性分析**: 各领域在不同消融模式下的表现差异
- **频域贡献度评估**: 低频和高频成分对推荐性能的贡献度

### 🎨 可视化特性
生成的消融对比图表包含：
- **双指标展示**: NDCG@10和Hit@10并排对比
- **领域趋势线**: Beauty、Games、MovieLens三条趋势线
- **数值标注**: 每个数据点的精确数值显示
- **期刊级质量**: 300 DPI PNG格式，适合论文发表
- **专业配色**: 领域特异性配色方案，视觉清晰

### 🔍 研究价值
#### 频域成分贡献分析
- **低频成分影响**: 评估长期用户偏好对推荐的贡献
- **高频成分影响**: 评估短期行为变化对推荐的贡献
- **协同效应验证**: 验证低频+高频融合的必要性

#### 领域差异发现
- **领域敏感性**: 识别对频域分解最敏感的领域
- **适应性验证**: 验证Fourier模块的领域适应能力
- **优化指导**: 为不同领域的模型优化提供数据支持

### 🚨 使用注意事项
1. **模型兼容性**: 要求模型具有`enhanced_rating_module`和消融模式支持
2. **内存要求**: 消融实验需要足够的GPU/CPU内存
3. **时间成本**: 完整消融实验需要较长时间（视数据集大小而定）
4. **参数自动加载**: 脚本会自动从模型目录加载训练参数，无需手动配置

## 🔄 向下兼容

所有脚本完全兼容旧的单编码器模型：

```bash
# 对于旧模型，脚本会自动检测并使用单编码器模式
python scripts/cli.py \
    --experiment_dir exp/old_sasrec_model \
    --user_id 123 \
    --output_dir exp/legacy_analysis
```

## 🚨 注意事项

1. **领域ID范围**: 当前支持3个领域 (0: beauty, 1: games, 2: ml-1m)
2. **平衡采样**: 系统自动确保每个领域获得相等的用户数量以进行公平对比
3. **内存使用**: 大规模批量分析时请注意内存使用
4. **GPU支持**: 所有分析都在CPU上进行，无需GPU（消融实验支持GPU加速）
5. **输出格式**: 支持PNG、PDF、SVG格式的图表输出
6. **消融实验要求** 🧪:
   - 模型必须具有`enhanced_rating_module`和消融模式支持
   - 需要训练时保存的`args.txt`文件用于参数加载
   - 完整消融实验耗时较长，建议使用GPU加速
   - 中间结果自动保存，支持实验中断后恢复

## 📖 更多示例

查看`exp/`目录下的示例输出，了解各种分析结果的格式和内容。

---

**最后更新**: 2024-01-XX  
**版本**: v2.1 - Multi-Domain Support + FFT Magnitude Spectrum Comparison ⭐
