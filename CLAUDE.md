# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

CMREC 是一个多领域推荐系统框架，实现了基于混合专家 (MoE) 架构的层次化自适应门控推荐模型。该项目专注于解决跨领域推荐中的知识协同与冲突问题，并通过细粒度偏好感知提升推荐效果。

## 核心架构

- **主模型**: HAGMRec - 层次化自适应门控混合专家推荐模型
- **基础框架**: 基于 SASRec (Self-Attentive Sequential Recommendation) 的 Transformer 架构
- **核心创新**: 在 Transformer 块中用 MoE FFN 替换传统前馈网络
- **专家分工**: 共享专家学习通用模式，领域专家学习特定领域模式

## 关键组件

### 模型架构 (keys/ 目录)
- `keys/model.py`: HAGMRec主模型定义，包含Transformer块和MoE集成，支持多种增强功能
- `keys/c_moe.py`: MoE实现，包括HAGMoEFFN和PointWiseFeedForward模块，支持专家路由策略
- `keys/utils.py`: 数据处理工具、评估指标计算、采样函数等核心工具
- `keys/rating_modules.py`: 增强评分嵌入模块，支持多种评分策略(simple/fourier等)

### 训练和实验基础设施
- `main.py`: 主训练脚本，支持多领域训练、推理、WandB集成和可视化
- `test_system.py`: 系统功能测试，验证模型组件正确性
- `test_visualization.py`: 可视化测试，生成专家路由热力图和t-SNE分析
- `process_datasets.py`: 数据集预处理和统计信息生成
- `data_process.py`: 数据处理辅助脚本

## 常用命令

### 环境设置
```bash
# 确保Python 3.11+和PyTorch 2.6.0+已安装
python --version  # 应显示 Python 3.11+
pip install torch numpy pandas matplotlib seaborn scikit-learn wandb tensorboard tqdm
```

### 数据预处理
```bash
python process_datasets.py  # 预处理数据集并生成统计信息
python data_process.py      # 数据处理辅助工具
```

### 训练模型
```bash
python main.py --train_dir train_1 \
    --batch_size 128 \
    --lr 0.001 \
    --maxlen 100 \
    --hidden_units 64 \
    --num_blocks 2 \
    --num_heads 2 \
    --num_epochs 20 \
    --use_datasets beauty_rated games ml-1m_rated \
    --use_moe true \
    --use_domain_info true \
    --use_context_attention true \
    --use_rating_emb true \
    --moe_routing_strategy shared_base
```

### 推理模式
```bash
python main.py --train_dir inference \
    --inference_only true \
    --state_dict_path exp/[experiment_dir]/[model_file].pth \
    --use_datasets beauty_rated games ml-1m_rated
```

### 系统测试和调试
```bash
python test_system.py        # 基本功能测试，验证模型组件
python test_visualization.py # 可视化测试，生成分析图表
python gradient_check.py     # 梯度检查和调试
```

### 使用示例脚本
```bash
bash examples_enhanced_rating.sh  # 增强评分策略使用示例
```

### 检查实验结果
```bash
# 查看实验目录结构
ls exp/[dataset_combination]_[train_dir]/
# 查看训练日志
cat exp/[experiment_dir]/log.txt
# 查看实验参数
cat exp/[experiment_dir]/args.txt
```

## 数据集

支持的数据集位于 `data/` 目录:
- `beauty_rated.txt`: 美妆产品评分数据
- `games.txt`: 游戏评分数据  
- `ml-1m_rated.txt`: MovieLens 1M 评分数据
- `ml-100k.txt`: MovieLens 100K 数据
- `Video.txt`, `Steam.txt`, `wikipedia.txt`: 其他领域数据

## 实验管理

### 实验目录结构
```
exp/
├── [dataset1-dataset2-dataset3]_[train_dir]/
│   ├── args.txt          # 实验参数
│   ├── log.txt           # 训练日志
│   ├── logs/             # TensorBoard 日志
│   └── *.pth             # 模型权重文件
```

### 重要参数详解
- `--use_datasets`: 指定训练数据集列表，支持多数据集联合训练
- `--use_moe`: 启用/禁用混合专家架构 (true/false)
- `--moe_routing_strategy`: 路由策略选择
  - `vanilla`: 标准MoE路由
  - `shared_base`: 共享基础专家策略 (推荐)
- `--use_domain_info`: 启用领域信息整合
- `--use_context`: 启用上下文信息机制
- `--use_rating_emb`: 启用评分嵌入功能
- `--rating_strategy`: 评分策略选择
  - `simple`: 传统简单评分嵌入
  - `fourier`: 傅里叶长短期特征提取 (推荐)
  - `legacy`: 向后兼容模式
- `--use_gated_fusion`: 启用门控融合机制
- `--use_specialization_loss`: 启用专业化损失
- `--use_contrastive_loss`: 启用对比学习损失
- `--gate_temperature`: 门控温度调节参数

## 依赖关系

主要依赖包括:
- PyTorch 2.6.0+
- NumPy
- Pandas
- Matplotlib/Seaborn (可视化)
- Scikit-learn (评估指标)
- WandB (实验跟踪)
- TensorBoard (日志可视化)
- tqdm (进度条)

## 可视化和监控

- **WandB 集成**: 自动记录训练指标、损失曲线和专家使用情况
- **TensorBoard**: 专家负载分布、领域-专家路由热力图
- **t-SNE 可视化**: 专家专业化分析
- **控制台输出**: 彩色格式化的训练进度和评估结果

## 评估指标

模型支持多种推荐评估指标:
- NDCG@10, NDCG@20: 归一化累计折扣增益
- HR@10, HR@20: 命中率
- MRR: 平均倒数排名
- 按领域分别计算和整体性能

## 开发注意事项

1. **设备兼容性**: 代码支持 CUDA 和 CPU，自动检测可用设备
2. **随机种子**: 使用 `--seed` 参数确保实验可重现
3. **内存管理**: 大批量训练时注意 GPU 内存使用
4. **数据格式**: 确保数据集格式正确 (用户ID, 物品ID, 评分)
5. **模型保存**: 最佳模型自动保存到实验目录

## 外部服务集成

### ArXiv MCP Server
项目已集成全局arxiv-mcp-server用于学术论文检索和引用支持：

**安装方式：**
```bash
uv tool install arxiv-mcp-server
```

**MCP配置文件位置：** 项目根目录的 `.mcp.json`
```json
{
  "mcpServers": {
    "arxiv-mcp-server": {
      "command": "uv",
      "args": [
        "tool",
        "run", 
        "arxiv-mcp-server"
      ],
      "env": {
        "ARXIV_STORAGE_PATH": "C:\\Users\\24153\\.arxiv-mcp-server\\papers"
      }
    }
  }
}
```

**主要功能：**
- `search_papers`: 搜索ArXiv论文，支持按类别、日期筛选
- `download_paper`: 下载论文PDF到本地存储
- `read_paper`: 读取已下载论文的内容
- `list_papers`: 列出本地存储的论文

**使用场景：**
- 搜索推荐系统、MoE架构、序列推荐相关论文
- 获取最新多领域推荐研究进展
- 为论文写作提供引用和对比分析
- 跟踪HAGMRec相关领域的研究动态

**示例用法：**
```python
# 搜索相关论文
search_papers({
  "query": "mixture of experts recommendation transformer",
  "max_results": 10,
  "categories": ["cs.IR", "cs.LG", "cs.AI"],
  "date_from": "2023-01-01"
})
```

## 论文相关

该项目对应的学术论文描述了层次化自适应门控MoE架构在多领域推荐系统中的应用，重点解决了知识协同与冲突以及细粒度偏好感知的问题。相关文档见 `md/` 目录。