### SynRec 评测参数与使用指南（对应 SynRec_revision_plan 中 A-评测项）

#### 新增与关键参数
- `--full_ranking_eval`（bool，默认 false）
  - 启用“全库评测”。对每个样本将真实物品与整个候选集合进行排序计算排名。
  - 若为 false，则使用“采样评测”（1 正 + N 负）。

- `--use_domain_sampling_for_evaluation`（bool，默认 false）
  - 评测候选是否限定在“域内物品集合”。
  - true：候选仅为该域的物品；false：候选为全局所有物品。

- `--eval_item_batch_size`（int，默认 4096）
  - 全库评测时对候选物品分批打分的批大小，避免显存溢出。

（相关实现：`keys/utils.py:evaluate_batched`，`main.py:parse_args`）

#### 典型用法

- 采样评测（默认 1 正 + 100 负），全局候选：
```bash
python main.py \
  --train_dir eval \
  --use_datasets beauty_5_5 games_5_5 ml-1m_5_5 \
  --inference_only true \
  --state_dict_path exp/beauty_5_5-games_5_5-ml-1m_5_5_eval/SASRec.epoch=200.lr=0.001.layer=2.head=2.hidden=64.maxlen=100.pth \
  --full_ranking_eval false \
  --use_domain_sampling_for_evaluation false
```

- 采样评测，域内候选：
```bash
python main.py \
  --train_dir eval_domain \
  --use_datasets beauty_5_5 games_5_5 ml-1m_5_5 \
  --inference_only true \
  --state_dict_path exp/beauty_5_5-games_5_5-ml-1m_5_5_eval_domain/SASRec.epoch=200.lr=0.001.layer=2.head=2.hidden=64.maxlen=100.pth \
  --full_ranking_eval false \
  --use_domain_sampling_for_evaluation true
```

- 全库评测，域内候选，分批打分：
```bash
python main.py \
  --train_dir eval_full_domain \
  --use_datasets beauty_5_5 games_5_5 ml-1m_5_5 \
  --inference_only true \
  --state_dict_path exp/beauty_5_5-games_5_5-ml-1m_5_5_eval_full_domain/SASRec.epoch=200.lr=0.001.layer=2.head=2.hidden=64.maxlen=100.pth \
  --full_ranking_eval true \
  --use_domain_sampling_for_evaluation true \
  --eval_item_batch_size 4096
```

- 全库评测，全局候选（最严格设置）：
```bash
python main.py \
  --train_dir eval_full_global \
  --use_datasets beauty_5_5 games_5_5 ml-1m_5_5 \
  --inference_only true \
  --state_dict_path exp/beauty_5_5-games_5_5-ml-1m_5_5_eval_full_global/SASRec.epoch=200.lr=0.001.layer=2.head=2.hidden=64.maxlen=100.pth \
  --full_ranking_eval true \
  --use_domain_sampling_for_evaluation false \
  --eval_item_batch_size 4096
```

#### 输出与记录
- 评测函数会输出分域与总体的 `NDCG@5/10`、`HT@5/10`、`MRR@5/10`；
- 结果会写入 `exp/<datasets>_<train_dir>/log.txt`；
- 建议在论文中同时报告采样评测与全库评测，特别是“域内/全局候选”的差异与公平性说明。

#### 与计划（plan）对齐的结论
- 已完成 `SynRec_revision_plan.md` 中 A-2 项“评测协议扩展与严格化”：
  - 支持全库评测与采样评测两种范式；
  - 支持域内/全局候选切换；
  - 提供大候选分批打分能力，便于复现实验与论文合规报告。


