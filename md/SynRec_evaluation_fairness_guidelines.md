### SynRec 公平评测指南（与 SynRec_revision_plan 对齐）

#### 目标与范围
- 统一并公开评测协议，确保方法间对比公平、可复现；
- 面向多域序列推荐（全局候选/域内候选）、采样评测与全库评测两种范式；
- 适用于本仓库的 `main.py` 与 `keys/utils.py` 的评测实现。

---

## 一、推荐评测协议（主+辅）
- 主协议（更严格，优先报告）：
  - 全库评测 + 全局候选（所有域的物品作为候选）
  - 指标：NDCG@{5,10,20}、HR/Hit@{5,10,20}、MRR@{5,10,20}
  - 聚合：分域指标 + Macro 平均（域均值）+ Micro 平均（全测试事件加权平均）
  - 运行示例：
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

- 辅助协议（补充视角）：
  1) 全库评测 + 域内候选（仅测试域内物品）
```bash
python main.py \
  --train_dir eval_full_domain \
  --use_datasets beauty_5_5 games_5_5 ml-1m_5_5 \
  --inference_only true \
  --state_dict_path exp/.../best.pth \
  --full_ranking_eval true \
  --use_domain_sampling_for_evaluation true \
  --eval_item_batch_size 4096
```
  2) 采样评测（1正+N负，N∈{100,1000}），全局与域内各一组：
```bash
python main.py \
  --train_dir eval_sampled \
  --use_datasets beauty_5_5 games_5_5 ml-1m_5_5 \
  --inference_only true \
  --state_dict_path exp/.../best.pth \
  --full_ranking_eval false \
  --use_domain_sampling_for_evaluation false
```

---

## 二、数据切分与防泄漏
- 切分：留一法（每用户倒数第1为测试，第2为验证，其余训练）。
- 候选过滤：评测时过滤用户历史已交互物品（训练/验证/测试历史均过滤）。
- 因果性：
  - 主干自注意力使用下三角掩码；
  - 频域编码器仅用历史前缀（已在实现中加入因果掩码，并对 padding 最终置零）。
- ID 唯一性：多域统一全局 ID（已采用 offset 策略），防止跨域冲突引入泄漏。

---

## 三、候选与采样公平性
- 全库 vs 采样：
  - 首选全库评测作为主结果；
  - 采样评测作为补充，必须在所有方法中使用同一随机种子与相同负例集合（"fixed negatives"）。
- 域内 vs 全局候选：
  - 全局候选更严格，域内候选刻画“域内可用性”；两者都建议报告。
- 负采样策略（采样评测）：
  - 统一随机负采样（或统一的受欢迎度采样），并固定 N 值；
  - 建议报告 N∈{100,1000} 的稳定性对比。

---

## 四、聚合与显著性
- 域内指标：每个域分别报告 NDCG/HR/MRR@{5,10,20}。
- Macro 平均：各域指标算术平均；Micro 平均：按测试量加权平均。
- 显著性与稳健性：
  - 多随机种子（≥5），报告均值±标准差；
  - 配对 t 检验或自举置信区间（p<0.01），对比主方法与最强对手；
  - 对 K 值（5/10/20）、采样 N（100/1000）做敏感性分析。

---

## 五、效率与实际性
- 报告参数量、训练吞吐（samples/s）、推理时延（ms/req）、显存占用。
- 与无 MoE、Vanilla MoE、SharedBase w/o Rating 进行效率对比。

---

## 六、冷启动与分桶评测（建议）
- 用户序列长度分桶（短/中/长），观察不同分桶的收益差异；
- 长尾物品与新物品子集（若可构造时间切分）评测；
- 零/少样本跨域：两域训练 + 第三域零样本/少样本微调评测。

---

## 七、复现清单（Checklist）
- [ ] 固定随机种子；
- [ ] 统一候选与负采样集合（采样评测时）；
- [ ] 报告全库（全局/域内）与采样两类结果；
- [ ] 过滤历史交互；
- [ ] 因果掩码（主干与频域）与 padding 置零；
- [ ] 多种子显著性与多 K、多 N 稳健性；
- [ ] Macro/Micro 聚合；
- [ ] 效率指标；
- [ ] 记录脚本/参数到 exp/<...>/args.txt 与 log.txt。

---

## 八、与代码参数映射
- `--full_ranking_eval`：true → 全库评测；false → 采样评测（1正+N负）
- `--use_domain_sampling_for_evaluation`：true → 域内候选；false → 全局候选
- `--eval_item_batch_size`：全库评测候选分批打分的批大小（如 4096）
- 指标输出：`exp/<datasets>_<train_dir>/log.txt`，含分域与overall（Macro 可由分域均值计算；Micro 需从原始统计生成）

---

## 九、论文报告建议（落到 SynRec.tex）
- 在“评测协议”小节明确上述主+辅协议与参数；
- 清晰说明候选集设置、负采样规模与是否固定负例；
- 报告 Macro 与 Micro 平均、显著性检验与稳健性分析；
- 解释域内与全局候选的意义，并两者皆给结果。


