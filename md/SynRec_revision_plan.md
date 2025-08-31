### SynRec 代码-论文一致性与审稿风险整改路线

#### 目标
- 明确实现与论文描述的差异与潜在退稿风险，形成分阶段、可验收的修复清单。
- 先修正评测与路由一致性等高风险问题，再补充公平基线与文本澄清，最终提升稿件把关质量。

#### 最高优先级问题（需尽快修复）
- 频域路由未用于推理：当前 `predict` 不传入 `rating_seqs`，导致推理时门控不受频域特征影响，与论文“频率引导路由”核心卖点不一致。
- 评测为采样而非全库：代码使用 1 正 + N 负的采样评测（默认 N=100），与论文“全库排序”表述不一致，易造成指标虚高与公平性争议。
- 频域侧可能时间泄漏：`OptimizedFourierRatingEncoder` 的双分支注意力未使用因果掩码，门控证据信息可能看到未来评分，违背自回归设定。

#### 行动计划（分组与验收标准）

## A. 实现与评测一致性
- 同步推理端频域路由（高优先级）
  - 修改 `keys/model.py: SynRec.predict` 支持 `rating_seqs` 与 `domain_ids`，与训练一致调用 `TemporalEnhancedRatingModule`。
  - 在评测数据管线中加入评分前缀：`EvalDataset`/`eval_collate_fn` 返回 `rating_seq`（去掉末位，与 `seq` 对齐）。
  - 验收：启用/禁用频域路由时推理结果有一致变化；与训练路径的特征使用严格一致。

- 评测协议扩展与严格化（高优先级）
  - 新增开关：`--full_ranking_eval`（全库评测）与 `--use_domain_sampling_for_evaluation`（域内候选）。
  - 全库评测实现：分批计算 `item_embs` 矩阵乘法，支持内存友好 Top-K。
  - 采样评测说明：明确负例数 N、是否域内采样、是否过滤训练交互；在论文中同步说明。
  - 验收：同一模型下，全库与采样评测的结果曲线、不同 N（100/1000）的稳定性报告。

- 频域因果掩码与前缀一致性（高优先级）
  - 在 `OptimizedFourierRatingEncoder` 的 `long_term_attention` / `short_term_attention` 中加入下三角因果掩码（与主干一致）。
  - 明确仅对“历史前缀”构建频域证据，必要时采用滑动窗口；确保任何时间步不访问未来评分。
  - 验收：单测/可视化验证门控所用的频域证据严格来自历史；训练与推理指标稳定、不过拟合于泄漏。

## B. 公平基线与稳健性
- 多域协同训练的公平基线（高优先级）
  - Baseline-1：`SharedBottom`（禁用 MoE，`use_moe=false`，共享 Transformer 底座）。
  - Baseline-2：`Vanilla MoE`（`moe_routing_strategy=vanilla`，Top-K 从全体专家选择）。
  - Baseline-3：`SharedBase w/o Rating`（保留共享+领域专家+meta gate，但关闭 `use_rating_emb/use_gated_fusion`）。
  -（可选）Baseline-4：简化 `PLE/STAR` 风格（若时间允许）。
  - 验收：所有方法在“单域训练”和“多域协同训练”两种范式下均评测，报告协同增益差值与统计显著性。

- 统计显著性与稳健性
  - 多随机种子（≥5），报告均值±标准差；配对 t 检验或自举置信区间（p<0.01）。
  - K 值敏感性（@5/@10/@20），不同负采样 N 的稳定性曲线。
  - 验收：曲线与表格入文，显著性结论明确、可复现。

## C. 效率与资源报告
- 参数量、训练吞吐（samples/s）、推理时延（ms/request）、显存占用。
- 对比无 MoE / Vanilla MoE / 本方法；展示负载均衡对专家利用的改善。
- 验收：固定批大小/硬件的可复现实验，结果入文。

## D. 论文文本与相关工作修订
- 评测协议与设定澄清（重要）
  - 明确本文报告的评测为“采样评测（1正+N负）”与/或“全库评测”，负采样规模、是否域内候选、过滤策略全部交代。
  - 问题设定：定位为“多数据源联合训练（默认不同域用户不重叠）”，并在附录给出“少量重叠用户”的小实验，连接到经典 CDR。

- 频域方法与实现细节对齐
  - 降低“theoretical validation”等措辞，改为“机制性与实证性验证”。
  - 对位说明：实现中的 `domain_emb + MLP(domain_emb)`、`rating_gate` 简化融合与论文公式的关系；`W_g^{meta}` 记号统一；`f_j^{domain}` 明确计算口径（按 token 或 batch 归一）。
  - 评分与隐式任务：解释“主任务为隐式 BCE，但频域模块使用显式评分作为路由证据”的合理性；给出无评分场景的替代信号（停留时长/强反馈次数等）。

- 相关工作补充
  - 增补频域/滤波增强的序列推荐代表作，并逐点对位差异：我们用于“路由决策与专家分工”，具备可学习分界与可解释可视化。

## E. 扩展实验（加分项）
- 稀疏/冷启动：短序列用户、长尾物品；
- 零/少样本跨域：两域训练+另一域零样本或少样本微调；
- 验收：对应子表与分析入文。

---

### 实施细节—代码定位与修改建议
- 频域路由用于推理
  - `keys/model.py`
    - `SynRec.predict(self, user_ids, log_seqs, item_indices, domain_ids, rating_seqs=None)`：签名加 `rating_seqs`，与 `log2feats` 一致传递。
  - `keys/utils.py`
    - `EvalDataset.__getitem__`：返回 `rating_seq`（去掉末位，与 `seq` 对齐）。
    - `eval_collate_fn`：新增打包 `rating_seqs`；
    - `evaluate_batched`：调用 `model.predict(..., rating_seqs=...)`。

- 全库评测与采样评测开关
  - `main.py`: 新增 `--full_ranking_eval`；
  - `keys/utils.py: evaluate_batched`：
    - 若全库：按域（或全局）候选矩阵分批打分；
    - 若采样：保留现有 1+N 负采样逻辑，参数化 N 与域内/全库候选。

- 频域因果掩码
  - `keys/temporal_rating_modules.py`
    - 在 `long_term_attention`/`short_term_attention` 前构造下三角 mask：
      - `batch_first=True` 时 `attn_mask` 形状 `(L, L)`，对 batch 广播；
      - 确保 rating 前缀与 `seq` 一致裁剪。

- 公平基线配置
  - `SharedBottom`：`--use_moe=false`（共享 Transformer，其他保持一致）。
  - `Vanilla MoE`：`--use_moe=true --moe_routing_strategy=vanilla`。
  - `SharedBase w/o Rating`：`--use_moe=true --moe_routing_strategy=shared_base --use_rating_emb=false --use_gated_fusion=false`。
  -（可选）PLE/STAR：视时间再行实现或引用。

- 统计显著性与效率
  - 新增脚本 `scripts/analyze_results.py`：聚合多随机种子日志、计算显著性与方差；
  - `main.py`：保留参数量统计，补充简单的吞吐/延时度量（epoch 内滑动平均）。

### 论文修订锚点（建议在 `SynRec.tex` 中增改）
- 方法节：
  - 增加“频域因果掩码与前缀一致性”的小节与伪码；
  - 公式与实现的符号统一（`W_g^{meta}` 等）。
- 实验节：
  - “评测协议”小节明确候选集设置、负采样规模、是否全库评测；
  - “公平基线”小节新增多域协同的对比方法与实现细节；
  - “稳健性与显著性”小节汇报多种子/多 K/多负采样规模结果。
- 讨论节：
  - 降低过强措辞（把 “theoretical validation” 改为 “机制性与实证性验证”）。

### 交付物清单与验收
- 代码：上述改动的最小可运行版本（含全库评测开关、推理频域路由、因果掩码）。
- 脚本：`scripts/analyze_results.py`；
- 文档：本整改路线、`README` 中新增评测与配置说明；
- 论文：方法与实验的同步修订稿。

### 后续里程碑（建议）
- M1（优先级最高）：A 组全部完成；
- M2：B/C 组完成并补充到论文；
- M3：E 组加分项按时间推进。


