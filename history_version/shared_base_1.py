
# --- 新策略: 共享专家作为基础，门控选择一个领域专家 ---

# 2.1. 共享专家输出 (平均)
shared_output = torch.stack(shared_expert_outputs_list, dim=0).mean(dim=0)

# 2.2. 门控仅对领域专家生效
gate_logits = self.gate(gate_input)
gate_scores = F.softmax(gate_logits, dim=-1) # (batch*seq, num_domain_experts)

# 2.3. 选择一个领域专家
selected_domain_indices = torch.argmax(gate_scores, dim=1)

# 2.4. 提取被选中的领域专家输出
all_domain_outputs = torch.stack(domain_expert_outputs_list, dim=1) # (batch*seq, num_domain_experts, hidden)
selected_domain_output = torch.gather(all_domain_outputs, 1, selected_domain_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.hidden_units)).squeeze(1)