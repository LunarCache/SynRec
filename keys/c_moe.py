import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate, args=None):
        super(PointWiseFeedForward, self).__init__()

        self.use_moe = getattr(args, 'use_moe', False)
        if self.use_moe and args is not None:
            self.moe_ffn = HAGMoEFFN(hidden_units, args)
        else:
            # Original FFN implementation
            self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
            self.dropout1 = torch.nn.Dropout(p=dropout_rate)
            self.relu = torch.nn.ReLU()
            self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
            self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs, log_feats, domain_ids=None, rating_embedding=None):
        if self.use_moe:
            # Our new MoE FFN handles everything internally
            return self.moe_ffn(inputs, log_feats, domain_ids, rating_embedding)
        else:
            # Original FFN logic
            outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
            outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
            outputs += inputs
            # Return a dict with zero losses and empty viz_data for compatibility
            return outputs, {}, {}

class HAGMoEFFN(nn.Module):
    def __init__(self, hidden_units, args):
        super(HAGMoEFFN, self).__init__()
        self.args = args # Store args
        self.hidden_units = hidden_units
        self.num_domain_experts = getattr(args, 'num_domains')
        self.num_experts = getattr(args, 'moe_num_experts')
        self.k = getattr(args, 'moe_k')
        
        self.routing_strategy = getattr(args, 'moe_routing_strategy', 'vanilla')

        # 参数验证
        if self.num_experts < self.num_domain_experts:
            raise ValueError(f"moe_num_experts ({self.num_experts}) must be >= num_domains ({self.num_domain_experts})")
        
        self.num_shared_experts = max(1, self.num_experts - self.num_domain_experts)
        
        # 发出警告如果配置不理想 (仅限vanilla策略)
        if self.routing_strategy == 'vanilla':
            if self.k > self.num_experts:
                print(f"Warning: moe_k ({self.k}) > num_experts ({self.num_experts}), will use all experts")
            
            if self.k > 1 and (self.k - 1) > self.num_domain_experts:
                print(f"Warning: moe_k-1 ({self.k-1}) > num_domain_experts ({self.num_domain_experts}), "
                      f"will only use {self.num_domain_experts} domain experts")
        
        # --- 专家网络 ---
        self._init_experts(args)

        # --- 门控网络 ---
        if self.routing_strategy == 'shared_base':
            # 门控网络只对领域专家生效
            self.gate = nn.Linear(hidden_units, self.num_domain_experts)
            # --- Meta-Gate for adaptive fusion ---
            self.meta_gate = nn.Linear(hidden_units, 2)
        else:
            # 保持原有逻辑
            self.gate = nn.Linear(hidden_units, self.num_experts)
        
        # --- 领域感知 ---
        self.use_domain_info = getattr(args, 'use_domain_info')
        if self.use_domain_info:
            self.domain_embedding = nn.Embedding(self.num_domain_experts, hidden_units)
            # 增强领域信息表达能力的线性变换层
            self.domain_transform = nn.Sequential(
                nn.Linear(self.hidden_units, self.hidden_units * 2),
                nn.ReLU(),
                nn.Linear(self.hidden_units * 2, self.hidden_units)
            )
        
        # Enhanced rating-aware gating components (简化版)
        if getattr(args, 'use_gated_fusion', False):
            # 保留基础的rating_gate用于简单融合
            self.rating_gate = nn.Sequential(
                nn.Linear(self.hidden_units, self.hidden_units),
                nn.Sigmoid()
            )

        # --- 负载均衡 ---
        self.load_balancing = getattr(args, 'moe_load_balancing')
        if self.load_balancing:
            self.balance_loss_weight = getattr(args, 'moe_balance_loss_weight')
        
        # --- Expert Specialization Optimization ---
        self.use_specialization_loss = getattr(args, 'use_specialization_loss', False)
        self.specialization_weight = getattr(args, 'specialization_weight', 0.1)
        
        self.use_contrastive_loss = getattr(args, 'use_contrastive_loss', False)
        self.contrastive_weight = getattr(args, 'contrastive_weight', 0.05)

    def _create_expert(self, hidden_units, args):
        # MLP expert with Swish activation (better for Transformer architectures)
        return nn.Sequential(
            nn.Linear(hidden_units, hidden_units * 2),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_units * 2, hidden_units),
            nn.Dropout(p=args.dropout_rate)
        )

    def _init_experts(self, args):
        self.shared_experts = nn.ModuleList([self._create_expert(self.hidden_units, args) for _ in range(self.num_shared_experts)])
        self.domain_experts = nn.ModuleList([self._create_expert(self.hidden_units, args) for _ in range(self.num_domain_experts)])


    def cv_squared(self, x):
        eps = 1e-10
        if x.dim() == 1: x = x.unsqueeze(0)
        return x.var() / (x.mean()**2 + eps)
    
    def compute_specialization_loss(self, gate_logits, domain_ids):
        """计算专业化损失：鼓励领域i主要使用专家i"""
        if domain_ids is None or not self.use_specialization_loss:
            return torch.tensor(0.0, device=gate_logits.device)
        
        # gate_logits: (batch*seq, num_domain_experts)
        # domain_ids: (batch,) -> 需要扩展到 (batch*seq,)
        batch_size = domain_ids.shape[0]
        seq_len = gate_logits.shape[0] // batch_size
        domain_ids_flat = domain_ids.unsqueeze(1).expand(-1, seq_len).reshape(-1)
        
        # 创建理想的专家分布（one-hot，领域i对应专家i）
        ideal_distribution = F.one_hot(domain_ids_flat, num_classes=self.num_domain_experts).float()
        
        # 计算当前门控分布与理想分布的KL散度
        kl_loss = F.kl_div(
            F.log_softmax(gate_logits, dim=-1), 
            ideal_distribution, 
            reduction='batchmean'
        )
        
        return kl_loss
    
    def compute_contrastive_loss(self, expert_outputs, domain_ids):
        """对比学习：让同领域的表示更相似，不同领域的表示更不同"""
        if domain_ids is None or not self.use_contrastive_loss:
            return torch.tensor(0.0, device=expert_outputs.device)
        
        batch_size, seq_len, hidden_dim = expert_outputs.shape
        # 正确扩展domain_ids到与expert_outputs_flat匹配的形状
        domain_ids_flat = domain_ids.unsqueeze(1).expand(-1, seq_len).reshape(-1)
        expert_outputs_flat = expert_outputs.view(-1, hidden_dim)
        
        # 计算领域中心
        unique_domains = torch.unique(domain_ids_flat)
        domain_centers = {}
        
        for domain_id in unique_domains:
            domain_mask = (domain_ids_flat == domain_id)
            if domain_mask.sum() > 1:
                domain_outputs = expert_outputs_flat[domain_mask]
                domain_centers[domain_id.item()] = domain_outputs.mean(dim=0)
        
        # 计算对比损失
        contrastive_loss = torch.tensor(0.0, device=expert_outputs.device)
        if len(domain_centers) > 1:
            centers_list = list(domain_centers.values())
            for i in range(len(centers_list)):
                for j in range(i + 1, len(centers_list)):
                    # 减少不同领域中心的相似性
                    similarity = F.cosine_similarity(centers_list[i], centers_list[j], dim=0)
                    contrastive_loss += similarity
        
        return contrastive_loss
    
    def forward(self, inputs, log_feats, domain_ids=None, rating_embedding=None):
        if domain_ids is not None and isinstance(domain_ids, np.ndarray):
            domain_ids = torch.LongTensor(domain_ids).to(inputs.device)
            
        batch_size, seq_len, _ = inputs.shape
        inputs_flat = inputs.view(-1, self.hidden_units)

        # --- 1. Construct Inputs ---
        # A. Construct Gate Input (for routing decisions)
        gate_input = inputs_flat
        domain_emb_expanded = None
        if self.use_domain_info and domain_ids is not None:
            domain_ids_expanded = domain_ids.unsqueeze(1).expand(-1, seq_len).reshape(-1)
            domain_emb_expanded = self.domain_embedding(domain_ids_expanded)
            gate_input = gate_input + self.domain_transform(domain_emb_expanded) + domain_emb_expanded
        # B. Construct Content Input (for expert processing)
        content_input = inputs_flat

        # Enhanced Rating-Aware Gating (简化版)
        if getattr(self.args, 'use_gated_fusion', False) and rating_embedding is not None:
            rating_emb_flat = rating_embedding.view(-1, self.hidden_units)
            
            # 简单的门控融合，不再调制门控网络本身
            gate_weight = self.rating_gate(gate_input)
            gate_input = gate_input + gate_weight * rating_emb_flat

        # --- 2. Expert Forward Pass ---
        # This is where the core logic of hybrid input and heterogeneous experts resides
        shared_expert_outputs_list = []
        domain_expert_outputs_list = []

        shared_expert_outputs_list = [expert(content_input) for expert in self.shared_experts]
        domain_expert_outputs_list = [expert(content_input) for expert in self.domain_experts]

        # --- 3. Gating and Fusion ---
        # This part remains the same, driven by the `gate_input`
        loss_dict = {}
        viz_data = {}

        if self.routing_strategy == 'shared_base':
            # --- 新策略: 共享专家作为基础，门控选择一个领域专家 ---
            
            # 2.1. 共享专家输出 (平均)
            shared_output = torch.stack(shared_expert_outputs_list, dim=0).mean(dim=0)

            # 2.2. 门控仅对领域专家生效
            gate_logits = self.gate(gate_input)
            gate_scores = F.softmax(gate_logits, dim=-1)
            
            # (batch*seq, num_domain_experts)

            # 2.3. 对所有领域专家进行加权融合
            all_domain_outputs = torch.stack(domain_expert_outputs_list, dim=1) # (batch*seq, num_domain_experts, hidden)
            weighted_domain_output = torch.einsum('be,beh->bh', gate_scores, all_domain_outputs)
            
            # 2.4. (新) Meta-Gate: 动态融合共享和领域专家
            meta_gate_logits = self.meta_gate(gate_input)
            meta_gate_weights = F.softmax(meta_gate_logits, dim=-1) # (batch*seq, 2)
            
            g_shared = meta_gate_weights[:, 0].unsqueeze(-1)
            g_domain = meta_gate_weights[:, 1].unsqueeze(-1)
            weighted_sum = g_shared * shared_output + g_domain * weighted_domain_output
            
            output = weighted_sum.view(batch_size, seq_len, self.hidden_units)

            # 2.6. 适配损失和可视化
            if self.training:
                # 计算专业化损失
                if self.use_specialization_loss and domain_ids is not None:
                    spec_loss = self.compute_specialization_loss(gate_logits, domain_ids)
                    loss_dict['specialization'] = self.specialization_weight * spec_loss
                
                # 计算对比学习损失
                if self.use_contrastive_loss and domain_ids is not None:
                    contrast_loss = self.compute_contrastive_loss(output, domain_ids)
                    loss_dict['contrastive'] = self.contrastive_weight * contrast_loss
                
                # 负载均衡
                if self.load_balancing:
                    expert_usage = gate_scores.sum(dim=0)
                    expert_load = expert_usage / (expert_usage.sum() + 1e-8)
                    lb_loss = self.balance_loss_weight * self.cv_squared(expert_load) * self.num_domain_experts
                    loss_dict['load_balancing'] = lb_loss
            
            if (self.load_balancing or self.args.visualize) and self.training:
                # 负载均衡仅基于领域专家的门控分数
                expert_usage = gate_scores.sum(dim=0)
                expert_load = expert_usage / (expert_usage.sum() + 1e-8)
                viz_data['expert_load'] = expert_load.detach()
                
                if self.args.visualize and domain_ids is not None:
                    token_domain_ids = domain_ids.unsqueeze(1).expand(-1, seq_len).reshape(-1)
                    
                    # --- 恢复并适配热力图数据生成 ---
                    # 新的热力图将显示 领域 -> 领域专家 的路由权重
                    domain_expert_load = torch.zeros(self.num_domain_experts, self.num_domain_experts, device=output.device)
                    domain_expert_load.scatter_add_(0, token_domain_ids.unsqueeze(1).expand(-1, self.num_domain_experts), gate_scores)
                    domain_token_counts = torch.bincount(token_domain_ids, minlength=self.num_domain_experts).float()
                    domain_expert_load = domain_expert_load / (domain_token_counts.unsqueeze(1) + 1e-8)
                    viz_data['domain_expert_load'] = domain_expert_load.detach()

                    viz_data['tsne_embeddings'] = weighted_sum.detach()
                    viz_data['tsne_labels'] = torch.argmax(gate_scores, dim=1).detach()
                    viz_data['tsne_domains'] = token_domain_ids.detach()

        else: # 'vanilla' 策略
            # --- 原有策略: 从所有专家中选择 Top-K ---
            gate_logits = self.gate(gate_input)
            gate_scores = F.softmax(gate_logits, dim=-1)
            
            if self.k >= self.num_experts:
                top_k_scores, top_k_indices = gate_scores, torch.arange(self.num_experts, device=gate_scores.device).expand(gate_scores.size(0), -1)
            else:
                top_k_scores, top_k_indices = torch.topk(gate_scores, self.k, dim=-1)

            top_k_scores_normalized = top_k_scores / (top_k_scores.sum(dim=-1, keepdim=True) + 1e-8)
            
            all_expert_outputs = torch.stack(shared_expert_outputs_list + domain_expert_outputs_list, dim=1)
            
            weighted_sum = torch.zeros_like(inputs_flat)
            for i in range(self.k):
                expert_indices = top_k_indices[:, i]
                scores = top_k_scores_normalized[:, i].unsqueeze(-1)
                selected_outputs = torch.gather(all_expert_outputs, 1, expert_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.hidden_units)).squeeze(1)
                weighted_sum += selected_outputs * scores
            
            output = weighted_sum.view(batch_size, seq_len, self.hidden_units)

            if (self.load_balancing or self.args.visualize) and self.training:
                expert_usage = torch.zeros(self.num_experts, device=output.device).scatter_add_(0, top_k_indices.flatten(), top_k_scores_normalized.flatten())
                expert_load = expert_usage / (expert_usage.sum() + 1e-8)
                viz_data['expert_load'] = expert_load.detach()

                if self.load_balancing:
                    lb_loss = self.balance_loss_weight * self.cv_squared(expert_load) * self.num_experts
                    loss_dict['load_balancing'] = lb_loss

                if self.args.visualize and domain_ids is not None:
                    token_domain_ids = domain_ids.unsqueeze(1).expand(-1, seq_len).reshape(-1)
                    domain_expert_load = torch.zeros(self.num_domain_experts, self.num_experts, device=output.device)
                    domain_expert_load.scatter_add_(0, token_domain_ids.unsqueeze(1).expand(-1, self.num_experts), gate_scores)
                    domain_token_counts = torch.bincount(token_domain_ids, minlength=self.num_domain_experts).float()
                    domain_expert_load = domain_expert_load / (domain_token_counts.unsqueeze(1) + 1e-8)
                    viz_data['domain_expert_load'] = domain_expert_load.detach()
                    
                    viz_data['tsne_embeddings'] = weighted_sum.detach()
                    viz_data['tsne_labels'] = torch.argmax(gate_scores, dim=1).detach()
                    viz_data['tsne_domains'] = token_domain_ids.detach()


        return output, loss_dict, viz_data