import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .domain_config import get_rating_config_for_datasets


class FourierRatingEncoder(nn.Module):
    """
    基于傅里叶变换的多尺度Rating建模
    
    创新点：
    1. 使用FFT提取rating序列的频域特征
    2. 双分支注意力机制自适应学习不同时间尺度模式
    3. 自适应融合机制整合多尺度信息
    4. 不预设短期/长期标签，让模型自主学习分化
    """
    
    def __init__(self, hidden_units, max_len=100, num_frequencies=16, 
                 branch1_heads=4, branch2_heads=2, dropout_rate=0.1):
        super(FourierRatingEncoder, self).__init__()
        self.hidden_units = hidden_units
        self.num_frequencies = num_frequencies
        self.max_len = max_len
        
        # 基础rating嵌入
        self.rating_embedding = nn.Embedding(6, hidden_units, padding_idx=0)
        
        # 频域特征投影
        self.freq_projection = nn.Sequential(
            nn.Linear(num_frequencies * 2, hidden_units),
            nn.LayerNorm(hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 多尺度注意力机制
        self.attention_branch1 = nn.MultiheadAttention(
            hidden_units, num_heads=branch1_heads, dropout=dropout_rate, batch_first=True
        )
        self.attention_branch2 = nn.MultiheadAttention(
            hidden_units, num_heads=branch2_heads, dropout=dropout_rate, batch_first=True
        )
        
        # 时间尺度融合网络
        self.scale_fusion = nn.Sequential(
            nn.Linear(hidden_units * 3, hidden_units * 2),  # 原始+分支1+分支2
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units * 2, hidden_units),
            nn.LayerNorm(hidden_units)
        )
        
        # 自适应权重生成
        self.adaptive_weighting = nn.Sequential(
            nn.Linear(hidden_units, hidden_units // 2),
            nn.Tanh(),
            nn.Linear(hidden_units // 2, 3),  # 3个尺度的权重
            nn.Softmax(dim=-1)
        )
        
    def forward(self, rating_seq):
        """
        Args:
            rating_seq: (batch_size, seq_len) - rating序列
        Returns:
            enhanced_rating_repr: (batch_size, seq_len, hidden_units)
            attention_weights: dict - 包含分支1和分支2注意力权重用于可视化
        """
        batch_size, seq_len = rating_seq.shape
        
        # 1. 基础rating嵌入
        rating_emb = self.rating_embedding(rating_seq)  # (batch, seq, hidden)
        
        # 2. 傅里叶频域分解
        freq_features = self._fourier_decomposition(rating_seq)  # (batch, seq, hidden)
        
        # 3. 多尺度注意力建模
        # 注意力分支1（4个头 - 可能学习细粒度模式）
        branch1_context, branch1_attn_weights = self.attention_branch1(
            freq_features, freq_features, freq_features
        )
        
        # 注意力分支2（2个头 - 可能学习粗粒度模式）
        branch2_context, branch2_attn_weights = self.attention_branch2(
            freq_features, freq_features, freq_features  
        )
        
        # 4. 多尺度信息融合
        # 拼接原始、分支1、分支2特征
        multi_scale_features = torch.cat([
            rating_emb, branch1_context, branch2_context
        ], dim=-1)
        
        fused_features = self.scale_fusion(multi_scale_features)
        
        # 5. 自适应权重融合
        adaptive_weights = self.adaptive_weighting(fused_features)  # (batch, seq, 3)
        
        final_repr = (
            adaptive_weights[..., 0:1] * rating_emb +
            adaptive_weights[..., 1:2] * branch1_context +  # 分支1权重
            adaptive_weights[..., 2:3] * branch2_context     # 分支2权重
        )
        
        # 6. 收集注意力权重用于可视化
        attention_weights = {
            'attention_branch_1': branch1_attn_weights,     # (batch, heads, seq, seq) - 第一个注意力分支
            'attention_branch_2': branch2_attn_weights,     # (batch, heads, seq, seq) - 第二个注意力分支  
            'adaptive_weights': adaptive_weights,           # (batch, seq, 3) - 自适应融合权重
            'branch_1_heads': self.attention_branch1.num_heads,  # 第一个分支的头数
            'branch_2_heads': self.attention_branch2.num_heads   # 第二个分支的头数
        }
        
        return final_repr, attention_weights
    
    def _fourier_decomposition(self, rating_seq):
        """傅里叶分解rating序列"""
        batch_size, seq_len = rating_seq.shape
        
        # 转换为浮点数进行FFT
        rating_float = rating_seq.float()
        
        # 进行实数FFT
        fft_result = torch.fft.rfft(rating_float, dim=-1)
        
        # 计算实际的频率成分数量
        actual_freq_components = fft_result.shape[-1]
        
        # 确保不超过可用的频率成分
        num_freq_to_use = min(self.num_frequencies, actual_freq_components)
        
        # 提取前num_freq_to_use个频率成分
        freq_real = fft_result.real[..., :num_freq_to_use]
        freq_imag = fft_result.imag[..., :num_freq_to_use]
        
        # 如果频率成分不够，用零填充
        if num_freq_to_use < self.num_frequencies:
            padding_size = self.num_frequencies - num_freq_to_use
            freq_real = torch.cat([
                freq_real, 
                torch.zeros(batch_size, padding_size, device=rating_seq.device)
            ], dim=-1)
            freq_imag = torch.cat([
                freq_imag,
                torch.zeros(batch_size, padding_size, device=rating_seq.device)
            ], dim=-1)
        
        # 拼接实部和虚部
        freq_features = torch.cat([freq_real, freq_imag], dim=-1)  # (batch, num_freq*2)
        
        # 投影到hidden_units维度并扩展到序列长度
        freq_projected = self.freq_projection(freq_features)  # (batch, hidden)
        
        # 改进：为每个位置创建不同的频域特征（几乎零开销）
        # 创建位置权重：从1开始，避免位置0的问题
        position_weights = torch.arange(1, seq_len+1, device=rating_seq.device).float()
        position_weights = position_weights.unsqueeze(0).unsqueeze(-1)  # (1, seq, 1)
        
        # 位置调制：使用正弦函数创建轻微的位置差异
        # 0.15是调制强度，0.05是频率因子，可以根据实验效果调整
        position_scale = 1.0 + 0.15 * torch.sin(position_weights * 0.05)  # (1, seq, 1)
        
        # 为每个位置创建差异化的频域特征
        freq_expanded = freq_projected.unsqueeze(1) * position_scale  # (batch, seq, hidden)
        
        return freq_expanded


class EnhancedRatingModule(nn.Module):
    """
    领域自适应增强Rating模块
    
    核心特性：
    1. 为每个领域创建专门的FourierRatingEncoder
    2. 在forward过程中根据domain_id动态选择encoder
    3. 不同领域使用完全隔离的参数配置
    4. 支持计算高效的自适应配置
    """
    
    def __init__(self, hidden_units, rating_strategy='fourier', dropout_rate=0.1, args=None):
        super(EnhancedRatingModule, self).__init__()
        self.rating_strategy = rating_strategy
        self.hidden_units = hidden_units
        
        if rating_strategy != 'fourier':
            raise ValueError(f"Unsupported rating strategy: {rating_strategy}. Only 'fourier' is supported.")
        
        # 获取数据集信息和领域配置
        self.dataset_names, self.domain_configs = self._load_domain_configs(args)
        
        # 为每个领域创建专门的FourierRatingEncoder
        self.domain_encoders = nn.ModuleDict()
        self.domain_name_to_id = {}
        
        for domain_id, dataset_name in enumerate(self.dataset_names):
            config = self.domain_configs[dataset_name]
            
            self.domain_encoders[str(domain_id)] = FourierRatingEncoder(
                hidden_units,
                num_frequencies=config['num_frequencies'],
                branch1_heads=config['branch1_heads'],
                branch2_heads=config['branch2_heads'],
                max_len=config['max_len'],
                dropout_rate=dropout_rate
            )
            
            self.domain_name_to_id[dataset_name] = domain_id
        
        print(f"🔧 Created {len(self.domain_encoders)} domain-specific FourierRatingEncoders:")
        for domain_id, dataset_name in enumerate(self.dataset_names):
            config = self.domain_configs[dataset_name]
            print(f"   Domain {domain_id} ({dataset_name}): "
                  f"{config['num_frequencies']}freq + {config['branch1_heads']}+{config['branch2_heads']}heads")
    
    def _load_domain_configs(self, args):
        """加载每个领域的专门配置"""
        if args is None or not hasattr(args, 'use_datasets'):
            # 默认单领域配置
            return ['default'], {'default': {
                'num_frequencies': 12,
                'branch1_heads': 1,
                'branch2_heads': 1,
                'max_len': 50
            }}
        
        dataset_names = args.use_datasets
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        
        domain_configs = {}
        
        # 检查是否启用自适应配置
        use_adaptive_config = getattr(args, 'use_adaptive_rating_config', True)
        
        if use_adaptive_config:
            try:
                # 为每个数据集单独生成配置
                for dataset_name in dataset_names:
                    single_config = get_rating_config_for_datasets([dataset_name])
                    domain_configs[dataset_name] = single_config
                
                print(f"🔧 Using adaptive domain-specific configs for: {dataset_names}")
                
            except Exception as e:
                print(f"⚠️  Failed to load adaptive configs: {e}")
                print("   Falling back to manual parameters")
                use_adaptive_config = False
        
        if not use_adaptive_config:
            # 所有领域使用相同的手动配置
            manual_config = {
                'num_frequencies': getattr(args, 'rating_num_frequencies', 12),
                'branch1_heads': getattr(args, 'rating_branch1_heads', 1),
                'branch2_heads': getattr(args, 'rating_branch2_heads', 1),
                'max_len': getattr(args, 'maxlen', 100)
            }
            
            for dataset_name in dataset_names:
                domain_configs[dataset_name] = manual_config.copy()
        
        return dataset_names, domain_configs
    
    def forward(self, rating_seq, domain_ids=None, implicit_signals=None):
        """
        动态领域自适应forward
        
        Args:
            rating_seq: (batch_size, seq_len) - rating序列
            domain_ids: (batch_size,) - 每个样本的领域标识
            implicit_signals: 暂不使用（为了接口兼容性保留）
        Returns:
            enhanced_rating_repr: (batch_size, seq_len, hidden_units)
            extra_info: dict - 包含attention权重等额外信息
        """
        if domain_ids is None:
            # 如果没有domain_ids，使用第一个encoder作为默认
            domain_id = 0
            encoder = self.domain_encoders[str(domain_id)]
            enhanced_rating_repr, attention_weights = encoder(rating_seq)
            
            extra_info = {'attention_weights': attention_weights}
            return enhanced_rating_repr, extra_info
        
        # 根据domain_ids动态选择encoder并分别处理
        batch_size, seq_len = rating_seq.shape
        device = rating_seq.device
        
        # 初始化输出张量
        enhanced_rating_repr = torch.zeros(batch_size, seq_len, self.hidden_units, device=device)
        
        # 按domain分组处理
        unique_domains = torch.unique(domain_ids)
        domain_attention_dict = {}  # 使用dict保存domain_id -> attention的映射
        
        for domain_id in unique_domains:
            domain_id_int = domain_id.item()
            
            # 检查domain_id是否有效
            if str(domain_id_int) not in self.domain_encoders:
                print(f"⚠️  Unknown domain_id {domain_id_int}, using domain 0 as fallback")
                domain_id_int = 0
            
            # 获取该domain的样本索引
            domain_mask = (domain_ids == domain_id)
            domain_indices = torch.where(domain_mask)[0]
            
            if len(domain_indices) == 0:
                continue
            
            # 提取该domain的rating序列
            domain_rating_seq = rating_seq[domain_indices]
            
            # 使用对应的encoder处理
            encoder = self.domain_encoders[str(domain_id_int)]
            domain_repr, domain_attention = encoder(domain_rating_seq)
            
            # 将结果写回对应位置
            enhanced_rating_repr[domain_indices] = domain_repr
            # 保存domain_id和对应的attention数据
            domain_attention_dict[domain_id_int] = domain_attention
        
        extra_info = {'attention_weights': domain_attention_dict}
        return enhanced_rating_repr, extra_info
    
    def get_config_summary(self):
        """获取所有领域的配置摘要"""
        summary = {
            'strategy': self.rating_strategy,
            'num_domains': len(self.domain_encoders),
            'domain_configs': {}
        }
        
        for domain_id, dataset_name in enumerate(self.dataset_names):
            if dataset_name in self.domain_configs:
                summary['domain_configs'][f'domain_{domain_id}_{dataset_name}'] = self.domain_configs[dataset_name]
        
        return summary