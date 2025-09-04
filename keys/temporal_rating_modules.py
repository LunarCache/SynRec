import torch
import torch.nn as nn
import numpy as np
from enum import Enum


class AblationMode(Enum):
    """消融实验模式"""
    FULL = "full"                # 完整模型（默认）
    LOW_FREQ_ONLY = "low_only"   # 仅低频成分
    HIGH_FREQ_ONLY = "high_only" # 仅高频成分


class OptimizedFourierRatingEncoder(nn.Module):
    """
    优化的基于傅里叶变换的时频域Rating编码器
    
    核心优化：
    1. Z-score标准化输入rating序列
    2. 在嵌入空间沿时间维进行FFT分解
    3. 基于cutoff_ratio的低频/高频分离
    4. 可学习的软边界机制
    5. 防谱泄露技术
    6. IFFT重构后的双分支注意力
    """
    
    def __init__(self, hidden_units, max_len=100, cutoff_ratio=0.3, 
                 learnable_cutoff=True, attention_heads=4, 
                 dropout_rate=0.1, use_windowing=True, ablation_mode=AblationMode.FULL):
        super(OptimizedFourierRatingEncoder, self).__init__()
        
        self.hidden_units = hidden_units
        self.max_len = max_len
        self.cutoff_ratio = cutoff_ratio
        self.learnable_cutoff = learnable_cutoff
        self.use_windowing = use_windowing
        self.ablation_mode = ablation_mode
        
        # 1. Rating嵌入层
        self.rating_embedding = nn.Embedding(6, hidden_units, padding_idx=0)
        
        # 2. 可学习的截止频率参数
        if learnable_cutoff:
            # 使用logit形式，通过sigmoid激活保证在(0,1)范围内
            self.cutoff_logit = nn.Parameter(
                torch.logit(torch.tensor(cutoff_ratio, dtype=torch.float32))
            )
            # 软边界的平滑度参数
            self.boundary_sharpness = nn.Parameter(torch.tensor(10.0))
        
        # 3. 双分支注意力机制 - 使用相同的头数
        # 长期分支：处理低频分量（长期趋势）
        self.long_term_attention = nn.MultiheadAttention(
            hidden_units, num_heads=attention_heads, dropout=dropout_rate, batch_first=True
        )
        
        # 短期分支：处理高频分量（短期波动）
        self.short_term_attention = nn.MultiheadAttention(
            hidden_units, num_heads=attention_heads, dropout=dropout_rate, batch_first=True
        )
        
        # 4. 特征增强网络
        self.long_term_enhancer = nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            nn.LayerNorm(hidden_units),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        self.short_term_enhancer = nn.Sequential(
            nn.Linear(hidden_units, hidden_units), 
            nn.LayerNorm(hidden_units),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # 5. 自适应融合网络
        self.adaptive_fusion = nn.Sequential(
            nn.Linear(hidden_units * 3, hidden_units * 2),  # 原始+长期+短期
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units * 2, hidden_units),
            nn.LayerNorm(hidden_units)
        )
        
        # 6. 融合权重生成器
        self.fusion_weight_generator = nn.Sequential(
            nn.Linear(hidden_units, hidden_units // 2),
            nn.Tanh(),
            nn.Linear(hidden_units // 2, 3),  # 原始、长期、短期权重
            nn.Softmax(dim=-1)
        )
        
        print(f"🔧 Initialized OptimizedFourierRatingEncoder:")
        print(f"   - Cutoff ratio: {cutoff_ratio} ({'learnable' if learnable_cutoff else 'fixed'})")
        print(f"   - Attention heads: {attention_heads} (same for both branches)")
        print(f"   - Windowing: {use_windowing}")
    
    def forward(self, rating_seq):
        """
        Args:
            rating_seq: (batch_size, seq_len) - rating序列
        Returns:
            enhanced_repr: (batch_size, seq_len, hidden_units) - 增强的rating表示
            analysis_info: dict - 包含频域分析信息
        """
        seq_len = rating_seq.size(1)
        device = rating_seq.device
        
        # 1. Z-score标准化
        rating_normalized = self._z_score_normalize(rating_seq)
        
        # 2. Rating嵌入
        rating_emb = self.rating_embedding(rating_normalized)  # (B, L, d)
        
        # 3. 防谱泄露预处理
        if self.use_windowing:
            windowed_emb = self._apply_windowing(rating_emb)
        else:
            windowed_emb = rating_emb
        
        # 4. FFT分解（沿时间维）
        fft_result = torch.fft.fft(windowed_emb, dim=1)  # (B, L, d)
        
        # 5. 低频/高频分离
        low_freq_mask, high_freq_mask, current_cutoff = self._create_frequency_masks(seq_len, device)
        
        # 6. 频域滤波
        low_freq_fft = fft_result * low_freq_mask.unsqueeze(0).unsqueeze(-1)   # (B, L, d)
        high_freq_fft = fft_result * high_freq_mask.unsqueeze(0).unsqueeze(-1) # (B, L, d)
        
        # 7. IFFT重构时域信号
        long_term_signal = torch.fft.ifft(low_freq_fft, dim=1).real   # (B, L, d)
        short_term_signal = torch.fft.ifft(high_freq_fft, dim=1).real # (B, L, d)
        
        # 8. 双分支注意力处理
        # 长期分支：关注长期依赖模式
        long_term_context, long_term_attn = self.long_term_attention(
            long_term_signal, long_term_signal, long_term_signal
        )
        long_term_enhanced = self.long_term_enhancer(long_term_context)
        
        # 短期分支：关注短期依赖模式  
        short_term_context, short_term_attn = self.short_term_attention(
            short_term_signal, short_term_signal, short_term_signal
        )
        short_term_enhanced = self.short_term_enhancer(short_term_context)
        
        # 9. 多尺度特征融合
        multi_scale_features = torch.cat([
            rating_emb,           # 原始特征
            long_term_enhanced,   # 长期特征  
            short_term_enhanced   # 短期特征
        ], dim=-1)
        
        fused_features = self.adaptive_fusion(multi_scale_features)
        
        # 10. 自适应权重融合
        fusion_weights = self.fusion_weight_generator(fused_features)  # (B, L, 3)
        
        # 消融实验逻辑：根据模式选择性地使用不同频率成分
        # 修复：统一使用融合权重，通过屏蔽特定组件实现消融
        if self.ablation_mode == AblationMode.LOW_FREQ_ONLY:
            # 仅使用低频成分：屏蔽短期分支，保留原始和长期分支
            final_repr = (
                fusion_weights[..., 0:1] * rating_emb +
                fusion_weights[..., 1:2] * long_term_enhanced +  
                torch.zeros_like(fusion_weights[..., 2:3]) * short_term_enhanced  # 短期权重设为0
            )
        elif self.ablation_mode == AblationMode.HIGH_FREQ_ONLY:
            # 仅使用高频成分：屏蔽长期分支，保留原始和短期分支
            final_repr = (
                fusion_weights[..., 0:1] * rating_emb +
                torch.zeros_like(fusion_weights[..., 1:2]) * long_term_enhanced +  # 长期权重设为0
                fusion_weights[..., 2:3] * short_term_enhanced
            )
        else:
            # 完整模型：自适应权重融合（默认）
            final_repr = (
                fusion_weights[..., 0:1] * rating_emb +
                fusion_weights[..., 1:2] * long_term_enhanced +  
                fusion_weights[..., 2:3] * short_term_enhanced
            )
        
        # 11. 分析信息收集
        analysis_info = {
            'cutoff_frequency': current_cutoff.item(),
            'long_term_attention': long_term_attn.detach(),
            'short_term_attention': short_term_attn.detach(), 
            'fusion_weights': fusion_weights.detach(),
            # 为可视化添加的数据，与简化后的plot_multi_domain_fourier_comparison_journal兼容
            'visualization_data': {
                'adaptive_weights': fusion_weights.detach()  # 只保留自适应权重用于可视化
            }
        }
        
        return final_repr, analysis_info
    
    def _z_score_normalize(self, rating_seq):
        """Z-score标准化rating序列"""
        # 忽略padding位置（0为padding）
        valid_mask = (rating_seq != 0).float()
        
        # 计算每个序列的均值和标准差（仅考虑非padding位置）
        seq_sum = torch.sum(rating_seq * valid_mask, dim=-1, keepdim=True)
        seq_count = torch.sum(valid_mask, dim=-1, keepdim=True) + 1e-8
        seq_mean = seq_sum / seq_count
        
        # 计算标准差
        squared_diff = ((rating_seq - seq_mean) ** 2) * valid_mask
        seq_var = torch.sum(squared_diff, dim=-1, keepdim=True) / seq_count
        seq_std = torch.sqrt(seq_var + 1e-8)
        
        # Z-score标准化
        rating_normalized = (rating_seq - seq_mean) / seq_std
        
        # 保持padding位置为0
        rating_normalized = rating_normalized * valid_mask
        
        # 转换为整数索引（四舍五入并限制范围）
        rating_normalized = torch.clamp(
            torch.round(rating_normalized) + 3, 0, 5  # 映射到[0,5]范围
        ).long()
        
        return rating_normalized
    
    def _apply_windowing(self, embeddings):
        """应用Hann窗函数防止谱泄露"""
        seq_len = embeddings.size(1)
        
        # 创建Hann窗
        hann_window = torch.hann_window(seq_len, device=embeddings.device)
        
        # 应用窗函数（沿时间维）
        windowed_emb = embeddings * hann_window.view(1, -1, 1)
        
        return windowed_emb
    
    def _create_frequency_masks(self, seq_len, device):
        """创建低频/高频分离掩码"""
        # 获取当前截止频率
        if self.learnable_cutoff:
            current_cutoff = torch.sigmoid(self.cutoff_logit)
        else:
            current_cutoff = self.cutoff_ratio
        
        # 计算截止频率索引
        cutoff_idx = int(seq_len * current_cutoff)
        
        # 频率索引
        freq_indices = torch.arange(seq_len, device=device, dtype=torch.float32)
        
        if self.learnable_cutoff:
            # 可学习的软边界
            # 使用sigmoid创建平滑过渡
            boundary_center = seq_len * current_cutoff
            soft_boundary = torch.sigmoid(
                self.boundary_sharpness * (boundary_center - freq_indices)
            )
            
            # 考虑FFT的对称性（负频率部分）
            # 对于实信号，FFT结果关于N/2对称
            symmetric_boundary = torch.where(
                freq_indices <= seq_len // 2,
                soft_boundary,
                torch.sigmoid(self.boundary_sharpness * (freq_indices - (seq_len - boundary_center)))
            )
            
            low_freq_mask = symmetric_boundary
            high_freq_mask = 1.0 - symmetric_boundary
            
        else:
            # 硬边界
            low_freq_mask = torch.zeros(seq_len, device=device)
            
            # 低频部分：[0, cutoff_idx) 和 [seq_len-cutoff_idx, seq_len)
            low_freq_mask[:cutoff_idx] = 1.0
            if cutoff_idx > 0:
                low_freq_mask[seq_len - cutoff_idx:] = 1.0
            
            high_freq_mask = 1.0 - low_freq_mask
        
        return low_freq_mask, high_freq_mask, current_cutoff


class TemporalEnhancedRatingModule(nn.Module):
    """
    时频域增强的Rating模块
    为每个领域创建独立的OptimizedFourierRatingEncoder实例
    """
    
    def __init__(self, hidden_units, rating_strategy='temporal_fourier', dropout_rate=0.1, args=None, ablation_mode=AblationMode.FULL):
        super(TemporalEnhancedRatingModule, self).__init__()
        self.rating_strategy = rating_strategy
        self.hidden_units = hidden_units
        self.ablation_mode = ablation_mode
        
        if rating_strategy != 'temporal_fourier':
            raise ValueError(f"Unsupported rating strategy: {rating_strategy}. Only 'temporal_fourier' is supported.")
        
        # 获取领域数量，默认为3个领域
        self.num_domains = getattr(args, 'num_domains', 3)
        
        # 为每个领域创建独立的编码器实例
        config = self._get_unified_config(args)
        self.domain_encoders = nn.ModuleDict()
        
        for domain_id in range(self.num_domains):
            self.domain_encoders[str(domain_id)] = OptimizedFourierRatingEncoder(
                hidden_units=self.hidden_units,
                max_len=config['max_len'],
                cutoff_ratio=config['cutoff_ratio'],
                learnable_cutoff=config['learnable_cutoff'],
                attention_heads=config['attention_heads'],
                dropout_rate=dropout_rate,
                use_windowing=config['use_windowing'],
                ablation_mode=ablation_mode  # 传递消融模式
            )
        
        print(f"🔧 Created domain-specific temporal-enhanced rating encoders:")
        print(f"   - Number of domains: {self.num_domains}")
        print(f"   - Initial cutoff ratio: {config['cutoff_ratio']} ({'learnable' if config['learnable_cutoff'] else 'fixed'})")
        print(f"   - Attention heads: {config['attention_heads']}") 
        print(f"   - Windowing: {config['use_windowing']}")
        print(f"   - Ablation mode: {ablation_mode.value}")
        print(f"   - Each domain has its own independent encoder")
    
    def set_ablation_mode(self, ablation_mode: AblationMode):
        """
        动态设置消融模式
        
        Args:
            ablation_mode: 新的消融模式
        """
        self.ablation_mode = ablation_mode
        for encoder in self.domain_encoders.values():
            encoder.ablation_mode = ablation_mode
        print(f"🔌 Switched to ablation mode: {ablation_mode.value}")
    
    def get_ablation_mode(self) -> AblationMode:
        """获取当前消融模式"""
        return self.ablation_mode
    
    def _get_unified_config(self, args=None):
        """获取统一的配置"""
        return {
            'max_len': 100,
            'cutoff_ratio': 0.3,      # 初始值，通过learnable_cutoff自适应
            'learnable_cutoff': True,  # 关键：让模型学习最优截止频率
            'attention_heads': 1,      # 统一的注意力头数
            'use_windowing': True      # 防谱泄露
        }
    
    def forward(self, rating_seq, domain_ids=None, implicit_signals=None):
        """
        Args:
            rating_seq: (batch_size, seq_len)
            domain_ids: (batch_size,) - 领域ID，必需参数
            implicit_signals: 兼容性参数
        Returns:
            enhanced_rating_repr: (batch_size, seq_len, hidden_units)
            extra_info: dict
        """
        batch_size, seq_len = rating_seq.shape
        device = rating_seq.device
        
        # 检查domain_ids是否提供
        if domain_ids is None:
            raise ValueError("domain_ids is required for domain-specific processing")
        
        # 确保domain_ids在正确的设备上
        if isinstance(domain_ids, list):
            domain_ids = torch.LongTensor(domain_ids).to(device)
        elif isinstance(domain_ids, torch.Tensor):
            domain_ids = domain_ids.to(device)
        
        # 初始化输出张量
        enhanced_repr = torch.zeros(batch_size, seq_len, self.hidden_units, 
                                   device=device, dtype=torch.float32)
        
        # 收集所有分析信息
        all_analysis_info = {}
        
        # 根据域ID分组处理
        unique_domains = torch.unique(domain_ids)
        
        for domain_id in unique_domains:
            domain_key = str(domain_id.item())
            
            # 检查是否存在该域的编码器
            if domain_key not in self.domain_encoders:
                raise ValueError(f"No encoder found for domain_id: {domain_id.item()}. "
                               f"Available domains: {list(self.domain_encoders.keys())}")
            
            # 获取属于当前域的样本索引
            domain_mask = (domain_ids == domain_id)
            domain_indices = torch.where(domain_mask)[0]
            
            if len(domain_indices) == 0:
                continue
            
            # 提取当前域的rating序列
            domain_rating_seq = rating_seq[domain_indices]  # (domain_batch_size, seq_len)
            
            # 使用对应域的编码器处理
            domain_enhanced_repr, domain_analysis_info = self.domain_encoders[domain_key](domain_rating_seq)
            
            # 将结果放回原始位置
            enhanced_repr[domain_indices] = domain_enhanced_repr
            
            # 收集分析信息
            all_analysis_info[f'domain_{domain_id.item()}'] = domain_analysis_info
        
        # 构建额外信息
        extra_info = {
            'frequency_analysis': all_analysis_info,
            'domain_processing_summary': {
                'processed_domains': [int(d.item()) for d in unique_domains],
                'samples_per_domain': {int(d.item()): int((domain_ids == d).sum().item()) 
                                     for d in unique_domains}
            }
        }
        
        return enhanced_repr, extra_info