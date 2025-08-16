import torch
import torch.nn as nn
import numpy as np


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
                 dropout_rate=0.1, use_windowing=True):
        super(OptimizedFourierRatingEncoder, self).__init__()
        
        self.hidden_units = hidden_units
        self.max_len = max_len
        self.cutoff_ratio = cutoff_ratio
        self.learnable_cutoff = learnable_cutoff
        self.use_windowing = use_windowing
        
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
        
        final_repr = (
            fusion_weights[..., 0:1] * rating_emb +
            fusion_weights[..., 1:2] * long_term_enhanced +  
            fusion_weights[..., 2:3] * short_term_enhanced
        )
        
        # 11. 分析信息收集
        analysis_info = {
            'cutoff_frequency': current_cutoff,
            'long_term_attention': long_term_attn,
            'short_term_attention': short_term_attn, 
            'fusion_weights': fusion_weights,
            'frequency_masks': {
                'low_freq_mask': low_freq_mask,
                'high_freq_mask': high_freq_mask
            },
            'signal_power': {
                'long_term_power': torch.mean(long_term_signal ** 2).item(),
                'short_term_power': torch.mean(short_term_signal ** 2).item()
            }
        }
        
        return final_repr, analysis_info
    
    def _z_score_normalize(self, rating_seq):
        """Z-score标准化rating序列"""
        # 忽略padding位置（假设0为padding）
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
    
    def get_frequency_analysis(self, rating_seq):
        """获取详细的频域分析"""
        with torch.no_grad():
            enhanced_repr, analysis_info = self.forward(rating_seq)
            
            return {
                'enhanced_representation': enhanced_repr,
                'frequency_analysis': analysis_info,
                'model_params': {
                    'cutoff_ratio': analysis_info['cutoff_frequency'],
                    'learnable_cutoff': self.learnable_cutoff,
                    'boundary_sharpness': self.boundary_sharpness.item() if self.learnable_cutoff else None
                }
            }


class TemporalEnhancedRatingModule(nn.Module):
    """
    时频域增强的Rating模块
    使用统一的OptimizedFourierRatingEncoder处理所有领域
    """
    
    def __init__(self, hidden_units, rating_strategy='temporal_fourier', dropout_rate=0.1, args=None):
        super(TemporalEnhancedRatingModule, self).__init__()
        self.rating_strategy = rating_strategy
        self.hidden_units = hidden_units
        
        if rating_strategy != 'temporal_fourier':
            raise ValueError(f"Unsupported rating strategy: {rating_strategy}. Only 'temporal_fourier' is supported.")
        
        # 使用统一的编码器处理所有领域
        config = self._get_unified_config(args)
        
        self.fourier_encoder = OptimizedFourierRatingEncoder(
            hidden_units=self.hidden_units,
            max_len=config['max_len'],
            cutoff_ratio=config['cutoff_ratio'],
            learnable_cutoff=config['learnable_cutoff'],
            attention_heads=config['attention_heads'],
            dropout_rate=dropout_rate,
            use_windowing=config['use_windowing']
        )
        
        print(f"🔧 Created unified temporal-enhanced rating encoder:")
        print(f"   - Initial cutoff ratio: {config['cutoff_ratio']} ({'learnable' if config['learnable_cutoff'] else 'fixed'})")
        print(f"   - Attention heads: {config['attention_heads']}") 
        print(f"   - Windowing: {config['use_windowing']}")
        print(f"   - Single encoder handles all domains adaptively")
    
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
            domain_ids: (batch_size,) - 兼容性参数，不再使用
            implicit_signals: 兼容性参数
        Returns:
            enhanced_rating_repr: (batch_size, seq_len, hidden_units)
            extra_info: dict
        """
        # 使用统一编码器处理所有样本
        enhanced_repr, analysis_info = self.fourier_encoder(rating_seq)
        
        extra_info = {'frequency_analysis': analysis_info}
        return enhanced_repr, extra_info