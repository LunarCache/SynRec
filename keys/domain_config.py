import json
import os
import numpy as np
from typing import Dict, Any, List, Optional


class DomainAdaptiveConfig:
    """
    轻量级领域自适应配置管理器
    
    基于数据集统计特征自动生成最优的FourierRatingEncoder参数配置，
    在减少计算开销的同时提升模型在各个领域的表现。
    """
    
    # 数据集名称到规范显示名称的映射
    DOMAIN_DISPLAY_NAMES = {
        'beauty_5_5': 'Beauty',
        'games_5_5': 'Games', 
        'ml-1m_5_5': 'MovieLens',
        'beauty_rated': 'Beauty',
        'games_rated': 'Games',
        'ml-1m_rated': 'MovieLens'
    }
    
    # 预定义的领域配置策略（计算高效版本）
    DOMAIN_CONFIGS = {
        'beauty_5_5': {
            'num_frequencies': 8,        # 简单序列，低频需求
            'branch1_heads': 1,       # 最小头数
            'branch2_heads': 1,        # 最小头数
            'max_len': 15,               # 基于数据统计的合理长度
            'description': 'Beauty domain: Short sequences, simple patterns'
        },
        'games_5_5': {
            'num_frequencies': 12,       # 中等复杂度
            'branch1_heads': 1,       # 保持高效
            'branch2_heads': 1,        # 保持高效
            'max_len': 25,               # 适中的序列长度
            'description': 'Games domain: Medium sequences, moderate complexity'
        },
        'ml-1m_5_5': {
            'num_frequencies': 16,       # 复杂序列，更多频率成分
            'branch1_heads': 2,       # 仅略微增加
            'branch2_heads': 1,        # 控制计算开销
            'max_len': 50,              # 长序列支持
            'description': 'MovieLens domain: Long sequences, complex patterns'
        }
    }   
    
    # 默认配置（未知数据集的安全配置）
    DEFAULT_CONFIG = {
        'num_frequencies': 12,
        'branch1_heads': 1,
        'branch2_heads': 1,
        'max_len': 50,
        'description': 'Default safe configuration'
    }
    
    def __init__(self, data_dir: str = "data"):
        """
        初始化配置管理器
        
        Args:
            data_dir: 数据集统计文件所在目录
        """
        self.data_dir = data_dir
        self.config_cache = {}
        self._load_dataset_stats()
    
    def _load_dataset_stats(self):
        """从data目录加载数据集统计信息"""
        self.dataset_stats = {}
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith('_5_5.json'):
                dataset_name = filename.replace('_5_5.json', '')
                filepath = os.path.join(self.data_dir, filename)
                
                try:
                    with open(filepath, 'r') as f:
                        stats = json.load(f)
                        self.dataset_stats[dataset_name] = stats
                except Exception as e:
                    print(f"Warning: Failed to load {filepath}: {e}")
    
    def get_domain_config(self, dataset_names: List[str]) -> Dict[str, Any]:
        """
        获取指定数据集的领域配置
        
        Args:
            dataset_names: 数据集名称列表
            
        Returns:
            适合该数据集的配置参数字典
        """
        if len(dataset_names) == 1:
            return self._get_single_domain_config(dataset_names[0])
        else:
            return self._get_multi_domain_config(dataset_names)
    
    def _get_single_domain_config(self, dataset_name: str) -> Dict[str, Any]:
        """获取单数据集的配置"""
        # 检查缓存
        if dataset_name in self.config_cache:
            return self.config_cache[dataset_name]
        
        # 首先检查预定义配置
        if dataset_name in self.DOMAIN_CONFIGS:
            config = self.DOMAIN_CONFIGS[dataset_name].copy()
        else:
            # 基于数据统计自动生成配置
            config = self._generate_config_from_stats(dataset_name)
        
        # 缓存配置
        self.config_cache[dataset_name] = config
        return config
    
    def _get_multi_domain_config(self, dataset_names: List[str]) -> Dict[str, Any]:
        """获取多数据集的平衡配置"""
        configs = [self._get_single_domain_config(name) for name in dataset_names]
        
        # 计算平衡配置（倾向于复杂度较高的配置以保证所有领域的效果）
        balanced_config = {
            'num_frequencies': max(config['num_frequencies'] for config in configs),
            'branch1_heads': max(config['branch1_heads'] for config in configs),
            'branch2_heads': max(config['branch2_heads'] for config in configs),
            'max_len': max(config['max_len'] for config in configs),
            'description': f'Balanced config for: {", ".join(dataset_names)}'
        }
        
        return balanced_config
    
    def _generate_config_from_stats(self, dataset_name: str) -> Dict[str, Any]:
        """基于数据集统计信息自动生成配置"""
        if dataset_name not in self.dataset_stats:
            print(f"Warning: No stats found for {dataset_name}, using default config")
            return self.DEFAULT_CONFIG.copy()
        
        stats = self.dataset_stats[dataset_name]
        basic_stats = stats.get('basic_stats', {})
        seq_stats = stats.get('sequence_length_distribution', {})
        
        # 提取关键特征
        avg_seq_len = basic_stats.get('avg_sequence_length', 50)
        seq_std = seq_stats.get('std', 10)
        sparsity = basic_stats.get('sparsity', 0.99)
        
        # 基于规则生成配置
        config = self._rule_based_config_generation(avg_seq_len, seq_std, sparsity)
        config['description'] = f'Auto-generated for {dataset_name} (avg_len={avg_seq_len:.1f})'
        
        return config
    
    def _rule_based_config_generation(self, avg_seq_len: float, seq_std: float, sparsity: float) -> Dict[str, Any]:
        """基于规则的配置生成"""
        
        # 根据平均序列长度确定复杂度需求
        if avg_seq_len < 15:
            # 短序列：简单配置
            num_frequencies = 8
            branch1_heads = 1
            branch2_heads = 1
            max_len = int(avg_seq_len * 2)
        elif avg_seq_len < 50:
            # 中等序列：平衡配置
            num_frequencies = 12
            branch1_heads = 1
            branch2_heads = 1
            max_len = int(avg_seq_len * 1.5)
        else:
            # 长序列：复杂配置（但控制头数）
            num_frequencies = 16
            branch1_heads = 2
            branch2_heads = 1
            max_len = min(int(avg_seq_len * 1.2), 300)  # 限制最大长度
        
        # 根据序列变化程度微调
        if seq_std > avg_seq_len * 0.5:
            # 变化很大，需要更多频率成分
            num_frequencies = min(num_frequencies + 4, 20)
        
        return {
            'num_frequencies': num_frequencies,
            'branch1_heads': branch1_heads,
            'branch2_heads': branch2_heads,
            'max_len': max_len
        }
    
    def print_config_summary(self, dataset_names: List[str]):
        """打印配置摘要信息"""
        config = self.get_domain_config(dataset_names)
        
        print("🔧 Domain Adaptive Configuration Summary")
        print("=" * 50)
        print(f"Datasets: {', '.join(dataset_names)}")
        print(f"Description: {config.get('description', 'N/A')}")
        print(f"Frequency components: {config['num_frequencies']}")
        print(f"branch1_heads attention heads: {config['branch1_heads']}")
        print(f"branch2_heads attention heads: {config['branch2_heads']}")
        print(f"Max sequence length: {config['max_len']}")
        
        # 计算相对于默认配置的计算开销
        default = self.DEFAULT_CONFIG
        compute_ratio = (
            (config['num_frequencies'] / default['num_frequencies']) *
            ((config['branch1_heads'] + config['branch2_heads']) / 
             (default['branch1_heads'] + default['branch2_heads']))
        )
        
        print(f"Relative compute cost: {compute_ratio:.2f}x")
        if compute_ratio < 1.0:
            print(f"💚 Compute savings: {(1-compute_ratio)*100:.1f}%")
        else:
            print(f"🔶 Compute increase: {(compute_ratio-1)*100:.1f}%")
        
        print("=" * 50)
    
    def get_available_datasets(self) -> List[str]:
        """获取可用的数据集列表"""
        available = list(self.dataset_stats.keys()) + list(self.DOMAIN_CONFIGS.keys())
        return sorted(set(available))
    
    def get_display_name(self, dataset_name: str) -> str:
        """
        获取数据集的规范显示名称
        
        Args:
            dataset_name: 原始数据集名称
            
        Returns:
            规范的显示名称
        """
        return self.DOMAIN_DISPLAY_NAMES.get(dataset_name, dataset_name)


# 工厂函数，便于在其他模块中使用
def create_domain_config(data_dir: str = "data") -> DomainAdaptiveConfig:
    """创建领域自适应配置管理器实例"""
    return DomainAdaptiveConfig(data_dir)


# 快速配置函数
def get_rating_config_for_datasets(dataset_names: List[str], data_dir: str = "data") -> Dict[str, Any]:
    """
    快速获取指定数据集的rating配置
    
    Args:
        dataset_names: 数据集名称列表
        data_dir: 数据目录路径
        
    Returns:
        rating配置字典
    """
    config_manager = create_domain_config(data_dir)
    return config_manager.get_domain_config(dataset_names)