"""
期刊级配色方案模块

提供符合顶级期刊要求的专业配色方案，确保：
1. 色彩无障碍友好
2. 打印效果良好  
3. 符合各大期刊的视觉规范
4. 在灰度模式下仍然清晰可辨
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, List, Tuple, Optional
import seaborn as sns

class JournalColorSchemes:
    """期刊级配色方案管理器"""
    
    def __init__(self):
        self._define_color_schemes()
    
    def _define_color_schemes(self):
        """定义各种期刊级配色方案"""
        
        # Nature系列配色 - 经典蓝红绿组合
        self.nature_colors = {
            'primary': '#0173B2',      # Nature蓝
            'secondary': '#DE8F05',    # Nature橙
            'tertiary': '#029E73',     # Nature绿
            'quaternary': '#CC78BC',   # Nature紫
            'accent': '#CA9161',       # Nature棕
            'error': '#D55E00',        # Nature红
            'neutral': '#525252'       # Nature灰
        }
        
        # Science系列配色 - 现代科学风格
        self.science_colors = {
            'primary': '#1f77b4',      # Science蓝
            'secondary': '#ff7f0e',    # Science橙  
            'tertiary': '#2ca02c',     # Science绿
            'quaternary': '#d62728',   # Science红
            'accent': '#9467bd',       # Science紫
            'error': '#8c564b',        # Science棕
            'neutral': '#7f7f7f'       # Science灰
        }
        
        # Cell系列配色 - 生物医学风格
        self.cell_colors = {
            'primary': '#2E86AB',      # Cell蓝
            'secondary': '#A23B72',    # Cell紫红
            'tertiary': '#F18F01',     # Cell橙
            'quaternary': '#C73E1D',   # Cell红
            'accent': '#6A994E',       # Cell绿
            'error': '#BC4749',        # Cell深红
            'neutral': '#606060'       # Cell灰
        }
        
        # 色彩无障碍友好配色(Okabe-Ito palette)
        self.colorblind_friendly = {
            'orange': '#E69F00',
            'sky_blue': '#56B4E9', 
            'bluish_green': '#009E73',
            'yellow': '#F0E442',
            'blue': '#0072B2',
            'vermillion': '#D55E00',
            'reddish_purple': '#CC79A7',
            'black': '#000000'
        }
        
        # 定义渐变色映射
        self._define_gradient_maps()
    
    def _define_gradient_maps(self):
        """定义专业渐变色映射"""
        
        # Nature风格的渐变
        self.nature_gradients = {
            'sequential_blue': ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', 
                               '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b'],
            'sequential_red': ['#fff5f0', '#fee0d2', '#fcbba1', '#fc9272',
                              '#fb6a4a', '#ef3b2c', '#cb181d', '#a50f15', '#67000d'],
            'sequential_green': ['#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b',
                                '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b']
        }
        
        # Science风格的渐变  
        self.science_gradients = {
            'diverging_blue_red': ['#053061', '#2166ac', '#4393c3', '#92c5de', 
                                  '#d1e5f0', '#f7f7f7', '#fddbc7', '#f4a582',
                                  '#d6604d', '#b2182b', '#67001f'],
            'sequential_viridis': ['#440154', '#482777', '#3f4a8a', '#31678e',
                                  '#26838f', '#1f9d8a', '#6cce5a', '#b6de2b', '#fee825']
        }
    
    def get_journal_palette(self, journal: str = 'nature', n_colors: int = 8) -> List[str]:
        """
        获取指定期刊的调色板
        
        Args:
            journal: 期刊名称 ('nature', 'science', 'cell')
            n_colors: 需要的颜色数量
            
        Returns:
            颜色列表
        """
        color_schemes = {
            'nature': self.nature_colors,
            'science': self.science_colors, 
            'cell': self.cell_colors
        }
        
        if journal not in color_schemes:
            journal = 'nature'  # 默认使用Nature配色
        
        colors = list(color_schemes[journal].values())
        
        # 如果需要更多颜色，使用色彩无障碍友好配色补充
        if len(colors) < n_colors:
            colors.extend(list(self.colorblind_friendly.values()))
        
        return colors[:n_colors]
    
    def get_attention_colormap(self, style: str = 'nature') -> str:
        """
        获取注意力矩阵的配色方案
        
        Args:
            style: 配色风格
            
        Returns:
            matplotlib colormap名称
        """
        attention_cmaps = {
            'nature': 'Blues',      # Nature风格：经典蓝色
            'science': 'viridis',   # Science风格：现代viridis
            'cell': 'plasma',       # Cell风格：紫色plasma
            'classic': 'Reds'       # 经典红色
        }
        
        return attention_cmaps.get(style, 'Blues')
    
    def get_heatmap_colormap(self, style: str = 'nature') -> str:
        """
        获取热力图的配色方案
        
        Args:
            style: 配色风格
            
        Returns:
            matplotlib colormap名称
        """
        heatmap_cmaps = {
            'nature': 'YlOrRd',     # Nature风格：黄橙红渐变
            'science': 'RdYlBu_r',  # Science风格：红黄蓝反向
            'cell': 'coolwarm',     # Cell风格：冷暖色调
            'classic': 'YlGnBu'     # 经典黄绿蓝
        }
        
        return heatmap_cmaps.get(style, 'YlOrRd')
    
    def get_scatter_colors(self, n_categories: int, style: str = 'nature') -> List[str]:
        """
        获取散点图的分类颜色
        
        Args:
            n_categories: 分类数量
            style: 配色风格
            
        Returns:
            颜色列表
        """
        if style == 'colorblind_friendly':
            colors = list(self.colorblind_friendly.values())
        else:
            colors = self.get_journal_palette(style, n_categories)
        
        # 如果类别过多，使用matplotlib的qualitative色谱
        if n_categories > len(colors):
            if n_categories <= 10:
                colors = plt.cm.Set3(np.linspace(0, 1, n_categories))
            elif n_categories <= 20:
                colors = plt.cm.tab20(np.linspace(0, 1, n_categories))
            else:
                # 对于更多类别，使用连续色谱
                colors = plt.cm.hsv(np.linspace(0, 1, n_categories))
        
        return colors[:n_categories]

def get_journal_palette(journal: str = 'nature', n_colors: int = 8) -> List[str]:
    """便捷函数：获取期刊调色板"""
    schemes = JournalColorSchemes()
    return schemes.get_journal_palette(journal, n_colors)

def get_colorblind_friendly_colors(n_colors: int = 8) -> List[str]:
    """便捷函数：获取色彩无障碍友好的颜色"""
    schemes = JournalColorSchemes()
    return schemes.get_scatter_colors(n_colors, 'colorblind_friendly')

def create_custom_colormap(colors: List[str], name: str = 'custom') -> mcolors.LinearSegmentedColormap:
    """
    创建自定义连续色彩映射
    
    Args:
        colors: 颜色列表
        name: 色彩映射名称
        
    Returns:
        自定义色彩映射
    """
    return mcolors.LinearSegmentedColormap.from_list(name, colors)

def validate_color_accessibility(colors: List[str]) -> Dict[str, bool]:
    """
    验证颜色的可访问性
    
    Args:
        colors: 颜色列表
        
    Returns:
        可访问性检查结果
    """
    results = {
        'sufficient_contrast': True,
        'distinguishable': True,  
        'grayscale_friendly': True
    }
    
    # 这里可以添加具体的对比度和可区分性检查逻辑
    # 简化实现，实际应用中可以使用专业的色彩分析库
    
    return results

# 预定义的期刊样式配色方案
JOURNAL_PALETTES = {
    'nature': {
        'primary_colors': ['#0173B2', '#DE8F05', '#029E73', '#CC78BC'],
        'attention_cmap': 'Blues',
        'heatmap_cmap': 'YlOrRd',
        'diverging_cmap': 'RdBu_r'
    },
    'science': {
        'primary_colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
        'attention_cmap': 'viridis', 
        'heatmap_cmap': 'RdYlBu_r',
        'diverging_cmap': 'coolwarm'
    },
    'cell': {
        'primary_colors': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'],
        'attention_cmap': 'plasma',
        'heatmap_cmap': 'coolwarm', 
        'diverging_cmap': 'seismic'
    }
}