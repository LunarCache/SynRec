"""
期刊样式模板模块

提供各大顶级期刊的专业样式配置，包括：
1. 字体规范和大小设置
2. 图表布局和间距标准
3. 线条样式和标记规范
4. 输出格式和分辨率要求
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
from matplotlib import cycler
from typing import Dict, Any, Optional, Tuple
import numpy as np

class JournalStyles:
    """期刊样式配置管理器"""
    
    def __init__(self):
        self._define_journal_styles()
    
    def _define_journal_styles(self):
        """定义各期刊的样式规范"""
        
        # Nature期刊样式
        self.nature_style = {
            # 字体配置
            'font.family': 'sans-serif',
            'font.size': 8,
            'font.weight': 'normal',
            'axes.titlesize': 9,
            'axes.labelsize': 8, 
            'xtick.labelsize': 7,
            'ytick.labelsize': 7,
            'legend.fontsize': 7,
            'figure.titlesize': 10,
            
            # 图表尺寸和DPI
            'figure.figsize': (5.2, 3.5),  # Nature单栏图片标准尺寸(英寸)
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.format': 'pdf',
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            
            # 线条和标记
            'lines.linewidth': 1.0,
            'lines.markersize': 4,
            'patch.linewidth': 0.5,
            'axes.linewidth': 0.8,
            
            # 网格和背景
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linewidth': 0.5,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.facecolor': 'white',
            'figure.facecolor': 'white',
            
            # 色彩设置
            'axes.prop_cycle': "cycler('color', ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#CA9161', '#D55E00'])",
            
            # 间距设置
            'figure.subplot.left': 0.15,
            'figure.subplot.bottom': 0.15,
            'figure.subplot.right': 0.95,
            'figure.subplot.top': 0.9,
            'figure.subplot.hspace': 0.3,
            'figure.subplot.wspace': 0.3
        }
        
        # Science期刊样式
        self.science_style = {
            # 字体配置
            'font.family': 'sans-serif',
            'font.size': 9,
            'font.weight': 'normal',
            'axes.titlesize': 10,
            'axes.labelsize': 9,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 8,
            'figure.titlesize': 11,
            
            # 图表尺寸和DPI
            'figure.figsize': (5.5, 4.0),  # Science标准尺寸
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.format': 'pdf',
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            
            # 线条和标记
            'lines.linewidth': 1.2,
            'lines.markersize': 5,
            'patch.linewidth': 0.6,
            'axes.linewidth': 1.0,
            
            # 网格和背景  
            'axes.grid': True,
            'grid.alpha': 0.25,
            'grid.linewidth': 0.6,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.facecolor': 'white',
            'figure.facecolor': 'white',
            
            # 色彩设置
            'axes.prop_cycle': "cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])",
            
            # 间距设置
            'figure.subplot.left': 0.15,
            'figure.subplot.bottom': 0.15,
            'figure.subplot.right': 0.93,
            'figure.subplot.top': 0.88,
            'figure.subplot.hspace': 0.35,
            'figure.subplot.wspace': 0.35
        }
        
        # Cell期刊样式
        self.cell_style = {
            # 字体配置
            'font.family': 'sans-serif',
            'font.size': 8,
            'font.weight': 'normal',
            'axes.titlesize': 9,
            'axes.labelsize': 8,
            'xtick.labelsize': 7,
            'ytick.labelsize': 7,
            'legend.fontsize': 7,
            'figure.titlesize': 10,
            
            # 图表尺寸和DPI
            'figure.figsize': (4.8, 3.6),  # Cell期刊偏向紧凑布局
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.format': 'pdf',
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.08,
            
            # 线条和标记
            'lines.linewidth': 1.0,
            'lines.markersize': 4,
            'patch.linewidth': 0.5,
            'axes.linewidth': 0.8,
            
            # 网格和背景
            'axes.grid': True,
            'grid.alpha': 0.2,
            'grid.linewidth': 0.4,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.facecolor': 'white',
            'figure.facecolor': 'white',
            
            # 色彩设置
            'axes.prop_cycle': "cycler('color', ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4749'])",
            
            # 间距设置
            'figure.subplot.left': 0.16,
            'figure.subplot.bottom': 0.16,
            'figure.subplot.right': 0.94,
            'figure.subplot.top': 0.9,
            'figure.subplot.hspace': 0.25,
            'figure.subplot.wspace': 0.25
        }
        
        # 自定义统一配色样式
        self.custom_style = {
            # 字体配置 - 现代简洁风格
            'font.family': 'sans-serif',
            'font.size': 9,
            'font.weight': 'normal',
            'axes.titlesize': 11,
            'axes.labelsize': 9,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 8,
            'figure.titlesize': 12,
            
            # 图表尺寸和分辨率
            'figure.figsize': (6.0, 4.0),
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.format': 'png',
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            
            # 线条和标记
            'lines.linewidth': 1.2,
            'lines.markersize': 5,
            'patch.linewidth': 0.8,
            'axes.linewidth': 1.0,
            
            # 网格和边框
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linewidth': 0.6,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.facecolor': 'white',
            'figure.facecolor': 'white',
            
            # 统一配色方案 - 9色配色表
            'axes.prop_cycle': "cycler('color', ['#8CD0C3', '#BCB9D8', '#F18072', '#80B1D2', '#F9B063', '#B3D46B', '#F7CBDF', '#D7D7D5', '#BA7FB5'])",
            
            # 间距设置
            'figure.subplot.left': 0.12,
            'figure.subplot.bottom': 0.15,
            'figure.subplot.right': 0.95,
            'figure.subplot.top': 0.9,
            'figure.subplot.hspace': 0.3,
            'figure.subplot.wspace': 0.25
        }
        
        # 通用高质量样式
        self.high_quality_style = {
            # 字体配置 - 使用默认serif字体
            'font.family': 'serif',
            'font.size': 10,
            'mathtext.fontset': 'dejavusans',  # 使用默认数学字体
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 14,
            
            # 超高分辨率
            'figure.figsize': (6.0, 4.5),
            'figure.dpi': 600,  # 超高DPI用于最终出版
            'savefig.dpi': 600,
            'savefig.format': 'pdf',
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            
            # 精细线条
            'lines.linewidth': 1.5,
            'lines.markersize': 6,
            'patch.linewidth': 0.8,
            'axes.linewidth': 1.2,
            
            # 专业网格
            'axes.grid': True,
            'grid.alpha': 0.15,
            'grid.linewidth': 0.8,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.facecolor': '#fafafa',  # 微妙的背景色
            'figure.facecolor': 'white'
        }
    
    def apply_journal_style(self, journal: str = 'nature') -> None:
        """
        应用指定期刊的样式配置
        
        Args:
            journal: 期刊名称 ('nature', 'science', 'cell', 'high_quality', 'custom')
        """
        style_map = {
            'nature': self.nature_style,
            'science': self.science_style,
            'cell': self.cell_style,
            'high_quality': self.high_quality_style,
            'custom': self.custom_style
        }
        
        if journal not in style_map:
            journal = 'nature'  # 默认使用Nature样式
        
        style = style_map[journal]
        
        # 应用样式配置
        for key, value in style.items():
            if key == 'axes.prop_cycle':
                # 特殊处理颜色循环
                rcParams[key] = eval(value)
            else:
                rcParams[key] = value
        
        # 确保LaTeX字体渲染(如果可用)
        try:
            rcParams['text.usetex'] = False  # 避免LaTeX依赖问题
            rcParams['pdf.fonttype'] = 42    # 确保PDF中的字体可编辑
            rcParams['ps.fonttype'] = 42     # PostScript字体类型
        except:
            pass
    
    def get_journal_figsize(self, journal: str = 'nature', 
                           layout: str = 'single') -> Tuple[float, float]:
        """
        获取期刊标准的图表尺寸
        
        Args:
            journal: 期刊名称
            layout: 布局类型 ('single', 'double', 'full')
            
        Returns:
            (宽度, 高度) 元组
        """
        base_sizes = {
            'nature': {
                'single': (5.2, 3.5),   # 单栏
                'double': (10.5, 7.0),  # 双栏  
                'full': (17.0, 11.0)    # 全页
            },
            'science': {
                'single': (5.5, 4.0),
                'double': (11.0, 8.0),
                'full': (18.0, 12.0)
            },
            'cell': {
                'single': (4.8, 3.6),
                'double': (9.6, 7.2),
                'full': (16.0, 10.5)
            },
            'custom': {
                'single': (6.0, 4.0),
                'double': (12.0, 8.0),
                'full': (18.0, 12.0)
            }
        }
        
        return base_sizes.get(journal, base_sizes['nature']).get(layout, (5.2, 3.5))
    
    def get_journal_colormap_config(self, journal: str = 'nature') -> Dict[str, Any]:
        """
        获取期刊特定的色彩映射配置
        
        Args:
            journal: 期刊名称
            
        Returns:
            色彩映射配置字典
        """
        colormap_configs = {
            'nature': {
                'sequential': 'Blues',
                'diverging': 'RdBu_r', 
                'qualitative': ['#0173B2', '#DE8F05', '#029E73', '#CC78BC'],
                'heatmap': 'YlOrRd'
            },
            'science': {
                'sequential': 'viridis',
                'diverging': 'coolwarm',
                'qualitative': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                'heatmap': 'RdYlBu_r'
            },
            'cell': {
                'sequential': 'plasma',
                'diverging': 'seismic',
                'qualitative': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'],
                'heatmap': 'coolwarm'
            }
        }
        
        return colormap_configs.get(journal, colormap_configs['nature'])

def apply_journal_style(journal: str = 'nature') -> None:
    """便捷函数：应用期刊样式"""
    styles = JournalStyles()
    styles.apply_journal_style(journal)
    # --- Publication legibility override (Revise4, Reviewer #9, Comment 5) ---
    # Manuscript figures are placed at text width; the per-journal presets above
    # use small (7-10 pt) fonts that become illegible once scaled down. Enforce
    # larger fonts / thicker lines / PNG output here. Rendering-only: this does
    # not change any plotted data values.
    rcParams.update({
        # 统一为标准无衬线字体 Liberation Sans（≈ Arial），数学符号用 stixsans 配套，
        # 取代默认的 DejaVu Sans，使图中英文更规范、与正文风格协调。仅排版，不改数据。
        'font.family': 'sans-serif',
        'font.sans-serif': ['Liberation Sans', 'Arial', 'Helvetica', 'Nimbus Sans', 'DejaVu Sans'],
        'mathtext.fontset': 'stixsans',
        'font.size': 13,
        'axes.titlesize': 15,
        'axes.labelsize': 13,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 16,
        'lines.linewidth': 2.0,
        'lines.markersize': 8,
        'axes.linewidth': 1.2,
        'grid.linewidth': 0.8,
        'savefig.format': 'png',
        'savefig.dpi': 600,
        'figure.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.08,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

def get_journal_config(journal: str = 'nature') -> Dict[str, Any]:
    """便捷函数：获取期刊完整配置"""
    styles = JournalStyles()
    return {
        'figsize': styles.get_journal_figsize(journal),
        'colormap': styles.get_journal_colormap_config(journal),
        'style_applied': journal
    }

def setup_publication_quality():
    """设置出版级质量的通用参数"""
    publication_params = {
        # 高质量渲染
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        
        # 专业字体
        'font.family': 'Arial',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        
        # 清晰边框
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 1.0,
        
        # 专业网格
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5
    }
    
    for key, value in publication_params.items():
        rcParams[key] = value

def create_subplot_layout(nrows: int, ncols: int, 
                         journal: str = 'nature',
                         **kwargs) -> Tuple[plt.Figure, np.ndarray]:
    """
    创建符合期刊标准的子图布局
    
    Args:
        nrows: 行数
        ncols: 列数  
        journal: 期刊样式
        **kwargs: 传递给plt.subplots的额外参数
        
    Returns:
        (figure, axes) 元组
    """
    styles = JournalStyles()
    
    # 根据子图数量调整整体尺寸
    base_width, base_height = styles.get_journal_figsize(journal, 'single')
    figsize = (base_width * ncols * 0.8, base_height * nrows * 0.8)
    
    # 设置默认参数
    subplot_kw = kwargs.pop('subplot_kw', {})
    subplot_kw.setdefault('facecolor', 'white')
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, 
                            subplot_kw=subplot_kw, **kwargs)
    
    # 应用期刊样式的间距调整
    style_spacing = {
        'nature': {'hspace': 0.3, 'wspace': 0.3},
        'science': {'hspace': 0.35, 'wspace': 0.35},
        'cell': {'hspace': 0.25, 'wspace': 0.25}
    }
    
    spacing = style_spacing.get(journal, style_spacing['nature'])
    plt.subplots_adjust(**spacing)
    
    return fig, axes

# 预定义的期刊样式配置
JOURNAL_CONFIGS = {
    'nature': {
        'figsize': (5.2, 3.5),
        'dpi': 300,
        'font_family': 'sans-serif',
        'font_size': 8,
        'title_size': 9,
        'label_size': 8,
        'colors': ['#0173B2', '#DE8F05', '#029E73', '#CC78BC']
    },
    'science': {
        'figsize': (5.5, 4.0),
        'dpi': 300,
        'font_family': 'sans-serif',
        'font_size': 9,
        'title_size': 10,
        'label_size': 9,
        'colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    },
    'cell': {
        'figsize': (4.8, 3.6),
        'dpi': 300,
        'font_family': 'sans-serif',
        'font_size': 8,
        'title_size': 9,
        'label_size': 8,
        'colors': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    }
}