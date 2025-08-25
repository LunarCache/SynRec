"""
可视化配置管理模块

提供统一的配置管理接口，支持：
1. 配置文件的加载和保存
2. 运行时配置的动态调整
3. 多种输出格式的配置
4. 期刊特定配置的快速切换
"""

import json
import os
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
import warnings

@dataclass
class VisualizationConfig:
    """可视化配置数据类"""
    
    # 期刊样式配置
    journal_style: str = 'nature'
    use_journal_colors: bool = True
    
    # 图表尺寸和质量
    figure_width: float = 5.2
    figure_height: float = 3.5
    dpi: int = 300
    figure_format: str = 'pdf'
    
    # 字体配置
    font_family: str = 'sans-serif'
    font_size: int = 8
    title_size: int = 9
    label_size: int = 8
    legend_size: int = 7
    
    # 色彩配置
    colorblind_friendly: bool = True
    use_custom_colormap: bool = False
    custom_colors: List[str] = field(default_factory=lambda: [])
    
    # 输出配置
    save_formats: List[str] = field(default_factory=lambda: ['pdf', 'png'])
    output_directory: str = 'figures'
    high_quality_export: bool = True
    transparent_background: bool = False
    
    # 布局配置
    tight_layout: bool = True
    subplot_spacing: float = 0.3
    margin_inches: float = 0.1
    
    # 专门的可视化配置
    attention_colormap: str = 'Blues'
    heatmap_colormap: str = 'YlOrRd'
    scatter_alpha: float = 0.7
    line_width: float = 1.0
    marker_size: float = 4.0
    
    # 平滑处理配置
    enable_smoothing: bool = True
    smooth_sigma: float = 0.6
    smooth_method: str = 'gaussian'  # 'gaussian', 'bilinear', 'bicubic'
    color_levels: int = 512  # 色彩层次数量
    interpolation_method: str = 'bilinear'  # 'nearest', 'bilinear', 'bicubic'
    
    # 高级配置
    enable_grid: bool = True
    grid_alpha: float = 0.3
    remove_top_spine: bool = True
    remove_right_spine: bool = True
    
    # 文本和注释
    use_math_fonts: bool = False
    enable_latex: bool = False
    annotation_fontsize: int = 7
    
    def __post_init__(self):
        """配置验证和初始化后处理"""
        self._validate_config()
        self._setup_derived_configs()
    
    def _validate_config(self):
        """验证配置参数的有效性"""
        if self.dpi < 72:
            warnings.warn("DPI过低，建议至少使用300DPI用于出版级质量")
        
        if self.figure_format not in ['pdf', 'png', 'svg', 'eps']:
            warnings.warn(f"不支持的图片格式: {self.figure_format}")
            self.figure_format = 'pdf'
        
        if self.journal_style not in ['nature', 'science', 'cell', 'high_quality']:
            warnings.warn(f"未知的期刊样式: {self.journal_style}")
            self.journal_style = 'nature'
    
    def _setup_derived_configs(self):
        """根据主要配置设置派生配置"""
        # 根据期刊样式调整默认参数
        journal_defaults = {
            'nature': {
                'figure_width': 5.2, 'figure_height': 3.5,
                'font_family': 'sans-serif', 'font_size': 8, 'title_size': 9
            },
            'science': {
                'figure_width': 5.5, 'figure_height': 4.0,
                'font_family': 'sans-serif', 'font_size': 9, 'title_size': 10
            },
            'cell': {
                'figure_width': 4.8, 'figure_height': 3.6,
                'font_family': 'sans-serif', 'font_size': 8, 'title_size': 9
            },
            'high_quality': {
                'figure_width': 6.0, 'figure_height': 4.5,
                'font_family': 'serif', 'font_size': 10, 'title_size': 12, 'dpi': 600
            }
        }
        
        if self.journal_style in journal_defaults:
            defaults = journal_defaults[self.journal_style]
            for key, value in defaults.items():
                if hasattr(self, key):
                    setattr(self, key, value)
    
    def get_figsize(self, layout: str = 'single') -> Tuple[float, float]:
        """
        获取图表尺寸
        
        Args:
            layout: 布局类型 ('single', 'double', 'full')
            
        Returns:
            (宽度, 高度) 元组
        """
        multipliers = {
            'single': (1.0, 1.0),
            'double': (2.0, 1.4), 
            'full': (3.2, 2.2)
        }
        
        mult_w, mult_h = multipliers.get(layout, (1.0, 1.0))
        return (self.figure_width * mult_w, self.figure_height * mult_h)
    
    def get_font_config(self) -> Dict[str, Any]:
        """获取字体配置字典"""
        return {
            'family': self.font_family,
            'size': self.font_size,
            'weight': 'normal'
        }
    
    def get_colormap_config(self) -> Dict[str, str]:
        """获取色彩映射配置"""
        return {
            'attention': self.attention_colormap,
            'heatmap': self.heatmap_colormap,
            'diverging': 'RdBu_r',
            'sequential': 'viridis'
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'VisualizationConfig':
        """从字典创建配置实例"""
        return cls(**config_dict)
    
    def update(self, **kwargs) -> None:
        """更新配置参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                warnings.warn(f"未知的配置参数: {key}")
    
    def apply_journal_preset(self, journal: str) -> None:
        """应用期刊预设配置"""
        self.journal_style = journal
        self._setup_derived_configs()

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: str = 'configs'):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.current_config = VisualizationConfig()
    
    def save_config(self, config: VisualizationConfig, 
                   name: str = 'default') -> None:
        """
        保存配置到文件
        
        Args:
            config: 配置实例
            name: 配置名称
        """
        config_path = self.config_dir / f"{name}.json"
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
            print(f"配置已保存到: {config_path}")
        except Exception as e:
            print(f"保存配置失败: {e}")
    
    def load_config(self, name: str = 'default') -> VisualizationConfig:
        """
        从文件加载配置
        
        Args:
            name: 配置名称
            
        Returns:
            配置实例
        """
        config_path = self.config_dir / f"{name}.json"
        
        if not config_path.exists():
            print(f"配置文件不存在: {config_path}，使用默认配置")
            return VisualizationConfig()
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            config = VisualizationConfig.from_dict(config_dict)
            self.current_config = config
            return config
        except Exception as e:
            print(f"加载配置失败: {e}，使用默认配置")
            return VisualizationConfig()
    
    def list_configs(self) -> List[str]:
        """列出所有可用的配置"""
        config_files = list(self.config_dir.glob("*.json"))
        return [f.stem for f in config_files]
    
    def create_journal_presets(self) -> None:
        """创建期刊预设配置"""
        journals = ['nature', 'science', 'cell', 'high_quality']
        
        for journal in journals:
            config = VisualizationConfig()
            config.apply_journal_preset(journal)
            self.save_config(config, f"preset_{journal}")
        
        print(f"已创建期刊预设配置: {journals}")
    
    def get_current_config(self) -> VisualizationConfig:
        """获取当前配置"""
        return self.current_config
    
    def update_current_config(self, **kwargs) -> None:
        """更新当前配置"""
        self.current_config.update(**kwargs)

# 全局配置管理器实例
_config_manager = ConfigManager()

def load_config(name: str = 'default') -> VisualizationConfig:
    """便捷函数：加载配置"""
    return _config_manager.load_config(name)

def save_config(config: VisualizationConfig, name: str = 'default') -> None:
    """便捷函数：保存配置"""
    _config_manager.save_config(config, name)

def get_current_config() -> VisualizationConfig:
    """便捷函数：获取当前配置"""
    return _config_manager.get_current_config()

def create_journal_config(journal: str) -> VisualizationConfig:
    """
    便捷函数：创建期刊特定配置
    
    Args:
        journal: 期刊名称
        
    Returns:
        配置实例
    """
    config = VisualizationConfig()
    config.apply_journal_preset(journal)
    return config

def setup_visualization_environment(config: Optional[VisualizationConfig] = None) -> None:
    """
    设置可视化环境
    
    Args:
        config: 配置实例，如果为None则使用默认配置
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    
    if config is None:
        config = VisualizationConfig()
    
    # 应用matplotlib配置
    plt.rcParams.update({
        'figure.figsize': (config.figure_width, config.figure_height),
        'figure.dpi': config.dpi,
        'savefig.dpi': config.dpi,
        'savefig.format': config.figure_format,
        'savefig.bbox': 'tight' if config.tight_layout else None,
        'savefig.pad_inches': config.margin_inches,
        'savefig.transparent': config.transparent_background,
        
        'font.family': config.font_family,
        'font.size': config.font_size,
        'axes.titlesize': config.title_size,
        'axes.labelsize': config.label_size,
        'legend.fontsize': config.legend_size,
        
        'lines.linewidth': config.line_width,
        'lines.markersize': config.marker_size,
        
        'axes.grid': config.enable_grid,
        'grid.alpha': config.grid_alpha,
        'axes.spines.top': not config.remove_top_spine,
        'axes.spines.right': not config.remove_right_spine,
        
        'text.usetex': config.enable_latex,
        'mathtext.fontset': 'stix' if config.use_math_fonts else 'dejavusans'
    })
    
    # 确保输出目录存在
    output_dir = Path(config.output_directory)
    output_dir.mkdir(exist_ok=True)

# 预定义配置模板
CONFIG_TEMPLATES = {
    'nature_single': {
        'journal_style': 'nature',
        'figure_width': 5.2,
        'figure_height': 3.5,
        'font_size': 8,
        'dpi': 300
    },
    'nature_double': {
        'journal_style': 'nature', 
        'figure_width': 10.5,
        'figure_height': 7.0,
        'font_size': 9,
        'dpi': 300
    },
    'science_presentation': {
        'journal_style': 'science',
        'figure_width': 8.0,
        'figure_height': 6.0,
        'font_size': 12,
        'dpi': 150
    },
    'high_quality_print': {
        'journal_style': 'high_quality',
        'figure_width': 6.0,
        'figure_height': 4.5,
        'font_size': 10,
        'dpi': 600,
        'figure_format': 'pdf'
    }
}