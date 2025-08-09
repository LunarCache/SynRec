"""
专业可视化模块 - 符合SCI顶级期刊标准

这个模块提供了针对CMREC模型的高质量可视化功能，
专门设计以满足Nature、Science等顶级期刊的图表要求。

主要功能:
- 期刊级别的配色方案
- 标准化的字体和布局
- 高分辨率矢量图输出
- 多种期刊样式模板
"""

from .color_schemes import (
    JournalColorSchemes,
    get_journal_palette,
    get_colorblind_friendly_colors
)
from .journal_styles import (
    JournalStyles,
    apply_journal_style,
    get_journal_config
)
from .config import (
    VisualizationConfig,
    load_config,
    save_config,
    create_journal_config
)
from .enhanced_plots import (
    EnhancedVisualization,
    plot_fourier_attention_journal,
    plot_expert_routing_journal,
    plot_tsne_specialization_journal,
    plot_multi_domain_fourier_comparison_journal
)
from .export_utils import (
    FigureExporter,
    export_figure_journal,
    create_publication_package
)

__version__ = "1.0.0"
__author__ = "CMREC Team"

# 默认配置
DEFAULT_JOURNAL = "nature"
DEFAULT_DPI = 300
DEFAULT_FORMAT = "pdf"

# 导出主要接口
__all__ = [
    "JournalColorSchemes",
    "JournalStyles", 
    "VisualizationConfig",
    "EnhancedVisualization",
    "plot_fourier_attention_journal",
    "plot_expert_routing_journal", 
    "plot_tsne_specialization_journal",
    "plot_multi_domain_fourier_comparison_journal",
    "FigureExporter",
    "export_figure_journal",
    "create_publication_package",
    "create_journal_config",
    "get_journal_palette",
    "apply_journal_style",
    "load_config"
]