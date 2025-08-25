"""
增强可视化配置示例

这个文件展示了如何创建和使用各种可视化配置。
"""

from visualization import (
    VisualizationConfig,
    ConfigManager,
    create_journal_config,
    apply_journal_style
)

def create_nature_config():
    """创建Nature期刊配置示例"""
    config = VisualizationConfig(
        journal_style='nature',
        figure_width=5.2,
        figure_height=3.5,
        dpi=300,
        font_family='sans-serif',
        font_size=8,
        title_size=9,
        label_size=8,
        colorblind_friendly=True,
        save_formats=['pdf', 'png'],
        output_directory='nature_figures',
        tight_layout=True
    )
    return config

def create_science_config():
    """创建Science期刊配置示例"""
    config = VisualizationConfig(
        journal_style='science',
        figure_width=5.5,
        figure_height=4.0,
        dpi=300,
        font_family='sans-serif',
        font_size=9,
        title_size=10,
        label_size=9,
        attention_colormap='viridis',
        heatmap_colormap='RdYlBu_r',
        save_formats=['pdf', 'svg'],
        output_directory='science_figures'
    )
    return config

def create_high_quality_config():
    """创建高质量出版配置示例"""
    config = VisualizationConfig(
        journal_style='high_quality',
        figure_width=6.0,
        figure_height=4.5,
        dpi=600,  # 超高DPI
        font_family='serif',
        font_size=10,
        title_size=12,
        label_size=10,
        save_formats=['pdf', 'png', 'svg', 'eps'],  # 全格式
        high_quality_export=True,
        transparent_background=False,
        output_directory='publication_figures'
    )
    return config

def create_presentation_config():
    """创建演示文稿配置示例"""
    config = VisualizationConfig(
        journal_style='science',
        figure_width=8.0,
        figure_height=6.0,
        dpi=150,  # 较低DPI用于快速预览
        font_family='sans-serif',
        font_size=12,  # 大字号便于投影
        title_size=16,
        label_size=14,
        legend_size=12,
        save_formats=['png'],
        output_directory='presentation_figures'
    )
    return config

def save_all_presets():
    """保存所有预设配置"""
    manager = ConfigManager()
    
    configs = {
        'nature_standard': create_nature_config(),
        'science_standard': create_science_config(),
        'high_quality': create_high_quality_config(),
        'presentation': create_presentation_config()
    }
    
    for name, config in configs.items():
        manager.save_config(config, name)
        print(f"✓ 已保存配置: {name}")

def load_and_apply_config(config_name='nature_standard'):
    """加载并应用配置示例"""
    manager = ConfigManager()
    
    try:
        config = manager.load_config(config_name)
        apply_journal_style(config.journal_style)
        print(f"✓ 已加载并应用配置: {config_name}")
        return config
    except Exception as e:
        print(f"✗ 配置加载失败: {e}")
        return None

def main():
    """运行配置示例"""
    print("=== 增强可视化配置示例 ===\\n")
    
    # 1. 创建并保存预设配置
    print("1. 创建预设配置...")
    save_all_presets()
    
    # 2. 列出所有配置
    print("\\n2. 可用配置:")
    manager = ConfigManager()
    configs = manager.list_configs()
    for config in configs:
        print(f"  - {config}")
    
    # 3. 加载和使用配置
    print("\\n3. 加载配置示例:")
    config = load_and_apply_config('nature_standard')
    if config:
        print(f"  期刊样式: {config.journal_style}")
        print(f"  图表尺寸: {config.figure_width}×{config.figure_height}英寸")
        print(f"  分辨率: {config.dpi} DPI")
        print(f"  输出格式: {', '.join(config.save_formats)}")

if __name__ == "__main__":
    main()