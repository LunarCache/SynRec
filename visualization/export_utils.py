"""
高质量图表导出功能模块

提供多种专业级的图表导出选项：
1. 多种格式支持 (PDF, PNG, SVG, EPS)
2. 自定义DPI和质量设置
3. 批量导出和元数据管理
4. 期刊特定的导出模板
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
try:
    import matplotlib.backends.backend_pdf as pdf_backend
except ImportError:
    pdf_backend = None
import warnings

from .config import VisualizationConfig

class FigureExporter:
    """专业图表导出器"""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.export_history = []
        self._setup_output_directory()
    
    def _setup_output_directory(self):
        """设置输出目录结构"""
        base_dir = Path(self.config.output_directory)
        
        # 创建子目录结构
        self.directories = {
            'base': base_dir,
            'pdf': base_dir / 'pdf',
            'png': base_dir / 'png', 
            'svg': base_dir / 'svg',
            'eps': base_dir / 'eps',
            'metadata': base_dir / 'metadata'
        }
        
        # 创建所有目录
        for dir_path in self.directories.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def export_figure(self, 
                     fig: plt.Figure,
                     filename: str,
                     formats: Optional[List[str]] = None,
                     custom_dpi: Optional[int] = None,
                     custom_quality: Optional[str] = None,
                     add_timestamp: bool = False,
                     save_metadata: bool = True) -> Dict[str, List[str]]:
        """
        导出图表到指定格式
        
        Args:
            fig: matplotlib图表对象
            filename: 文件名（不含扩展名）
            formats: 导出格式列表
            custom_dpi: 自定义DPI
            custom_quality: 质量级别 ('draft', 'standard', 'high', 'publication')
            add_timestamp: 是否添加时间戳
            save_metadata: 是否保存元数据
            
        Returns:
            导出结果字典
        """
        if formats is None:
            formats = self.config.save_formats
        
        # 处理文件名
        if add_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename}_{timestamp}"
        
        # 设置DPI
        dpi = custom_dpi or self._get_quality_dpi(custom_quality)
        
        export_results = {
            'success': [],
            'failed': [],
            'metadata': {}
        }
        
        # 导出各种格式
        for fmt in formats:
            try:
                filepath = self._export_single_format(fig, filename, fmt, dpi)
                export_results['success'].append(str(filepath))
                
            except Exception as e:
                error_msg = f"格式 {fmt} 导出失败: {str(e)}"
                export_results['failed'].append(error_msg)
                warnings.warn(error_msg)
        
        # 保存元数据
        if save_metadata and export_results['success']:
            metadata = self._create_metadata(fig, filename, formats, dpi)
            metadata_path = self._save_metadata(metadata, filename)
            export_results['metadata'] = metadata
            export_results['metadata_path'] = str(metadata_path)
        
        # 记录导出历史
        self.export_history.append({
            'filename': filename,
            'formats': formats,
            'timestamp': datetime.now().isoformat(),
            'success_count': len(export_results['success']),
            'failed_count': len(export_results['failed'])
        })
        
        return export_results
    
    def _get_quality_dpi(self, quality: Optional[str]) -> int:
        """根据质量级别获取DPI"""
        quality_dpi_map = {
            'draft': 150,
            'standard': 300,
            'high': 600,
            'publication': 1200
        }
        
        if quality and quality in quality_dpi_map:
            return quality_dpi_map[quality]
        
        return self.config.dpi
    
    def _export_single_format(self, fig: plt.Figure, filename: str, 
                             fmt: str, dpi: int) -> Path:
        """导出单个格式"""
        output_dir = self.directories.get(fmt, self.directories['base'])
        filepath = output_dir / f"{filename}.{fmt}"
        
        # 格式特定的导出参数
        export_kwargs = {
            'dpi': dpi,
            'bbox_inches': 'tight' if self.config.tight_layout else None,
            'pad_inches': self.config.margin_inches,
            'transparent': self.config.transparent_background
        }
        
        # 格式特定设置
        if fmt == 'pdf':
            export_kwargs.update({
                'metadata': {
                    'Title': filename,
                    'Author': 'CMREC Enhanced Visualization',
                    'Subject': 'Scientific Figure',
                    'Creator': 'matplotlib + Enhanced Visualization Module',
                    'CreationDate': datetime.now()
                }
            })
        elif fmt == 'png':
            export_kwargs.update({
                'facecolor': 'white' if not self.config.transparent_background else None,
                'edgecolor': 'none'
            })
        elif fmt == 'svg':
            export_kwargs.update({
                'transparent': True,
                'metadata': {
                    'Date': datetime.now().isoformat()
                }
            })
        elif fmt == 'eps':
            export_kwargs['transparent'] = False  # EPS不支持透明
        
        # 执行导出
        fig.savefig(filepath, format=fmt, **export_kwargs)
        
        return filepath
    
    def _create_metadata(self, fig: plt.Figure, filename: str, 
                        formats: List[str], dpi: int) -> Dict[str, Any]:
        """创建图表元数据"""
        # 获取图表信息
        axes_info = []
        for i, ax in enumerate(fig.get_axes()):
            ax_info = {
                'index': i,
                'title': ax.get_title(),
                'xlabel': ax.get_xlabel(),
                'ylabel': ax.get_ylabel(),
                'xlim': ax.get_xlim(),
                'ylim': ax.get_ylim(),
                'has_legend': ax.get_legend() is not None
            }
            axes_info.append(ax_info)
        
        metadata = {
            'filename': filename,
            'export_timestamp': datetime.now().isoformat(),
            'formats': formats,
            'dpi': dpi,
            'figure_size': fig.get_size_inches().tolist(),
            'num_axes': len(fig.get_axes()),
            'axes_info': axes_info,
            'config': {
                'journal_style': self.config.journal_style,
                'font_family': self.config.font_family,
                'font_size': self.config.font_size,
                'colorblind_friendly': self.config.colorblind_friendly
            }
        }
        
        return metadata
    
    def _save_metadata(self, metadata: Dict[str, Any], filename: str) -> Path:
        """保存元数据到JSON文件"""
        metadata_path = self.directories['metadata'] / f"{filename}_metadata.json"
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        
        return metadata_path
    
    def batch_export(self, figures: Dict[str, plt.Figure],
                    formats: Optional[List[str]] = None,
                    quality: str = 'standard') -> Dict[str, Dict]:
        """
        批量导出多个图表
        
        Args:
            figures: 图表字典 {filename: figure}
            formats: 导出格式
            quality: 质量级别
            
        Returns:
            批量导出结果
        """
        batch_results = {}
        
        for filename, fig in figures.items():
            try:
                result = self.export_figure(
                    fig, filename, formats, 
                    custom_quality=quality, save_metadata=True
                )
                batch_results[filename] = result
                
            except Exception as e:
                batch_results[filename] = {
                    'success': [],
                    'failed': [f"批量导出失败: {str(e)}"],
                    'metadata': {}
                }
        
        # 保存批量导出摘要
        self._save_batch_summary(batch_results)
        
        return batch_results
    
    def _save_batch_summary(self, batch_results: Dict[str, Dict]):
        """保存批量导出摘要"""
        summary = {
            'batch_timestamp': datetime.now().isoformat(),
            'total_figures': len(batch_results),
            'successful_exports': sum(
                len(result['success']) for result in batch_results.values()
            ),
            'failed_exports': sum(
                len(result['failed']) for result in batch_results.values()
            ),
            'details': batch_results
        }
        
        summary_path = self.directories['metadata'] / f"batch_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    
    def create_journal_template(self, journal: str = 'nature') -> Dict[str, Any]:
        """
        创建期刊特定的导出模板
        
        Args:
            journal: 期刊名称
            
        Returns:
            导出模板配置
        """
        journal_templates = {
            'nature': {
                'formats': ['pdf', 'png'],
                'dpi': 300,
                'max_width_inches': 5.2,
                'max_height_inches': 7.0,
                'font_requirements': {
                    'family': 'Arial',
                    'min_size': 6,
                    'max_size': 12
                },
                'color_requirements': {
                    'colorblind_friendly': True,
                    'grayscale_compatible': True
                }
            },
            'science': {
                'formats': ['pdf', 'svg'],
                'dpi': 300,
                'max_width_inches': 5.5,
                'max_height_inches': 8.0,
                'font_requirements': {
                    'family': 'Arial',
                    'min_size': 7,
                    'max_size': 14
                },
                'color_requirements': {
                    'colorblind_friendly': True,
                    'grayscale_compatible': True
                }
            },
            'cell': {
                'formats': ['pdf', 'png'],
                'dpi': 300,
                'max_width_inches': 4.8,
                'max_height_inches': 6.5,
                'font_requirements': {
                    'family': 'Arial',
                    'min_size': 6,
                    'max_size': 11
                },
                'color_requirements': {
                    'colorblind_friendly': True,
                    'grayscale_compatible': True
                }
            }
        }
        
        return journal_templates.get(journal, journal_templates['nature'])
    
    def validate_figure_for_journal(self, fig: plt.Figure, 
                                   journal: str = 'nature') -> Dict[str, Any]:
        """
        验证图表是否符合期刊要求
        
        Args:
            fig: 图表对象
            journal: 期刊名称
            
        Returns:
            验证结果
        """
        template = self.create_journal_template(journal)
        validation_results = {
            'passed': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        # 检查图表尺寸
        fig_width, fig_height = fig.get_size_inches()
        max_width = template['max_width_inches']
        max_height = template['max_height_inches']
        
        if fig_width > max_width:
            validation_results['errors'].append(
                f"图表宽度 {fig_width:.1f}英寸 超过{journal}期刊限制 {max_width}英寸"
            )
            validation_results['passed'] = False
        
        if fig_height > max_height:
            validation_results['errors'].append(
                f"图表高度 {fig_height:.1f}英寸 超过{journal}期刊限制 {max_height}英寸"
            )
            validation_results['passed'] = False
        
        # 检查字体大小
        font_reqs = template['font_requirements']
        for ax in fig.get_axes():
            # 检查标题字体
            title_size = ax.title.get_fontsize()
            if title_size < font_reqs['min_size']:
                validation_results['warnings'].append(
                    f"标题字体过小: {title_size}pt, 建议最小 {font_reqs['min_size']}pt"
                )
            elif title_size > font_reqs['max_size']:
                validation_results['warnings'].append(
                    f"标题字体过大: {title_size}pt, 建议最大 {font_reqs['max_size']}pt"
                )
            
            # 检查轴标签字体
            xlabel_size = ax.xaxis.label.get_fontsize()
            ylabel_size = ax.yaxis.label.get_fontsize()
            
            for label, size in [('X轴标签', xlabel_size), ('Y轴标签', ylabel_size)]:
                if size < font_reqs['min_size']:
                    validation_results['warnings'].append(
                        f"{label}字体过小: {size}pt, 建议最小 {font_reqs['min_size']}pt"
                    )
        
        # 添加建议
        if not validation_results['errors']:
            validation_results['recommendations'].extend([
                f"确保所有文本在{journal}期刊的打印版本中清晰可读",
                "考虑在提交前检查灰度模式下的显示效果",
                "验证所有颜色在色盲读者看来是否可区分"
            ])
        
        return validation_results
    
    def get_export_summary(self) -> Dict[str, Any]:
        """获取导出历史摘要"""
        if not self.export_history:
            return {'total_exports': 0, 'history': []}
        
        total_success = sum(h['success_count'] for h in self.export_history)
        total_failed = sum(h['failed_count'] for h in self.export_history)
        
        return {
            'total_exports': len(self.export_history),
            'total_successful_files': total_success,
            'total_failed_files': total_failed,
            'success_rate': total_success / (total_success + total_failed) if (total_success + total_failed) > 0 else 0,
            'recent_exports': self.export_history[-5:],  # 最近5次导出
            'export_directory': str(self.directories['base'])
        }

# 便捷函数
def export_figure_journal(fig: plt.Figure, filename: str, 
                         journal: str = 'nature',
                         quality: str = 'standard') -> List[str]:
    """
    便捷函数：按期刊标准导出图表
    
    Args:
        fig: 图表对象
        filename: 文件名
        journal: 期刊名称
        quality: 质量级别
        
    Returns:
        导出文件路径列表
    """
    from .config import create_journal_config
    
    config = create_journal_config(journal)
    exporter = FigureExporter(config)
    
    result = exporter.export_figure(
        fig, filename, 
        custom_quality=quality,
        save_metadata=True
    )
    
    return result['success']

def create_publication_package(figures: Dict[str, plt.Figure],
                              package_name: str = 'publication_figures',
                              journal: str = 'nature') -> str:
    """
    创建出版级图表包
    
    Args:
        figures: 图表字典
        package_name: 包名称
        journal: 目标期刊
        
    Returns:
        包输出目录路径
    """
    from .config import create_journal_config
    
    config = create_journal_config(journal)
    config.output_directory = f"{package_name}_{journal}"
    
    exporter = FigureExporter(config)
    
    # 批量导出
    results = exporter.batch_export(figures, quality='publication')
    
    # 生成README文件
    readme_path = Path(config.output_directory) / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(f"# {package_name} - {journal.title()} Journal Format\\n\\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
        f.write("## Contents\\n\\n")
        
        for filename, result in results.items():
            f.write(f"- **{filename}**: {len(result['success'])} files exported\\n")
        
        f.write("\\n## Quality Settings\\n\\n")
        f.write(f"- DPI: {config.dpi}\\n")
        f.write(f"- Formats: {', '.join(config.save_formats)}\\n")
        f.write(f"- Journal Style: {journal}\\n")
    
    return config.output_directory