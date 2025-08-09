"""
增强的可视化绘图模块

提供期刊级别的三个核心可视化功能：
1. Fourier评分注意力可视化
2. 领域-专家路由热力图
3. t-SNE专家专业化分析

所有函数都符合SCI顶级期刊的视觉标准和技术要求。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.manifold import TSNE
import torch
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
from pathlib import Path
from scipy import ndimage  # 添加用于平滑处理
from scipy.interpolate import interp2d  # 添加用于插值

from .config import VisualizationConfig, get_current_config
from .color_schemes import JournalColorSchemes, get_journal_palette
from .journal_styles import apply_journal_style
import gc

class EnhancedVisualization:
    """增强的可视化类，提供期刊级别的图表生成"""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or get_current_config()
        self.color_schemes = JournalColorSchemes()
        self._setup_style()
    
    def _setup_style(self):
        """设置可视化样式"""
        apply_journal_style(self.config.journal_style)
        
        # 应用自定义配置
        plt.rcParams.update({
            'figure.figsize': self.config.get_figsize(),
            'figure.dpi': self.config.dpi,
            'font.size': self.config.font_size,
            'axes.titlesize': self.config.title_size,
            'axes.labelsize': self.config.label_size,
            'legend.fontsize': self.config.legend_size,
            'lines.linewidth': self.config.line_width,
            'lines.markersize': self.config.marker_size,
            'axes.grid': self.config.enable_grid,
            'grid.alpha': self.config.grid_alpha,
            'savefig.dpi': self.config.dpi,
            'savefig.format': self.config.figure_format,
            'savefig.bbox': 'tight' if self.config.tight_layout else None,
            'savefig.pad_inches': self.config.margin_inches,
            'savefig.transparent': self.config.transparent_background
        })
    
    def _create_figure_layout(self, layout: str = 'single', 
                             nrows: int = 1, ncols: int = 1) -> Tuple[plt.Figure, np.ndarray]:
        """创建期刊标准的图表布局"""
        figsize = self.config.get_figsize(layout)
        
        # 根据子图数量调整尺寸
        if nrows > 1 or ncols > 1:
            figsize = (figsize[0] * ncols * 0.9, figsize[1] * nrows * 0.9)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                                facecolor='white', edgecolor='none')
        
        # 标准化axes为数组格式
        if nrows == 1 and ncols == 1:
            axes = np.array([axes])
        elif nrows == 1 or ncols == 1:
            axes = axes.flatten()
        
        # 应用期刊样式的间距
        plt.subplots_adjust(
            hspace=self.config.subplot_spacing,
            wspace=self.config.subplot_spacing,
            left=0.1, right=0.95, top=0.9, bottom=0.15
        )
        
        return fig, axes
    
    def _apply_professional_styling(self, ax: plt.Axes, 
                                  title: str = "", xlabel: str = "", ylabel: str = "") -> None:
        """应用专业的坐标轴样式"""
        if title:
            ax.set_title(title, fontsize=self.config.title_size, 
                        fontweight='bold', pad=10)
        
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=self.config.label_size, 
                         fontweight='normal')
        
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=self.config.label_size, 
                         fontweight='normal')
        
        # 移除顶部和右侧边框（期刊标准）
        if self.config.remove_top_spine:
            ax.spines['top'].set_visible(False)
        if self.config.remove_right_spine:
            ax.spines['right'].set_visible(False)
        
        # 设置边框样式
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color('#333333')
        
        # 设置刻度样式
        ax.tick_params(axis='both', which='major', 
                      labelsize=self.config.font_size,
                      color='#333333', width=0.8, length=4)
        ax.tick_params(axis='both', which='minor', 
                      color='#333333', width=0.6, length=2)
        
        # 网格样式
        if self.config.enable_grid:
            ax.grid(True, alpha=self.config.grid_alpha, 
                   linestyle='-', linewidth=0.5, color='#cccccc')
            ax.set_axisbelow(True)
    
    def _smooth_attention_matrix(self, attention_matrix: np.ndarray, 
                                smooth_method: str = 'gaussian',
                                smooth_sigma: float = 0.8,
                                upsample_factor: int = 2) -> np.ndarray:
        """
        对注意力矩阵进行平滑处理，提升视觉效果
        
        Args:
            attention_matrix: 注意力矩阵
            smooth_method: 平滑方法 ('gaussian', 'bilinear', 'bicubic')
            smooth_sigma: 高斯平滑的sigma参数
            upsample_factor: 上采样倍数
            
        Returns:
            平滑后的注意力矩阵
        """
        if attention_matrix.size == 0:
            return attention_matrix
        
        try:
            # 第一步：轻微的高斯平滑，去除噪声但保持结构
            if smooth_method == 'gaussian':
                smoothed = ndimage.gaussian_filter(attention_matrix, sigma=smooth_sigma)
            else:
                smoothed = attention_matrix.copy()
            
            # 第二步：如果需要，进行上采样以提高分辨率
            if upsample_factor > 1:
                from scipy.ndimage import zoom
                smoothed = zoom(smoothed, upsample_factor, order=3)  # 双三次插值
            
            # 确保数值范围正确
            smoothed = np.clip(smoothed, 0, None)
            
            # 重新归一化（如果原矩阵是概率分布）
            if np.allclose(attention_matrix.sum(axis=-1), 1.0, rtol=1e-3):
                # 按行归一化
                row_sums = smoothed.sum(axis=-1, keepdims=True)
                row_sums[row_sums == 0] = 1  # 避免除零
                smoothed = smoothed / row_sums
            
            return smoothed
            
        except Exception as e:
            warnings.warn(f"注意力平滑处理失败: {e}")
            return attention_matrix
    
    def _create_smooth_colormap(self, base_cmap: str, n_levels: int = 256) -> mcolors.ListedColormap:
        """
        创建更平滑的色彩映射
        
        Args:
            base_cmap: 基础色彩映射名称
            n_levels: 色彩层次数量
            
        Returns:
            平滑的色彩映射
        """
        try:
            # 获取基础色彩映射
            base = plt.cm.get_cmap(base_cmap)
            
            # 生成更多的色彩层次
            colors = base(np.linspace(0, 1, n_levels))
            
            # 创建平滑色彩映射
            smooth_cmap = mcolors.ListedColormap(colors, name=f'smooth_{base_cmap}')
            
            return smooth_cmap
            
        except Exception:
            return plt.cm.get_cmap(base_cmap)
    
    def _save_figure(self, fig: plt.Figure, filename: str, 
                    additional_formats: Optional[List[str]] = None) -> List[str]:
        """保存图表到多种格式"""
        output_dir = Path(self.config.output_directory)
        output_dir.mkdir(exist_ok=True)
        
        saved_files = []
        formats = self.config.save_formats.copy()
        
        if additional_formats:
            formats.extend(additional_formats)
        
        # 去重并保持顺序
        formats = list(dict.fromkeys(formats))
        
        for fmt in formats:
            filepath = output_dir / f"{filename}.{fmt}"
            
            try:
                # 设置格式特定的参数
                save_kwargs = {
                    'dpi': self.config.dpi,
                    'bbox_inches': 'tight' if self.config.tight_layout else None,
                    'pad_inches': self.config.margin_inches,
                    'transparent': self.config.transparent_background
                }
                
                if fmt == 'pdf':
                    save_kwargs['metadata'] = {
                        'Title': filename,
                        'Author': 'CMREC Visualization System',
                        'Creator': 'Enhanced Visualization Module'
                    }
                elif fmt == 'png':
                    if self.config.high_quality_export:
                        save_kwargs['dpi'] = max(self.config.dpi, 300)
                elif fmt == 'svg':
                    save_kwargs['transparent'] = True
                
                fig.savefig(filepath, format=fmt, **save_kwargs)
                saved_files.append(str(filepath))
                
            except Exception as e:
                warnings.warn(f"保存格式 {fmt} 失败: {e}")
        
        return saved_files

def plot_fourier_attention_journal(attention_data: Dict[str, torch.Tensor],
                                  domain_info: Dict[str, Any],
                                  layer_idx: int,
                                  epoch: int,
                                  config: Optional[VisualizationConfig] = None,
                                  save_plots: bool = True) -> Tuple[plt.Figure, List[str]]:
    """
    期刊级Fourier注意力可视化
    
    Args:
        attention_data: 包含注意力矩阵的字典
        domain_info: 领域信息
        layer_idx: 层索引
        epoch: 训练轮次
        config: 可视化配置
        save_plots: 是否保存图片
        
    Returns:
        (figure, saved_files) 元组
    """
    viz = EnhancedVisualization(config)
    
    # 提取数据
    branch1 = attention_data.get('branch1')
    branch2 = attention_data.get('branch2') 
    adaptive_weights = attention_data.get('adaptive_weights')
    
    if branch1 is None or branch2 is None:
        raise ValueError("缺少必要的注意力分支数据")
    
    # 转换为numpy并处理维度
    branch1_np = branch1.detach().cpu().numpy()
    branch2_np = branch2.detach().cpu().numpy()
    
    # 处理不同的维度情况
    if branch1_np.ndim == 4:  # (batch, heads, seq, seq)
        branch1_np = branch1_np.mean(axis=(0, 1))  # 平均over batch和heads
    elif branch1_np.ndim == 3:  # (batch, seq, seq) - 已平均over heads
        branch1_np = branch1_np.mean(axis=0)  # 平均over batch
    elif branch1_np.ndim == 2:  # (seq, seq) - 单样本
        pass  # 直接使用
    
    if branch2_np.ndim == 4:  # (batch, heads, seq, seq)
        branch2_np = branch2_np.mean(axis=(0, 1))  # 平均over batch和heads
    elif branch2_np.ndim == 3:  # (batch, seq, seq) - 已平均over heads
        branch2_np = branch2_np.mean(axis=0)  # 平均over batch
    elif branch2_np.ndim == 2:  # (seq, seq) - 单样本
        pass  # 直接使用
    
    # 获取有效序列长度
    effective_len = domain_info.get('effective_length', min(50, branch1_np.shape[0]))
    branch1_np = branch1_np[:effective_len, :effective_len]
    branch2_np = branch2_np[:effective_len, :effective_len]
    
    # 应用平滑处理以获得更好的视觉效果（如果启用）
    if viz.config.enable_smoothing:
        branch1_smooth = viz._smooth_attention_matrix(branch1_np, 
                                                     smooth_method=viz.config.smooth_method,
                                                     smooth_sigma=viz.config.smooth_sigma,
                                                     upsample_factor=1)
        branch2_smooth = viz._smooth_attention_matrix(branch2_np,
                                                     smooth_method=viz.config.smooth_method, 
                                                     smooth_sigma=viz.config.smooth_sigma,
                                                     upsample_factor=1)
    else:
        branch1_smooth = branch1_np
        branch2_smooth = branch2_np
    
    # 创建三子图布局
    fig, axes = viz._create_figure_layout('single', 1, 3)
    
    try:
        # 获取期刊配色
        colors = viz.color_schemes.get_journal_palette(viz.config.journal_style, 3)
        
        # 创建平滑的色彩映射 - 使用与multi_domain_comparison一致的配色
        smooth_cmap_viridis = viz._create_smooth_colormap('viridis', n_levels=512)
        smooth_cmap_plasma = viz._create_smooth_colormap('plasma', n_levels=512)
        smooth_cmap_blue_yellow = viz._create_smooth_colormap('RdYlBu_r', n_levels=512)  # 蓝黄平滑配色
        
        # 子图1: 短期注意力模式 (高频) - 使用viridis配色与多领域对比保持一致
        im1 = axes[0].imshow(branch1_smooth, cmap=smooth_cmap_viridis, aspect='auto', 
                            interpolation='bilinear', alpha=0.95,
                            extent=[0, effective_len, effective_len, 0])  # 设置正确的坐标范围
        viz._apply_professional_styling(
            axes[0], 
            title='Fourier Branch 1',
            xlabel='Position',
            ylabel='Position'
        )
        
        # 添加颜色条
        cbar1 = plt.colorbar(im1, ax=axes[0], shrink=0.8, aspect=15)
        cbar1.set_label('Attention Weight', fontsize=viz.config.font_size)
        cbar1.ax.tick_params(labelsize=viz.config.font_size-1)
        
        # 子图2: 长期注意力模式 (低频) - 使用plasma配色与多领域对比保持一致
        im2 = axes[1].imshow(branch2_smooth, cmap=smooth_cmap_plasma, aspect='auto',
                            interpolation='bilinear', alpha=0.95,
                            extent=[0, effective_len, effective_len, 0])
        viz._apply_professional_styling(
            axes[1],
            title='Fourier Branch 2', 
            xlabel='Position',
            ylabel='Position'
        )
        
        cbar2 = plt.colorbar(im2, ax=axes[1], shrink=0.8, aspect=15)
        cbar2.set_label('Attention Weight', fontsize=viz.config.font_size)
        cbar2.ax.tick_params(labelsize=viz.config.font_size-1)
        
        # 子图3: 自适应权重分布
        if adaptive_weights is not None:
            adaptive_np = adaptive_weights.detach().cpu().numpy()
            
            # 处理adaptive_weights的维度
            if adaptive_np.ndim == 3:  # (batch, seq, scales)
                adaptive_np = adaptive_np.mean(axis=0)  # 平均over batch
            elif adaptive_np.ndim == 2:  # (seq, scales)
                pass  # 直接使用
            
            # 裁剪到有效长度
            adaptive_np = adaptive_np[:effective_len, :]
            
            # 对自适应权重也进行轻微平滑
            adaptive_smooth = viz._smooth_attention_matrix(adaptive_np.T, 
                                                         smooth_method='gaussian',
                                                         smooth_sigma=0.4,
                                                         upsample_factor=1).T
            
            im3 = axes[2].imshow(adaptive_smooth.T, cmap=smooth_cmap_blue_yellow, aspect='auto',
                               interpolation='bilinear', alpha=0.95,
                               extent=[0, effective_len, 0, adaptive_smooth.shape[1]])
            viz._apply_professional_styling(
                axes[2],
                title='Adaptive Scale Weights\n(Original, Branch1, Branch2)',
                xlabel='Position',
                ylabel='Scale Type'
            )
            
            # 设置y轴标签
            axes[2].set_yticks([0.5, 1.5, 2.5])
            axes[2].set_yticklabels(['Original', 'Branch1', 'Branch2'])
            
            cbar3 = plt.colorbar(im3, ax=axes[2], shrink=0.8, aspect=15)
            cbar3.set_label('Weight Value', fontsize=viz.config.font_size)
            cbar3.ax.tick_params(labelsize=viz.config.font_size-1)
        else:
            axes[2].text(0.5, 0.5, 'Adaptive Weights\nNot Available',
                        ha='center', va='center', transform=axes[2].transAxes,
                        fontsize=viz.config.font_size, style='italic')
            viz._apply_professional_styling(
                axes[2],
                title='Adaptive Scale Weights\n(Not Available)'
            )
        
        # 添加整体标题
        domain_name = domain_info.get('display_name', 'Unknown Domain')
        fig.suptitle(f'Fourier Rating Attention Analysis - {domain_name}\nLayer {layer_idx} (Epoch {epoch})',
                    fontsize=viz.config.title_size + 1, fontweight='bold', y=0.95)
        
        # 应用紧凑布局
        if viz.config.tight_layout:
            plt.tight_layout(rect=[0, 0, 1, 0.92])
        
        # 保存图片
        saved_files = []
        if save_plots:
            filename = f"fourier_attention_layer{layer_idx}_epoch{epoch}_{domain_name}"
            saved_files = viz._save_figure(fig, filename)
        
        return fig, saved_files
        
    except Exception as e:
        plt.close(fig)
        raise RuntimeError(f"Fourier注意力可视化失败: {e}")
    finally:
        # 内存清理
        gc.collect()

def plot_expert_routing_journal(routing_weights: torch.Tensor,
                               domain_labels: List[str],
                               expert_labels: List[str], 
                               epoch: int,
                               config: Optional[VisualizationConfig] = None,
                               save_plots: bool = True) -> Tuple[plt.Figure, List[str]]:
    """
    期刊级专家路由热力图
    
    Args:
        routing_weights: 路由权重矩阵 [num_domains, num_experts]
        domain_labels: 领域标签列表
        expert_labels: 专家标签列表
        epoch: 训练轮次
        config: 可视化配置
        save_plots: 是否保存图片
        
    Returns:
        (figure, saved_files) 元组
    """
    viz = EnhancedVisualization(config)
    
    if routing_weights.numel() == 0:
        raise ValueError("路由权重数据为空")
    
    # 转换为numpy
    data = routing_weights.detach().cpu().numpy()
    
    # 创建单个热力图布局
    figsize = viz.config.get_figsize('single')
    # 根据数据尺寸调整图表大小
    figsize = (max(figsize[0], len(expert_labels) * 0.8), 
              max(figsize[1], len(domain_labels) * 0.6))
    
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    try:
        # 使用期刊级配色方案
        colormap = viz.color_schemes.get_heatmap_colormap(viz.config.journal_style)
        
        # 创建热力图
        im = sns.heatmap(data, 
                        annot=True,           # 显示数值
                        fmt='.3f',            # 三位小数
                        cmap=colormap,        # 期刊配色
                        ax=ax,
                        xticklabels=expert_labels,
                        yticklabels=domain_labels,
                        cbar_kws={
                            'label': 'Routing Weight',
                            'shrink': 0.8,
                            'aspect': 20
                        },
                        square=False,         # 不强制正方形
                        linewidths=0.5,       # 网格线宽度
                        linecolor='white',    # 网格线颜色
                        annot_kws={
                            'fontsize': viz.config.font_size - 1,
                            'fontweight': 'normal'
                        })
        
        # 应用专业样式
        viz._apply_professional_styling(
            ax,
            title=f'Domain-Expert Routing Distribution (Epoch {epoch})',
            xlabel='Expert Models',
            ylabel='Domain Categories'
        )
        
        # 优化标签显示
        ax.set_xticklabels(expert_labels, rotation=45, ha='right',
                          fontsize=viz.config.font_size)
        ax.set_yticklabels(domain_labels, rotation=0, ha='right',
                          fontsize=viz.config.font_size)
        
        # 设置颜色条样式
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=viz.config.font_size-1)
        cbar.set_label('Routing Weight', fontsize=viz.config.font_size,
                      fontweight='normal')
        
        # 应用紧凑布局
        if viz.config.tight_layout:
            plt.tight_layout()
        
        # 保存图片
        saved_files = []
        if save_plots:
            filename = f"expert_routing_heatmap_epoch{epoch}"
            saved_files = viz._save_figure(fig, filename)
        
        return fig, saved_files
        
    except Exception as e:
        plt.close(fig)
        raise RuntimeError(f"专家路由热力图生成失败: {e}")
    finally:
        gc.collect()

def plot_tsne_specialization_journal(embeddings: torch.Tensor,
                                    expert_labels: torch.Tensor,
                                    domain_labels: torch.Tensor,
                                    domain_map: Dict[int, str],
                                    epoch: int,
                                    config: Optional[VisualizationConfig] = None,
                                    save_plots: bool = True,
                                    tsne_params: Optional[Dict[str, Any]] = None) -> Tuple[plt.Figure, List[str]]:
    """
    期刊级t-SNE专家专业化分析
    
    Args:
        embeddings: 嵌入向量 [num_samples, embedding_dim]
        expert_labels: 专家标签 [num_samples]
        domain_labels: 领域标签 [num_samples]
        domain_map: 领域ID到名称的映射
        epoch: 训练轮次
        config: 可视化配置
        save_plots: 是否保存图片
        tsne_params: t-SNE参数配置
        
    Returns:
        (figure, saved_files) 元组
    """
    viz = EnhancedVisualization(config)
    
    if embeddings.numel() == 0:
        raise ValueError("嵌入数据为空")
    
    # 默认t-SNE参数
    default_tsne_params = {
        'n_components': 2,
        'perplexity': 30,
        'learning_rate': 200,
        'max_iter': 1000,
        'random_state': 42
    }
    
    if tsne_params:
        default_tsne_params.update(tsne_params)
    
    # 执行t-SNE降维
    tsne = TSNE(**default_tsne_params)
    embeddings_2d = tsne.fit_transform(embeddings.cpu().numpy())
    
    # 创建双子图布局
    fig, axes = viz._create_figure_layout('double', 1, 2)
    
    try:
        # 获取唯一的专家和领域数量
        num_experts = len(torch.unique(expert_labels))
        num_domains = len(domain_map)
        
        # 获取期刊配色
        expert_colors = viz.color_schemes.get_scatter_colors(
            num_experts, viz.config.journal_style)
        domain_colors = viz.color_schemes.get_scatter_colors(
            num_domains, viz.config.journal_style)
        
        # 子图1: 按专家着色
        expert_ids = expert_labels.cpu().numpy()
        for i, expert_id in enumerate(np.unique(expert_ids)):
            mask = expert_ids == expert_id
            axes[0].scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                           c=[expert_colors[i]], label=f'Expert {expert_id}',
                           alpha=viz.config.scatter_alpha, s=20,
                           edgecolors='white', linewidths=0.3)
        
        viz._apply_professional_styling(
            axes[0],
            title=f't-SNE: Expert Specialization\n(Colored by Expert ID)',
            xlabel='t-SNE Dimension 1',
            ylabel='t-SNE Dimension 2'
        )
        
        # 添加专家图例
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                      fontsize=viz.config.legend_size, frameon=True,
                      fancybox=True, shadow=True, framealpha=0.9)
        
        # 子图2: 按领域着色
        domain_ids = domain_labels.cpu().numpy()
        for i, domain_id in enumerate(np.unique(domain_ids)):
            mask = domain_ids == domain_id
            raw_domain_name = domain_map.get(domain_id, f'Domain {domain_id}')
            domain_name = _normalize_domain_name(raw_domain_name) if raw_domain_name.startswith('Domain ') == False else raw_domain_name
            axes[1].scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                           c=[domain_colors[i]], label=domain_name,
                           alpha=viz.config.scatter_alpha, s=20,
                           edgecolors='black', linewidths=0.3)
        
        viz._apply_professional_styling(
            axes[1],
            title=f't-SNE: Domain Distribution\n(Colored by Domain)',
            xlabel='t-SNE Dimension 1',
            ylabel='t-SNE Dimension 2'
        )
        
        # 添加领域图例
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                      fontsize=viz.config.legend_size, frameon=True,
                      fancybox=True, shadow=True, framealpha=0.9)
        
        # 为两个子图添加背景网格
        for ax in axes:
            ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
            ax.set_facecolor('#fafafa')  # 浅灰背景增强对比度
        
        # 添加整体标题
        fig.suptitle(f'Expert Specialization Analysis via t-SNE (Epoch {epoch})',
                    fontsize=viz.config.title_size + 1, fontweight='bold', y=0.95)
        
        # 应用紧凑布局
        if viz.config.tight_layout:
            plt.tight_layout(rect=[0, 0, 0.85, 0.92])
        
        # 保存图片
        saved_files = []
        if save_plots:
            filename = f"tsne_specialization_epoch{epoch}"
            saved_files = viz._save_figure(fig, filename)
        
        return fig, saved_files
        
    except Exception as e:
        plt.close(fig)
        raise RuntimeError(f"t-SNE专业化分析失败: {e}")
    finally:
        gc.collect()

# 便捷函数
def _normalize_domain_name(raw_domain_name: str) -> str:
    """
    规范化领域名称显示
    
    Args:
        raw_domain_name: 原始领域名称 (可能包含_5_5等后缀)
        
    Returns:
        规范化的显示名称
    """
    # 导入领域配置
    try:
        from keys.domain_config import DomainAdaptiveConfig
        display_names = DomainAdaptiveConfig.DOMAIN_DISPLAY_NAMES
        
        # 如果有完全匹配的映射，使用映射值
        if raw_domain_name in display_names:
            return display_names[raw_domain_name]
            
        # 移除常见的后缀并重新尝试匹配
        cleaned_name = raw_domain_name.replace('_5_5', '').replace('_rated', '')
        if cleaned_name in display_names:
            return display_names[cleaned_name]
            
        # 进一步清理和美化
        if 'beauty' in cleaned_name.lower():
            return 'Beauty'
        elif 'games' in cleaned_name.lower():
            return 'Games'
        elif 'ml-1m' in cleaned_name.lower() or 'movielens' in cleaned_name.lower():
            return 'MovieLens'
        elif 'steam' in cleaned_name.lower():
            return 'Steam'
        elif 'video' in cleaned_name.lower():
            return 'Video'
        else:
            # 如果无法识别，返回首字母大写的清理版本
            return cleaned_name.replace('_', ' ').replace('-', ' ').title()
            
    except ImportError:
        # 如果无法导入配置，使用基本的清理逻辑
        cleaned = raw_domain_name.replace('_5_5', '').replace('_rated', '')
        return cleaned.replace('_', ' ').replace('-', ' ').title()


def plot_multi_domain_fourier_comparison_journal(multi_domain_data: Dict[int, Dict],
                                                domain_map: Dict[int, str],
                                                layer_idx: int,
                                                epoch: int,
                                                config: Optional[VisualizationConfig] = None,
                                                save_plots: bool = True) -> Tuple[plt.Figure, List[str]]:
    """
    期刊级多领域Fourier注意力对比可视化
    
    Args:
        multi_domain_data: 多领域数据 {domain_id: attention_dict}
        domain_map: 领域ID到名称的映射
        layer_idx: 层索引
        epoch: 训练轮次
        config: 可视化配置
        save_plots: 是否保存图片
        
    Returns:
        (figure, saved_files) 元组
    """
    viz = EnhancedVisualization(config)
    
    if not multi_domain_data:
        raise ValueError("多领域数据为空")
    
    # 准备数据
    domain_ids = sorted(multi_domain_data.keys())
    num_domains = len(domain_ids)
    
    # 获取领域显示名称 - 使用规范化函数
    domain_names = []
    for domain_id in domain_ids:
        if domain_id in domain_map:
            raw_name = domain_map[domain_id]
            normalized_name = _normalize_domain_name(raw_name)
            domain_names.append(normalized_name)
        else:
            domain_names.append(f"Domain_{domain_id}")
    
    # 创建2行N列的布局 (Branch1 + Branch2 对比)
    figsize = viz.config.get_figsize('double')
    fig_width = max(figsize[0], num_domains * 4.5)  # 根据领域数量调整宽度
    fig_height = 8.0  # 固定高度适合2行布局
    
    fig, axes = plt.subplots(2, num_domains, figsize=(fig_width, fig_height),
                            facecolor='white', edgecolor='none')
    
    # 确保axes是2D数组
    if num_domains == 1:
        axes = axes.reshape(2, 1)
    
    try:
        # 处理每个领域的数据
        for col, domain_id in enumerate(domain_ids):
            attention_dict = multi_domain_data[domain_id]
            domain_name = domain_names[col]
            
            # 提取注意力数据
            branch1 = attention_dict.get('attention_branch_1')
            branch2 = attention_dict.get('attention_branch_2')
            
            if branch1 is None or branch2 is None:
                # 如果数据缺失，显示空白
                for row in range(2):
                    axes[row, col].text(0.5, 0.5, 'Data Not Available',
                                       ha='center', va='center',
                                       transform=axes[row, col].transAxes,
                                       fontsize=viz.config.font_size)
                    viz._apply_professional_styling(axes[row, col])
                continue
            
            # 处理数据维度
            branch1_np = branch1.detach().cpu().numpy()
            branch2_np = branch2.detach().cpu().numpy()
            
            # 维度处理
            if branch1_np.ndim == 4:  # (batch, heads, seq, seq)
                branch1_np = branch1_np.mean(axis=(0, 1))
            elif branch1_np.ndim == 3:  # (batch, seq, seq)
                branch1_np = branch1_np.mean(axis=0)
            
            if branch2_np.ndim == 4:
                branch2_np = branch2_np.mean(axis=(0, 1))
            elif branch2_np.ndim == 3:
                branch2_np = branch2_np.mean(axis=0)
            
            # 获取有效长度
            effective_len = min(50, branch1_np.shape[0])  # 可以根据需要调整
            branch1_np = branch1_np[:effective_len, :effective_len]
            branch2_np = branch2_np[:effective_len, :effective_len]
            
            # 应用平滑处理
            branch1_smooth = viz._smooth_attention_matrix(branch1_np,
                                                         smooth_method='gaussian',
                                                         smooth_sigma=0.6,
                                                         upsample_factor=1)
            branch2_smooth = viz._smooth_attention_matrix(branch2_np,
                                                         smooth_method='gaussian',
                                                         smooth_sigma=0.6,
                                                         upsample_factor=1)
            
            # 创建平滑的色彩映射
            smooth_viridis = viz._create_smooth_colormap('viridis', n_levels=512)
            smooth_plasma = viz._create_smooth_colormap('plasma', n_levels=512)
            
            # 绘制Branch 1 (第一行) - 使用平滑数据
            im1 = axes[0, col].imshow(branch1_smooth, cmap=smooth_viridis, aspect='auto',
                                     interpolation='bilinear', alpha=0.95,
                                     extent=[0, effective_len, effective_len, 0])
            
            # 设置标题和标签
            axes[0, col].set_title(f'{domain_name}\nBranch 1', 
                                  fontsize=viz.config.title_size, 
                                  fontweight='bold', pad=10)
            
            if col == 0:  # 只在第一列显示y轴标签
                axes[0, col].set_ylabel('Position', fontsize=viz.config.label_size)
            
            # 添加颜色条
            cbar1 = plt.colorbar(im1, ax=axes[0, col], shrink=0.8, aspect=15)
            cbar1.set_label('Attention Weight', fontsize=viz.config.font_size-1)
            cbar1.ax.tick_params(labelsize=viz.config.font_size-2)
            
            # 绘制Branch 2 (第二行) - 使用平滑数据
            im2 = axes[1, col].imshow(branch2_smooth, cmap=smooth_plasma, aspect='auto',
                                     interpolation='bilinear', alpha=0.95,
                                     extent=[0, effective_len, effective_len, 0])
            
            axes[1, col].set_title(f'{domain_name}\nBranch 2', 
                                  fontsize=viz.config.title_size, 
                                  fontweight='bold', pad=10)
            
            if col == 0:  # 只在第一列显示y轴标签
                axes[1, col].set_ylabel('Position', fontsize=viz.config.label_size)
            
            # 底部一行显示x轴标签
            axes[1, col].set_xlabel('Position', fontsize=viz.config.label_size)
            
            # 添加颜色条
            cbar2 = plt.colorbar(im2, ax=axes[1, col], shrink=0.8, aspect=15)
            cbar2.set_label('Attention Weight', fontsize=viz.config.font_size-1)
            cbar2.ax.tick_params(labelsize=viz.config.font_size-2)
            
            # 应用专业样式
            for row in range(2):
                viz._apply_professional_styling(axes[row, col])
                
                # 设置刻度
                axes[row, col].tick_params(axis='both', which='major',
                                          labelsize=viz.config.font_size-1)
        
        # 添加整体标题
        fig.suptitle(f'Multi-Domain Fourier Rating Attention Comparison - Layer {layer_idx} (Epoch {epoch})',
                    fontsize=viz.config.title_size + 2, fontweight='bold', y=0.95)
        
        # 调整布局
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.subplots_adjust(hspace=0.35, wspace=0.3)
        
        # 保存图片
        saved_files = []
        if save_plots:
            filename = f"multi_domain_fourier_comparison_layer{layer_idx}_epoch{epoch}"
            saved_files = viz._save_figure(fig, filename)
        
        return fig, saved_files
        
    except Exception as e:
        plt.close(fig)
        raise RuntimeError(f"多领域Fourier对比可视化失败: {e}")
    finally:
        gc.collect()

def create_journal_visualization(journal: str = 'nature') -> EnhancedVisualization:
    """创建期刊特定的可视化实例"""
    from .config import create_journal_config
    config = create_journal_config(journal)
    return EnhancedVisualization(config)