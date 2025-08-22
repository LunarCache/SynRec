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
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
import torch
from typing import Dict, List, Optional, Tuple, Any
import warnings
from pathlib import Path
from scipy import ndimage  # 用于平滑处理
import matplotlib.patches as patches

from .config import VisualizationConfig, get_current_config
from .color_schemes import JournalColorSchemes, create_custom_colormaps
from .journal_styles import apply_journal_style
import gc

class EnhancedVisualization:
    """增强的可视化类，提供期刊级别的图表生成"""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or get_current_config()
        self.color_schemes = JournalColorSchemes()
        self._setup_style()
        
        # 如果使用custom配色方案，创建自定义colormap
        if self.config.journal_style == 'custom':
            self.custom_cmaps = create_custom_colormaps()
        else:
            self.custom_cmaps = None
    
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
    
    def _get_effective_length_from_attention(self, attention_matrix: np.ndarray, 
                                             max_display_len: int = 50) -> Tuple[int, np.ndarray]:
        """
        从注意力矩阵推断有效序列长度，排除padding区域
        
        Args:
            attention_matrix: 注意力矩阵 (seq_len, seq_len)
            max_display_len: 最大显示长度
            
        Returns:
            (effective_length, valid_indices) 元组
        """
        try:
            # 方法1: 基于注意力值的非零模式识别有效长度
            # 计算每行的注意力总和，有效位置通常有更高的注意力值
            row_sums = np.sum(attention_matrix, axis=1)
            col_sums = np.sum(attention_matrix, axis=0)
            
            # 找到最后一个有显著注意力值的位置
            # 使用动态阈值：均值的10%
            threshold = max(np.mean(row_sums) * 0.1, 1e-6)
            
            # 从后往前找最后一个有效位置
            effective_len = attention_matrix.shape[0]
            for i in range(attention_matrix.shape[0] - 1, -1, -1):
                if row_sums[i] > threshold or col_sums[i] > threshold:
                    effective_len = i + 1
                    break
            
            # 限制在合理范围内，避免显示过多无用信息
            effective_len = min(effective_len, max_display_len)
            effective_len = max(effective_len, 10)  # 至少显示10个位置
            
            # 创建有效位置的索引
            valid_indices = np.arange(effective_len)
            
            return effective_len, valid_indices
            
        except Exception as e:
            warnings.warn(f"Failed to compute effective length: {e}")
            # 回退到固定长度
            fallback_len = min(max_display_len, attention_matrix.shape[0])
            return fallback_len, np.arange(fallback_len)
    
    def _filter_padding_attention(self, attention_matrix: np.ndarray, 
                                 valid_indices: np.ndarray) -> np.ndarray:
        """
        过滤注意力矩阵中的padding区域
        
        Args:
            attention_matrix: 原始注意力矩阵
            valid_indices: 有效位置索引
            
        Returns:
            过滤后的注意力矩阵
        """
        try:
            # 只保留有效位置的注意力
            filtered_matrix = attention_matrix[np.ix_(valid_indices, valid_indices)]
            return filtered_matrix
        except Exception as e:
            warnings.warn(f"Failed to filter padding: {e}")
            return attention_matrix[:len(valid_indices), :len(valid_indices)]
    
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


def _epoch_suffix(epoch: Any) -> str:
    """根据 epoch 值生成文件名后缀。
    若 epoch 为数字（int/float/可转为数字的字符串），返回 _epoch{num}；否则返回空串。
    """
    try:
        # 允许形如 'inference' 的字符串跳过 epoch 后缀
        if isinstance(epoch, (int, float)):
            return f"_epoch{int(epoch)}"
        # 字符串且可解析为数值
        if isinstance(epoch, str) and epoch.strip().isdigit():
            return f"_epoch{int(epoch)}"
    except Exception:
        pass
    return ""


def _epoch_title(epoch: Any) -> str:
    """根据 epoch 生成标题中的补充信息。
    数值 epoch 返回 " (Epoch X)"，否则返回空串（推理阶段不显示）。
    """
    try:
        if isinstance(epoch, (int, float)):
            return f" (Epoch {int(epoch)})"
        if isinstance(epoch, str) and epoch.strip().isdigit():
            return f" (Epoch {int(epoch)})"
    except Exception:
        pass
    return ""

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
    
    # 获取有效序列长度并过滤padding区域
    effective_len, valid_indices = viz._get_effective_length_from_attention(branch1_np)
    
    # 过滤padding区域的注意力
    branch1_filtered = viz._filter_padding_attention(branch1_np, valid_indices)
    branch2_filtered = viz._filter_padding_attention(branch2_np, valid_indices)
    
    # 应用平滑处理以获得更好的视觉效果（如果启用）
    if viz.config.enable_smoothing:
        branch1_smooth = viz._smooth_attention_matrix(branch1_filtered, 
                                                     smooth_method=viz.config.smooth_method,
                                                     smooth_sigma=viz.config.smooth_sigma,
                                                     upsample_factor=1)
        branch2_smooth = viz._smooth_attention_matrix(branch2_filtered,
                                                     smooth_method=viz.config.smooth_method, 
                                                     smooth_sigma=viz.config.smooth_sigma,
                                                     upsample_factor=1)
    else:
        branch1_smooth = branch1_filtered
        branch2_smooth = branch2_filtered
    
    # 创建三子图布局
    fig, axes = viz._create_figure_layout('single', 1, 3)
    
    try:
        # 选择配色方案
        if viz.config.journal_style == 'custom' and viz.custom_cmaps:
            # 使用自定义配色
            cmap_branch1 = viz.custom_cmaps['custom_attention']
            cmap_branch2 = viz.custom_cmaps['custom_heatmap'] 
            cmap_weights = viz.custom_cmaps['custom_diverging']
        else:
            # 使用默认配色
            smooth_cmap_viridis = viz._create_smooth_colormap('viridis', n_levels=512)
            smooth_cmap_plasma = viz._create_smooth_colormap('plasma', n_levels=512)
            smooth_cmap_blue_yellow = viz._create_smooth_colormap('RdYlBu_r', n_levels=512)
            cmap_branch1 = smooth_cmap_viridis
            cmap_branch2 = smooth_cmap_plasma
            cmap_weights = smooth_cmap_blue_yellow
        
        # 子图1: 短期注意力模式 (高频)
        # 判断是否为概率权重(0-1)，用于统一色条范围
        is_prob_scale = (np.nanmin(branch1_smooth) >= -1e-6 and np.nanmax(branch1_smooth) <= 1.05)
        im1 = axes[0].imshow(
            branch1_smooth,
            cmap=cmap_branch1,
            aspect='auto',
            interpolation='bilinear',
            alpha=0.95,
            extent=[0, effective_len, effective_len, 0],
            vmin=0.0 if is_prob_scale else None,
            vmax=1.0 if is_prob_scale else None,
        )
        viz._apply_professional_styling(
            axes[0], 
            title='Fourier Branch 1',
            xlabel='Position',
            ylabel='Position'
        )
        
        # 添加颜色条
        cbar1 = plt.colorbar(im1, ax=axes[0], shrink=0.8, aspect=15)
        if is_prob_scale:
            cbar1.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
        cbar1.set_label('Attention Weight', fontsize=viz.config.font_size)
        cbar1.ax.tick_params(labelsize=viz.config.font_size-1)
        
        # 子图2: 长期注意力模式 (低频)
        im2 = axes[1].imshow(
            branch2_smooth,
            cmap=cmap_branch2,
            aspect='auto',
            interpolation='bilinear',
            alpha=0.95,
            extent=[0, effective_len, effective_len, 0],
            vmin=0.0 if is_prob_scale else None,
            vmax=1.0 if is_prob_scale else None,
        )
        viz._apply_professional_styling(
            axes[1],
            title='Fourier Branch 2', 
            xlabel='Position',
            ylabel='Position'
        )
        
        cbar2 = plt.colorbar(im2, ax=axes[1], shrink=0.8, aspect=15)
        if is_prob_scale:
            cbar2.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
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
            
            # 自适应权重也按概率尺度显示
            is_prob_adapt = (np.nanmin(adaptive_smooth) >= -1e-6 and np.nanmax(adaptive_smooth) <= 1.05)
            im3 = axes[2].imshow(
                adaptive_smooth.T,
                cmap=cmap_weights,
                aspect='auto',
                interpolation='bilinear',
                alpha=0.95,
                extent=[0, effective_len, 0, adaptive_smooth.shape[1]],
                vmin=0.0 if is_prob_adapt else None,
                vmax=1.0 if is_prob_adapt else None,
            )
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
            if is_prob_adapt:
                cbar3.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
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
        fig.suptitle(f'Fourier Rating Attention Analysis - {domain_name}\nLayer {layer_idx}{_epoch_title(epoch)}',
                    fontsize=viz.config.title_size + 1, fontweight='bold', y=0.95)
        
        # 应用紧凑布局
        if viz.config.tight_layout:
            plt.tight_layout(rect=[0, 0, 1, 0.92])
        
        # 保存图片
        saved_files = []
        if save_plots:
            epoch_suffix = _epoch_suffix(epoch)
            filename = f"fourier_attention_layer{layer_idx}{epoch_suffix}_{domain_name}"
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
        # 选择配色方案
        if viz.config.journal_style == 'custom' and viz.custom_cmaps:
            colormap = viz.custom_cmaps['custom_heatmap']
        else:
            colormap = viz.color_schemes.get_heatmap_colormap(viz.config.journal_style)
        
        # 统一色条范围：若权重看起来为概率(0-1)，则固定到[0,1]，并设置标准刻度
        data_max = float(np.nanmax(data))
        data_min = float(np.nanmin(data))
        use_prob_scale = (data_max <= 1.05) and (data_min >= -1e-6)
        cbar_kwargs = {
            'label': 'Routing Weight',
            'shrink': 0.8,
            'aspect': 20
        }
        if use_prob_scale:
            cbar_kwargs['ticks'] = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        # 创建热力图
        sns.heatmap(
            data,
            annot=True,                 # 显示数值
            fmt='.2f',                  # 统一为两位小数
            cmap=colormap,              # 自定义或期刊配色
            ax=ax,
            xticklabels=expert_labels,
            yticklabels=domain_labels,
            cbar_kws=cbar_kwargs,
            vmin=0.0 if use_prob_scale else None,
            vmax=1.0 if use_prob_scale else None,
            square=False,               # 不强制正方形
            linewidths=0.5,             # 网格线宽度
            linecolor='white',          # 网格线颜色
            annot_kws={
                'fontsize': viz.config.font_size - 1,
                'fontweight': 'normal',
                'color': '#222222'
            }
        )
        
        # 应用专业样式
        viz._apply_professional_styling(
            ax,
            title=f'Domain-Expert Routing Distribution{_epoch_title(epoch)}',
            xlabel='Expert Models',
            ylabel='Domain Categories'
        )
        
        # 优化标签显示
        ax.set_xticklabels(expert_labels, rotation=45, ha='right',
                          fontsize=viz.config.font_size)
        ax.set_yticklabels(domain_labels, rotation=0, ha='right',
                          fontsize=viz.config.font_size)
        
        # 设置颜色条样式（与全局一致）
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=viz.config.font_size-1)
        cbar.set_label('Routing Weight', fontsize=viz.config.font_size, fontweight='normal')
        
        # 应用紧凑布局
        if viz.config.tight_layout:
            plt.tight_layout()
        
        # 保存图片
        saved_files = []
        if save_plots:
            epoch_suffix = _epoch_suffix(epoch)
            filename = f"expert_routing_heatmap{epoch_suffix}"
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
        
        # 获取配色方案：当选择“小清新”风格时使用自定义配色
        if viz.config.journal_style == 'custom':
            expert_colors = viz.color_schemes.get_scatter_colors(num_experts, 'custom')
            domain_colors = viz.color_schemes.get_scatter_colors(num_domains, 'custom')
        else:
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
                           alpha=viz.config.scatter_alpha, s=22,
                           edgecolors='white', linewidths=0.4)
        
        viz._apply_professional_styling(
            axes[0],
            title=f't-SNE Expert Specialization\n(Colored by Expert Assignment)',
            xlabel='t-SNE Dimension 1',
            ylabel='t-SNE Dimension 2'
        )
        
        # 添加专家图例 - 学术风格（无阴影、略透明边框）
        legend1 = axes[0].legend(loc='upper right', fontsize=viz.config.legend_size-1, 
                      frameon=True, fancybox=False, shadow=False, framealpha=0.85,
                      borderpad=0.5, columnspacing=0.8, handletextpad=0.5,
                      markerscale=1.0)
        
        # 如果专家数量较多，调整图例为多列显示
        if num_experts > 4:
            legend1.set_ncol(2 if num_experts <= 8 else 3)
        
        # 子图2: 按领域着色
        domain_ids = domain_labels.cpu().numpy()
        for i, domain_id in enumerate(np.unique(domain_ids)):
            mask = domain_ids == domain_id
            raw_domain_name = domain_map.get(domain_id, f'Domain {domain_id}')
            domain_name = _normalize_domain_name(raw_domain_name) if raw_domain_name.startswith('Domain ') == False else raw_domain_name
            axes[1].scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                           c=[domain_colors[i]], label=domain_name,
                           alpha=viz.config.scatter_alpha, s=22,
                           edgecolors='white', linewidths=0.4)
        
        viz._apply_professional_styling(
            axes[1],
            title=f't-SNE Domain Distribution\n(Colored by Domain Assignment)',
            xlabel='t-SNE Dimension 1',
            ylabel='t-SNE Dimension 2'
        )
        
        # 添加领域图例 - 学术风格（与左图一致）
        legend2 = axes[1].legend(loc='upper right', fontsize=viz.config.legend_size-1,
                      frameon=True, fancybox=False, shadow=False, framealpha=0.85,
                      borderpad=0.5, columnspacing=0.8, handletextpad=0.5,
                      markerscale=1.0)
        
        # 如果领域数量较多，调整图例为多列显示
        if num_domains > 4:
            legend2.set_ncol(2 if num_domains <= 8 else 3)
        
        # 为两个子图添加背景网格和等比例坐标轴
        for ax in axes:
            ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
            ax.set_facecolor('#fafafa')  # 浅灰背景增强对比度
            # 保持等比例坐标系
            ax.set_aspect('equal', adjustable='box')

        # 将左右子图锚定到各自单元格内侧边，减少中间空白
        if len(axes) >= 2:
            try:
                axes[0].set_anchor('E')  # 左图靠右
                axes[1].set_anchor('W')  # 右图靠左
            except Exception:
                pass
            
            # 统一两个子图的坐标范围，确保在等比例下大小一致
            global_xmin, global_xmax = embeddings_2d[:, 0].min(), embeddings_2d[:, 0].max()
            global_ymin, global_ymax = embeddings_2d[:, 1].min(), embeddings_2d[:, 1].max()
            x_center = (global_xmin + global_xmax) / 2.0
            y_center = (global_ymin + global_ymax) / 2.0
            max_range = max(global_xmax - global_xmin, global_ymax - global_ymin) * 1.05
            half = max_range / 2.0
            for ax_sync in axes:
                ax_sync.set_xlim(x_center - half, x_center + half)
                ax_sync.set_ylim(y_center - half, y_center + half)
        
        # 添加整体标题 - 调整到两个子图上方
        fig.suptitle(f'Expert Specialization Analysis via t-SNE{_epoch_title(epoch)}',
                    fontsize=viz.config.title_size + 1, fontweight='bold', y=1)
        
        # 应用紧凑布局 - 优化子图间距和边距
        if viz.config.tight_layout:
            plt.tight_layout()
            # 调整布局：标题在子图上方，减少子图间距
            plt.subplots_adjust(wspace=0.15, hspace=0.1, left=0.05, right=0.95, top=0.85, bottom=0.1)
        
        # 保存图片
        saved_files = []
        if save_plots:
            epoch_suffix = _epoch_suffix(epoch)
            filename = f"tsne_specialization{epoch_suffix}"
            saved_files = viz._save_figure(fig, filename)
        
        return fig, saved_files
        
    except Exception as e:
        plt.close(fig)
        raise RuntimeError(f"t-SNE专业化分析失败: {e}")
    finally:
        gc.collect()


def plot_inference_combined_overview(
    routing_weights: torch.Tensor,
    domain_labels_list: List[str],
    expert_labels_list: List[str],
    embeddings: torch.Tensor,
    expert_assignments: torch.Tensor,
    domain_assignments: torch.Tensor,
    domain_map: Dict[int, str],
    config: Optional[VisualizationConfig] = None,
    save_plots: bool = True,
    max_tsne_samples: int = 1000
) -> Tuple[plt.Figure, List[str]]:
    """
    推理可视化综合图（1x3 布局）：
    左：专家路由热力图；中：t-SNE（按专家着色）；右：t-SNE（按领域着色）。
    不显示 Epoch 信息，统一学术风格。
    """
    viz = EnhancedVisualization(config)
    saved_files: List[str] = []

    # 1x3 布局
    fig, axes = viz._create_figure_layout('single', 1, 3)

    # --- 左：专家路由热力图 ---
    try:
        data = routing_weights.detach().cpu().numpy()
        if viz.config.journal_style == 'custom' and viz.custom_cmaps:
            colormap = viz.custom_cmaps['custom_heatmap']
        else:
            colormap = viz.color_schemes.get_heatmap_colormap(viz.config.journal_style)

        data_max = float(np.nanmax(data))
        data_min = float(np.nanmin(data))
        use_prob_scale = (data_max <= 1.05) and (data_min >= -1e-6)
        cbar_kwargs = {
            'label': 'Routing Weight',
            'shrink': 0.8,
            'aspect': 20
        }
        if use_prob_scale:
            cbar_kwargs['ticks'] = [0.0, 0.25, 0.5, 0.75, 1.0]

        sns.heatmap(
            data,
            annot=True,
            fmt='.2f',
            cmap=colormap,
            ax=axes[0],
            xticklabels=expert_labels_list,
            yticklabels=domain_labels_list,
            cbar_kws=cbar_kwargs,
            vmin=0.0 if use_prob_scale else None,
            vmax=1.0 if use_prob_scale else None,
            square=False,
            linewidths=0.5,
            linecolor='white',
            annot_kws={
                'fontsize': viz.config.font_size - 1,
                'fontweight': 'normal',
                'color': '#222222'
            }
        )

        viz._apply_professional_styling(
            axes[0],
            title='(a) Domain-Expert Routing Distribution',
            xlabel='Expert Models',
            ylabel='Domain Categories'
        )

        cbar = axes[0].collections[0].colorbar
        cbar.ax.tick_params(labelsize=viz.config.font_size-1)
        cbar.set_label('Routing Weight', fontsize=viz.config.font_size, fontweight='normal')
    except Exception as e:
        axes[0].text(0.5, 0.5, f'Routing heatmap error:\n{e}', ha='center', va='center', transform=axes[0].transAxes)

    # --- 中/右：t-SNE 两视图 ---
    try:
        # 转为 numpy 并进行采样，避免 t-SNE 过慢
        emb_np = embeddings.detach().cpu().numpy()
        exp_np = expert_assignments.detach().cpu().numpy()
        dom_np = domain_assignments.detach().cpu().numpy()

        n = len(emb_np)
        if n > max_tsne_samples:
            idx = np.random.choice(n, max_tsne_samples, replace=False)
            emb_np = emb_np[idx]
            exp_np = exp_np[idx]
            dom_np = dom_np[idx]

        # t-SNE 嵌入（若已是 2D 则直接用）
        if emb_np.ndim == 2 and emb_np.shape[1] == 2:
            embeddings_2d = emb_np
        else:
            perplexity = int(min(30, max(5, len(emb_np) // 4)))
            tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=200, max_iter=1000, random_state=42)
            embeddings_2d = tsne.fit_transform(emb_np)

        # 颜色
        expert_ids = exp_np
        domain_ids = dom_np

        expert_colors = viz.color_schemes.get_scatter_colors(len(np.unique(expert_ids)), viz.config.journal_style)
        domain_colors = viz.color_schemes.get_scatter_colors(len(domain_map), viz.config.journal_style)

        # 中：按专家
        for i, expert_id in enumerate(np.unique(expert_ids)):
            mask = expert_ids == expert_id
            axes[1].scatter(
                embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                c=[expert_colors[i]], label=f'Expert {expert_id}',
                alpha=viz.config.scatter_alpha, s=22,
                edgecolors='white', linewidths=0.4
            )
        viz._apply_professional_styling(
            axes[1],
            title='(b) t-SNE Expert Specialization',
            xlabel='t-SNE Dimension 1',
            ylabel='t-SNE Dimension 2'
        )
        legend1 = axes[1].legend(loc='upper right', fontsize=viz.config.legend_size-1,
                      frameon=True, fancybox=False, shadow=False, framealpha=0.85,
                      borderpad=0.5, columnspacing=0.8, handletextpad=0.5, markerscale=1.0)

        # 右：按领域
        for i, domain_id in enumerate(np.unique(domain_ids)):
            mask = domain_ids == domain_id
            raw_domain_name = domain_map.get(int(domain_id), f'Domain {domain_id}')
            domain_name = _normalize_domain_name(raw_domain_name) if not raw_domain_name.startswith('Domain ') else raw_domain_name
            axes[2].scatter(
                embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                c=[domain_colors[i]], label=domain_name,
                alpha=viz.config.scatter_alpha, s=22,
                edgecolors='white', linewidths=0.4
            )
        viz._apply_professional_styling(
            axes[2],
            title='(c) t-SNE Domain Distribution',
            xlabel='t-SNE Dimension 1',
            ylabel='t-SNE Dimension 2'
        )
        legend2 = axes[2].legend(loc='upper right', fontsize=viz.config.legend_size-1,
                      frameon=True, fancybox=False, shadow=False, framealpha=0.85,
                      borderpad=0.5, columnspacing=0.8, handletextpad=0.5, markerscale=1.0)

        # 统一两图纵横比与范围
        for ax in [axes[1], axes[2]]:
            ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
            ax.set_facecolor('#fafafa')
            ax.set_aspect('equal', adjustable='box')

        global_xmin, global_xmax = embeddings_2d[:, 0].min(), embeddings_2d[:, 0].max()
        global_ymin, global_ymax = embeddings_2d[:, 1].min(), embeddings_2d[:, 1].max()
        x_center = (global_xmin + global_xmax) / 2.0
        y_center = (global_ymin + global_ymax) / 2.0
        max_range = max(global_xmax - global_xmin, global_ymax - global_ymin) * 1.05
        half = max_range / 2.0
        for ax_sync in [axes[1], axes[2]]:
            ax_sync.set_xlim(x_center - half, x_center + half)
            ax_sync.set_ylim(y_center - half, y_center + half)
    except Exception as e:
        axes[1].text(0.5, 0.5, f't-SNE error:\n{e}', ha='center', va='center', transform=axes[1].transAxes)
        axes[2].set_visible(False)

    # 总标题（推理，不含 Epoch）
    fig.suptitle('Inference Overview: Routing and t-SNE', fontsize=viz.config.title_size + 1, fontweight='bold', y=1.0)

    if viz.config.tight_layout:
        plt.tight_layout()
        # 恢复常规边距
        plt.subplots_adjust(wspace=0.12, left=0.06, right=0.97, top=0.88, bottom=0.12)

    if save_plots:
        saved_files = viz._save_figure(fig, 'inference_overview_routing_tsne')

    return fig, saved_files

def _intelligent_sampling(embeddings: np.ndarray, labels: np.ndarray, 
                         target_size: int = 500, strategy: str = 'auto') -> np.ndarray:
    """
    智能采样策略，减少数据点同时保持代表性
    
    Args:
        embeddings: 原始嵌入向量 [N, dim]
        labels: 对应的标签 [N]
        target_size: 目标采样数量
        strategy: 采样策略 ('auto', 'random', 'density', 'hybrid')
    
    Returns:
        采样后的索引数组
    """
    n_samples = len(embeddings)
    
    if n_samples <= target_size:
        return np.arange(n_samples)
    
    # 自动选择策略
    if strategy == 'auto':
        if n_samples < 800:
            strategy = 'random'
        elif n_samples < 2000:
            strategy = 'density'
        else:
            strategy = 'hybrid'
    
    if strategy == 'random':
        return np.random.choice(n_samples, target_size, replace=False)
    
    elif strategy == 'density':
        # 基于密度的采样：在高密度区域采样更多点
        kde = KernelDensity(bandwidth=0.5)
        kde.fit(embeddings)
        densities = np.exp(kde.score_samples(embeddings))
        
        # 将密度转换为采样权重
        weights = densities / np.sum(densities)
        indices = np.random.choice(n_samples, target_size, replace=False, p=weights)
        return indices
    
    elif strategy == 'hybrid':
        # 混合策略：70%密度采样 + 30%随机采样
        density_count = int(target_size * 0.7)
        random_count = target_size - density_count
        
        # 密度采样部分
        kde = KernelDensity(bandwidth=0.5)
        kde.fit(embeddings)
        densities = np.exp(kde.score_samples(embeddings))
        weights = densities / np.sum(densities)
        density_indices = np.random.choice(n_samples, density_count, replace=False, p=weights)
        
        # 随机采样部分（从剩余点中选择）
        remaining_indices = np.setdiff1d(np.arange(n_samples), density_indices)
        if len(remaining_indices) >= random_count:
            random_indices = np.random.choice(remaining_indices, random_count, replace=False)
        else:
            random_indices = remaining_indices
        
        return np.concatenate([density_indices, random_indices])
    
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")


def _draw_spherical_points(ax: plt.Axes, x: np.ndarray, y: np.ndarray, 
                          colors: List[str], labels: List[str], 
                          base_size: int = 20, size_variation: float = 0.6,
                          alpha: float = 0.8, edge_alpha: float = 0.9) -> None:
    """
    绘制拟球形效果的散点图
    
    Args:
        ax: matplotlib轴对象
        x, y: 散点坐标
        colors: 颜色列表
        labels: 标签列表  
        base_size: 基础点大小
        size_variation: 大小变化系数
        alpha: 主体透明度
        edge_alpha: 边框透明度
    """
    # 计算局部密度用于调整点大小
    points = np.column_stack([x, y])
    
    if len(points) > 50:  # 只有足够的点才计算密度
        try:
            kde = KernelDensity(bandwidth=0.3)
            kde.fit(points)
            densities = np.exp(kde.score_samples(points))
            # 归一化密度到 [0, 1]
            densities = (densities - densities.min()) / (densities.max() - densities.min())
        except:
            densities = np.ones(len(points)) * 0.5
    else:
        densities = np.ones(len(points)) * 0.5
    
    unique_colors = list(set(colors))
    color_to_label = dict(zip(unique_colors, labels[:len(unique_colors)]))
    
    for color in unique_colors:
        mask = np.array(colors) == color
        if not np.any(mask):
            continue
            
        x_subset = x[mask]
        y_subset = y[mask]
        densities_subset = densities[mask]
        
        # 计算动态大小
        sizes = base_size + (base_size * size_variation * densities_subset)
        
        # 1. 绘制阴影层（投影效果）
        shadow_offset = 2
        ax.scatter(x_subset + shadow_offset, y_subset - shadow_offset, 
                  s=sizes * 0.8, c='black', alpha=0.15, 
                  marker='o', linewidths=0)
        
        # 2. 绘制主体层（径向渐变效果模拟）
        # 主体颜色（稍深）
        main_color = mcolors.to_rgba(color, alpha=alpha)
        ax.scatter(x_subset, y_subset, s=sizes, c=[main_color], 
                  marker='o', linewidths=2, 
                  edgecolors='white', alpha=1.0,
                  label=color_to_label.get(color, ''))
        
        # 3. 绘制内部渐变效果（模拟球面曲率）
        inner_color = mcolors.to_rgba(color, alpha=alpha * 0.6)
        ax.scatter(x_subset, y_subset, s=sizes * 0.6, c=[inner_color], 
                  marker='o', linewidths=0)
        
        # 4. 绘制高光效果
        highlight_offset_x = sizes * 0.15
        highlight_offset_y = sizes * 0.15
        ax.scatter(x_subset - highlight_offset_x/100, y_subset + highlight_offset_y/100, 
                  s=sizes * 0.2, c='white', alpha=0.8, 
                  marker='o', linewidths=0)


# 便捷函数
def _normalize_domain_name(raw_domain_name: str) -> str:
    """
    规范化领域名称显示
    
    Args:
        raw_domain_name: 原始领域名称 (可能包含_5_5等后缀)
        
    Returns:
        规范化的显示名称
    """
    # 简化的领域名称规范化，不再依赖复杂的配置
    cleaned_name = raw_domain_name.replace('_5_5', '').replace('_rated', '')
    
    # 基本的名称映射
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


# Multi-domain Fourier comparison visualization - Detailed branch view
def plot_multi_domain_fourier_comparison_journal(fourier_attn_data: Dict[int, Dict[str, torch.Tensor]],
                                                domain_map: Dict[int, str],
                                                layer_idx: int,
                                                epoch: int,
                                                config: Optional[VisualizationConfig] = None,
                                                save_plots: bool = True,
                                                adaptive_style: str = 'heatmap') -> Tuple[plt.Figure, List[str]]:
    """
    期刊级多领域自适应权重对比可视化
    专注展示各领域的自适应权重分布对比
    
    Args:
        fourier_attn_data: 包含多个领域的注意力数据 {domain_id: attention_data}
        domain_map: 领域ID到名称的映射
        layer_idx: 层索引
        epoch: 训练轮次
        config: 可视化配置
        save_plots: 是否保存图片
        
    Returns:
        (figure, saved_files) 元组
    """
    viz = EnhancedVisualization(config)
    
    if not fourier_attn_data or len(fourier_attn_data) < 1:
        # 如果数据不足，创建空图
        fig, ax = plt.subplots(1, 1, figsize=viz.config.get_figsize('single'))
        ax.text(0.5, 0.5, 'Insufficient data for multi-domain comparison', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'Multi-Domain Adaptive Weights Comparison (Layer {layer_idx})')
        return fig, []
    
    # 布局设计：1行 x N列（N为领域数）- 只显示自适应权重
    num_domains = len(fourier_attn_data)
    nrows = 1  # 固定1行：只显示Adaptive weights
    ncols = num_domains
    
    # 调整图片尺寸
    base_figsize = viz.config.get_figsize('multi')
    figsize = (base_figsize[0] * ncols * 0.8, base_figsize[1] * 0.8)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, 
                            facecolor='white', edgecolor='none')
    
    # 标准化axes为1D数组（单行）
    if ncols == 1:
        axes = np.array([axes])
    elif not isinstance(axes, np.ndarray):
        axes = np.array(axes)
    
    try:
        # 定义自适应权重的配色
        weights_cmap = viz._create_smooth_colormap('RdYlBu_r', n_levels=256) # 权重用发散色
        
        # 先将所有tensor移到CPU并脱离计算图，避免GPU内存问题
        domain_ids = sorted(fourier_attn_data.keys())
        
        # 安全的数据处理
        processed_data = {}
        for domain_id in domain_ids:
            try:
                attention_data = fourier_attn_data[domain_id]
                processed_data[domain_id] = {}
                
                for key, tensor in attention_data.items():
                    if torch.is_tensor(tensor):
                        # 安全地移动到CPU并脱离计算图
                        with torch.no_grad():
                            processed_data[domain_id][key] = tensor.detach().cpu()
                    else:
                        processed_data[domain_id][key] = tensor
            except Exception as data_error:
                warnings.warn(f"Data processing error for domain {domain_id}: {data_error}")
                continue
        
        if not processed_data:
            raise ValueError("No valid data could be processed")
        
        # 用于折线样式的全局图例句柄来源轴
        line_legend_axes = None

        for col, domain_id in enumerate(processed_data.keys()):
            attention_data = processed_data[domain_id]
            domain_name = _normalize_domain_name(domain_map.get(domain_id, f'Domain {domain_id}'))
            
            # 直接处理自适应权重 (Adaptive Weights)
            ax_weights = axes[col]
            effective_len = 50  # 默认有效长度，如果有adaptive_weights数据会更新
                
            if 'adaptive_weights' in attention_data:
                adaptive_weights = attention_data['adaptive_weights'].numpy()
                
                # 处理adaptive weights的维度
                if adaptive_weights.ndim >= 3:
                    adaptive_weights = adaptive_weights.mean(axis=0) if adaptive_weights.ndim == 3 else adaptive_weights.mean(axis=(0,1))
                
                # 转置以便更好地显示或处理 (3, seq_len) -> (seq_len, 3)
                if adaptive_weights.shape[0] == 3:  # (3, seq_len)
                    adaptive_weights = adaptive_weights.T
                
                # 更新有效长度
                effective_len = min(adaptive_weights.shape[0], 100)  # 限制最大显示长度
                adaptive_weights = adaptive_weights[:effective_len, :]

                if adaptive_style == 'lines':
                    # 折线样式：三条曲线更直观地反映关系
                    x = np.arange(effective_len)
                    labels = ['Original', 'Long-term', 'Short-term']
                    colors = viz.color_schemes.get_journal_palette(viz.config.journal_style, 3)
                    # 平滑曲线（轻微）
                    try:
                        y0 = ndimage.gaussian_filter1d(adaptive_weights[:, 0], sigma=0.8)
                        y1 = ndimage.gaussian_filter1d(adaptive_weights[:, 1], sigma=0.8)
                        y2 = ndimage.gaussian_filter1d(adaptive_weights[:, 2], sigma=0.8)
                    except Exception:
                        y0, y1, y2 = adaptive_weights[:, 0], adaptive_weights[:, 1], adaptive_weights[:, 2]

                    ax_weights.plot(x, y0, color=colors[0], linewidth=2.0, label=labels[0])
                    ax_weights.plot(x, y1, color=colors[1], linewidth=2.0, label=labels[1])
                    ax_weights.plot(x, y2, color=colors[2], linewidth=2.0, label=labels[2])

                    # 若数据像概率，固定0-1范围并设置刻度
                    is_prob = (np.nanmin(adaptive_weights) >= -1e-6 and np.nanmax(adaptive_weights) <= 1.05)
                    if is_prob:
                        ax_weights.set_ylim(0.0, 1.0)
                        ax_weights.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

                    # 记录一个含有线条标签的轴以生成全局图例（底部）
                    if line_legend_axes is None:
                        line_legend_axes = ax_weights
                else:
                    # 渐变热力图（原样式）
                    im3 = ax_weights.imshow(adaptive_weights.T, cmap=weights_cmap, aspect='auto',
                                          interpolation='bilinear', alpha=0.9,
                                          extent=[0, effective_len, 0, 3])
                    # 设置y轴标签
                    ax_weights.set_yticks([0.5, 1.5, 2.5])
                    ax_weights.set_yticklabels(['Original', 'Long-term', 'Short-term'])
                
            else:
                # 如果没有adaptive weights数据
                ax_weights.text(0.5, 0.5, 'No Adaptive\nWeights Data', 
                               ha='center', va='center', transform=ax_weights.transAxes,
                               fontsize=viz.config.font_size, style='italic')
            
            viz._apply_professional_styling(
                ax_weights,
                title=f'{domain_name}\nAdaptive Weights',
                xlabel='Position',
                ylabel=('Weight' if adaptive_style == 'lines' else ('Weight Type' if col == 0 else ''))
            )

            # 不再显示 L 注记，长度可由刻度读取
                
            # 添加颜色条（仅在最后一列）
            if adaptive_style != 'lines' and col == len(domain_ids) - 1 and 'adaptive_weights' in attention_data:
                cbar3 = plt.colorbar(im3, ax=ax_weights, shrink=0.6, aspect=15)
                cbar3.set_label('Weight Value', fontsize=viz.config.font_size-1)
                cbar3.ax.tick_params(labelsize=viz.config.font_size-2)
        
        # 设置总标题 - 专注于自适应权重对比
        fig.suptitle(f'Multi-Domain Adaptive Weights Comparison (Layer {layer_idx}){_epoch_title(epoch)}', 
                    fontsize=viz.config.title_size + 1, fontweight='bold', y=0.95)
        
        # 折线样式：在底部添加全局图例并预留空间
        if adaptive_style == 'lines' and line_legend_axes is not None:
            try:
                handles, labels = line_legend_axes.get_legend_handles_labels()
                if len(handles) > 0:
                    fig.legend(
                        handles, labels,
                        loc='lower center', ncol=len(labels),
                        frameon=False, fancybox=False, shadow=False,
                        fontsize=viz.config.legend_size+2,
                        bbox_to_anchor=(0.5, 0.01),
                        borderpad=0.8, columnspacing=1.0, handletextpad=0.8,
                        handlelength=2.4, markerscale=1.2
                    )
            except Exception:
                pass

        # 调整布局 - 为底部图例留空间
        if adaptive_style == 'lines':
            # 增加底部留白，确保底部图例不与图像重叠
            plt.tight_layout(rect=[0.02, 0.2, 1, 0.92])
            plt.subplots_adjust(bottom=0.32)
        else:
            # 单行布局无需行标签空间
            plt.tight_layout(rect=[0.02, 0.02, 1, 0.92])  # 为标题留空间
        
        # 保存图片
        saved_files = []
        if save_plots:
            epoch_suffix = _epoch_suffix(epoch)
            filename = f'multi_domain_adaptive_weights_layer_{layer_idx}{epoch_suffix}'
            saved_files = viz._save_figure(fig, filename)
        
        return fig, saved_files
        
    except Exception as e:
        # 完全安全地处理错误，避免影响训练
        try:
            warnings.warn(f"多领域自适应权重可视化失败: {e}", RuntimeWarning)
            # 关闭所有可能的matplotlib图形
            plt.close('all')
            
            # 创建简单错误图作为回退
            fig, ax = plt.subplots(1, 1, figsize=(6, 4), facecolor='white')
            ax.text(0.5, 0.5, f'Visualization temporarily unavailable\nTraining continues normally', 
                    ha='center', va='center', transform=ax.transAxes, fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
            ax.set_title(f'Adaptive Weights Analysis (Layer {layer_idx}){_epoch_title(epoch)}', fontsize=14)
            ax.axis('off')  # 隐藏轴以简化
            
            return fig, []
            
        except Exception as fallback_error:
            # 最终安全网：即使创建错误图失败，也要返回某个图形
            warnings.warn(f"Fallback visualization also failed: {fallback_error}", RuntimeWarning)
            # 返回空的Figure对象
            return plt.figure(figsize=(1, 1)), []
    
    finally:
        # 确保内存清理和资源释放
        try:
            # 强制垃圾收集
            import gc
            gc.collect()
            
            # 清理matplotlib缓存
            if plt.get_fignums():  # 如果有未关闭的图形
                for fignum in plt.get_fignums():
                    try:
                        plt.figure(fignum)
                        plt.close()
                    except:
                        pass
            
            # 清理GPU内存（如果使用CUDA）
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception:
            # 即使清理失败也不能影响训练
            pass


def plot_multi_domain_fourier_comparison_enhanced_journal(
    fourier_attn_data: Dict[int, Dict[str, torch.Tensor]],
    domain_map: Dict[int, str],
    layer_idx: int,
    epoch: int,
    config: Optional[VisualizationConfig] = None,
    save_plots: bool = True,
    show_frequency_analysis: bool = True,
    show_branch_comparison: bool = True
) -> Tuple[plt.Figure, List[str]]:
    """
    增强版多领域Fourier注意力对比可视化
    
    Args:
        fourier_attn_data: 包含多个领域的注意力数据
        domain_map: 领域映射
        layer_idx: 层索引
        epoch: 训练轮次
        config: 可视化配置
        save_plots: 是否保存图片
        show_frequency_analysis: 是否显示频率能量分析
        show_branch_comparison: 是否显示分支对比
        
    Returns:
        (figure, saved_files) 元组
    """
    viz = EnhancedVisualization(config)
    
    if not fourier_attn_data or len(fourier_attn_data) < 1:
        fig, ax = plt.subplots(1, 1, figsize=viz.config.get_figsize('single'))
        ax.text(0.5, 0.5, 'Insufficient data for multi-domain comparison', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'Enhanced Multi-Domain Fourier Analysis (Layer {layer_idx})')
        return fig, []
    
    num_domains = len(fourier_attn_data)
    
    # 决定子图布局
    if show_frequency_analysis and show_branch_comparison:
        # 3行布局：每个领域的整体视图 + 分支对比 + 频率分析
        nrows = 3
        ncols = num_domains
    elif show_frequency_analysis or show_branch_comparison:
        # 2行布局
        nrows = 2
        ncols = num_domains
    else:
        # 基本布局
        nrows = 1
        ncols = num_domains
    
    fig, axes = viz._create_figure_layout('multi', nrows, ncols)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)
    
    try:
        # 获取配色方案
        if viz.config.journal_style == 'custom':
            colors_list = list(viz.color_schemes.custom_colors.values())
        else:
            colors_list = viz.color_schemes.get_journal_palette(viz.config.journal_style, 9)
        
        domain_ids = sorted(fourier_attn_data.keys())
        
        for col, domain_id in enumerate(domain_ids):
            if col >= ncols:
                break
                
            attention_data = fourier_attn_data[domain_id]
            domain_name = _normalize_domain_name(domain_map.get(domain_id, f'Domain {domain_id}'))
            
            if 'branch1' in attention_data and 'branch2' in attention_data:
                branch1 = attention_data['branch1'].detach().cpu().numpy()
                branch2 = attention_data['branch2'].detach().cpu().numpy()
                
                # 处理维度
                if branch1.ndim >= 3:
                    branch1 = branch1.mean(axis=(0, 1)) if branch1.ndim == 4 else branch1.mean(axis=0)
                if branch2.ndim >= 3:
                    branch2 = branch2.mean(axis=(0, 1)) if branch2.ndim == 4 else branch2.mean(axis=0)
                
                # 获取有效序列长度并过滤padding区域 - 使用动态检测
                effective_len, valid_indices = viz._get_effective_length_from_attention(branch1)
                
                # 过滤padding区域的注意力
                branch1_filtered = viz._filter_padding_attention(branch1, valid_indices)
                branch2_filtered = viz._filter_padding_attention(branch2, valid_indices)
                
                # 第1行：综合注意力视图
                ax_main = axes[0, col]
                combined_attn = (branch1_filtered + branch2_filtered) / 2.0
                
                if viz.config.enable_smoothing:
                    combined_attn = viz._smooth_attention_matrix(
                        combined_attn, 
                        smooth_method=viz.config.smooth_method,
                        smooth_sigma=viz.config.smooth_sigma
                    )
                
                # 选择配色
                if viz.config.journal_style == 'custom' and viz.custom_cmaps:
                    cmap = viz.custom_cmaps['custom_attention']
                else:
                    cmap = viz._create_smooth_colormap('viridis', n_levels=256)
                
                im_main = ax_main.imshow(combined_attn, cmap=cmap, aspect='auto', 
                                       interpolation='bilinear', alpha=0.9,
                                       extent=[0, effective_len, effective_len, 0])
                
                viz._apply_professional_styling(
                    ax_main,
                    title=f'{domain_name}\nCombined Attention',
                    xlabel='Position' if nrows == 1 else '',
                    ylabel='Position' if col == 0 else ''
                )
                
                # 第2行：分支对比（如果启用）
                if show_branch_comparison and nrows >= 2:
                    ax_branches = axes[1, col]
                    
                    # 将两个分支并排显示
                    combined_branches = np.concatenate([branch1_filtered, branch2_filtered], axis=1)
                    
                    # 为两个分支使用不同的配色
                    cmap_branch = viz._create_smooth_colormap('RdYlBu_r', n_levels=256)
                    
                    im_branches = ax_branches.imshow(combined_branches, cmap=cmap_branch, 
                                                   aspect='auto', interpolation='bilinear',
                                                   extent=[0, effective_len*2, effective_len, 0])
                    
                    # 添加分割线
                    mid_point = effective_len
                    ax_branches.axvline(x=mid_point - 0.5, color='white', linewidth=2, linestyle='--')
                    
                    viz._apply_professional_styling(
                        ax_branches,
                        title='High-Freq | Low-Freq',
                        xlabel='Position' if nrows == 2 else '',
                        ylabel='Position' if col == 0 else ''
                    )
                
                # 第3行：频率能量分析（如果启用）
                if show_frequency_analysis and nrows >= 3:
                    ax_freq = axes[2, col]
                    
                    # 计算频率能量分布 - 使用过滤后的数据
                    high_freq_energy = np.mean(branch1_filtered ** 2)
                    low_freq_energy = np.mean(branch2_filtered ** 2)
                    total_energy = high_freq_energy + low_freq_energy
                    
                    if total_energy > 0:
                        high_ratio = high_freq_energy / total_energy
                        low_ratio = low_freq_energy / total_energy
                    else:
                        high_ratio = low_ratio = 0.5
                    
                    # 绘制能量分布饼图
                    sizes = [high_ratio, low_ratio]
                    labels = ['High-Freq\n(Short-term)', 'Low-Freq\n(Long-term)']
                    colors = [colors_list[2], colors_list[1]]  # 高频用红色，低频用蓝色
                    
                    ax_freq.pie(
                        sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                        startangle=90, textprops={'fontsize': viz.config.font_size-2}
                    )
                    
                    ax_freq.set_title(f'Energy Distribution\nH:{high_freq_energy:.3f} L:{low_freq_energy:.3f}',
                                     fontsize=viz.config.font_size)
        
        # 隐藏多余的子图
        for row in range(nrows):
            for col in range(len(domain_ids), ncols):
                if col < axes.shape[1]:
                    axes[row, col].set_visible(False)
        
        # 设置总标题
        title_parts = [f'Enhanced Multi-Domain Fourier Analysis (Layer {layer_idx}){_epoch_title(epoch)}']
        if show_branch_comparison:
            title_parts.append('with Branch Comparison')
        if show_frequency_analysis:
            title_parts.append('and Energy Analysis')
        
        fig.suptitle(' '.join(title_parts), 
                    fontsize=viz.config.title_size, fontweight='bold', y=0.95)
        
        # 调整布局
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        
        # 保存图片
        saved_files = []
        if save_plots:
            suffix = '_enhanced'
            if show_branch_comparison:
                suffix += '_branches'
            if show_frequency_analysis:
                suffix += '_energy'
            
            filename = f'multi_domain_fourier_attention{suffix}_layer_{layer_idx}_epoch_{epoch}'
            saved_files = viz._save_figure(fig, filename)
        
        return fig, saved_files
        
    except Exception as e:
        warnings.warn(f"增强多领域Fourier可视化失败: {e}")
        fig, ax = plt.subplots(1, 1, figsize=viz.config.get_figsize('single'))
        ax.text(0.5, 0.5, f'Enhanced visualization error: {str(e)}', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Enhanced Multi-Domain Fourier Analysis - Error')
        return fig, []
    finally:
        gc.collect()


def plot_frequency_ablation_comparison_journal(
    ablation_results: Dict[str, Dict[str, float]],
    domain_map: Dict[int, str],
    epoch: int,
    config: Optional[VisualizationConfig] = None,
    save_plots: bool = True
) -> Tuple[plt.Figure, List[str]]:
    """
    消融实验结果对比可视化
    
    Args:
        ablation_results: 消融实验结果 {mode: {metric: value}}
        domain_map: 领域映射
        epoch: 训练轮次
        config: 可视化配置
        save_plots: 是否保存图片
        
    Returns:
        (figure, saved_files) 元组
    """
    viz = EnhancedVisualization(config)
    
    if not ablation_results:
        fig, ax = plt.subplots(1, 1, figsize=viz.config.get_figsize('single'))
        ax.text(0.5, 0.5, 'No ablation results available', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Frequency Ablation Comparison - No Data')
        return fig, []
    
    # 提取所有模式和指标
    modes = list(ablation_results.keys())
    if not modes:
        return plt.figure(), []
    
    # 提取可用指标
    all_metrics = set()
    for mode_results in ablation_results.values():
        all_metrics.update(mode_results.keys())
    
    metrics = sorted(list(all_metrics))
    
    if not metrics:
        fig, ax = plt.subplots(1, 1, figsize=viz.config.get_figsize('single'))
        ax.text(0.5, 0.5, 'No metrics available in ablation results', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Frequency Ablation Comparison - No Metrics')
        return fig, []
    
    # 创建子图布局：按指标数量决定布局
    num_metrics = len(metrics)
    if num_metrics <= 2:
        nrows, ncols = 1, num_metrics
    elif num_metrics <= 4:
        nrows, ncols = 2, 2
    elif num_metrics <= 6:
        nrows, ncols = 2, 3
    else:
        nrows, ncols = 3, 3
    
    fig, axes = viz._create_figure_layout('multi', nrows, ncols)
    if num_metrics == 1:
        axes = [axes]
    elif nrows == 1 or ncols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    try:
        # 获取配色方案
        if viz.config.journal_style == 'custom':
            colors_list = list(viz.color_schemes.custom_colors.values())
        else:
            colors_list = viz.color_schemes.get_journal_palette(viz.config.journal_style, len(modes) + 2)
        
        # 为每个模式分配颜色
        mode_colors = {
            'full': colors_list[0],      # 完整模型
            'low_only': colors_list[1],  # 低频模式
            'high_only': colors_list[2]  # 高频模式
        }
        
        # 为每个指标生成对比柱状图
        for i, metric in enumerate(metrics):
            if i >= len(axes):
                break
            
            ax = axes[i]
            
            # 提取各模式下的该指标值
            values = []
            labels = []
            colors = []
            
            for mode in modes:
                if metric in ablation_results[mode]:
                    values.append(ablation_results[mode][metric])
                    labels.append(_format_mode_name(mode))
                    colors.append(mode_colors.get(mode, colors_list[len(colors) % len(colors_list)]))
            
            if not values:
                ax.text(0.5, 0.5, f'No data for {metric}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{metric.upper()}')
                continue
            
            # 绘制柱状图
            bars = ax.bar(labels, values, color=colors, alpha=0.8, 
                         edgecolor='white', linewidth=1.2)
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.4f}',
                       ha='center', va='bottom', fontsize=viz.config.font_size-1)
            
            # 设置标题和标签
            ax.set_title(f'{metric.upper()}', fontsize=viz.config.title_size, fontweight='bold')
            ax.set_ylabel('Score', fontsize=viz.config.label_size)
            
            # 旋转x轴标签以避免重叠
            ax.tick_params(axis='x', rotation=45, labelsize=viz.config.font_size-1)
            
            # 添加网格
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_axisbelow(True)
        
        # 隐藏多余的子图
        for j in range(len(metrics), len(axes)):
            axes[j].set_visible(False)
        
        # 设置总标题
        domains_str = ', '.join([_normalize_domain_name(name) for name in domain_map.values()][:3])
        fig.suptitle(f'Frequency Ablation Analysis\n{domains_str}{_epoch_title(epoch)}', 
                    fontsize=viz.config.title_size, fontweight='bold', y=0.95)
        
        # 添加说明文本
        explanation = (
            "Full: Complete model with both frequency components\n"
            "Low-Only: Only low-frequency (long-term trend) components\n"
            "High-Only: Only high-frequency (short-term variation) components"
        )
        fig.text(0.02, 0.02, explanation, fontsize=viz.config.font_size-2, 
                style='italic', alpha=0.7, verticalalignment='bottom')
        
        plt.tight_layout(rect=[0, 0.1, 1, 0.92])  # 为标题和说明留出空间
        
        # 保存图片
        saved_files = []
        if save_plots:
            filename = f'frequency_ablation_comparison_epoch_{epoch}'
            saved_files = viz._save_figure(fig, filename)
        
        return fig, saved_files
        
    except Exception as e:
        warnings.warn(f"消融实验可视化失败: {e}")
        # 创建错误信息图
        fig, ax = plt.subplots(1, 1, figsize=viz.config.get_figsize('single'))
        ax.text(0.5, 0.5, f'Visualization error: {str(e)}', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Frequency Ablation Comparison - Error')
        return fig, []
    finally:
        gc.collect()


def create_journal_visualization(journal: str = 'nature') -> EnhancedVisualization:
    """创建期刊特定的可视化实例"""
    from .config import create_journal_config
    config = create_journal_config(journal)
    return EnhancedVisualization(config)


def _format_mode_name(mode: str) -> str:
    """格式化模式名称为显示友好的形式"""
    mode_mapping = {
        'full': 'Full Model',
        'low_only': 'Low-Freq Only',
        'high_only': 'High-Freq Only'
    }
    return mode_mapping.get(mode, mode.replace('_', ' ').title())