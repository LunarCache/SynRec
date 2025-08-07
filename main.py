import os
import time
import torch
import argparse
from tqdm import tqdm
from keys.model import HAGMRec
from keys.utils import *
import random
from torch.utils.tensorboard import SummaryWriter
import swanlab
# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np
from collections import OrderedDict, defaultdict
from keys.c_moe import PointWiseFeedForward
import torch.optim as optim

# ANSI color codes for terminal output
RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"

def setup_matplotlib():
    """Setup matplotlib for non-interactive use to prevent tkinter errors"""
    matplotlib.use('Agg')  # Use non-interactive backend
    plt.ioff()  # Turn off interactive mode
    # Set additional rcParams to prevent GUI-related errors
    import matplotlib as mpl
    mpl.rcParams['backend'] = 'Agg'
    mpl.rcParams['figure.max_open_warning'] = 0

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    
def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

def check_rating_strategy_compatibility(args):
    """
    检查增强Rating策略与其他功能的兼容性，并提供优化建议
    """
    rating_strategy = getattr(args, 'rating_strategy', 'simple')
    
    # 当前实现已简化，fourier策略与专业化机制兼容
    if rating_strategy == 'fourier':
        print(f"✅ Using Fourier rating strategy with simplified gating integration.")
        print(f"   This strategy is compatible with expert specialization mechanisms.")
    elif rating_strategy not in ['simple', 'legacy', 'fourier']:
        print(f"⚠️  WARNING: Rating strategy '{rating_strategy}' is not supported in current implementation.")
        print(f"   Supported strategies: 'simple', 'legacy', 'fourier'")
        print(f"   Falling back to 'fourier' strategy.")
        args.rating_strategy = 'fourier'
    
    return args  # Fix: Return the args object
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', required=True)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=100, type=int, help='Maximum sequence length.')
    parser.add_argument('--hidden_units', default=64, type=int, help='Size of hidden vectors.')
    parser.add_argument('--num_blocks', default=2, type=int, help='Number of transformer blocks.')
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--num_heads', default=2, type=int, help='Number of attention heads.')
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', default=False, type=str2bool)
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--seed', default=42, type=int, help='Seed for reproducibility.')
    parser.add_argument('--use_domain_sampling', default=False, type=str2bool, help='Enable domain-specific sampling,if false use global sampling')
    parser.add_argument('--use_domain_sampling_for_evaluation', default=False, type=str2bool, help='Enable domain-specific sampling for evaluation, if false use global sampling')
    # --- MoE Integration: Add MoE specific args ---
    parser.add_argument('--use_moe', default=True, type=str2bool, help='Enable/Disable MoE')
    parser.add_argument('--use_datasets', nargs='+', default=['beauty_5_5', 'games_5_5', 'ml-1m_5_5'], help='Datasets to use for multi-domain training')
    parser.add_argument('--use_domain_info', default=True, type=str2bool, help='Use domain info in MoE gating')
    parser.add_argument('--use_context', default=True, type=str2bool, help='Use context information in MoE gating')
    parser.add_argument('--use_rating_emb', default=True, type=str2bool, help='Use rating embedding to inform gating')
    parser.add_argument('--use_gated_fusion', default=True, type=str2bool, help='Use a gated mechanism to fuse rating embedding')
    parser.add_argument('--rating_pos_emb', default=False, type=str2bool, help='Add positional embedding to rating embeddings')
    # --- Enhanced Rating Module Parameters (简化版 + 自适应配置) ---
    parser.add_argument('--rating_strategy', default='fourier', type=str, 
                       choices=['simple', 'legacy', 'fourier'],
                       help='Strategy for rating information modeling: simple/legacy (backward compatibility), fourier (Fourier-based long/short-term feature extraction)')
    
    # 自适应配置参数
    parser.add_argument('--use_adaptive_rating_config', default=True, type=str2bool,
                       help='Enable adaptive rating configuration based on dataset characteristics')
    
    # 手动配置参数（当策略为fourier且自适应配置关闭时使用）
    parser.add_argument('--rating_num_frequencies', default=12, type=int, 
                       help='Number of frequency components for Fourier rating encoding (used when adaptive config disabled)')
    parser.add_argument('--rating_short_term_heads', default=1, type=int,
                       help='Number of attention heads for short-term rating patterns (used when adaptive config disabled)')
    parser.add_argument('--rating_long_term_heads', default=1, type=int,
                       help='Number of attention heads for long-term rating trends (used when adaptive config disabled)')
    # --- End Enhanced Rating Module Parameters ---
    parser.add_argument('--moe_num_experts', default=4, type=int, help='Number of experts in MoE')
    parser.add_argument('--moe_k', default=2, type=int, help='Number of experts to use for each token')
    parser.add_argument('--moe_noisy_gating', default=True, type=str2bool, help='Use noisy gating in MoE')
    parser.add_argument('--moe_routing_strategy', default='shared_base', type=str, choices=['vanilla', 'shared_base'], help='MoE routing strategy')
    parser.add_argument('--moe_load_balancing', default=True, type=str2bool, help='Use load balancing in MoE')
    parser.add_argument('--moe_balance_loss_weight', default=0.01, type=float, help='Weight for MoE load balancing loss')
    # --- Expert Specialization Optimization Parameters ---
    parser.add_argument('--gate_temperature', default=2.0, type=float, help='Initial temperature for gate softmax')
    parser.add_argument('--min_gate_temperature', default=0.1, type=float, help='Minimum temperature for gate softmax')
    parser.add_argument('--temperature_decay', default=0.995, type=float, help='Temperature decay rate per step')
    parser.add_argument('--use_specialization_loss', default=True, type=str2bool, help='Enable specialization loss for expert specialization')
    parser.add_argument('--specialization_weight', default=0.01, type=float, help='Weight for specialization loss')
    parser.add_argument('--use_contrastive_loss', default=True, type=str2bool, help='Enable contrastive learning for expert specialization')
    parser.add_argument('--contrastive_weight', default=0.01, type=float, help='Weight for contrastive loss')
    parser.add_argument('--use_adaptive_balance', default=True, type=str2bool, help='Use adaptive load balancing based on specialization')
    # --- End Expert Specialization Optimization ---
    parser.add_argument('--visualize', default=True, type=str2bool, help='Enable visualization of expert usage')
    parser.add_argument('--log_freq', default=100, type=int, help='Frequency of logging visualizations (in steps)')
    parser.add_argument('--tsne_log_freq', default=1, type=int, help='Frequency of logging t-SNE plots (in epochs)')
    parser.add_argument('--tsne_sample_size', default=512, type=int, help='Number of points to sample for t-SNE plot')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers for data loading.')
    # --- End MoE Integration ---
    # --- SwanLab Integration ---
    parser.add_argument('--swanlab_project', type=str, default='CMREC', help='SwanLab project name')
    parser.add_argument('--use_swanlab', default=True, type=str2bool, help='Enable/Disable SwanLab')
    # --- End SwanLab Integration ---
    args = parser.parse_args()
    
    # Check compatibility between rating strategy and other options
    args = check_rating_strategy_compatibility(args)
    
    return args

def log_fourier_rating_detailed_heatmap(writer, step, fourier_attention_data, layer_idx, prefix="Fourier_Rating_Detail", domain_config_manager=None, domain_map=None):
    """
    Log detailed Fourier rating attention patterns for multi-domain scenarios with adaptive sizing
    
    Args:
        writer: TensorBoard writer
        step: Current epoch
        fourier_attention_data: Dict mapping {domain_id: attention_dict} or single attention_dict
        layer_idx: Current transformer layer index
        prefix: Prefix for the log name
        domain_config_manager: DomainAdaptiveConfig instance for getting domain-specific configurations
        domain_map: Dict mapping domain_id to dataset_name
    """
    if not fourier_attention_data:
        return
    
    if isinstance(fourier_attention_data, dict):
        # 检查是否为多领域格式（键为数字domain_id）
        keys = list(fourier_attention_data.keys())
        if keys and isinstance(keys[0], (int, str)) and str(keys[0]).isdigit():
            # 多领域格式：{domain_id: attention_dict}
            for domain_id, fourier_attention_dict in fourier_attention_data.items():
                if fourier_attention_dict is None:
                    continue
                    
                # 提取attention组件
                short_term_attn = fourier_attention_dict.get('short_term_attention')
                long_term_attn = fourier_attention_dict.get('long_term_attention')
                adaptive_weights = fourier_attention_dict.get('adaptive_weights')
                
                if short_term_attn is None or long_term_attn is None:
                    continue
                
                # 🎯 获取领域特定配置
                domain_config = None
                domain_name = None
                if domain_config_manager and domain_map and domain_id in domain_map:
                    dataset_name = domain_map[domain_id]
                    domain_config = domain_config_manager.get_domain_config([dataset_name])
                    domain_name = dataset_name
                
                # 处理这个领域的attention数据
                _log_single_domain_fourier_attention(
                    writer, step, short_term_attn, long_term_attn, adaptive_weights,
                    layer_idx, f"{prefix}_Domain{domain_id}", domain_config, domain_name, domain_config_manager
                )
        else:
            # 单领域格式：直接包含attention数据的dict
            short_term_attn = fourier_attention_data.get('short_term_attention')
            long_term_attn = fourier_attention_data.get('long_term_attention')
            adaptive_weights = fourier_attention_data.get('adaptive_weights')
            
            if short_term_attn is not None and long_term_attn is not None:
                # 🎯 单领域情况下的配置处理
                domain_config = None
                domain_name = None
                if domain_config_manager and domain_map and len(domain_map) == 1:
                    # 单领域情况，使用第一个（也是唯一的）领域配置
                    first_domain_id = list(domain_map.keys())[0]
                    dataset_name = domain_map[first_domain_id]
                    domain_config = domain_config_manager.get_domain_config([dataset_name])
                    domain_name = dataset_name
                
                _log_single_domain_fourier_attention(
                    writer, step, short_term_attn, long_term_attn, adaptive_weights,
                    layer_idx, prefix, domain_config, domain_name, domain_config_manager
                )

def _log_single_domain_fourier_attention(writer, step, short_term_attn, long_term_attn, adaptive_weights, layer_idx, prefix, domain_config=None, domain_name=None, domain_config_manager=None):
    """Helper function to log attention data for a single domain with adaptive sizing"""
    # Convert to numpy and handle attention weight dimensions
    short_term_np = short_term_attn.detach().cpu().numpy()
    long_term_np = long_term_attn.detach().cpu().numpy()
    
    # Handle different dimensionalities
    if short_term_np.ndim == 4:  # (batch, heads, seq, seq)
        short_term_np = short_term_np.mean(axis=(0, 1))  # Average over batch and heads
    elif short_term_np.ndim == 3:  # (batch, seq, seq) - already averaged over heads
        short_term_np = short_term_np.mean(axis=0)  # Average over batch
    elif short_term_np.ndim == 2:  # (seq, seq) - single sample
        pass  # Use as is
    
    if long_term_np.ndim == 4:  # (batch, heads, seq, seq)
        long_term_np = long_term_np.mean(axis=(0, 1))  # Average over batch and heads
    elif long_term_np.ndim == 3:  # (batch, seq, seq) - already averaged over heads
        long_term_np = long_term_np.mean(axis=0)  # Average over batch
    elif long_term_np.ndim == 2:  # (seq, seq) - single sample
        pass  # Use as is
    
    #  Domain-aware adaptive sizing based on DomainAdaptiveConfig
    original_len = short_term_np.shape[0]
    if domain_config and 'max_len' in domain_config:
        effective_len = min(domain_config['max_len'], original_len)
        # 获取规范的显示名称
        if domain_name and domain_config_manager:
            display_name = domain_config_manager.get_display_name(domain_name)
            domain_info = f" - {display_name}"
        else:
            domain_info = f" - {domain_name}" if domain_name else ""
        # adaptive_info = f" (Adaptive: {effective_len}x{effective_len})"
        # 静默处理，不需要每次都打印尺寸变换信息
    else:
        effective_len = min(50, original_len)  # 默认安全值
        domain_info = ""
        # adaptive_info = f" (Default: {effective_len}x{effective_len})"
    
    # Crop attention matrices to effective region
    short_term_np = short_term_np[:effective_len, :effective_len]
    long_term_np = long_term_np[:effective_len, :effective_len]
    
    # Create adaptive figure size based on effective length
    base_size = max(4, effective_len * 0.12)  # 根据有效长度动态调整基础尺寸
    fig_width = min(base_size * 3, 24)  # 限制最大宽度
    fig_height = min(base_size, 8)  # 限制最大高度
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height))
    
    try:
        # Plot 1: Short-term attention (high-frequency patterns)
        im1 = axes[0].imshow(short_term_np, cmap='Reds', aspect='auto')
        axes[0].set_title(f'Short-term Rating Attention{domain_info}\n(High-frequency Patterns)')
        axes[0].set_xlabel('Rating Sequence Position (Key)')
        axes[0].set_ylabel('Rating Sequence Position (Query)')
        plt.colorbar(im1, ax=axes[0])
        
        # Plot 2: Long-term attention (low-frequency trends)
        im2 = axes[1].imshow(long_term_np, cmap='Blues', aspect='auto')
        axes[1].set_title(f'Long-term Rating Attention{domain_info}\n(Low-frequency Trends)')
        axes[1].set_xlabel('Rating Sequence Position (Key)')
        axes[1].set_ylabel('Rating Sequence Position (Query)')
        plt.colorbar(im2, ax=axes[1])
        
        # Plot 3: Adaptive weights distribution
        if adaptive_weights is not None:
            adaptive_np = adaptive_weights.detach().cpu().numpy().mean(axis=0)  # (seq, 3)
            # 也需要裁剪adaptive weights到有效长度
            adaptive_np = adaptive_np[:effective_len, :]
            im3 = axes[2].imshow(adaptive_np.T, cmap='Greens', aspect='auto')
            axes[2].set_title(f'Adaptive Scale Weights{domain_info}\n(Original, Short-term, Long-term)')
            axes[2].set_xlabel('Rating Sequence Position')
            axes[2].set_ylabel('Scale Type (0:Original, 1:Short, 2:Long)')
            axes[2].set_yticks([0, 1, 2])
            axes[2].set_yticklabels(['Original', 'Short-term', 'Long-term'])
            plt.colorbar(im3, ax=axes[2])
        else:
            axes[2].text(0.5, 0.5, 'Adaptive Weights\nNot Available', 
                        ha='center', va='center', transform=axes[2].transAxes)
            axes[2].set_title(f'Adaptive Scale Weights{domain_info}')
        
        plt.suptitle(f'Fourier Rating Attention Analysis{domain_info} - Layer {layer_idx} (Epoch {step})')
        plt.tight_layout()
        
        # Log to TensorBoard
        writer.add_figure(f'{prefix}/Layer_{layer_idx}_Detailed', fig, step)
        
        # Log to SwanLab if enabled
        if swanlab.get_run() is not None:
            # 使用与TensorBoard一致的命名，包含领域信息
            swanlab_key = f"rating_attention_detailed/{prefix.lower()}_layer_{layer_idx}_fourier_analysis"
            swanlab.log({
                swanlab_key: swanlab.Image(fig), 
                "epoch": step
            })
            
    except Exception as e:
        print(f"Warning: Failed to create visualization: {e}")
    finally:
        # Always close the figure and clear memory
        plt.close(fig)
        plt.close('all')  # Close any remaining figures
        del fig
        import gc
        gc.collect()  # Force garbage collection


def log_multi_domain_fourier_comparison(writer, step, all_fourier_data, domain_config_manager, domain_map, layer_idx):
    """
    创建多领域Fourier attention对比视图，展示不同领域的特性差异
    
    Args:
        writer: TensorBoard writer
        step: Current epoch
        all_fourier_data: Dict mapping {domain_id: attention_dict}
        domain_config_manager: DomainAdaptiveConfig instance
        domain_map: Dict mapping domain_id to dataset_name
        layer_idx: Current layer index
    """
    if not all_fourier_data or not isinstance(all_fourier_data, dict):
        return
    
    num_domains = len(all_fourier_data)
    if num_domains < 2:
        return  # 不足两个领域，无需对比
    
    try:
        # 创建对比布局: 2行 x N列 (短期 + 长期)
        fig, axes = plt.subplots(2, num_domains, figsize=(6 * num_domains, 12))
        if num_domains == 1:
            axes = axes.reshape(2, 1)
        
        for col, (domain_id, fourier_dict) in enumerate(sorted(all_fourier_data.items())):
            if fourier_dict is None:
                continue
                
            dataset_name = domain_map.get(domain_id, f"Domain{domain_id}")
            domain_config = domain_config_manager.get_domain_config([dataset_name]) if domain_config_manager else {}
            effective_len = domain_config.get('max_len', 50)
            
            # 获取规范的显示名称
            display_name = domain_config_manager.get_display_name(dataset_name) if domain_config_manager else dataset_name
            
            # 获取attention数据
            short_term_attn = fourier_dict.get('short_term_attention')
            long_term_attn = fourier_dict.get('long_term_attention')
            
            if short_term_attn is None or long_term_attn is None:
                continue
            
            # 处理维度并裁剪到有效长度
            short_term_np = short_term_attn.detach().cpu().numpy()
            long_term_np = long_term_attn.detach().cpu().numpy()
            
            if short_term_np.ndim > 2:
                short_term_np = short_term_np.mean(axis=tuple(range(short_term_np.ndim - 2)))
            if long_term_np.ndim > 2:
                long_term_np = long_term_np.mean(axis=tuple(range(long_term_np.ndim - 2)))
            
            # 裁剪到有效区域
            short_term_np = short_term_np[:effective_len, :effective_len]
            long_term_np = long_term_np[:effective_len, :effective_len]
            
            # 绘制短期attention
            im1 = axes[0, col].imshow(short_term_np, cmap='Reds', aspect='auto')
            axes[0, col].set_title(f'{display_name}\nShort-term ({effective_len}x{effective_len})')
            axes[0, col].set_xlabel('Position')
            axes[0, col].set_ylabel('Position')
            plt.colorbar(im1, ax=axes[0, col])
            
            # 绘制长期attention
            im2 = axes[1, col].imshow(long_term_np, cmap='Blues', aspect='auto')
            axes[1, col].set_title(f'{display_name}\nLong-term ({effective_len}x{effective_len})')
            axes[1, col].set_xlabel('Position')
            axes[1, col].set_ylabel('Position')
            plt.colorbar(im2, ax=axes[1, col])
        
        plt.suptitle(f'Multi-Domain Fourier Rating Attention Comparison - Layer {layer_idx} (Epoch {step})', fontsize=16)
        plt.tight_layout()
        
        # 记录到TensorBoard和WandB
        writer.add_figure(f'Multi_Domain_Fourier_Comparison/Layer_{layer_idx}', fig, step)
        if swanlab.get_run() is not None:
            swanlab.log({
                f"multi_domain_fourier_comparison/layer_{layer_idx}": swanlab.Image(fig),
                "epoch": step
            })
            
    except Exception as e:
        print(f"Warning: Failed to create multi-domain comparison: {e}")
    finally:
        plt.close(fig)
        plt.close('all')
        del fig
        import gc
        gc.collect()


def log_domain_expert_heatmap(writer, step, data, num_shared_experts, domain_map, strategy='vanilla', domain_config_manager=None):
    """Logs a heatmap of domain-to-expert routing, adapting to the routing strategy."""
    if data is None or data.numel() == 0: return
    
    fig, ax = plt.subplots(figsize=(12, max(8, data.shape[0] * 0.8)))
    
    try:
        # --- Adapt labels based on the routing strategy ---
        if strategy == 'shared_base':
            # For 'shared_base', use neutral expert labels to observe if specialization emerges.
            expert_labels = [f'Expert {i}' for i in range(data.shape[1])]
        else: # 'vanilla'
            # For 'vanilla', we explicitly mark which experts are designated as shared.
            expert_labels = []
            for i in range(data.shape[1]):
                label = str(i)
                if i < num_shared_experts:
                    label += " (Shared)"
                expert_labels.append(label)

        # 生成规范的领域标签
        domain_labels = []
        for i in range(data.shape[0]):
            dataset_name = domain_map.get(i, f"Unknown Domain {i}")
            if domain_config_manager:
                display_name = domain_config_manager.get_display_name(dataset_name)
                domain_labels.append(display_name)
            else:
                domain_labels.append(dataset_name)
        
        # 🌿 使用清新的nature配色方案 - YlGn (黄绿渐变)
        # 其他可选的nature配色: "BuGn"(蓝绿), "GnBu"(绿蓝), "YlOrRd"(黄橙红)
        sns.heatmap(data.cpu().numpy(), annot=True, fmt=".2f", cmap="YlGn", ax=ax,
                    xticklabels=expert_labels, yticklabels=domain_labels,
                    cbar_kws={'label': 'Routing Weight'})
        
        ax.set_title(f'Domain-to-Expert Routing - Epoch {step}', fontsize=14, pad=20)
        ax.set_xlabel('Experts', fontsize=12)
        ax.set_ylabel('Domains', fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        writer.add_figure('Domain_Expert_Routing_Heatmap', fig, global_step=step)
        if swanlab.get_run() is not None:
            # Log heatmap with epoch for consistency
            swanlab.log({"Domain_Expert_Routing_Heatmap": swanlab.Image(fig), "epoch": step})
            
    except Exception as e:
        print(f"Warning: Failed to create domain expert heatmap: {e}")
    finally:
        # Always close the figure and clear memory
        plt.close(fig)
        plt.close('all')  # Close any remaining figures
        del fig
        import gc
        gc.collect()  # Force garbage collection

def log_tsne_expert_specialization(writer, step, embeddings, labels, domains, num_experts, domain_map, args, sample_size=1024, domain_config_manager=None):
    """Logs a t-SNE plot of token embeddings, colored by expert or domain."""
    if embeddings is None or embeddings.numel() == 0: return
    
    # --- Sample data to avoid excessive computation ---
    num_points = embeddings.shape[0]
    if num_points > sample_size:
        indices = np.random.choice(num_points, sample_size, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]
        domains = domains[indices]

    # --- Perform t-SNE ---
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, max_iter=1000, random_state=args.seed)
    embeddings_2d = tsne.fit_transform(embeddings.cpu().numpy())

    # --- Create Plots (colored by expert and by domain) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    
    try:
        #  使用高对比度颜色图提升区分度
        # Set1和Dark2提供更鲜明的离散颜色对比
        cmap_experts = plt.get_cmap('Set1', num_experts) if num_experts <= 9 else plt.get_cmap('tab20', num_experts)
        cmap_domains = plt.get_cmap('Dark2', len(domain_map)) if len(domain_map) <= 8 else plt.get_cmap('tab10', len(domain_map))

        # Plot 1: Colored by Expert ID - 增强对比度
        scatter1 = ax1.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                              c=labels.cpu().numpy(), cmap=cmap_experts, 
                              alpha=0.85, s=20, vmin=-0.5, vmax=num_experts-0.5,
                              edgecolors='white', linewidths=0.3)
        ax1.set_title(f' t-SNE of Embeddings Colored by Expert ID (Epoch {step})', fontsize=14)
        ax1.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax1.set_ylabel('t-SNE Dimension 2', fontsize=12)
        # --- 优化: 根据路由策略生成更清晰的图例 ---
        if args.moe_routing_strategy == 'shared_base':
            # 在新策略下，专家是领域专家
            expert_legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'Expert {i}',
                                             markerfacecolor=cmap_experts(i), markersize=12,
                                             markeredgecolor='white', markeredgewidth=0.5) for i in range(len(domain_map))]
            ax1.legend(handles=expert_legend_elements, title="Experts", fontsize=10)
        else:
            # 在旧策略下，使用通用标签
            expert_legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'Expert {i}',
                                             markerfacecolor=cmap_experts(i), markersize=12,
                                             markeredgecolor='white', markeredgewidth=0.5) for i in range(num_experts)]
            ax1.legend(handles=expert_legend_elements, title="Experts", fontsize=10)

        # Plot 2: Colored by Domain ID - 增强对比度
        scatter2 = ax2.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                              c=domains.cpu().numpy(), cmap=cmap_domains, 
                              alpha=0.85, s=20, vmin=-0.5, vmax=len(domain_map)-0.5,
                              edgecolors='black', linewidths=0.3)
        ax2.set_title(f' t-SNE of Embeddings Colored by Domain ID (Epoch {step})', fontsize=14)
        ax2.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax2.set_ylabel('t-SNE Dimension 2', fontsize=12)
        # 生成规范的领域图例标签
        domain_legend_labels = []
        for i in range(len(domain_map)):
            dataset_name = domain_map[i]
            if domain_config_manager:
                display_name = domain_config_manager.get_display_name(dataset_name)
                domain_legend_labels.append(display_name)
            else:
                domain_legend_labels.append(dataset_name)
        
        domain_legend_elements = [Line2D([0], [0], marker='o', color='w', label=domain_legend_labels[i],
                                         markerfacecolor=cmap_domains(i), markersize=12, 
                                         markeredgecolor='black', markeredgewidth=0.5) for i in range(len(domain_map))]
        ax2.legend(handles=domain_legend_elements, title="Domains", fontsize=10)

        #  优化整体布局和背景
        for ax in [ax1, ax2]:
            ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
            ax.set_facecolor('#fafafa')  # 浅灰背景增强对比度
        
        plt.tight_layout()
        writer.add_figure('t-SNE_Specialization', fig, global_step=step)
        if swanlab.get_run() is not None:
            # Log t-SNE plots to SwanLab with epoch as step for consistency with evaluation metrics
            swanlab.log({"t-SNE_Specialization": swanlab.Image(fig), "epoch": step})
            
    except Exception as e:
        print(f"Warning: Failed to create t-SNE visualization: {e}")
    finally:
        # Always close the figure and clear memory
        plt.close(fig)
        plt.close('all')  # Close any remaining figures
        del fig
        import gc
        gc.collect()  # Force garbage collection


def main():
    # Setup matplotlib early to prevent tkinter issues
    setup_matplotlib()
    
    args = parse_args()
    # Set the seed for the entire environment
    if args.seed is not None:
        set_seed(args.seed)

    # Initialize SwanLab
    if args.use_swanlab:
        swanlab.init(
            project=args.swanlab_project,
            experiment_name='-'.join(args.use_datasets),
            config=vars(args)
        )

    # --- MoE Integration: Create a unified name for the experiment directory ---
    dataset_name_str = '-'.join(args.use_datasets)
    experiment_dir = os.path.join('exp', dataset_name_str + '_' + args.train_dir)
    if not os.path.isdir(experiment_dir):
        os.makedirs(experiment_dir)
    with open(os.path.join(experiment_dir, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
    f.close()

    # global dataset
    # --- MoE Integration: Use new multi-domain data partition and sampler ---
    dataset = partition_multi_domain(args.use_datasets)
    [user_train, user_valid, user_test, user_to_domain, usernum, itemnum, domain_to_item_range] = dataset
    args.num_domains = len(args.use_datasets) # Save number of domains
    domain_map = {i: name for i, name in enumerate(args.use_datasets)}
    # --- End MoE Integration ---
    
    # --- Visualization Setup ---
    writer = None
    if args.visualize:
        log_dir = os.path.join(experiment_dir, 'logs')
        writer = SummaryWriter(log_dir=log_dir)
    # --- End Visualization Setup ---

    # num_batch = len(user_train) // args.batch_size # tail? + ((len(user_train) % args.batch_size) != 0)
    num_batch = (len(user_train) - 1) // args.batch_size + 1
    asl = {f'domain_{i}': 0 for i in range(args.num_domains)} # total sequence length for each domain
    usrnum_of_domain = {f'domain_{i}': 0 for i in range(args.num_domains)} # user number of each domain

    print('\n{:-^100}'.format('Average sequence length of each domain'))
    for u in user_train:
        asl[f'domain_{user_to_domain[u]}'] += len(user_train[u])
        usrnum_of_domain[f'domain_{user_to_domain[u]}'] += 1
    
    for i in range(args.num_domains):
        print(f'{domain_map[i]}: {asl[f"domain_{i}"] / usrnum_of_domain[f"domain_{i}"]:.2f}')

    f = open(os.path.join(experiment_dir, 'log.txt'), 'w')
    f.write('epoch\tvalid_metrics\ttest_metrics\n')
    
    # --- MoE Integration: Use PyTorch DataLoader with custom StratifiedSampler for efficient and correct data loading ---
    train_dataset = MoerecDataset(user_train, user_to_domain, usernum, itemnum, args.maxlen, args, domain_to_item_range)
    train_sampler = StratifiedSampler(train_dataset)
    train_collator = MoerecCollator(maxlen=args.maxlen)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler, # Use our custom sampler
        shuffle=False, # Shuffle must be False when using a custom sampler
        num_workers=args.num_workers,
        collate_fn=train_collator,
        pin_memory=True # Speeds up data transfer to GPU
    )
    # --- End MoE Integration ---
    model = HAGMRec(usernum, itemnum, args).to(args.device) # no ReLU activation in original SASRec implementation?
    
    # --- MoE Integration: Calculate and print parameter counts ---
    total_params = 0
    original_params = 0
    moe_params = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if 'moe_ffn' in name:
            moe_params += num_params
        else:
            original_params += num_params

    print('\n{:-^100}'.format("Model Parameters"))
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Transformer Block Parameters: {original_params:,}")
    print(f"  MoE-related Parameters: {moe_params:,}")
    print("-" * 100,"\n")
    # --- End MoE Integration ---

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers

    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0

    # this fails embedding init 'Embedding' object has no attribute 'dim'
    # model.apply(torch.nn.init.xavier_uniform_)
    
    model.train() # enable model training
    
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except: # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb; pdb.set_trace()
            
    
    if args.inference_only:
        model.eval()
        print('Running inference-only evaluation...')
        
        # 创建日志文件（如果不存在）
        f = open(os.path.join(experiment_dir, 'log.txt'), 'w')
        f.write('epoch\tvalid_metrics\ttest_metrics\n')
        
        # 进行评估
        t_valid = evaluate_batched(model, dataset, args, 'valid')
        t_test = evaluate_batched(model, dataset, args, 'test')

        # 获取epoch信息（从权重文件名提取，如果可能的话）
        epoch_num = 'inference'
        if args.state_dict_path and 'epoch=' in args.state_dict_path:
            try:
                tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
                epoch_num = int(tail[:tail.find('.')])
            except:
                epoch_num = 'inference'
        
        # 打印结果
        print(f'Inference Results - Valid: NDCG@10={t_valid["overall_NDCG@10"]:.4f}, HR@10={t_valid["overall_HT@10"]:.4f} | Test: NDCG@10={t_test["overall_NDCG@10"]:.4f}, HR@10={t_test["overall_HT@10"]:.4f}')
        print(f"Valid metrics: {t_valid}")
        print(f"Test metrics: {t_test}")
        
        # 保存结果到log.txt文件（与训练模式相同格式）
        valid_metrics_str = ",".join([f"{k}:{v:.4f}" for k, v in sorted(t_valid.items())])
        test_metrics_str = ",".join([f"{k}:{v:.4f}" for k, v in sorted(t_test.items())])
        f.write(f'{epoch_num}\t{valid_metrics_str}\t{test_metrics_str}\n')
        f.close()
        
        print(f"Results saved to: {os.path.join(experiment_dir, 'log.txt')}")
        print("You can now use analyze_results.py to visualize and compare these results.")
        return

    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
    bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T = 0.0
    
    # 初始化领域配置管理器用于自适应可视化
    domain_config_manager = None
    if args.visualize:
        try:
            from keys.domain_config import create_domain_config
            domain_config_manager = create_domain_config()
            print(f"🔧 Initialized domain-aware adaptive visualization for {len(args.use_datasets)} domains")
            print(f"   Domains: {', '.join(args.use_datasets)}")
        except Exception as e:
            print(f"⚠️  Failed to load domain config for visualization: {e}")
            print("   Falling back to default visualization")
            domain_config_manager = None
    
    t0 = time.time()
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break # just to decrease identition
        
        # --- Initialize epoch-level accumulators for visualization ---
        epoch_domain_expert_load = None
        epoch_viz_step_count = 0
        # --- End epoch-level accumulators ---
        
        pbar = tqdm(train_loader, desc=f"{BOLD}{BLUE}Epoch {epoch}/{args.num_epochs}{RESET}", colour='blue')
        for step, (u, seq, rating_seq, pos, neg, domain_id) in enumerate(pbar):
            # Move batch to device
            u, seq, rating_seq, pos, neg, domain_id = u.to(args.device), seq.to(args.device), rating_seq.to(args.device), pos.to(args.device), neg.to(args.device), domain_id.to(args.device)

            pos_logits, neg_logits, moe_loss_dict, viz_data = model(u, seq, pos, neg, rating_seqs=rating_seq, domain_ids=domain_id)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            
            adam_optimizer.zero_grad()
            indices = torch.where(pos != 0)
            
            postfix_data = OrderedDict()
            # --- MoE Integration: Calculate and log individual losses ---
            bpr_loss = bce_criterion(pos_logits[indices], pos_labels[indices]) + bce_criterion(neg_logits[indices], neg_labels[indices])
            postfix_data['bpr_loss'] = f"{BOLD}{BLUE}{bpr_loss.item():.4f}{RESET}"

            loss = bpr_loss
            
            for i in moe_loss_dict.keys():
                if torch.is_tensor(moe_loss_dict[i]):
                    loss = loss + moe_loss_dict[i]
                    postfix_data[i] = f"{BOLD}{BLUE}{moe_loss_dict[i].item():.4f}{RESET}"

            # --- End MoE Integration ---
            if args.l2_emb > 0:
                # Apply L2 regularization to item embeddings
                for param in model.item_emb.parameters(): loss = loss + args.l2_emb * torch.norm(param)
            
            loss.backward()
            adam_optimizer.step()

            postfix_data['loss'] = f"{BOLD}{BLUE}{loss.item():.4f}{RESET}"
            pbar.set_postfix(postfix_data)

            # --- SwanLab Integration: Log training metrics ---
            if args.use_swanlab:
                log_data = {
                    'train/loss': loss.item(),
                    'train/bpr_loss': bpr_loss.item(),
                    'learning_rate': adam_optimizer.param_groups[0]['lr']
                }
                for k, v_val in moe_loss_dict.items(): # Renamed v to v_val
                    if torch.is_tensor(v_val):
                        log_data[f'train/{k}'] = v_val.item()
                # Calculate global step for SwanLab
                swanlab_global_step = (epoch - 1) * len(train_loader) + step
                swanlab.log(log_data, step=swanlab_global_step)
            # --- End SwanLab Integration ---
            
            # --- Visualization Logging ---
            if args.visualize:
                # --- Accumulate data for epoch-level logging ---
                if 'domain_expert_load' in viz_data:
                    if epoch_domain_expert_load is None:
                        epoch_domain_expert_load = viz_data['domain_expert_load'].clone()
                    else:
                        epoch_domain_expert_load += viz_data['domain_expert_load']
                    epoch_viz_step_count += 1
                
                # Rating attention weights will be processed only at epoch level to reduce overhead
                
                # Log expert load scalars at step level (keeping this for training monitoring)
                if step % args.log_freq == 0 and 'expert_load' in viz_data:
                    global_step = (epoch - 1) * len(train_loader) + step
                    expert_load = viz_data['expert_load']
                    # --- 优化: 根据路由策略生成更清晰的标量标签 ---
                    if args.moe_routing_strategy == 'shared_base':
                        # 在新策略下，负载是针对领域专家的
                        scalar_dict = {f'Load/Domain_{domain_map[i]}': val.item() for i, val in enumerate(expert_load)}
                    else:
                        # 在旧策略下，使用通用标签
                        scalar_dict = {f'Load/Expert_{i}': val.item() for i, val in enumerate(expert_load)}
                    writer.add_scalars('Expert_Load_Distribution', scalar_dict, global_step)
                    if args.use_swanlab:
                        swanlab.log({f'train_expert_load/Domain_{domain_map[i]}' if args.moe_routing_strategy == 'shared_base' else f'train_expert_load/Expert_{i}': val.item() for i, val in enumerate(expert_load)}, step=global_step)

        if args.visualize and epoch % args.tsne_log_freq == 0:
             # Log t-SNE plot at the end of the epoch
            if 'tsne_embeddings' in viz_data:
                log_tsne_expert_specialization(
                    writer, epoch,
                    viz_data['tsne_embeddings'],
                    viz_data['tsne_labels'],
                    viz_data['tsne_domains'],
                    # 适配 'shared_base' 策略，此时专家数量为领域专家数量
                    model.forward_layers[0].moe_ffn.num_domain_experts if args.moe_routing_strategy == 'shared_base' else model.forward_layers[0].moe_ffn.num_experts,
                    domain_map,
                    args,
                    sample_size=args.tsne_sample_size,
                    domain_config_manager=domain_config_manager
                )

        # --- Log epoch-level domain-expert heatmap ---
        if args.visualize and epoch_domain_expert_load is not None and epoch_viz_step_count > 0:
            # Calculate average domain-expert load over the entire epoch
            avg_domain_expert_load = epoch_domain_expert_load / epoch_viz_step_count
            log_domain_expert_heatmap(
                writer, epoch, avg_domain_expert_load, 
                model.forward_layers[0].moe_ffn.num_shared_experts, 
                domain_map, 
                strategy=args.moe_routing_strategy,
                domain_config_manager=domain_config_manager
            )
        
        # --- Log epoch-level rating attention heatmap ---
        if args.visualize and 'fourier_rating_attention_detailed' in viz_data:
            fourier_detailed = viz_data['fourier_rating_attention_detailed']
            for layer_idx, fourier_attn_data in enumerate(fourier_detailed):
                if fourier_attn_data is not None:
                    # 🎯 传递领域配置信息给可视化函数
                    log_fourier_rating_detailed_heatmap(
                        writer, epoch, fourier_attn_data, layer_idx,
                        domain_config_manager=domain_config_manager,
                        domain_map=domain_map
                    )
                    
                    # 🎯 额外生成多领域对比视图（如果是多领域数据）
                    if (isinstance(fourier_attn_data, dict) and 
                        len(fourier_attn_data) > 1 and 
                        domain_config_manager is not None):
                        log_multi_domain_fourier_comparison(
                            writer, epoch, fourier_attn_data, 
                            domain_config_manager, domain_map, layer_idx
                        )
        # --- End epoch-level heatmap logging ---

        if epoch % 1 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            t_valid = evaluate_batched(model, dataset, args, 'valid')
            t_test = evaluate_batched(model, dataset, args, 'test')
            print('epoch:%d, time: %f(s)' % (epoch, T))

            def pretty_print_metrics(metrics_dict, title, color_code):
                print(f"\n  {BOLD}{color_code}[{title}]{RESET}")
                
                domain_metrics = defaultdict(dict)
                overall_metrics = {}
                
                for k, v in sorted(metrics_dict.items()):
                    if k.startswith('domain_'):
                        parts = k.split('_')
                        domain_id = int(parts[1])
                        metric_name = '_'.join(parts[2:])
                        domain_metrics[domain_id][metric_name] = v
                    elif k.startswith('overall_'):
                        metric_name = k.replace('overall_', '')
                        overall_metrics[metric_name] = v

                for domain_id, d_metrics in sorted(domain_metrics.items()):
                    domain_name = domain_map.get(domain_id, f"Unknown Domain {domain_id}")
                    print(f"    - {BOLD}Domain: {domain_name}{RESET}")
                    metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in sorted(d_metrics.items())])
                    print(f"        {metrics_str}")
                
                if overall_metrics:
                    print("    " + "-"*50)
                    print(f"    - {BOLD}Overall{RESET}")
                    metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in sorted(overall_metrics.items())])
                    print(f"        {metrics_str}")

            pretty_print_metrics(t_valid, "Full Valid Metrics", GREEN)
            pretty_print_metrics(t_test, "Full Test Metrics", CYAN)

            # --- SwanLab Integration: Log validation and test metrics against epoch ---
            if args.use_swanlab:
                eval_log_dict = {"epoch": epoch}
                # Consolidate all validation and test metrics into a single dictionary
                for key, value in t_valid.items():
                    metric_name = f'eval/valid_{key}'
                    eval_log_dict[metric_name] = value
                for key, value in t_test.items():
                    metric_name = f'eval/test_{key}'
                    eval_log_dict[metric_name] = value
                
                # Log all evaluation metrics at once against the 'epoch' step
                swanlab.log(eval_log_dict)
            # --- End SwanLab Integration ---

            # Using NDCG@10 for model saving criteria
            if t_valid['overall_NDCG@10'] > best_val_ndcg or t_valid['overall_HT@10'] > best_val_hr:
                best_val_ndcg = t_valid['overall_NDCG@10']
                best_val_hr = t_valid['overall_HT@10']
                best_test_ndcg = t_test['overall_NDCG@10']
                best_test_hr = t_test['overall_HT@10']
                folder = experiment_dir
                fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
                fname = fname.format(epoch, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
                model_path = os.path.join(folder, fname)
                torch.save(model.state_dict(), model_path)
                # --- SwanLab Integration: Save model artifact ---
                # if args.use_swanlab:
                #     swanlab.save(model_path)
                # --- End SwanLab Integration ---

            # Format the metrics string for log file
            valid_metrics_str = ",".join([f"{k}:{v:.4f}" for k, v in sorted(t_valid.items())])
            test_metrics_str = ",".join([f"{k}:{v:.4f}" for k, v in sorted(t_test.items())])
            f.write(f'{epoch}\t{valid_metrics_str}\t{test_metrics_str}\n')
            f.flush()
            t0 = time.time()
            model.train()
    
        if epoch == args.num_epochs:
            folder = experiment_dir
            fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            torch.save(model.state_dict(), os.path.join(folder, fname))
    
    f.close()
    if writer is not None:
        writer.close()
    # --- SwanLab Integration ---
    if args.use_swanlab:
        swanlab.finish()
    # --- End SwanLab Integration ---
    # sampler.close() # The new sampler does not need to be closed.
    print("Done")

if __name__ == '__main__':
    main()
