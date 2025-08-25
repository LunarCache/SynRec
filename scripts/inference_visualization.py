#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference_visualization.py
--------------------------
推理时专家热力图和t-SNE图可视化脚本

该脚本加载训练好的模型，在推理数据上运行并生成与main.py训练时相同的可视化：
1. 专家路由热力图 (Expert routing heatmap)
2. t-SNE专家专业化图 (t-SNE specialization plot) 
3. 可选的Fourier频率分析

支持期刊级别的高质量图表输出。
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path

# 确保项目根目录在Python路径中
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 设置matplotlib非交互模式
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 核心模块导入
from keys.model import HAGMRec
from keys.utils import partition_multi_domain, MoerecDataset, MoerecCollator, StratifiedSampler

# 可视化模块导入
try:
    from visualization import (
        VisualizationConfig,
        EnhancedVisualization,
        plot_expert_routing_journal,
        plot_tsne_specialization_journal,
        plot_inference_combined_overview,
        plot_multi_domain_fourier_comparison_journal,
        export_figure_journal,
        create_journal_config,
        apply_journal_style
    )
    ENHANCED_VIZ_AVAILABLE = True
    print("✓ Enhanced visualization modules loaded successfully")
except ImportError as e:
    ENHANCED_VIZ_AVAILABLE = False
    print(f"⚠ Enhanced visualization not available: {e}")
    print("  Falling back to basic visualization")

# ANSI颜色代码
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RED = "\033[91m"


def str2bool(s):
    """字符串转布尔值"""
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


def set_seed(seed):
    """设置所有随机种子以确保可重复性"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"🎲 Set random seed to {seed} for reproducible results")


def load_experiment_args(experiment_dir):
    """加载实验参数"""
    args_file = os.path.join(experiment_dir, 'args.txt')
    if not os.path.exists(args_file):
        raise FileNotFoundError(f"Args file not found: {args_file}")
    
    training_args = {}
    with open(args_file, 'r') as f:
        for line in f:
            if ',' in line:
                key, value = line.strip().split(',', 1)
                training_args[key] = value
    
    print(f"✓ Loaded training arguments from: {args_file}")
    return training_args


def create_model_args(training_args, datasets, device):
    """根据训练参数创建模型配置"""
    class ModelArgs:
        def __init__(self):
            self.device = device
            self.hidden_units = int(training_args.get('hidden_units', 64))
            self.maxlen = int(training_args.get('maxlen', 100))
            self.dropout_rate = float(training_args.get('dropout_rate', 0.5))
            self.num_heads = int(training_args.get('num_heads', 2))
            self.num_blocks = int(training_args.get('num_blocks', 2))
            self.l2_emb = float(training_args.get('l2_emb', 0.0))
            
            # MoE 参数
            self.use_moe = True
            self.moe_num_experts = int(training_args.get('moe_num_experts', 4))
            self.moe_k = int(training_args.get('moe_k', 2))
            self.moe_routing_strategy = training_args.get('moe_routing_strategy', 'shared_base')
            self.moe_load_balancing = True
            self.moe_balance_loss_weight = float(training_args.get('moe_balance_loss_weight', 0.01))
            self.moe_noisy_gating = True
            
            # Rating embedding 参数
            self.use_rating_emb = True
            self.rating_strategy = training_args.get('rating_strategy', 'temporal_fourier')
            self.rating_pos_emb = False
            
            # 采样参数（MoerecDataset需要）
            self.use_domain_sampling = str(training_args.get('use_domain_sampling', 'False')).lower() == 'true'
            
            # 可视化参数（MoE模块需要）
            self.visualize = True  # 推理可视化时始终启用
            
            # 领域相关参数
            self.num_domains = len(datasets)
            self.use_domain_info = True
            self.use_gated_fusion = True
            
            # 损失权重
            self.use_specialization_loss = True
            self.specialization_weight = float(training_args.get('specialization_weight', 0.01))
            self.use_contrastive_loss = True
            self.contrastive_weight = float(training_args.get('contrastive_weight', 0.01))
    
    return ModelArgs()


def setup_enhanced_visualization(args):
    """设置增强可视化系统"""
    if not ENHANCED_VIZ_AVAILABLE:
        return None
    
    try:
        # 创建期刊特定的配置
        config = create_journal_config(args.journal_style)
        config.dpi = args.viz_dpi
        config.figure_format = args.viz_format
        config.save_formats = [args.viz_format, 'png'] if args.save_publication_figs else [args.viz_format]
        config.output_directory = args.output_dir
        
        # 应用期刊样式
        apply_journal_style(args.journal_style)
        
        print(f"✓ Enhanced visualization configured for {args.journal_style} journal style")
        print(f"  DPI: {config.dpi}, Format: {config.figure_format}")
        print(f"  Output: {config.output_directory}")
        
        return config
    except Exception as e:
        print(f"⚠ Failed to setup enhanced visualization: {e}")
        return None


def collect_visualization_data(model, data_loader, args, domain_map, max_batches=None):
    """
    在推理过程中收集可视化数据
    
    Args:
        model: 训练好的模型
        data_loader: 数据加载器
        args: 参数配置
        domain_map: 领域映射
        max_batches: 最大处理批次数量（用于快速测试）
    
    Returns:
        dict: 聚合的可视化数据
    """
    print(f"{BOLD}{BLUE}🔍 Collecting visualization data during inference...{RESET}")
    
    # 重要：为了生成可视化数据，需要将模型设置为训练模式
    # 因为可视化数据的生成代码在 self.training 条件下
    original_mode = model.training
    model.train()  # 临时设置为训练模式以生成可视化数据
    print(f"  ⚙️ Temporarily setting model to train mode for visualization data collection")
    
    # 初始化可视化数据收集器
    collected_data = {
        'domain_expert_load_sum': None,
        'domain_expert_load_count': 0,
        'tsne_embeddings': [],
        'tsne_labels': [],
        'tsne_domains': [],
        'fourier_data': []
    }
    
    with torch.no_grad():  # 仍然使用no_grad来避免梯度计算
        pbar = tqdm(data_loader, desc=f"{BLUE}Collecting viz data{RESET}", colour='blue')
        for batch_idx, (u, seq, rating_seq, pos, neg, domain_id) in enumerate(pbar):
            if max_batches and batch_idx >= max_batches:
                break
                
            # 移动数据到设备
            u = u.to(args.device)
            seq = seq.to(args.device)
            rating_seq = rating_seq.to(args.device)
            pos = pos.to(args.device)
            neg = neg.to(args.device)
            domain_id = domain_id.to(args.device)
            
            # 运行模型前向传播获取可视化数据
            try:
                pos_logits, neg_logits, moe_loss_dict, viz_data = model(
                    u, seq, pos, neg, rating_seqs=rating_seq, domain_ids=domain_id
                )
                
                # 收集专家负载数据
                if 'domain_expert_load' in viz_data:
                    domain_expert_load = viz_data['domain_expert_load'].detach().cpu()
                    if collected_data['domain_expert_load_sum'] is None:
                        collected_data['domain_expert_load_sum'] = domain_expert_load.clone()
                    else:
                        collected_data['domain_expert_load_sum'] += domain_expert_load
                    collected_data['domain_expert_load_count'] += 1
                
                # 收集t-SNE数据
                if 'tsne_embeddings' in viz_data:
                    collected_data['tsne_embeddings'].append(viz_data['tsne_embeddings'].detach().cpu())
                if 'tsne_labels' in viz_data:
                    collected_data['tsne_labels'].append(viz_data['tsne_labels'].detach().cpu())
                if 'tsne_domains' in viz_data:
                    collected_data['tsne_domains'].append(viz_data['tsne_domains'].detach().cpu())
                
                # 收集Fourier数据（如果可用）
                if 'fourier_rating_attention_detailed' in viz_data:
                    collected_data['fourier_data'].append(viz_data['fourier_rating_attention_detailed'])
                
                # 更新进度条
                pbar.set_postfix({
                    'Batches': f"{batch_idx + 1}",
                    'Expert loads': f"{collected_data['domain_expert_load_count']}",
                    't-SNE samples': f"{len(collected_data['tsne_embeddings'])}"
                })
                
            except Exception as e:
                print(f"⚠ Error processing batch {batch_idx}: {e}")
                continue
    
    # 恢复原始模型模式
    if original_mode:
        model.train()
    else:
        model.eval()
    print(f"  ⚙️ Restored model to original mode: {'train' if original_mode else 'eval'}")
    
    # 处理收集的数据
    if collected_data['domain_expert_load_count'] > 0:
        collected_data['avg_domain_expert_load'] = (
            collected_data['domain_expert_load_sum'] / collected_data['domain_expert_load_count']
        )
    
    # 合并t-SNE数据
    if collected_data['tsne_embeddings']:
        collected_data['combined_tsne_embeddings'] = torch.cat(collected_data['tsne_embeddings'], dim=0)
        collected_data['combined_tsne_labels'] = torch.cat(collected_data['tsne_labels'], dim=0)
        collected_data['combined_tsne_domains'] = torch.cat(collected_data['tsne_domains'], dim=0)
    
    print(f"✓ Collected visualization data from {collected_data['domain_expert_load_count']} batches")
    print(f"  - Expert load data: {collected_data['domain_expert_load_count']} batches")
    print(f"  - t-SNE samples: {len(collected_data['tsne_embeddings'])} batches")
    print(f"  - Fourier data: {len(collected_data['fourier_data'])} batches")
    
    return collected_data


def generate_expert_routing_visualization(collected_data, domain_map, routing_strategy, args, viz_config):
    """生成专家路由热力图"""
    if not ENHANCED_VIZ_AVAILABLE or viz_config is None:
        print(f"⚠️ Enhanced visualization not available - expert routing visualization skipped")
        return None
    
    if 'avg_domain_expert_load' not in collected_data:
        print(f"⚠️ No expert routing data available")
        return None
    
    try:
        print(f"{BOLD}{GREEN}🎨 Generating expert routing heatmap...{RESET}")
        
        # 准备数据 - 确保数据类型正确
        avg_load = collected_data['avg_domain_expert_load']
        if isinstance(avg_load, np.ndarray):
            avg_load = torch.from_numpy(avg_load)
        
        # 准备标签
        domain_labels = []
        for i in range(len(avg_load)):
            raw_dataset_name = domain_map.get(i, f"Unknown Domain {i}")
            if not raw_dataset_name.startswith("Unknown Domain"):
                try:
                    from visualization.enhanced_plots import _normalize_domain_name
                    normalized_name = _normalize_domain_name(raw_dataset_name)
                    domain_labels.append(normalized_name)
                except ImportError:
                    domain_labels.append(raw_dataset_name)
            else:
                domain_labels.append(raw_dataset_name)
        
        if routing_strategy == 'shared_base':
            expert_labels = [f"Expert {i}" for i in range(len(avg_load[0]))]
        else:
            num_shared = 1 if len(avg_load[0]) > len(domain_map) else 0
            expert_labels = []
            for i in range(len(avg_load[0])):
                if i < num_shared:
                    expert_labels.append(f"Shared Expert {i}")
                else:
                    expert_labels.append(f"Domain Expert {i-num_shared}")
        
        # 使用增强可视化
        fig, saved_files = plot_expert_routing_journal(
            avg_load, domain_labels, expert_labels, 
            epoch="inference", config=viz_config, 
            save_plots=args.save_publication_figs
        )
        
        if saved_files:
            print(f"  ✓ Expert routing heatmap saved: {len(saved_files)} files")
            for file_path in saved_files:
                print(f"    - {file_path}")
        
        return fig, saved_files
        
    except Exception as e:
        print(f"⚠ Expert routing visualization failed: {e}")
        return None


def generate_tsne_specialization_visualization(collected_data, domain_map, args, viz_config):
    """生成t-SNE专家专业化图"""
    if not ENHANCED_VIZ_AVAILABLE or viz_config is None:
        print(f"⚠️ Enhanced visualization not available - t-SNE visualization skipped")
        return None
    
    if 'combined_tsne_embeddings' not in collected_data:
        print(f"⚠️ No t-SNE data available")
        return None
    
    try:
        print(f"{BOLD}{GREEN}🎨 Generating t-SNE specialization plot...{RESET}")
        
        # 准备数据 - 确保数据类型正确  
        embeddings = collected_data['combined_tsne_embeddings']
        labels = collected_data['combined_tsne_labels']
        domains = collected_data['combined_tsne_domains']
        
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings)
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)
        if isinstance(domains, np.ndarray):
            domains = torch.from_numpy(domains)
        
        # 确保是numpy格式用于后续处理
        embeddings = embeddings.numpy()
        labels = labels.numpy()
        domains = domains.numpy()
        
        # 采样数据以避免过大的t-SNE计算
        max_samples = getattr(args, 'tsne_sample_size', 1000)
        if len(embeddings) > max_samples:
            indices = np.random.choice(len(embeddings), max_samples, replace=False)
            embeddings = embeddings[indices]
            labels = labels[indices]
            domains = domains[indices]
            print(f"  ⚡ Sampled {max_samples} points for t-SNE visualization")
        
        # 使用增强可视化 - 需要转换为torch tensor格式
        embeddings_tensor = torch.from_numpy(embeddings) if isinstance(embeddings, np.ndarray) else embeddings
        labels_tensor = torch.from_numpy(labels) if isinstance(labels, np.ndarray) else labels
        domains_tensor = torch.from_numpy(domains) if isinstance(domains, np.ndarray) else domains
        
        fig, saved_files = plot_tsne_specialization_journal(
            embeddings_tensor, labels_tensor, domains_tensor, domain_map, 
            epoch="inference", config=viz_config,
            save_plots=args.save_publication_figs,
            tsne_params={'perplexity': min(30, len(embeddings)//4)}
        )
        
        if saved_files:
            print(f"  ✓ t-SNE specialization plot saved: {len(saved_files)} files")
            for file_path in saved_files:
                print(f"    - {file_path}")
        
        return fig, saved_files
        
    except Exception as e:
        print(f"⚠ t-SNE visualization failed: {e}")
        return None


def generate_fourier_visualization(collected_data, domain_map, args, viz_config):
    """生成Fourier频率分析可视化（可选）"""
    if not ENHANCED_VIZ_AVAILABLE or viz_config is None:
        print(f"⚠️ Enhanced visualization not available - Fourier visualization skipped")
        return None
    
    if not collected_data['fourier_data']:
        print(f"⚠️ No Fourier data available")
        return None
    
    try:
        print(f"{BOLD}{GREEN}🎨 Generating Fourier analysis visualization...{RESET}")
        
        # 处理Fourier数据（取第一个batch的数据作为示例）
        fourier_detailed = collected_data['fourier_data'][0]
        saved_files_list = []
        
        for layer_idx, fourier_attn_data in enumerate(fourier_detailed):
            if fourier_attn_data is not None and len(fourier_attn_data) > 1:
                try:
                    multi_fig, multi_saved_files = plot_multi_domain_fourier_comparison_journal(
                        fourier_attn_data, domain_map, layer_idx, "inference",
                        viz_config, save_plots=args.save_publication_figs, adaptive_style='lines'
                    )
                    
                    plt.close(multi_fig)
                    
                    if multi_saved_files:
                        saved_files_list.extend(multi_saved_files)
                        print(f"  ✓ Fourier analysis (Layer {layer_idx}): {len(multi_saved_files)} files")
                
                except Exception as e:
                    print(f"  ⚠ Fourier visualization failed for layer {layer_idx}: {e}")
        
        if saved_files_list:
            print(f"  ✓ Total Fourier visualizations saved: {len(saved_files_list)} files")
            
        return saved_files_list
        
    except Exception as e:
        print(f"⚠ Fourier visualization failed: {e}")
        return None


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Inference-time visualization for expert routing and t-SNE")
    
    # 输入参数
    parser.add_argument('--experiment_dir', required=True, type=str,
                       help='Path to training experiment directory')
    parser.add_argument('--state_dict_path', required=True, type=str,
                       help='Path to trained model checkpoint')
    parser.add_argument('--dataset_type', default='test', choices=['valid', 'test'],
                       help='Dataset type for inference visualization')
    
    # 数据参数
    parser.add_argument('--use_datasets', nargs='+', default=None,
                       help='Datasets to use; if None, will load from experiment args')
    parser.add_argument('--batch_size', default=256, type=int,
                       help='Batch size for inference')
    parser.add_argument('--max_batches', default=None, type=int,
                       help='Maximum number of batches to process (for quick testing)')
    parser.add_argument('--num_workers', default=4, type=int,
                       help='Number of workers for data loading')
    
    # 可视化参数
    parser.add_argument('--journal_style', default='custom', type=str,
                       choices=['nature', 'science', 'cell', 'high_quality', 'custom'],
                       help='Journal style for visualizations')
    parser.add_argument('--viz_dpi', default=600, type=int,
                       help='DPI for visualization outputs')
    parser.add_argument('--viz_format', default='pdf', type=str,
                       choices=['pdf', 'png', 'svg', 'eps'],
                       help='Primary format for visualization exports')
    parser.add_argument('--save_publication_figs', default=True, type=str2bool,
                       help='Save publication-quality figures')
    parser.add_argument('--tsne_sample_size', default=1000, type=int,
                       help='Maximum number of samples for t-SNE visualization')
    
    # 输出参数
    parser.add_argument('--output_dir', default='exp/inference_visualization', type=str,
                       help='Output directory for visualizations')
    parser.add_argument('--include_fourier', default=False, type=str2bool,
                       help='Include Fourier frequency analysis visualization')
    
    # 设备参数
    parser.add_argument('--device', default='cuda', type=str,
                       help='Device for inference (cuda/cpu)')
    
    # 可重复性参数
    parser.add_argument('--seed', default=42, type=int,
                       help='Random seed for reproducible results')
    
    return parser.parse_args()


def main():
    """主函数"""
    print(f"{BOLD}{BLUE}🚀 Starting inference-time visualization{RESET}")
    
    # 解析参数
    args = parse_args()
    
    # 设置随机种子以确保可重复性
    set_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"📁 Output directory: {args.output_dir}")
    
    # 加载训练参数
    print(f"{BOLD}📖 Loading training configuration...{RESET}")
    training_args = load_experiment_args(args.experiment_dir)
    
    # 确定使用的数据集
    if args.use_datasets is None:
        # 从训练参数中解析数据集
        datasets_str = training_args.get('use_datasets', "['beauty_5_5', 'games_5_5', 'ml-1m_5_5']")
        try:
            import ast
            args.use_datasets = ast.literal_eval(datasets_str)
        except:
            args.use_datasets = ['beauty_5_5', 'games_5_5', 'ml-1m_5_5']
    
    print(f"📊 Using datasets: {args.use_datasets}")
    
    # 加载数据
    print(f"{BOLD}📂 Loading multi-domain data...{RESET}")
    dataset = partition_multi_domain(args.use_datasets)
    [user_train, user_valid, user_test, user_to_domain, usernum, itemnum, domain_to_item_range] = dataset
    
    domain_map = {i: name for i, name in enumerate(args.use_datasets)}
    print(f"✓ Data loaded: users={usernum}, items={itemnum}, domains={len(args.use_datasets)}")
    
    # 确定设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print(f"⚠ CUDA not available, falling back to CPU")
        args.device = 'cpu'
    device = torch.device(args.device)
    print(f"🔧 Using device: {device}")
    
    # 创建模型参数
    model_args = create_model_args(training_args, args.use_datasets, device)
    model_args.domain_to_item_range = domain_to_item_range
    
    # 加载模型
    print(f"{BOLD}🤖 Loading trained model...{RESET}")
    model = HAGMRec(usernum, itemnum, model_args).to(device)
    
    # 加载权重
    try:
        state_dict = torch.load(args.state_dict_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"✓ Model loaded from: {args.state_dict_path}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # 设置增强可视化
    print(f"{BOLD}🎨 Setting up enhanced visualization...{RESET}")
    viz_config = setup_enhanced_visualization(args)
    if viz_config is None:
        print(f"❌ Enhanced visualization setup failed")
        return
    
    # 准备数据加载器 - 为了可视化，我们需要使用能提供rating_seq的训练格式数据
    print(f"{BOLD}📦 Preparing data loader for visualization...{RESET}")
    print(f"  Note: Using training data format to ensure compatibility with visualization")
    
    # 根据数据集类型选择合适的用户集合进行可视化
    # 这里我们使用训练数据的格式，但可以限制用户范围来模拟推理场景
    if args.dataset_type == 'valid':
        # 使用验证集中的用户，但使用它们的训练数据进行可视化
        viz_users = {u: seq for u, seq in user_train.items() if u in user_valid and len(seq) > 1}
        print(f"  📊 Visualization on validation users: {len(viz_users)} users")
    else:  # test
        # 使用测试集中的用户，但使用它们的训练数据进行可视化
        viz_users = {u: seq for u, seq in user_train.items() if u in user_test and len(seq) > 1}
        print(f"  📊 Visualization on test users: {len(viz_users)} users")
    
    if len(viz_users) == 0:
        print(f"❌ No users found for visualization in {args.dataset_type} set!")
        return
    
    # 创建可视化专用数据集
    viz_dataset = MoerecDataset(
        viz_users, user_to_domain, usernum, itemnum,
        int(training_args.get('maxlen', 100)), model_args, domain_to_item_range
    )
    
    viz_sampler = StratifiedSampler(viz_dataset)
    viz_collator = MoerecCollator(maxlen=int(training_args.get('maxlen', 100)))
    viz_loader = torch.utils.data.DataLoader(
        viz_dataset,
        batch_size=args.batch_size,
        sampler=viz_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=viz_collator,
        pin_memory=True
    )
    
    print(f"✓ Data loader ready: {len(viz_loader)} batches")
    
    # 收集可视化数据
    start_time = time.time()
    collected_data = collect_visualization_data(
        model, viz_loader, model_args, domain_map, args.max_batches
    )
    collection_time = time.time() - start_time
    print(f"⏱️ Data collection completed in {collection_time:.2f} seconds")
    
    # 生成可视化
    print(f"\n{BOLD}{GREEN}🖼️ Generating visualizations...{RESET}")
    
    generated_files = []
    
    # 1. 不再导出单独的专家路由热力图（综合图已包含）
    
    # 2. t-SNE专家专业化图 或 1x3 综合图
    try:
        # 构造 1x3 综合图所需数据
        if 'avg_domain_expert_load' in collected_data and 'combined_tsne_embeddings' in collected_data:
            avg_load = collected_data['avg_domain_expert_load']
            if isinstance(avg_load, np.ndarray):
                avg_load = torch.from_numpy(avg_load)
            # 准备标签
            domain_labels = []
            for i in range(len(avg_load)):
                raw_dataset_name = domain_map.get(i, f"Unknown Domain {i}")
                try:
                    from visualization.enhanced_plots import _normalize_domain_name
                    domain_labels.append(_normalize_domain_name(raw_dataset_name))
                except ImportError:
                    domain_labels.append(raw_dataset_name)
            expert_labels = [f"Expert {i}" for i in range(len(avg_load[0]))]

            fig, files = plot_inference_combined_overview(
                routing_weights=avg_load,
                domain_labels_list=domain_labels,
                expert_labels_list=expert_labels,
                embeddings=collected_data['combined_tsne_embeddings'],
                expert_assignments=collected_data['combined_tsne_labels'],
                domain_assignments=collected_data['combined_tsne_domains'],
                domain_map=domain_map,
                config=viz_config,
                save_plots=args.save_publication_figs,
                max_tsne_samples=args.tsne_sample_size,
            )
            if files:
                generated_files.extend(files)
            plt.close(fig)
        else:
            tsne_result = generate_tsne_specialization_visualization(
                collected_data, domain_map, args, viz_config
            )
            if tsne_result:
                fig, files = tsne_result
                generated_files.extend(files)
                plt.close(fig)
    except Exception as e:
        print(f"⚠ Combined overview generation failed: {e}")
    
    # 3. Fourier频率分析（可选）
    if args.include_fourier:
        fourier_files = generate_fourier_visualization(
            collected_data, domain_map, args, viz_config
        )
        if fourier_files:
            generated_files.extend(fourier_files)
    
    # 总结
    total_time = time.time() - start_time + collection_time
    print(f"\n{BOLD}{GREEN}🎉 Inference visualization completed!{RESET}")
    print(f"⏱️ Total time: {total_time:.2f} seconds")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"📊 Generated {len(generated_files)} visualization files:")
    for file_path in generated_files:
        print(f"  - {file_path}")
    
    print(f"\n💡 These visualizations show the same expert routing and specialization")
    print(f"   patterns as during training, but derived from inference on {args.dataset_type} data.")


if __name__ == '__main__':
    main()