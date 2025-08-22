
# -*- coding: utf-8 -*-
"""
cli.py (merged)
--------------
Command-line entrypoint that wires project-specific components (data/model)
with the generic plotting utilities from viz_lib.
"""
from __future__ import annotations

import os
import re
import ast
import argparse
from pathlib import Path

import numpy as np
import torch

# Ensure project root on sys.path (so that `keys.*` can be imported when running directly)
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
import sys
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Project imports
from keys.utils import partition_multi_domain
from keys.model import HAGMRec
from keys.temporal_rating_modules import OptimizedFourierRatingEncoder  # noqa: F401

# Local library
from viz_lib import (
    select_user_with_min_len,
    build_rating_series,
    compute_scalar_fft_decomposition,
    compute_embedding_sequences,
    energy_ratio,
    auto_cutoff_80,
    find_top_peaks,
    r2_trend_plus_mean_consistent,
    resolve_cutoff_idx,
    plot_time_and_spectrum,
    plot_decomposition_overlay,
    plot_tsne_embeddings,
    save_metrics_json,
    get_user_domain_id,
)


def str2bool(v: str) -> bool:
    return str(v).lower() in ("true", "1", "yes", "y")


def _to_bool(s):
    if isinstance(s, bool):
        return s
    return str(s).lower() in ('true', '1', 'yes', 'y')


def _to_list(s):
    try:
        val = ast.literal_eval(s)
        if isinstance(val, list):
            return [str(x) for x in val]
    except Exception:
        pass
    return [x for x in re.split(r'[\s,]+', s) if x]


def generate_single_domain_analysis(rating_series, model, domain_id, args):
    """生成单域分析"""
    # 1) Time vs Frequency
    freqs, amplitude, low_s, high_s, cutoff_ratio = compute_scalar_fft_decomposition(
        rating_series, model, domain_id
    )
    L = len(rating_series)
    half = L // 2
    amp_half = amplitude[:half]
    kc_80 = auto_cutoff_80(amp_half)

    # choose cutoff per policy
    kc_model = min(int(cutoff_ratio * L), half - 1)
    chosen_k = resolve_cutoff_idx(L, amp_half, cutoff_ratio, policy=args.cutoff_policy, fixed_k=args.fixed_k)
    low_frac, high_frac = energy_ratio(amp_half, chosen_k)

    domain_suffix = f"_domain_{domain_id}" if domain_id is not None else ""
    
    plot_time_and_spectrum(
        rating_series, freqs, amplitude,
        cutoff_idx=chosen_k,
        kc_80=kc_80,
        low_frac=low_frac, high_frac=high_frac,
        out_path=os.path.join(args.output_dir, f'fig1_time_vs_frequency{domain_suffix}.png'),
        journal=args.journal_style,
        label_policy=args.cutoff_policy,
        peaks=find_top_peaks(amp_half, n=3, min_distance=3)
    )

    # 2) Decomposition overlay
    r2_reg = r2_trend_plus_mean_consistent(rating_series, model, domain_id)
    r2_energy = low_frac  # same口径
    plot_decomposition_overlay(
        rating_series, low_s, high_s,
        r2_energy=r2_energy, r2_reg=r2_reg,
        out_path=os.path.join(args.output_dir, f'fig2_decomposition_overlay{domain_suffix}.png'),
        journal=args.journal_style
    )

    # 3) Embedding t-SNE
    pre_emb, long_emb, short_emb = compute_embedding_sequences(rating_series, model, domain_id)
    sil, acc = plot_tsne_embeddings(
        pre_emb, long_emb, short_emb,
        out_path=os.path.join(args.output_dir, f'fig3_tsne_embeddings{domain_suffix}.png'),
        journal=args.journal_style,
        perplexity=args.tsne_perplexity,
        seed=args.tsne_seed
    )

    # Save metrics
    if args.save_metrics:
        metrics = {
            'analysis_domain': domain_id,
            'time_frequency': {
                'L': int(L),
                'chosen_policy': args.cutoff_policy,
                'chosen_cutoff_idx': int(chosen_k),
                'model_cutoff_idx': int(kc_model),
                'auto_cutoff_80_idx': int(kc_80),
                'low_energy_fraction': float(low_frac),
                'high_energy_fraction': float(high_frac),
                'model_cutoff_ratio': float(cutoff_ratio),
            },
            'decomposition': {
                'R2_energy_lowfrac': float(r2_energy),
                'R2_reg_trend_plus_mean': float(r2_reg),
            },
            'embedding': {
                'silhouette_cosine': float(sil),
                'logreg_accuracy': float(acc),
                'tsne_seed': int(args.tsne_seed),
                'tsne_perplexity': int(args.tsne_perplexity),
            }
        }
        save_metrics_json(metrics, args.output_dir, f'analysis_metrics{domain_suffix}.json')

    print(f"Saved single-domain analysis (domain {domain_id}) to: {args.output_dir}")


def generate_domain_comparison_analysis(rating_series, model, user_domain, args, datasets, user_to_domain):
    """生成多域对比分析"""
    print(f"Starting domain comparison analysis for user domain {user_domain}...")
    
    # 获取可用的领域数量
    num_domains = len(datasets)
    domain_metrics = {}
    
    # 对每个域进行分析
    for domain_id in range(num_domains):
        print(f"Analyzing with domain {domain_id} encoder...")
        
        try:
            # 使用特定域的编码器分析
            freqs, amplitude, low_s, high_s, cutoff_ratio = compute_scalar_fft_decomposition(
                rating_series, model, domain_id
            )
            L = len(rating_series)
            half = L // 2
            amp_half = amplitude[:half]
            kc_80 = auto_cutoff_80(amp_half)
            kc_model = min(int(cutoff_ratio * L), half - 1)
            chosen_k = resolve_cutoff_idx(L, amp_half, cutoff_ratio, policy=args.cutoff_policy, fixed_k=args.fixed_k)
            low_frac, high_frac = energy_ratio(amp_half, chosen_k)
            
            # R²指标
            r2_reg = r2_trend_plus_mean_consistent(rating_series, model, domain_id)
            
            # 嵌入分析
            pre_emb, long_emb, short_emb = compute_embedding_sequences(rating_series, model, domain_id)
            
            # 计算silhouette score
            try:
                from sklearn.metrics import silhouette_score
                X = np.vstack([pre_emb, long_emb, short_emb])
                labels = np.array(['Pre'] * pre_emb.shape[0] + 
                                 ['Long'] * long_emb.shape[0] + 
                                 ['Short'] * short_emb.shape[0])
                sil = float(silhouette_score(X, labels, metric='cosine'))
            except Exception:
                sil = float('nan')
            
            domain_metrics[domain_id] = {
                'cutoff_ratio': float(cutoff_ratio),
                'low_energy_fraction': float(low_frac),
                'high_energy_fraction': float(high_frac),
                'R2_reg': float(r2_reg),
                'silhouette': float(sil),
                'kc_model': int(kc_model),
                'kc_80': int(kc_80),
                'chosen_k': int(chosen_k),
                'freqs': freqs,
                'amplitude': amplitude,
                'low_s': low_s,
                'high_s': high_s,
            }
            
            # 生成每个域的单独图表
            generate_single_domain_analysis(rating_series, model, domain_id, args)
            
        except Exception as e:
            print(f"Error analyzing domain {domain_id}: {e}")
            continue
    
    # 生成对比图表
    if len(domain_metrics) > 1:
        generate_comparison_plots(domain_metrics, rating_series, args, user_domain)
    
    # 保存对比指标（只保存JSON可序列化的数据）
    if args.save_metrics:
        # 创建只包含可序列化数据的域指标
        serializable_domain_metrics = {}
        for d, m in domain_metrics.items():
            serializable_domain_metrics[f'domain_{d}'] = {
                'cutoff_ratio': float(m['cutoff_ratio']),
                'low_energy_fraction': float(m['low_energy_fraction']),
                'high_energy_fraction': float(m['high_energy_fraction']),
                'R2_reg': float(m['R2_reg']),
                'silhouette': float(m['silhouette']),
                'kc_model': int(m['kc_model']),
                'kc_80': int(m['kc_80']),
                'chosen_k': int(m['chosen_k']),
                # 排除NumPy数组：freqs, amplitude, low_s, high_s
            }
        
        comparison_metrics = {
            'user_domain': int(user_domain),
            'analyzed_domains': list(domain_metrics.keys()),
            'comparison_summary': {
                f'domain_{d}': {
                    'low_energy_fraction': float(m['low_energy_fraction']),
                    'R2_reg': float(m['R2_reg']),
                    'silhouette': float(m['silhouette']),
                    'cutoff_ratio': float(m['cutoff_ratio']),
                } for d, m in domain_metrics.items()
            },
            'domain_metrics': serializable_domain_metrics
        }
        save_metrics_json(comparison_metrics, args.output_dir, 'domain_comparison_metrics.json')
    
    print(f"Saved domain comparison analysis to: {args.output_dir}")


def generate_comparison_plots(domain_metrics, rating_series, args, user_domain):
    """生成域对比图表"""
    import matplotlib.pyplot as plt
    from viz_lib import apply_journal_style
    
    apply_journal_style(args.journal_style)
    
    domains = sorted(domain_metrics.keys())
    # matplotlib会自动使用apply_journal_style()设置的prop_cycle颜色
    
    # 1. 低频能量分数对比
    fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=300)
    low_fracs = [domain_metrics[d]['low_energy_fraction'] for d in domains]
    bars = ax.bar(range(len(domains)), low_fracs)
    
    # 高亮用户实际域
    if user_domain in domains:
        user_idx = domains.index(user_domain)
        bars[user_idx].set_edgecolor('red')
        bars[user_idx].set_linewidth(2)
    
    ax.set_xlabel('Domain')
    ax.set_ylabel('Low-Frequency Energy Fraction')
    ax.set_title(f'Domain Comparison: Low-Freq Energy (User actual domain: {user_domain})')
    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels([f'Domain {d}' for d in domains])
    ax.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, v in enumerate(low_fracs):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'comparison_low_energy_fraction.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # 2. R² 指标对比
    fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=300)
    r2_vals = [domain_metrics[d]['R2_reg'] for d in domains]
    bars = ax.bar(range(len(domains)), r2_vals)
    
    if user_domain in domains:
        user_idx = domains.index(user_domain)
        bars[user_idx].set_edgecolor('red')
        bars[user_idx].set_linewidth(2)
    
    ax.set_xlabel('Domain')
    ax.set_ylabel('R² (Trend + Mean)')
    ax.set_title(f'Domain Comparison: Decomposition R² (User actual domain: {user_domain})')
    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels([f'Domain {d}' for d in domains])
    ax.grid(True, alpha=0.3)
    
    for i, v in enumerate(r2_vals):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'comparison_r2_decomposition.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    print("Generated domain comparison plots")


def main():
    parser = argparse.ArgumentParser(description="User behavior frequency analysis and visualization (2-file version)")

    # Data & model
    parser.add_argument('--datasets', nargs='+', default=None, help='Datasets under data/*.txt; default from experiment args.txt')
    parser.add_argument('--user_id', type=int, default=None, help='Specific global user ID to visualize')
    parser.add_argument('--maxlen', type=int, default=100, help='Max sequence length to consider')
    parser.add_argument('--hidden_units', type=int, default=64, help='Hidden units for rating embedding')
    parser.add_argument('--experiment_dir', type=str, default=None, help='Path to training experiment dir containing args.txt & checkpoints')
    parser.add_argument('--state_dict_path', type=str, default=None, help='Path to trained model checkpoint (.pth)')

    # Viz & metrics
    parser.add_argument('--journal_style', type=str, default='custom', choices=['nature', 'science', 'cell', 'high_quality', 'custom'])
    parser.add_argument('--output_dir', type=str, default='exp/user_frequency_figs')
    parser.add_argument('--save_metrics', type=str2bool, default=True)

    # Domain analysis (NEW)
    parser.add_argument('--domain_id', type=int, default=None, help='Specific domain ID to use for analysis (overrides auto-detection)')
    parser.add_argument('--compare_domains', type=str2bool, default=False, help='Generate comparison plots across all domains')
    parser.add_argument('--auto_domain', type=str2bool, default=True, help='Auto-detect user domain from user_to_domain mapping')

    # Cutoff policy
    parser.add_argument('--cutoff_policy', type=str, default='model', choices=['model', 'energy80', 'fixed_k'],
                        help='Region shading policy in spectrum plot')
    parser.add_argument('--fixed_k', type=int, default=None, help='Cutoff index when --cutoff_policy=fixed_k')

    # t-SNE reproducibility
    parser.add_argument('--tsne_perplexity', type=int, default=20)
    parser.add_argument('--tsne_seed', type=int, default=42)

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # --- Load experiment args if provided ---
    exp_args = {}
    if args.experiment_dir is not None:
        exp_dir = Path(args.experiment_dir)
        if not exp_dir.exists():
            cand = _PROJECT_ROOT / args.experiment_dir.lstrip(os.sep)
            if cand.exists():
                exp_dir = cand
                args.experiment_dir = str(exp_dir)
        args_txt = exp_dir / 'args.txt'
        if args_txt.exists():
            with open(args_txt, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or ',' not in line:
                        continue
                    k, v = line.split(',', 1)
                    exp_args[k.strip()] = v.strip()

    # Resolve datasets: CLI > exp_args > default
    datasets = args.datasets
    if datasets is None:
        if 'use_datasets' in exp_args:
            datasets = _to_list(exp_args['use_datasets'])
        else:
            datasets = ['beauty_5_5']

    # Load multi-domain & pick user
    dataset = partition_multi_domain(datasets)
    [user_train, user_valid, user_test, user_to_domain, usernum, itemnum, domain_to_item_range] = dataset
    if not user_train:
        raise RuntimeError('No training data found.')

    if args.user_id is None or args.user_id not in user_train:
        uid = select_user_with_min_len(user_train, min_len=20)
    else:
        uid = args.user_id
    if uid is None:
        raise RuntimeError('No valid user found for visualization.')

    user_seq = user_train[uid]
    rating_series = build_rating_series(user_seq, maxlen=args.maxlen)
    if rating_series.size == 0:
        raise RuntimeError('Selected user has empty rating series.')

    # --- Determine domain for analysis ---
    user_domain = get_user_domain_id(uid, user_to_domain, default_domain=0)
    
    if args.domain_id is not None:
        # 用户指定了域ID
        analysis_domain = args.domain_id
        print(f"Using user-specified domain_id: {analysis_domain} (user {uid} actual domain: {user_domain})")
    elif args.auto_domain:
        # 自动检测用户域
        analysis_domain = user_domain
        print(f"Auto-detected domain for user {uid}: {analysis_domain}")
    else:
        # 不使用域信息（向下兼容模式）
        analysis_domain = None
        print(f"Domain-agnostic analysis for user {uid}")

    print(f"Selected user: {uid} (domain: {user_domain}, analysis domain: {analysis_domain})")
    print(f"Rating series length: {len(rating_series)}")
    print(f"Rating range: {rating_series.min():.2f} - {rating_series.max():.2f}")
    print(f"Compare domains mode: {args.compare_domains}")

    # --- Build model with config & load weights (CPU inference) ---
    class _Cfg: pass
    cfg = _Cfg()
    setattr(cfg, 'device', 'cpu')
    setattr(cfg, 'maxlen', int(exp_args.get('maxlen', args.maxlen)))
    setattr(cfg, 'hidden_units', int(exp_args.get('hidden_units', args.hidden_units)))
    setattr(cfg, 'num_blocks', int(exp_args.get('num_blocks', 2)))
    setattr(cfg, 'num_heads', int(exp_args.get('num_heads', 2)))
    setattr(cfg, 'dropout_rate', float(exp_args.get('dropout_rate', 0.5)))
    setattr(cfg, 'use_rating_emb', _to_bool(exp_args.get('use_rating_emb', True)))
    setattr(cfg, 'rating_strategy', exp_args.get('rating_strategy', 'temporal_fourier'))
    setattr(cfg, 'rating_pos_emb', _to_bool(exp_args.get('rating_pos_emb', False)))
    setattr(cfg, 'use_moe', _to_bool(exp_args.get('use_moe', True)))
    setattr(cfg, 'moe_num_experts', int(exp_args.get('moe_num_experts', 4)))
    setattr(cfg, 'moe_k', int(exp_args.get('moe_k', 2)))
    setattr(cfg, 'moe_routing_strategy', exp_args.get('moe_routing_strategy', 'shared_base'))
    setattr(cfg, 'moe_load_balancing', _to_bool(exp_args.get('moe_load_balancing', True)))
    setattr(cfg, 'moe_balance_loss_weight', float(exp_args.get('moe_balance_loss_weight', 0.01)))
    setattr(cfg, 'use_domain_info', _to_bool(exp_args.get('use_domain_info', True)))
    setattr(cfg, 'use_gated_fusion', _to_bool(exp_args.get('use_gated_fusion', True)))
    setattr(cfg, 'use_specialization_loss', _to_bool(exp_args.get('use_specialization_loss', True)))
    setattr(cfg, 'specialization_weight', float(exp_args.get('specialization_weight', 0.01)))
    setattr(cfg, 'use_contrastive_loss', _to_bool(exp_args.get('use_contrastive_loss', True)))
    setattr(cfg, 'contrastive_weight', float(exp_args.get('contrastive_weight', 0.01)))
    setattr(cfg, 'num_workers', int(exp_args.get('num_workers', 0)))
    setattr(cfg, 'num_domains', len(datasets))

    model = HAGMRec(usernum, itemnum, cfg).to('cpu')

    # checkpoint
    ckpt_path = args.state_dict_path
    if ckpt_path is None and args.experiment_dir is not None:
        p = Path(args.experiment_dir)
        all_pth = sorted(p.glob('SASRec.epoch=*.pth'))
        if all_pth:
            def _epoch_num(pp):
                m = re.search(r'epoch=(\d+)', pp.name)
                return int(m.group(1)) if m else -1
            all_pth.sort(key=_epoch_num)
            ckpt_path = str(all_pth[-1])
    if ckpt_path is None or not Path(ckpt_path).exists():
        raise RuntimeError('Trained checkpoint not found. Provide --state_dict_path or --experiment_dir with checkpoints.')

    state = torch.load(ckpt_path, map_location=torch.device('cpu'))
    model.load_state_dict(state, strict=False)
    model.eval()

    if not hasattr(model, 'enhanced_rating_module'):
        raise RuntimeError('Model does not have enhanced_rating_module; ensure use_rating_emb=True and rating_strategy=temporal_fourier.')

    # --- Generate analysis ---
    if args.compare_domains:
        # 多域对比模式
        generate_domain_comparison_analysis(
            rating_series, model, user_domain, args, 
            datasets, user_to_domain
        )
    else:
        # 单域分析模式 
        generate_single_domain_analysis(
            rating_series, model, analysis_domain, args
        )


if __name__ == '__main__':
    main()
