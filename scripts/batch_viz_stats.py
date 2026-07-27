
# -*- coding: utf-8 -*-
"""
batch_viz_stats.py
------------------
Batch visualization & statistics for user-behavior frequency analysis.

This script:
1) Loads data/model once (CPU).
2) Samples N users (per overall dataset list).
3) For each user, computes FFT metrics (LowFrac/HighFrac, auto 80% cutoff, peaks),
   decomposition R^2 (trend+mean), and embedding separability (silhouette/logreg).
4) Optionally saves per-user figures (the three plots) using `viz_lib` utilities.
5) Aggregates results into CSV + JSON, and renders several *group-level* plots:
   - Distribution of LowFrac (hist/box)
   - Correlation scatter: LowFrac vs silhouette/logreg
   - Policy comparison boxplot: model vs energy80 vs fixed_k (if configured)

USAGE EXAMPLE
-------------
python batch_viz_stats.py \
  --experiment_dir /path/to/exp \
  --state_dict_path /path/to/SASRec.epoch=XX.pth \
  --datasets beauty_5_5 \
  --output_dir exp/batch_stats \
  --n_users 200 \
  --sample_strategy longest \
  --cutoff_policies model,energy80 \
  --tsne_seed 42 --tsne_perplexity 20 \
  --save_user_figs false

Outputs:
- metrics_user_level.csv
- summary_stats.json
- group_*.png (group plots)
- Per-user figures (optional): user_<uid>_fig[1-3].png
"""
from __future__ import annotations

import os
import re
import ast
import json
import math
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Optional scipy for advanced analysis
try:
    from scipy.stats import gaussian_kde
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Optional pandas for CSV/summary
try:
    import pandas as pd
except Exception:
    pd = None

# Ensure project root on sys.path (so that `keys.*` can be imported when running directly)
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
import sys
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Project imports
from keys.utils import partition_multi_domain
from keys.model import SynRec

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
    apply_journal_style,
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


def sample_users(user_train: Dict[int, list], n_users: int, strategy: str, seed: int) -> List[int]:
    rng = random.Random(seed)
    if not user_train:
        return []
    users = list(user_train.keys())
    if strategy == "random":
        rng.shuffle(users)
        return users[:n_users]
    # longest
    users_sorted = sorted(users, key=lambda u: len(user_train[u]), reverse=True)
    return users_sorted[:n_users]


def sample_users_balanced(user_train: Dict[int, list], user_to_domain: Dict[int, int], 
                          n_users: int, strategy: str, seed: int, min_seq_len: int = 20) -> List[int]:
    """
    分层平衡采样：确保每个领域都有充分代表，且满足最小序列长度要求
    
    Args:
        user_train: 用户训练数据
        user_to_domain: 用户到领域的映射
        n_users: 总采样用户数
        strategy: 采样策略 ('longest', 'random')
        seed: 随机种子
        min_seq_len: 最小序列长度要求
    
    Returns:
        采样的用户ID列表
    """
    rng = random.Random(seed)
    
    # 按领域分组用户，同时过滤序列长度
    users_by_domain = {}
    for user_id, seq in user_train.items():
        # 只包含满足最小序列长度要求的用户
        if len(seq) >= min_seq_len:
            domain_id = get_user_domain_id(user_id, user_to_domain, 0)
            if domain_id not in users_by_domain:
                users_by_domain[domain_id] = []
            users_by_domain[domain_id].append(user_id)
    
    if not users_by_domain:
        return []
    
    print(f"📊 合格用户分布 (序列长度≥{min_seq_len}):")
    total_available = 0
    for domain_id, users in users_by_domain.items():
        print(f"   - 领域 {domain_id}: {len(users)} 用户")
        total_available += len(users)
    
    # 检查总的合格用户数
    if total_available < n_users:
        print(f"⚠️  警告: 合格用户总数 ({total_available}) 小于请求数量 ({n_users})")
        print(f"   建议降低 --min_seq_len (当前:{min_seq_len}) 或减少 --n_users")
        n_users = total_available  # 调整为实际可用数量
    
    # 平衡分层采样：所有策略都使用平衡分配
    # strategy参数控制的是每个领域内的采样方法
    num_domains = len(users_by_domain)
    target_per_domain = n_users // num_domains
    remainder = n_users % num_domains
    
    samples_per_domain = {}
    for i, domain_id in enumerate(sorted(users_by_domain.keys())):
        samples_per_domain[domain_id] = target_per_domain
        if i < remainder:  # 余数分配给前几个领域
            samples_per_domain[domain_id] += 1
    
    print(f"🎯 平衡分层采样目标 (领域内策略: {strategy}):")
    for domain_id, target in samples_per_domain.items():
        available = len(users_by_domain[domain_id])
        actual = min(target, available)
        print(f"   - 领域 {domain_id}: 目标 {target}, 实际 {actual}")
    
    # 执行分层采样
    selected_users = []
    for domain_id, target_count in samples_per_domain.items():
        available_users = users_by_domain[domain_id]
        actual_count = min(target_count, len(available_users))
        
        if actual_count == 0:
            continue
        
        # 在该领域内采样（使用指定的strategy）
        if strategy == "longest":
            # 在领域内按序列长度降序采样
            domain_users_sorted = sorted(available_users, 
                                       key=lambda u: len(user_train[u]), reverse=True)
            domain_selected = domain_users_sorted[:actual_count]
        elif strategy == "random":
            # 在领域内随机采样
            domain_selected = rng.sample(available_users, actual_count)
        else:  # proportional或其他策略的后备处理
            # 默认使用random策略
            domain_selected = rng.sample(available_users, actual_count)
        
        selected_users.extend(domain_selected)
    
    print(f"✅ 最终采样结果: 总计 {len(selected_users)} 用户")
    
    # 验证最终分布
    final_domain_counts = {}
    for user_id in selected_users:
        domain_id = get_user_domain_id(user_id, user_to_domain, 0)
        final_domain_counts[domain_id] = final_domain_counts.get(domain_id, 0) + 1
    
    for domain_id in sorted(final_domain_counts.keys()):
        count = final_domain_counts[domain_id]
        percentage = count / len(selected_users) * 100
        print(f"   - 领域 {domain_id}: {count} 用户 ({percentage:.1f}%)")
    
    return selected_users


def group_boxplot(values_by_group: Dict[str, List[float]], title: str, ylabel: str, out_path: str):
    """Simple boxplot for multiple groups using Matplotlib only."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 3.5), dpi=600)
    groups = list(values_by_group.keys())
    data = [values_by_group[g] for g in groups if len(values_by_group[g]) > 0]
    groups = [g for g in groups if len(values_by_group[g]) > 0]
    if not data:
        plt.close(fig); return
    ax.boxplot(data, showmeans=True, meanline=True)
    ax.set_xticks(range(1, len(groups)+1))
    ax.set_xticklabels(groups, rotation=0)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25, linestyle='--')
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def hist_plot(vals: List[float], title: str, xlabel: str, out_path: str, bins: int = 20):
    fig, ax = plt.subplots(1, 1, figsize=(6, 3.0), dpi=600)
    if len(vals) == 0:
        plt.close(fig); return
    ax.hist(vals, bins=bins, edgecolor='black', alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.25, linestyle='--')
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def scatter_plot(x: List[float], y: List[float], title: str, xlabel: str, ylabel: str, out_path: str):
    fig, ax = plt.subplots(1, 1, figsize=(6, 3.0), dpi=600)
    if len(x) == 0:
        plt.close(fig); return
    ax.scatter(x, y, s=12, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25, linestyle='--')
    # simple correlation (robust to zero-variance and NaNs)
    try:
        if len(x) >= 3:
            xm = np.array(x, dtype=float); ym = np.array(y, dtype=float)
            mask = np.isfinite(xm) & np.isfinite(ym)
            if mask.sum() >= 3:
                xs = xm[mask]; ys = ym[mask]
                if np.std(xs) > 0 and np.std(ys) > 0:
                    with np.errstate(invalid='ignore', divide='ignore'):
                        r = np.corrcoef(xs, ys)[0, 1]
                    if np.isfinite(r):
                        ax.text(0.02, 0.98, f'r={r:.2f}', transform=ax.transAxes, ha='left', va='top')
    except Exception:
        pass
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Batch visualization & statistics")
    # Data/model
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--experiment_dir', type=str, default=None)
    parser.add_argument('--state_dict_path', type=str, default=None)
    parser.add_argument('--maxlen', type=int, default=100)
    parser.add_argument('--min_seq_len', type=int, default=20)

    # Sampling
    parser.add_argument('--n_users', type=int, default=200)
    parser.add_argument('--sample_strategy', type=str, default='longest', 
                        choices=['longest', 'random'],
                        help='Within-domain sampling strategy: longest (high-quality users), random (unbiased sampling)')
    parser.add_argument('--seed', type=int, default=42)

    # Policies
    parser.add_argument('--cutoff_policies', type=str, default='model,energy80',
                        help='Comma list: model,energy80,fixed_k')
    parser.add_argument('--fixed_k', type=int, default=None)

    # t-SNE settings
    parser.add_argument('--tsne_perplexity', type=int, default=20)
    parser.add_argument('--tsne_seed', type=int, default=42)

    # IO & output
    parser.add_argument('--output_dir', type=str, default='exp/batch_stats')
    parser.add_argument('--save_user_figs', type=str2bool, default=False)
    parser.add_argument('--journal_style', type=str, default='custom', choices=['nature', 'science', 'cell', 'high_quality', 'custom'])

    # Domain analysis (NEW)
    parser.add_argument('--analyze_domain', type=int, default=None, help='Analyze only users from specific domain')
    parser.add_argument('--domain_aware', type=str2bool, default=True, help='Use domain-aware analysis (user correct encoder per domain)')
    parser.add_argument('--compare_domains', type=str2bool, default=True, help='Generate domain comparison statistics and plots')

    # Device
    parser.add_argument('--device', type=str, default='auto',
                        help='cpu | cuda | cuda:0 | auto (default: auto)')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load experiment args (optional)
    exp_args: Dict[str, str] = {}
    if args.experiment_dir is not None:
        exp_dir = Path(args.experiment_dir)
        if not exp_dir.exists():
            cand = _PROJECT_ROOT / args.experiment_dir.lstrip(os.sep)
            if cand.exists():
                exp_dir = cand
        args_txt = exp_dir / 'args.txt'
        if args_txt.exists():
            with open(args_txt, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or ',' not in line:
                        continue
                    k, v = line.split(',', 1)
                    exp_args[k.strip()] = v.strip()

    datasets = args.datasets
    if datasets is None:
        if 'use_datasets' in exp_args:
            datasets = _to_list(exp_args['use_datasets'])
        else:
            datasets = ['beauty_5_5']

    # Load data
    dataset = partition_multi_domain(datasets)
    [user_train, user_valid, user_test, user_to_domain, usernum, itemnum, domain_to_item_range] = dataset
    if not user_train:
        raise RuntimeError("No training data found.")

    # Build model
    class _Cfg: pass
    cfg = _Cfg()
    # Resolve device
    if args.device is None or str(args.device).lower() == 'auto':
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device_str = str(args.device)
    device = torch.device(device_str)
    print(f"Using device: {device}")
    setattr(cfg, 'device', device)
    setattr(cfg, 'maxlen', int(exp_args.get('maxlen', args.maxlen)))
    setattr(cfg, 'hidden_units', int(exp_args.get('hidden_units', 64)))
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

    model = SynRec(usernum, itemnum, cfg).to(device)

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
        raise RuntimeError("Checkpoint not found.")

    # Load checkpoint on the chosen device (safe to map to CPU then .to(device), but map directly for speed)
    try:
        state = torch.load(ckpt_path, map_location=device)
    except Exception:
        state = torch.load(ckpt_path, map_location=torch.device('cpu'))
    model.load_state_dict(state, strict=False)
    model.eval()

    if not hasattr(model, 'enhanced_rating_module'):
        raise RuntimeError('Model lacks enhanced_rating_module.')
    
    print(f"Model architecture: {'Multi-domain' if hasattr(model.enhanced_rating_module, 'domain_encoders') else 'Single encoder'}")

    # Balanced domain sampling with flexible within-domain strategy
    if args.analyze_domain is not None:
        # Filter users by domain and sequence length
        domain_users = {uid: seq for uid, seq in user_train.items() 
                       if get_user_domain_id(uid, user_to_domain, 0) == args.analyze_domain 
                       and len(seq) >= args.min_seq_len}
        if not domain_users:
            raise RuntimeError(f"No users found in domain {args.analyze_domain}")
        user_ids = sample_users(domain_users, n_users=args.n_users, strategy=args.sample_strategy, seed=args.seed)
        print(f"Sampling {len(user_ids)} users from domain {args.analyze_domain} using {args.sample_strategy} strategy")
    else:
        # Always use balanced domain allocation with chosen within-domain strategy
        user_ids = sample_users_balanced(
            user_train, user_to_domain, 
            n_users=args.n_users, 
            strategy=args.sample_strategy,  # Use the chosen strategy within each domain
            seed=args.seed,
            min_seq_len=args.min_seq_len  # 添加最小序列长度过滤
        )
        print(f"🎯 平衡采样: 每个领域均等分配，领域内使用 {args.sample_strategy} 策略")
    if len(user_ids) == 0:
        raise RuntimeError("No users selected.")

    # Prepare policy list
    policies = [p.strip() for p in args.cutoff_policies.split(',') if p.strip()]
    policies = [p for p in policies if p in ('model', 'energy80', 'fixed_k')]
    if 'fixed_k' in policies and args.fixed_k is None:
        print("[WARN] fixed_k policy selected but --fixed_k is None; it will be ignored at runtime.")

    # Accumulate user-level metrics
    rows: List[Dict] = []

    # Iterate users
    for idx, uid in enumerate(user_ids, 1):
        seq = user_train[uid]
        rating_series = build_rating_series(seq, maxlen=args.maxlen)
        if rating_series.size == 0:
            continue

        # Skip users with too-short sequences to avoid degenerate FFT metrics
        if rating_series.size < args.min_seq_len:
            continue
        
        # Determine analysis domain
        user_domain = get_user_domain_id(uid, user_to_domain, default_domain=0)
        analysis_domain = user_domain if args.domain_aware else None

        try:
            # Domain-aware FFT decomposition
            freqs, amplitude, low_s, high_s, cutoff_ratio = compute_scalar_fft_decomposition(
                rating_series, model, analysis_domain
            )
        except Exception as e:
            print(f"Error in FFT decomposition for user {uid} (domain {user_domain}): {e}")
            continue
            
        L = len(rating_series)
        half = L // 2
        amp_half = amplitude[:half]

        # Model cutoff (for reference) & auto 80%
        kc_model = min(int(cutoff_ratio * L), half - 1)
        kc_80 = auto_cutoff_80(amp_half)

        # Peaks (top-1)
        peaks = find_top_peaks(amp_half, n=3, min_distance=3)
        k1 = int(peaks[0]) if len(peaks) > 0 else None
        T1 = float(L / k1) if k1 and k1 > 0 else None

        # Energy fractions per policy
        energy_by_policy: Dict[str, float] = {}
        cutoff_idx_by_policy: Dict[str, int] = {}
        for pol in policies:
            kc = resolve_cutoff_idx(L, amp_half, cutoff_ratio, policy=pol, fixed_k=args.fixed_k)
            cutoff_idx_by_policy[pol] = int(kc)
            low_frac, high_frac = energy_ratio(amp_half, kc)
            energy_by_policy[pol] = float(low_frac)

        # R^2 with model trend (consistent with encoder low_s)
        try:
            r2_reg = r2_trend_plus_mean_consistent(rating_series, model, analysis_domain)
        except Exception as e:
            print(f"Error in R2 calculation for user {uid}: {e}")
            r2_reg = float('nan')

        # Embedding separability
        try:
            pre_emb, long_emb, short_emb = compute_embedding_sequences(rating_series, model, analysis_domain)
            from sklearn.metrics import silhouette_score
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import make_pipeline
            X = np.vstack([pre_emb, long_emb, short_emb])
            labels = np.array(['Pre'] * pre_emb.shape[0] + ['Long'] * long_emb.shape[0] + ['Short'] * short_emb.shape[0])
            sil = float(silhouette_score(X, labels, metric='cosine'))
            X_tr, X_te, y_tr, y_te = train_test_split(X, labels, test_size=0.3, random_state=args.tsne_seed, stratify=labels)
            clf = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=1000))
            clf.fit(X_tr, y_tr)
            acc = float((clf.predict(X_te) == y_te).mean())
        except Exception as e:
            print(f"Error in embedding analysis for user {uid}: {e}")
            sil, acc = float('nan'), float('nan')

        row = {
            "user_id": int(uid),
            "length": int(len(rating_series)),
            "domain": int(user_domain),
            "analysis_domain": analysis_domain,
            "domain_aware": args.domain_aware,
            "k_model": int(kc_model),
            "k_energy80": int(kc_80),
            "peak_k1": None if k1 is None else int(k1),
            "peak_T1": None if T1 is None else float(T1),
            "model_cutoff_ratio": float(cutoff_ratio),
            "R2_reg_trend_plus_mean": float(r2_reg),
            "silhouette_cosine": float(sil),
            "logreg_accuracy": float(acc),
        }
        # merge energy by policy
        for pol in policies:
            row[f"lowfrac_{pol}"] = float(energy_by_policy[pol])
            row[f"k_{pol}"] = int(cutoff_idx_by_policy[pol])
        rows.append(row)

        # Optional save per-user figures (using chosen policy for shading)
        if args.save_user_figs:
            chosen_policy = policies[0] if len(policies) > 0 else "model"
            kc_chosen = resolve_cutoff_idx(L, amp_half, cutoff_ratio, policy=chosen_policy, fixed_k=args.fixed_k)
            low_frac, high_frac = energy_ratio(amp_half, kc_chosen)

            user_dir = os.path.join(args.output_dir, f"user_{uid}")
            os.makedirs(user_dir, exist_ok=True)

            domain_suffix = f" (Domain {analysis_domain})" if analysis_domain is not None else ""
            
            plot_time_and_spectrum(
                rating_series, freqs, amplitude,
                cutoff_idx=kc_chosen,
                kc_80=kc_80,
                low_frac=low_frac, high_frac=high_frac,
                out_path=os.path.join(user_dir, f'fig1_time_vs_frequency_d{analysis_domain}.png'),
                journal=args.journal_style,
                label_policy=chosen_policy,
                peaks=peaks
            )
            plot_decomposition_overlay(
                rating_series, low_s, high_s,
                r2_energy=low_frac, r2_reg=r2_reg,
                out_path=os.path.join(user_dir, f'fig2_decomposition_overlay_d{analysis_domain}.png'),
                journal=args.journal_style
            )

        if idx % 20 == 0:
            print(f"Processed {idx}/{len(user_ids)} users...")

    print(f"Completed analysis of {len(rows)} users with domain-aware processing: {args.domain_aware}")

    # Save CSV
    if pd is not None:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(args.output_dir, 'metrics_user_level.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8')
    else:
        # Fallback JSONL
        csv_path = os.path.join(args.output_dir, 'metrics_user_level.jsonl')
        with open(csv_path, 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # -------- Group-level summaries --------
    # Basic aggregates
    def _safe(vals):
        return [float(x) for x in vals if isinstance(x, (float,int)) and not math.isnan(float(x))]

    lowfrac_overall = _safe([r.get('lowfrac_model', np.nan) if 'lowfrac_model' in r else
                             r.get('lowfrac_energy80', np.nan) for r in rows])
    r2_vals = _safe([r['R2_reg_trend_plus_mean'] for r in rows])
    sil_vals = _safe([r['silhouette_cosine'] for r in rows])
    acc_vals = _safe([r['logreg_accuracy'] for r in rows])
    T1_vals = _safe([r['peak_T1'] for r in rows])

    summary = {
        "n_users": len(rows),
        "lowfrac_median": float(np.median(lowfrac_overall)) if lowfrac_overall else None,
        "lowfrac_q1": float(np.percentile(lowfrac_overall, 25)) if lowfrac_overall else None,
        "lowfrac_q3": float(np.percentile(lowfrac_overall, 75)) if lowfrac_overall else None,
        "pct_lowfrac_gt_0p7": float(np.mean(np.array(lowfrac_overall) > 0.7)) if lowfrac_overall else None,
        "R2_reg_median": float(np.median(r2_vals)) if r2_vals else None,
        "silhouette_mean": float(np.mean(sil_vals)) if sil_vals else None,
        "logreg_acc_mean": float(np.mean(acc_vals)) if acc_vals else None,
        "period_T1_median": float(np.median(T1_vals)) if T1_vals else None,
    }
    save_metrics_json(summary, args.output_dir, filename='summary_stats.json')

    # -------- Domain-specific Analysis (NEW) --------
    if args.compare_domains and len(rows) > 0:
        generate_domain_comparison_analysis(rows, args)

    # Group plots
    apply_journal_style(args.journal_style)

    # Histogram of LowFrac (use model policy if present else energy80)
    if lowfrac_overall:
        hist_plot(lowfrac_overall, title="Distribution of Low-Frequency Energy Fraction",
                  xlabel="LowFrac", out_path=os.path.join(args.output_dir, "group_lowfrac_hist.png"))

    # Boxplot across policies (if multiple)
    vals_by_policy: Dict[str, List[float]] = {}
    for pol in policies:
        vals_by_policy[pol] = _safe([r.get(f"lowfrac_{pol}", np.nan) for r in rows])
    if any(len(v) > 0 for v in vals_by_policy.values()):
        group_boxplot(vals_by_policy, title="LowFrac across policies", ylabel="LowFrac",
                      out_path=os.path.join(args.output_dir, "group_lowfrac_box_by_policy.png"))

    # Correlation: LowFrac vs silhouette/logreg
    if lowfrac_overall and sil_vals:
        scatter_plot(lowfrac_overall[:len(sil_vals)], sil_vals[:len(lowfrac_overall)],
                     title="LowFrac vs Silhouette (embedding separability)",
                     xlabel="LowFrac", ylabel="Silhouette",
                     out_path=os.path.join(args.output_dir, "group_corr_lowfrac_silhouette.png"))
    if lowfrac_overall and acc_vals:
        scatter_plot(lowfrac_overall[:len(acc_vals)], acc_vals[:len(lowfrac_overall)],
                     title="LowFrac vs Linear separability (logreg acc)",
                     xlabel="LowFrac", ylabel="LogReg Acc",
                     out_path=os.path.join(args.output_dir, "group_corr_lowfrac_logreg.png"))

    print(f"Saved batch outputs to: {args.output_dir}")
    print(f"- user-level metrics: {csv_path}")
    print(f"- summary: {os.path.join(args.output_dir, 'summary_stats.json')}")
    print("Done.")


def generate_domain_comparison_analysis(rows: List[Dict], args):
    """生成领域对比分析和可视化"""
    print("\n🔍 Generating domain comparison analysis...")
    
    # 按领域分组数据
    domain_data = {}
    for row in rows:
        domain_id = row['domain']
        if domain_id not in domain_data:
            domain_data[domain_id] = []
        domain_data[domain_id].append(row)
    
    print(f"Found {len(domain_data)} domains: {list(domain_data.keys())}")
    
    # 计算领域级统计
    domain_stats = {}
    for domain_id, domain_rows in domain_data.items():
        if len(domain_rows) == 0:
            continue
            
        # 提取指标
        low_fracs = [r.get('lowfrac_model', r.get('lowfrac_energy80', np.nan)) for r in domain_rows]
        r2_vals = [r['R2_reg_trend_plus_mean'] for r in domain_rows]
        sil_vals = [r['silhouette_cosine'] for r in domain_rows]
        acc_vals = [r['logreg_accuracy'] for r in domain_rows]
        cutoff_ratios = [r.get('model_cutoff_ratio', np.nan) for r in domain_rows]
        
        # 过滤NaN值
        def _safe_vals(vals):
            return [float(x) for x in vals if isinstance(x, (float, int)) and not math.isnan(float(x))]
        
        low_fracs_clean = _safe_vals(low_fracs)
        r2_vals_clean = _safe_vals(r2_vals)
        sil_vals_clean = _safe_vals(sil_vals)
        acc_vals_clean = _safe_vals(acc_vals)
        cutoff_ratios_clean = _safe_vals(cutoff_ratios)
        
        domain_stats[domain_id] = {
            'n_users': len(domain_rows),
            'lowfrac_mean': float(np.mean(low_fracs_clean)) if low_fracs_clean else np.nan,
            'lowfrac_std': float(np.std(low_fracs_clean)) if low_fracs_clean else np.nan,
            'lowfrac_median': float(np.median(low_fracs_clean)) if low_fracs_clean else np.nan,
            'R2_mean': float(np.mean(r2_vals_clean)) if r2_vals_clean else np.nan,
            'R2_std': float(np.std(r2_vals_clean)) if r2_vals_clean else np.nan,
            'silhouette_mean': float(np.mean(sil_vals_clean)) if sil_vals_clean else np.nan,
            'logreg_acc_mean': float(np.mean(acc_vals_clean)) if acc_vals_clean else np.nan,
            'cutoff_ratio_mean': float(np.mean(cutoff_ratios_clean)) if cutoff_ratios_clean else np.nan,
            'cutoff_ratio_std': float(np.std(cutoff_ratios_clean)) if cutoff_ratios_clean else np.nan,
        }
    
    # 保存领域统计
    save_metrics_json(domain_stats, args.output_dir, filename='domain_comparison_stats.json')
    
    # 生成领域对比图表
    generate_domain_comparison_plots(domain_data, domain_stats, args)
    
    print(f"✅ Domain comparison analysis saved to: {args.output_dir}")


def _plot_metric_boxplot(ax, domain_data: Dict[int, List[Dict]], domains: List[int], 
                        domain_colors: Dict[int, str], domain_names: Dict[int, str],
                        metric_key: str, ylabel: str, title: str, subplot_label: str = None):
    """统一的箱线图绘制函数"""
    import math
    import numpy as np
    
    data_by_domain = []
    domain_labels = []
    colors_list = []
    
    for domain_id in domains:
        rows = domain_data[domain_id]
        
        # 处理不同的metric_key格式
        if metric_key == 'lowfrac':
            values = [r.get('lowfrac_model', r.get('lowfrac_energy80', np.nan)) for r in rows]
        else:
            values = [r.get(metric_key, np.nan) for r in rows]
        
        values_clean = [x for x in values if not math.isnan(x)]
        
        if len(values_clean) > 0:
            data_by_domain.append(values_clean)
            domain_name = domain_names.get(domain_id, f'Domain {domain_id}')
            domain_labels.append(f'{domain_name}\n(n={len(values_clean)})')
            colors_list.append(domain_colors.get(domain_id, '#888888'))
    
    if len(data_by_domain) > 1:
        bp = ax.boxplot(data_by_domain, patch_artist=True, showmeans=True, meanline=True)
        
        # 应用统一配色
        for i, patch in enumerate(bp['boxes']):
            if i < len(colors_list):
                patch.set_facecolor(colors_list[i])
                patch.set_alpha(0.7)
        
        ax.set_xticklabels(domain_labels, rotation=0, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 确保纵轴范围合适 - 针对R²值的特殊处理
        if 'R2' in metric_key or 'r2' in metric_key.lower():
            all_values = [val for sublist in data_by_domain for val in sublist]
            if all_values:
                min_val = min(all_values)
                max_val = max(all_values)
                margin = (max_val - min_val) * 0.1  # 10%边距
                ax.set_ylim(max(0, min_val - margin), min(1, max_val + margin))
        
    else:
        ax.text(0.5, 0.5, 'Insufficient data\nfor comparison', 
               ha='center', va='center', transform=ax.transAxes, 
               fontsize=14, style='italic')
        ax.set_title(title, fontsize=15, fontweight='bold')
    
    # 在设置完所有内容后添加子图标签，使用更安全的位置
    if subplot_label:
        ax.text(-0.15, 0.98, subplot_label, transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='right')


def _plot_peak_frequency_analysis(ax, domain_data: Dict[int, List[Dict]], domains: List[int],
                                 domain_colors: Dict[int, str], domain_names: Dict[int, str], subplot_label: str = None):
    """峰值频率分析 - 新增功能"""
    import math
    import numpy as np
    
    # 收集峰值周期数据
    peak_periods_by_domain = []
    domain_labels = []
    colors_list = []
    
    for domain_id in domains:
        rows = domain_data[domain_id]
        # 提取峰值周期 T1 (使用正确的字段名)
        periods = []
        for r in rows:
            if 'peak_T1' in r and r['peak_T1'] is not None and not math.isnan(r['peak_T1']):
                periods.append(r['peak_T1'])
        
        if len(periods) > 0:
            # 过滤异常值 (周期 > 100 可能不合理)
            periods_clean = [p for p in periods if 1 <= p <= 50]
            if len(periods_clean) > 0:
                peak_periods_by_domain.append(periods_clean)
                domain_name = domain_names.get(domain_id, f'Domain {domain_id}')
                domain_labels.append(f'{domain_name}\n(n={len(periods_clean)})')
                colors_list.append(domain_colors.get(domain_id, '#888888'))
    
    if len(peak_periods_by_domain) >= 1:
        # 使用密度直方图显示峰值周期分布
        all_periods = np.concatenate(peak_periods_by_domain)
        
        # 为每个领域绘制密度曲线
        x_range = np.linspace(1, min(50, max(all_periods) if len(all_periods) > 0 else 20), 100)
        
        for i, (periods, label, color) in enumerate(zip(peak_periods_by_domain, domain_labels, colors_list)):
            if len(periods) > 1:
                # 使用核密度估计如果可用
                if SCIPY_AVAILABLE:
                    try:
                        kde = gaussian_kde(periods)
                        density = kde(x_range)
                        domain_name = label.split('\n')[0]  # 只取领域名
                        ax.plot(x_range, density, color=color, linewidth=2, alpha=0.8, label=domain_name)
                        ax.fill_between(x_range, density, alpha=0.2, color=color)
                    except Exception:
                        # 回退到简单直方图
                        ax.hist(periods, bins=15, alpha=0.6, color=color, 
                               label=label.split('\n')[0], density=True)
                else:
                    # 使用简单直方图
                    ax.hist(periods, bins=15, alpha=0.6, color=color, 
                           label=label.split('\n')[0], density=True)
        
        ax.set_xlabel('Peak Period (Time Steps)', fontsize=14)
        ax.set_ylabel('Density', fontsize=14)
        ax.set_title('Peak Frequency Patterns by Domain', fontsize=15, fontweight='bold')
        ax.legend(fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, min(50, max(all_periods) if len(all_periods) > 0 else 20))
    else:
        ax.text(0.5, 0.5, 'No peak frequency\ndata available', 
               ha='center', va='center', transform=ax.transAxes, 
               fontsize=14, style='italic')
        ax.set_title('Peak Frequency Patterns', fontsize=15, fontweight='bold')
    
    # 添加子图标签
    if subplot_label:
        ax.text(-0.15, 0.98, subplot_label, transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='right')


def _plot_lowfrac_across_policies(ax, domain_data: Dict[int, List[Dict]], domains: List[int],
                                 domain_colors: Dict[int, str], domain_names: Dict[int, str], subplot_label: str = None):
    """低频能量跨策略对比 - 新增功能"""
    import math
    import numpy as np
    
    # 定义可用策略和颜色
    policies = ['model', 'energy80']
    policy_labels = {'model': 'Model Policy', 'energy80': 'Energy80 Policy'}
    
    # 准备数据
    data_by_domain_policy = {}
    
    for domain_id in domains:
        rows = domain_data[domain_id]
        data_by_domain_policy[domain_id] = {}
        
        for policy in policies:
            field_name = f'lowfrac_{policy}'
            values = []
            for r in rows:
                if field_name in r and not math.isnan(r[field_name]):
                    values.append(r[field_name])
            
            if values:
                data_by_domain_policy[domain_id][policy] = values
    
    # 检查是否有数据
    has_data = any(len(data_by_domain_policy[d]) > 0 for d in domains if d in data_by_domain_policy)
    
    if not has_data:
        ax.text(0.5, 0.5, 'No policy comparison\ndata available', 
               ha='center', va='center', transform=ax.transAxes, 
               fontsize=14, style='italic')
        ax.set_title('LowFrac across Policies', fontsize=15, fontweight='bold')
        return
    
    # 计算位置
    n_domains = len(domains)
    n_policies = len(policies)
    width = 0.35  # 策略间的间距
    x_positions = np.arange(n_domains)
    
    # 绘制箱线图
    box_data = []
    box_positions = []
    box_colors = []
    labels = []
    
    for i, domain_id in enumerate(domains):
        domain_name = domain_names.get(domain_id, f'Domain {domain_id}')
        base_color = domain_colors.get(domain_id, '#888888')
        
        for j, policy in enumerate(policies):
            if domain_id in data_by_domain_policy and policy in data_by_domain_policy[domain_id]:
                values = data_by_domain_policy[domain_id][policy]
                box_data.append(values)
                
                # 位置计算: 每个领域内部的策略偏移
                pos = i + (j - 0.5) * width
                box_positions.append(pos)
                
                # 颜色: 同领域使用同色系，不同透明度
                alpha = 0.8 if policy == 'model' else 0.5
                box_colors.append(base_color)
                
                # 标签
                if i == 0:  # 只在第一个领域添加策略标签
                    labels.append(policy_labels[policy])
    
    if box_data:
        # 绘制箱线图
        bp = ax.boxplot(box_data, positions=box_positions, patch_artist=True, 
                       showmeans=True, meanline=True, widths=width*0.8)
        
        # 应用颜色
        for i, patch in enumerate(bp['boxes']):
            if i < len(box_colors):
                patch.set_facecolor(box_colors[i])
                domain_idx = i // n_policies
                policy_idx = i % n_policies
                alpha = 0.8 if policies[policy_idx] == 'model' else 0.5
                patch.set_alpha(alpha)
        
        # 设置 x 轴
        domain_labels = [domain_names.get(d, f'Domain {d}') for d in domains]
        ax.set_xticks(x_positions)
        ax.set_xticklabels(domain_labels, fontsize=14)
        
        ax.set_ylabel('Low-Frequency Energy Fraction', fontsize=14)
        ax.set_title('LowFrac across Policies', fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 添加图例（简化版）
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='gray', alpha=0.8, label='Model Policy'),
            Patch(facecolor='gray', alpha=0.5, label='Energy80 Policy')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=13)
    else:
        ax.text(0.5, 0.5, 'Insufficient policy\ndata for comparison', 
               ha='center', va='center', transform=ax.transAxes, 
               fontsize=14, style='italic')
        ax.set_title('LowFrac across Policies', fontsize=15, fontweight='bold')
    
    # 添加子图标签
    if subplot_label:
        ax.text(-0.15, 0.98, subplot_label, transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='right')


def _plot_domain_metrics_summary(ax, domain_stats: Dict[int, Dict], domains: List[int],
                                domain_colors: Dict[int, str], domain_names: Dict[int, str], subplot_label: str = None):
    """领域指标摘要表 - 新增功能"""
    import math
    
    if len(domain_stats) <= 1:
        ax.text(0.5, 0.5, 'Insufficient domain\nstatistics available', 
               ha='center', va='center', transform=ax.transAxes, 
               fontsize=14, style='italic')
        ax.set_title('Domain Metrics Summary', fontsize=15, fontweight='bold')
        return
    
    # 准备数据表格
    metrics_info = [
        ('lowfrac_mean', 'Low-Freq', '.3f'),
        ('R2_mean', 'R² Score', '.3f'),
        ('silhouette_mean', 'Silhouette', '.3f'),
        ('logreg_acc_mean', 'LogReg Acc', '.3f'),
    ]
    
    # 构建表格数据
    table_data = []
    row_colors = []
    
    # 表头
    headers = ['Domain'] + [info[1] for info in metrics_info]
    table_data.append(headers)
    row_colors.append(['lightgray'] * len(headers))
    
    # 数据行
    for domain_id in domains:
        if domain_id in domain_stats:
            domain_name = domain_names.get(domain_id, f'Domain {domain_id}')
            row = [domain_name]
            
            for metric_key, _, fmt in metrics_info:
                if metric_key in domain_stats[domain_id]:
                    value = domain_stats[domain_id][metric_key]
                    if not math.isnan(value):
                        row.append(f'{value:{fmt}}')
                    else:
                        row.append('N/A')
                else:
                    row.append('N/A')
            
            table_data.append(row)
            domain_color = domain_colors.get(domain_id, '#f0f0f0')
            row_colors.append([domain_color] + ['white'] * (len(headers) - 1))
    
    # 绘制表格
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=table_data[1:],  # 数据行
                    colLabels=table_data[0],   # 表头
                    cellColours=row_colors[1:], # 数据行颜色
                    colColours=row_colors[0],   # 表头颜色
                    cellLoc='center',
                    loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # 调整表格样式
    for i in range(len(table_data)):
        for j in range(len(headers)):
            cell = table[(i, j)] if i > 0 else table[(i, j)]
            if i == 0:  # 表头
                cell.set_text_props(weight='bold')
            cell.set_edgecolor('black')
            cell.set_linewidth(0.5)
    
    ax.set_title('Domain Performance Summary', fontsize=15, fontweight='bold')
    
    # 添加子图标签
    if subplot_label:
        ax.text(-0.15, 0.98, subplot_label, transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='right')


def generate_domain_comparison_plots(domain_data: Dict[int, List[Dict]], domain_stats: Dict[int, Dict], args):
    """生成统一的领域对比图表 - 2x3布局版本"""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import numpy as np
    
    apply_journal_style(args.journal_style)
    
    domains = sorted(domain_data.keys())
    if len(domains) <= 1:
        print("警告: 需要至少两个领域才能生成对比图")
        return
    
    # 统一的领域配色方案
    domain_colors = {
        0: '#e74c3c',    # Beauty - 红色
        1: '#2ecc71',    # Games - 绿色  
        2: '#3498db',    # MovieLens - 蓝色
        3: '#f39c12',    # 兼容第4个领域 - 橙色
        4: '#9b59b6'     # 兼容第5个领域 - 紫色
    }
    
    domain_names = {
        0: 'Beauty',
        1: 'Games', 
        2: 'MovieLens',
        3: 'Domain 3',
        4: 'Domain 4'
    }
    
    # 创建2x3统一布局，添加学术标准标签
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=600)
    fig.suptitle('Multi-Domain Frequency Analysis Comparison', fontsize=16, fontweight='bold', y=0.95)
    
    # 定义子图标签
    subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    
    # 调整子图间距，为标签预留空间
    plt.subplots_adjust(hspace=0.3, wspace=0.25)
    
    # ============ 第一行: 箱线图三连 ============
    
    # (0,0) Low-Frequency Energy 分布
    ax_lowfrac = axes[0, 0]
    _plot_metric_boxplot(ax_lowfrac, domain_data, domains, domain_colors, domain_names,
                        'lowfrac', 'Low-Frequency Energy Fraction', 
                        'Distribution of Low-Frequency Energy', subplot_labels[0])
    
    # (0,1) R² Score 分布  
    ax_r2 = axes[0, 1]
    _plot_metric_boxplot(ax_r2, domain_data, domains, domain_colors, domain_names,
                        'R2_reg_trend_plus_mean', 'R² (Trend + Mean)', 
                        'Distribution of Decomposition R²', subplot_labels[1])
    
    # (0,2) Silhouette Score 分布
    ax_sil = axes[0, 2] 
    _plot_metric_boxplot(ax_sil, domain_data, domains, domain_colors, domain_names,
                        'silhouette_cosine', 'Silhouette Score',
                        'Distribution of Embedding Separability', subplot_labels[2])
    
    # ============ 第二行: 分析增强 ============
    
    # (1,0) LowFrac across Policies - 新增功能
    ax_policies = axes[1, 0]
    _plot_lowfrac_across_policies(ax_policies, domain_data, domains, domain_colors, domain_names, subplot_labels[3])
    
    # (1,1) Peak Frequency Analysis - 新增功能
    ax_peak = axes[1, 1]
    _plot_peak_frequency_analysis(ax_peak, domain_data, domains, domain_colors, domain_names, subplot_labels[4])
    
    # (1,2) Domain Metrics Summary - 新增功能
    ax_summary = axes[1, 2]
    _plot_domain_metrics_summary(ax_summary, domain_stats, domains, domain_colors, domain_names, subplot_labels[5])
    
    # 添加统一图例，优化位置
    legend_elements = [Patch(facecolor=domain_colors[d], alpha=0.8, label=domain_names[d]) 
                      for d in domains if d in domain_colors]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02), 
              ncol=len(domains), fontsize=15, frameon=True, fancybox=True, shadow=True)
    
    # 保存统一图表，优化布局
    plt.tight_layout(rect=[0, 0.08, 1, 0.92])  # 为标题和图例预留足够空间
    plt.savefig(os.path.join(args.output_dir, 'domain_comparison_unified.png'),
               bbox_inches='tight', dpi=600, facecolor='white')
    plt.close()
    
    print(f"✅ 生成统一的多领域对比图表: domain_comparison_unified.png")


if __name__ == "__main__":
    main()
