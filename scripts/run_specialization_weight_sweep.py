#!/usr/bin/env python3
"""对 specialization_weight 进行批量实验并绘制各领域指标变化（期刊风格）。

功能与参数设计参考 `run_contrastive_weight_sweep.py`，只是目标超参改为 specialization_weight。

核心逻辑:
1. 根据 --weights 列表依次运行 main.py，传入 --specialization_weight <w>
2. 解析 exp/<datasets>_specialization_weight_<w>/log.txt 中 test 指标
3. 对每个领域独立选取 NDCG@10 最佳 epoch 并记录该 epoch 的 (NDCG@10, HR@10)
4. 输出 CSV / JSON / PNG / PDF (1000dpi) 图；支持 journal 风格、x 轴模式等

使用示例:
  python scripts/run_specialization_weight_sweep.py \
      --weights 0.0 0.001 0.005 0.01 0.05 0.1 \
      --epochs 150 \
      --use_swanlab false

仅解析已有日志并画图:
  python scripts/run_specialization_weight_sweep.py --weights 0.0 0.01 0.05 --no_run
"""

from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
import json
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd

try:
    from visualization.config import create_journal_config, setup_visualization_environment
    from visualization.color_schemes import JournalColorSchemes
except ImportError:
    create_journal_config = None
    setup_visualization_environment = None
    JournalColorSchemes = None

DEFAULT_DATASETS = ["beauty_5_5", "games_5_5", "ml-1m_5_5"]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', nargs='+', type=float, required=True)
    p.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS)
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--batch_size', type=int, default=1024)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--maxlen', type=int, default=100)
    p.add_argument('--hidden_units', type=int, default=64)
    p.add_argument('--num_blocks', type=int, default=2)
    p.add_argument('--num_heads', type=int, default=2)
    p.add_argument('--dropout_rate', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--use_swanlab', type=str, default='false', choices=['true','false'])
    p.add_argument('--skip_if_exists', action='store_true')
    p.add_argument('--no_run', action='store_true')
    p.add_argument('--output_dir', type=str, default='exp/specialization_weight_sweep')
    p.add_argument('--extra_main_args', type=str, default='')
    p.add_argument('--journal_style', type=str, default='nature')
    p.add_argument('--line_width', type=float, default=1.0)
    p.add_argument('--marker_size', type=float, default=5.5)
    p.add_argument('--font_size', type=int, default=10)
    p.add_argument('--x_mode', type=str, default='linear', choices=['linear','log','categorical'])
    p.add_argument('--log_zero_shift', type=float, default=1e-5)
    # 细节放大相关
    p.add_argument('--tight_ylim', action='store_true', help='对每个子图自适应最小区间放大 y 轴')
    p.add_argument('--ylim_padding_frac', type=float, default=0.05, help='tight 模式下上下 padding 比例')
    p.add_argument('--plot_delta', action='store_true', help='输出第二套图：显示相对基线(首个 weight)的百分比变化')
    p.add_argument('--delta_as_percent', action='store_true', help='delta 图使用百分比(默认)；若关闭则显示绝对差值')
    p.add_argument('--value_labels', action='store_true', help='在点旁标注数值')
    p.add_argument('--value_format', type=str, default='{:.4f}', help='数值标注格式 (用于绝对值与绝对 delta)')
    p.add_argument('--percent_format', type=str, default='{:+.2f}%', help='百分比标注格式 (用于百分比 delta)')
    p.add_argument('--infer_weighted', action='store_true', help='若日志无 overall_weighted_ 指标则根据 domain 指标与提供的 domain_counts 推断')
    p.add_argument('--domain_counts', nargs='+', type=int, help='各域样本数(顺序需与 --datasets 对应)，用于推断加权指标')
    return p.parse_args()

def run_training(weight: float, args) -> Path:
    datasets_str = '-'.join(args.datasets)
    train_dir = f"specialization_weight_{weight}"
    exp_dir = Path('exp') / f"{datasets_str}_{train_dir}"
    log_path = exp_dir / 'log.txt'
    if args.no_run:
        return log_path
    if args.skip_if_exists and log_path.exists():
        print(f"[Skip] {weight} 已存在 {log_path}")
        return log_path
    cmd = [
        sys.executable, 'main.py',
        '--train_dir', train_dir,
        '--specialization_weight', str(weight),
        '--num_epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--lr', str(args.lr),
        '--device', args.device,
        '--maxlen', str(args.maxlen),
        '--hidden_units', str(args.hidden_units),
        '--num_blocks', str(args.num_blocks),
        '--num_heads', str(args.num_heads),
        '--dropout_rate', str(args.dropout_rate),
        '--seed', str(args.seed),
        '--use_swanlab', args.use_swanlab,
        '--use_datasets', *args.datasets,
        '--use_specialization_loss', 'true'
    ]
    if args.extra_main_args:
        cmd.extend(args.extra_main_args.strip().split())
    print('\n=== Running specialization_weight', weight, '===')
    print(' '.join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"[Error] weight {weight} 训练失败, returncode={proc.returncode}")
    return log_path

def parse_log_per_domain(log_path: Path) -> Dict[str, Dict[str, float]]:
    """解析日志，返回每个 domain 和 overall_weighted 的最佳 (NDCG@10, HR@10)。

    返回示例:
    {
        'domain_0': {...},
        'domain_1': {...},
        'domain_2': {...},
        'overall_weighted': {'best_epoch': e, 'NDCG@10': x, 'HR@10': y}
    }
    """
    if not log_path.exists():
        return {}
    per_domain_records: Dict[str, Dict] = {}
    overall_weighted_epochs: List[Tuple[int, float, float]] = []
    try:
        with log_path.open('r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('epoch\tvalid_metrics'):
                    continue
                parts = line.split('\t')
                if len(parts) < 3:
                    continue
                epoch_str, _valid, test_metrics = parts[0], parts[1], parts[2]
                try:
                    epoch = int(epoch_str)
                except ValueError:
                    continue
                metrics = {}
                for kv in test_metrics.split(','):
                    if ':' not in kv:
                        continue
                    k, v = kv.split(':', 1)
                    try:
                        metrics[k] = float(v)
                    except ValueError:
                        pass
                # 域指标收集
                for k in metrics.keys():
                    if k.startswith('domain_') and ('NDCG@10' in k or 'HT@10' in k):
                        parts_k = k.split('_')
                        if len(parts_k) < 3:
                            continue
                        domain_id = parts_k[0] + '_' + parts_k[1]
                        per_domain_records.setdefault(domain_id, {})
                for domain_id in list(per_domain_records.keys()):
                    ndcg_key = f'{domain_id}_NDCG@10'
                    hr_key = f'{domain_id}_HT@10'
                    ndcg = metrics.get(ndcg_key)
                    hr = metrics.get(hr_key)
                    if ndcg is not None and hr is not None:
                        per_domain_records.setdefault(domain_id, {}).setdefault('epochs', []).append((epoch, ndcg, hr))
                # overall_weighted
                ow_ndcg = metrics.get('overall_weighted_NDCG@10')
                ow_hr = metrics.get('overall_weighted_HT@10')
                if ow_ndcg is not None and ow_hr is not None:
                    overall_weighted_epochs.append((epoch, ow_ndcg, ow_hr))
    except Exception as e:
        print(f"解析失败 {log_path}: {e}")
        return {}
    result: Dict[str, Dict[str, float]] = {}
    for domain_id, store in per_domain_records.items():
        epoch_list = store.get('epochs', [])
        if not epoch_list:
            continue
        best = max(epoch_list, key=lambda x: x[1])
        result[domain_id] = {'best_epoch': best[0], 'NDCG@10': best[1], 'HR@10': best[2]}
    if overall_weighted_epochs:
        bestw = max(overall_weighted_epochs, key=lambda x: x[1])
        result['overall_weighted'] = {'best_epoch': bestw[0], 'NDCG@10': bestw[1], 'HR@10': bestw[2]}
    return result

def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    records: List[Dict] = []
    domain_ids_order: List[str] = []  # 仅存 domain_i
    has_weighted = False
    for w in args.weights:
        log_path = run_training(w, args)
        per_domain = parse_log_per_domain(log_path)
        if per_domain and not domain_ids_order:
            # 过滤掉 overall_weighted
            domain_ids_order = sorted([k for k in per_domain.keys() if k.startswith('domain_')], key=lambda x: int(x.split('_')[1]))
        has_weighted = has_weighted or ('overall_weighted' in per_domain)
        rec = {'specialization_weight': w, 'log_path': str(log_path)}
        for d, vals in per_domain.items():
            rec[f'{d}_best_epoch'] = vals['best_epoch']
            rec[f'{d}_NDCG@10'] = vals['NDCG@10']
            rec[f'{d}_HR@10'] = vals['HR@10']
        print(f"weight={w} -> {per_domain}")
        records.append(rec)

    # 如果没有 weighted 但用户希望推断
    if not has_weighted and args.infer_weighted:
        if not domain_ids_order:
            print('无法推断 weighted: 未找到任何 domain_* 指标')
        else:
            if args.domain_counts and len(args.domain_counts) == len(domain_ids_order):
                counts = args.domain_counts
            else:
                # 默认各域等权
                counts = [1]*len(domain_ids_order)
                if args.domain_counts:
                    print('domain_counts 长度不匹配，使用等权。')
            total_c = sum(counts)
            for rec in records:
                ndcgs = []; hrs = []
                for i, d in enumerate(domain_ids_order):
                    ndcg_col = f'{d}_NDCG@10'
                    hr_col = f'{d}_HR@10'
                    if ndcg_col in rec and hr_col in rec:
                        ndcgs.append(rec[ndcg_col]*counts[i])
                        hrs.append(rec[hr_col]*counts[i])
                if ndcgs and hrs:
                    rec['overall_weighted_best_epoch'] = -1  # 无法精确对应
                    rec['overall_weighted_NDCG@10'] = sum(ndcgs)/total_c
                    rec['overall_weighted_HR@10'] = sum(hrs)/total_c
            has_weighted = True
    df = pd.DataFrame(records).sort_values('specialization_weight')
    csv_path = Path(args.output_dir) / 'specialization_weight_domain_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"域级结果已保存: {csv_path}")
    json_path = Path(args.output_dir) / 'specialization_weight_domain_results.json'
    with open(json_path, 'w') as jf:
        json.dump(records, jf, indent=2)
    print(f"JSON 已保存: {json_path}")
    if not domain_ids_order:
        print('未解析到 domain 指标，结束。')
        return
    # 配置样式
    palette = None
    if create_journal_config and setup_visualization_environment:
        cfg = create_journal_config(args.journal_style)
        cfg.dpi = 1000
        cfg.font_size = args.font_size
        cfg.legend_size = args.font_size - 1
        cfg.line_width = args.line_width
        cfg.marker_size = args.marker_size
        setup_visualization_environment(cfg)
        if JournalColorSchemes:
            palette = JournalColorSchemes().get_journal_palette(args.journal_style, 4)
        import matplotlib as mpl
        mpl.rcParams.update({
            'figure.facecolor': 'white','axes.facecolor': 'white','savefig.facecolor':'white',
            'legend.frameon': False,'grid.color':'#CCCCCC','grid.linewidth':0.6
        })
    # x 轴处理
    weights_sorted = df['specialization_weight'].tolist()
    if args.x_mode == 'categorical':
        x_positions = list(range(len(weights_sorted)))
        x_tick_labels = [f"{w:g}" for w in weights_sorted]
    elif args.x_mode == 'log':
        x_positions = [(args.log_zero_shift if w == 0 else w) for w in weights_sorted]
        x_tick_labels = [('0' if w == 0 else f"{w:g}") for w in weights_sorted]
    else:
        x_positions = weights_sorted
        x_tick_labels = [f"{w:g}" for w in weights_sorted]
    # 图布局
    import math
    plot_units = domain_ids_order.copy()
    if has_weighted:
        plot_units.append('overall_weighted')
    num_plots = len(plot_units)
    # 固定 2x2 布局（若未来域数变化可自适应）
    rows, cols = 2, 2
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.1*rows), dpi=1000)
    axes_list = axes.flatten()
    dataset_names = args.datasets
    def _norm_name(n: str) -> str:
        l = n.lower()
        if 'beauty' in l: return 'Beauty'
        if 'games' in l: return 'Games'
        if 'ml' in l or 'movielens' in l: return 'MovieLens'
        return n
    first_handles = None
    first_labels = None
    for idx, unit_id in enumerate(plot_units):
        ax = axes_list[idx]
        ndcg_col = f'{unit_id}_NDCG@10'
        hr_col = f'{unit_id}_HR@10'
        y_ndcg = df[ndcg_col].tolist() if ndcg_col in df.columns else [float('nan')]*len(weights_sorted)
        y_hr = df[hr_col].tolist() if hr_col in df.columns else [float('nan')]*len(weights_sorted)
        if unit_id.startswith('domain_'):
            dom_index = int(unit_id.split('_')[1])
            raw_name = dataset_names[dom_index] if dom_index < len(dataset_names) else unit_id
            dom_name = _norm_name(raw_name)
        else:
            dom_name = 'Weighted Overall'
        c1 = palette[0] if palette else '#1f77b4'
        c2 = palette[1] if palette else '#ff7f0e'
        line1, = ax.plot(x_positions, y_ndcg, marker='o', label='NDCG@10', color=c1, linewidth=args.line_width)
        line2, = ax.plot(x_positions, y_hr, marker='s', label='HR@10', color=c2, linewidth=args.line_width)
        ax.set_title(dom_name)
        ax.set_xlabel('Specialization Weight')
        ax.set_ylabel('Score')
        ax.grid(alpha=0.3)
        if args.tight_ylim:
            all_vals = [v for v in y_ndcg + y_hr if v == v]
            if all_vals:
                vmin, vmax = min(all_vals), max(all_vals)
                if vmax == vmin:
                    eps = max(1e-6, abs(vmax) * 1e-3)
                    vmin -= eps; vmax += eps
                span = vmax - vmin
                pad = span * args.ylim_padding_frac
                ax.set_ylim(vmin - pad, vmax + pad)
        if args.value_labels:
            for x_, y_ in zip(x_positions, y_ndcg):
                if y_ == y_:
                    ax.text(x_, y_, args.value_format.format(y_), fontsize=max(args.font_size-3,6), color=c1, ha='center', va='bottom')
            for x_, y_ in zip(x_positions, y_hr):
                if y_ == y_:
                    ax.text(x_, y_, args.value_format.format(y_), fontsize=max(args.font_size-3,6), color=c2, ha='center', va='bottom')
        if args.x_mode == 'categorical':
            ax.set_xticks(x_positions); ax.set_xticklabels(x_tick_labels, rotation=30)
        elif args.x_mode == 'log':
            ax.set_xscale('log'); ax.set_xticks(x_positions); ax.set_xticklabels(x_tick_labels, rotation=30)
        else:
            ax.set_xticks(x_positions); ax.set_xticklabels(x_tick_labels, rotation=30)
        if first_handles is None:
            first_handles = [line1, line2]
            first_labels = [h.get_label() for h in first_handles]
    for j in range(len(plot_units), len(axes_list)):
        fig.delaxes(axes_list[j])
    plt.tight_layout(rect=[0,0.05,1,0.92])
    fig.suptitle('Specialization Weight – Domain-wise Top-K Metrics', fontsize=13, y=0.97, fontweight='bold')
    if first_handles is not None:
        fig.legend(first_handles, first_labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.01), frameon=False)
    fig_base = Path(args.output_dir) / 'specialization_weight_domains'
    for fmt in ['png','pdf']:
        out_path = f'{fig_base}.{fmt}'
        fig.savefig(out_path, dpi=1000, bbox_inches='tight', pad_inches=0.05)
        print(f"图已保存: {out_path}")

    # -------- Delta 图（相对提升） --------
    if args.plot_delta:
        base_w = weights_sorted[0]
        base_row = df[df['specialization_weight'] == base_w].iloc[0]
        fig2, axes2 = plt.subplots(rows, cols, figsize=(4*cols, 3.0*rows), dpi=1000)
        axes2_list = axes2.flatten() if hasattr(axes2, 'flatten') else [axes2]
        for idx, unit_id in enumerate(plot_units):
            ax2 = axes2_list[idx]
            ndcg_col = f'{unit_id}_NDCG@10'
            hr_col = f'{unit_id}_HR@10'
            base_ndcg = float(base_row.get(ndcg_col, 'nan'))
            base_hr = float(base_row.get(hr_col, 'nan'))
            y_ndcg_abs = df[ndcg_col].tolist()
            y_hr_abs = df[hr_col].tolist()
            # 计算 delta
            if args.delta_as_percent:
                def pct(cur, base):
                    if base == 0 or base != base or cur != cur:
                        return float('nan')
                    return (cur - base) / base * 100.0
                y_ndcg = [pct(v, base_ndcg) for v in y_ndcg_abs]
                y_hr = [pct(v, base_hr) for v in y_hr_abs]
            else:
                y_ndcg = [v - base_ndcg if v == v and base_ndcg == base_ndcg else float('nan') for v in y_ndcg_abs]
                y_hr = [v - base_hr if v == v and base_hr == base_hr else float('nan') for v in y_hr_abs]
            if unit_id.startswith('domain_'):
                dom_index = int(unit_id.split('_')[1])
                raw_name = dataset_names[dom_index] if dom_index < len(dataset_names) else unit_id
                dom_name = _norm_name(raw_name)
            else:
                dom_name = 'Weighted Overall'
            c1 = palette[0] if palette else '#1f77b4'
            c2 = palette[1] if palette else '#ff7f0e'
            line1, = ax2.plot(x_positions, y_ndcg, marker='o', label='ΔNDCG@10', color=c1, linewidth=args.line_width)
            line2, = ax2.plot(x_positions, y_hr, marker='s', label='ΔHR@10', color=c2, linewidth=args.line_width)
            ax2.set_title(dom_name)
            ax2.set_xlabel('Specialization Weight')
            ax2.set_ylabel('Change (%)' if args.delta_as_percent else 'Absolute Δ')
            ax2.axhline(0, color='#888888', linewidth=0.8)
            ax2.grid(alpha=0.3)
            if args.x_mode == 'categorical':
                ax2.set_xticks(x_positions); ax2.set_xticklabels(x_tick_labels, rotation=30)
            elif args.x_mode == 'log':
                ax2.set_xscale('log'); ax2.set_xticks(x_positions); ax2.set_xticklabels(x_tick_labels, rotation=30)
            else:
                ax2.set_xticks(x_positions); ax2.set_xticklabels(x_tick_labels, rotation=30)
            if args.tight_ylim:
                vals = [v for v in y_ndcg + y_hr if v == v]
                if vals:
                    vmin, vmax = min(vals), max(vals)
                    if vmin == vmax:
                        eps = max(1e-6, abs(vmax) * 1e-3)
                        vmin -= eps; vmax += eps
                    span = vmax - vmin
                    pad = span * args.ylim_padding_frac
                    ax2.set_ylim(vmin - pad, vmax + pad)
            if args.value_labels:
                fmt = args.percent_format if args.delta_as_percent else args.value_format
                for x_, y_ in zip(x_positions, y_ndcg):
                    if y_ == y_:
                        ax2.text(x_, y_, fmt.format(y_), fontsize=max(args.font_size-3,6), color=c1, ha='center', va='bottom')
                for x_, y_ in zip(x_positions, y_hr):
                    if y_ == y_:
                        ax2.text(x_, y_, fmt.format(y_), fontsize=max(args.font_size-3,6), color=c2, ha='center', va='bottom')
            if idx == 0:
                second_handles = [line1, line2]
                second_labels = [h.get_label() for h in second_handles]
        for j in range(len(plot_units), len(axes2_list)):
            fig2.delaxes(axes2_list[j])
        plt.tight_layout(rect=[0,0.05,1,0.92])
        # 与主图标题风格统一：加前缀 “Specialization Weight – ” 并显式 Relative/Absolute
        title_delta = 'Relative Change' if args.delta_as_percent else 'Absolute Change'
        fig2.suptitle(f'Specialization Weight – {title_delta} vs. Base', fontsize=13, y=0.97, fontweight='bold')
        if 'second_handles' in locals():
            fig2.legend(second_handles, second_labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.01), frameon=False)
        fig2_base = Path(args.output_dir) / 'specialization_weight_domains_delta'
        for fmt in ['png','pdf']:
            out_path2 = f'{fig2_base}.{fmt}'
            fig2.savefig(out_path2, dpi=1000, bbox_inches='tight', pad_inches=0.05)
            print(f"Delta 图已保存: {out_path2}")
    print('\n完成。')

if __name__ == '__main__':
    main()
