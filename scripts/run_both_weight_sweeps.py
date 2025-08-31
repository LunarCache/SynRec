#!/usr/bin/env python3
"""顺序执行 contrastive_weight 与 specialization_weight 两套 sweep 脚本。

功能概述:
1. 根据提供的权重列表调用 `run_contrastive_weight_sweep.py` 与 `run_specialization_weight_sweep.py`。
2. 允许分别/共同配置常用训练与绘图参数 (datasets, epochs, batch_size, x_mode, delta 等)。
3. 支持透传额外参数到两个子脚本 (`--contrastive_extra_args`, `--specialization_extra_args`)。
4. 可选择只跑其中一类 (通过 --skip_contrastive 或 --skip_specialization)。
5. 可选择 no_run 模式（仅解析日志并绘图）。
6. 子脚本执行完成后，汇总 CSV 路径与图像路径输出到一个 JSON 汇总文件。

使用示例:
  python scripts/run_both_weight_sweeps.py \
      --contrastive_weights 0.0 0.001 0.005 0.01 0.05 0.1 \
      --specialization_weights 0.0 0.001 0.005 0.01 0.05 0.1 \
      --epochs_contrastive 50 --epochs_specialization 150 \
      --datasets beauty_5_5 games_5_5 ml-1m_5_5 \
      --plot_delta --delta_as_percent --tight_ylim --value_labels \
      --infer_weighted --domain_counts 22332 15264 6040 \
      --output_dir exp/combined_sweeps

只做解析与绘图 (不重新训练):
  python scripts/run_both_weight_sweeps.py --no_run \
      --contrastive_weights 0.0 0.01 0.05 \
      --specialization_weights 0.0 0.01 0.05 \
      --infer_weighted --domain_counts 22332 15264 6040
"""

from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
import json
import shlex
from datetime import datetime


def parse_args():
    p = argparse.ArgumentParser()
    # 权重列表
    p.add_argument('--contrastive_weights', nargs='+', type=float, help='contrastive_weight 列表 (为空则跳过)')
    p.add_argument('--specialization_weights', nargs='+', type=float, help='specialization_weight 列表 (为空则跳过)')
    # 数据集 / 训练参数
    p.add_argument('--datasets', nargs='+', default=['beauty_5_5','games_5_5','ml-1m_5_5'])
    p.add_argument('--epochs_contrastive', type=int, default=50)
    p.add_argument('--epochs_specialization', type=int, default=150)
    p.add_argument('--batch_size', type=int, default=1024)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--maxlen', type=int, default=100)
    p.add_argument('--hidden_units', type=int, default=64)
    p.add_argument('--num_blocks', type=int, default=2)
    p.add_argument('--num_heads', type=int, default=2)
    p.add_argument('--dropout_rate', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=42)
    # 绘图 & 解析相关共用开关 (若子脚本不支持会被忽略)
    p.add_argument('--x_mode', type=str, default='categorical', choices=['linear','log','categorical'])
    p.add_argument('--plot_delta', action='store_true')
    p.add_argument('--delta_as_percent', action='store_true')
    p.add_argument('--tight_ylim', action='store_true')
    p.add_argument('--value_labels', action='store_true')
    p.add_argument('--infer_weighted', action='store_true')
    p.add_argument('--domain_counts', nargs='+', type=int)
    p.add_argument('--journal_style', type=str, default='nature')
    # 运行控制
    p.add_argument('--no_run', action='store_true', help='传递给子脚本，仅解析日志与绘图')
    p.add_argument('--skip_contrastive', action='store_true')
    p.add_argument('--skip_specialization', action='store_true')
    p.add_argument('--skip_if_exists', action='store_true', help='子脚本存在 log.txt 时跳过训练')
    # 额外参数透传
    p.add_argument('--contrastive_extra_args', type=str, default='', help='透传给 contrastive 脚本的原样参数')
    p.add_argument('--specialization_extra_args', type=str, default='', help='透传给 specialization 脚本的原样参数')
    # 输出
    p.add_argument('--output_dir', type=str, default='exp/combined_sweeps', help='汇总输出目录')
    return p.parse_args()


def build_common_flags(args) -> list[str]:
    flags = [
        '--datasets', *args.datasets,
        '--batch_size', str(args.batch_size),
        '--lr', str(args.lr),
        '--device', args.device,
        '--maxlen', str(args.maxlen),
        '--hidden_units', str(args.hidden_units),
        '--num_blocks', str(args.num_blocks),
        '--num_heads', str(args.num_heads),
        '--dropout_rate', str(args.dropout_rate),
        '--seed', str(args.seed),
        '--x_mode', args.x_mode,
        '--journal_style', args.journal_style,
    ]
    if args.plot_delta: flags.append('--plot_delta')
    if args.delta_as_percent: flags.append('--delta_as_percent')
    if args.tight_ylim: flags.append('--tight_ylim')
    if args.value_labels: flags.append('--value_labels')
    if args.infer_weighted: flags.append('--infer_weighted')
    if args.no_run: flags.append('--no_run')
    if args.skip_if_exists: flags.append('--skip_if_exists')
    if args.domain_counts:
        flags.extend(['--domain_counts', *[str(c) for c in args.domain_counts]])
    return flags


def run_subprocess(cmd: list[str]):
    print('\n[Run] ' + ' '.join(shlex.quote(c) for c in cmd))
    start = datetime.now()
    proc = subprocess.run(cmd)
    duration = datetime.now() - start
    if proc.returncode != 0:
        print(f"[Error] 子进程失败 returncode={proc.returncode}, 用时 {duration}")
    else:
        print(f"[OK] 完成，用时 {duration}")
    return proc.returncode, duration


def locate_outputs(output_root: Path):
    """尝试从子目录中发现主 CSV / 主图 / delta 图 (存在则记录)。"""
    result = {}
    # contrastive
    c_dir = output_root / 'contrastive_weight_sweep'
    if c_dir.exists():
        result['contrastive'] = {
            'csv': str(c_dir / 'contrastive_weight_domain_results.csv'),
            'json': str(c_dir / 'contrastive_weight_domain_results.json'),
            'figure_main_png': str(c_dir / 'contrastive_weight_domains.png'),
            'figure_main_pdf': str(c_dir / 'contrastive_weight_domains.pdf'),
            'figure_delta_png': str(c_dir / 'contrastive_weight_domains_delta.png'),
            'figure_delta_pdf': str(c_dir / 'contrastive_weight_domains_delta.pdf'),
        }
    # specialization
    s_dir = output_root / 'specialization_weight_sweep'
    if s_dir.exists():
        result['specialization'] = {
            'csv': str(s_dir / 'specialization_weight_domain_results.csv'),
            'json': str(s_dir / 'specialization_weight_domain_results.json'),
            'figure_main_png': str(s_dir / 'specialization_weight_domains.png'),
            'figure_main_pdf': str(s_dir / 'specialization_weight_domains.pdf'),
            'figure_delta_png': str(s_dir / 'specialization_weight_domains_delta.png'),
            'figure_delta_pdf': str(s_dir / 'specialization_weight_domains_delta.pdf'),
        }
    return result


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 为保持子脚本原有默认输出结构，这里不强制改其 output_dir，而是允许用户通过 extra_args 自行指定
    common = build_common_flags(args)

    summary = {
        'contrastive': None,
        'specialization': None,
        'errors': []
    }

    if not args.skip_contrastive and args.contrastive_weights:
        cmd = [sys.executable, 'scripts/run_contrastive_weight_sweep.py', '--weights', *[str(w) for w in args.contrastive_weights], '--epochs', str(args.epochs_contrastive)] + common
        if args.contrastive_extra_args:
            cmd.extend(args.contrastive_extra_args.strip().split())
        rc, dur = run_subprocess(cmd)
        if rc != 0:
            summary['errors'].append('contrastive_returncode_'+str(rc))
    else:
        print('[Skip] contrastive sweep (无权重或被跳过)')

    if not args.skip_specialization and args.specialization_weights:
        cmd = [sys.executable, 'scripts/run_specialization_weight_sweep.py', '--weights', *[str(w) for w in args.specialization_weights], '--epochs', str(args.epochs_specialization)] + common
        if args.specialization_extra_args:
            cmd.extend(args.specialization_extra_args.strip().split())
        rc, dur = run_subprocess(cmd)
        if rc != 0:
            summary['errors'].append('specialization_returncode_'+str(rc))
    else:
        print('[Skip] specialization sweep (无权重或被跳过)')

    # 汇总输出信息（如果用户使用默认子脚本 output_dir）
    discovered = locate_outputs(Path('exp'))
    for k, v in discovered.items():
        summary[k] = v

    summary_path = out_dir / 'combined_summary.json'
    with summary_path.open('w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Summary] 汇总信息已保存: {summary_path}")
    print(json.dumps(summary, indent=2))
    if summary['errors']:
        print('[WARN] 子任务存在错误, 请检查 logs / 返回码。')
    else:
        print('[DONE] 所有子任务完成。')


if __name__ == '__main__':
    main()
