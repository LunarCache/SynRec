#!/usr/bin/env python3
"""Redesigned Fig. 4 (hyperparameter sensitivity) as a single 2x2 figure.

Layout (one \\textwidth figure):
  rows = regularization weight   (top: Specialization lambda1, bottom: Contrastive lambda2)
  cols = metric                  (left: NDCG@10, right: HR@10)
Each panel overlays the RELATIVE change (%) of Beauty, Games, MovieLens and
Weighted Overall against the weight value (baseline = weight 0).

Uses the ORIGINAL sweep result CSVs (same data as the previous Fig. 4); no
retraining and no change to the underlying numbers -- only the presentation.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Domain column -> display name (domain_0/1/2 follow the dataset order used in
# the sweeps: Beauty, Games, MovieLens; plus the weighted overall curve).
DOMAIN_MAP = [
    ('domain_0', 'Beauty'),
    ('domain_1', 'Games'),
    ('domain_2', 'MovieLens'),
    ('overall_weighted', 'Weighted Overall'),
]
COLORS = {'Beauty': '#1f77b4', 'Games': '#ff7f0e',
          'MovieLens': '#2ca02c', 'Weighted Overall': '#000000'}
MARKERS = {'Beauty': 'o', 'Games': 's', 'MovieLens': '^', 'Weighted Overall': 'D'}


def rel_change(values):
    base = values[0]
    if base == 0 or base != base:
        return [float('nan')] * len(values)
    return [(v - base) / base * 100.0 if v == v else float('nan') for v in values]


def load(csv_path, weight_col):
    df = pd.read_csv(csv_path).sort_values(weight_col).reset_index(drop=True)
    return df, df[weight_col].tolist()


def main():
    ap = argparse.ArgumentParser(description='Combined 2x2 sensitivity figure (Fig.4)')
    ap.add_argument('--spec_csv', default='mypaper/specialization_weight_sweep/specialization_weight_domain_results.csv')
    ap.add_argument('--contr_csv', default='mypaper/contrastive_weight_sweep/contrastive_weight_domain_results.csv')
    ap.add_argument('--output_dir', default='Revise/Revise4/regenerated_figs/sensitivity')
    ap.add_argument('--out_name', default='sensitivity_combined')
    ap.add_argument('--font_size', type=int, default=17)
    ap.add_argument('--line_width', type=float, default=2.2)
    ap.add_argument('--marker_size', type=float, default=7.0)
    ap.add_argument('--figw', type=float, default=12.0)
    ap.add_argument('--figh', type=float, default=8.5)
    args = ap.parse_args()

    fs = args.font_size
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Liberation Sans', 'Arial', 'DejaVu Sans'],
        'mathtext.fontset': 'stixsans',
        'font.size': fs,
        'axes.titlesize': fs + 1,
        'axes.labelsize': fs + 1,
        'xtick.labelsize': fs,
        'ytick.labelsize': fs,
        'legend.fontsize': fs,
        'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
        'axes.grid': True, 'grid.alpha': 0.3, 'grid.linewidth': 0.6, 'grid.color': '#CCCCCC',
        'axes.edgecolor': '#333333',
    })

    spec_df, weights = load(args.spec_csv, 'specialization_weight')
    contr_df, _ = load(args.contr_csv, 'contrastive_weight')
    x = list(range(len(weights)))
    xlabels = [('0' if w == 0 else f'{w:g}') for w in weights]

    rows = [('Specialization Weight', spec_df), ('Contrastive Weight', contr_df)]
    metrics = ['NDCG@10', 'HR@10']

    fig, axes = plt.subplots(2, 2, figsize=(args.figw, args.figh))
    handles_labels = None
    for r, (wname, df) in enumerate(rows):
        for c, metric in enumerate(metrics):
            ax = axes[r, c]
            for dom_col, dom_name in DOMAIN_MAP:
                col = f'{dom_col}_{metric}'
                if col not in df.columns:
                    continue
                y = rel_change(df[col].tolist())
                emphasized = dom_name == 'Weighted Overall'
                ax.plot(
                    x, y,
                    marker=MARKERS[dom_name], color=COLORS[dom_name], label=dom_name,
                    linewidth=args.line_width + (0.8 if emphasized else 0.0),
                    markersize=args.marker_size + (1.0 if emphasized else 0.0),
                    zorder=6 if emphasized else 3,
                )
            ax.axhline(0, color='#888888', linewidth=0.9)
            ax.set_xticks(x)
            ax.set_xticklabels(xlabels)
            ax.set_title(f'{wname}: Δ{metric}', fontweight='bold')
            if c == 0:
                ax.set_ylabel('Change (%)')
            if handles_labels is None:
                handles_labels = ax.get_legend_handles_labels()

    fig.legend(*handles_labels, loc='lower center', ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=[0, 0.055, 1, 1])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        path = out_dir / f'{args.out_name}.{ext}'
        fig.savefig(path, dpi=600)
        print(f'saved: {path}')


if __name__ == '__main__':
    main()
