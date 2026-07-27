# -*- coding: utf-8 -*-
"""
multi_domain_user_comparison.py
-------------------------------
Multi-domain representative user comparison analysis.

This script selects representative users from each domain and generates
a unified 2x2 comparison analysis showing:
1. Time-frequency patterns across domains
2. Decomposition comparison (original/low/high freq)
3. Embedding space visualization (t-SNE)
4. Representative users summary table

Usage:
    python scripts/multi_domain_user_comparison.py \
        --experiment_dir exp/e1 \
        --state_dict_path model.pth \
        --output_dir exp/multi_domain_comparison \
        --selection_strategy longest
"""
from __future__ import annotations

import os
import re
import ast
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Ensure project root on sys.path
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
import sys
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Project imports
from keys.utils import partition_multi_domain
from keys.model import SynRec
from keys.temporal_rating_modules import OptimizedFourierRatingEncoder

# Local library
from viz_lib import (
    build_rating_series,
    compute_scalar_fft_decomposition,
    compute_embedding_sequences,
    energy_ratio,
    auto_cutoff_80,
    find_top_peaks,
    r2_trend_plus_mean_consistent,
    resolve_cutoff_idx,
    save_metrics_json,
    apply_journal_style,
    get_user_domain_id,
)


def str2bool(v: str) -> bool:
    return str(v).lower() in ("true", "1", "yes", "y")


def select_representative_users(user_train: Dict[int, List[int]], 
                              user_to_domain: Dict[int, int], 
                              strategy: str = 'longest',
                              min_seq_len: int = 20) -> Dict[int, Dict]:
    """
    Select representative users from each domain.
    
    Args:
        user_train: User training sequences
        user_to_domain: User to domain mapping
        strategy: Selection strategy ('longest', 'median', 'random')
        min_seq_len: Minimum sequence length requirement
        
    Returns:
        Dict mapping domain_id to user info dict
    """
    representatives = {}
    domain_names = {0: 'Beauty', 1: 'Games', 2: 'MovieLens'}
    
    for domain_id in [0, 1, 2]:
        # Find users in this domain with sufficient sequence length
        domain_users = {
            uid: seq for uid, seq in user_train.items() 
            if get_user_domain_id(uid, user_to_domain, 0) == domain_id
            and len(seq) >= min_seq_len
        }
        
        if not domain_users:
            print(f"Warning: No users found in domain {domain_id} with min length {min_seq_len}")
            continue
        
        # Select representative user based on strategy
        if strategy == 'longest':
            best_uid, best_seq = max(domain_users.items(), key=lambda x: len(x[1]))
        elif strategy == 'median':
            lengths = [len(seq) for seq in domain_users.values()]
            median_len = np.median(lengths)
            best_uid, best_seq = min(domain_users.items(), 
                                   key=lambda x: abs(len(x[1]) - median_len))
        elif strategy == 'random':
            import random
            best_uid, best_seq = random.choice(list(domain_users.items()))
        else:
            raise ValueError(f"Unknown selection strategy: {strategy}")
        
        representatives[domain_id] = {
            'user_id': best_uid,
            'sequence': best_seq,
            'length': len(best_seq),
            'domain_name': domain_names[domain_id],
            'domain_id': domain_id
        }
        
        print(f"Selected representative user for {domain_names[domain_id]}: "
              f"ID={best_uid}, Length={len(best_seq)}")
    
    return representatives


def analyze_representative_user(user_info: Dict, model: SynRec, args) -> Dict:
    """
    Perform comprehensive analysis for a representative user.
    
    Returns:
        Dict containing all analysis results
    """
    user_id = user_info['user_id']
    sequence = user_info['sequence']
    domain_id = user_info['domain_id']
    
    # Build rating series
    rating_series = build_rating_series(sequence, maxlen=args.maxlen)
    
    if len(rating_series) < 10:
        print(f"Warning: User {user_id} has short rating series ({len(rating_series)})")
        return {}
    
    try:
        # FFT decomposition using user's own domain encoder
        freqs, amplitude, low_s, high_s, cutoff_ratio = compute_scalar_fft_decomposition(
            rating_series, model, domain_id
        )
        
        L = len(rating_series)
        half = L // 2
        amp_half = amplitude[:half]
        
        # Cutoff calculations
        kc_model = min(int(cutoff_ratio * L), half - 1)
        kc_80 = auto_cutoff_80(amp_half)
        chosen_k = resolve_cutoff_idx(L, amp_half, cutoff_ratio, 
                                    policy=args.cutoff_policy, fixed_k=args.fixed_k)
        
        # Energy fractions
        low_frac, high_frac = energy_ratio(amp_half, chosen_k)
        
        # Peak analysis
        peaks = find_top_peaks(amp_half, n=3, min_distance=3)
        k1 = int(peaks[0]) if len(peaks) > 0 else None
        T1 = float(L / k1) if k1 and k1 > 0 else None
        
        # R² decomposition quality
        r2_reg = r2_trend_plus_mean_consistent(rating_series, model, domain_id)
        
        # Embedding analysis
        pre_emb, long_emb, short_emb = compute_embedding_sequences(rating_series, model, domain_id)
        
        # Silhouette score
        try:
            from sklearn.metrics import silhouette_score
            X = np.vstack([pre_emb, long_emb, short_emb])
            labels = np.array(['Pre'] * pre_emb.shape[0] + 
                             ['Long'] * long_emb.shape[0] + 
                             ['Short'] * short_emb.shape[0])
            sil = float(silhouette_score(X, labels, metric='cosine'))
        except Exception as e:
            print(f"Warning: Silhouette calculation failed for user {user_id}: {e}")
            sil = float('nan')
        
        return {
            'user_info': user_info,
            'rating_series': rating_series,
            'fft_data': {
                'freqs': freqs,
                'amplitude': amplitude,
                'low_s': low_s,
                'high_s': high_s,
                'cutoff_ratio': cutoff_ratio,
            },
            'metrics': {
                'L': L,
                'kc_model': kc_model,
                'kc_80': kc_80,
                'chosen_k': chosen_k,
                'low_frac': low_frac,
                'high_frac': high_frac,
                'peak_k1': k1,
                'peak_T1': T1,
                'r2_reg': r2_reg,
                'silhouette': sil,
            },
            'embeddings': {
                'pre': pre_emb,
                'long': long_emb,  
                'short': short_emb,
            }
        }
        
    except Exception as e:
        print(f"Error analyzing user {user_id}: {e}")
        return {}


def generate_multi_domain_comparison_plot(analysis_results: Dict[int, Dict], args):
    """
    Generate unified 3x3 independent domain comparison plot with academic subplot labels.
    """
    apply_journal_style(args.journal_style)
    
    # Domain colors (consistent with batch analysis)
    domain_colors = {
        0: '#e74c3c',    # Beauty - Red
        1: '#2ecc71',    # Games - Green  
        2: '#3498db',    # MovieLens - Blue
    }
    
    domain_names = {0: 'Beauty', 1: 'Games', 2: 'MovieLens'}
    domains = sorted(analysis_results.keys())
    
    # Create 3x3 layout with optimized sizing for journal publication
    fig, axes = plt.subplots(3, 3, figsize=(20, 18), dpi=600)  # Increased size for better readability
    fig.suptitle('Multi-Domain Representative User Comparison', 
                fontsize=18, fontweight='bold', y=0.97)  # Adjusted title position
    
    # Define subplot labels for academic standard
    subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)']
    
    # Optimized spacing for better layout
    plt.subplots_adjust(hspace=0.35, wspace=0.3)  # Increased spacing for subplot labels
    
    # First row: Independent time-frequency analysis for each domain
    for i, domain_id in enumerate(domains):
        if domain_id in analysis_results:
            ax_timefreq = axes[0, i]
            _plot_independent_time_frequency(ax_timefreq, analysis_results[domain_id], 
                                           domain_colors[domain_id], domain_names[domain_id])
            # Add academic subplot label with improved positioning
            ax_timefreq.text(-0.15, 1.2, subplot_labels[i], transform=ax_timefreq.transAxes,
                            fontsize=16, fontweight='bold', va='top', ha='right')
    
    # Second row: Independent decomposition analysis for each domain
    for i, domain_id in enumerate(domains):
        if domain_id in analysis_results:
            ax_decomp = axes[1, i]
            _plot_independent_decomposition(ax_decomp, analysis_results[domain_id], 
                                          domain_colors[domain_id], domain_names[domain_id])
            # Add academic subplot label with improved positioning
            label_idx = i + 3  # Second row starts from index 3
            ax_decomp.text(-0.15, 1.2, subplot_labels[label_idx], transform=ax_decomp.transAxes,
                          fontsize=16, fontweight='bold', va='top', ha='right')
    
    # Third row: FFT magnitude spectrum comparison for each domain
    for i, domain_id in enumerate(domains):
        if domain_id in analysis_results:
            ax_spectrum = axes[2, i]
            _plot_independent_fft_spectrum(ax_spectrum, analysis_results[domain_id], 
                                         domain_colors[domain_id], domain_names[domain_id])
            # Add academic subplot label with improved positioning
            label_idx = i + 6  # Third row starts from index 6
            ax_spectrum.text(-0.15, 1.2, subplot_labels[label_idx], transform=ax_spectrum.transAxes,
                            fontsize=16, fontweight='bold', va='top', ha='right')
    
    # Note: Embedding analysis and summary moved to separate table generation
    # The text summary has been removed from the plot for journal publication standards
    # Use generate_domain_comparison_table() function to create LaTeX table instead
    
    # Add optimized unified legend with better positioning
    legend_elements = [Patch(facecolor=domain_colors[d], alpha=0.8, 
                           label=domain_names[d]) 
                      for d in domains if d in analysis_results]
    fig.legend(handles=legend_elements, loc='lower center', 
              bbox_to_anchor=(0.5, 0.02), ncol=len(domains), 
              fontsize=14, frameon=True, fancybox=True, shadow=True)
    
    # Save plot with adjusted layout (removed space for text summary)
    plt.tight_layout(rect=[0, 0.05, 1, 0.94])  # Reduced bottom margin from 0.1 to 0.05
    output_path = os.path.join(args.output_dir, 'multi_domain_comparison.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=600, 
               facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✅ Multi-domain comparison plot saved: {output_path}")
    return output_path


def _plot_multi_domain_time_frequency(ax, analysis_results: Dict, 
                                     domain_colors: Dict, domain_names: Dict):
    """Plot time-frequency analysis for all domains."""
    ax.set_title('Multi-Domain Time-Frequency Analysis', 
                fontsize=15, fontweight='bold')
    
    max_length = max(len(result['rating_series']) for result in analysis_results.values())
    
    for domain_id, result in analysis_results.items():
        if not result:
            continue
            
        rating_series = result['rating_series']
        color = domain_colors[domain_id]
        domain_name = domain_names[domain_id]
        user_id = result['user_info']['user_id']
        
        # Plot time series
        time_steps = np.arange(len(rating_series))
        ax.plot(time_steps, rating_series, 
               color=color, linewidth=2, alpha=0.8,
               label=f'{domain_name} (User {user_id})')
    
    ax.set_xlabel('Time Steps', fontsize=14)
    ax.set_ylabel('Rating Value', fontsize=14)
    ax.legend(loc='upper right', fontsize=13)
    ax.grid(True, alpha=0.3)


def _plot_multi_domain_decomposition(ax, analysis_results: Dict,
                                   domain_colors: Dict, domain_names: Dict):
    """Plot decomposition comparison for all domains."""
    ax.set_title('Multi-Domain Decomposition Comparison',
                fontsize=15, fontweight='bold')
    
    for domain_id, result in analysis_results.items():
        if not result:
            continue
            
        rating_series = result['rating_series']
        low_s = result['fft_data']['low_s']
        high_s = result['fft_data']['high_s']
        color = domain_colors[domain_id]
        domain_name = domain_names[domain_id]
        r2 = result['metrics']['r2_reg']
        
        time_steps = np.arange(len(rating_series))
        
        # Plot original, low-freq, high-freq
        ax.plot(time_steps, rating_series, color=color, linewidth=2, 
               alpha=0.8, linestyle='-', label=f'{domain_name} Original')
        ax.plot(time_steps, low_s, color=color, linewidth=1.5,
               alpha=0.6, linestyle='--', label=f'{domain_name} Low-freq')
        ax.plot(time_steps, high_s, color=color, linewidth=1,
               alpha=0.4, linestyle=':', label=f'{domain_name} High-freq')
        
        # Add R² annotation
        ax.text(0.02, 0.98 - domain_id * 0.05, f'{domain_name} R²: {r2:.3f}',
               transform=ax.transAxes, fontsize=13, color=color,
               verticalalignment='top')
    
    ax.set_xlabel('Time Steps', fontsize=14)
    ax.set_ylabel('Rating Value', fontsize=14)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=13)
    ax.grid(True, alpha=0.3)


def _plot_multi_domain_embeddings(ax, analysis_results: Dict,
                                 domain_colors: Dict, domain_names: Dict):
    """Plot embedding space comparison using t-SNE."""
    ax.set_title('Multi-Domain Embedding Space (t-SNE)',
                fontsize=15, fontweight='bold')
    
    try:
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
        
        # Collect all embeddings
        all_embeddings = []
        all_labels = []
        all_domains = []
        
        for domain_id, result in analysis_results.items():
            if not result:
                continue
                
            pre_emb = result['embeddings']['pre']
            long_emb = result['embeddings']['long']
            short_emb = result['embeddings']['short']
            
            all_embeddings.extend([pre_emb, long_emb, short_emb])
            all_labels.extend(['Pre', 'Long', 'Short'])
            all_domains.extend([domain_id] * 3)
        
        if len(all_embeddings) > 0:
            # Standardize embeddings
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(np.vstack(all_embeddings))
            
            # t-SNE
            tsne = TSNE(n_components=2, perplexity=min(10, len(X_scaled)-1), 
                       random_state=42)
            X_2d = tsne.fit_transform(X_scaled)
            
            # Plot points
            markers = {'Pre': 'o', 'Long': 's', 'Short': '^'}
            
            for i, (domain_id, label) in enumerate(zip(all_domains, all_labels)):
                color = domain_colors[domain_id]
                marker = markers[label]
                alpha = 0.8 if label == 'Pre' else 0.6
                
                ax.scatter(X_2d[i, 0], X_2d[i, 1], 
                          c=color, marker=marker, s=100, alpha=alpha,
                          edgecolors='black', linewidth=0.5)
            
            # Create custom legend
            legend_elements = []
            for domain_id in sorted(analysis_results.keys()):
                legend_elements.append(
                    plt.Line2D([0], [0], marker='o', color='w', 
                              markerfacecolor=domain_colors[domain_id], 
                              markersize=8, label=domain_names[domain_id])
                )
            ax.legend(handles=legend_elements, loc='upper right', fontsize=13)
            
        else:
            ax.text(0.5, 0.5, 'No embedding data available', 
                   ha='center', va='center', transform=ax.transAxes)
            
    except Exception as e:
        ax.text(0.5, 0.5, f'Embedding analysis failed:\n{str(e)}', 
               ha='center', va='center', transform=ax.transAxes, fontsize=14)
    
    ax.set_xlabel('t-SNE Component 1', fontsize=14)
    ax.set_ylabel('t-SNE Component 2', fontsize=14)
    ax.grid(True, alpha=0.3)


def _plot_users_summary_table(ax, analysis_results: Dict,
                             domain_colors: Dict, domain_names: Dict):
    """Plot representative users summary table."""
    ax.set_title('Representative Users Summary', fontsize=15, fontweight='bold')
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    headers = ['Domain', 'User ID', 'Seq Len', 'Low-Freq', 'R² Score', 'Silhouette']
    table_data = []
    row_colors = []
    
    # Header row
    table_data.append(headers)
    row_colors.append(['lightgray'] * len(headers))
    
    # Data rows
    for domain_id in sorted(analysis_results.keys()):
        result = analysis_results[domain_id]
        if not result:
            continue
            
        user_info = result['user_info']
        metrics = result['metrics']
        
        row = [
            domain_names[domain_id],
            str(user_info['user_id']),
            str(user_info['length']),
            f"{metrics['low_frac']:.3f}",
            f"{metrics['r2_reg']:.3f}",
            f"{metrics['silhouette']:.3f}" if not np.isnan(metrics['silhouette']) else 'N/A'
        ]
        
        table_data.append(row)
        domain_color = domain_colors[domain_id]
        row_colors.append([domain_color] + ['white'] * (len(headers) - 1))
    
    # Create table
    table = ax.table(cellText=table_data[1:],  # Data rows
                    colLabels=table_data[0],   # Headers
                    cellColours=row_colors[1:], # Data row colors
                    colColours=row_colors[0],   # Header colors
                    cellLoc='center',
                    loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Style the table
    for i in range(len(table_data)):
        for j in range(len(headers)):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_text_props(weight='bold')
            cell.set_edgecolor('black')
            cell.set_linewidth(0.5)


def _plot_independent_time_frequency(ax, result: Dict, domain_color: str, domain_name: str):
    """Plot independent time-frequency analysis for a single domain."""
    if not result:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', 
               transform=ax.transAxes, fontsize=15, style='italic')
        ax.set_title(f'{domain_name}\nTime-Frequency Analysis', 
                    fontsize=14, fontweight='bold')
        return
    
    rating_series = result['rating_series']
    user_id = result['user_info']['user_id']
    seq_length = len(rating_series)
    
    # Plot time series
    time_steps = np.arange(seq_length)
    ax.plot(time_steps, rating_series, color=domain_color, linewidth=2.5, alpha=0.9)
    
    # Highlight peaks if available
    if 'fft_data' in result:
        L = len(rating_series)
        half = L // 2
        amplitude = result['fft_data']['amplitude'][:half]
        
        # Find and mark top peaks in time domain
        try:
            from viz_lib import find_top_peaks
            peaks = find_top_peaks(amplitude, n=2, min_distance=3)
            if len(peaks) > 0:
                k1 = int(peaks[0])
                if k1 > 0:
                    T1 = L / k1  # Period
                    # Mark periodic points
                    for i in range(0, seq_length, int(T1)):
                        if i < seq_length:
                            ax.axvline(x=i, color=domain_color, alpha=0.3, linestyle='--')
        except Exception:
            pass
    
    # Formatting
    ax.set_xlabel('Time Steps', fontsize=14)
    ax.set_ylabel('Rating Value', fontsize=14)
    ax.set_title(f'{domain_name} (User {user_id})\nSeq Length: {seq_length}', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add key metrics as text
    if 'metrics' in result:
        metrics = result['metrics']
        info_text = f"Low-Freq: {metrics.get('low_frac', 0):.3f}"
        if metrics.get('peak_T1'):
            info_text += f"\nPeak Period: {metrics['peak_T1']:.1f}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
               fontsize=13, verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))


def _plot_independent_decomposition(ax, result: Dict, domain_color: str, domain_name: str):
    """Plot independent decomposition analysis for a single domain."""
    if not result or 'fft_data' not in result:
        ax.text(0.5, 0.5, 'No decomposition data\navailable', ha='center', va='center', 
               transform=ax.transAxes, fontsize=15, style='italic')
        ax.set_title(f'{domain_name}\nDecomposition Analysis', 
                    fontsize=14, fontweight='bold')
        return
    
    rating_series = result['rating_series']
    low_s = result['fft_data']['low_s']
    high_s = result['fft_data']['high_s']
    r2_reg = result['metrics'].get('r2_reg', 0)
    
    time_steps = np.arange(len(rating_series))
    
    # Plot original, low-freq, high-freq components
    ax.plot(time_steps, rating_series, color=domain_color, linewidth=2.5, 
           alpha=0.9, label='Original', linestyle='-')
    ax.plot(time_steps, low_s, color=domain_color, linewidth=2, 
           alpha=0.7, label='Low-Freq', linestyle='--')
    ax.plot(time_steps, high_s, color=domain_color, linewidth=1.5, 
           alpha=0.5, label='High-Freq', linestyle=':')
    
    # Formatting
    ax.set_xlabel('Time Steps', fontsize=14)
    ax.set_ylabel('Rating Value', fontsize=14)
    ax.set_title(f'{domain_name}\nR² Score: {r2_reg:.3f}', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=13)
    ax.grid(True, alpha=0.3)
    
    # Add decomposition quality info
    if 'metrics' in result:
        metrics = result['metrics']
        quality_text = f"Low-Freq Ratio: {metrics.get('low_frac', 0):.3f}\nHigh-Freq Ratio: {metrics.get('high_frac', 0):.3f}"
        ax.text(0.02, 0.02, quality_text, transform=ax.transAxes, 
               fontsize=13, verticalalignment='bottom',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))


def _plot_independent_fft_spectrum(ax, result: Dict, domain_color: str, domain_name: str):
    """Plot independent FFT magnitude spectrum for a single domain, following cli.py design."""
    if not result or 'fft_data' not in result:
        ax.text(0.5, 0.5, 'No FFT spectrum data\navailable', ha='center', va='center', 
               transform=ax.transAxes, fontsize=15, style='italic')
        ax.set_title(f'{domain_name}\nFFT Magnitude Spectrum', 
                    fontsize=14, fontweight='bold')
        return
    
    # Extract FFT data (same as cli.py approach)
    fft_data = result['fft_data']
    amplitude = fft_data['amplitude']
    freqs = fft_data['freqs']
    cutoff_ratio = fft_data['cutoff_ratio']
    
    # Get metrics for additional info
    metrics = result.get('metrics', {})
    L = len(amplitude)
    half = L // 2
    amp_half = amplitude[:half]
    freqs_half = freqs[:half]
    
    # Calculate cutoff indices (following cli.py logic)
    kc_model = min(int(cutoff_ratio * L), half - 1)
    
    # Calculate kc_80 (80% energy cutoff) - following cli.py
    try:
        from viz_lib import auto_cutoff_80
        kc_80 = auto_cutoff_80(amp_half)
    except Exception:
        kc_80 = kc_model
    
    # Use chosen_k if available, otherwise use kc_model
    chosen_k = metrics.get('chosen_k', kc_model)
    chosen_k = int(max(1, min(chosen_k, half - 1)))
    
    # Main amplitude plot (following cli.py style)
    ax.plot(freqs_half, amp_half, lw=1.5, label="Amplitude", color=domain_color)
    
    # 使用统一配色方案 - 从prop_cycle获取颜色 (same as cli.py)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors_list = prop_cycle.by_key()['color']
    
    # Highlight low/high frequency regions (following cli.py design)
    ax.axvspan(0, chosen_k, color=colors_list[0], alpha=0.15,
              label=f"Low-Freq Region (model)")
    ax.axvspan(chosen_k, half, color=colors_list[2], alpha=0.12,
              label="High-Freq Region")
    
    # Lines (following cli.py style)
    ax.axvline(kc_80, color=colors_list[4], ls="--", lw=1.0, 
              label=f"80% energy cutoff (k={kc_80})")
    ax.axvline(chosen_k, color=colors_list[0], ls="--", lw=1.0, 
              label=f"Chosen cutoff (k={chosen_k})")
    
    # Peaks with approximate periods (following cli.py implementation)
    try:
        from viz_lib import find_top_peaks
        peaks = find_top_peaks(amp_half, n=3, min_distance=3)
        for k in peaks:
            if k > 0 and k < len(amp_half):
                T = int(round(L / max(k, 1)))
                ax.axvline(k, color="#888888", ls=":", lw=0.8)
                ax.text(k + 0.5, amp_half[k], f"k={k}\n~{T} pts", 
                       fontsize=13, color="#444444")
    except Exception:
        pass
    
    # Secondary axis: normalized frequency (following cli.py)
    try:
        secax = ax.secondary_xaxis('top', functions=(lambda k: k / L, lambda f: f * L))
        secax.set_xlabel('Normalized frequency (k/N)', fontsize=13)
    except Exception:
        pass
    
    # Energy text box (following cli.py style)
    low_frac = metrics.get('low_frac', 0)
    high_frac = metrics.get('high_frac', 0)
    ax.text(0.02, 0.98, f"LowFrac={low_frac:.2f}\nHighFrac={high_frac:.2f}",
           transform=ax.transAxes, ha="left", va="top", fontsize=13,
           bbox=dict(boxstyle="round", fc="white", ec="#cccccc", alpha=0.8))
    
    # Formatting (following cli.py)
    ax.set_xlabel("Frequency Index", fontsize=14)
    ax.set_ylabel("Amplitude", fontsize=14)
    ax.set_title(f"{domain_name}\nFFT Magnitude Spectrum", 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Legend (following cli.py placement style)
    ax.legend(loc="upper right", fontsize=13)
    
    # Set reasonable axis limits
    ax.set_xlim(0, min(half, len(freqs_half)))
    if len(amp_half) > 0:
        ax.set_ylim(0, np.max(amp_half) * 1.1)


def generate_domain_comparison_table(analysis_results: Dict[int, Dict], output_dir: str, domain_names: Dict[int, str]):
    """
    Generate LaTeX table for domain comparison analysis to replace text summary.
    
    Args:
        analysis_results: Dictionary containing analysis results for each domain
        output_dir: Output directory to save the LaTeX table
        domain_names: Mapping from domain IDs to domain names
    
    Returns:
        Path to the generated LaTeX table file
    """
    domains = sorted(analysis_results.keys())
    
    # Prepare table data
    table_rows = []
    
    for domain_id in domains:
        result = analysis_results.get(domain_id, {})
        if not result:
            continue
            
        user_info = result.get('user_info', {})
        metrics = result.get('metrics', {})
        
        domain_name = domain_names.get(domain_id, f'Domain {domain_id}')
        user_id = user_info.get('user_id', 'N/A')
        seq_len = user_info.get('length', 0)
        low_frac = metrics.get('low_frac', 0)
        r2_score = metrics.get('r2_reg', 0)
        peak_k1 = metrics.get('peak_k1', 0)
        peak_T1 = metrics.get('peak_T1', 0)
        silhouette = metrics.get('silhouette', 0)
        
        peak_k_str = str(int(peak_k1)) if peak_k1 else 'N/A'
        peak_T_str = f"{peak_T1:.1f}" if peak_T1 else 'N/A'
        sil_str = f"{silhouette:.3f}" if not np.isnan(silhouette) else 'N/A'
        
        table_rows.append({
            'domain': domain_name,
            'user_id': str(user_id),
            'seq_len': str(seq_len),
            'low_frac': f"{low_frac:.3f}",
            'r2_score': f"{r2_score:.3f}",
            'peak_k': peak_k_str,
            'peak_T': peak_T_str,
            'silhouette': sil_str
        })
    
    # Generate LaTeX table
    latex_content = []
    latex_content.append("% Multi-Domain Representative User Comparison Table")
    latex_content.append("% This table replaces the text summary from the figure")
    latex_content.append("")
    latex_content.append("\\begin{table}[htbp]")
    latex_content.append("\\centering")
    latex_content.append("\\caption{Multi-Domain Representative User Analysis Summary}")
    latex_content.append("\\label{tab:multi_domain_comparison}")
    latex_content.append("\\begin{tabular}{l|c|c|c|c|c|c|c}")
    latex_content.append("\\hline")
    latex_content.append("\\textbf{Domain} & \\textbf{User ID} & \\textbf{Seq Len} & \\textbf{Low-Freq} & \\textbf{R² Score} & \\textbf{Peak k} & \\textbf{Period T} & \\textbf{Silhouette} \\\\")
    latex_content.append("\\hline")
    
    # Add data rows
    for row in table_rows:
        latex_row = f"{row['domain']} & {row['user_id']} & {row['seq_len']} & {row['low_frac']} & {row['r2_score']} & {row['peak_k']} & {row['peak_T']} & {row['silhouette']} \\\\"
        latex_content.append(latex_row)
    
    latex_content.append("\\hline")
    latex_content.append("\\end{tabular}")
    latex_content.append("\\begin{tablenotes}")
    latex_content.append("\\item[*] Low-Freq: Low-frequency energy fraction from FFT analysis")
    latex_content.append("\\item[*] R² Score: Decomposition quality measure (trend + mean)")
    latex_content.append("\\item[*] Peak k: Dominant frequency component index")
    latex_content.append("\\item[*] Period T: Corresponding temporal period in time steps")
    latex_content.append("\\item[*] Silhouette: Embedding separability score (cosine metric)")
    latex_content.append("\\end{tablenotes}")
    latex_content.append("\\end{table}")
    
    # Save to file
    table_path = os.path.join(output_dir, 'domain_comparison_table.tex')
    with open(table_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_content))
    
    print(f"✅ LaTeX table saved: {table_path}")
    return table_path


def main():
    parser = argparse.ArgumentParser(
        description="Multi-domain representative user comparison analysis"
    )
    
    # Data & model
    parser.add_argument('--datasets', nargs='+', default=None,
                       help='Datasets; default from experiment args.txt')
    parser.add_argument('--maxlen', type=int, default=100,
                       help='Max sequence length')
    parser.add_argument('--hidden_units', type=int, default=64,
                       help='Hidden units for rating embedding')
    parser.add_argument('--experiment_dir', type=str, required=True,
                       help='Path to training experiment dir')
    parser.add_argument('--state_dict_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    
    # Analysis parameters
    parser.add_argument('--selection_strategy', type=str, default='longest',
                       choices=['longest', 'median', 'random'],
                       help='Representative user selection strategy')
    parser.add_argument('--min_sequence_length', type=int, default=20,
                       help='Minimum sequence length for user selection')
    parser.add_argument('--cutoff_policy', type=str, default='model',
                       choices=['model', 'energy80', 'fixed_k'],
                       help='Frequency cutoff policy')
    parser.add_argument('--fixed_k', type=int, default=None,
                       help='Fixed cutoff index when policy=fixed_k')
    
    # Output
    parser.add_argument('--journal_style', type=str, default='custom',
                       choices=['nature', 'science', 'cell', 'high_quality', 'custom'])
    parser.add_argument('--output_dir', type=str, default='exp/multi_domain_comparison')
    parser.add_argument('--save_individual_analysis', type=str2bool, default=False,
                       help='Save individual user analysis details')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🚀 Starting Multi-Domain Representative User Comparison Analysis")
    print(f"   Experiment dir: {args.experiment_dir}")
    print(f"   Output dir: {args.output_dir}")
    print(f"   Selection strategy: {args.selection_strategy}")
    
    # Load training arguments
    args_file = os.path.join(args.experiment_dir, 'args.txt')
    if os.path.exists(args_file):
        print(f"📖 Loading training arguments from: {args_file}")
        with open(args_file, 'r') as f:
            training_args = {}
            for line in f:
                if ',' in line:
                    key, value = line.strip().split(',', 1)
                    training_args[key] = value
        
        if args.datasets is None:
            # Default datasets
            args.datasets = ['beauty_5_5', 'games_5_5', 'ml-1m_5_5']
        print(f"✅ Using datasets: {args.datasets}")
    else:
        print(f"⚠️ Args file not found: {args_file}")
        if args.datasets is None:
            args.datasets = ['beauty_5_5', 'games_5_5', 'ml-1m_5_5']
    
    # Load data
    print("📂 Loading multi-domain data...")
    user_train, user_valid, user_test, user_to_domain, n_users, n_items, domain_to_item_range = partition_multi_domain(
        fnames=args.datasets
    )
    print(f"📊 Dataset info: users={n_users}, items={n_items}, domains={len(args.datasets)}")
    
    # Load model
    print("🔧 Loading trained model...")
    device = torch.device('cpu')  # Use CPU for analysis to avoid device mismatch issues
    
    # Create model args object from training args
    class ModelArgs:
        def __init__(self):
            self.device = device
            self.hidden_units = args.hidden_units
            self.maxlen = args.maxlen
            self.dropout_rate = float(training_args.get('dropout_rate', 0.5))
            self.num_heads = int(training_args.get('num_heads', 2))
            self.num_blocks = int(training_args.get('num_blocks', 2))
            self.l2_emb = float(training_args.get('l2_emb', 0.0))
            # MoE parameters
            self.use_moe = True
            self.moe_num_experts = int(training_args.get('moe_num_experts', 4))
            self.moe_k = int(training_args.get('moe_k', 2))
            self.moe_routing_strategy = training_args.get('moe_routing_strategy', 'shared_base')
            self.moe_load_balancing = True
            self.moe_balance_loss_weight = float(training_args.get('moe_balance_loss_weight', 0.01))
            self.moe_noisy_gating = True
            # Rating embedding parameters
            self.use_rating_emb = True
            self.rating_strategy = training_args.get('rating_strategy', 'temporal_fourier')
            self.rating_pos_emb = False
            # Domain-specific parameters
            self.num_domains = len(args.datasets)
            self.domain_to_item_range = domain_to_item_range
            self.use_domain_info = True
            self.use_gated_fusion = True
            # Loss weights
            self.use_specialization_loss = True
            self.specialization_weight = float(training_args.get('specialization_weight', 0.01))
            self.use_contrastive_loss = True
            self.contrastive_weight = float(training_args.get('contrastive_weight', 0.01))
    
    model_args = ModelArgs()
    
    model = SynRec(
        user_num=n_users,
        item_num=n_items,
        args=model_args
    ).to(device)
    
    # Load state dict
    state = torch.load(args.state_dict_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    
    if not hasattr(model, 'enhanced_rating_module'):
        raise RuntimeError('Model lacks enhanced_rating_module.')
    
    print(f"Model architecture: {'Multi-domain' if hasattr(model.enhanced_rating_module, 'domain_encoders') else 'Single encoder'}")
    
    # Select representative users
    print("\n🎯 Selecting representative users...")
    representatives = select_representative_users(
        user_train, user_to_domain, 
        strategy=args.selection_strategy,
        min_seq_len=args.min_sequence_length
    )
    
    if len(representatives) == 0:
        raise RuntimeError("No representative users found!")
    
    # Save representative users info
    rep_info = {str(k): v for k, v in representatives.items()}  # JSON serializable
    with open(os.path.join(args.output_dir, 'representative_users.json'), 'w') as f:
        json.dump(rep_info, f, indent=2)
    
    # Analyze each representative user
    print("\n🔬 Analyzing representative users...")
    analysis_results = {}
    
    for domain_id, user_info in representatives.items():
        print(f"   Analyzing {user_info['domain_name']} user {user_info['user_id']}...")
        result = analyze_representative_user(user_info, model, args)
        if result:
            analysis_results[domain_id] = result
    
    if len(analysis_results) == 0:
        raise RuntimeError("No successful user analyses!")
    
    # Generate comparison plot
    print("\n🎨 Generating multi-domain comparison plot...")
    plot_path = generate_multi_domain_comparison_plot(analysis_results, args)
    
    # Generate LaTeX table to replace text summary
    print("\n📊 Generating LaTeX comparison table...")
    table_path = generate_domain_comparison_table(analysis_results, args.output_dir, 
                                                  {0: 'Beauty', 1: 'Games', 2: 'MovieLens'})
    
    # Save comparison metrics
    print("💾 Saving comparison metrics...")
    comparison_metrics = {}
    for domain_id, result in analysis_results.items():
        if result:
            user_info = result['user_info']
            metrics = result['metrics']
            comparison_metrics[str(domain_id)] = {
                'domain_name': user_info['domain_name'],
                'user_id': user_info['user_id'],
                'sequence_length': user_info['length'],
                'low_freq_fraction': metrics['low_frac'],
                'r2_decomposition': metrics['r2_reg'],
                'silhouette_score': metrics['silhouette'],
                'peak_period_T1': metrics['peak_T1'],
            }
    
    with open(os.path.join(args.output_dir, 'comparison_metrics.json'), 'w') as f:
        json.dump(comparison_metrics, f, indent=2)
    
    print(f"\n🎉 Multi-domain comparison analysis completed!")
    print(f"📁 Results saved in: {args.output_dir}")
    print(f"📊 Main plot: {plot_path}")
    print(f"📋 LaTeX table: {table_path}")
    print(f"📄 Use the LaTeX table in your paper to replace the previous text summary")


if __name__ == "__main__":
    main()
