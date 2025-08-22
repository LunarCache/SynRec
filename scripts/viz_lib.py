
# -*- coding: utf-8 -*-
"""
viz_lib.py
-----------------
Reusable utilities for user-behavior frequency analysis and visualization.

This module is *project-agnostic* except for two functions that accept a
trained Fourier encoder from your project. If the optional dependency
`visualization.journal_styles.apply_journal_style` is not available,
a lightweight default style will be used.
"""
from __future__ import annotations

import json
import math
from typing import Tuple, List, Optional, Dict

import numpy as np
import torch
import matplotlib.pyplot as plt

# Optional, project-specific style. Fall back to a sane default.
try:
    from visualization.journal_styles import apply_journal_style as _apply_style_impl  # type: ignore
except Exception:  # pragma: no cover
    _apply_style_impl = None

def apply_journal_style(name: str = "high_quality") -> None:
    """Apply a publication-ready style if available; otherwise use defaults."""
    if _apply_style_impl is not None:
        try:
            _apply_style_impl(name)
            return
        except Exception:
            pass
    # Fallback: a clean Matplotlib setup
    plt.rcParams.update({
        "figure.autolayout": False,
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 9,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
    })


# ----------------------------
# Data helpers
# ----------------------------
def get_domain_encoder(model, domain_id: Optional[int] = None):
    """
    获取指定领域的编码器，支持新的多领域架构和旧的单编码器架构
    
    Args:
        model: 训练好的模型
        domain_id: 领域ID，如果为None则尝试使用单编码器模式
    
    Returns:
        对应的编码器实例
    """
    if not hasattr(model, 'enhanced_rating_module'):
        raise RuntimeError('Model does not have enhanced_rating_module')
    
    rating_module = model.enhanced_rating_module
    
    # 检查是否为新的多领域架构
    if hasattr(rating_module, 'domain_encoders'):
        if domain_id is None:
            raise ValueError("domain_id is required for multi-domain model")
        domain_key = str(domain_id)
        if domain_key not in rating_module.domain_encoders:
            available_domains = list(rating_module.domain_encoders.keys())
            raise ValueError(f"Domain {domain_id} not found. Available domains: {available_domains}")
        return rating_module.domain_encoders[domain_key]
    
    # 回退到旧的单编码器架构
    elif hasattr(rating_module, 'fourier_encoder'):
        if domain_id is not None:
            print(f"Warning: domain_id={domain_id} specified but model uses single encoder")
        return rating_module.fourier_encoder
    
    else:
        raise RuntimeError('Model enhanced_rating_module lacks both domain_encoders and fourier_encoder')


def get_user_domain_id(user_id: int, user_to_domain: dict, default_domain: int = 0) -> int:
    """
    获取用户的领域ID
    
    Args:
        user_id: 用户ID
        user_to_domain: 用户到领域的映射字典
        default_domain: 默认领域ID
    
    Returns:
        用户的领域ID
    """
    return user_to_domain.get(user_id, default_domain)


def select_user_with_min_len(user_train: Dict[int, list], min_len: int = 20) -> Optional[int]:
    """Pick a user whose sequence length >= min_len; fallback to the longest one."""
    candidates = [(u, len(seq)) for u, seq in user_train.items() if len(seq) >= min_len]
    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    if not user_train:
        return None
    all_users = sorted(((u, len(seq)) for u, seq in user_train.items()), key=lambda x: x[1], reverse=True)
    return all_users[0][0]


def build_rating_series(user_sequence: list, maxlen: int = 100) -> np.ndarray:
    """Convert a user's sequence of (item, rating) into a 1D array with right trim/pad."""
    ratings = [float(r) for (_item, r) in user_sequence]
    if len(ratings) > maxlen:
        ratings = ratings[-maxlen:]
    return np.asarray(ratings, dtype=np.float32)


# ----------------------------
# FFT & decomposition (scalar)
# ----------------------------
def compute_scalar_fft_decomposition(
    rating_series: np.ndarray,
    model_or_encoder,
    domain_id: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Perform FFT on scalar rating series and reconstruct low/high frequency components
    using the encoder's learnable cutoff and soft boundary for consistency with model design.
    
    Args:
        rating_series: 1D rating sequence
        model_or_encoder: 模型实例或编码器实例
        domain_id: 领域ID，用于多领域模型
    
    Returns
    -------
    freqs : np.ndarray (L,)
    amplitude : np.ndarray (L,)
    low_series : np.ndarray (L,)
    high_series : np.ndarray (L,)
    cutoff_ratio : float
    """
    # 如果传入的是模型，获取对应的编码器
    if hasattr(model_or_encoder, 'enhanced_rating_module'):
        encoder = get_domain_encoder(model_or_encoder, domain_id)
    else:
        encoder = model_or_encoder  # 直接传入编码器的情况（向下兼容）
    
    # 使用编码器所在设备，避免CPU/GPU混用
    try:
        device = next(encoder.parameters()).device
    except Exception:
        device = torch.device("cpu")
    with torch.no_grad():
        x = torch.tensor(rating_series, dtype=torch.float32, device=device)
        L = x.shape[0]
        hann = torch.hann_window(L, device=device)
        xw = x * hann
        X = torch.fft.fft(xw, dim=0)

        low_mask, high_mask, cutoff = encoder._create_frequency_masks(L, device)  # type: ignore[attr-defined]
        X_low = X * low_mask
        X_high = X * high_mask

        x_low = torch.fft.ifft(X_low, dim=0).real
        x_high = torch.fft.ifft(X_high, dim=0).real

        amplitude = torch.abs(X)
        freqs = np.arange(L)

        cutoff_val = float(cutoff.detach().cpu()) if isinstance(cutoff, torch.Tensor) else float(cutoff)

        return freqs, amplitude.detach().cpu().numpy(), x_low.detach().cpu().numpy(), x_high.detach().cpu().numpy(), cutoff_val


# ----------------------------
# Embedding space pipeline
# ----------------------------
def compute_embedding_sequences(
    rating_series: np.ndarray,
    model_or_encoder,
    domain_id: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build three embedding sequences for t‑SNE visualization:
    - pre: rating embedding before frequency decomposition
    - long: long-term enhanced embedding (low-frequency branch + attention + enhancer)
    - short: short-term enhanced embedding (high-frequency branch + attention + enhancer)
    
    Args:
        rating_series: 1D rating sequence
        model_or_encoder: 模型实例或编码器实例
        domain_id: 领域ID，用于多领域模型
    """
    # 如果传入的是模型，获取对应的编码器
    if hasattr(model_or_encoder, 'enhanced_rating_module'):
        encoder = get_domain_encoder(model_or_encoder, domain_id)
    else:
        encoder = model_or_encoder  # 直接传入编码器的情况（向下兼容）
    
    # 使用编码器所在设备，避免CPU/GPU混用
    try:
        device = next(encoder.parameters()).device
    except Exception:
        device = torch.device('cpu')
    with torch.no_grad():
        r = torch.tensor(rating_series, dtype=torch.float32, device=device).unsqueeze(0)

        # Private util may vary across versions → robust fallback
        try:
            r_idx = encoder._z_score_normalize(r)  # type: ignore[attr-defined]
        except Exception:
            r_std = (r - r.mean()) / (r.std() + 1e-6)
            r_idx = torch.clamp((r_std + 3) / 6 * 5, 0, 5).long()

        rating_emb = encoder.rating_embedding(r_idx)  # (1, L, d)

        if getattr(encoder, "use_windowing", False):
            emb_win = encoder._apply_windowing(rating_emb)  # type: ignore[attr-defined]
        else:
            emb_win = rating_emb

        L = emb_win.size(1)
        F = torch.fft.fft(emb_win, dim=1)
        low_mask, high_mask, _ = encoder._create_frequency_masks(L, device)  # type: ignore[attr-defined]
        F_low = F * low_mask.view(1, L, 1)
        F_high = F * high_mask.view(1, L, 1)

        long_signal = torch.fft.ifft(F_low, dim=1).real
        short_signal = torch.fft.ifft(F_high, dim=1).real

        long_ctx, _ = encoder.long_term_attention(long_signal, long_signal, long_signal)
        long_enh = encoder.long_term_enhancer(long_ctx)

        short_ctx, _ = encoder.short_term_attention(short_signal, short_signal, short_signal)
        short_enh = encoder.short_term_enhancer(short_ctx)

        return (
            rating_emb.squeeze(0).detach().cpu().numpy(),
            long_enh.squeeze(0).detach().cpu().numpy(),
            short_enh.squeeze(0).detach().cpu().numpy(),
        )


# ----------------------------
# Metrics & helpers
# ----------------------------
def energy_ratio(amplitude_half: np.ndarray, kc: int) -> Tuple[float, float]:
    """Low/High energy ratio on one-sided spectrum (exclude DC at idx 0)."""
    amp2 = amplitude_half ** 2
    total = float(np.sum(amp2[1:]) + 1e-12)
    kc = max(1, int(kc))
    low = float(np.sum(amp2[1:kc + 1]))
    low_frac = low / total if total > 0 else 0.0
    return low_frac, 1.0 - low_frac


def auto_cutoff_80(amplitude_half: np.ndarray) -> int:
    """Index where cumulative energy (exclude DC) first reaches 80%."""
    # Handle degenerate cases (very short sequences)
    if amplitude_half is None or amplitude_half.size <= 1:
        return 1
    amp2 = amplitude_half ** 2
    cumsum = np.cumsum(amp2[1:])
    if cumsum.size == 0:
        return 1
    total = float(cumsum[-1]) + 1e-12
    target = 0.8 * total
    idx = int(np.searchsorted(cumsum, target)) + 1
    return max(1, idx)


def find_top_peaks(amplitude_half: np.ndarray, n: int = 3, min_distance: int = 3) -> List[int]:
    """Simple local maxima finder on one-sided spectrum (exclude DC)."""
    a = amplitude_half
    candidates = np.where((a[1:-1] > a[:-2]) & (a[1:-1] > a[2:]))[0] + 1
    candidates = sorted(candidates, key=lambda k: a[k], reverse=True)
    selected: List[int] = []
    for k in candidates:
        if all(abs(k - s) >= min_distance for s in selected):
            selected.append(int(k))
        if len(selected) >= n:
            break
    return selected


def r2_trend_plus_mean(y: np.ndarray, trend: np.ndarray) -> float:
    """Regression-style R^2 comparing y with (mean + trend)."""
    mu = float(np.mean(y))
    ss_tot = float(np.sum((y - mu) ** 2) + 1e-12)
    ss_res = float(np.sum((y - (trend + mu)) ** 2))
    return 1.0 - ss_res / ss_tot


def resolve_cutoff_idx(
    L: int,
    amp_half: np.ndarray,
    cutoff_ratio: float,
    policy: str = "model",
    fixed_k: Optional[int] = None
) -> int:
    """Resolve cutoff index per policy."""
    half = L // 2
    if policy == "model":
        return min(int(cutoff_ratio * L), half - 1)
    if policy == "energy80":
        return min(auto_cutoff_80(amp_half), half - 1)
    if policy == "fixed_k" and fixed_k is not None:
        return min(int(fixed_k), half - 1)
    return min(int(cutoff_ratio * L), half - 1)

def trend_from_raw_fft(y, model_or_encoder, domain_id: Optional[int] = None):
    """
    计算原始FFT的趋势分量
    
    Args:
        y: 输入序列
        model_or_encoder: 模型实例或编码器实例  
        domain_id: 领域ID，用于多领域模型
    """
    # 如果传入的是模型，获取对应的编码器
    if hasattr(model_or_encoder, 'enhanced_rating_module'):
        encoder = get_domain_encoder(model_or_encoder, domain_id)
    else:
        encoder = model_or_encoder  # 直接传入编码器的情况（向下兼容）
    
    # 使用编码器所在设备，避免CPU/GPU混用
    try:
        device = next(encoder.parameters()).device
    except Exception:
        device = torch.device('cpu')
    with torch.no_grad():
        x = torch.tensor(y, dtype=torch.float32, device=device)
        L = x.shape[0]
        X = torch.fft.fft(x, dim=0)              # raw FFT
        low_mask, _, _ = encoder._create_frequency_masks(L, device)
        X_low = X * low_mask
        X_low[0] = 0.0 + 0.0j                    # <<< 去掉 DC，避免与均值重复
        x_low0 = torch.fft.ifft(X_low, dim=0).real
        return x_low0.detach().cpu().numpy()

def r2_trend_plus_mean_consistent(y, model_or_encoder, domain_id: Optional[int] = None):
    """
    计算一致的R²指标
    
    Args:
        y: 输入序列
        model_or_encoder: 模型实例或编码器实例
        domain_id: 领域ID，用于多领域模型
    """
    y0 = y - np.mean(y)                          # 中心化
    trend0 = trend_from_raw_fft(y, model_or_encoder, domain_id)      # 零均值趋势
    ss_tot = float(np.sum(y0**2) + 1e-12)
    ss_res = float(np.sum((y0 - trend0)**2))
    return 1.0 - ss_res / ss_tot




# ----------------------------
# Plotting
# ----------------------------
def plot_time_and_spectrum(
    rating_series: np.ndarray,
    freqs: np.ndarray,
    amplitude: np.ndarray,
    cutoff_idx: int,
    kc_80: int,
    low_frac: float,
    high_frac: float,
    out_path: str,
    journal: str = "high_quality",
    label_policy: str = "model",
    peaks: Optional[List[int]] = None,
) -> None:
    """Plot time series and its FFT magnitude spectrum with marked regions."""
    apply_journal_style(journal)
    L = len(rating_series)
    fig, axes = plt.subplots(2, 1, figsize=(6.0, 4.5), dpi=300)

    # --- Time domain
    axes[0].plot(np.arange(L), rating_series, lw=1.5, label="Rating (time)")
    axes[0].set_xlabel("Position")
    axes[0].set_ylabel("Rating")
    axes[0].set_title("Time-domain Rating Sequence")
    axes[0].grid(True, alpha=0.3)

    # --- Frequency domain (one-sided)
    half = L // 2
    amp_half = amplitude[:half]
    axes[1].plot(freqs[:half], amp_half, lw=1.5, label="Amplitude")
    axes[1].set_xlabel("Frequency Index")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title("FFT Magnitude Spectrum")
    axes[1].grid(True, alpha=0.3)

    # 使用统一配色方案 - 从prop_cycle获取颜色
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors_list = prop_cycle.by_key()['color']
    
    cutoff_idx = int(max(1, min(cutoff_idx, half - 1)))
    axes[1].axvspan(0, cutoff_idx, color=colors_list[0], alpha=0.15,
                    label=f"Low-Freq Region ({label_policy})")
    axes[1].axvspan(cutoff_idx, half, color=colors_list[2], alpha=0.12,
                    label="High-Freq Region")
    # Lines
    axes[1].axvline(kc_80, color=colors_list[4], ls="--", lw=1.0, label=f"80% energy cutoff (k={kc_80})")
    axes[1].axvline(cutoff_idx, color=colors_list[0], ls="--", lw=1.0, label=f"Chosen cutoff (k={cutoff_idx})")

    # Peaks with approximate periods
    if peaks is None:
        peaks = find_top_peaks(amp_half, n=3, min_distance=3)
    for k in peaks:
        T = int(round(L / max(k, 1)))
        axes[1].axvline(k, color="#888888", ls=":", lw=0.8)
        axes[1].text(k + 0.5, amp_half[k], f"k={k}\n~{T} pts", fontsize=7, color="#444444")

    # Secondary axis: normalized frequency
    try:
        secax = axes[1].secondary_xaxis('top', functions=(lambda k: k / L, lambda f: f * L))
        secax.set_xlabel('Normalized frequency (k/N)')
    except Exception:
        pass

    # Legends
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    # Energy text box
    axes[1].text(0.98, 0.98, f"LowFrac={low_frac:.2f}\nHighFrac={high_frac:.2f}",
                 transform=axes[1].transAxes, ha="right", va="top", fontsize=8,
                 bbox=dict(boxstyle="round", fc="white", ec="#cccccc", alpha=0.8))

    fig.suptitle("Time vs Frequency Analysis (User Ratings)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _hf_rms_curve(high_series: np.ndarray, win: int) -> np.ndarray:
    win = max(5, int(win))
    return np.sqrt(np.convolve(high_series**2, np.ones(win) / win, mode="same"))


def plot_decomposition_overlay(
    rating_series: np.ndarray,
    low_series: np.ndarray,
    high_series: np.ndarray,
    r2_energy: Optional[float],
    r2_reg: Optional[float],
    out_path: str,
    journal: str = "high_quality"
) -> None:
    """Overlay original / low-freq / high-freq with HF-RMS and R² stats."""
    apply_journal_style(journal)
    L = len(rating_series)
    x = np.arange(L)

    fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.0), dpi=300)
    
    # 获取matplotlib当前的prop_cycle颜色
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors_list = prop_cycle.by_key()['color']
    
    # 明确指定颜色确保良好的区分度
    ax.plot(x, rating_series, lw=1.5, color=colors_list[0], label="Original (Rating)")
    ax.plot(x, low_series, lw=1.5, color=colors_list[5], label="Low-Freq (Trend)")  
    ax.plot(x, high_series, lw=1.0, color=colors_list[2], alpha=0.85, label="High-Freq (Variation)")
    ax.set_xlabel("Position")
    ax.set_ylabel("Value")

    title = "Time-domain Decomposition Overlay"
    if r2_energy is not None and r2_reg is not None:
        title += f" (R^2_energy={r2_energy:.2f}, R^2_reg={r2_reg:.2f})"
    elif r2_energy is not None:
        title += f" (R^2_energy={r2_energy:.2f})"
    elif r2_reg is not None:
        title += f" (R^2_reg={r2_reg:.2f})"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # 高频变化强度曲线 (使用独特颜色和样式)
    ax2 = ax.twinx()
    rms = _hf_rms_curve(high_series, win=max(5, L // 20))
    ax2.plot(x, rms, lw=1.2, color=colors_list[4], linestyle='--', alpha=0.8, label="Variation Intensity")
    ax2.set_ylabel("Variation Intensity")
    ax2.grid(False)

    # 将图例放置在图左侧，避免与图形内容重叠
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(lines + lines2, labels + labels2, 
              loc='center left', bbox_to_anchor=(-0.1, 0.5), 
              fontsize=8, borderaxespad=0.0)

    # 调整布局，为左侧图例留出空间
    fig.tight_layout()
    fig.subplots_adjust(left=0.25)  # 为左侧图例留出空间
    fig.savefig(out_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_tsne_embeddings(
    pre_emb: np.ndarray,
    long_emb: np.ndarray,
    short_emb: np.ndarray,
    out_path: str,
    journal: str = "high_quality",
    perplexity: int = 20,
    seed: int = 42
) -> Tuple[float, float]:
    """t‑SNE projection and separability metrics; returns (silhouette, logreg_acc)."""
    apply_journal_style(journal)

    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    L = pre_emb.shape[0]
    X = np.vstack([pre_emb, long_emb, short_emb])
    labels = np.array(["Pre"] * L + ["Long"] * L + ["Short"] * L)

    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, max(5, X.shape[0] // 4)),
        learning_rate=200,
        random_state=seed,
    )
    X2 = tsne.fit_transform(X)

    fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.5), dpi=300)
    
    # 获取matplotlib当前的prop_cycle颜色
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors_list = prop_cycle.by_key()['color']
    
    # 使用统一配色方案: 前3个颜色对应Pre, Long, Short
    colors = {"Pre": colors_list[0], "Long": colors_list[5], "Short": colors_list[2]}
    for c in ["Pre", "Long", "Short"]:
        mask = labels == c
        ax.scatter(X2[mask, 0], X2[mask, 1], s=12, alpha=0.75, c=colors[c],
                   label=c, edgecolors="white", linewidths=0.3)

    try:
        sil = float(silhouette_score(X, labels, metric="cosine"))
    except Exception:
        sil = float("nan")
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, labels, test_size=0.3, random_state=seed, stratify=labels
        )
        clf = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=1000))
        clf.fit(X_tr, y_tr)
        acc = float((clf.predict(X_te) == y_te).mean())
    except Exception:
        acc = float("nan")

    ax.set_title(f"t-SNE (Pre/Long/Short)\nSilhouette={sil:.2f}, LR-Acc={acc:.2f}")
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(out_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return sil, acc


def save_metrics_json(metrics: dict, output_dir: str, filename: str = "analysis_metrics.json") -> str:
    """Save metrics into JSON; return file path."""
    path = f"{output_dir.rstrip('/')}/{filename}"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return path
