"""Plotting utilities for the XAI benchmark."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import gnuplot_style as gp
    gp.use("all")
except ImportError:
    pass


DISPLAY_NAMES = {
    "exact_cis":              "Exact CIS",
    "exact_shapley":          "Exact Shapley",
    "shap_explainer":         "SHAP",
    "sampling_shap":          "Sampling SHAP",
    "kernel_shap":            "Kernel SHAP",
    "pfi":                    "PFI",
    "original_shapg":         "Original ShapG",
    "sampling_owen":          "Sampling Owen",
    "improved_shapg_fixed":   "Improved ShapG (fixed)",
    "improved_shapg_adaptive": "Improved ShapG (adaptive)",
}

# Fixed legend / plotting order for the feature-dropping figure.
PLOT_ORDER = [
    "pfi", "shap_explainer", "sampling_shap", "kernel_shap",
    "exact_cis", "exact_shapley", "sampling_owen", "original_shapg",
    "improved_shapg_fixed", "improved_shapg_adaptive",
]

# Consistent (color, marker) per method so curves look identical across
# datasets and models.
METHOD_STYLE = {
    "pfi":                    {"color": "#1f77b4", "marker": "o"},
    "shap_explainer":         {"color": "#ff7f0e", "marker": "s"},
    "sampling_shap":          {"color": "#2ca02c", "marker": "^"},
    "kernel_shap":            {"color": "#d62728", "marker": "v"},
    "exact_cis":              {"color": "#9467bd", "marker": "D"},
    "exact_shapley":          {"color": "#8c564b", "marker": "P"},
    "original_shapg":         {"color": "#e377c2", "marker": "X"},
    "improved_shapg_fixed":   {"color": "#17becf", "marker": "*"},
    "improved_shapg_adaptive": {"color": "#bcbd22", "marker": "h"},
    "sampling_owen":          {"color": "#7f7f7f", "marker": "<"},
}

_FALLBACK_STYLE = {"color": "#000000", "marker": "."}

# Methods excluded from every benchmark figure.
EXCLUDED_METHODS = {"pfi"}


def _display(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)


def _ordered_methods(dropping: dict) -> list:
    """Methods present in ``dropping`` ordered per PLOT_ORDER, extras appended."""
    ordered = [n for n in PLOT_ORDER if n in dropping]
    ordered += [n for n in dropping if n not in PLOT_ORDER]
    return ordered


def _save(fig, out_dir: Path, stem: str, dpi: int = 300):
    for fmt in ["eps", "png"]:
        fig.savefig(out_dir / f"{stem}.{fmt}", format=fmt,
                    dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ── 1. Feature-dropping curves ────────────────────────────────────────────────

def plot_feature_dropping(dropping: dict, initial_kpi: float,
                          dataset: str, model: str, kpi_label: str,
                          beta: float, out_dir: Path, dpi: int = 300):
    """KPI vs number of features dropped, one curve per method."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for name in _ordered_methods(dropping):
        if name in EXCLUDED_METHODS:
            continue
        res     = dropping[name]
        metrics = res["metrics"]
        slope   = res["slope"]
        label   = f"{_display(name)}  (S={slope:.4f})"
        style   = METHOD_STYLE.get(name, _FALLBACK_STYLE)
        ax.plot(range(len(metrics)), metrics,
                label=label, linewidth=1.5, alpha=0.85,
                color=style["color"], marker=style["marker"],
                markersize=5)

    ax.set_xlabel("Number of Features Dropped", fontsize=12)
    ax.set_ylabel(kpi_label, fontsize=12)
    ax.set_title(f"Feature Dropping — {dataset}  [{model.upper()}, {kpi_label}, β={beta}]")
    ax.set_xticks(range(len(next(iter(dropping.values()))["metrics"])))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    plt.tight_layout()
    _save(fig, out_dir, "feature_dropping", dpi)


# ── 2. Weighted slope bar chart ────────────────────────────────────────────────

def plot_slope_bar(dropping: dict, dataset: str, model: str,
                   kpi_label: str, out_dir: Path, dpi: int = 300):
    """Horizontal bar chart of weighted slopes S, sorted descending."""
    names  = [n for n in dropping if n not in EXCLUDED_METHODS]
    slopes = [dropping[n]["slope"] for n in names]

    order  = np.argsort(slopes)[::-1]
    names  = [names[i] for i in order]
    slopes = [slopes[i] for i in order]
    labels = [_display(n) for n in names]

    fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.55)))
    colors = [METHOD_STYLE.get(n, _FALLBACK_STYLE)["color"] for n in names]
    bars = ax.barh(range(len(labels)), slopes, color=colors, alpha=0.85)

    # value labels
    for bar, val in zip(bars, slopes):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Weighted Slope S", fontsize=11)
    ax.set_title(f"Weighted Slope Comparison — {dataset}  [{model.upper()}, {kpi_label}]")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    _save(fig, out_dir, "slope_bar", dpi)


# ── 3. Runtime bar chart ───────────────────────────────────────────────────────

def plot_timing(explainer_results: dict, dataset: str, model: str,
                out_dir: Path, dpi: int = 300):
    """Horizontal bar chart of mean runtimes with std error bars."""
    names  = [n for n in explainer_results if n not in EXCLUDED_METHODS]
    means  = [explainer_results[n]["duration"] for n in names]
    stds   = [0.0] * len(names)

    order  = np.argsort(means)
    names  = [names[i]  for i in order]
    means  = [means[i]  for i in order]
    stds   = [stds[i]   for i in order]
    labels = [_display(n) for n in names]

    fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.55)))
    colors = [METHOD_STYLE.get(n, _FALLBACK_STYLE)["color"] for n in names]
    ax.barh(range(len(labels)), means, xerr=stds,
            alpha=0.8, capsize=3, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_title(f"Runtime Comparison — {dataset}  [{model.upper()}]")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    _save(fig, out_dir, "timing", dpi)


# ── 4. Feature ranking heatmap ─────────────────────────────────────────────────

def plot_ranking_heatmap(explainer_results: dict, dataset: str, model: str,
                         top_k: int = 10, out_dir: Path = None, dpi: int = 300):
    """Heatmap: rows = top-k features (by most-agreed rank), columns = methods.

    Cell value = rank assigned by that method (1 = most important).
    Darker = ranked higher (more important).
    """
    # Collect all unique features and their rank per method
    methods = [n for n in explainer_results if n not in EXCLUDED_METHODS]
    rank_data = {}
    for name in methods:
        res = explainer_results[name]
        for rank, (feat, _) in enumerate(res["values"], start=1):
            rank_data.setdefault(feat, {})[name] = rank

    # Select top_k features by median rank across methods
    all_feats = list(rank_data.keys())
    median_rank = {f: np.median(list(rank_data[f].values())) for f in all_feats}
    top_feats = sorted(all_feats, key=lambda f: median_rank[f])[:top_k]

    # Build matrix (features × methods)
    n_feat = len(top_feats)
    n_meth = len(methods)
    matrix = np.full((n_feat, n_meth), np.nan)
    for j, meth in enumerate(methods):
        for i, feat in enumerate(top_feats):
            matrix[i, j] = rank_data[feat].get(meth, np.nan)

    col_labels = [_display(m) for m in methods]

    fig, ax = plt.subplots(figsize=(max(8, n_meth * 1.2), max(4, n_feat * 0.5)))
    im = ax.imshow(matrix, aspect="auto", cmap="Blues_r",
                   vmin=1, vmax=len(all_feats))

    # Axis labels
    ax.set_xticks(range(n_meth))
    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(top_feats, fontsize=9)

    # Cell annotations
    for i in range(n_feat):
        for j in range(n_meth):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, str(int(val)), ha="center", va="center",
                        fontsize=7, color="white" if val <= 3 else "black")

    plt.colorbar(im, ax=ax, label="Rank (1 = most important)")
    ax.set_title(f"Feature Ranking Heatmap (top {top_k}) — "
                 f"{dataset}  [{model.upper()}]")
    plt.tight_layout()
    _save(fig, out_dir, "ranking_heatmap", dpi)
