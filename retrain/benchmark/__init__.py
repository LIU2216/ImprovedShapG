"""Benchmark modules for Improved ShapG — LGB/MLP edition."""

from .data import load_dataset, DATASET_META
from .models import build_model
from .kpi import compute_kpi, compute_weighted_slope, calculate_kpi_for_dropping
from .characteristic import build_char_func, build_shapg_custom_function
from .explainers import run_explainers, EXPLAINER_REGISTRY
from .plotting import (plot_feature_dropping, plot_slope_bar,
                       plot_timing, plot_ranking_heatmap)
