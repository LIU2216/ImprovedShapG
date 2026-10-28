"""
Run the full XAI benchmark for one dataset × model combination.

Usage
-----
    python run_benchmark.py --dataset abalone  --model lgb
    python run_benchmark.py --dataset diabetes --model mlp --n_runs 10
"""

import os

# Paper setting: rank deletion processes nodes in ASCENDING order of relevance to the target (least
# relevant first). shapG reads RANK_DESC when building rank-deletion graphs; its default "1" is descending.
os.environ["RANK_DESC"] = "0"

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import time
from pathlib import Path

import numpy as np

from benchmark.characteristic import build_char_func, build_shapg_custom_function
from benchmark.data import DATASET_META, load_dataset
from benchmark.explainers import run_explainers
from benchmark.kpi import calculate_kpi_for_dropping, compute_weighted_slope
from benchmark.plotting import (plot_feature_dropping, plot_ranking_heatmap,
                                 plot_slope_bar, plot_timing)


# ── Load existing results ─────────────────────────────────────────────────────

def _load_and_print(results_dir: Path) -> None:
    """Load results.json, print the summary table, and regenerate all plots."""
    with open(results_dir / "results.json") as f:
        data = json.load(f)

    dataset     = data["dataset"]
    model       = data["model"]
    kpi_label   = data["kpi"]
    n_features  = data["n_features"]
    initial_kpi = data["initial_kpi"]
    beta        = data.get("beta", 0.8)

    # Reconstruct explainer_results and dropping from saved data.
    # Feature scores are not stored; use descending rank as dummy values.
    explainer_results = {}
    dropping = {}
    for name, m in data["methods"].items():
        ranking = m["ranking"]
        n = len(ranking)
        explainer_results[name] = {
            "values":        [(f, n - i) for i, f in enumerate(ranking)],
            "duration": m["duration"],

        }
        dropping[name] = {
            "slope":   m["slope"],
            "metrics": m["metrics"],
        }

    print(f"\n{'='*60}")
    print(f"RESULTS  (initial {kpi_label} = {initial_kpi:.4f},  β = {beta})")
    print(f"[Loaded from {results_dir}]")
    print(f"{'='*60}")
    header = f"{'Method':<28}  {'Slope':>8}  {'Time(s)':>9}  Top-5 features"
    print(header)
    print("-" * len(header))
    for name, result in explainer_results.items():
        slope = dropping[name]["slope"]
        t     = result["duration"]
        top5  = "  ".join(f for f, _ in result["values"][:5])
        print(f"{name:<28}  {slope:>8.4f}  {t:>9.3f}  {top5}")

    print("\nGenerating plots...")
    plot_feature_dropping(dropping, initial_kpi,
                          dataset=dataset, model=model,
                          kpi_label=kpi_label, beta=beta,
                          out_dir=results_dir)
    plot_slope_bar(dropping, dataset=dataset, model=model,
                   kpi_label=kpi_label, out_dir=results_dir)
    plot_timing(explainer_results, dataset=dataset, model=model,
                out_dir=results_dir)
    plot_ranking_heatmap(explainer_results, dataset=dataset, model=model,
                         top_k=min(10, n_features), out_dir=results_dir)
    print(f"Done → {results_dir}/")


# ── Per-dataset method overrides ──────────────────────────────────────────────

_DATASET_DISABLED = {
    "adult": {"kernel_shap"},
    "H1N1":  {"kernel_shap"},
    "Churn": {"sampling_shap", "kernel_shap"},
}

# ── Config builder ────────────────────────────────────────────────────────────

def build_cfg(n_features: int, n_runs: int = 1, seed: int = 42,
              disabled: set = None, enabled: set = None, dataset: str = None) -> dict:
    """Build the experiment config dict programmatically.

    exact_shapley is automatically disabled for datasets with >15 features.
    disabled : set of method names to force-disable (on top of the above).
    """
    disabled = ((disabled or set()) | _DATASET_DISABLED.get(dataset or "", set())) - (enabled or set())
    all_methods = [
        "exact_cis", "exact_shapley", "shap_explainer",
        "sampling_shap", "kernel_shap", "pfi",
        "original_shapg", "sampling_owen",
        "improved_shapg_fixed", "improved_shapg_adaptive",
    ]
    explainers = {}
    for m in all_methods:
        if m in disabled:
            explainers[m] = {"enabled": False}
        elif m == "exact_shapley":
            explainers[m] = {"enabled": n_features <= 15}
        else:
            explainers[m] = {"enabled": True}
    return {"seed": seed, "n_runs": n_runs, "explainers": explainers}


# ── Feature-dropping evaluation ───────────────────────────────────────────────

def get_drop_limit(n_features: int) -> int:
    """Adaptive drop limit based on feature count.

    ≤ 10 features : drop until 1 remains  (n − 1 steps)
    11 – 25       : drop 10 features
    > 25          : drop 20 features

    Rationale: weighted slope weights decay as β^i (β=0.8), so
    contributions beyond step 10 (<11%) and step 20 (<1%) are negligible.
    Loop runs range(1, drop_limit + 1) so the actual drop count is drop_limit.
    """
    if n_features <= 10:
        return n_features - 1
    elif n_features <= 25:
        return 10
    else:
        return 20


def evaluate_feature_dropping(X, y, explainer_results: dict,
                               model_type: str, task: str,
                               test_size: float = 0.2,
                               random_state: int = 42,
                               beta: float = 0.8,
                               dataset: str = None,
                               n_repeats: int = 1) -> tuple:
    """Compute feature-dropping KPI curves and weighted slopes for every method.

    For each method, features are dropped one by one in the order of their
    importance ranking (most important first).  The model is retrained after
    each drop to measure the new KPI.

    Returns
    -------
    dropping_results : dict  {method_name: {"metrics": [...], "slope": float}}
    initial_kpi      : float  KPI with all features
    """
    all_features = X.columns.tolist()
    n = len(all_features)
    drop_limit = get_drop_limit(n)

    # Shared cache across all methods: {frozenset(remaining) → kpi}.
    # Not counted in per-method timing, so caching is safe and reduces
    # redundant retraining when methods share feature subsets in their
    # dropping sequences.
    kpi_cache: dict = {}

    def _kpi(feat_subset):
        key = frozenset(feat_subset)
        if key not in kpi_cache:
            kpi_cache[key] = calculate_kpi_for_dropping(
                X[list(feat_subset)], y, model_type, task, test_size,
                random_state, dataset=dataset, n_repeats=n_repeats,
            )
        return kpi_cache[key]

    initial_kpi = _kpi(all_features)

    dropping_results = {}
    for name, result in explainer_results.items():
        feature_rank = [f for f, _ in result["values"]]
        metrics = [initial_kpi]

        for i in range(1, drop_limit + 1):
            to_drop = set(feature_rank[:i])
            remaining = [f for f in all_features if f not in to_drop]
            metrics.append(_kpi(remaining) if remaining else 0.0)

        dropping_results[name] = {
            "metrics": metrics,
            "slope":   compute_weighted_slope(metrics, beta),
        }

    return dropping_results, initial_kpi


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import sys
    # Stream progress live (flush stdout per line) for tail -f visibility.
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(
        description="XAI benchmark — Improved ShapG paper"
    )
    parser.add_argument(
        "--dataset", required=True, choices=list(DATASET_META.keys()),
    )
    parser.add_argument(
        "--model", required=True, choices=["lgb", "mlp"],
    )
    parser.add_argument(
        "--n_runs", type=int, default=1,
        help="Number of runs per method (default: 1)",
    )
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--beta",      type=float, default=0.8,
                        help="Weighted-slope decay factor β (default: 0.8)")
    parser.add_argument(
        "--n_repeats_mlp", type=int, default=1,
        help="Number of retrain repeats per dropping step for MLP (averaged "
             "to reduce initialisation variance). Default: 1 (paper setting).",
    )
    parser.add_argument(
        "--results_dir", default="results",
        help="Unused, kept for compatibility",
    )
    parser.add_argument(
        "--disable", nargs="+", default=[],
        metavar="METHOD",
        help="Methods to skip, e.g. --disable sampling_shap kernel_shap",
    )
    parser.add_argument(
        "--enable", nargs="+", default=[],
        metavar="METHOD",
        help="Force-enable methods that are in _DATASET_DISABLED, e.g. --enable sampling_shap",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run even if results already exist",
    )
    parser.add_argument(
        "--rerun", nargs="+", default=[],
        metavar="METHOD",
        help="Force re-run specific methods even if already in saved results, "
             "e.g. --rerun improved_shapg_adaptive",
    )
    args = parser.parse_args()

    # ── Load dataset ──────────────────────────────────────────────────────
    X, y, task = load_dataset(args.dataset)
    n_features = X.shape[1]
    kpi_label  = "R²" if task == "regression" else "F1"

    # ── Determine what needs to run ───────────────────────────────────────
    cfg = build_cfg(n_features, args.n_runs, args.seed,
                    disabled=set(args.disable), enabled=set(args.enable),
                    dataset=args.dataset)
    requested = {name for name, ecfg in cfg["explainers"].items()
                 if ecfg.get("enabled", True)}

    bench_dir     = Path("results") / f"benchmark_{args.dataset}_{args.model}"
    existing_data = None
    if not args.force:
        results_path = bench_dir / "results.json"
        if results_path.exists():
            with open(results_path) as f:
                existing_data = json.load(f)

    already_done = set(existing_data["methods"].keys()) if existing_data else set()
    already_done -= set(args.rerun)   # force re-run the specified methods
    to_run = requested - already_done

    if not to_run:
        # If feature-dropping results are already present, just display them.
        # Otherwise fall through to recompute (e.g. interrupted after phase 1).
        if existing_data and all("slope" in m for m in existing_data["methods"].values()):
            _load_and_print(bench_dir)
            return

    if existing_data:
        print(f"Found existing results: {bench_dir}")
        print(f"  Cached  : {sorted(already_done)}")
        print(f"  To run  : {sorted(to_run)}")
        # Disable already-done methods so they are not re-run.
        for name in already_done:
            if name in cfg["explainers"]:
                cfg["explainers"][name] = {"enabled": False}

    print(f"\n{'='*60}")
    print(f"Dataset  : {args.dataset}  ({X.shape[0]} samples × {n_features} features)")
    print(f"Task     : {task}  →  KPI = {kpi_label}")
    print(f"Model    : {args.model}")
    n_repeats_drop = args.n_repeats_mlp if args.model == "mlp" else 1
    print(f"Seed     : {args.seed}   Test size: {args.test_size}   β: {args.beta}")
    if args.model == "mlp":
        print(f"Dropping : n_repeats={n_repeats_drop} (MLP averaged retraining)")
    print(f"Drop     : {get_drop_limit(n_features)} features")
    print(f"{'='*60}\n")

    _ORDER = [
        "exact_cis", "exact_shapley", "shap_explainer",
        "sampling_shap", "kernel_shap", "pfi",
        "original_shapg", "sampling_owen",
        "improved_shapg_fixed", "improved_shapg_adaptive",
    ]

    # ── Build characteristic functions ────────────────────────────────────
    char_func, X_train, X_test, y_train, y_test = build_char_func(
        X, y, args.model, task, args.test_size, args.seed,
        dataset=args.dataset,
    )
    shapg_custom_func, cache, cache_stats = build_shapg_custom_function(
        X, y, args.model, task, args.test_size, args.seed,
        dataset=args.dataset,
    )

    # ── Initialize output from existing data or fresh ─────────────────────
    bench_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "dataset":     args.dataset,
        "model":       args.model,
        "task":        task,
        "kpi":         kpi_label,
        "n_samples":   int(X.shape[0]),
        "n_features":  n_features,
        "seed":        args.seed,
        "test_size":   args.test_size,
        "beta":        args.beta,
        "initial_kpi": None,
        "methods": {
            name: {k: m[k] for k in ("ranking", "scores", "duration", "extra") if k in m}
            for name, m in (existing_data["methods"].items() if existing_data else {}.items())
        },
    }

    # ── Phase 1: run each missing method, checkpoint ranking + duration ───
    print("Running explainers...\n")
    t_wall_start = time.time()

    for name in _ORDER:
        if name not in to_run:
            continue

        cfg_one = {**cfg, "explainers": {
            n: (ecfg if n == name else {"enabled": False})
            for n, ecfg in cfg["explainers"].items()
        }}

        result = run_explainers(
            cfg_one, X, y,
            char_func, X_train, X_test, y_train, y_test,
            shapg_custom_func,
            model_type=args.model,
            task=task,
            dataset=args.dataset,
            results_dir=args.results_dir,
            shapg_cache=cache,
        )

        if name not in result:
            continue

        r = result[name]
        output["methods"][name] = {
            "ranking":       [f for f, _ in r["values"]],
            "scores":        [float(np.mean(np.abs(v))) if np.ndim(v) else float(v) for _, v in r["values"]],
            "duration": r["duration"],
            **({"extra": r["extra"]} if r.get("extra") else {}),
        }

        with open(bench_dir / "results.json", "w") as f:
            json.dump(output, f, indent=2)

    t_wall = time.time() - t_wall_start

    # ── Phase 2: feature dropping for ALL methods in checkpoint ───────────
    print("\nEvaluating feature-dropping curves...")

    all_explainer_results = {
        name: {
            "values":        [(f, len(m["ranking"]) - i) for i, f in enumerate(m["ranking"])],
            "duration": m["duration"],

        }
        for name, m in output["methods"].items()
    }

    dropping, initial_kpi = evaluate_feature_dropping(
        X, y, all_explainer_results,
        args.model, task,
        args.test_size, args.seed, args.beta,
        dataset=args.dataset,
        n_repeats=n_repeats_drop,
    )
    output["initial_kpi"] = initial_kpi
    for name, d in dropping.items():
        output["methods"][name]["slope"]   = d["slope"]
        output["methods"][name]["metrics"] = d["metrics"]

    with open(bench_dir / "results.json", "w") as f:
        json.dump(output, f, indent=2)

    # ── Sort into canonical order for print & plots ───────────────────────
    explainer_results = {k: all_explainer_results[k] for k in _ORDER if k in all_explainer_results}
    dropping          = {k: dropping[k]               for k in _ORDER if k in dropping}

    # ── Print summary table ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"RESULTS  (initial {kpi_label} = {initial_kpi:.4f},  β = {args.beta})")
    print(f"{'='*60}")
    header = f"{'Method':<28}  {'Slope':>8}  {'Time(s)':>9}  Top-5 features"
    print(header)
    print("-" * len(header))

    for name, result in explainer_results.items():
        slope = dropping[name]["slope"]
        t     = result["duration"]
        top5  = "  ".join(f for f, _ in result["values"][:5])
        print(f"{name:<28}  {slope:>8.4f}  {t:>9.3f}  {top5}")

    print(f"\nNew methods wall time : {t_wall:.1f}s")
    print(f"Cache (ShapG)         : {cache_stats['hits']} hits / "
          f"{cache_stats['misses']} misses")

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\nGenerating plots...")

    plot_feature_dropping(
        dropping, initial_kpi,
        dataset=args.dataset, model=args.model,
        kpi_label=kpi_label, beta=args.beta,
        out_dir=bench_dir,
    )
    plot_slope_bar(
        dropping,
        dataset=args.dataset, model=args.model,
        kpi_label=kpi_label, out_dir=bench_dir,
    )
    plot_timing(
        explainer_results,
        dataset=args.dataset, model=args.model,
        out_dir=bench_dir,
    )
    plot_ranking_heatmap(
        explainer_results,
        dataset=args.dataset, model=args.model,
        top_k=min(10, n_features),
        out_dir=bench_dir,
    )

    print(f"Results saved → {bench_dir}/")


if __name__ == "__main__":
    main()
