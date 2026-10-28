"""XAI method implementations — LGB / MLP edition.

SHAP-based methods (shap_explainer, sampling_shap, kernel_shap):
  - Train the full model internally, then apply SHAP attribution.
  - LGB  → TreeExplainer (exact, fast).
  - MLP  → KernelExplainer on a background summary (approximate).

Coalition-based methods (exact_cis, exact_shapley, original_shapg, sampling_owen,
                         improved_shapg_fixed, improved_shapg_adaptive):
  - Use char_func / shapg_custom_func which retrain the model per coalition.
  - Model-type-agnostic at this layer.

Model-based non-Shapley methods (pfi):
  - Train the full model internally, then apply model-agnostic attribution.
"""

import json
import random
import time
from itertools import combinations
from math import factorial

import numpy as np
import shap
from sklearn.inspection import permutation_importance

from shapG import GraphBuilder, MultilinearExplainer, ShapGExplainer

from .models import build_model


# ── helpers ──────────────────────────────────────────────────────────────────

def _train(model_type, task, X_train, y_train, dataset=None):
    model = build_model(model_type, task, dataset=dataset)
    model.fit(X_train, y_train)
    return model


def _predict_fn(model, task):
    """Return a scalar-output predict callable suitable for SHAP."""
    if task == "regression":
        return model.predict
    # Binary classification: probability of positive class
    return lambda X: model.predict_proba(X)[:, 1]


# ── SHAP-based explainers ─────────────────────────────────────────────────────

def run_shap_explainer(X, y, char_func, model_type, task,
                       X_train, X_test, y_train, y_test,
                       dataset=None, **kwargs):
    """Native SHAP explainer.

    LGB  → TreeExplainer (exact Shapley for tree ensembles).
    MLP  → PermutationExplainer (full train background, full test set).
    """
    features = X_train.columns.tolist()
    model = _train(model_type, task, X_train, y_train, dataset=dataset)

    if model_type == "lgb":
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_test)
        # Binary LGB may return list [neg_class, pos_class]; take pos_class
        if isinstance(sv, list):
            sv = sv[1]
    else:  # mlp — PermutationExplainer (full train background, full test set)
        explainer = shap.PermutationExplainer(_predict_fn(model, task), X_train)
        sv = explainer.shap_values(X_test)
        if isinstance(sv, list):
            sv = sv[1]

    importance = np.abs(np.array(sv)).mean(axis=0)
    return sorted(zip(features, importance), key=lambda x: x[1], reverse=True)


def run_sampling_shap(X, y, char_func, model_type, task,
                      X_train, X_test, y_train, y_test,
                      dataset=None, **kwargs):
    """Sampling SHAP (model-agnostic, uses predict function)."""
    features = X_train.columns.tolist()
    model = _train(model_type, task, X_train, y_train, dataset=dataset)
    predict = _predict_fn(model, task)
    explainer = shap.SamplingExplainer(lambda X: predict(X), X_train)
    sv = explainer.shap_values(X_test)
    importance = np.abs(np.array(sv)).mean(axis=0)
    return sorted(zip(features, importance), key=lambda x: x[1], reverse=True)


def run_kernel_shap(X, y, char_func, model_type, task,
                    X_train, X_test, y_train, y_test,
                    dataset=None, **kwargs):
    """Kernel SHAP (model-agnostic, uses predict function, full data)."""
    features = X_train.columns.tolist()
    model = _train(model_type, task, X_train, y_train, dataset=dataset)
    predict = _predict_fn(model, task)
    explainer = shap.KernelExplainer(lambda X: predict(X), X_train)
    sv = explainer.shap_values(X_test)
    if isinstance(sv, list):
        sv = sv[1]
    importance = np.abs(np.array(sv)).mean(axis=0)
    return sorted(zip(features, importance), key=lambda x: x[1], reverse=True)


# ── Coalition-based explainers ────────────────────────────────────────────────

def run_exact_cis(X, y, char_func, **kwargs):
    """CIS (Center of Imputation Set) — O(n+1) coalition evaluations."""
    features = X.columns.tolist()
    grand = char_func(features)
    individual = {f: char_func([f]) for f in features}
    surplus = grand - sum(individual.values())
    share = surplus / len(features)
    cis = {f: individual[f] + share for f in features}
    return sorted(cis.items(), key=lambda x: x[1], reverse=True)


def run_exact_shapley(X, y, char_func, **kwargs):
    """Exact Shapley Values — O(2^n) coalition evaluations.

    Only run on datasets with ≤15 features (enforced by config).
    A local cache ensures each unique coalition is evaluated only once
    (reducing calls from n·2^n to 2^n).
    """
    features = X.columns.tolist()
    n = len(features)
    _cache = {}

    def _cf(feats):
        key = frozenset(feats)
        if key not in _cache:
            _cache[key] = char_func(feats)
        return _cache[key]

    sv = {}
    for feat in features:
        val = 0.0
        others = [f for f in features if f != feat]
        for size in range(n):
            for S in combinations(others, size):
                S = list(S)
                weight = factorial(len(S)) * factorial(n - len(S) - 1) / factorial(n)
                val += weight * (_cf(S + [feat]) - _cf(S))
        sv[feat] = val
    return sorted(sv.items(), key=lambda x: x[1], reverse=True)


def run_pfi(X, y, char_func, model_type, task,
            X_train, X_test, y_train, y_test,
            n_repeats: int = 10, dataset=None, **kwargs):
    """Permutation Feature Importance (PFI).

    Trains the full model on X_train, then measures the mean KPI drop on X_test
    when each feature column is randomly permuted n_repeats times.
    Higher drop → more important feature.
    """
    features = X_train.columns.tolist()
    model = _train(model_type, task, X_train, y_train, dataset=dataset)
    scoring = "r2" if task == "regression" else "f1"
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats,
        scoring=scoring,
        random_state=42,
    )
    importance = result.importances_mean
    return sorted(zip(features, importance), key=lambda x: x[1], reverse=True)


def run_original_shapg(X, y, char_func, shapg_custom_func=None, **kwargs):
    """Original ShapG on a minimal-edge graph built from Kendall-tau.

    NOT a spanning tree: create_minimal_edge_graph adds edges in descending |tau|
    with no cycle check, stopping only once every node is attached, so the edge
    count is driven by the worst-connected feature and far exceeds N-1
    (e.g. admission 13 edges vs 6; crimedata 4003 vs 120).

    Note the association measure is Kendall-tau, the library default, whereas the
    published ShapG specifies Pearson.
    """
    G = GraphBuilder.from_kendalltau_minimal_edge(X, reverse=True, version="v3")
    explainer = ShapGExplainer(
        characteristic_function=shapg_custom_func,
        depth=1, n_samples=3,
        approximate_by_ratio=False, scale=False, verbose=False,
    )
    sv = explainer.fit_explain(G)
    return sorted(sv.items(), key=lambda x: x[1], reverse=True)


def run_sampling_owen(X, y, char_func, shapg_custom_func=None,
                      seed: int = 42, **kwargs):
    """Sampling Owen Values via MultilinearExplainer (TabM, inverse_coalition leverage).

    Uses the same Kendall-tau minimal-edge graph as Original ShapG (not a spanning
    tree -- see run_original_shapg).
    """
    G = GraphBuilder.from_kendalltau_minimal_edge(X, reverse=True, version="v3")
    explainer = MultilinearExplainer(
        characteristic_function=shapg_custom_func,
        leverage_type="inverse_coalition",
        seed=seed,
        verbose=False,
    )
    owen_vals = explainer.fit_explain(G)
    return sorted(owen_vals.items(), key=lambda x: x[1], reverse=True)


def run_improved_shapg_fixed(X, y, char_func, shapg_custom_func=None, **kwargs):
    """Improved ShapG (fixed): one ShapG run on the rank-deletion graph with round(3N) edges
    (average degree 6, capped at the complete graph). Edge weights: cosine (regression), AMI
    (classification, the same AMI rank-deletion path as Improved ShapG (adaptive))."""
    from .adaptive import ami_graph_with_edges

    task, n_features = kwargs.get("task"), X.shape[1]
    max_edges = n_features * (n_features - 1) // 2
    target_edges = min(round(3 * n_features), max_edges)
    if task == "classification":
        G = ami_graph_with_edges(X, y, kwargs.get("dataset"), target_edges)
    else:
        # density_ratio = target / max_edges; the +0.5 offsets from_rank_deletion's int() truncation
        G = GraphBuilder.from_rank_deletion(
            X, y, density_ratio=min((target_edges + 0.5) / max_edges, 1.0),
            correlation_method="cosine", similarity_method="cosine",
        )
    explainer = ShapGExplainer(
        characteristic_function=shapg_custom_func,
        depth=1, n_samples=3,
        approximate_by_ratio=False, scale=False, verbose=False,
    )
    return sorted(explainer.fit_explain(G).items(), key=lambda x: x[1], reverse=True)


# ── Registry ──────────────────────────────────────────────────────────────────

from .adaptive import run_improved_shapg_adaptive        # noqa: E402

EXPLAINER_REGISTRY = {
    # name → (function, is_stochastic)
    "exact_cis":               (run_exact_cis,               False),
    "exact_shapley":           (run_exact_shapley,           False),
    "shap_explainer":          (run_shap_explainer,          False),
    "sampling_shap":           (run_sampling_shap,           True),
    "kernel_shap":             (run_kernel_shap,             True),
    "pfi":                     (run_pfi,                     True),
    "original_shapg":          (run_original_shapg,          False),
    "sampling_owen":           (run_sampling_owen,           True),
    "improved_shapg_fixed":    (run_improved_shapg_fixed,    False),
    "improved_shapg_adaptive": (run_improved_shapg_adaptive, False),
}


def run_explainers(cfg, X, y, char_func, X_train, X_test, y_train, y_test,
                   shapg_custom_func, model_type: str, task: str,
                   dataset: str = None, results_dir: str = "results",
                   shapg_cache: dict = None):
    """Run all enabled explainers.

    Parameters
    ----------
    dataset     : dataset name (used by the AMI graphs of the Improved ShapG variants).
    results_dir : unused, kept for interface compatibility.

    Returns
    -------
    dict : {method_name: {"values", "all_values", "durations",
                          "duration",
                          "extra" (stopping information of Improved ShapG (adaptive)), ...}}
    """
    explainer_cfg = cfg["explainers"]
    n_features = X.shape[1]
    base_seed = cfg.get("seed", 42)
    n_runs = cfg.get("n_runs", 1)
    seeds = [base_seed + i for i in range(n_runs)]
    results = {}

    for name, ecfg in explainer_cfg.items():
        if not ecfg.get("enabled", True):
            continue
        if name not in EXPLAINER_REGISTRY:
            print(f"[WARNING] Unknown explainer: {name}, skipping.")
            continue

        if name == "exact_shapley" and n_features > 15:
            print(f"  {name}: skipped (n_features={n_features} > 15)")
            continue

        func, is_stochastic = EXPLAINER_REGISTRY[name]
        params = dict(ecfg.get("params", {}) or {})

        # side_channel lets a method pass back extra information (adaptive: where it stopped)
        side_channel: dict = {}

        common_kwargs = dict(
            char_func=char_func,
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test,
            shapg_custom_func=shapg_custom_func,
            model_type=model_type,
            task=task,
            dataset=dataset,
            results_dir=results_dir,
            shapg_cache=shapg_cache,
            _side_channel=side_channel,
            **params,
        )

        all_values, all_durations = [], []
        run_count = n_runs

        for run_idx in range(run_count):
            np.random.seed(seeds[run_idx])
            random.seed(seeds[run_idx])
            label = f"  {name} (run {run_idx+1}/{run_count})" if run_count > 1 else f"  {name}"
            print(f"{label}...", end=" ", flush=True)

            side_channel.clear()
            # Real standalone timing: reset the shared ShapG coalition cache
            # before every method so no method freeloads coalitions another
            # method already computed. Within-method caching (e.g. across the
            # graphs of the adaptive search) still applies.
            if shapg_cache is not None:
                shapg_cache.clear()
            t0 = time.time()
            values = func(X, y, **common_kwargs)
            elapsed = time.time() - t0

            total_elapsed = elapsed

            all_values.append(values)
            all_durations.append(total_elapsed)

            print(f"{total_elapsed:.3f}s")

        results[name] = {
            "values": all_values[0],
            "all_values": all_values,
            "durations": all_durations,
            "duration": float(all_durations[0]),
            "is_stochastic": is_stochastic,
            "n_runs": run_count,
            "extra": dict(side_channel),
        }

    return results
