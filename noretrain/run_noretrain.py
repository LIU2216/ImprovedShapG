"""
No-retraining (SHAP-style) game: Original ShapG, Improved ShapG (fixed), Improved ShapG (adaptive) and
Exact Shapley (N <= 15). SHAP needs no retraining and is reported from the retraining run.

For every explained sample x (X_test) the characteristic function is the fitted model's output with
the features outside S replaced by the Train mean,

    v_x(S) = f(x_S, mean(X_train)_{S-bar}),

using shapG's BatchedModelCharacteristic in per-sample mode. Global importance is mean_x |phi_i(x)|,
the same aggregation the SHAP methods use. At most 1,000 test rows are explained.

Everything else as in the retraining run: single 80/20 split (seed 42), models.py, ascending rank
deletion (RANK_DESC=0), ShapG depth 1, m 3, scale False, the same graphs. S is the retraining
feature-dropping slope, so it is comparable with the retraining run. Each method is timed from an
empty coalition cache.

    python run_noretrain.py --datasets admission --models lgb mlp

Needs ../retrain/results/benchmark_<dataset>_<model>/results.json (run the retraining benchmark first).
Output: results/<dataset>_<model>.json
"""

import os
import sys

os.environ["RANK_DESC"] = "0"                    # paper setting: ascending node order in rank deletion

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import random
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parent
CODE = ROOT.parent / "retrain"                   # shared benchmark code
sys.path.insert(0, str(CODE))

from benchmark.characteristic import build_char_func
from benchmark.data import DATASET_META, load_dataset
from benchmark.kpi import compute_weighted_slope
from benchmark.models import build_model
from shapG import ExactExplainer, GraphBuilder, ShapGExplainer
from shapG.characteristic.model_based import BatchedModelCharacteristic
from benchmark import adaptive as ADAPTIVE

RETRAIN = ROOT.parent / "retrain" / "results"   # retraining run of the same pairs
OUT = ROOT / "results"
SEED, ALPHA = 42, 0.8
EXACT_MAX_N = 15
MAX_EXPLAIN = 1000                               # explained test rows cap
REFERENCE = ["shap_explainer", "original_shapg", "improved_shapg_fixed", "improved_shapg_adaptive", "exact_shapley"]


# ── the SHAP-style game ───────────────────────────────────────────────────────

class OutputModel:
    """Model output as SHAP uses it: regression prediction, or P(y = 1)."""

    def __init__(self, model, features, task):
        self.model, self.features, self.task = model, features, task

    def predict(self, X):
        X = pd.DataFrame(X, columns=self.features)
        if self.task == "regression":
            return self.model.predict(X)
        return self.model.predict_proba(X)[:, 1]


class CachedOutputGame(BatchedModelCharacteristic):
    """BatchedModelCharacteristic returning per-sample outputs, with a coalition
    cache (the adaptive search re-visits coalitions across candidate graphs)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = {}

    def clear(self):
        self.cache.clear()

    def batch_compute(self, coalitions, context=None, chunk_size=None):
        keys = [frozenset(c) for c in coalitions]
        missing = list({k for k in keys if k not in self.cache})
        if missing:
            values = super().batch_compute([set(k) for k in missing], context, chunk_size)
            for k, val in zip(missing, values):
                self.cache[k] = np.asarray(val)
        return np.stack([self.cache[k] for k in keys])


# ── helpers ───────────────────────────────────────────────────────────────────

def drop_limit(n):
    return n - 1 if n <= 10 else (10 if n <= 25 else 20)


def global_importance(phi, features):
    """{node index → per-sample φ} → ranking and mean |φ| scores."""
    score = {features[i]: float(np.mean(np.abs(np.asarray(v)))) for i, v in phi.items()}
    ranking = sorted(score, key=score.get, reverse=True)
    return ranking, [score[f] for f in ranking]


def explain_shapg(G, game, features):
    Gi = nx.relabel_nodes(G, {f: i for i, f in enumerate(features)})
    random.seed(SEED)
    np.random.seed(SEED)                          # seeded before every ShapG run
    phi = ShapGExplainer(characteristic_function=game, depth=1, n_samples=3,
                         approximate_by_ratio=False, scale=False, verbose=False).fit_explain(Gi)
    return global_importance(phi, features)


def tau(a, b):
    pos = {f: i for i, f in enumerate(b)}
    return float(kendalltau([pos[f] for f in a], range(len(a)))[0])


# ── one dataset × model ───────────────────────────────────────────────────────

def run_pair(dataset, model_type, chunk_size):
    X, y, task = load_dataset(dataset)
    features = sorted(X.columns)
    N, K = len(features), drop_limit(len(features))
    M = N * (N - 1) // 2

    char_func, X_train, X_test, y_train, y_test = build_char_func(
        X, y, model_type, task, test_size=0.2, random_state=SEED, dataset=dataset)

    retrain_cache = {}

    def v_retrain(feats):                         # retraining KPI, for S
        key = frozenset(feats)
        if not key:
            return 0.0
        if key not in retrain_cache:
            retrain_cache[key] = float(char_func(list(key)))
        return retrain_cache[key]

    def dropping(ranking):
        """Retraining feature-dropping curve and its weighted slope S."""
        curve = [v_retrain(features)] + [v_retrain(ranking[k:]) for k in range(1, K + 1)]
        return float(compute_weighted_slope(curve, ALPHA)), curve

    def slope(ranking):
        return dropping(ranking)[0]

    model = build_model(model_type, task, dataset=dataset)
    model.fit(X_train[features], y_train)
    # explained rows: all of X_test, capped at MAX_EXPLAIN rows drawn without
    # replacement; S is still evaluated on all of X_test
    explain = np.arange(len(X_test))
    if len(X_test) > MAX_EXPLAIN:
        d_idx = list(DATASET_META).index(dataset)
        explain = np.sort(np.random.default_rng([6000, d_idx]).choice(len(X_test), MAX_EXPLAIN, replace=False))
    game = CachedOutputGame(
        model=OutputModel(model, features, task),
        X=X_test[features].to_numpy()[explain], y=y_test.to_numpy()[explain],
        metric_fn=lambda y_true, y_pred: np.asarray(y_pred),   # per-sample output → per-sample mode
        mask_value=X_train[features].mean().to_numpy(),
        chunk_size=chunk_size,
    )

    out = {"dataset": dataset, "model": model_type, "task": task, "n_features": N, "K": K,
           "n_train": len(X_train), "n_test": len(X_test), "n_explained": len(explain), "methods": {}}

    def record(name, ranking, scores, seconds, extra=None):
        S, curve = dropping(ranking)
        out["methods"][name] = {"ranking": ranking, "scores": scores, "S": S, "metrics": curve,
                                "seconds": seconds, "coalitions": len(game.cache), **(extra or {})}
        print(f"  {name:10s} S={out['methods'][name]['S']:.4f}  {seconds:7.1f}s  "
              f"coalitions={len(game.cache)}", flush=True)

    run_methods(X, y, task, dataset, features, N, M, game, record)
    return finish(out, dataset, model_type)


def run_methods(X, y, task, dataset, features, N, M, game, record):
    # Original ShapG: Kendall minimal-edge graph on raw X (as in the retraining run)
    game.clear()
    t0 = time.perf_counter()
    G = GraphBuilder.from_kendalltau_minimal_edge(X, reverse=True, version="v3")
    ranking, scores = explain_shapg(G, game, features)
    record("original", ranking, scores, time.perf_counter() - t0, {"n_edges": G.number_of_edges()})

    # Improved ShapG (fixed): round(3N) edges (average degree 6), cosine / AMI
    game.clear()
    t0 = time.perf_counter()
    random.seed(SEED); np.random.seed(SEED)       # as the retraining runner before every method
    target = min(round(3 * N), M)
    if task == "regression":
        G = GraphBuilder.from_rank_deletion(X, y, density_ratio=min((target + 0.5) / M, 1.0),
                                            correlation_method="cosine", similarity_method="cosine")
    else:                                         # AMI rank-deletion graph, as adaptive
        G = ADAPTIVE.ami_graph_with_edges(X, y, dataset, target)
    ranking, scores = explain_shapg(G, game, features)
    record("fixed", ranking, scores, time.perf_counter() - t0, {"n_edges": G.number_of_edges()})

    # Improved ShapG (adaptive) with this game: same path and stopping rule
    game.clear()
    t0 = time.perf_counter()
    graphs = ADAPTIVE.candidate_graphs(X, y, task, dataset)

    def explain(G):
        rk, sc = explain_shapg(G, game, features)
        return dict(zip(rk, sc))

    _, sc, info = ADAPTIVE.search(graphs, explain)
    ranking = sorted(sc, key=lambda f: -sc[f])
    record("adaptive", ranking, [sc[f] for f in ranking], time.perf_counter() - t0, info)

    # Exact Shapley of the same game (reference), N <= 15
    if N <= EXACT_MAX_N:
        game.clear()
        t0 = time.perf_counter()
        phi = ExactExplainer(characteristic_function=game).fit_explain(nx.complete_graph(N))
        ranking, scores = global_importance(phi, features)
        record("exact", ranking, scores, time.perf_counter() - t0)


def finish(out, dataset, model_type):
    # agreement with the exact game and with the retraining rankings
    ref = json.loads((RETRAIN / f"benchmark_{dataset}_{model_type}" / "results.json").read_text())["methods"]
    for name, rec in out["methods"].items():
        rec["tau_vs_exact_mask"] = tau(rec["ranking"], out["methods"]["exact"]["ranking"]) \
            if "exact" in out["methods"] else None
        rec["tau_vs_retrain"] = {m: tau(rec["ranking"], ref[m]["ranking"]) for m in REFERENCE if m in ref}
    out["retrain_S"] = {m: ref[m]["slope"] for m in REFERENCE if m in ref}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["admission", "ucs_scm"])
    ap.add_argument("--models", nargs="+", default=["lgb", "mlp"])
    ap.add_argument("--chunk", type=int, default=256, help="coalitions per model.predict() call")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets:
        for m in args.models:
            print(f"{ds} × {m}", flush=True)
            r = run_pair(ds, m, args.chunk)
            (OUT / f"{ds}_{m}.json").write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
