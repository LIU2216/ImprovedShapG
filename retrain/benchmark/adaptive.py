"""Improved ShapG (adaptive).

Edge weights: cosine similarity (regression) or adjusted mutual information, AMI (classification).
Search: start from the sparsest connected rank-deletion graph (a spanning tree) and add the deleted
edges back in reverse order, floor(n_add / 5) edges per step, up to average degree 10
(E_max = min(5N, floor(0.9 N(N-1)/2))). ShapG is run on every graph of the path. Stop at the first
window of 3 consecutive graphs whose pairwise Kendall tau_b (scores aligned by feature, ties kept,
undefined tau = not stable) are all >= 0.95 and return the window's last graph; otherwise return the
cap graph.

The path and the stopping rule do not depend on the game, so the same functions serve the
retraining game (run_improved_shapg_adaptive) and the no-retraining game (noretrain/).
Also provides ami_graph_with_edges, used by Improved ShapG (fixed) on classification.
"""
import math
import random
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import adjusted_mutual_info_score

K_BINS, MAX_ROWS = 10, 10_000
NOMINAL = {"H1N1": {"employment_industry", "employment_occupation"}, "adult": {"occupation"}}
SEED, WINDOW, TAU0 = 42, 3, 0.95


def _tie_respecting_bins(col: pd.Series, k: int) -> np.ndarray:
    """Near-equal-frequency binning that never splits a group of equal values: cuts are allowed only
    between two distinct values; for each target quantile 1/k, ..., (k-1)/k the cut whose cumulative
    share of rows is closest is chosen (ties -> the smaller cut), duplicates merged. Independent of the
    row order; a column with >= 2 distinct values always gets >= 2 bins."""
    values, counts = np.unique(col.to_numpy(), return_counts=True)
    cum = np.cumsum(counts)[:-1] / counts.sum()          # share of rows up to and incl. values[i], i < last
    cuts = sorted({int(np.argmin(np.abs(cum - t))) for t in np.arange(1, k) / k}) if len(cum) else []
    upper = values[cuts]                                  # a value v goes to bin = #cuts with upper < v
    return np.searchsorted(upper, col.to_numpy(), side="left")


def _discretize(col: pd.Series, nominal: bool) -> np.ndarray:
    """Nominal features and features with <= K_BINS distinct values keep their labels; other features
    are cut into at most K_BINS near-equal-frequency bins that keep equal values together."""
    if nominal or col.nunique() <= K_BINS:
        return col.to_numpy()
    return _tie_respecting_bins(col, K_BINS)


def _ami_inputs(X: pd.DataFrame, y: pd.Series, dataset: str):
    """Raw (untruncated) AMI edge weights and AMI node relevance on <= MAX_ROWS rows drawn uniformly at
    random (seed 0). Only their order is used (rank deletion); ShapG ignores edge weights."""
    rows = np.arange(len(X))
    if len(X) > MAX_ROWS:
        rows = np.sort(np.random.default_rng(0).choice(len(X), MAX_ROWS, replace=False))
    Xs, ys = X.iloc[rows], y.iloc[rows].to_numpy()
    D = {c: _discretize(Xs[c], c in NOMINAL.get(dataset, set())) for c in X.columns}
    # A column without variation in these rows carries no association: its scores are set to 0 instead
    # of sklearn's clustering convention AMI(constant, constant) = 1.
    const = {c for c in X.columns if len(np.unique(D[c])) < 2}

    def score(u, v, cu, cv):
        return 0.0 if (cu or cv) else float(adjusted_mutual_info_score(u, v))

    W = {(a, b): score(D[a], D[b], a in const, b in const) for a, b in combinations(X.columns, 2)}
    r = {c: score(D[c], ys, c in const, len(np.unique(ys)) < 2) for c in X.columns}
    return W, r


def _ami_deletion_order(features, W, r):
    """rank deletion with AMI weights and AMI node order (ascending relevance, ties by name)"""
    G = nx.Graph()
    for (a, b), w in W.items():
        G.add_edge(a, b, weight=w)
    deleted = []
    for node in sorted(features, key=lambda f: (r[f], f)):
        for u, v in sorted(G.edges(node), key=lambda e: (-G[e[0]][e[1]]["weight"], tuple(sorted(e)))):
            w = G[u][v]["weight"]
            G.remove_edge(u, v)
            if nx.is_connected(G):
                deleted.append((u, v, w))
            else:
                G.add_edge(u, v, weight=w)
    return G, deleted


def e_max(n):
    return min(5 * n, math.floor(0.9 * n * (n - 1) / 2))


def candidate_graphs(X, y, task, dataset):
    """Candidate path: sparsest connected graph, deleted edges added back in reverse order,
    floor(n_add/5) per step up to E_max."""
    from .graph_order import get_deletion_order
    np.random.seed(SEED); random.seed(SEED)
    if task == "regression":
        _, sparse, deleted = get_deletion_order(X, y, corr_method="cosine", sim_method="cosine")
    else:
        W, r = _ami_inputs(X, y, dataset)
        sparse, deleted = _ami_deletion_order(X.columns.tolist(), W, r)
    addable = list(reversed(deleted))
    n_add = max(0, min(e_max(X.shape[1]) - sparse.number_of_edges(), len(addable)))
    step = max(1, n_add // 5)
    graphs, G = [], sparse.copy()
    for batch in [[]] + [addable[:n_add][i:i + step] for i in range(0, n_add, step)]:
        for u, v, w in batch:
            G.add_edge(u, v, weight=w)
        graphs.append(G.copy())
    return graphs


def score_tau_b(a: dict, b: dict) -> float:
    """Kendall tau_b on scores aligned by feature, ties kept; undefined -> -1 (not stable)"""
    F = list(a)
    t = float(kendalltau([a[f] for f in F], [b[f] for f in F], variant="b")[0])
    return t if np.isfinite(t) else -1.0


def search(graphs, explain):
    """explain(G) -> {feature: score}. Returns (selected index, scores of it, info)."""
    runs = []

    def phi(i):
        while len(runs) <= i:
            runs.append(explain(graphs[len(runs)]))
        return runs[i]

    Q, sel, converged = len(graphs), len(graphs) - 1, False
    for q in range(Q - WINDOW + 1):
        if min(score_tau_b(phi(i), phi(j)) for i, j in combinations(range(q, q + WINDOW), 2)) >= TAU0:
            sel, converged = q + WINDOW - 1, True
            break
    scores = phi(sel)
    n = len(scores)
    info = {"selected_index": sel, "converged": converged, "graphs_run": len(runs), "n_candidates": Q,
            "selected_edges": graphs[sel].number_of_edges(),
            "selected_avg_degree": 2 * graphs[sel].number_of_edges() / n}
    return sel, scores, info


def run_improved_shapg_adaptive(X, y, char_func, shapg_custom_func=None, task=None, dataset=None,
                          _side_channel=None, **kwargs):
    """Retraining game (shapg_custom_func); the runner's timing covers the whole search."""
    from shapG import ShapGExplainer

    def explain(G):
        np.random.seed(SEED); random.seed(SEED)
        sv = ShapGExplainer(characteristic_function=shapg_custom_func, depth=1, n_samples=3,
                            approximate_by_ratio=False, scale=False, verbose=False).fit_explain(G)
        return {f: float(v) for f, v in sv.items()}

    _, scores, info = search(candidate_graphs(X, y, task, dataset), explain)
    if _side_channel is not None:
        _side_channel["adaptive"] = info
    print(f"[adaptive: deg {info['selected_avg_degree']:.1f} conv={info['converged']} "
          f"graphs {info['graphs_run']}/{info['n_candidates']}]", end=" ", flush=True)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def ami_graph_with_edges(X, y, dataset, n_edges):
    """AMI rank-deletion graph with n_edges edges: the sparsest tree plus the deleted edges added back in
    reverse order (= rank deletion stopped at n_edges). Used by fixed (average degree 6) on classification."""
    W, r = _ami_inputs(X, y, dataset)
    sparse, deleted = _ami_deletion_order(X.columns.tolist(), W, r)
    G = sparse.copy()
    for u, v, w in list(reversed(deleted))[:max(0, n_edges - sparse.number_of_edges())]:
        G.add_edge(u, v, weight=w)
    return G
