"""Paper figures from the saved results (no explainer is re-run): the four feature-dropping panels
(Life and Adult, LightGBM and MLP), the Life graphs (complete, Original ShapG, Improved ShapG fixed and
adaptive LightGBM) and the average degree of the Original ShapG graph against the number of features.
No in-figure title, one legend size shared by the four dropping panels, circular layout over the
feature names so the graph panels differ only in their edges.
Usage: python make_paper_figures.py <output Figures directory>
"""
import os
os.environ["RANK_DESC"] = "0"          # must precede the shapG import
import json
import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "retrain"))
from benchmark.plotting import DISPLAY_NAMES, PLOT_ORDER, METHOD_STYLE, EXCLUDED_METHODS, _FALLBACK_STYLE
from benchmark.data import load_dataset
from benchmark import adaptive as A
from shapG import GraphBuilder

OUT = sys.argv[1]
RES = ROOT / "retrain" / "results"
FIGSIZE, LABEL_FS, TICK_FS = (7.0, 4.2), 13, 11
COMPACT = dict(labelspacing=0.28, handlelength=1.4, handletextpad=0.45, borderpad=0.35, borderaxespad=0.4)
DROP_LEGEND_LOC = {("life", "lgb"): "lower left", ("life", "mlp"): "lower left",
                   ("adult", "lgb"): "lower left", ("adult", "mlp"): "lower left"}
EDGE_COLOR, NODE_COLOR = "0.55", "#1f4e79"
DATASETS = ["admission", "abalone", "ucs_scm", "life", "crimedata",
            "Churn", "adult", "diabetes", "H1N1", "MIC"]
K0 = 6
def _curve_points(ax, skip_labels=()):
    """All plotted curve points in display coordinates, densely interpolated."""
    import numpy as np
    pts = []
    for ln in ax.get_lines():
        if ln.get_label() in skip_labels or ln.get_label().startswith("Best:"):
            continue
        x, y = np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)
        if len(x) < 2:
            continue
        t = np.linspace(0, len(x) - 1, (len(x) - 1) * 12 + 1)
        xi = np.interp(t, range(len(x)), x)
        yi = np.interp(t, range(len(y)), y)
        pts.append(ax.transData.transform(np.column_stack([xi, yi])))
    return np.vstack(pts) if pts else np.empty((0, 2))


def fit_legend(fig, ax, loc, hi=LABEL_FS, lo=5.0, step=0.5, margin=2.0,
               vlines=(), **kw):
    """Largest legend font size at `loc` whose frame clears every curve.

    Capped at the axis-label size so the legend never outgrows the labels.
    `vlines` gives data x-positions of full-height vertical lines (the
    ``axvline`` marking the best ratio). Those span the whole axis, so only
    their horizontal overlap with the legend box is meaningful.
    """
    pts = _curve_points(ax)
    vx = [ax.transData.transform((x, 0))[0] for x in vlines]
    best = None
    fs = hi
    while fs >= lo:
        leg = ax.legend(loc=loc, fontsize=fs, frameon=True, **kw)
        fig.canvas.draw()
        bb = leg.get_window_extent().expanded(1.0, 1.0)
        inside = ((pts[:, 0] > bb.x0 - margin) & (pts[:, 0] < bb.x1 + margin) &
                  (pts[:, 1] > bb.y0 - margin) & (pts[:, 1] < bb.y1 + margin))
        crosses = any(bb.x0 - margin < x < bb.x1 + margin for x in vx)
        if not inside.any() and not crosses:
            best = fs
            break
        fs -= step
    if best is None:                      # nothing fits: keep the smallest tried
        best = lo
        ax.legend(loc=loc, fontsize=best, frameon=True, **kw)
    return best


def save(fig, name):
    for ext in ("pdf", "eps"):
        fig.savefig(f"{OUT}/{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  {name}")


def feature_dropping(dataset, model, force_fs=None):
    d = json.load(open(RES / f"benchmark_{dataset}_{model}" / "results.json"))
    methods, kpi = d["methods"], d["kpi"]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for name in PLOT_ORDER:
        if name in EXCLUDED_METHODS or name not in methods:
            continue
        res = methods[name]
        style = METHOD_STYLE.get(name, _FALLBACK_STYLE)
        ax.plot(range(len(res["metrics"])), res["metrics"], label=f'{DISPLAY_NAMES[name]}  (S={res["slope"]:.4f})',
                linewidth=1.5, alpha=0.85, color=style["color"], marker=style["marker"], markersize=5)
    n = len(next(iter(methods.values()))["metrics"])
    ax.set_xlabel("Number of features dropped", fontsize=LABEL_FS)
    ax.set_ylabel(kpi, fontsize=LABEL_FS)
    ax.set_xticks(range(n))
    ax.tick_params(labelsize=TICK_FS)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    loc = DROP_LEGEND_LOC[(dataset, model)]
    if force_fs is None:
        fs = fit_legend(fig, ax, loc, **COMPACT)
        plt.close(fig)
        return fs
    ax.legend(loc=loc, fontsize=force_fs, frameon=True, **COMPACT)
    save(fig, f"{dataset}_{model}_feature_dropping")


def original_graph(X):
    """Original ShapG graph: Kendall-tau minimal-edge graph, as in run_original_shapg."""
    return GraphBuilder.from_kendalltau_minimal_edge(X, reverse=True, version="v3")


def graph_panels(dataset="life", model="lgb"):
    X, y, task = load_dataset(dataset)
    X = pd.DataFrame(X)
    N, cols = X.shape[1], list(X.columns)
    pos = nx.circular_layout(cols)
    fixed = A.ami_graph_with_edges(X, y, dataset, min(round(6 * N / 2), N * (N - 1) // 2))
    sel = json.load(open(RES / f"benchmark_{dataset}_{model}" / "results.json"))[
        "methods"]["improved_shapg_adaptive"]["extra"]["adaptive"]["selected_index"]
    adaptive = A.candidate_graphs(X, y, task, dataset)[sel]
    graphs = [(f"{dataset}_graph_complete", nx.complete_graph(cols)),
              (f"{dataset}_graph_original", original_graph(X)),
              (f"{dataset}_graph_fixed", fixed),
              (f"{dataset}_graph_adaptive_{model}", adaptive)]
    for name, G in graphs:
        H = nx.Graph(); H.add_nodes_from(cols); H.add_edges_from(G.edges())
        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        nx.draw_networkx_edges(H, pos, ax=ax, edge_color=EDGE_COLOR, width=0.45)
        nx.draw_networkx_nodes(H, pos, ax=ax, node_color=NODE_COLOR, node_size=34, linewidths=0)
        ax.set_axis_off(); ax.set_aspect("equal")
        fig.tight_layout(pad=0.05)
        for ext in ("pdf", "eps"):
            fig.savefig(f"{OUT}/{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        e = H.number_of_edges()
        print(f"  {name:32s} {e} edges  rho={2 * e / (N * (N - 1)):.2f}  avg degree {2 * e / N:.2f}")


def degree_figure():
    Ns, orig_d, names = [], [], []
    for ds in DATASETS:
        X, _, _ = load_dataset(ds)
        X = pd.DataFrame(X)
        N = X.shape[1]
        e = original_graph(X).number_of_edges()
        Ns.append(N); orig_d.append(2 * e / N); names.append(ds)
    order = np.argsort(Ns)
    Ns = np.array(Ns)[order]; orig_d = np.array(orig_d)[order]
    names = [names[i] for i in order]
    # Markers only: these are ten separate datasets, not samples of a curve.
    offsets = {"admission": (4, -13), "abalone": (-24, 3), "Churn": (-9, 8),
               "adult": (5, -13), "ucs_scm": (6, 3), "diabetes": (6, -2),
               "life": (6, -2), "H1N1": (6, -2), "MIC": (4, -13),
               "crimedata": (-36, -2)}
    label = {"ucs_scm": "UCS-SCM", "crimedata": "Crime", "admission": "Admission",
             "abalone": "Abalone", "life": "Life", "diabetes": "Diabetes", "adult": "Adult"}
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(Ns, orig_d, "o", color="#c0392b", ms=6, label="Original ShapG")
    ax.axhline(K0, color="#1f4e79", ls="--", lw=1.6,
               label=f"Improved ShapG (fixed), average degree {K0}")
    for N, d, nm in zip(Ns, orig_d, names):
        ax.annotate(label.get(nm, nm), (N, d), textcoords="offset points",
                    xytext=offsets.get(nm, (5, 4)), fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xticks([7, 10, 15, 20, 35, 60, 120])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(6.2, 175)
    ax.set_ylim(-2, 74)
    ax.set_xlabel("Number of features")
    ax.set_ylabel("Average degree")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "eps"):
        fig.savefig(f"{OUT}/avg_degree_vs_n.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  avg_degree_vs_n: " + ", ".join(f"{n}(N={N}) d={d:.1f}" for n, N, d in zip(names, Ns, orig_d)))


if __name__ == "__main__":
    panels = [(ds, m) for ds in ("life", "adult") for m in ("lgb", "mlp")]
    common = min(feature_dropping(*p) for p in panels)
    print(f"feature dropping legend {common}pt")
    for p in panels:
        feature_dropping(*p, force_fs=common)
    graph_panels()
    degree_figure()
