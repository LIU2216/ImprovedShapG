"""LaTeX table bodies and aggregate numbers for the paper, from the saved retraining results.

Nothing is re-run. Writes paper_tables/{combined_reg,combined_cls,adaptive_graphs,avg_rank}.tex and
paper_tables/aggregate.txt, and prints them. Conventions as in the previous tables: nine methods
(PFI excluded), bold = best slope / fastest time per row, ranks assigned within a row over the methods
present in that row (ties broken by column order), averaged over the rows a method was evaluated on.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent
RES = ROOT / "retrain" / "results"
OUT = ROOT / "paper_tables"

METHODS = ["exact_cis", "exact_shapley", "shap_explainer", "sampling_shap", "kernel_shap",
           "original_shapg", "sampling_owen", "improved_shapg_fixed", "improved_shapg_adaptive"]
DISP = {"exact_cis": "Exact CIS", "exact_shapley": "Exact Shapley", "shap_explainer": "SHAP",
        "sampling_shap": "SamplingSHAP", "kernel_shap": "KernelSHAP", "original_shapg": "Original ShapG",
        "sampling_owen": "Sampling Owen", "improved_shapg_fixed": "Improved ShapG (fixed)",
        "improved_shapg_adaptive": "Improved ShapG (adaptive)"}
REG = ["admission", "abalone", "ucs_scm", "life", "crimedata"]
CLS = ["Churn", "adult", "diabetes", "H1N1", "MIC"]
MODELS = ["lgb", "mlp"]
NAME = {"admission": "Admission", "abalone": "Abalone", "ucs_scm": "UCS-SCM", "life": "Life",
        "crimedata": "Crime", "Churn": "Churn", "adult": "Adult", "diabetes": "Diabetes", "H1N1": "H1N1",
        "MIC": "MIC"}


def load(ds, mdl):
    return json.load(open(RES / f"benchmark_{ds}_{mdl}" / "results.json"))


def fmt_time(v):
    if v < 1:
        return f"{v:.3f}s"
    if v < 60:
        return f"{v:.1f}s"
    return f"{v / 60:.1f}m"


def combined_block(datasets):
    lines = []
    for ds in datasets:
        rows = []
        for mdl in MODELS:
            m = load(ds, mdl)["methods"]
            S = {k: m[k]["slope"] for k in METHODS if k in m}
            T = {k: m[k]["duration"] for k in METHODS if k in m}
            bs, bt = max(S.values()), min(T.values())
            cells = []
            for k in METHODS:
                if k not in S:
                    cells.append("--")
                    continue
                s, t = f"{S[k]:.4f}", fmt_time(T[k])
                s = rf"\textbf{{{s}}}" if S[k] == bs else s
                t = rf"\textbf{{{t}}}" if T[k] == bt else t
                cells.append(rf"\makecell{{{s}\\{t}}}")
            rows.append((mdl, cells))
        lines.append(rf"\multirow{{2}}{{*}}{{\textbf{{{NAME[ds]}}}}} & \texttt{{{rows[0][0]}}} & "
                     + " & ".join(rows[0][1]) + r" \\")
        lines.append(r"\cmidrule(lr){2-11}")
        lines.append(rf" & \texttt{{{rows[1][0]}}} & " + " & ".join(rows[1][1]) + r" \\")
        if ds != datasets[-1]:
            lines.append(r"\midrule")
    return "\n".join(lines)


def adaptive_block():
    lines = []
    for task, dss, kpi in [("regression", REG, "R^2"), ("classification", CLS, "F_1")]:
        for ds in dss:
            per = []
            for mdl in MODELS:
                d = load(ds, mdl)
                a = d["methods"]["improved_shapg_adaptive"]["extra"]["adaptive"]
                N = d["n_features"]
                per.append((mdl, N, a, d["methods"]["improved_shapg_adaptive"]["slope"]))
            N = per[0][1]
            M = N * (N - 1) // 2
            for i, (mdl, _, a, s) in enumerate(per):
                head = (rf"\multirow{{2}}{{*}}{{\textbf{{{NAME[ds]}}}}} & \multirow{{2}}{{*}}{{{task} (${kpi}$)}} & "
                        if i == 0 else " &  & ")
                nf = rf"\multirow{{2}}{{*}}{{{N}}}" if i == 0 else ""
                conv = "yes" if a["converged"] else "no"
                lines.append(head + rf"{mdl} & {nf} & {a['selected_edges']}/{M} & "
                             rf"{a['selected_edges'] / M:.4f} & {a['selected_avg_degree']:.2f} & "
                             rf"{a['graphs_run']}/{a['n_candidates']} & {conv} & {s:.4f} \\")
            if ds == REG[-1]:
                lines.append(r"\midrule\midrule")
            elif ds != CLS[-1]:
                lines.append(r"\midrule")
    return "\n".join(lines)


FULL = ["exact_cis", "shap_explainer", "original_shapg", "sampling_owen", "improved_shapg_fixed",
        "improved_shapg_adaptive"]          # evaluated on all ten datasets
PARTIAL = ["sampling_shap", "kernel_shap", "exact_shapley"]


def avg_ranks(model):
    """Ranks within each row over the methods evaluated on all ten datasets only, so that every
    average is taken over the same rows."""
    sr = {k: [] for k in FULL}
    tr = {k: [] for k in FULL}
    for ds in REG + CLS:
        m = load(ds, model)["methods"]
        for i, k in enumerate(sorted(FULL, key=lambda k: -m[k]["slope"])):
            sr[k].append(i + 1)
        for i, k in enumerate(sorted(FULL, key=lambda k: m[k]["duration"])):
            tr[k].append(i + 1)
    return {k: (np.mean(sr[k]), np.mean(tr[k])) for k in FULL}


def rank_block():
    lgb, mlp = avg_ranks("lgb"), avg_ranks("mlp")
    lines = []
    for k in sorted(FULL, key=lambda k: (lgb[k][0] + mlp[k][0]) / 2):
        lines.append(rf"{DISP[k]} & {lgb[k][0]:.1f} & {lgb[k][1]:.1f} & {mlp[k][0]:.1f} & {mlp[k][1]:.1f} \\")
    return "\n".join(lines)


def partial_block():
    """Adaptive against the methods not evaluated on every dataset, on the pairs where they were."""
    out = []
    for k in PARTIAL:
        w = l = t = 0
        for ds in REG + CLS:
            for mdl in MODELS:
                m = load(ds, mdl)["methods"]
                if k in m:
                    a, b = m["improved_shapg_adaptive"]["slope"], m[k]["slope"]
                    w, l, t = w + (a > b), l + (a < b), t + (a == b)
        out.append(f"adaptive vs {DISP[k]}: {w + l + t} pairs, higher {w}, lower {l}, equal {t}")
    return "\n".join(out)


def aggregate():
    pairs = [(ds, mdl) for ds in REG + CLS for mdl in MODELS]
    D = {p: load(*p)["methods"] for p in pairs}
    out = []
    o = np.array([D[p]["original_shapg"]["slope"] for p in pairs])
    for k in ["original_shapg", "sampling_owen", "improved_shapg_fixed", "improved_shapg_adaptive"]:
        s = np.array([D[p][k]["slope"] for p in pairs])
        t = sum(D[p][k]["duration"] for p in pairs)
        line = f"{DISP[k]:28s} time {t:9.0f}s = {t / 3600:.2f} h  mean S {s.mean():.4f}"
        if k != "original_shapg":
            d = s - o
            line += (f"  vs Original {(d > 0).sum()}/{(d < 0).sum()}/{(d == 0).sum()} (W/L/T)"
                     f"  mean diff {d.mean():+.4f}  Wilcoxon p {wilcoxon(d[d != 0]).pvalue:.3f}")
        out.append(line)
    a = np.array([D[p]["improved_shapg_adaptive"]["slope"] for p in pairs])
    for k in ["sampling_owen", "improved_shapg_fixed"]:
        s = np.array([D[p][k]["slope"] for p in pairs]); d = a - s
        out.append(f"adaptive vs {DISP[k]}: {(d > 0).sum()}/{(d < 0).sum()}/{(d == 0).sum()}  p {wilcoxon(d[d != 0]).pvalue:.3f}")
    for task, dss in [("regression", REG), ("classification", CLS)]:
        sub = [p for p in pairs if p[0] in dss]
        for k in ["improved_shapg_fixed", "improved_shapg_adaptive"]:
            d = np.array([D[p][k]["slope"] - D[p]["original_shapg"]["slope"] for p in sub])
            out.append(f"{task:14s} {DISP[k]:28s} vs Original {(d > 0).sum()}/{(d < 0).sum()}  mean diff {d.mean():+.4f}")
    losses = [f"{p[0]}_{p[1]}" for p in pairs if D[p]["improved_shapg_adaptive"]["slope"] < D[p]["original_shapg"]["slope"]]
    out.append("adaptive below Original: " + ", ".join(losses))
    return "\n".join(out)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    blocks = {"combined_reg.tex": combined_block(REG), "combined_cls.tex": combined_block(CLS),
              "adaptive_graphs.tex": adaptive_block(), "avg_rank.tex": rank_block(),
              "aggregate.txt": aggregate() + "\n" + partial_block()}
    for name, text in blocks.items():
        (OUT / name).write_text(text + "\n")
        print(f"%%% {name}\n{text}\n")
