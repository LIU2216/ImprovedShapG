# Improved ShapG — benchmark code

Code, data and saved results for the experiments of the Improved ShapG paper: Original ShapG,
Improved ShapG (fixed and adaptive) and eight baseline explainers on ten tabular datasets
(five regression, five binary classification), each with a LightGBM and an MLP model (20 cases).

## Layout
```
retrain/                     retraining characteristic function (one model retrained per coalition)
  run_benchmark.py           one dataset-model case: all methods, then the feature-dropping evaluation (S)
  benchmark/
    adaptive.py              Improved ShapG (adaptive) and the AMI / cosine graph construction
    graph_order.py           rank-deletion edge order
    explainers.py            Original ShapG, Sampling Owen, Improved ShapG (fixed), SHAP baselines, PFI, ...
    characteristic.py        characteristic functions;  kpi.py  F1 / R2 and weighted slope S
    data.py  models.py  plotting.py
  data/                      the ten datasets (preprocessed CSV)
  results/benchmark_<dataset>_<model>/results.json
noretrain/
  run_noretrain.py           no-retraining (SHAP-style) game for Original, fixed, adaptive, Exact Shapley
  results/<dataset>_<model>.json
run_all.sh                   all 20 cases (retraining, then no-retraining), small to large
make_paper_tables.py         paper tables / aggregate numbers from the saved results -> paper_tables/
make_paper_figures.py        paper figures from the saved results: python make_paper_figures.py <out_dir>
```

## Methods
- **Original ShapG**: Kendall-tau minimal-edge graph (shapG `GraphBuilder.from_kendalltau_minimal_edge`).
- **Improved ShapG (fixed)**: one ShapG run on a rank-deletion graph of average degree 6 (round(3N) edges);
  association = adjusted mutual information (classification) / cosine (regression).
- **Improved ShapG (adaptive)**: rank-deletion path from the spanning tree, floor(n_add/5) edges per step up to
  average degree 10; stops at the first 3 consecutive graphs whose pairwise Kendall tau_b of the scores is
  >= 0.95 and returns the last graph of that window (otherwise the cap graph).
- Baselines: Exact Shapley (N <= 15), Sampling Owen, SHAP, SamplingSHAP, KernelSHAP, exact CIS, PFI.

Graph statistics (AMI / cosine, node relevance) are computed on all rows of each dataset
(classification AMI: at most 10,000 rows sampled uniformly, seed 0); no model is trained on test rows.

## Setup
Python 3.10. `pip install -r requirements.txt` (versions used for the paper).

## Reproduce
```
./run_all.sh                       # about 24 h on one machine; re-running skips finished cases
cd retrain && python run_benchmark.py --dataset life --model lgb       # a single case
cd noretrain && python run_noretrain.py --datasets life --models lgb   # needs the retraining result
python make_paper_tables.py
python make_paper_figures.py figures/
```
Settings used for the paper: seed 42, one 80/20 train/test split, library-default threads, methods run
serially and each timed standalone from an empty coalition cache. `RANK_DESC=0` (ascending node order) is
set inside the entry scripts; the shapG library default is descending, so keep it when calling shapG directly.
Running times depend on the machine.
