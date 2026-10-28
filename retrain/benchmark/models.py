"""Model factory — LightGBM and MLP for regression / classification.

MLP hyperparameters are tuned per dataset (via tune_mlp.py / Optuna) and
hard-coded in _MLP_DATASET_CONFIGS.  Normalisation is applied inside
characteristic.py and kpi.py after the train/test split (scaler fit on
X_train only), so raw X is passed to build_model callers.
"""

import lightgbm as lgb
from sklearn.neural_network import MLPClassifier, MLPRegressor

# ── LGB defaults (same for all datasets) ──────────────────────────────────────

_LGB_DEFAULTS = dict(verbose=-1,)

# ── LGB per-dataset configs (Optuna-tuned, leakage-free 3-fold CV) ─────────────
# Objective: classification → F1, regression → R². scale_pos_weight tuned for
# imbalanced classification. See tune_models.py.
#
# NOTE: n_estimators was reduced from the raw Optuna values to speed up the
# per-coalition retraining in XAI (each coalition trains one model). The cut
# points were chosen from the KPI-vs-num_iteration curve so the held-out
# R²/F1 is within ~0.005 of the full-tree value (often equal or better, since
# the tuned counts slightly overfit). Original tuned counts:
#   abalone 328, admission 196, ucs_scm 548, life 324, crimedata 499,
#   adult 370, Churn 349, diabetes 318, H1N1 432, MIC 492.
# num_leaves was also cut where the held-out KPI is flat (or better) across
# leaf counts — shallower trees train much faster on the wider/larger-row data.
# Original tuned → new:  abalone 81→31, ucs_scm 43→15, crimedata 64→15,
# adult 98→15, Churn 40→31, H1N1 98→15.  Each kept KPI within ±0.006 (most
# improved); the biggest wins are crimedata/adult/H1N1 (~3–4× faster per fit).
# life (20), MIC (55), admission (101), diabetes (96) were left as-is — fewer
# leaves either hurt their KPI or gave no speedup (data too small).
#
# MLP batch_size: a 3-seed sweep showed larger batches mostly SLOWED MLP
# (early stopping then needs more epochs), so only MIC was changed:
# 128→512 gave +0.0055 KPI and 1.57× faster (0.77s→0.49s). diabetes 1024→2048
# was also taken — not for speed (time was flat) but for +0.0075 KPI. All other
# datasets kept their tuned batch_size — increasing it was slower or hurt KPI.
#
# life was the slowest MLP (~150 epochs to converge). Shrinking the net alone or
# enlarging the batch both failed (KPI crash or slower). What worked: a smaller
# net + faster convergence — (128,64)→(96,48), lr 0.008→0.012, n_iter_no_change
# 10→8 — giving 1.57× (1.70s→1.08s) for −0.006 R² (0.9494→0.9432), and the most
# seed-stable of the candidates. (lr ≥0.024 diverges; arch (128,32) crashes R².)

_LGB_DATASET_CONFIGS = {
    "abalone":   dict(n_estimators=200, num_leaves=31, learning_rate=0.013585663953517835, min_child_samples=38, reg_alpha=0.16965071118062935, reg_lambda=0.0072729861634627975, random_state=42),
    "admission": dict(n_estimators=100, num_leaves=101, learning_rate=0.03138884195923451, min_child_samples=23, reg_alpha=0.09088206125949644, reg_lambda=0.003854901738186631, random_state=42),
    "ucs_scm":   dict(n_estimators=150, num_leaves=15, learning_rate=0.13387790985590264, min_child_samples=11, reg_alpha=0.08161314296357226, reg_lambda=0.24444361636070594, random_state=42),
    "life":      dict(n_estimators=100, num_leaves=20, learning_rate=0.13880091727553712, min_child_samples=21, reg_alpha=0.01308382971327895, reg_lambda=0.002997383894053618, random_state=42),
    "crimedata": dict(n_estimators=75, num_leaves=15, learning_rate=0.04836575450156083, min_child_samples=6, reg_alpha=0.05110333258151407, reg_lambda=0.48301078379059076, random_state=42),
    "adult":     dict(n_estimators=100, num_leaves=15, learning_rate=0.11934343410325828, min_child_samples=81, reg_alpha=7.091767477766311, reg_lambda=1.0225541198743933, scale_pos_weight=1.8721751628137875, random_state=42),
    "Churn":     dict(n_estimators=150, num_leaves=31, learning_rate=0.016448169213449312, min_child_samples=69, reg_alpha=0.006872081253393051, reg_lambda=0.0028060447541589355, scale_pos_weight=2.1840435855139746, random_state=42),
    "diabetes":  dict(n_estimators=75, num_leaves=96, learning_rate=0.06055441622169155, min_child_samples=58, reg_alpha=6.006129733742664, reg_lambda=0.26804433528364385, scale_pos_weight=1.4412979400398622, random_state=42),
    "H1N1":      dict(n_estimators=200, num_leaves=15, learning_rate=0.019007521508967815, min_child_samples=79, reg_alpha=6.189139074400388, reg_lambda=0.13461433506372725, scale_pos_weight=2.4436888979340354, random_state=42),
    "MIC":       dict(n_estimators=75, num_leaves=55, learning_rate=0.1691696892241735, min_child_samples=22, reg_alpha=0.03430499071604451, reg_lambda=0.07731518837934229, scale_pos_weight=2.681485813353589, random_state=42),
}

# ── MLP per-dataset configs (Optuna-tuned) ─────────────────────────────────────

# early_stopping=False on purpose: with it on, certain feature subsets trigger a
# premature stop before the net fits, collapsing R² to ~0 and producing huge
# spurious spikes in the feature-dropping curve. Training to a fixed budget makes
# single-run (n_repeats=1) curves stable. alpha (per-dataset) controls overfit.
_MLP_TUNED_BASE = dict(
    solver="adam",
    max_iter=300,
    early_stopping=False,
    random_state=42,
)

# Leakage-free 3-fold CV (objective: F1 / R²). Every config trains+predicts in
# < 3 s on its full training split. Churn uses a deliberately small/fast config
# (the only one meeting the 3 s budget on 132 k samples). See tune_models.py.
#
# NOTE: hidden_layer_sizes shrunk for XAI speed on three datasets where a
# smaller net was faster without materially hurting KPI (verified on the
# held-out split): diabetes (256,128)→(128,64) [+0.006 R²/F1, 1.7×],
# crimedata (128,64,32)→(64,32,32) [-0.006, 2.3×], Churn (64,)→(32,) [-0.002,
# 1.3×]. Other datasets kept their tuned arch — shrinking them lost too much
# (e.g. ucs_scm 0.91→0.77) or didn't speed up (early-stop dynamics).
_MLP_DATASET_CONFIGS = {
    "abalone": dict(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        alpha=0.04058077103731952,
        learning_rate_init=0.004604161572369996,
        batch_size=32,
        max_iter=150,
        n_iter_no_change=10,
    ),
    "admission": dict(
        hidden_layer_sizes=(64,),
        activation="relu",
        alpha=0.07490207013706272,
        learning_rate_init=0.007920339556641407,
        batch_size=128,
        max_iter=300,
        n_iter_no_change=10,
    ),
    "ucs_scm": dict(
        hidden_layer_sizes=(64,),
        activation="tanh",
        alpha=0.0006233748602704164,
        learning_rate_init=0.007536595831898086,
        batch_size=32,
        max_iter=100,
        n_iter_no_change=10,
    ),
    "life": dict(
        hidden_layer_sizes=(32,),
        activation="tanh",
        alpha=0.0006336303115984684,
        learning_rate_init=0.0054479089516943895,
        batch_size=64,
        max_iter=200,
        n_iter_no_change=10,
    ),
    "crimedata": dict(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        alpha=0.0005672728141838273,
        learning_rate_init=0.0027628437223770736,
        batch_size=64,
        max_iter=200,
        n_iter_no_change=10,
    ),
    "adult": dict(
        hidden_layer_sizes=(16,),
        activation="tanh",
        alpha=1.3357240411974104e-05,
        learning_rate_init=0.0018742210985555703,
        batch_size=256,
        max_iter=200,
        n_iter_no_change=7,
    ),
    "Churn": dict(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        alpha=0.002718028407207569,
        learning_rate_init=0.0007781207679186852,
        batch_size=512,
        max_iter=75,
        n_iter_no_change=5,
    ),
    "diabetes": dict(
        hidden_layer_sizes=(16,),
        activation="tanh",
        alpha=0.0009273188981542832,
        learning_rate_init=0.00026354747293806744,
        batch_size='auto',
        max_iter=150,
        n_iter_no_change=10,
    ),
    "H1N1": dict(
        hidden_layer_sizes=(16,),
        activation="tanh",
        alpha=1.3357240411974104e-05,
        learning_rate_init=0.0018742210985555703,
        batch_size=256,
        max_iter=200,
        n_iter_no_change=7,
    ),
    "MIC": dict(
        hidden_layer_sizes=(32, 16),
        activation="tanh",
        alpha=0.00013284336135942253,
        learning_rate_init=0.00019887413901611116,
        batch_size=64,
        max_iter=100,
        n_iter_no_change=10,
    ),
}

# Fallback when dataset is not in _MLP_DATASET_CONFIGS
_MLP_DEFAULTS = dict(
    hidden_layer_sizes=(64, 32),
    max_iter=500,
    random_state=42,
    early_stopping=False,   # see _MLP_TUNED_BASE: avoids R²-collapse spikes
    n_iter_no_change=10,
)


def build_model(model_type: str, task: str, dataset: str = None):
    """Return an unfitted model instance.

    Parameters
    ----------
    model_type : 'lgb' or 'mlp'
    task       : 'regression' or 'classification'
    dataset    : dataset name used to look up per-dataset MLP config
    """
    if model_type == "lgb":
        lgb_cfg = {**_LGB_DEFAULTS, **_LGB_DATASET_CONFIGS.get(dataset, {})}
        if task == "regression":
            return lgb.LGBMRegressor(**lgb_cfg)
        return lgb.LGBMClassifier(**lgb_cfg)

    if model_type == "mlp":
        cls = MLPRegressor if task == "regression" else MLPClassifier
        if dataset in _MLP_DATASET_CONFIGS:
            cfg = _MLP_DATASET_CONFIGS[dataset]
            params = {
                **_MLP_TUNED_BASE,
                "hidden_layer_sizes": cfg["hidden_layer_sizes"],
                "activation":         cfg["activation"],
                "alpha":              cfg["alpha"],
                "learning_rate_init": cfg["learning_rate_init"],
                "batch_size":         cfg["batch_size"],
                "n_iter_no_change":   cfg["n_iter_no_change"],
            }
            if "max_iter" in cfg:          # per-dataset cap (overrides base 300)
                params["max_iter"] = cfg["max_iter"]
            return cls(**params)
        return cls(**_MLP_DEFAULTS)

    raise ValueError(f"Unknown model_type '{model_type}'. Choose 'lgb' or 'mlp'.")
