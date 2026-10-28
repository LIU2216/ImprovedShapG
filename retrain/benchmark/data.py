"""Dataset loaders — 10 datasets for Improved ShapG paper."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Graph construction (correlation/similarity) runs on RAW X for every method.
# Keeping cosine uncentred makes it distinct from pearsonr; rank-based methods
# (kendalltau/spearmanr) and mutual_info are scale-invariant anyway. Model
# training still uses StandardScaler-scaled data separately (see kpi.py `_scale`).

DATASET_META = {
    # ── Regression ──────────────────────────────────────────────────────────
    "abalone":   {"file": "abalone_processing.csv",  "target": "Rings",            "task": "regression"},
    "admission": {"file": "admission.csv",            "target": "Chance of Admit ", "task": "regression"},
    "ucs_scm":   {"file": "ucs_scm.csv",             "target": "strength",         "task": "regression"},
    "life":      {"file": "life.csv",                 "target": "Life expectancy ", "task": "regression"},
    "crimedata": {"file": "crimedata_processing.csv", "target": "ViolentCrimesPerPop", "task": "regression"},
    # ── Classification (binary, all targets are 0/1) ─────────────────────────
    "adult":     {"file": "adult_encoded.csv",        "target": "income",           "task": "classification"},
    "Churn":     {"file": "Churn.csv",                "target": "Exited",           "task": "classification"},
    "diabetes":  {"file": "diabetes.csv",             "target": "Diabetes",         "task": "classification"},
    "H1N1":      {"file": "H1N1.csv",                 "target": "h1n1_vaccine",     "task": "classification"},
    "MIC":       {"file": "MIC_processing.csv",       "target": "LET_IS",           "task": "classification"},
}


def load_dataset(name: str):
    """Load a dataset by name.

    Returns
    -------
    X : pd.DataFrame   feature matrix
    y : pd.Series      target vector
    task : str         'regression' or 'classification'
    """
    if name not in DATASET_META:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASET_META)}")
    meta = DATASET_META[name]
    df = pd.read_csv(DATA_DIR / meta["file"])
    X = df.drop(columns=[meta["target"]])
    y = df[meta["target"]]
    return X, y, meta["task"]


def get_task(name: str) -> str:
    return DATASET_META[name]["task"]
