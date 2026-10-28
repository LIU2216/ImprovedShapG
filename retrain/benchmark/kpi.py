"""KPI computation — R² for regression, F1 for classification."""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .models import build_model


def _set_mlp_seed(model, seed: int) -> None:
    """Override random_state on an MLP."""
    if hasattr(model, "random_state"):
        model.set_params(random_state=seed)


def _scale(X_train, X_test):
    """Fit StandardScaler on X_train, return scaled DataFrames."""
    scaler = StandardScaler()
    cols = X_train.columns.tolist()
    Xtr = pd.DataFrame(scaler.fit_transform(X_train), columns=cols)
    Xte = pd.DataFrame(scaler.transform(X_test),      columns=cols)
    return Xtr, Xte


def compute_char_kpi(y_test, y_pred, task: str) -> float:
    """KPI for the characteristic function v(S).

    regression     → R²
    classification → F1
    """
    if task == "regression":
        return r2_score(y_test, y_pred)
    return f1_score(y_test, y_pred)


def compute_kpi(y_test, y_pred, task: str) -> float:
    """KPI for feature-dropping evaluation (slope).

    regression     → R²
    classification → F1

    Same metric as compute_char_kpi: v(S) and the dropping curve both measure
    model performance, so ShapG's coalition ranking is optimised for exactly
    what the slope measures. PFI's scoring is also set to F1 for a fair, aligned
    comparison.
    """
    if task == "regression":
        return r2_score(y_test, y_pred)
    return f1_score(y_test, y_pred)


def compute_weighted_slope(metrics: list, beta: float = 0.8) -> float:
    """Weighted decreasing slope with exponential decay.

    S = Σ_{i=0}^{L-2}  β^i · (metrics[i] − metrics[i+1])

    Larger S means the method identified the most important features first.
    """
    deltas = [metrics[i] - metrics[i + 1] for i in range(len(metrics) - 1)]
    weights = [beta ** i for i in range(len(deltas))]
    return float(np.dot(deltas, weights))


def calculate_kpi_for_dropping(
    X_sub,
    y,
    model_type: str,
    task: str,
    test_size: float = 0.2,
    random_state: int = 42,
    dataset: str = None,
    n_repeats: int = 1,
) -> float:
    """Retrain model on X_sub and return the KPI on the held-out test set.

    Normalisation is applied after splitting (scaler fit on X_train only)
    so raw X can be passed here without prior scaling.

    When n_repeats > 1, retrain with n_repeats different random seeds and
    return the mean KPI to reduce initialisation variance.

    Columns are reordered to a canonical (sorted) order before training: MLP is
    sensitive to input-column ordering, so the dropping-curve KPI must match the
    coalition KPI (char_func) for the same feature set regardless of the order
    the columns were passed in.
    """
    X_sub = X_sub[sorted(X_sub.columns)]
    if n_repeats == 1:
        X_train, X_test, y_train, y_test = train_test_split(
            X_sub, y, test_size=test_size, random_state=random_state
        )
        X_train, X_test = _scale(X_train, X_test)
        model = build_model(model_type, task, dataset=dataset)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return compute_kpi(y_test, y_pred, task)

    scores = []
    for i in range(n_repeats):
        seed = random_state + i
        X_train, X_test, y_train, y_test = train_test_split(
            X_sub, y, test_size=test_size, random_state=seed
        )
        X_train, X_test = _scale(X_train, X_test)
        model = build_model(model_type, task, dataset=dataset)
        _set_mlp_seed(model, seed)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores.append(compute_kpi(y_test, y_pred, task))
    return float(np.mean(scores))
