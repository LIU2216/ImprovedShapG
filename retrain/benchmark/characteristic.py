"""Characteristic functions for coalition-based XAI methods (CIS, Exact Shapley, ShapG).

The characteristic function v(S) = KPI of a model trained only on features S.
Regression → R²,  Classification → F1.

Normalisation is applied after the train/test split (scaler fit on X_train only)
so graph-building code receives raw X while model training always sees scaled data.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from shapG import CustomFunction

from .kpi import compute_char_kpi
from .models import build_model


def _scale(X_train, X_test):
    """Fit StandardScaler on X_train, return scaled DataFrames."""
    scaler = StandardScaler()
    Xtr = pd.DataFrame(scaler.fit_transform(X_train),
                        columns=X_train.columns, index=X_train.index)
    Xte = pd.DataFrame(scaler.transform(X_test),
                        columns=X_test.columns, index=X_test.index)
    return Xtr, Xte


def build_char_func(X, y, model_type: str, task: str,
                    test_size: float = 0.2, random_state: int = 42,
                    dataset: str = None):
    """Build a characteristic function closure and the train/test split.

    X_train / X_test returned are normalised (scaler fit on train only).
    Callers that need raw X for graph building should use the original X directly.

    Returns
    -------
    char_func     : callable(features) → float   (no caching)
    X_train, X_test, y_train, y_test  (normalised)
    """
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train, X_test = _scale(X_train_raw, X_test_raw)

    def char_func(features):
        # Canonical (sorted) column order: MLP is sensitive to input-column
        # ordering, so a coalition's KPI must not depend on the order it arrives
        # in. Sorting makes v(S) a pure function of the feature SET.
        cols = sorted(features) if features else []
        if not cols:
            return 0.0
        model = build_model(model_type, task, dataset=dataset)
        model.fit(X_train[cols], y_train)
        y_pred = model.predict(X_test[cols])
        return compute_char_kpi(y_test, y_pred, task)

    return char_func, X_train, X_test, y_train, y_test


def build_shapg_custom_function(X, y, model_type: str, task: str,
                                 test_size: float = 0.2, random_state: int = 42,
                                 dataset: str = None):
    """Build a shapG CustomFunction with coalition-level caching.

    The cache persists across multiple ShapG runs (different density ratios),
    so each unique feature coalition is trained only once.
    X_train / X_test are normalised (scaler fit on train only).

    Returns
    -------
    custom_func : shapG.CustomFunction
    cache       : dict  {frozenset → float}
    stats       : dict  {"hits": int, "misses": int}
    """
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train, X_test = _scale(X_train_raw, X_test_raw)

    _cache: dict = {}
    _stats = {"hits": 0, "misses": 0}

    def _raw(features):
        # Canonical (sorted) column order — see char_func above. Keeps v(S)
        # order-independent so the frozenset-keyed cache is always consistent.
        cols = sorted(features) if features else []
        if not cols:
            return 0.0
        model = build_model(model_type, task, dataset=dataset)
        model.fit(X_train[cols], y_train)
        y_pred = model.predict(X_test[cols])
        return compute_char_kpi(y_test, y_pred, task)

    def wrapper(coalition, context):
        key = frozenset(coalition)
        if key in _cache:
            _stats["hits"] += 1
            return _cache[key]
        _stats["misses"] += 1
        val = _raw(list(coalition))
        _cache[key] = val
        return val

    kpi_label = "R²" if task == "regression" else "F1"
    name = f"{model_type.upper()} {kpi_label} (cached)"
    return CustomFunction(wrapper, name=name), _cache, _stats
