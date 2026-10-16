"""
Naive baselines for comparison against RF/ANN.

A flexible model matching a deterministic physical formula closely (see
README's "Evaluation methodology") isn't informative on its own -- the
question a reviewer will ask is "how much better is this than something
trivial?". This module answers that with two standard naive baselines,
evaluated on the exact same held-out test split as RF/ANN:

- Persistence: predict target[t] = target[t-1] (previous-hour value).
- Linear: a single-feature OLS fit, rail_temp = a * air_temp_c + b, fit on
  the training split only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from train_models import chronological_split_3way, evaluate


def persistence_baseline(work: pd.DataFrame, target_column: str, test_df: pd.DataFrame) -> dict:
    """target[t] = target[t-1], evaluated on the test split only. The first
    test row has no in-window predecessor if it's also the first row of
    `work`, which can't happen here since test always follows train+val."""
    shifted = work[target_column].shift(1)
    pred = shifted.loc[test_df.index]
    actual = test_df[target_column]
    valid = pred.notna()
    return evaluate(actual[valid].to_numpy(), pred[valid].to_numpy())


def linear_airtemp_baseline(
    train_df: pd.DataFrame, test_df: pd.DataFrame, target_column: str,
    feature_column: str = "air_temp_c",
) -> dict:
    model = LinearRegression()
    model.fit(train_df[[feature_column]], train_df[target_column])
    pred = model.predict(test_df[[feature_column]])
    metrics = evaluate(np.asarray(test_df[target_column].to_numpy(), dtype=float), np.asarray(pred, dtype=float))
    metrics["coef"] = float(model.coef_[0])
    metrics["intercept"] = float(model.intercept_)
    return metrics


def compute_baselines(df: pd.DataFrame, feature_columns: list[str], target_column: str) -> dict[str, dict]:
    """Same dropna + chronological_split_3way as _train_and_evaluate(), so
    these baselines are scored on the identical test rows as RF/ANN."""
    work = df.dropna(subset=feature_columns + [target_column]).copy().sort_index()
    train_df, _val_df, test_df = chronological_split_3way(work)
    return {
        "persistence": persistence_baseline(work, target_column, test_df),
        "linear_airtemp": linear_airtemp_baseline(train_df, test_df, target_column),
    }
