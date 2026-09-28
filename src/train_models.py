"""
Random Forest + small PyTorch ANN for rail surface temperature prediction.

======================================================================
NO REAL GROUND-TRUTH RAIL TEMPERATURE DATA EXISTS FOR THIS PROJECT YET.
======================================================================
Obtaining that requires contacting Uzbek Railways / Tashkent Metro
directly -- a separate, unresolved step (see README.md).

Because of that, this module trains RF and the ANN on the PHYSICAL
BASELINE MODEL'S OWN OUTPUT (src/physical_model.py) as a proxy target.
That means: the RF/ANN metrics below measure only how well each model can
reproduce the physical baseline's formula from weather features -- they
say NOTHING about real-world rail temperature prediction accuracy. Do
not quote these RMSE/MAE/R2 numbers as real-world performance anywhere.

Once real measured rail temperature is available, call
`train_and_evaluate_real_target()` -- the single function in this file
meant for that purpose -- instead of `train_and_evaluate_proxy_target()`.
Everything else (features, splitting, model architectures, metrics) is
shared and does not need to change.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn

FEATURE_COLUMNS = [
    "air_temp_c",
    "rel_humidity_pct",
    "wind_speed_ms",
    "pressure_hpa",
    "cloud_cover_okta",
    "solar_zenith_deg",
    "solar_elevation_deg",
    "clearsky_ghi_wm2",
    "clearsky_dni_wm2",
    "clearsky_dhi_wm2",
]

# Proxy target: the physical baseline's own output (src/physical_model.py).
# NOT real measured rail temperature.
PROXY_TARGET_COLUMN = "rail_temp_physical_c"

# Real target: expected column name once real measurements are obtained
# and merged into the feature dataframe by timestamp. Does not exist yet.
REAL_TARGET_COLUMN = "rail_temp_measured_c"


@dataclass
class TrainedModels:
    label: str  # "PROXY (physical-baseline-as-target)" or "REAL (measured data)"
    feature_columns: list[str]
    target_column: str
    rf_model: RandomForestRegressor
    ann_model: "RailTempANN"
    ann_x_scaler: StandardScaler
    ann_y_scaler: StandardScaler
    predictions: pd.DataFrame  # full-dataset predictions, all rows
    test_metrics: dict = field(default_factory=dict)


class RailTempANN(nn.Module):
    """Small feed-forward network: n_features -> 32 -> 16 -> 1."""

    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def chronological_split(df: pd.DataFrame, test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-ordered split (not random) -- appropriate for autocorrelated hourly data."""
    df = df.sort_index()
    n_test = int(len(df) * test_frac)
    return df.iloc[:-n_test], df.iloc[-n_test:]


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_ann(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    epochs: int = 500, lr: float = 1e-3, patience: int = 40, batch_size: int = 64,
) -> tuple[RailTempANN, StandardScaler, StandardScaler]:
    x_scaler = StandardScaler().fit(X_train)
    X_train_s = x_scaler.transform(X_train)
    X_val_s = x_scaler.transform(X_val)

    # Target is also standardized: rail temp spans ~9-80 C here, and MSE loss
    # on that raw scale converges far slower than on a standardized target.
    y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
    y_train_s = np.asarray(y_scaler.transform(y_train.reshape(-1, 1))).ravel()
    y_val_s = np.asarray(y_scaler.transform(y_val.reshape(-1, 1))).ravel()

    torch.manual_seed(42)
    model = RailTempANN(n_features=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_s, dtype=torch.float32)
    X_val_t = torch.tensor(X_val_s, dtype=torch.float32)
    y_val_t = torch.tensor(y_val_s, dtype=torch.float32)

    n = X_train_t.shape[0]
    generator = torch.Generator().manual_seed(42)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=generator)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad()
            pred = model(X_train_t[idx])
            loss = loss_fn(pred, y_train_t[idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t).item()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, x_scaler, y_scaler


def _predict_ann(model: RailTempANN, x_scaler: StandardScaler, y_scaler: StandardScaler,
                  X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        X_s = x_scaler.transform(X)
        pred_s = model(torch.tensor(X_s, dtype=torch.float32)).numpy()
        return y_scaler.inverse_transform(pred_s.reshape(-1, 1)).ravel()


def _train_and_evaluate(
    df: pd.DataFrame, feature_columns: list[str], target_column: str, label: str,
) -> TrainedModels:
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in dataframe. "
            f"Available columns: {list(df.columns)}"
        )
    missing_features = [c for c in feature_columns if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    work = df.dropna(subset=feature_columns + [target_column]).copy()
    train_df, test_df = chronological_split(work, test_frac=0.2)

    X_train_df = cast(pd.DataFrame, train_df[feature_columns])
    X_test_df = cast(pd.DataFrame, test_df[feature_columns])
    y_train_series = cast(pd.Series, train_df[target_column])
    y_test_series = cast(pd.Series, test_df[target_column])
    y_train = np.asarray(y_train_series.to_numpy(), dtype=float)
    y_test = np.asarray(y_test_series.to_numpy(), dtype=float)

    rf_model = train_random_forest(X_train_df, y_train_series)
    ann_model, ann_x_scaler, ann_y_scaler = train_ann(
        np.asarray(X_train_df.to_numpy(), dtype=float), y_train,
        np.asarray(X_test_df.to_numpy(), dtype=float), y_test,
    )

    rf_pred_test = rf_model.predict(X_test_df)
    ann_pred_test = _predict_ann(
        ann_model, ann_x_scaler, ann_y_scaler, np.asarray(X_test_df.to_numpy(), dtype=float)
    )

    test_metrics = {
        "random_forest": evaluate(y_test, rf_pred_test),
        "ann": evaluate(y_test, ann_pred_test),
        "n_train": len(train_df),
        "n_test": len(test_df),
    }

    full_X_df = cast(pd.DataFrame, work[feature_columns])
    predictions = pd.DataFrame(index=work.index)
    predictions[target_column] = work[target_column]
    predictions["rf_pred_c"] = rf_model.predict(full_X_df)
    predictions["ann_pred_c"] = _predict_ann(
        ann_model, ann_x_scaler, ann_y_scaler, np.asarray(full_X_df.to_numpy(), dtype=float)
    )
    predictions["split"] = ["train"] * len(train_df) + ["test"] * len(test_df)

    return TrainedModels(
        label=label,
        feature_columns=feature_columns,
        target_column=target_column,
        rf_model=rf_model,
        ann_model=ann_model,
        ann_x_scaler=ann_x_scaler,
        ann_y_scaler=ann_y_scaler,
        predictions=predictions,
        test_metrics=test_metrics,
    )


def train_and_evaluate_proxy_target(
    df: pd.DataFrame, feature_columns: list[str] = FEATURE_COLUMNS,
) -> TrainedModels:
    """Train RF + ANN against the PHYSICAL-BASELINE OUTPUT as a stand-in target.

    NOT validated against real measurements. Metrics measure only how well
    each model reproduces the physical formula, not real-world accuracy.
    """
    warnings.warn(
        "Training against physical-baseline-as-proxy target. Metrics do NOT "
        "reflect real-world rail temperature prediction accuracy.",
        stacklevel=2,
    )
    return _train_and_evaluate(
        df, feature_columns, PROXY_TARGET_COLUMN,
        label="PROXY (physical-baseline-as-target, NOT real-world validated)",
    )


def train_and_evaluate_real_target(
    df: pd.DataFrame, feature_columns: list[str] = FEATURE_COLUMNS,
    target_column: str = REAL_TARGET_COLUMN,
) -> TrainedModels:
    """Train + evaluate RF and ANN against REAL measured rail temperature.

    THIS IS THE FUNCTION TO USE ONCE REAL DATA IS OBTAINED. Merge measured
    rail temperature into `df` (indexed the same way as the weather feature
    dataframe, one value per timestamp) under a column named
    `target_column` (default: 'rail_temp_measured_c'), then call this
    function. Metrics it returns are then real, honest evaluation numbers.

    Raises a clear error if the target column is not present, so this
    can't silently run on the wrong data.
    """
    return _train_and_evaluate(
        df, feature_columns, target_column,
        label="REAL (trained on measured rail temperature)",
    )
