"""Checked chronological data and training-only metric/reference statistics."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


class AnnualWindows:
    """Read-only overlapping views; retain one time series instead of 1826 copies."""
    def __init__(self, values, conditions, length=8760, stride=168):
        if values.shape[-1] != conditions.shape[-1] or length < 1 or stride < 1:
            raise ValueError("Invalid window dimensions")
        self.values, self.conditions = values, conditions
        self.length, self.stride = length, stride
        self.count = max(0, (values.shape[-1] - length) // stride + 1)

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if not 0 <= index < self.count:
            raise IndexError(index)
        start = index * self.stride
        return self.values[:, start:start + self.length], self.conditions[:, start:start + self.length]


def prepare_data(root, legacy, stride=168, lazy=False, data_path=None):
    root = Path(root)
    path = Path(data_path) if data_path is not None else root / "data/HR_data.parquet"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != "9a21558378d6359f308e3b7b7a3068d053195fa9123792c1d8aa9ac177101022":
        raise ValueError("Dataset differs from the frozen study input")
    frame = pd.read_parquet(path)
    dates = pd.to_datetime(frame["time"])
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Timestamps must be unique and ordered")
    if not (dates.diff().dropna() == pd.Timedelta(hours=1)).all():
        raise ValueError("Input must be a complete hourly timeline before removing leap days")
    frame = frame.loc[~((dates.dt.month == 2) & (dates.dt.day == 29))].copy()
    years = pd.to_datetime(frame["time"]).dt.year.to_numpy()
    raw = frame[["solar", "wind", "temperature"]].to_numpy(dtype=np.float64)
    if not np.isfinite(raw).all() or raw[:, 1].min() <= -1:
        raise ValueError("Invalid source values")
    if sorted(np.unique(years).tolist()) != list(range(1980, 2025)):
        raise ValueError("Expected complete 1980-2024 input")
    if any((years == year).sum() != 8760 for year in range(1980, 2025)):
        raise ValueError("Each non-leap year must have 8760 hours")
    annual = raw.reshape(45, 8760, 3).transpose(0, 2, 1)
    train = annual[:36]
    if lazy:
        import torch
        transformed = raw.copy()
        transformed[:, 1] = np.log1p(transformed[:, 1])
        low = transformed[:36 * 8760].min(axis=0).reshape(1, 3)
        high = transformed[:36 * 8760].max(axis=0).reshape(1, 3)
        width = np.clip(high - low, 1e-9, None)

        def split(first, last, training):
            values = np.clip((transformed[first * 8760:last * 8760] - low) / width, 0, 1)
            h = np.arange(len(values))
            condition = np.stack((np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                                  np.sin(2 * np.pi * h / 8760), np.cos(2 * np.pi * h / 8760)), axis=1)
            if training:
                return torch.from_numpy(values.astype(np.float32).T.copy()), torch.from_numpy(condition.astype(np.float32).T.copy())
            return (torch.from_numpy(values.reshape(-1, 8760, 3).astype(np.float32)).transpose(1, 2),
                    torch.from_numpy(condition.reshape(-1, 8760, 4).astype(np.float32)).transpose(1, 2))
        tr, tc = split(0, 36, True)
        va, vc = split(36, 40, False)
        te, ec = split(40, 45, False)
        tensors = tr, tc, va, vc, te, ec, low.reshape(1, 3, 1), high.reshape(1, 3, 1)
    else:
        tensors = legacy.load_data(str(path), stride_hours=stride)
    return tensors, {"data_sha256": digest, "train": train, "validation": annual[36:40],
                     "test": annual[40:], "scales": train.std(axis=(0, 2)),
                     "night_mask": train[:, 0].max(axis=0) <= .005,
                     "low_wind_threshold": float(np.percentile(train[:, 1], 10)),
                     "high_wind_threshold": float(np.percentile(train[:, 1], 90))}


def physical(values, low, high):
    result = np.asarray(values, dtype=np.float64) * (high - low) + low
    result[:, 1] = np.expm1(result[:, 1])
    return result
