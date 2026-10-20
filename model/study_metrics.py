"""Common individual-path evaluation, with training-only scale and night mask."""
import numpy as np
from scipy.stats import wasserstein_distance

CHANNELS = ("solar", "wind", "temperature")


def acf(values, lag):
    a, b = values[:-lag], values[lag:]
    if min(a.std(), b.std()) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def episodes(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(int))
    lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    return {"count": int(len(lengths)), "mean_hours": float(lengths.mean()) if len(lengths) else 0.,
            "p95_hours": float(np.percentile(lengths, 95)) if len(lengths) else 0.,
            "max_hours": int(lengths.max()) if len(lengths) else 0}


def path_metrics(real, generated, scales, night_mask, year, mode, seed, temperature,
                 low_wind_threshold, high_wind_threshold):
    if real.shape != (3, 8760) or generated.shape != real.shape:
        raise ValueError("Metrics require one annual path with shape [3,8760]")
    if not np.isfinite(generated).all():
        raise FloatingPointError("Nonfinite generated path")
    rows = []
    for k, channel in enumerate(CHANNELS):
        r, g = real[k], generated[k]
        row = {"year": int(year), "mode": mode, "sample_seed": int(seed), "temperature": float(temperature),
               "channel": channel, "wasserstein": float(wasserstein_distance(r, g)),
               "train_std": float(scales[k]), "real_mean": float(r.mean()), "generated_mean": float(g.mean()),
               "real_std": float(r.std()), "generated_std": float(g.std()),
               "generated_negative_fraction": float((g < 0).mean())}
        for q in (1, 5, 50, 95, 99, 100):
            row[f"real_p{q}"] = float(np.percentile(r, q))
            row[f"generated_p{q}"] = float(np.percentile(g, q))
        errors = []
        for lag in (1, 24, 168):
            ra, ga = acf(r, lag), acf(g, lag)
            row[f"real_ACF_{lag}"] = ra
            row[f"generated_ACF_{lag}"] = ga
            # A constant synthetic series is an invalid persistence match, not a missing easy score.
            errors.append(abs(ga - ra) if ga is not None and ra is not None else 1.)
        for name, x in (("real", r), ("generated", g)):
            row[f"{name}_ramp_abs_p95"] = float(np.percentile(np.abs(np.diff(x)), 95))
        if mode in ("reconstruction", "posterior_mean"):
            row["paired_MAE"] = float(np.abs(g - r).mean())
        if channel == "wind":
            for label, threshold, operation in (("low", low_wind_threshold, np.less_equal),
                                                  ("high", high_wind_threshold, np.greater_equal)):
                for name, x in (("real", r), ("generated", g)):
                    row.update({f"{name}_{label}_wind_{key}": value
                                for key, value in episodes(operation(x, threshold)).items()})
        if channel == "solar":
            row["training_calendar_night_hours"] = int(np.asarray(night_mask).sum())
            if np.any(night_mask):
                row["generated_calendar_night_mean"] = float(np.abs(g[night_mask]).mean())
                row["real_calendar_night_mean"] = float(np.abs(r[night_mask]).mean())
        row["selection_component"] = (row["wasserstein"] / max(scales[k], 1e-9)
            + .25 * abs(g.std() / max(r.std(), 1e-9) - 1)
            + .25 * abs(row["generated_p99"] - row["real_p99"]) / max(scales[k], 1e-9)
            + .5 * float(np.mean(errors)))
        rows.append(row)
    joint = {"year": int(year), "mode": mode, "sample_seed": int(seed), "temperature": float(temperature)}
    for a, b in ((0, 1), (0, 2), (1, 2)):
        for name, values in (("real", real), ("generated", generated)):
            joint[f"{name}_corr_{CHANNELS[a]}_{CHANNELS[b]}"] = (
                float(np.corrcoef(values[a], values[b])[0, 1])
                if min(values[a].std(), values[b].std()) > 1e-12 else None)
    return rows, joint
