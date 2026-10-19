"""Train-only Gaussian approximation to an aggregated global posterior.

This is an explicit post-hoc sampler diagnostic, not the standard VAE prior.
It does not modify decoder weights or use validation/test weather for fitting.
"""
import numpy as np


def fit_gaussian(mu, logvar, shrinkage=.1):
    mu, logvar = np.asarray(mu, dtype=np.float64), np.asarray(logvar, dtype=np.float64)
    if mu.ndim != 2 or mu.shape != logvar.shape or mu.shape[0] < 2 or mu.shape[1] > 128:
        raise ValueError("Expected at least two global posterior vectors, dimension <=128")
    if not np.isfinite(mu).all() or not np.isfinite(logvar).all() or not 0 <= shrinkage <= 1:
        raise ValueError("Invalid posterior parameters")
    empirical = np.atleast_2d(np.cov(mu, rowvar=False))
    covariance = (1 - shrinkage) * empirical + shrinkage * np.diag(np.diag(empirical))
    covariance += np.diag(np.exp(np.clip(logvar, -10, 5)).mean(axis=0))
    covariance += np.eye(mu.shape[1]) * 1e-6
    return mu.mean(axis=0), np.linalg.cholesky(covariance)


def sample_gaussian(model, condition, mean, cholesky):
    import torch
    mean = torch.as_tensor(mean, dtype=condition.dtype, device=condition.device)
    factor = torch.as_tensor(cholesky, dtype=condition.dtype, device=condition.device)
    noise = torch.randn(condition.shape[0], mean.numel(), device=condition.device, dtype=condition.dtype)
    return model.decode(mean[None] + noise @ factor.T, condition)
