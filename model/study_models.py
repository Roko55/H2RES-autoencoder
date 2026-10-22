"""Conditional variational autoencoders and annual-profile loss functions."""
import importlib.util
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent


def legacy_module(device="cpu"):
    spec = importlib.util.spec_from_file_location("preserved_legacy", ROOT / "src/ai_engine.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.DEVICE = torch.device(device)
    return module


class GlobalCompact(nn.Module):
    """Global-latent annual CVAE with 691,639 trainable parameters."""
    latent_dim = 32

    def __init__(self, night_mask=None):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(7, 16, 7, 2, 3), nn.LeakyReLU(.2),
            nn.Conv1d(16, 32, 7, 2, 3), nn.LeakyReLU(.2),
            nn.Conv1d(32, 48, 7, 2, 3), nn.LeakyReLU(.2),
            nn.AdaptiveAvgPool1d(12), nn.Flatten(), nn.Linear(576, 128), nn.LeakyReLU(.2))
        self.mu = nn.Linear(128, 32)
        self.lv = nn.Linear(128, 32)
        self.input = nn.Linear(32, 48 * 365)
        self.dec = nn.ModuleList([nn.Conv1d(52, 32, 7, padding=3),
                                  nn.Conv1d(36, 16, 7, padding=3), nn.Conv1d(20, 3, 7, padding=3)])

    def decode(self, z, c):
        h = self.input(z).reshape(-1, 48, 365)
        for i, (layer, size) in enumerate(zip(self.dec, (1095, 4380, 8760))):
            h = layer(torch.cat([F.interpolate(h, size=size, mode="linear", align_corners=False),
                                F.interpolate(c, size=size, mode="linear", align_corners=False)], 1))
            h = torch.sigmoid(h) if i == 2 else F.leaky_relu(h, .2)
        return h

    def forward(self, x, c):
        h = self.enc(torch.cat([x, c], 1))
        mu, lv = self.mu(h), self.lv(h)
        return self.decode(mu + torch.randn_like(mu) * torch.exp(.5 * lv.clamp(-10, 5)), c), None, mu, lv

    def sample(self, c, temperature=1.):
        return self.decode(torch.randn(c.size(0), 32, device=c.device) * temperature, c)


class DilatedBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
                                 nn.GroupNorm(4, channels), nn.SiLU(), nn.Conv1d(channels, channels, 1))

    def forward(self, x):
        return F.silu(x + self.net(x))


class LocalCompact(nn.Module):
    """Fully convolutional annual CVAE with temporal latents and calendar inputs.

    Independent latent innovations are transformed by dilated convolutions;
    no held-out weather values or training-year resampling enter its sampler.
    The solar zero mask is estimated only from training calendar hours.
    """
    latent_dim = 8

    def __init__(self, night_mask=None, latent_step_hours=8):
        super().__init__()
        if latent_step_hours not in (2, 8):
            raise ValueError("The study supports 2-hour and 8-hour temporal latents")
        self.latent_step_hours = latent_step_hours
        self.latent_dim = latent_step_hours
        self.latent_length = 8760 // latent_step_hours
        later_stride = 2 if latent_step_hours == 8 else 1
        self.enc = nn.Sequential(nn.Conv1d(7, 24, 7, 2, 3), nn.SiLU(),
                                 nn.Conv1d(24, 48, 7, later_stride, 3), nn.SiLU(),
                                 nn.Conv1d(48, 64, 7, later_stride, 3), nn.SiLU(),
                                 DilatedBlock(64, 1), DilatedBlock(64, 4))
        self.mu = nn.Conv1d(64, self.latent_dim, 1)
        self.lv = nn.Conv1d(64, self.latent_dim, 1)
        nn.init.zeros_(self.lv.weight)
        nn.init.zeros_(self.lv.bias)
        self.latent_net = nn.Sequential(nn.Conv1d(self.latent_dim + 4, 64, 3, padding=1), nn.SiLU(),
                                        *(DilatedBlock(64, d) for d in (1, 4, 16, 64)))
        self.decoder = nn.ModuleList([nn.Conv1d(68, 48, 7, padding=3),
                                      nn.Conv1d(52, 24, 7, padding=3), nn.Conv1d(28, 16, 7, padding=3)])
        self.output = nn.Conv1d(16, 3, 7, padding=3)
        self.decoder_sizes = (2190, 4380, 8760) if latent_step_hours == 8 else (4380, 4380, 8760)
        self.register_buffer("night_mask", torch.zeros(8760, dtype=torch.bool) if night_mask is None
                             else torch.as_tensor(night_mask, dtype=torch.bool))

    def decode(self, z, c):
        h = self.latent_net(torch.cat([z, F.interpolate(c, size=z.shape[-1], mode="linear", align_corners=False)], 1))
        for layer, size in zip(self.decoder, self.decoder_sizes):
            h = F.silu(layer(torch.cat([F.interpolate(h, size=size, mode="linear", align_corners=False),
                                       F.interpolate(c, size=size, mode="linear", align_corners=False)], 1)))
        out = torch.sigmoid(self.output(h))
        # Phase comes from the existing calendar condition, including shifted windows.
        hour = torch.remainder(torch.round(torch.atan2(c[:, 2], c[:, 3]) * (8760 / (2 * torch.pi))).long(), 8760)
        solar = out[:, 0] * (~self.night_mask[hour]).to(out.dtype)
        return torch.stack((solar, out[:, 1], out[:, 2]), 1)

    def forward(self, x, c):
        h = self.enc(torch.cat([x, c], 1))
        mu, lv = self.mu(h), self.lv(h).clamp(-10, 5)
        return self.decode(mu + torch.randn_like(mu) * torch.exp(.5 * lv), c), None, mu, lv

    def sample(self, c, temperature=1.):
        return self.decode(torch.randn(c.size(0), self.latent_dim, self.latent_length, device=c.device) * temperature, c)


def make_model(variant, legacy, night_mask=None):
    if variant == "compact_global_legacy":
        return GlobalCompact()
    if variant in ("compact_local_balanced", "compact_local_prior_matching"):
        return LocalCompact(night_mask)
    if variant == "compact_local_hires_prior_matching":
        return LocalCompact(night_mask, latent_step_hours=2)
    if variant == "legacy_original":
        return legacy.CVAE_1D()
    if variant == "legacy_balanced":
        class ConsistentLegacy(legacy.CVAE_1D):
            def forward(self, x, cond):
                h = self.enc_net(torch.cat([x, cond], dim=1))
                mu, lv = self.fc_mu(h), self.fc_logvar(h).clamp(-10, 5)
                return self.decode(self.reparameterize(mu, lv), cond), None, mu, lv
        return ConsistentLegacy()
    raise ValueError(f"Unknown variant: {variant}")


def balanced_loss(pred, target, mu, lv, beta):
    """Dimension-normalized reconstruction, temporal/distribution losses and KL.

    KL and sampling use the same bounded log variance. Loss terms are averaged
    per channel and sequence instead of scaling with annual FFT magnitude.
    """
    mse = (pred - target).square().mean()
    gradient = ((pred[..., 1:] - pred[..., :-1]) - (target[..., 1:] - target[..., :-1])).abs().mean()
    moments = (pred.mean(-1) - target.mean(-1)).square().mean()
    moments = moments + (pred.std(-1) - target.std(-1)).square().mean()
    spectrum = pred.new_zeros(())
    for scale in (1, 24, 168):
        p, t = (pred, target) if scale == 1 else (F.avg_pool1d(pred, scale), F.avg_pool1d(target, scale))
        fp, ft = torch.fft.rfft(p, norm="ortho").abs(), torch.fft.rfft(t, norm="ortho").abs()
        spectrum = spectrum + (torch.log1p(fp) - torch.log1p(ft)).abs().mean() / 3
    # A small sorted-value term targets marginals without forcing unrelated peaks to align.
    marginal = (pred.sort(dim=-1).values - target.sort(dim=-1).values).abs().mean()
    kl = .5 * (mu.square() + lv.exp() - 1 - lv).mean()
    reconstruction = mse + .25 * gradient + 2 * moments + .25 * spectrum + .25 * marginal
    return reconstruction + beta * kl, reconstruction, kl


def prior_statistics_loss(generated, target):
    """Unpaired training-prior constraints; no hourly alignment of random weather.

    Match annual marginal distributions, spectral magnitude, persistence and
    channel dependence using training targets only.
    """
    moments = (generated.mean(-1) - target.mean(-1)).square().mean()
    moments = moments + (generated.std(-1) - target.std(-1)).square().mean()
    marginal = (generated.sort(dim=-1).values - target.sort(dim=-1).values).abs().mean()
    spectral = generated.new_zeros(())
    for scale in (1, 24, 168):
        g, t = (generated, target) if scale == 1 else (F.avg_pool1d(generated, scale), F.avg_pool1d(target, scale))
        fg, ft = torch.fft.rfft(g, norm="ortho").abs(), torch.fft.rfft(t, norm="ortho").abs()
        spectral = spectral + (torch.log1p(fg) - torch.log1p(ft)).abs().mean() / 3

    def correlation(a, b):
        a, b = a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True)
        return (a * b).mean(-1) / (a.square().mean(-1) * b.square().mean(-1)).clamp_min(1e-12).sqrt()

    persistence = generated.new_zeros(())
    for lag in (1, 24, 168):
        persistence = persistence + (correlation(generated[..., :-lag], generated[..., lag:])
            - correlation(target[..., :-lag], target[..., lag:])).abs().mean() / 3
    joint = generated.new_zeros(())
    for a, b in ((0, 1), (0, 2), (1, 2)):
        joint = joint + (correlation(generated[:, a], generated[:, b])
            - correlation(target[:, a], target[:, b])).square().mean() / 3
    return 2 * moments + .5 * marginal + .25 * spectral + .1 * persistence + .1 * joint
