"""Meaningful compact-model and metric checks; no historical-size allocation."""
import os
os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="-1")
import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
import json
import numpy as np
import torch
from study_models import GlobalCompact, LocalCompact, balanced_loss, prior_statistics_loss
from study_metrics import episodes, path_metrics
from latent_prior import fit_gaussian
from study_data import AnnualWindows


def calendar(start=0):
    h = torch.arange(start, start + 8760, dtype=torch.float32)
    return torch.stack((torch.sin(2 * torch.pi * h / 24), torch.cos(2 * torch.pi * h / 24),
                        torch.sin(2 * torch.pi * h / 8760), torch.cos(2 * torch.pi * h / 8760)))[None]


class ScientificMetrics(unittest.TestCase):
    def test_lazy_windows_keep_stride_and_ownership(self):
        values = torch.arange(120).reshape(3, 40)
        conditions = torch.arange(160).reshape(4, 40)
        data = AnnualWindows(values, conditions, length=8, stride=3)
        self.assertEqual(len(data), 11)
        x, c = data[3]
        self.assertTrue(torch.equal(x, values[:, 9:17]))
        self.assertTrue(torch.equal(c, conditions[:, 9:17]))
        self.assertEqual(x.untyped_storage().data_ptr(), values.untyped_storage().data_ptr())
        with self.assertRaises(IndexError):
            data[len(data)]
    def test_aggregate_prior_covariance_and_rejection(self):
        mu = np.array([[1., 2.], [3., 4.], [2., 3.]])
        mean, factor = fit_gaussian(mu, np.zeros_like(mu))
        np.testing.assert_allclose(mean, [2., 3.])
        self.assertTrue(np.all(np.linalg.eigvalsh(factor @ factor.T) > 0))
        with self.assertRaises(ValueError):
            fit_gaussian(mu[:1], np.zeros_like(mu[:1]))
    def test_identity_and_episode_boundaries(self):
        t = np.arange(8760)
        values = np.stack((np.maximum(0, np.sin(t / 24)), .2 + .1 * np.sin(t / 48), 10 + 5 * np.sin(t / 100)))
        rows, _ = path_metrics(values, values.copy(), np.ones(3), np.zeros(8760, bool), 2016,
                               "reconstruction", 101, 1., .1, .3)
        self.assertTrue(all(r["selection_component"] < 1e-12 for r in rows))
        self.assertTrue(all(r["paired_MAE"] == 0 for r in rows))
        self.assertEqual(episodes([True, True, False, True])["max_hours"], 2)
        self.assertEqual(episodes([True, True, False, True])["count"], 2)
        self.assertEqual(episodes([False, False])["max_hours"], 0)

    def test_constant_prior_does_not_hide_missing_acf(self):
        rng = np.random.default_rng(17)
        real = rng.random((3, 8760))
        rows, _ = path_metrics(real, np.zeros_like(real), np.ones(3), np.zeros(8760, bool),
                               2016, "prior", 101, 1., .1, .9)
        self.assertTrue(all(r["selection_component"] >= .5 for r in rows))
        self.assertTrue(all("paired_MAE" not in r for r in rows))


class CompactModels(unittest.TestCase):
    def test_hires_latents_preserve_annual_size_and_gradients(self):
        torch.manual_seed(17)
        model = LocalCompact(latent_step_hours=2)
        c = calendar()
        pred, _, mu, lv = model(torch.rand(1, 3, 8760), c)
        self.assertEqual(tuple(mu.shape), (1, 2, 4380))
        self.assertEqual(tuple(pred.shape), (1, 3, 8760))
        self.assertEqual(mu.numel(), 8760)
        loss, _, _ = balanced_loss(pred, torch.rand_like(pred), mu, lv, .02)
        loss = loss + .5 * prior_statistics_loss(model.sample(c), torch.rand_like(pred))
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
    def test_unpaired_prior_statistics_have_finite_gradient(self):
        torch.manual_seed(17)
        generated = torch.rand(1, 3, 8760, requires_grad=True)
        target = torch.rand_like(generated)
        loss = prior_statistics_loss(generated, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(generated.grad).all())
        self.assertGreater(float(generated.grad.abs().sum()), 0)
        constant = torch.full((1, 3, 8760), .5, requires_grad=True)
        prior_statistics_loss(constant, target).backward()
        self.assertTrue(torch.isfinite(constant.grad).all())
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_global_parameter_count_and_seeded_prior(self):
        model = GlobalCompact().eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 691639)
        with torch.no_grad():
            torch.manual_seed(101)
            first = model.sample(calendar())
            torch.manual_seed(101)
            second = model.sample(calendar())
        self.assertEqual(tuple(first.shape), (1, 3, 8760))
        self.assertTrue(torch.equal(first, second))

    def test_local_calendar_shift_gradients_and_prior(self):
        torch.manual_seed(17)
        night = np.arange(8760) % 24 < 6
        model = LocalCompact(night)
        c = calendar(168)
        x = torch.rand(1, 3, 8760)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        pred, _, mu, lv = model(x, c)
        self.assertEqual(tuple(pred.shape), (1, 3, 8760))
        self.assertEqual(tuple(mu.shape), (1, 8, 1095))
        self.assertTrue(torch.equal(pred[:, 0, :6], torch.zeros_like(pred[:, 0, :6])))
        loss, _, kl = balanced_loss(pred, x, mu, lv, .02)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(kl.detach()), 0)
        before = model.output.weight.detach().clone()
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(before, model.output.weight))
        model.eval()
        with torch.no_grad():
            torch.manual_seed(101)
            a = model.sample(c)
            torch.manual_seed(102)
            b = model.sample(c)
        self.assertTrue(torch.isfinite(a).all())
        self.assertGreater(float((a - b).abs().mean()), 0)
        model.train()
        model.zero_grad(set_to_none=True)
        prior_statistics_loss(model.sample(c), x).backward()
        self.assertTrue(torch.isfinite(model.output.weight.grad).all())
        self.assertGreater(float(model.output.weight.grad.abs().sum()), 0)


class InferenceFailures(unittest.TestCase):
    def run_failure(self, checkpoint_exists):
        import generate_profiles
        scratch = Path(__file__).resolve().parent / "tmp"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temporary:
            root = Path(temporary)
            root.resolve().relative_to(scratch.resolve())  # Verify owned cleanup target before recursive removal.
            run = root / "runs/test"
            run.mkdir(parents=True)
            (run / "report.json").write_text(json.dumps({"checkpoint": {"file": "missing.pth", "sha256": "0" * 64}}))
            if checkpoint_exists:
                (run / "missing.pth").write_bytes(b"mismatched checkpoint; must never reach deserialization")
            with patch.object(generate_profiles, "ROOT", root), patch.object(generate_profiles, "ResourceGuard"), \
                 patch("sys.argv", ["generate_profiles.py", "--run", "runs/test"]):
                with self.assertRaises(ValueError if checkpoint_exists else FileNotFoundError):
                    generate_profiles.main()
            self.assertFalse((run / "generated_profiles.npy").exists())

    def test_missing_weights_never_generate_zeros(self):
        self.run_failure(False)

    def test_mismatched_weights_never_reach_loader(self):
        self.run_failure(True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
