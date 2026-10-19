"""Leakage and artifact tests use synthetic data; no private dataset is needed."""
from pathlib import Path
import hashlib
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
import audit
from study_data import AnnualWindows, prepare_data
from study_models import LocalCompact


class SplitIsolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        time = pd.date_range("1980-01-01", "2025-01-01", freq="h", inclusive="left")
        h = np.arange(len(time), dtype=float)
        cls.frame = pd.DataFrame({"time":time,"solar":np.maximum(0,np.sin(2*np.pi*h/24))*.3,
                                  "wind":.2+.1*np.sin(h/37),"temperature":10+8*np.sin(h/1394)})

    def prepare(self, frame):
        # Bypass only the byte-identity gate for synthetic data. The real-data
        # CLI independently enforces the pinned SHA256 before preprocessing.
        pinned = SimpleNamespace(hexdigest=lambda:"d92748ca6625d0092b073b9bc16345da960a04c42da29926da5b125a63ad122d")
        with patch("study_data.Path.read_bytes",return_value=b"fixture"), \
             patch("study_data.hashlib.sha256",return_value=pinned), \
             patch("study_data.pd.read_parquet",return_value=frame):
            return prepare_data(Path("."),None,lazy=True)

    def test_future_weather_cannot_change_training_statistics_or_windows(self):
        tensors, info = self.prepare(self.frame)
        changed = self.frame.copy()
        mask = changed.time.dt.year >= 2016
        changed.loc[mask,["solar","wind","temperature"]] += [100.,50.,200.]
        other, future = self.prepare(changed)
        for index in (0,1,6,7):
            np.testing.assert_array_equal(tensors[index],other[index])
        for key in ("scales","night_mask","low_wind_threshold","high_wind_threshold"):
            np.testing.assert_array_equal(info[key],future[key])
        self.assertFalse(np.array_equal(info["test"],future["test"]))
        self.assertEqual(tensors[0].shape[-1],36*8760)
        windows = AnnualWindows(tensors[0],tensors[1])
        self.assertEqual(len(windows),1826)
        np.testing.assert_array_equal(windows[len(windows)-1][0],tensors[0][:,-8760:])

    def test_prior_generation_does_not_receive_future_weather(self):
        from generate_profiles import calendar_condition
        first, info = self.prepare(self.frame)
        np.testing.assert_array_equal(calendar_condition()[0], first[5][0])
        changed = self.frame.copy()
        changed.loc[changed.time.dt.year >= 2020,"wind"] *= 100
        second, other = self.prepare(changed)
        torch.set_num_threads(1)
        model = LocalCompact(info["night_mask"]).eval()
        with torch.no_grad():
            torch.manual_seed(101)
            a = model.sample(first[5][:1])
            torch.manual_seed(101)
            b = model.sample(second[5][:1])
        self.assertTrue(torch.equal(a,b))
        self.assertFalse(np.array_equal(info["test"],other["test"]))

    def test_reporting_reference_remains_unclipped(self):
        changed = self.frame.copy()
        changed.loc[changed.time.dt.year >= 2020,"wind"] = 2.
        tensors, info = self.prepare(changed)
        self.assertEqual(float(tensors[4][:,1].max()),1.)
        self.assertEqual(float(info["test"][:,1].max()),2.)


class ArtifactIsolation(unittest.TestCase):
    def fixture(self, root):
        run=root/"runs/example";run.mkdir(parents=True)
        (run/"best_weights.pth").write_bytes(b"weights must be checked before deserialization")
        np.savez(run/"normalization.npz",data_min=np.zeros((1,3,1)),data_max=np.ones((1,3,1)),night_mask=np.zeros(8760,dtype=bool))
        report={"status":"completed","checkpoint":{"file":"best_weights.pth","sha256":audit.sha256(run/"best_weights.pth")},
                "normalization_sha256":audit.sha256(run/"normalization.npz")}
        (run/"report.json").write_text(json.dumps(report))
        freeze={"checkpoints":[{"run":"runs/example","sha256":report["checkpoint"]["sha256"],
                                "normalization_sha256":report["normalization_sha256"]}]}
        return run,report,freeze

    def test_changed_normalization_is_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);run,report,freeze=self.fixture(root)
            audit.check_run(root,"runs/example",freeze)
            (run/"normalization.npz").write_bytes(b"changed normalization")
            with self.assertRaisesRegex(ValueError,"Normalization"):
                audit.check_run(root,"runs/example",freeze)

    def test_modified_report_cannot_override_frozen_normalization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);run,report,freeze=self.fixture(root)
            (run/"normalization.npz").write_bytes(b"changed normalization")
            report["normalization_sha256"]=audit.sha256(run/"normalization.npz")
            (run/"report.json").write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,"frozen"):
                audit.check_run(root,"runs/example",freeze)

    def test_partial_first_epoch_is_not_a_validated_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);run,report,freeze=self.fixture(root)
            report["checkpoint_partial_epoch"]=True
            (run/"report.json").write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,"partial"):
                audit.check_run(root,"runs/example",freeze)

    def test_escape_outside_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);run,report,freeze=self.fixture(root)
            report["checkpoint"]["file"]="../../outside.pth"
            (run/"report.json").write_text(json.dumps(report))
            with self.assertRaises(ValueError):audit.check_run(root,"runs/example",freeze)

    def test_known_upstream_overlap_cannot_be_labelled_strict_holdout(self):
        provenance=json.loads((audit.ROOT/"data_provenance.json").read_text())
        self.assertFalse(audit.input_provenance(provenance)["strict_temporal_holdout"])
        with self.assertRaisesRegex(ValueError,"Upstream calibration"):
            audit.input_provenance(provenance,strict=True)

    def test_freeze_rejects_reporting_summary_as_validation(self):
        import freeze_evaluation
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);run,report,freeze=self.fixture(root)
            output=root/"exports/validation";output.mkdir(parents=True)
            (output/"summary.json").write_text(json.dumps({"split":"test","checkpoint_sha256":report["checkpoint"]["sha256"]}))
            arguments=["freeze_evaluation.py","--runs","runs/example","--selected-run","runs/example","--validation-dir","exports/validation","--output","freeze.json"]
            with patch.object(freeze_evaluation,"ROOT",root),patch("sys.argv",arguments):
                with self.assertRaisesRegex(ValueError,"validation output"):
                    freeze_evaluation.main()
            self.assertFalse((root/"freeze.json").exists())

    def test_new_prior_validation_can_be_frozen_without_moving_files(self):
        import freeze_evaluation
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);run,report,freeze=self.fixture(root)
            output=root/"exports/validation";output.mkdir(parents=True)
            (output/"summary.json").write_text(json.dumps({"split":"validation","checkpoint_sha256":report["checkpoint"]["sha256"],"normalization_sha256":report["normalization_sha256"]}))
            for name in ("overnight_protocol.json","study_runner.py","study_models.py","study_data.py","study_metrics.py","latent_prior.py"):
                (root/name).write_text("synthetic source fixture")
            arguments=["freeze_evaluation.py","--runs","runs/example","--selected-run","runs/example","--validation-dir","exports/validation","--output","freeze.json"]
            with patch.object(freeze_evaluation,"ROOT",root),patch("sys.argv",arguments):
                freeze_evaluation.main()
            recorded=json.loads((root/"freeze.json").read_text())
            self.assertEqual(recorded["checkpoints"][0]["normalization_sha256"],report["normalization_sha256"])
            self.assertFalse(recorded["test_based_tuning_allowed"])


if __name__=="__main__":unittest.main(verbosity=2)
