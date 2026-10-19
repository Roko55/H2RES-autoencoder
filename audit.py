"""Check checkpoint provenance and distinguish model isolation from input calibration.

This audit does not train, select a checkpoint, or claim an untouched test set.
The original source and executed reports remain available for comparison.
"""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parent
SELECTED_RUN = "runs/study_confirmation_compact_local_hires_prior_matching_seed17_3600s"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inside(path, parent):
    """Resolve before checking scope, including any symlink components."""
    resolved = Path(path).resolve()
    resolved.relative_to(Path(parent).resolve())
    return resolved


def input_provenance(provenance, strict=False):
    end = provenance["training_years"][1]
    overlap = {channel: [year for year in record["calibration_years"] if year > end]
               for channel, record in provenance["upstream_calibration"].items()}
    independent = not any(overlap.values())
    if strict and not independent:
        raise ValueError("Upstream calibration uses post-training years; strict temporal holdout is not supported")
    return {"strict_temporal_holdout": independent, "upstream_post_training_years": overlap,
            "evaluation_scope": "retrospective comparison on a shared pre-calibrated corpus"}


def check_manifest(root):
    root = Path(root)
    manifest = json.loads((root / "artifact_manifest.json").read_text())
    for name, record in manifest["files"].items():
        path = inside(root / name, root)
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError("Published artifact differs from manifest: " + name)
    return len(manifest["files"])


def check_run(root, run_name, freeze=None):
    """Reject a changed checkpoint or normalization before loading any weights."""
    import numpy as np
    root = Path(root)
    run = inside(root / run_name, root / "runs")
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    if report["status"] not in ("completed", "budget_stopped") or report.get("checkpoint_partial_epoch"):
        raise ValueError("Evaluation requires a completed validation checkpoint, not a partial first epoch")
    recorded = report["checkpoint"]
    checkpoint = inside(run / recorded.get("file", recorded.get("filename", "best_weights.pth")), run)
    normalization = run / "normalization.npz"
    actual, norm_hash = sha256(checkpoint), sha256(normalization)
    if actual != recorded["sha256"]:
        raise ValueError("Checkpoint differs from the training record")
    if not report.get("normalization_sha256") or norm_hash != report["normalization_sha256"]:
        raise ValueError("Normalization differs from the training record")
    if freeze is not None:
        entries = {entry["run"]: entry for entry in freeze["checkpoints"]}
        entry = entries.get(run_name)
        if not entry or actual != entry["sha256"] or norm_hash != entry.get("normalization_sha256"):
            raise ValueError("Checkpoint or normalization differs from the frozen record")
    with np.load(normalization, allow_pickle=False) as archive:
        low, high, night = archive["data_min"].copy(), archive["data_max"].copy(), archive["night_mask"].copy()
    if low.size != 3 or high.size != 3 or night.shape != (8760,) or night.dtype != np.bool_:
        raise ValueError("Unexpected normalization or calendar-mask dimensions")
    if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(high <= low):
        raise ValueError("Invalid normalization bounds")
    return report, checkpoint, low.reshape(1, 3, 1), high.reshape(1, 3, 1), night


def check_dataset(path, low, high, night):
    """Recompute fitting statistics from 1980–2015 only and compare exact values."""
    import numpy as np
    from study_data import prepare_data
    tensors, info = prepare_data(ROOT, None, lazy=True, data_path=path)
    np.testing.assert_array_equal(tensors[-2], low)
    np.testing.assert_array_equal(tensors[-1], high)
    np.testing.assert_array_equal(info["night_mask"], night)
    return tensors, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=SELECTED_RUN)
    parser.add_argument("--data", help="Optional exact Parquet; audit train-only statistics too")
    parser.add_argument("--strict-holdout", action="store_true", help="Reject the supplied corpus's post-training calibration")
    parser.add_argument("--verify-files", action="store_true", help="Verify every delivered file against artifact_manifest.json")
    args = parser.parse_args()
    provenance = json.loads((ROOT / "data_provenance.json").read_text())
    result = input_provenance(provenance, strict=args.strict_holdout)
    if args.verify_files:
        result["manifest_files_verified"] = check_manifest(ROOT)
    freeze = json.loads((ROOT / "provenance/test_freeze.json").read_text())
    # The selected and consumer checkpoints belong to the original study freeze.
    if "faculty_cpu900" in args.run:
        freeze = json.loads((ROOT / "provenance/cpu900_freeze.json").read_text())
    report, checkpoint, low, high, night = check_run(ROOT, args.run, freeze)
    if args.data:
        check_dataset(args.data, low, high, night)
    result.update(status="passed_with_documented_input_limitation", checkpoint_sha256=sha256(checkpoint),
                  normalization_sha256=report["normalization_sha256"],
                  training_statistics_recomputed=bool(args.data), checkpoint_and_normalization_verified=True)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
