"""Train or evaluate the frozen, time-bounded H2RES CVAE study.

Training never reads test metrics. Evaluation uses individual annual paths,
unclipped physical references and training-only normalization/statistics.
"""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MPLBACKEND="Agg",
                  MPLCONFIGDIR=str(ROOT / "cache/matplotlib"), CUDA_CACHE_PATH=str(ROOT / "cache/cuda"),
                  TEMP=str(ROOT / "tmp"), TMP=str(ROOT / "tmp"), PYTHONDONTWRITEBYTECODE="1")

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import platform
import time
import traceback

from study_runtime import OwnBudgetReached, ResourceGuard, atomic_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure(device):
    for relative in ("tmp", "cache/matplotlib", "cache/cuda"):
        (ROOT / relative).mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "vendor"))
    if device == "cuda":
        sys.path.insert(0, str(ROOT / "vendor_cuda"))
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" if device == "cuda" else "-1"
    import numpy as np
    import torch
    import pyarrow as pa
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    return np, torch


def new_run(path):
    out = (ROOT / path).resolve()
    out.relative_to((ROOT / "runs").resolve())
    out.mkdir(parents=True, exist_ok=False)
    return out


def train(args):
    out = new_run(args.output)
    report = {"status": "starting", "variant": args.variant, "training_seed": args.seed,
              "started_utc": datetime.now(timezone.utc).isoformat(), "history": [],
              "test_evaluated": False, "configuration": vars(args).copy(),
              "implementation_sha256": {n: digest(ROOT / n) for n in
                  ("study_models.py", "study_data.py", "study_runner.py", "study_runtime.py", "overnight_protocol.json")}}
    atomic_json(out / "report.json", report)
    model = None
    best = float("inf")
    started = time.monotonic()
    try:
        guard = ResourceGuard(args.device, args.seconds)
        np, torch = configure(args.device)
        import psutil
        from study_models import legacy_module, make_model, balanced_loss, prior_statistics_loss
        from study_data import prepare_data, AnnualWindows
        if args.device == "cuda":
            report["GPU"] = guard.configure_cuda()
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        legacy = legacy_module(args.device)
        if digest(ROOT / "src/ai_engine.py") != "1b20e3210a4431921e17ba03046e51761dc794e2e878829527123eecba5c5c48":
            raise ValueError("Historical source hash changed")
        lazy = args.variant in ("compact_local_balanced", "compact_local_prior_matching", "compact_local_hires_prior_matching")
        match_prior = args.variant in ("compact_local_prior_matching", "compact_local_hires_prior_matching")
        tensors, info = prepare_data(ROOT, legacy, lazy=lazy)
        tr, tc, va, vc, _, _, low, high = tensors
        del tensors
        guard.check()
        model = make_model(args.variant, legacy, info["night_mask"]).to(args.device)
        original_loss = args.variant in ("legacy_original", "compact_global_legacy")
        batch = 4 if original_loss or args.variant == "legacy_balanced" else 8
        lr = 3e-5 if original_loss else (1e-4 if args.variant == "legacy_balanced" else 3e-4)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=2e-3 if original_loss else 1e-4)
        # Keep the learning-rate schedule independent of the wall-time limit.
        schedule_epochs = 100 if original_loss else 300
        effective_epochs = min(args.epochs, 100) if original_loss else args.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=schedule_epochs, eta_min=1e-6)
        dataset = AnnualWindows(tr, tc) if lazy else torch.utils.data.TensorDataset(tr, tc)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True)
        report.update(status="training", parameters=sum(p.numel() for p in model.parameters()),
                      hardware={"platform": platform.platform(), "logical_CPUs": os.cpu_count(),
                                "processor": platform.processor(), "total_RAM_GiB": psutil.virtual_memory().total / 2 ** 30,
                                "reserved_free_RAM_GiB": guard.reserve / 2 ** 30},
                      latent_elements_per_annual_path=8760 if lazy else model.latent_dim,
                      temporal_latent_step_hours=getattr(model, "latent_step_hours", None),
                      torch=torch.__version__, numpy=np.__version__, python=sys.version,
                      train_windows=len(dataset), batch_size=batch, learning_rate=lr,
                      data_materialization="lazy overlapping views of one training series" if lazy else "preserved eager annual windows",
                      data_sha256=info["data_sha256"], source_sha256=digest(ROOT / "src/ai_engine.py"),
                      loss="preserved legacy" if original_loss else "new dimension-normalized balanced objective",
                      cosine_schedule_horizon_epochs=schedule_epochs,
                      effective_epoch_limit=effective_epochs,
                      optimizer={"name": "Adam", "learning_rate": lr,
                                 "weight_decay": 2e-3 if original_loss else 1e-4,
                                 "max_beta": 1e-4 if original_loss else .02,
                                 "beta_warmup_epochs": 50 if original_loss else 10,
                                 "gradient_norm_clip": .5 if original_loss else 1.},
                      prior_statistics_weight=.5 if match_prior else 0.,
                      optimizer_steps=0, completed_epochs=0, initialization_seconds=time.monotonic() - started)
        np.savez_compressed(out / "normalization.npz", data_min=low, data_max=high, night_mask=info["night_mask"])
        report["normalization_sha256"] = digest(out / "normalization.npz")
        no_improve = 0
        for epoch in range(effective_epochs):
            guard.check()
            begin = time.monotonic()
            model.train()
            total, count = 0., 0
            beta = min(1e-4, 1e-4 * epoch / 50) if original_loss else .02 * min(1., (epoch + 1) / 10)
            for x, c in loader:
                guard.check()
                x, c = x.to(args.device), c.to(args.device)
                optimizer.zero_grad(set_to_none=True)
                pred, _, mu, lv = model(x + torch.randn_like(x) * .005, c)
                loss_fn = legacy.vae_loss_1d if original_loss else balanced_loss
                loss, _, _ = loss_fn(pred, x, mu, lv, beta=beta)
                if match_prior:
                    loss = loss + .5 * min(1., (epoch + 1) / 10) * prior_statistics_loss(model.sample(c), x)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .5 if original_loss else 1.)
                if not torch.isfinite(norm):
                    raise FloatingPointError("Nonfinite gradient norm")
                optimizer.step()
                total += float(loss.detach())
                count += 1
                report["optimizer_steps"] += 1
            guard.check()
            model.eval()
            # Reuse validation noise for the dimension-normalized objective.
            context = nullcontext() if original_loss else torch.random.fork_rng(devices=[0] if args.device == "cuda" else [])
            with torch.no_grad(), context:
                if not original_loss:
                    torch.manual_seed(1000)
                x, c = va.to(args.device), vc.to(args.device)
                pred, _, mu, lv = model(x, c)
                # Reconstruction uses the scheduled beta; balanced loss uses beta 0.02.
                vloss = float(loss_fn(pred, x, mu, lv, beta=beta if original_loss else .02)[0])
                if match_prior:
                    vloss += .5 * float(prior_statistics_loss(model.sample(c), x))
                mae = (pred - x).abs().mean(dim=(0, 2)).cpu().tolist()
                var_ratio = float(pred[:, 0].var() / (x[:, 0].var() + 1e-9))
                peak_ratio = float(pred[:, 0].max() / (x[:, 0].max() + 1e-9))
                kl = float(.5 * (mu.square() + lv.exp() - 1 - lv).mean())
            if not np.isfinite(vloss):
                raise FloatingPointError("Nonfinite validation loss")
            scheduler.step()
            row = {"epoch": epoch + 1, "train_loss": total / count, "validation_loss": vloss,
                   "normalized_reconstruction_MAE": dict(zip(("solar", "wind", "temperature"), mae)),
                   "posterior_KL_per_element": kl, "beta": beta, "seconds": time.monotonic() - begin,
                   "elapsed_seconds": time.monotonic() - started}
            report["history"].append(row)
            report["completed_epochs"] = epoch + 1
            if vloss < best:
                best = vloss
                torch.save(model.state_dict(), out / "best_weights.pth")
                report["best_epoch"] = epoch + 1
                no_improve = 0
            else:
                no_improve += 1
            report.update(elapsed_seconds=time.monotonic() - started, best_validation_loss=best,
                          peak_working_set_MiB=getattr(psutil.Process().memory_info(), "peak_wset", psutil.Process().memory_info().rss) / 2 ** 20)
            if args.device == "cuda":
                report["gpu_peak_allocated_MiB"] = torch.cuda.max_memory_allocated() / 2 ** 20
                report["gpu_peak_reserved_MiB"] = torch.cuda.max_memory_reserved() / 2 ** 20
            atomic_json(out / "report.json", report)
            print(json.dumps(row), flush=True)
            gap = (vloss - total / count) / max(total / count, 1e-9) * 100
            if original_loss and ((epoch > 15 and gap > 30) or (epoch > 10 and var_ratio < .3)
                                  or (epoch > 20 and peak_ratio < .5) or no_improve >= 50):
                report["stop_reason"] = "Original early-stopping rule"
                break
            if not original_loss and no_improve >= 25:
                report["stop_reason"] = "25 epochs without fixed validation objective improvement"
                break
        report["status"] = "completed"
        report.setdefault("stop_reason", "Epoch limit")
    except OwnBudgetReached as error:
        report.update(status="budget_stopped", stop_reason=str(error))
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    if model is not None and not (out / "best_weights.pth").exists():
        # An interrupted first epoch is retained and explicitly marked, never a converged result.
        import torch
        torch.save(model.state_dict(), out / "partial_weights.pth")
        report["checkpoint_partial_epoch"] = True
    checkpoint = out / "best_weights.pth"
    if not checkpoint.exists():
        checkpoint = out / "partial_weights.pth"
    if checkpoint.exists():
        report["checkpoint"] = {"file": checkpoint.name, "sha256": digest(checkpoint), "bytes": checkpoint.stat().st_size}
    report.update(elapsed_seconds=time.monotonic() - started, finished_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(out / "report.json", report)
    print(report["status"], flush=True)
    return 1 if report["status"] == "failed" or not checkpoint.exists() else 0


def evaluate(args):
    np, torch = configure("cpu")
    import pandas as pd
    import io
    from study_models import legacy_module, make_model
    from study_data import prepare_data, physical
    from study_metrics import path_metrics
    guard = ResourceGuard("cpu", 300)
    run = (ROOT / args.run).resolve()
    run.relative_to((ROOT / "runs").resolve())
    destination = run / ("study_" + args.split)
    destination.mkdir(exist_ok=False)
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    legacy = legacy_module("cpu")
    protocol = json.loads((ROOT / "overnight_protocol.json").read_text(encoding="utf-8"))
    if digest(ROOT / "src/ai_engine.py") != protocol["legacy_source_sha256"]:
        raise ValueError("Historical source changed before evaluation")
    tensors, info = prepare_data(ROOT, legacy, stride=8760)
    tr, tc, va, vc, te, ec, low, high = tensors
    x, c = (va, vc) if args.split == "validation" else (te, ec)
    real = info[args.split]
    years = list(range(2016, 2020)) if args.split == "validation" else list(range(2020, 2025))
    variant = report.get("variant", "legacy_original")
    model = make_model(variant, legacy, info["night_mask"]).eval()
    checkpoint = run / report.get("checkpoint", {}).get("file", "best_weights.pth")
    if not checkpoint.exists():
        checkpoint = run / "partial_weights.pth"
    payload = checkpoint.read_bytes()
    checkpoint_hash = hashlib.sha256(payload).hexdigest()
    if args.split == "test":
        freeze_path = (ROOT / args.freeze_manifest).resolve()
        freeze_path.relative_to(ROOT.resolve())
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        registered = {item["run"]: item for item in freeze["checkpoints"]}
        if args.run not in registered or registered[args.run]["sha256"] != checkpoint_hash:
            raise ValueError("Test evaluation requires this exact checkpoint in the frozen manifest")
        if freeze["protocol_sha256"] != digest(ROOT / "overnight_protocol.json"):
            raise ValueError("Protocol changed after the test freeze")
        for name, expected in freeze.get("evaluation_implementation_sha256", {}).items():
            if name not in ("study_runner.py", "study_models.py", "study_data.py", "study_metrics.py", "latent_prior.py") or digest(ROOT / name) != expected:
                raise ValueError("Evaluation implementation changed after the test freeze")
    model.load_state_dict(torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True))
    del payload
    rows, correlations = [], []
    outputs = {}
    with torch.no_grad():
        torch.manual_seed(100)
        pred, _, mu, lv = model(x, c)
        outputs[("reconstruction", 100, 1.)] = physical(pred.numpy(), low, high)
        posterior = {"mu_RMS": float(mu.square().mean().sqrt()), "logvar_min": float(lv.min()),
                     "logvar_max": float(lv.max()), "fraction_logvar_outside_sampling_clamp": float(((lv < -10) | (lv > 5)).float().mean())}
        outputs[("posterior_mean", 100, 0.)] = physical(model.decode(mu, c).numpy(), low, high)
        for temperature in (0., .5, 1.):
            for seed in ([101] if temperature == 0 else protocol["prior_seeds"]):
                guard.check()
                torch.manual_seed(seed)
                outputs[("prior", seed, temperature)] = physical(model.sample(c, temperature=temperature).numpy(), low, high)
        calibration = run / "latent_calibration.npz"
        if calibration.exists():
            from latent_prior import sample_gaussian
            calibrated = json.loads((run / "latent_calibration.json").read_text(encoding="utf-8"))
            if calibrated["checkpoint_sha256"] != checkpoint_hash or calibrated["calibration_sha256"] != digest(calibration):
                raise ValueError("Latent calibration/checkpoint provenance mismatch")
            if args.split == "test" and registered[args.run].get("calibration_sha256") != digest(calibration):
                raise ValueError("Latent calibration changed after test freeze")
            with np.load(calibration) as fitted:
                for seed in protocol["prior_seeds"]:
                    torch.manual_seed(seed)
                    outputs[("prior_aggregate", seed, 1.)] = physical(sample_gaussian(
                        model, c, fitted["mean"], fitted["cholesky"]).numpy(), low, high)
    for (mode, seed, temperature), generated in outputs.items():
        guard.check()
        for i, year in enumerate(years):
            values, joint = path_metrics(real[i], generated[i], info["scales"], info["night_mask"], year,
                                        mode, seed, temperature, info["low_wind_threshold"], info["high_wind_threshold"])
            rows.extend(values)
            correlations.append(joint)
        # Individual seed files stay small enough for the remote file API.
        if mode in ("prior", "prior_aggregate") and temperature == 1.:
            np.savez_compressed(destination / f"{mode}_seed{seed}.npz", values=generated.astype(np.float32), years=years)
    pd.DataFrame(rows).to_csv(destination / "metrics.csv", index=False)
    pd.DataFrame(correlations).to_csv(destination / "cross_channel.csv", index=False)
    scores = [r["selection_component"] for r in rows if r["mode"] == "prior" and r["temperature"] == 1.]
    diversity = [outputs[("prior", seed, 1.)] for seed in protocol["prior_seeds"]]
    diversity_std = np.stack(diversity).std(axis=0).mean(axis=(0, 2))
    summary = {"status": "completed", "split": args.split, "test_evaluated": args.split == "test",
               "variant": variant, "training_seed": report.get("training_seed", report.get("configuration", {}).get("seed")),
               "checkpoint_sha256": checkpoint_hash, "years": years, "metric_rows": len(rows),
               "selection_score_T1": float(np.mean(scores)), "posterior": posterior,
               "mean_pointwise_diversity_std": dict(zip(("solar", "wind", "temperature"), diversity_std.tolist())),
               "diversity_fraction_training_std": dict(zip(("solar", "wind", "temperature"), (diversity_std / info["scales"]).tolist())),
               "prior_non_degenerate": bool(np.all(diversity_std / info["scales"] >= protocol["minimum_diversity_fraction_training_std"])),
               "metric_implementation_sha256": digest(ROOT / "study_metrics.py"),
               "evaluator_sha256": digest(__file__), "protocol_sha256": digest(ROOT / "overnight_protocol.json"),
               "units": "Original source units; inverse log1p for wind; no m/s assumption",
               "reference_clipping": "Raw unclipped reference; model training/conditioning follows legacy scaling"}
    calibrated_scores = [r["selection_component"] for r in rows if r["mode"] == "prior_aggregate"]
    if calibrated_scores:
        summary["aggregate_prior_score_T1"] = float(np.mean(calibrated_scores))
        aggregate_diversity = np.stack([outputs[("prior_aggregate", seed, 1.)]
            for seed in protocol["prior_seeds"]]).std(axis=0).mean(axis=(0, 2)) / info["scales"]
        summary["aggregate_diversity_fraction_training_std"] = dict(zip(("solar", "wind", "temperature"), aggregate_diversity.tolist()))
        summary["aggregate_prior_non_degenerate"] = bool(np.all(aggregate_diversity >= protocol["minimum_diversity_fraction_training_std"]))
    atomic_json(destination / "summary.json", summary)
    # Export one individual path directly compatible with the H2RES [N,8760,3] contract.
    np.save(destination / "h2res_profiles_seed101.npy", outputs[("prior", 101, 1.)].transpose(0, 2, 1).astype(np.float32))
    print(json.dumps(summary), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    training = sub.add_parser("train")
    training.add_argument("--variant", required=True, choices=("legacy_original", "compact_global_legacy", "legacy_balanced", "compact_local_balanced", "compact_local_prior_matching", "compact_local_hires_prior_matching"))
    training.add_argument("--seed", type=int, required=True)
    training.add_argument("--seconds", type=int, required=True)
    training.add_argument("--epochs", type=int, default=100)
    training.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    training.add_argument("--output", required=True)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--run", required=True)
    evaluation.add_argument("--split", choices=("validation", "test"), required=True)
    evaluation.add_argument("--freeze-manifest", default="study_20261019/test_freeze.json")
    args = parser.parse_args()
    if args.operation == "train":
        if not 1 <= args.seconds <= 14400 or not 1 <= args.epochs <= 300:
            raise ValueError("Budget exceeds the study's per-run safety bounds")
        return train(args)
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
