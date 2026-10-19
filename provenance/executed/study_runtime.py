"""Conservative guards affecting only this experiment's process."""
import csv
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parent


def own_cpu_hybrid_context(pid, psutil_module):
    """Recognize only an active CPU child launched by this comparison queue.

    The XGBoost CUDA-capable library registers a context even for device=cpu.
    This exception never covers an unrelated Python process or a CUDA worker.
    """
    try:
        state = json.loads((ROOT / "measurements.json").read_text(encoding="utf-8"))
        if not any(job.get("pid") == pid and job.get("status") == "running"
                   and job.get("label", "").startswith("hybrid_seed")
                   and job.get("label", "").endswith("_faculty_cpu") for job in state.get("jobs", [])):
            return False
        process = psutil_module.Process(pid)
        directory = Path(process.cwd()).resolve()
        command = process.cmdline()
        if directory != ROOT.resolve() or len(command) < 2:
            return False
        script = Path(command[1])
        if not script.is_absolute():
            script = directory / script
        if script.resolve() != (ROOT / "hybrid_trial.py").resolve():
            return False
        if "--device" in command:
            index = command.index("--device")
            if index + 1 >= len(command) or command[index + 1] != "cpu":
                return False
        return True
    except (OSError, ValueError, psutil_module.Error):
        return False


class OwnBudgetReached(RuntimeError):
    pass


class ResourceGuard:
    def __init__(self, device, seconds):
        import psutil
        self.psutil = psutil
        self.started = time.monotonic()
        self.seconds = seconds
        self.device = device
        self.last_gpu_check = -100.
        self.gpu_tool = shutil.which("nvidia-smi")
        self.job_handle = None
        policy = ROOT / "resource_policy.json"
        self.reserve = max(1., float(os.environ.get("STUDY_RESERVE_RAM_GIB", "1"))) * 1024 ** 3
        if policy.exists():
            from runtime_limits import adaptive_memory_budget, system_memory_reserve_bytes, windows_memory_limit
            self.reserve = system_memory_reserve_bytes()
            if os.name == "nt":
                self.job_handle = windows_memory_limit(adaptive_memory_budget())
        if os.name == "nt":
            psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        self.check(force_gpu=True)

    def check(self, force_gpu=False):
        elapsed = time.monotonic() - self.started
        if elapsed >= self.seconds:
            raise OwnBudgetReached("Predeclared wall-clock training budget reached")
        if self.psutil.virtual_memory().available < self.reserve:
            raise OwnBudgetReached("Reserved free system RAM reached; stopping only this run")
        if (ROOT / "STOP_STUDY").exists():
            raise OwnBudgetReached("Own STOP_STUDY marker is present")
        if self.device != "cuda" or (not force_gpu and elapsed - self.last_gpu_check < 15):
            return
        self.last_gpu_check = elapsed
        if not self.gpu_tool:
            raise OwnBudgetReached("Cannot inspect shared GPU; defer own run")
        flags = 0x08000000 if os.name == "nt" else 0
        result = subprocess.run([self.gpu_tool, "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=5, creationflags=flags)
        if result.returncode:
            raise OwnBudgetReached("Cannot verify shared GPU status")
        desktop = {"dwm.exe", "logonui.exe", "shellhost.exe", "crossdeviceresume.exe", "explorer.exe",
                   "searchhost.exe", "startmenuexperiencehost.exe", "msedgewebview2.exe",
                   "shellexperiencehost.exe", "phoneexperiencehost.exe", "wudfhost.exe",
                   "windowsterminal.exe", "msedge.exe"}
        for line in result.stdout.splitlines():
            if not line.strip().isdigit() or int(line.strip()) == os.getpid():
                continue
            pid = int(line.strip())
            if own_cpu_hybrid_context(pid, self.psutil):
                continue
            try:
                name = self.psutil.Process(pid).name()
            except self.psutil.NoSuchProcess:
                continue
            except self.psutil.AccessDenied:
                if os.name != "nt":
                    raise OwnBudgetReached("Unidentified GPU process")
                lookup = subprocess.run(["C:/Windows/System32/tasklist.exe", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                                        capture_output=True, text=True, timeout=5, creationflags=flags)
                matches = [r for r in csv.reader(io.StringIO(lookup.stdout)) if len(r) > 1 and r[1].strip() == str(pid)]
                if not matches and not self.psutil.pid_exists(pid):
                    continue
                name = matches[0][0] if matches else "unidentified"
            if name.lower() not in desktop:
                raise OwnBudgetReached("Another or unidentified GPU computation is present; defer own run")

    def configure_cuda(self):
        import torch
        free, total = torch.cuda.mem_get_info()
        if total - free > 1024 * 2 ** 20:
            raise OwnBudgetReached("Shared GPU already has more than 1 GiB occupied")
        fraction = min(.70, (free - 1024 * 2 ** 20) / total)
        if fraction <= 0:
            raise OwnBudgetReached("Insufficient GPU headroom")
        torch.cuda.set_per_process_memory_fraction(fraction)
        return {"name": torch.cuda.get_device_name(0), "total_MiB": total / 2 ** 20,
                "initial_free_MiB": free / 2 ** 20, "allocator_fraction": fraction}


def atomic_json(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    pending.replace(path)
