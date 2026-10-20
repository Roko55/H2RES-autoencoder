"""Resource limits for our own Windows experiment process only."""
import os
import json
from pathlib import Path


def system_memory_reserve_bytes():
    path = Path(__file__).resolve().parent/"resource_policy.json"
    return int(json.loads(path.read_text(encoding="utf-8"))["reserve_system_RAM_GiB"]*2**30) if path.exists() else 8*2**30


def adaptive_memory_budget():
    path = Path(__file__).resolve().parent/"resource_policy.json"
    if not path.exists():
        return 19
    policy = json.loads(path.read_text(encoding="utf-8"))
    if os.name=="nt":
        import ctypes as c
        class Memory(c.Structure):
            _fields_=[("length",c.c_uint32),("load",c.c_uint32)]+[(name,c.c_uint64) for name in
                ("total","available","page_total","page_available","virtual_total","virtual_available","extended")]
        memory = Memory()
        memory.length = c.sizeof(memory)
        if not c.windll.kernel32.GlobalMemoryStatusEx(c.byref(memory)):
            raise c.WinError()
        total,available = memory.total,memory.available
    else:
        import psutil
        memory = psutil.virtual_memory()
        total,available = memory.total,memory.available
    reserve = float(policy["reserve_system_RAM_GiB"])
    fraction = float(policy["maximum_fraction_of_total_RAM"])
    if not 0.1 <= fraction <= 0.9 or reserve < 64:
        raise ValueError("Resource policy must preserve at least 64 GiB and 10% total RAM")
    budget = int(min(total/2**30*fraction,available/2**30-reserve))
    if budget < 8:
        raise MemoryError("Insufficient free memory after the reserved RAM")
    return budget


def windows_memory_limit(gib=None):
    """Apply a commit limit to this process's job, without touching other jobs."""
    if os.name != "nt":
        raise RuntimeError("This resource guard was verified for Windows only")
    if gib is None:
        gib = adaptive_memory_budget()
    import ctypes as c
    from ctypes import wintypes as w
    class Basic(c.Structure):
        _fields_=[("process_time",c.c_longlong),("job_time",c.c_longlong),("flags",w.DWORD),
            ("min_ws",c.c_size_t),("max_ws",c.c_size_t),("process_limit",w.DWORD),
            ("affinity",c.c_size_t),("priority",w.DWORD),("scheduling",w.DWORD)]
    class IO(c.Structure):
        _fields_=[(name,c.c_ulonglong) for name in ("read_ops","write_ops","other_ops","read_bytes","write_bytes","other_bytes")]
    class Extended(c.Structure):
        _fields_=[("basic",Basic),("io",IO),("process_memory",c.c_size_t),("job_memory",c.c_size_t),
                  ("peak_process",c.c_size_t),("peak_job",c.c_size_t)]
    k = c.WinDLL("kernel32",use_last_error=True)
    k.CreateJobObjectW.argtypes=[c.c_void_p,w.LPCWSTR]
    k.CreateJobObjectW.restype=w.HANDLE
    k.SetInformationJobObject.argtypes=[w.HANDLE,c.c_int,c.c_void_p,w.DWORD]
    k.SetInformationJobObject.restype=w.BOOL
    k.AssignProcessToJobObject.argtypes=[w.HANDLE,w.HANDLE]
    k.AssignProcessToJobObject.restype=w.BOOL
    k.GetCurrentProcess.restype=w.HANDLE
    handle = k.CreateJobObjectW(None,None)
    if not handle:
        raise c.WinError(c.get_last_error())
    limits = Extended()
    limits.basic.flags = 0x200  # JOB_OBJECT_LIMIT_JOB_MEMORY
    limits.job_memory = int(gib*2**30)
    if not k.SetInformationJobObject(handle,9,c.byref(limits),c.sizeof(limits)):
        raise c.WinError(c.get_last_error())
    if not k.AssignProcessToJobObject(handle,k.GetCurrentProcess()):
        raise c.WinError(c.get_last_error())
    return handle  # Retain until worker exit; no other process is assigned.
