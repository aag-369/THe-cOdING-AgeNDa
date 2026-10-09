"""Environment capture: interpreter, installed packages, platform, accelerators.

All lazy: :func:`accelerator_info` imports ``torch`` only if it is already
importable, and otherwise shells out to ``nvidia-smi`` best-effort. Nothing
here imports a heavyweight library at module import time.
"""

from __future__ import annotations

import importlib.util
import platform
import socket
import subprocess
import sys
from importlib import metadata
from typing import TypedDict

__all__ = [
    "ENV_ALLOWLIST",
    "PlatformInfo",
    "PythonInfo",
    "accelerator_info",
    "package_versions",
    "platform_info",
    "python_info",
]

ENV_ALLOWLIST = (
    "CUDA_VISIBLE_DEVICES",
    "PYTHONHASHSEED",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "RESEARCH_RUNS_DIR",
)
"""Environment variables copied verbatim into ``manifest["invocation"]["env"]``."""


class PythonInfo(TypedDict):
    version: str
    executable: str
    implementation: str


class PlatformInfo(TypedDict):
    system: str
    release: str
    machine: str
    hostname: str


def python_info() -> PythonInfo:
    """Interpreter version string, executable path, implementation name."""
    return {
        "version": platform.python_version(),
        "executable": sys.executable,
        "implementation": platform.python_implementation(),
    }


def package_versions() -> dict[str, str]:
    """Every installed distribution mapped ``name -> version`` (sorted)."""
    versions: dict[str, str] = {}
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            versions[name] = dist.version
    return dict(sorted(versions.items(), key=lambda item: item[0].lower()))


def platform_info() -> PlatformInfo:
    """OS name/release, machine architecture, hostname."""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "hostname": socket.gethostname(),
    }


def accelerator_info() -> list[dict[str, object]]:
    """Best-effort accelerator inventory.

    If ``torch`` is importable, report CUDA version, device count, and
    per-device name/capability, plus MPS availability. Otherwise try
    ``nvidia-smi``. On failure, record the reason rather than raising; returns
    ``[]`` when nothing is detected.

    The broad ``except`` here is deliberate and matches the module contract:
    accelerator probing must never abort a run.
    """
    if importlib.util.find_spec("torch") is not None:
        return _torch_accelerators()
    return _nvidia_smi_accelerators()


def _torch_accelerators() -> list[dict[str, object]]:
    try:
        import torch

        result: list[dict[str, object]] = []
        cuda: dict[str, object] = {
            "kind": "cuda",
            "available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "cuda_version": torch.version.cuda,
            "devices": [],
        }
        devices: list[dict[str, object]] = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "name": props.name,
                    "capability": f"{props.major}.{props.minor}",
                    "total_memory": int(props.total_memory),
                }
            )
        cuda["devices"] = devices
        result.append(cuda)
        mps = getattr(torch.backends, "mps", None)
        if mps is not None:
            result.append({"kind": "mps", "available": bool(mps.is_available())})
        return result
    except Exception as exc:
        return [{"kind": "torch", "error": repr(exc)}]


def _nvidia_smi_accelerators() -> list[dict[str, object]]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, ValueError):
        return []
    if proc.returncode != 0:
        return []
    gpus: list[dict[str, object]] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        gpus.append(
            {
                "kind": "cuda",
                "name": parts[0] if parts else "",
                "memory_total": parts[1] if len(parts) > 1 else None,
                "driver_version": parts[2] if len(parts) > 2 else None,
            }
        )
    return gpus
