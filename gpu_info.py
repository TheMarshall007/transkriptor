"""Best-effort NVIDIA telemetry for the local desktop panel."""

from __future__ import annotations

import subprocess


def query_gpu_info() -> dict:
    result = {
        "available": False,
        "name": "GPU indisponível",
        "utilization_percent": None,
        "memory_used_mb": None,
        "memory_total_mb": None,
        "temperature_c": None,
        "torch_allocated_mb": None,
        "torch_reserved_mb": None,
        "reason": "nvidia-smi não encontrado ou GPU NVIDIA indisponível.",
    }
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=1.5,
            check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        fields = [part.strip() for part in completed.stdout.splitlines()[0].split(",")]
        result.update({
            "available": True,
            "name": fields[0],
            "utilization_percent": int(fields[1]),
            "memory_used_mb": int(fields[2]),
            "memory_total_mb": int(fields[3]),
            "temperature_c": int(fields[4]),
            "reason": "",
        })
    except (OSError, subprocess.SubprocessError, IndexError, ValueError) as exc:
        result["reason"] = str(exc) or result["reason"]

    try:
        import torch

        if torch.cuda.is_available():
            result["torch_allocated_mb"] = round(torch.cuda.memory_allocated(0) / 1048576)
            result["torch_reserved_mb"] = round(torch.cuda.memory_reserved(0) / 1048576)
    except Exception:
        pass
    return result
