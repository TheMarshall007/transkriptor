"""Best-effort NVIDIA telemetry for the local desktop panel."""

from __future__ import annotations

import subprocess


_nvml_initialized = False


def query_gpu_process_metrics(pid: int, last_seen_us: int | None = None) -> dict:
    """Return only samples and VRAM attributed by NVML to this process."""
    global _nvml_initialized
    result = {"available": False, "utilization_available": False, "memory_available": False,
              "utilization_percent": None, "memory_mb": None,
              "last_seen_us": last_seen_us, "reason": "NVML não disponível."}
    try:
        import pynvml
        if not _nvml_initialized:
            pynvml.nvmlInit()
            _nvml_initialized = True
        count = pynvml.nvmlDeviceGetCount()
        utilization = []
        memory = []
        for index in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            try:
                for process in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                    if int(process.pid) == int(pid) and process.usedGpuMemory is not None:
                        memory.append(int(process.usedGpuMemory) / 1048576)
            except Exception:
                pass
            try:
                samples = pynvml.nvmlDeviceGetProcessUtilization(handle, int(last_seen_us or 0))
                for sample in samples:
                    if int(sample.pid) == int(pid):
                        utilization.append((int(sample.timestamp), float(sample.smUtil)))
            except Exception as exc:
                result["reason"] = str(exc) or "Driver sem amostras de utilização por processo."
        if utilization:
            newest = max(utilization, key=lambda item: item[0])
            result["utilization_percent"] = newest[1]
            result["last_seen_us"] = newest[0]
        if memory:
            result["memory_mb"] = sum(memory)
        result["utilization_available"] = bool(utilization)
        result["memory_available"] = bool(memory)
        result["available"] = bool(utilization or memory)
        if result["available"]:
            result["reason"] = ""
        elif not result["reason"]:
            result["reason"] = "Driver não expõe telemetria por processo para este PID."
        if not result["available"] and result["reason"] == "NVML não disponível.":
            result["reason"] = "Driver não expõe amostras de GPU atribuídas a este PID."
    except Exception as exc:
        result["reason"] = f"NVML indisponível: {exc}"
    return result


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
