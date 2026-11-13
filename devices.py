"""Device selection and local CUDA capability checks."""

from __future__ import annotations


def cuda_status() -> tuple[bool, str]:
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "PyTorch não detectou uma GPU CUDA neste ambiente."
        return True, torch.cuda.get_device_name(0)
    except Exception as exc:
        return False, f"CUDA indisponível: {exc}"


def resolve_device(requested: str = "auto") -> str:
    requested = (requested or "auto").lower()
    if requested == "auto":
        return "cuda" if cuda_status()[0] else "cpu"
    if requested == "cuda":
        available, detail = cuda_status()
        if not available:
            raise RuntimeError(detail)
        return "cuda"
    if requested != "cpu":
        raise ValueError(f"Dispositivo inválido: {requested}")
    return "cpu"
