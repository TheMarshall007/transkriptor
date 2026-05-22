import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from formatting import human_size
from models import FileJob, FileProgress


def clamp_workers(requested: int) -> int:
    if requested < 1:
        return 1
    return min(requested, 2)


def suggested_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return 2 if cpu_count >= 8 else 1


def build_metrics_payload(
    input_path: Path,
    output_dir: Path,
    log_dir: Path,
    jobs: List[FileJob],
    args,
    success_count: int,
    error_count: int,
    total_script_sec: float,
    total_mb: float,
    avg_file_sec: float,
    throughput_mb_s: float,
    success_rate: float,
    model_load_total_sec: float,
    worker_metrics: Dict[int, dict],
    progress_state: Dict[str, FileProgress],
    resource_metrics: Dict[str, dict],
    file_analysis: Dict[str, dict],
    total_media_duration_sec: float,
    total_word_count: int,
    avg_realtime_factor: float,
    folder_analysis: Dict[str, dict],
):
    files_payload = []
    for job in jobs:
        state = progress_state[job.path.name]
        analysis = file_analysis.get(job.path.name, {})

        files_payload.append({
            "file_name": job.path.name,
            "path": str(job.path),
            "group_name": job.group_name,
            "discovered_index": job.discovered_index,
            "size_bytes": job.size_bytes,
            "size_human": human_size(job.size_bytes),
            "status": state.status,
            "duration_sec": state.duration_sec,
            "error": state.error,
            "pid": state.pid,
            "priority_index": analysis.get("priority_index"),
            "media_duration_sec": analysis.get("media_duration_sec"),
            "processing_duration_sec": analysis.get("processing_duration_sec"),
            "realtime_factor": analysis.get("realtime_factor"),
            "word_count": analysis.get("word_count"),
            "char_count": analysis.get("char_count"),
            "segment_count": analysis.get("segment_count"),
            "mb_per_processing_second": analysis.get("mb_per_processing_second"),
            "words_per_processing_second": analysis.get("words_per_processing_second"),
            "words_per_media_minute": analysis.get("words_per_media_minute"),
        })

    workers_payload = {}
    for pid, info in worker_metrics.items():
        resource_info = resource_metrics.get("workers", {}).get(str(pid), {})
        merged = dict(info)
        merged["resource"] = resource_info
        workers_payload[str(pid)] = merged

    try:
        import psutil  # noqa: F401
        psutil_available = True
    except ImportError:
        psutil_available = False

    overall_realtime_factor = (
        total_media_duration_sec / total_script_sec
        if total_script_sec > 0 and total_media_duration_sec > 0
        else 0.0
    )

    return {
        "run": {
            "input_path": str(input_path),
            "output_dir": str(output_dir),
            "log_dir": str(log_dir),
            "model": args.model,
            "language": args.language,
            "workers": args.workers,
            "workers_effective": clamp_workers(args.workers),
            "group_by_folder": getattr(args, "group_by_folder", False),
            "success_count": success_count,
            "error_count": error_count,
            "total_files": len(jobs),
            "total_groups": len(folder_analysis),
            "success_rate_percent": success_rate,
            "total_script_sec": total_script_sec,
            "total_processed_mb": total_mb,
            "total_media_duration_sec": total_media_duration_sec,
            "total_word_count": total_word_count,
            "avg_file_sec": avg_file_sec,
            "throughput_mb_s": throughput_mb_s,
            "model_load_total_sec": model_load_total_sec,
            "overall_realtime_factor": overall_realtime_factor,
            "avg_realtime_factor": avg_realtime_factor,
            "psutil_available": psutil_available,
        },
        "files": files_payload,
        "groups": folder_analysis,
        "workers": workers_payload,
        "resource_metrics": resource_metrics,
        "generated_at": datetime.now().isoformat(),
    }