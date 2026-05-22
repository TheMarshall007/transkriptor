import time
from pathlib import Path
from typing import Dict, List, Optional

from constants import SUPPORTED_EXTENSIONS
from models import FileJob, FileProgress


def _resolve_group_name(input_path: Path, file_path: Path) -> str:
    if input_path.is_file():
        return "Arquivo avulso"

    try:
        relative_parent = file_path.parent.relative_to(input_path)
    except ValueError:
        return file_path.parent.name or "Raiz"

    if not relative_parent.parts:
        return "Raiz"

    return " / ".join(relative_parent.parts)


def scan_files(input_path: Path) -> List[FileJob]:
    if input_path.is_file():
        files = [input_path] if input_path.suffix.lower() in SUPPORTED_EXTENSIONS else []
    else:
        files = sorted(
            f for f in input_path.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    jobs: List[FileJob] = []

    for index, file_path in enumerate(files, start=1):
        jobs.append(
            FileJob(
                path=file_path,
                size_bytes=file_path.stat().st_size,
                group_name=_resolve_group_name(input_path, file_path),
                discovered_index=index,
            )
        )

    return jobs


def estimate_eta(
    progress_state: Dict[str, FileProgress],
    pending_count: int
) -> Optional[float]:
    completed_durations = [
        state.duration_sec
        for state in progress_state.values()
        if state.status == "Done" and state.duration_sec is not None
    ]

    processing_states = [
        state for state in progress_state.values()
        if state.status == "Processing" and state.started_at is not None
    ]

    if not completed_durations:
        return None

    avg_duration = sum(completed_durations) / len(completed_durations)
    processing_remaining = 0.0

    for state in processing_states:
        elapsed = time.time() - state.started_at
        processing_remaining += max(0.0, avg_duration - elapsed)

    return processing_remaining + (pending_count * avg_duration)