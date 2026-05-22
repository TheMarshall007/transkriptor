import re
from pathlib import Path
from typing import Dict, List

from models import FileJob


def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)}{unit}"
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{num_bytes}B"


def format_duration(seconds: float) -> str:
    total_seconds = int(round(max(0.0, seconds)))
    minutes = total_seconds // 60
    secs = total_seconds % 60
    hours = minutes // 60
    minutes = minutes % 60

    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def format_timestamp(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    secs = milliseconds // 1000
    milliseconds %= 1000
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


def sanitize_filename(value: str) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*]+', "_", value)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    return sanitized


def natural_sort_key(value: str):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    ]


def truncate_text(value: str, max_len: int) -> str:
    if max_len <= 0:
        return ""
    if len(value) <= max_len:
        return value
    if max_len == 1:
        return value[:1]
    return value[: max_len - 1] + "…"


def relative_media_folder(input_root: Path, input_file: Path) -> Path:
    if input_root.is_file():
        return Path()

    try:
        relative_parent = input_file.parent.relative_to(input_root)
        return relative_parent
    except ValueError:
        return Path()


def get_output_paths(output_dir: Path, input_root: Path, input_file: Path):
    relative_folder = relative_media_folder(input_root, input_file)
    target_dir = output_dir / relative_folder
    target_dir.mkdir(parents=True, exist_ok=True)

    base = input_file.stem
    return (
        target_dir / f"{base}.txt",
        target_dir / f"{base}.srt",
        target_dir / f"{base}.json",
    )


def write_srt(result: dict, srt_path: Path) -> None:
    segments = result.get("segments", [])
    lines: List[str] = []

    for idx, segment in enumerate(segments, start=1):
        start = format_timestamp(segment["start"])
        end = format_timestamp(segment["end"])
        text = segment["text"].strip()

        lines.append(str(idx))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")

    srt_path.write_text("\n".join(lines), encoding="utf-8")


def build_grouped_output_filename(root_input_path: Path, folder: Path) -> str:
    try:
        relative = folder.relative_to(root_input_path)
        parts = list(relative.parts)
    except ValueError:
        parts = [folder.name]

    if not parts:
        parts = [folder.name]

    joined = " - ".join(parts)
    return sanitize_filename(f"{joined}__agrupado.txt")


def write_grouped_transcription_file(
    root_input_path: Path,
    folder: Path,
    jobs: List[FileJob],
    output_dir: Path,
    file_analysis: Dict[str, dict],
) -> dict:
    grouped_dir = output_dir / "_agrupados"
    grouped_dir.mkdir(parents=True, exist_ok=True)

    sorted_jobs = sorted(jobs, key=lambda job: natural_sort_key(job.path.name))

    total_media_duration_sec = 0.0
    total_processing_duration_sec = 0.0
    total_word_count = 0
    included_files = 0

    lines: List[str] = []
    lines.append("# TRANSCRIÇÃO AGRUPADA")
    lines.append(f"Pasta: {folder}")
    lines.append(f"Arquivos incluídos: {len(sorted_jobs)}")
    lines.append("")

    for index, job in enumerate(sorted_jobs, start=1):
        txt_path, _, _ = get_output_paths(output_dir, root_input_path, job.path)
        if not txt_path.exists():
            continue

        content = txt_path.read_text(encoding="utf-8").strip()
        analysis = file_analysis.get(job.path.name, {})

        media_duration_sec = analysis.get("media_duration_sec") or 0.0
        processing_duration_sec = analysis.get("processing_duration_sec") or 0.0
        word_count = analysis.get("word_count") or 0
        segment_count = analysis.get("segment_count") or 0
        rtf = analysis.get("realtime_factor")

        total_media_duration_sec += media_duration_sec
        total_processing_duration_sec += processing_duration_sec
        total_word_count += word_count
        included_files += 1

        lines.append("=" * 80)
        lines.append(f"AULA {index}: {job.path.name}")
        lines.append("=" * 80)
        lines.append(f"Tamanho: {human_size(job.size_bytes)}")
        lines.append(
            f"Duração da mídia: "
            f"{format_duration(media_duration_sec) if media_duration_sec else 'N/A'}"
        )
        lines.append(
            f"Tempo de processamento: "
            f"{format_duration(processing_duration_sec) if processing_duration_sec else 'N/A'}"
        )
        lines.append(f"Palavras: {word_count}")
        lines.append(f"Segmentos: {segment_count}")
        lines.append(
            f"Realtime factor: {f'{rtf:.2f}x' if rtf is not None else 'N/A'}"
        )
        lines.append("")
        lines.append(content)
        lines.append("")

    grouped_filename = build_grouped_output_filename(root_input_path, folder)
    grouped_path = grouped_dir / grouped_filename
    grouped_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "folder": str(folder),
        "output_path": str(grouped_path),
        "file_count": included_files,
        "total_media_duration_sec": total_media_duration_sec,
        "total_processing_duration_sec": total_processing_duration_sec,
        "total_word_count": total_word_count,
    }


def build_status_row(
    group_name: str,
    file_name: str,
    size_human: str,
    percent: int,
    status: str,
    queue_pos: str,
    elapsed_text: str = "",
    worker_text: str = "",
    group_width: int = 20,
    file_width: int = 34,
) -> str:
    group_col = truncate_text(group_name, group_width)
    file_col = truncate_text(file_name, file_width)
    elapsed_col = truncate_text(elapsed_text, 10)
    worker_col = truncate_text(worker_text, 8)

    return (
        f"{group_col:<{group_width}} "
        f"{file_col:<{file_width}} "
        f"{size_human:>8} "
        f"{percent:>4}% "
        f"{status:<12} "
        f"{queue_pos:>4} "
        f"{worker_col:<8} "
        f"{elapsed_col:<10}"
    )