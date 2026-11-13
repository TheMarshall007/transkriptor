"""Build deterministic, collision-safe jobs from multiple GUI inputs."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import subprocess

from constants import SUPPORTED_EXTENSIONS
from models import FileJob


def _unique_folder_names(folders: list[Path]) -> dict[Path, str]:
    counts: Counter[str] = Counter()
    used: set[str] = set()
    names: dict[Path, str] = {}
    for folder in folders:
        base = folder.name or folder.drive.rstrip(":\\") or "Pasta"
        counts[base.casefold()] += 1
        occurrence = counts[base.casefold()]
        candidate = base if occurrence == 1 else f"{base} ({occurrence})"
        while candidate.casefold() in used:
            occurrence += 1
            candidate = f"{base} ({occurrence})"
        used.add(candidate.casefold())
        names[folder] = candidate
    return names


def probe_duration(path: Path) -> float | None:
    try:
        completed = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=4, check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return float(completed.stdout.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def collect_transcription_jobs(sources: list[str | Path], output_dir: str | Path) -> tuple[list[FileJob], list[str]]:
    output_root = Path(output_dir).expanduser().resolve()
    selected: list[Path] = []
    warnings: list[str] = []
    seen: set[Path] = set()
    for source in sources:
        path = Path(source).expanduser().resolve()
        if path in seen:
            continue
        seen.add(path)
        if not path.exists():
            warnings.append(f"Caminho inexistente: {path}")
        elif path.is_file():
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                selected.append(path)
            else:
                warnings.append(f"Formato não suportado: {path.name}")
        elif path.is_dir():
            selected.append(path)
        else:
            warnings.append(f"Entrada não suportada: {path}")

    folders = [path for path in selected if path.is_dir()]
    folders = [folder for folder in folders if not any(folder != other and folder.is_relative_to(other) for other in folders)]
    folder_names = _unique_folder_names(folders)
    individual_files = [path for path in selected if path.is_file() and not any(path.is_relative_to(folder) for folder in folders)]

    pending: list[tuple[Path, Path, Path, str, str]] = []
    used_stems: set[str] = set()
    for source in selected:
        if source.is_dir():
            if source not in folders:
                continue
            media = sorted((p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS),
                           key=lambda p: str(p.relative_to(source)).casefold())
            if not media:
                warnings.append(f"Nenhum arquivo compatível na pasta: {source}")
                continue
            folder_output = output_root / folder_names[source]
            for media_file in media:
                relative = media_file.relative_to(source)
                pending.append((media_file, source, folder_output, media_file.stem,
                                f"{folder_names[source]}\\{relative}"))
        elif source.is_file() and source in individual_files:
            candidate = source.stem
            if candidate.casefold() in used_stems:
                parent = source.parent.name or "arquivo"
                candidate = f"{candidate}__{parent}"
                suffix = 2
                while candidate.casefold() in used_stems:
                    candidate = f"{source.stem}__{parent}_{suffix}"
                    suffix += 1
            used_stems.add(candidate.casefold())
            pending.append((source, source.parent, output_root, candidate, source.name))

    jobs = []
    for index, (file_path, input_root, destination, output_stem, display_path) in enumerate(pending, start=1):
        jobs.append(FileJob(
            path=file_path,
            size_bytes=file_path.stat().st_size,
            group_name=input_root.name or "Raiz",
            discovered_index=index,
            job_id=f"{file_path.resolve()}|{destination.resolve()}|{output_stem}",
            input_root=input_root,
            output_dir=destination,
            output_stem=output_stem,
            display_path=display_path,
        ))
    return jobs, warnings
