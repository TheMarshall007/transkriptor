import os
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from file_discovery import estimate_eta
from formatting import (
    build_status_row,
    format_duration,
    human_size,
    natural_sort_key,
)
from models import FileJob, FileProgress


def clear_screen(no_clear: bool) -> None:
    if not no_clear:
        return


def render_header(
    input_path: Path,
    jobs: List[FileJob],
    model_name: str,
    workers: int,
    language: str,
    output_dir: Path,
) -> List[str]:
    group_counter = Counter(job.group_name or "Raiz" for job in jobs)
    total_size_bytes = sum(job.size_bytes for job in jobs)

    lines: List[str] = []
    lines.append("=" * 100)
    lines.append("TRANSKRIPTOR")
    lines.append("=" * 100)
    lines.append(f"Entrada: {input_path}")
    lines.append(f"Saída:   {output_dir}")
    lines.append(
        f"Arquivos encontrados: {len(jobs)} | "
        f"Grupos: {len(group_counter)} | "
        f"Tamanho total: {human_size(total_size_bytes)}"
    )
    lines.append(
        f"Modelo: {model_name} | Workers: {workers} | Idioma: {language}"
    )
    lines.append("-" * 100)

    if group_counter:
        lines.append("Resumo por grupo:")
        for group_name, count in sorted(group_counter.items(), key=lambda item: natural_sort_key(item[0])):
            lines.append(f"  - {group_name}: {count} arquivo(s)")
        lines.append("-" * 100)

    lines.append("Ordem de exibição: agrupada por pasta e nome do arquivo.")
    lines.append("A ordem real de processamento pode ser diferente conforme os workers.")
    lines.append("=" * 100)

    return lines


def progress_bar(percent: int, width: int = 28) -> str:
    percent = max(0, min(100, percent))
    filled = int(percent * width / 100)
    return "[" + ("=" * filled) + ("." * (width - filled)) + "]"


def render_worker_lines(
    worker_metrics: Dict[int, dict],
    resource_metrics: Dict[str, dict]
) -> List[str]:
    lines = []
    lines.append("WORKERS")
    lines.append("-" * 100)

    if not worker_metrics:
        lines.append("Nenhum worker iniciou o carregamento do modelo ainda.")
        return lines

    workers_resource = resource_metrics.get("workers", {})

    for pid in sorted(worker_metrics.keys()):
        info = worker_metrics[pid]
        model_name = info.get("model_name", "-")
        files_processed = info.get("files_processed", 0)
        total_sec = info.get("total_processing_sec", 0.0)

        if info.get("model_loading"):
            status = "Carregando modelo..."
        elif info.get("model_loaded"):
            load_sec = info.get("model_load_duration_sec", 0.0)
            status = f"Modelo carregado em {format_duration(load_sec)}"
        else:
            status = "Aguardando"

        worker_resource = workers_resource.get(str(pid), {})
        cpu_text = ""
        ram_text = ""

        if worker_resource:
            cpu_text = (
                f" | CPU avg {worker_resource.get('cpu_avg', 0.0):.1f}%"
                f" pico {worker_resource.get('cpu_peak', 0.0):.1f}%"
            )
            ram_text = (
                f" | RAM avg {worker_resource.get('rss_avg_mb', 0.0):.1f}MB"
                f" pico {worker_resource.get('rss_peak_mb', 0.0):.1f}MB"
            )

        lines.append(
            f"PID {pid} | modelo={model_name} | arquivos={files_processed} | "
            f"tempo={format_duration(total_sec)} | {status}{cpu_text}{ram_text}"
        )

    return lines


def _sorted_jobs_for_display(jobs: List[FileJob]) -> List[FileJob]:
    return sorted(
        jobs,
        key=lambda job: (
            natural_sort_key(job.group_name or "Raiz"),
            natural_sort_key(job.path.name),
            job.discovered_index,
        ),
    )


def _queue_positions(
    jobs: List[FileJob],
    progress_state: Dict[str, FileProgress],
) -> Dict[str, int]:
    waiting_jobs = [
        job for job in _sorted_jobs_for_display(jobs)
        if progress_state[job.key].status in {"Waiting", "Queued"}
    ]
    return {
        job.key: index
        for index, job in enumerate(waiting_jobs, start=1)
    }


def _worker_current_files(
    jobs: List[FileJob],
    progress_state: Dict[str, FileProgress],
) -> Dict[int, str]:
    current: Dict[int, str] = {}

    for job in jobs:
        state = progress_state.get(job.key)
        if not state:
            continue
        if state.status == "Processing" and state.pid is not None:
            current[state.pid] = job.display_path or job.path.name

    return current


def write_live_status_tsv(
    output_dir: Path,
    jobs: List[FileJob],
    progress_state: Dict[str, FileProgress],
) -> Path:
    status_dir = output_dir / "_status"
    status_dir.mkdir(parents=True, exist_ok=True)
    status_file = status_dir / "live_status.tsv"

    queue_positions = _queue_positions(jobs, progress_state)

    lines = [
        "\t".join([
            "group_name",
            "file_name",
            "size_bytes",
            "size_human",
            "status",
            "percent",
            "pid",
            "queue_position",
            "discovered_index",
            "elapsed_or_duration",
            "stage",
            "current_frames",
            "total_frames",
            "eta_seconds",
            "stage_elapsed",
            "model_name",
            "device",
            "model_load_seconds",
            "processing_seconds",
            "media_duration_seconds",
            "realtime_factor",
            "cpu_avg_percent",
            "cpu_peak_percent",
            "ram_avg_mb",
            "ram_peak_mb",
            "gpu_utilization_percent",
            "gpu_memory_used_mb",
            "gpu_memory_total_mb",
        ])
    ]

    for job in _sorted_jobs_for_display(jobs):
        state = progress_state[job.key]

        elapsed_or_duration = ""
        if state.status == "Processing" and state.started_at:
            elapsed_or_duration = format_duration(time.time() - state.started_at)
        elif state.duration_sec is not None:
            elapsed_or_duration = format_duration(state.duration_sec)

        stage_elapsed = ""
        if state.status == "Processing" and state.stage_started_at is not None:
            stage_elapsed = format_duration(time.time() - state.stage_started_at)

        lines.append("\t".join([
            job.group_name or "Raiz",
            job.display_path or job.path.name,
            str(job.size_bytes),
            human_size(job.size_bytes),
            state.status,
            str(state.percent) if state.percent is not None else "",
            str(state.pid or ""),
            str(queue_positions.get(job.key, "")),
            str(job.discovered_index),
            elapsed_or_duration,
            state.stage,
            str(state.current_frames) if state.current_frames is not None else "",
            str(state.total_frames) if state.total_frames is not None else "",
            f"{state.eta_sec:.1f}" if state.eta_sec is not None else "",
            stage_elapsed,
            state.model_name,
            state.device,
            f"{state.model_load_sec:.3f}" if state.model_load_sec is not None else "",
            f"{state.processing_duration_sec:.3f}" if state.processing_duration_sec is not None else "",
            f"{state.media_duration_sec:.3f}" if state.media_duration_sec is not None else "",
            f"{state.realtime_factor:.3f}" if state.realtime_factor is not None else "",
            f"{state.cpu_avg:.2f}" if state.cpu_avg is not None else "",
            f"{state.cpu_peak:.2f}" if state.cpu_peak is not None else "",
            f"{state.ram_avg_mb:.2f}" if state.ram_avg_mb is not None else "",
            f"{state.ram_peak_mb:.2f}" if state.ram_peak_mb is not None else "",
            f"{state.gpu_utilization_percent:.2f}" if state.gpu_utilization_percent is not None else "",
            str(state.gpu_memory_used_mb) if state.gpu_memory_used_mb is not None else "",
            str(state.gpu_memory_total_mb) if state.gpu_memory_total_mb is not None else "",
        ]))

    status_file.write_text("\n".join(lines), encoding="utf-8")
    return status_file


def render_screen(
    header_lines: List[str],
    worker_metrics: Dict[int, dict],
    progress_state: Dict[str, FileProgress],
    jobs: List[FileJob],
    script_started_at: float,
    no_clear: bool,
    resource_metrics: Dict[str, dict],
    output_dir: Optional[Path] = None,
) -> None:
    clear_screen(no_clear)

    print("")
    print("=" * 100)

    for line in header_lines:
        print(line)

    elapsed = time.time() - script_started_at

    pending_count = sum(
        1 for state in progress_state.values()
        if state.status in {"Waiting", "Queued"}
    )
    processing_count = sum(
        1 for state in progress_state.values()
        if state.status == "Processing"
    )
    done_count = sum(
        1 for state in progress_state.values()
        if state.status == "Done"
    )
    error_count = sum(
        1 for state in progress_state.values()
        if state.status == "Error"
    )

    eta_sec = estimate_eta(progress_state, pending_count)

    print(f"Tempo total decorrido: {format_duration(elapsed)}")
    print(
        f"Concluídos: {done_count}/{len(jobs)} | "
        f"Processando: {processing_count} | "
        f"Na fila: {pending_count} | "
        f"Erros: {error_count}"
    )

    if eta_sec is not None:
        print(f"ETA estimado: {format_duration(eta_sec)}")
    else:
        print("ETA estimado: calculando...")

    print("")

    run_resource = resource_metrics.get("run", {})
    if run_resource:
        print("RECURSOS DA EXECUÇÃO")
        print("-" * 100)
        print(
            f"CPU avg {run_resource.get('cpu_avg', 0.0):.1f}% | "
            f"CPU pico {run_resource.get('cpu_peak', 0.0):.1f}% | "
            f"RAM avg {run_resource.get('rss_avg_mb', 0.0):.1f}MB | "
            f"RAM pico {run_resource.get('rss_peak_mb', 0.0):.1f}MB"
        )
        print("")

    for line in render_worker_lines(worker_metrics, resource_metrics):
        print(line)

    print("")
    print("FILA / ETAPA")
    print("-" * 100)
    print(
        f"{'GRUPO':<20} "
        f"{'ARQUIVO':<34} "
        f"{'TAM':>8} "
        f"{'ETAPA':<22} "
        f"{'ÁUDIO':>13} "
        f"{'PROG':>5} "
        f"{'ETA':>9} "
        f"{'FILA':>4} "
        f"{'WORKER':<8} "
        f"{'T. ETAPA':<9} "
        f"{'TEMPO':<10}"
    )

    queue_positions = _queue_positions(jobs, progress_state)

    for job in _sorted_jobs_for_display(jobs):
        state = progress_state[job.key]

        if state.status in {"Waiting", "Queued"}:
            queue_pos = str(queue_positions.get(job.key, "-"))
        else:
            queue_pos = "-"

        elapsed_text = ""
        if state.status == "Processing" and state.started_at:
            elapsed_text = format_duration(time.time() - state.started_at)
        elif state.duration_sec is not None:
            elapsed_text = format_duration(state.duration_sec)

        stage_elapsed_text = ""
        if state.status == "Processing" and state.stage_started_at is not None:
            stage_elapsed_text = format_duration(time.time() - state.stage_started_at)

        worker_text = f"PID {state.pid}" if state.pid else "-"
        progress_text = f"{state.percent}%" if state.percent is not None else "--"
        audio_text = "--"
        if state.current_frames is not None and state.total_frames is not None:
            audio_text = f"{state.current_frames}/{state.total_frames}"
        eta_text = format_duration(state.eta_sec) if state.eta_sec is not None else "--"

        print(
            build_status_row(
                group_name=job.group_name or "Raiz",
                file_name=job.display_path or job.path.name,
                size_human=human_size(job.size_bytes),
                stage=state.stage,
                audio_text=audio_text,
                progress_text=progress_text,
                eta_text=eta_text,
                queue_pos=queue_pos,
                stage_elapsed_text=stage_elapsed_text,
                elapsed_text=elapsed_text,
                worker_text=worker_text,
            )
        )

    if output_dir is not None:
        status_file = write_live_status_tsv(output_dir, jobs, progress_state)
        print("")
        print(f"Status detalhado salvo em: {status_file}")
