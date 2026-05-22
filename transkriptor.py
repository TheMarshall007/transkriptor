from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import subprocess
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional

try:
    import psutil
except ImportError:
    psutil = None

from event_handlers import fake_progress_tick, queue_listener_loop
from file_discovery import scan_files
from formatting import (
    build_grouped_output_filename,
    format_duration,
    human_size,
    natural_sort_key,
    write_grouped_transcription_file,
)
from metrics import build_metrics_payload, clamp_workers, suggested_workers
from models import FileJob, FileProgress
from monitoring import resource_monitor_loop
from ui import render_header, render_screen
from worker import transcribe_file_worker


def setup_logger(log_dir: Path) -> logging.Logger:
    from datetime import datetime

    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("transkriptor")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        log_dir / f"transkriptor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def renderer_loop(
    header_lines,
    worker_metrics: Dict[int, dict],
    progress_state: Dict[str, FileProgress],
    jobs,
    script_started_at: float,
    stop_event: threading.Event,
    no_clear: bool,
    resource_metrics: Dict[str, dict],
    output_dir: Path,
) -> None:
    while not stop_event.is_set():
        for state in progress_state.values():
            fake_progress_tick(state)

        render_screen(
            header_lines=header_lines,
            worker_metrics=worker_metrics,
            progress_state=progress_state,
            jobs=jobs,
            script_started_at=script_started_at,
            no_clear=no_clear,
            resource_metrics=resource_metrics,
            output_dir=output_dir,
        )
        time.sleep(0.4)

    render_screen(
        header_lines=header_lines,
        worker_metrics=worker_metrics,
        progress_state=progress_state,
        jobs=jobs,
        script_started_at=script_started_at,
        no_clear=no_clear,
        resource_metrics=resource_metrics,
        output_dir=output_dir,
    )


def get_media_duration_seconds(file_path: Path) -> Optional[float]:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                str(file_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        payload = json.loads(result.stdout)
        duration = payload.get("format", {}).get("duration")
        return float(duration) if duration is not None else None
    except Exception:
        return None


def build_initial_file_analysis(jobs) -> Dict[str, dict]:
    file_analysis: Dict[str, dict] = {}

    for execution_priority_index, job in enumerate(jobs, start=1):
        media_duration_sec = get_media_duration_seconds(job.path)
        size_mb = job.size_bytes / (1024 * 1024)

        file_analysis[job.path.name] = {
            "execution_priority_index": execution_priority_index,
            "size_bytes": job.size_bytes,
            "size_mb": size_mb,
            "media_duration_sec": media_duration_sec,
            "processing_duration_sec": None,
            "realtime_factor": None,
            "word_count": 0,
            "char_count": 0,
            "segment_count": 0,
            "mb_per_processing_second": None,
            "words_per_processing_second": None,
            "words_per_media_minute": None,
        }

    return file_analysis


def display_sort_key(job: FileJob, input_root: Path):
    try:
        relative = job.path.relative_to(input_root)
        parts = relative.parts
        return [natural_sort_key(part) for part in parts]
    except ValueError:
        return [natural_sort_key(job.path.name)]


def group_jobs_by_folder(jobs: list[FileJob]) -> dict[Path, list[FileJob]]:
    groups: dict[Path, list[FileJob]] = {}
    for job in jobs:
        folder = job.path.parent
        groups.setdefault(folder, []).append(job)
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcrição local com Whisper")
    parser.add_argument("input_path", help="Arquivo ou pasta de entrada")
    parser.add_argument("--output-dir", default="transcricoes", help="Pasta de saída")
    parser.add_argument("--log-dir", default="logs", help="Pasta dos logs")
    parser.add_argument("--workers", type=int, default=1, help="Quantidade de workers")
    parser.add_argument("--model", default="small", help="Modelo Whisper")
    parser.add_argument("--language", default="pt", help="Idioma do áudio")
    parser.add_argument("--metrics-json", default=None, help="Salvar métricas em JSON")
    parser.add_argument(
        "--group-by-folder",
        action="store_true",
        help="Gerar um arquivo agrupado para cada pasta que contém mídias",
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Não limpar a tela a cada atualização",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=1.0,
        help="Intervalo de amostragem de CPU/RAM em segundos",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    log_dir = Path(args.log_dir).expanduser().resolve()

    logger = setup_logger(log_dir)

    if not input_path.exists():
        logger.error(f"Caminho não encontrado: {input_path}")
        raise SystemExit(1)

    discovered_jobs = scan_files(input_path)

    if not discovered_jobs:
        logger.warning("Nenhum arquivo suportado encontrado.")
        raise SystemExit(0)

    # Ordem de execução: maior -> menor
    execution_jobs = sorted(discovered_jobs, key=lambda job: job.size_bytes, reverse=True)

    # Ordem de exibição: ordem natural de pasta/arquivo
    display_jobs = sorted(discovered_jobs, key=lambda job: display_sort_key(job, input_path))

    workers = clamp_workers(args.workers)
    if args.workers != workers:
        logger.warning(
            f"Workers ajustado de {args.workers} para {workers} "
            f"por segurança de CPU/memória."
        )

    logger.info(f"Workers sugerido para esta máquina: {suggested_workers()}")
    if workers > 1:
        logger.warning(
            "Whisper em CPU com múltiplos processos pode consumir muita CPU/RAM. "
            "Se houver falhas, volte para --workers 1."
        )

    if psutil is None:
        logger.warning(
            "psutil não está instalado. "
            "Métricas de CPU/RAM ficarão desabilitadas."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    progress_state: Dict[str, FileProgress] = {
        job.path.name: FileProgress(status="Queued", percent=0)
        for job in discovered_jobs
    }

    worker_metrics: Dict[int, dict] = {}
    resource_metrics: Dict[str, dict] = {"run": {}, "workers": {}}
    file_analysis = build_initial_file_analysis(execution_jobs)
    folder_analysis: Dict[str, dict] = {}

    logger.info("Estrutura encontrada (ordem natural por pasta/arquivo):")
    for job in display_jobs:
        try:
            relative = job.path.relative_to(input_path)
        except ValueError:
            relative = job.path

        analysis = file_analysis[job.path.name]
        media_duration_sec = analysis.get("media_duration_sec")
        media_duration_text = (
            format_duration(media_duration_sec)
            if media_duration_sec is not None
            else "desconhecida"
        )
        logger.info(
            f"[AULA] {relative} | {human_size(job.size_bytes)} | mídia={media_duration_text}"
        )

    logger.info("Fila de execução por tamanho (maior -> menor):")
    for execution_index, job in enumerate(execution_jobs, start=1):
        try:
            relative = job.path.relative_to(input_path)
        except ValueError:
            relative = job.path

        logger.info(
            f"[EXEC {execution_index}] {relative} | {human_size(job.size_bytes)}"
        )

    if args.group_by_folder:
        groups_preview = group_jobs_by_folder(discovered_jobs)
        logger.info("Agrupamento por pasta habilitado:")
        for folder, folder_jobs in sorted(groups_preview.items(), key=lambda item: str(item[0]).lower()):
            preview_name = build_grouped_output_filename(input_path, folder)
            logger.info(
                f"[GROUP] {folder} | arquivos={len(folder_jobs)} | saída={preview_name}"
            )

    success_count = 0
    error_count = 0
    script_started_at = time.time()
    stop_event = threading.Event()

    with mp.Manager() as manager:
        event_queue = manager.Queue()

        logger.info(
            f"Iniciando processamento | arquivos={len(discovered_jobs)} | "
            f"workers={workers} | modelo={args.model} | idioma={args.language}"
        )

        header_lines = render_header(
            input_path=input_path,
            jobs=display_jobs,
            model_name=args.model,
            workers=workers,
            language=args.language,
            output_dir=output_dir,
        )

        listener_thread = threading.Thread(
            target=queue_listener_loop,
            args=(event_queue, progress_state, worker_metrics, logger, stop_event),
            daemon=True,
        )
        listener_thread.start()

        ui_thread = threading.Thread(
            target=renderer_loop,
            args=(
                header_lines,
                worker_metrics,
                progress_state,
                display_jobs,
                script_started_at,
                stop_event,
                args.no_clear,
                resource_metrics,
                output_dir,
            ),
            daemon=True,
        )
        ui_thread.start()

        resource_thread = None
        if psutil is not None:
            resource_thread = threading.Thread(
                target=resource_monitor_loop,
                args=(
                    stop_event,
                    worker_metrics,
                    resource_metrics,
                    logger,
                    args.sample_interval,
                ),
                daemon=True,
            )
            resource_thread.start()

        futures = {}

        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                for job in execution_jobs:
                    future = executor.submit(
                        transcribe_file_worker,
                        str(job.path),
                        str(input_path),
                        str(output_dir),
                        args.model,
                        args.language,
                        event_queue,
                    )
                    futures[future] = job

                for future in as_completed(futures):
                    job = futures[future]
                    analysis = file_analysis[job.path.name]

                    try:
                        result = future.result()

                        if result.get("status") == "success":
                            success_count += 1

                            processing_duration_sec = result.get("duration_sec", 0.0)
                            word_count = result.get("word_count", 0)
                            char_count = result.get("char_count", 0)
                            segment_count = result.get("segment_count", 0)

                            media_duration_sec = analysis.get("media_duration_sec")
                            size_mb = analysis.get("size_mb", 0.0)

                            realtime_factor = (
                                media_duration_sec / processing_duration_sec
                                if media_duration_sec and processing_duration_sec > 0
                                else None
                            )

                            mb_per_processing_second = (
                                size_mb / processing_duration_sec
                                if processing_duration_sec > 0
                                else None
                            )

                            words_per_processing_second = (
                                word_count / processing_duration_sec
                                if processing_duration_sec > 0
                                else None
                            )

                            words_per_media_minute = (
                                word_count / (media_duration_sec / 60)
                                if media_duration_sec and media_duration_sec > 0
                                else None
                            )

                            analysis.update({
                                "processing_duration_sec": processing_duration_sec,
                                "realtime_factor": realtime_factor,
                                "word_count": word_count,
                                "char_count": char_count,
                                "segment_count": segment_count,
                                "mb_per_processing_second": mb_per_processing_second,
                                "words_per_processing_second": words_per_processing_second,
                                "words_per_media_minute": words_per_media_minute,
                            })

                        else:
                            error_count += 1

                            state = progress_state[job.path.name]
                            state.status = "Error"
                            state.percent = 100
                            state.error = result.get("error", "Erro desconhecido")
                            state.duration_sec = result.get("duration_sec")

                            analysis.update({
                                "processing_duration_sec": result.get("duration_sec"),
                            })

                            logger.error(
                                f"[ERROR/FALLBACK] {job.path.name} | "
                                f"erro={state.error}"
                            )
                            logger.error(
                                f"[TRACEBACK/FALLBACK] {job.path.name}\n"
                                f"{result.get('traceback', '')}"
                            )

                    except Exception:
                        error_count += 1
                        state = progress_state[job.path.name]
                        state.status = "Error"
                        state.percent = 100
                        state.error = "Falha ao recuperar resultado do processo"

                        logger.error(
                            f"Erro ao processar {job.path.name}:\n"
                            f"{traceback.format_exc()}"
                        )

        finally:
            stop_event.set()
            listener_thread.join(timeout=2)
            ui_thread.join(timeout=2)

            if resource_thread is not None:
                resource_thread.join(timeout=2)

    if args.group_by_folder:
        grouped_jobs = group_jobs_by_folder(discovered_jobs)
        for folder, folder_jobs in sorted(grouped_jobs.items(), key=lambda item: str(item[0]).lower()):
            group_result = write_grouped_transcription_file(
                root_input_path=input_path,
                folder=folder,
                jobs=folder_jobs,
                output_dir=output_dir,
                file_analysis=file_analysis,
            )
            folder_analysis[str(folder)] = group_result
            logger.info(
                f"[GROUP DONE] {folder} | "
                f"arquivos={group_result['file_count']} | "
                f"mídia_total={format_duration(group_result['total_media_duration_sec'])} | "
                f"palavras={group_result['total_word_count']} | "
                f"saída={group_result['output_path']}"
            )

    total_script_sec = time.time() - script_started_at

    total_mb = sum(a["size_mb"] for a in file_analysis.values())
    total_media_duration_sec = sum(
        a["media_duration_sec"] or 0.0 for a in file_analysis.values()
    )
    total_word_count = sum(a["word_count"] or 0 for a in file_analysis.values())

    completed_processing_durations = [
        a["processing_duration_sec"]
        for a in file_analysis.values()
        if a["processing_duration_sec"] is not None
    ]
    avg_file_sec = (
        sum(completed_processing_durations) / len(completed_processing_durations)
        if completed_processing_durations else 0.0
    )

    throughput_mb_s = total_mb / total_script_sec if total_script_sec > 0 else 0.0
    success_rate = (success_count / len(discovered_jobs)) * 100 if discovered_jobs else 0.0

    realtime_factors = [
        a["realtime_factor"]
        for a in file_analysis.values()
        if a["realtime_factor"] is not None
    ]
    avg_realtime_factor = (
        sum(realtime_factors) / len(realtime_factors)
        if realtime_factors else 0.0
    )
    overall_realtime_factor = (
        total_media_duration_sec / total_script_sec
        if total_script_sec > 0 and total_media_duration_sec > 0
        else 0.0
    )

    model_load_total_sec = sum(
        info.get("model_load_duration_sec", 0.0)
        for info in worker_metrics.values()
    )

    run_resource = resource_metrics.get("run", {})

    fastest_item = None
    slowest_item = None
    completed_items = [
        (name, a)
        for name, a in file_analysis.items()
        if a.get("processing_duration_sec") is not None
    ]
    if completed_items:
        fastest_item = min(
            completed_items,
            key=lambda item: item[1]["processing_duration_sec"],
        )
        slowest_item = max(
            completed_items,
            key=lambda item: item[1]["processing_duration_sec"],
        )

    logger.info("\n===== RESUMO POR ARQUIVO =====")
    for job in display_jobs:
        name = job.path.name
        a = file_analysis[name]
        media_text = (
            format_duration(a["media_duration_sec"])
            if a["media_duration_sec"] is not None
            else "N/A"
        )
        proc_text = (
            format_duration(a["processing_duration_sec"])
            if a["processing_duration_sec"] is not None
            else "N/A"
        )
        rtf_text = (
            f"{a['realtime_factor']:.2f}x"
            if a["realtime_factor"] is not None
            else "N/A"
        )
        wps_text = (
            f"{a['words_per_processing_second']:.2f}"
            if a["words_per_processing_second"] is not None
            else "N/A"
        )
        wpm_media_text = (
            f"{a['words_per_media_minute']:.2f}"
            if a["words_per_media_minute"] is not None
            else "N/A"
        )

        logger.info(
            f"{job.path} | execução={a['execution_priority_index']} | "
            f"{a['size_mb']:.2f} MB | mídia={media_text} | proc={proc_text} | "
            f"RTF={rtf_text} | palavras={a['word_count']} | "
            f"wps={wps_text} | words/min mídia={wpm_media_text}"
        )

    logger.info(
        "Processamento finalizado | "
        f"sucesso={success_count} | erro={error_count} | total={len(discovered_jobs)} | "
        f"tempo_total={format_duration(total_script_sec)} | "
        f"tempo_carga_modelos={format_duration(model_load_total_sec)} | "
        f"media_por_arquivo={format_duration(avg_file_sec) if avg_file_sec else '0s'} | "
        f"throughput={throughput_mb_s:.2f} MB/s | "
        f"rtf_medio={avg_realtime_factor:.2f}x | "
        f"rtf_geral={overall_realtime_factor:.2f}x | "
        f"palavras_totais={total_word_count} | "
        f"grupos_gerados={len(folder_analysis)}"
    )

    if run_resource:
        logger.info(
            "Recursos da execução | "
            f"cpu_avg={run_resource.get('cpu_avg', 0.0):.1f}% | "
            f"cpu_peak={run_resource.get('cpu_peak', 0.0):.1f}% | "
            f"ram_avg={run_resource.get('rss_avg_mb', 0.0):.1f}MB | "
            f"ram_peak={run_resource.get('rss_peak_mb', 0.0):.1f}MB"
        )

    print("")
    print("=" * 80)
    print("RESUMO FINAL")
    print("=" * 80)
    print(f"Sucesso: {success_count}")
    print(f"Erro: {error_count}")
    print(f"Total: {len(discovered_jobs)}")
    print(f"Taxa de sucesso: {success_rate:.1f}%")
    print(f"Tempo total do script: {format_duration(total_script_sec)}")
    print(f"Tempo total de carga de modelo: {format_duration(model_load_total_sec)}")
    print(
        f"Tempo médio por arquivo: "
        f"{format_duration(avg_file_sec) if avg_file_sec else '0s'}"
    )
    print(f"Total processado: {total_mb:.1f}MB")
    print(f"Tempo total de mídia: {format_duration(total_media_duration_sec)}")
    print(f"Throughput médio: {throughput_mb_s:.2f} MB/s")
    print(f"Realtime médio: {avg_realtime_factor:.2f}x")
    print(f"Realtime geral: {overall_realtime_factor:.2f}x")
    print(f"Total de palavras transcritas: {total_word_count}")
    print(f"Grupos gerados: {len(folder_analysis)}")
    print(f"Workers iniciados: {len(worker_metrics)}")

    if run_resource:
        print(f"CPU média da execução: {run_resource.get('cpu_avg', 0.0):.1f}%")
        print(f"CPU pico da execução: {run_resource.get('cpu_peak', 0.0):.1f}%")
        print(f"RAM média da execução: {run_resource.get('rss_avg_mb', 0.0):.1f}MB")
        print(f"RAM pico da execução: {run_resource.get('rss_peak_mb', 0.0):.1f}MB")

    for pid in sorted(worker_metrics.keys()):
        info = worker_metrics[pid]
        resource_info = resource_metrics.get("workers", {}).get(str(pid), {})
        print("")
        print(
            f"Worker PID {pid}: "
            f"arquivos={info.get('files_processed', 0)} | "
            f"tempo_total={format_duration(info.get('total_processing_sec', 0.0))} | "
            f"carga_modelo={format_duration(info.get('model_load_duration_sec', 0.0))}"
        )
        if resource_info:
            print(
                f"  CPU avg {resource_info.get('cpu_avg', 0.0):.1f}% | "
                f"CPU pico {resource_info.get('cpu_peak', 0.0):.1f}% | "
                f"RAM avg {resource_info.get('rss_avg_mb', 0.0):.1f}MB | "
                f"RAM pico {resource_info.get('rss_peak_mb', 0.0):.1f}MB"
            )

    if fastest_item:
        print(
            f"Arquivo mais rápido: {fastest_item[0]} "
            f"({format_duration(fastest_item[1]['processing_duration_sec'])})"
        )

    if slowest_item:
        print(
            f"Arquivo mais lento: {slowest_item[0]} "
            f"({format_duration(slowest_item[1]['processing_duration_sec'])})"
        )

    print("")
    print("RESUMO POR ARQUIVO")
    print("-" * 80)
    for job in display_jobs:
        name = job.path.name
        a = file_analysis[name]
        media_text = (
            format_duration(a["media_duration_sec"])
            if a["media_duration_sec"] is not None
            else "N/A"
        )
        proc_text = (
            format_duration(a["processing_duration_sec"])
            if a["processing_duration_sec"] is not None
            else "N/A"
        )
        rtf_text = (
            f"{a['realtime_factor']:.2f}x"
            if a["realtime_factor"] is not None
            else "N/A"
        )
        try:
            relative = job.path.relative_to(input_path)
        except ValueError:
            relative = job.path

        print(
            f"{relative} | execução={a['execution_priority_index']} | "
            f"{a['size_mb']:.1f}MB | mídia={media_text} | "
            f"proc={proc_text} | RTF={rtf_text} | "
            f"palavras={a['word_count']} | segmentos={a['segment_count']}"
        )

    if folder_analysis:
        print("")
        print("RESUMO POR GRUPO")
        print("-" * 80)
        for folder, group_data in sorted(folder_analysis.items(), key=lambda item: item[0].lower()):
            print(
                f"{folder} | arquivos={group_data['file_count']} | "
                f"mídia_total={format_duration(group_data['total_media_duration_sec'])} | "
                f"palavras={group_data['total_word_count']} | "
                f"saida={group_data['output_path']}"
            )

    print(f"Saída: {output_dir}")
    print(f"Logs: {log_dir}")

    if args.metrics_json:
        metrics_path = Path(args.metrics_json).expanduser().resolve()
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

        payload = build_metrics_payload(
            input_path=input_path,
            output_dir=output_dir,
            log_dir=log_dir,
            jobs=discovered_jobs,
            args=args,
            success_count=success_count,
            error_count=error_count,
            total_script_sec=total_script_sec,
            total_mb=total_mb,
            avg_file_sec=avg_file_sec,
            throughput_mb_s=throughput_mb_s,
            success_rate=success_rate,
            model_load_total_sec=model_load_total_sec,
            worker_metrics=worker_metrics,
            progress_state=progress_state,
            resource_metrics=resource_metrics,
            file_analysis=file_analysis,
            total_media_duration_sec=total_media_duration_sec,
            total_word_count=total_word_count,
            avg_realtime_factor=avg_realtime_factor,
            folder_analysis=folder_analysis,
        )

        metrics_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Métricas JSON: {metrics_path}")
        logger.info(f"Métricas salvas em JSON: {metrics_path}")


if __name__ == "__main__":
    mp.freeze_support()
    main()