"""
Event handling for worker and progress updates.
"""

import logging
import queue
import time
from typing import Dict

from formatting import format_duration
from models import FileProgress


def _set_stage(state: FileProgress, stage: str, timestamp: float) -> None:
    if state.stage != stage:
        state.stage = stage
        state.stage_started_at = timestamp


def queue_listener_loop(
    event_queue,
    progress_state: Dict[str, FileProgress],
    worker_metrics: Dict[int, dict],
    logger: logging.Logger,
    stop_event,
) -> None:
    """
    Listen to events from worker processes and update progress state.
    
    Args:
        event_queue: Queue containing events from workers.
        progress_state: Dictionary tracking progress of each file.
        worker_metrics: Dictionary tracking metrics for each worker.
        logger: Logger instance.
        stop_event: Threading event to signal loop shutdown.
    """
    while not stop_event.is_set():
        try:
            event = event_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        except Exception:
            continue

        event_type = event.get("type")

        if event_type == "model_loading":
            pid = event.get("pid")
            model_name = event.get("model_name")

            worker_metrics.setdefault(pid, {})
            worker_metrics[pid]["model_name"] = model_name
            worker_metrics[pid]["model_loading"] = True
            worker_metrics[pid]["model_loaded"] = False
            worker_metrics[pid]["model_load_started_at"] = event.get("timestamp")
            worker_metrics[pid].setdefault("files_processed", 0)
            worker_metrics[pid].setdefault("total_processing_sec", 0.0)

            logger.info(f"[MODEL LOADING] worker pid={pid} | modelo={model_name}")
            continue

        if event_type == "model_loaded":
            pid = event.get("pid")
            model_name = event.get("model_name")
            load_duration_sec = event.get("load_duration_sec", 0.0)

            worker_metrics.setdefault(pid, {})
            worker_metrics[pid]["model_name"] = model_name
            worker_metrics[pid]["model_loading"] = False
            worker_metrics[pid]["model_loaded"] = True
            worker_metrics[pid]["model_load_duration_sec"] = load_duration_sec
            worker_metrics[pid].setdefault("files_processed", 0)
            worker_metrics[pid].setdefault("total_processing_sec", 0.0)

            logger.info(
                f"[MODEL LOADED] worker pid={pid} | modelo={model_name} | "
                f"tempo={format_duration(load_duration_sec)}"
            )
            continue

        file_name = event.get("file_name")
        if file_name not in progress_state:
            continue

        state = progress_state[file_name]
        timestamp = event.get("timestamp", time.time())

        if event_type == "started":
            state.status = "Processing"
            state.started_at = timestamp
            state.pid = event.get("pid")
            state.percent = None
            state.current_frames = None
            state.total_frames = None
            state.eta_sec = None
            state.progress_started_at = None
            state.progress_start_frames = 0
            _set_stage(state, "Carregando modelo", timestamp)

            pid = state.pid
            worker_metrics.setdefault(pid, {})
            worker_metrics[pid].setdefault("files_processed", 0)
            worker_metrics[pid].setdefault("total_processing_sec", 0.0)

            logger.info(f"[START] {file_name} | pid={state.pid}")

        elif event_type == "stage_changed":
            _set_stage(state, event.get("stage", state.stage), timestamp)

        elif event_type == "transcription_progress":
            _set_stage(state, "Transcrevendo", timestamp)
            current_frames = max(0, int(event.get("current_frames") or 0))
            total_frames = event.get("total_frames")
            total_frames = int(total_frames) if total_frames is not None else None
            if total_frames is not None and total_frames <= 0:
                total_frames = None
            elif total_frames is not None:
                current_frames = min(current_frames, total_frames)

            state.current_frames = current_frames
            state.total_frames = total_frames

            if state.progress_started_at is None:
                state.progress_started_at = timestamp
                state.progress_start_frames = current_frames

            state.eta_sec = None
            if total_frames is not None and current_frames > 0:
                state.percent = min(99, int(current_frames * 100 / total_frames))

            if total_frames is not None and current_frames > state.progress_start_frames:
                elapsed = timestamp - state.progress_started_at
                advanced_frames = current_frames - state.progress_start_frames
                if elapsed > 0 and advanced_frames > 0:
                    frames_per_sec = advanced_frames / elapsed
                    state.eta_sec = max(0.0, (total_frames - current_frames) / frames_per_sec)

        elif event_type == "finished":
            state.status = "Done"
            state.finished_at = timestamp
            state.duration_sec = event.get("duration_sec", 0.0)
            state.percent = 100
            state.eta_sec = None
            state.stage = "Concluído"
            state.stage_started_at = timestamp

            pid = state.pid
            if pid is not None:
                worker_metrics.setdefault(pid, {})
                worker_metrics[pid]["files_processed"] = (
                    worker_metrics[pid].get("files_processed", 0) + 1
                )
                worker_metrics[pid]["total_processing_sec"] = (
                    worker_metrics[pid].get("total_processing_sec", 0.0)
                    + state.duration_sec
                )

            duration_sec = event.get("duration_sec", 0.0)
            outputs = event.get("outputs", {})
            logger.info(
                f"[DONE] {file_name} | tempo={format_duration(duration_sec)} "
                f"| txt={outputs.get('txt')} | srt={outputs.get('srt')} | "
                f"json={outputs.get('json')}"
            )

        elif event_type == "error":
            state.status = "Error"
            state.finished_at = timestamp
            state.duration_sec = event.get("duration_sec", 0.0)
            failed_stage = event.get("stage", state.stage)
            state.stage = f"Falha: {failed_stage}"
            state.stage_started_at = timestamp
            state.eta_sec = None
            state.error = event.get("error", "Erro desconhecido")

            pid = state.pid
            if pid is not None:
                worker_metrics.setdefault(pid, {})
                worker_metrics[pid]["total_processing_sec"] = (
                    worker_metrics[pid].get("total_processing_sec", 0.0)
                    + state.duration_sec
                )

            duration_sec = event.get("duration_sec", 0.0)
            tb = event.get("traceback", "")

            logger.error(
                f"[ERROR] {file_name} | etapa={failed_stage} | "
                f"tempo={format_duration(duration_sec)} | "
                f"erro={state.error}"
            )
            logger.error(f"[TRACEBACK] {file_name}\n{tb}")

