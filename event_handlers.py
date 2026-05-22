"""
Event handling for worker and progress updates.
"""

import logging
import queue
import time
from typing import Dict

from formatting import format_duration
from models import FileProgress


def fake_progress_tick(state: FileProgress) -> None:
    """
    Simulate progress increment while processing.
    This creates smooth visual progress even when actual progress is unknown.
    """
    if state.status != "Processing":
        return

    if state.percent < 3:
        state.percent += 1
    elif state.percent < 10:
        state.percent += 2
    elif state.percent < 20:
        state.percent += 1
    elif state.percent < 40:
        state.percent += 2
    elif state.percent < 60:
        state.percent += 1
    elif state.percent < 75:
        state.percent += 1
    elif state.percent < 85:
        state.percent += 1
    elif state.percent < 93:
        state.percent += 1
    else:
        state.percent = 95

    state.percent = min(state.percent, 95)


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

        if event_type == "started":
            state.status = "Processing"
            state.started_at = event.get("timestamp", time.time())
            state.pid = event.get("pid")
            state.percent = max(state.percent, 1)

            pid = state.pid
            worker_metrics.setdefault(pid, {})
            worker_metrics[pid].setdefault("files_processed", 0)
            worker_metrics[pid].setdefault("total_processing_sec", 0.0)

            logger.info(f"[START] {file_name} | pid={state.pid}")

        elif event_type == "finished":
            state.status = "Done"
            state.finished_at = event.get("timestamp", time.time())
            state.duration_sec = event.get("duration_sec", 0.0)
            state.percent = 100

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
            state.finished_at = event.get("timestamp", time.time())
            state.duration_sec = event.get("duration_sec", 0.0)
            state.percent = 100
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
                f"[ERROR] {file_name} | tempo={format_duration(duration_sec)} | "
                f"erro={state.error}"
            )
            logger.error(f"[TRACEBACK] {file_name}\n{tb}")

