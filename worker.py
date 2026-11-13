import importlib
import json
import os
import time
import traceback
from pathlib import Path
from typing import Optional

import whisper

from formatting import get_output_paths, write_srt
from devices import resolve_device

_MODELS: dict[tuple[str, str], object] = {}
_MODEL_LOAD_SECONDS: dict[tuple[str, str], float] = {}


def transcribe_with_progress(
    model,
    input_file: Path,
    language: str,
    event_queue,
    stage_state: dict,
    job_id: str = "",
    device: str = "cpu",
):
    """Forward Whisper's real audio-frame progress to the parent process."""
    transcribe_module = importlib.import_module("whisper.transcribe")
    tqdm_module = transcribe_module.tqdm
    original_tqdm = tqdm_module.tqdm

    class ProgressTqdm(original_tqdm):
        def __init__(self, *args, **kwargs):
            self._tracks_audio_frames = kwargs.get("unit") == "frames"
            if self._tracks_audio_frames:
                kwargs["disable"] = True
            super().__init__(*args, **kwargs)
            if self._tracks_audio_frames:
                self._report_progress()

        def update(self, n=1):
            if self._tracks_audio_frames:
                self.n += n
                self._report_progress()
                return None
            result = super().update(n)
            return result

        def _report_progress(self):
            timestamp = time.time()
            stage_state["stage"] = "Transcrevendo"
            event_queue.put({
                "type": "transcription_progress",
                "file_name": input_file.name,
                "job_id": job_id,
                "pid": os.getpid(),
                "current_frames": self.n,
                "total_frames": self.total,
                "timestamp": timestamp,
            })

    tqdm_module.tqdm = ProgressTqdm
    try:
        return model.transcribe(
            str(input_file),
            language=language,
            fp16=device == "cuda",
            verbose=False,
        )
    finally:
        tqdm_module.tqdm = original_tqdm


def get_or_load_model(model_name: str, device: str, event_queue):
    key = (model_name, device)
    if key in _MODELS:
        return _MODELS[key], 0.0

    load_start = time.time()

    event_queue.put({
        "type": "model_loading",
        "pid": os.getpid(),
        "model_name": model_name,
        "device": device,
        "timestamp": load_start,
    })

    model = whisper.load_model(model_name, device=device)
    _MODELS[key] = model
    _MODEL_LOAD_SECONDS[key] = time.time() - load_start

    event_queue.put({
        "type": "model_loaded",
        "pid": os.getpid(),
        "model_name": model_name,
        "device": device,
        "timestamp": time.time(),
        "load_duration_sec": _MODEL_LOAD_SECONDS[key],
    })

    return model, _MODEL_LOAD_SECONDS[key]


def transcribe_file_worker(
    input_file_str: str,
    input_root_str: str,
    output_dir_str: str,
    model_name: str,
    language: str,
    event_queue,
    device: str = "auto",
    job_id: str = "",
    output_stem: Optional[str] = None,
) -> dict:
    input_file = Path(input_file_str)
    input_root = Path(input_root_str)
    output_dir = Path(output_dir_str)
    start_time = time.time()
    stage_state = {"stage": "Carregando modelo"}

    try:
        device = resolve_device(device)
        output_dir.mkdir(parents=True, exist_ok=True)

        event_queue.put({
            "type": "started",
            "file_name": input_file.name,
            "job_id": job_id,
            "pid": os.getpid(),
            "timestamp": start_time,
            "model_name": model_name,
            "device": device,
        })

        model, model_load_sec = get_or_load_model(model_name, device, event_queue)

        stage_state["stage"] = "Preparando áudio"
        event_queue.put({
            "type": "stage_changed",
            "file_name": input_file.name,
            "job_id": job_id,
            "stage": stage_state["stage"],
            "timestamp": time.time(),
        })
        transcription_started_at = time.time()
        result = transcribe_with_progress(
            model,
            input_file,
            language,
            event_queue,
            stage_state,
            job_id,
            device,
        )
        transcription_duration_sec = time.time() - transcription_started_at

        stage_state["stage"] = "Salvando resultados"
        event_queue.put({
            "type": "stage_changed",
            "file_name": input_file.name,
            "job_id": job_id,
            "stage": stage_state["stage"],
            "timestamp": time.time(),
        })

        text = result.get("text", "").strip()
        segments = result.get("segments", [])

        txt_path, srt_path, json_path = get_output_paths(output_dir, input_root, input_file, output_stem)

        txt_path.write_text(text, encoding="utf-8")
        write_srt(result, srt_path)
        json_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        end_time = time.time()
        duration_sec = end_time - start_time

        word_count = len(text.split()) if text else 0
        char_count = len(text)
        segment_count = len(segments)

        event_queue.put({
            "type": "finished",
            "file_name": input_file.name,
            "job_id": job_id,
            "timestamp": end_time,
            "duration_sec": duration_sec,
            "stage": "Concluído",
            "model_name": model_name,
            "device": device,
            "model_load_sec": model_load_sec,
            "processing_duration_sec": transcription_duration_sec,
            "outputs": {
                "txt": str(txt_path),
                "srt": str(srt_path),
                "json": str(json_path),
            }
        })

        return {
            "status": "success",
            "file_name": input_file.name,
            "job_id": job_id,
            "duration_sec": duration_sec,
            "model_name": model_name,
            "device": device,
            "model_load_sec": model_load_sec,
            "processing_duration_sec": transcription_duration_sec,
            "stage": "Concluído",
            "word_count": word_count,
            "char_count": char_count,
            "segment_count": segment_count,
            "outputs": {
                "txt": str(txt_path),
                "srt": str(srt_path),
                "json": str(json_path),
            }
        }

    except Exception as exc:
        end_time = time.time()
        error_msg = str(exc)
        tb_str = traceback.format_exc()
        duration_sec = end_time - start_time

        event_queue.put({
            "type": "error",
            "file_name": input_file.name,
            "job_id": job_id,
            "timestamp": end_time,
            "duration_sec": duration_sec,
            "stage": stage_state["stage"],
            "model_name": model_name,
            "device": device,
            "error": error_msg,
            "traceback": tb_str,
        })

        return {
            "status": "error",
            "file_name": input_file.name,
            "job_id": job_id,
            "duration_sec": duration_sec,
            "model_name": model_name,
            "device": device,
            "stage": stage_state["stage"],
            "error": error_msg,
            "traceback": tb_str,
        }
