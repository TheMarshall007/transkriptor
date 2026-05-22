import json
import os
import time
import traceback
from pathlib import Path
from typing import Optional

import whisper

from formatting import get_output_paths, write_srt

_MODEL: Optional[object] = None
_MODEL_NAME: Optional[str] = None
_MODEL_LOAD_SECONDS: Optional[float] = None


def get_or_load_model(model_name: str, event_queue):
    global _MODEL, _MODEL_NAME, _MODEL_LOAD_SECONDS

    if _MODEL is not None and _MODEL_NAME == model_name:
        return _MODEL

    load_start = time.time()

    event_queue.put({
        "type": "model_loading",
        "pid": os.getpid(),
        "model_name": model_name,
        "timestamp": load_start,
    })

    _MODEL = whisper.load_model(model_name)
    _MODEL_NAME = model_name
    _MODEL_LOAD_SECONDS = time.time() - load_start

    event_queue.put({
        "type": "model_loaded",
        "pid": os.getpid(),
        "model_name": model_name,
        "timestamp": time.time(),
        "load_duration_sec": _MODEL_LOAD_SECONDS,
    })

    return _MODEL


def transcribe_file_worker(
    input_file_str: str,
    input_root_str: str,
    output_dir_str: str,
    model_name: str,
    language: str,
    event_queue,
) -> dict:
    input_file = Path(input_file_str)
    input_root = Path(input_root_str)
    output_dir = Path(output_dir_str)
    start_time = time.time()

    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        event_queue.put({
            "type": "started",
            "file_name": input_file.name,
            "pid": os.getpid(),
            "timestamp": start_time,
        })

        model = get_or_load_model(model_name, event_queue)

        result = model.transcribe(
            str(input_file),
            language=language,
            fp16=False,
            verbose=False,
        )

        text = result.get("text", "").strip()
        segments = result.get("segments", [])

        txt_path, srt_path, json_path = get_output_paths(output_dir, input_root, input_file)

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
            "timestamp": end_time,
            "duration_sec": duration_sec,
            "outputs": {
                "txt": str(txt_path),
                "srt": str(srt_path),
                "json": str(json_path),
            }
        })

        return {
            "status": "success",
            "file_name": input_file.name,
            "duration_sec": duration_sec,
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
            "timestamp": end_time,
            "duration_sec": duration_sec,
            "error": error_msg,
            "traceback": tb_str,
        })

        return {
            "status": "error",
            "file_name": input_file.name,
            "duration_sec": duration_sec,
            "error": error_msg,
            "traceback": tb_str,
        }