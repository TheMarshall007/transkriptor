from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import whisper

from services.media_service import MediaService


WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]


class TranscriptionService:
    def __init__(self) -> None:
        self._loaded_models: dict[str, Any] = {}

    def _get_model(self, model_name: str):
        if model_name not in WHISPER_MODELS:
            raise ValueError(f"Unsupported model: {model_name}")

        if model_name not in self._loaded_models:
            self._loaded_models[model_name] = whisper.load_model(model_name)

        return self._loaded_models[model_name]

    def transcribe_file_with_progress(
        self,
        input_file: Path,
        model_name: str = "small",
        language: str = "pt",
        chunk_duration: int = 30,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> dict:
        if not input_file.exists():
            raise FileNotFoundError(f"File not found: {input_file}")

        total_duration = MediaService.get_duration_seconds(input_file)
        model = self._get_model(model_name)

        full_text_parts: list[str] = []
        all_segments: list[dict] = []

        current_start = 0.0
        processed_until = 0.0

        while current_start < total_duration:
            current_chunk_duration = min(chunk_duration, total_duration - current_start)

            if progress_callback is not None:
                progress_callback(
                    (processed_until / total_duration) * 100,
                    f"Processing chunk starting at {current_start:.0f}s",
                )

            chunk_path = MediaService.extract_audio_chunk(
                input_file=input_file,
                start_time=current_start,
                duration=current_chunk_duration,
            )

            try:
                result = model.transcribe(
                    str(chunk_path),
                    language=language,
                    fp16=False,
                )
            finally:
                if chunk_path.exists():
                    chunk_path.unlink()

            chunk_text = result.get("text", "").strip()
            if chunk_text:
                full_text_parts.append(chunk_text)

            for segment in result.get("segments", []):
                adjusted_segment = dict(segment)
                adjusted_segment["start"] = float(segment["start"]) + current_start
                adjusted_segment["end"] = float(segment["end"]) + current_start
                all_segments.append(adjusted_segment)

            processed_until = current_start + current_chunk_duration
            current_start += current_chunk_duration

            if progress_callback is not None:
                progress_callback(
                    min((processed_until / total_duration) * 100, 100.0),
                    f"Processed {processed_until:.0f}s of {total_duration:.0f}s",
                )

        final_result = {
            "text": "\n".join(full_text_parts).strip(),
            "segments": all_segments,
            "language": language,
        }

        if progress_callback is not None:
            progress_callback(100.0, "Transcription completed")

        return final_result