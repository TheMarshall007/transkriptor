from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path


class MediaService:
    @staticmethod
    def get_duration_seconds(input_file: Path) -> float:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(input_file),
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )

        data = json.loads(result.stdout)
        return float(data["format"]["duration"])

    @staticmethod
    def extract_audio_chunk(
        input_file: Path,
        start_time: float,
        duration: float,
    ) -> Path:
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_path = Path(temp_file.name)
        temp_file.close()

        command = [
            "ffmpeg",
            "-y",
            "-ss",
            str(start_time),
            "-i",
            str(input_file),
            "-t",
            str(duration),
            "-ar",
            "16000",
            "-ac",
            "1",
            str(temp_path),
        ]

        subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )

        return temp_path