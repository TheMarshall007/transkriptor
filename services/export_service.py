from __future__ import annotations

import json
from pathlib import Path

from utils.time_utils import format_timestamp


def save_txt(output_path: Path, text: str) -> None:
    output_path.write_text(text.strip(), encoding="utf-8")


def save_srt(output_path: Path, segments: list[dict]) -> None:
    lines: list[str] = []

    for index, segment in enumerate(segments, start=1):
        start = format_timestamp(segment["start"])
        end = format_timestamp(segment["end"])
        text = segment["text"].strip()

        lines.append(str(index))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_json(output_path: Path, result: dict) -> None:
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_all_outputs(output_dir: Path, base_name: str, result: dict) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_path = output_dir / f"{base_name}.txt"
    srt_path = output_dir / f"{base_name}.srt"
    json_path = output_dir / f"{base_name}.json"

    save_txt(txt_path, result["text"])
    save_srt(srt_path, result.get("segments", []))
    save_json(json_path, result)

    return {
        "txt": txt_path,
        "srt": srt_path,
        "json": json_path,
    }