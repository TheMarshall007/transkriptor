from __future__ import annotations

import argparse
from pathlib import Path

from services.transcription_service import TranscriptionService, WHISPER_MODELS


def run_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Local transcription with Whisper"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input audio or video file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="transcricoes",
        help="Directory where output files will be saved",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="small",
        choices=WHISPER_MODELS,
        help="Whisper model",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="pt",
        help="Audio language, for example: pt, en, es",
    )

    args = parser.parse_args()

    service = TranscriptionService()

    print(f"Loading model: {args.model}")
    print(f"Transcribing file: {args.input_file}")

    response = service.transcribe_file(
        input_file=Path(args.input_file),
        output_dir=Path(args.output_dir),
        model_name=args.model,
        language=args.language,
    )

    print("\nTranscription completed successfully.")
    print(f"TXT:  {response['paths']['txt']}")
    print(f"SRT:  {response['paths']['srt']}")
    print(f"JSON: {response['paths']['json']}")