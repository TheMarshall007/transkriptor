from pathlib import Path
import json
import argparse
import whisper


def format_timestamp(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hours = millis // 3_600_000
    millis %= 3_600_000
    minutes = millis // 60_000
    millis %= 60_000
    secs = millis // 1000
    millis %= 1000
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def save_txt(output_path: Path, text: str) -> None:
    output_path.write_text(text.strip(), encoding="utf-8")


def save_srt(output_path: Path, segments: list) -> None:
    lines = []
    for i, seg in enumerate(segments, start=1):
        start = format_timestamp(seg["start"])
        end = format_timestamp(seg["end"])
        text = seg["text"].strip()
        lines.append(f"{i}")
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_json(output_path: Path, result: dict) -> None:
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def transcribe_file(
    input_file: Path,
    output_dir: Path,
    model_name: str = "small",
    language: str = "pt"
) -> None:
    if not input_file.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {input_file}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Carregando modelo: {model_name}")
    model = whisper.load_model(model_name)

    print(f"Transcrevendo arquivo: {input_file.name}")
    result = model.transcribe(
        str(input_file),
        language=language,
        fp16=False
    )

    base_name = input_file.stem
    txt_path = output_dir / f"{base_name}.txt"
    srt_path = output_dir / f"{base_name}.srt"
    json_path = output_dir / f"{base_name}.json"

    save_txt(txt_path, result["text"])
    save_srt(srt_path, result.get("segments", []))
    save_json(json_path, result)

    print("\nTranscrição concluída com sucesso.")
    print(f"TXT:  {txt_path}")
    print(f"SRT:  {srt_path}")
    print(f"JSON: {json_path}")


def transkriptor() -> None:
    parser = argparse.ArgumentParser(
        description="Transcrição local com Whisper"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Caminho do arquivo de áudio ou vídeo"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="transcricoes",
        help="Pasta onde os arquivos serão salvos"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="small",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Modelo Whisper"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="pt",
        help="Idioma do áudio, ex: pt, en, es"
    )

    args = parser.parse_args()

    transcribe_file(
        input_file=Path(args.input_file),
        output_dir=Path(args.output_dir),
        model_name=args.model,
        language=args.language
    )


if __name__ == "__main__":
    transkriptor()