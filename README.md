# Transkriptor

Command-line tool for local audio and video transcription using OpenAI Whisper.

## Overview

Transkriptor performs offline transcription of audio and video files and exports the results in multiple formats. The script uses Whisper for speech recognition and provides a simple CLI interface for selecting the input file, output directory, model size, and language.

## Features

- Local transcription without external API calls
- Support for audio and video input files
- Output generation in TXT, SRT, and JSON formats
- Configurable Whisper model selection
- Language selection through CLI arguments
- Automatic creation of the output directory

## Requirements

- Python 3.9 or higher
- pip
- ffmpeg available in the system environment

## Installation

Install the Python dependencies:

```bash
pip install openai-whisper torch
```

Depending on the environment, `torch` may already be installed as a dependency of `openai-whisper`. Installing it explicitly helps avoid runtime issues on some systems.

## Usage

Run the script from the project directory:

```bash
python transkriptor.py "input_file.mp3"
```

Example:

```bash
python transkriptor.py "meeting_recording.mp3"
```

## CLI Arguments

```bash
python transkriptor.py "input_file.mp3" --output_dir output --model medium --language pt
```

### Positional Argument

| Argument | Description |
|----------|-------------|
| `input_file` | Path to the input audio or video file |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--output_dir` | `transcricoes` | Directory where output files will be saved |
| `--model` | `small` | Whisper model to use: `tiny`, `base`, `small`, `medium`, `large` |
| `--language` | `pt` | Input language code, such as `pt`, `en`, or `es` |

## Output Files

For an input file named `input_file.mp3`, the script generates:

```text
transcricoes/
├── input_file.txt
├── input_file.srt
└── input_file.json
```

### Output Description

- `TXT`: plain transcription text
- `SRT`: subtitle file with timestamps
- `JSON`: full Whisper response, including metadata and segments

## Notes

- The selected Whisper model is downloaded automatically on first use if it is not already available locally.
- Larger models generally provide better transcription accuracy at the cost of higher memory usage and longer processing time.
- Clear audio quality improves transcription results significantly.
- If `ffmpeg` is not installed or is not available in the system PATH, Whisper may fail to process the input file.

## Example Workflow

```bash
python transkriptor.py "lecture.wav" --output_dir results --model medium --language pt
```

This command will:

1. Load the `medium` Whisper model
2. Transcribe `lecture.wav`
3. Create the `results` directory if it does not exist
4. Save the transcription as TXT, SRT, and JSON

## Technology Stack

- Python
- OpenAI Whisper
- argparse

## License

MIT
