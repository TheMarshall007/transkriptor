# Transkriptor

Aplicativo local para transcrição de áudio e vídeo com OpenAI Whisper. O processamento é feito neste computador; os resultados são exportados como TXT, SRT e JSON.

## Recursos

- Interface desktop para selecionar modelo, idioma, CPU ou GPU CUDA.
- Entrada por arrastar e soltar e seletores de arquivos/pastas, com varredura recursiva.
- Tabela única de fila, com uma linha por mídia atualizada até conclusão ou falha.
- Progresso por frames reais do Whisper, etapa e ETA por arquivo.
- Modal de métricas por arquivo: atual, mínimo, média e máximo de CPU/RAM do processo e, quando disponível por PID, uso GPU e VRAM.
- CLI mantido para execução individual ou em lote.

## Instalação no Windows

No terminal, dentro da pasta do projeto:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Instale FFmpeg e confirme que `ffmpeg` e `ffprobe` estão no `PATH`. Para CUDA, instale uma versão de PyTorch com suporte à sua GPU usando o seletor oficial do [PyTorch](https://pytorch.org/get-started/locally/); depois confirme `torch.cuda.is_available()`. O aplicativo permite CPU se CUDA não estiver disponível.

Whisper baixa automaticamente os pesos do modelo na primeira utilização. O modelo `medium` exige mais memória e espaço em disco que os modelos menores.

## Abrir a interface

```powershell
.\.venv\Scripts\python.exe transkriptor_gui.py
```

## Criar o executável para Windows

O build gera uma pasta distribuível em `dist\Transkriptor`, com o aplicativo e as dependências incluídas. Para criar:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-build.txt
powershell -NoProfile -ExecutionPolicy Bypass -File .\build_windows.ps1
```

Para usar outro interpretador Python:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\build_windows.ps1 -Python "C:\caminho\para\python.exe"
```

Execute `dist\Transkriptor\Transkriptor.exe` ou distribua a pasta `dist\Transkriptor` completa. O build inclui Whisper, PyTorch e PySide6; por isso, a pasta é grande. O FFmpeg e o ffprobe precisam estar instalados no computador e disponíveis no `PATH`. Os pesos Whisper são baixados para o cache local na primeira execução de cada modelo. Para aceleração CUDA, o computador de destino também precisa de GPU NVIDIA, driver compatível e bibliotecas CUDA que o PyTorch instalado consiga carregar. O executável não requer uma instalação separada de Python.

Na tela, selecione modelo, dispositivo e idioma; adicione arquivos e pastas pelo botão ou arraste-os para a tabela. Pastas são percorridas recursivamente e itens repetidos são deduplicados. Remova linhas selecionadas antes de iniciar e use o botão de olho para consultar as métricas daquele arquivo. A tabela atualiza status, etapa, progresso e ETA na própria linha. Os resultados e o `live_status.tsv` da GUI ficam na subpasta configurada (por padrão, `transcricoes`) dentro da pasta de cada mídia. O CLI continua usando a pasta global indicada por `--output-dir`.

O tema acompanha as configurações do sistema por padrão. As opções claro e escuro ficam disponíveis no menu Tema.

## CLI

Para transcrever um arquivo ou uma pasta inteira:

```powershell
.\.venv\Scripts\python.exe transkriptor.py "C:\caminho\para\audio-ou-pasta" --model medium --language pt --device auto --workers 1
```

`--device` aceita `auto`, `cpu` ou `cuda`; `auto` escolhe CUDA quando disponível. Também estão disponíveis `--output-dir`, `--log-dir`, `--workers` e `--metrics-json`. Consulte todas as opções com:

```powershell
.\.venv\Scripts\python.exe transkriptor.py --help
```

## Saídas e progresso

Transcrições e `live_status.tsv` são gravados em `transcricoes` por padrão; logs vão para `logs`. O TSV mantém as colunas anteriores e acrescenta modelo, dispositivo, frames, ETA e tempos de carga/processamento. O percentual da etapa de transcrição vem do avanço real de frames informado pelo Whisper.

O CLI mantém o relatório textual. A GUI atualiza a tabela e grava o TSV durante a execução, preservando as colunas antigas e acrescentando métricas por PID. O painel GPU mostra dados globais da placa; o modal e o TSV registram apenas dados atribuídos ao processo Transkriptor por NVML. Se o driver não fornecer telemetria por processo, a métrica aparece como indisponível e a transcrição continua normalmente.
