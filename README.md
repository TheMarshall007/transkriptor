# Transkriptor

Aplicativo local para transcrição de áudio e vídeo com OpenAI Whisper. O processamento é feito neste computador; os resultados são exportados como TXT, SRT e JSON.

## Recursos

- Interface desktop para selecionar modelo, idioma, CPU ou GPU CUDA.
- Entrada por arrastar e soltar, seletores de arquivo/pasta e varredura recursiva.
- Fila sequencial com progresso por frames reais do Whisper, etapa e ETA.
- Métricas de modelo, dispositivo, tempo de carga, inferência, duração da mídia e fator de tempo real.
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

Na tela, selecione modelo, dispositivo e idioma; adicione arquivos ou pastas arrastando para a área pontilhada ou usando os botões; ajuste a saída se necessário e inicie. Pastas são percorridas recursivamente e sua árvore é mantida na saída. Arquivos avulsos são salvos na raiz da saída; nomes repetidos recebem sufixos para evitar sobrescrita.

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

O CLI mantém o relatório textual. A GUI atualiza a fila e grava o TSV durante a execução. Métricas da GPU dependem de `nvidia-smi`; a transcrição não depende dessa telemetria.
