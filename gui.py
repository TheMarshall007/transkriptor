"""PySide6 desktop interface for local Transkriptor runs."""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

from PySide6.QtCore import QDir, QSettings, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QCursor, QDragEnterEvent, QPalette, QDropEvent
from PySide6.QtWidgets import (
    QApplication, QAbstractItemView, QComboBox, QDialog, QDialogButtonBox,
    QFileSystemModel, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPushButton,
    QProgressBar, QTreeView, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from devices import cuda_status
from gpu_info import query_gpu_info
from models import FileProgress
from queue_runner import collect_transcription_jobs, probe_duration
from ui import write_live_status_tsv
from worker import transcribe_file_worker


class SourcePickerDialog(QDialog):
    """Browse the filesystem and select media files, folders, or both."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Adicionar arquivos ou pastas")
        self.resize(860, 600)
        self.current_dir = Path.home()
        if not self.current_dir.is_dir():
            self.current_dir = Path.cwd()

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Selecione um ou mais arquivos e/ou pastas. Pastas serão lidas recursivamente."))
        path_row = QHBoxLayout()
        self.path_edit = QLineEdit(str(self.current_dir))
        self.path_edit.returnPressed.connect(self._go_to_path)
        self.up_button = QPushButton("↑")
        self.up_button.setAccessibleName("Pasta acima")
        self.up_button.clicked.connect(self._go_up)
        path_row.addWidget(self.path_edit, 1)
        path_row.addWidget(self.up_button)
        layout.addLayout(path_row)

        self.model = QFileSystemModel(self)
        self.model.setFilter(QDir.Filter.AllEntries | QDir.Filter.NoDotAndDotDot)
        self.model.setRootPath(str(self.current_dir))
        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.setRootIndex(self.model.index(str(self.current_dir)))
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tree.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tree.setColumnHidden(1, True)
        self.tree.setColumnHidden(3, True)
        self.tree.setColumnWidth(0, 560)
        self.tree.doubleClicked.connect(self._open_directory)
        layout.addWidget(self.tree, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.add_button = buttons.addButton("Adicionar selecionados", QDialogButtonBox.ButtonRole.AcceptRole)
        self.add_button.clicked.connect(self._accept_selection)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_paths(self) -> list[str]:
        return [self.model.filePath(index) for index in self.tree.selectionModel().selectedRows(0)]

    def _set_directory(self, path: Path):
        if not path.is_dir():
            QMessageBox.warning(self, "Pasta não encontrada", f"Não foi possível abrir a pasta:\n{path}")
            return
        self.current_dir = path.resolve()
        self.path_edit.setText(str(self.current_dir))
        self.model.setRootPath(str(self.current_dir))
        self.tree.setRootIndex(self.model.index(str(self.current_dir)))

    def _go_to_path(self):
        self._set_directory(Path(self.path_edit.text()).expanduser())

    def _go_up(self):
        parent = self.current_dir.parent
        if parent != self.current_dir:
            self._set_directory(parent)

    def _open_directory(self, index):
        if self.model.isDir(index):
            self._set_directory(Path(self.model.filePath(index)))

    def _accept_selection(self):
        selected = self.selected_paths()
        if not selected:
            QMessageBox.information(self, "Nenhuma entrada selecionada", "Selecione arquivos ou pastas para adicionar.")
            return
        self.accept()


class DropZone(QFrame):
    paths_dropped = Signal(list)
    choose_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setObjectName("dropZone")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName("Adicionar arquivos ou pastas")
        self.setToolTip("Clique para procurar ou solte arquivos e pastas aqui")
        layout = QVBoxLayout(self)
        title = QLabel("Clique para procurar ou solte arquivos e pastas aqui")
        title.setObjectName("dropTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        hint = QLabel("Pastas são percorridas recursivamente; arquivos e pastas podem ser combinados")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        layout.addWidget(title)
        layout.addWidget(hint)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.choose_requested.emit()
            event.accept()
            return
        super().mousePressEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.choose_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            self.paths_dropped.emit(paths)
            event.acceptProposedAction()
        else:
            event.ignore()


class HoverButton(QPushButton):
    entered = Signal()
    left = Signal()

    def enterEvent(self, event):
        self.entered.emit()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.left.emit()
        super().leaveEvent(event)


class BatchThread(QThread):
    event_received = Signal(dict)
    item_finished = Signal(str, dict)
    batch_finished = Signal()

    def __init__(self, jobs, model: str, language: str, device: str):
        super().__init__()
        self.jobs = jobs
        self.model = model
        self.language = language
        self.device = device

    def run(self):
        events: queue.Queue = queue.Queue()
        stop = threading.Event()

        def forward_events():
            while not stop.is_set() or not events.empty():
                try:
                    event = events.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    self.event_received.emit(event)
                finally:
                    events.task_done()

        pump = threading.Thread(target=forward_events, daemon=True)
        pump.start()
        for job in self.jobs:
            job.media_duration_sec = probe_duration(job.path)
            self.event_received.emit({
                "type": "media_duration",
                "job_id": job.key,
                "media_duration_sec": job.media_duration_sec,
            })
            result = transcribe_file_worker(
                str(job.path),
                str(job.input_root or job.path.parent),
                str(job.output_dir or Path("transcricoes")),
                self.model,
                self.language,
                events,
                self.device,
                job.key,
                job.output_stem,
            )
            events.join()
            self.item_finished.emit(job.key, result)
        stop.set()
        pump.join(timeout=2)
        self.batch_finished.emit()


class GpuMonitor(QThread):
    info_updated = Signal(dict)

    def run(self):
        while not self.isInterruptionRequested():
            self.info_updated.emit(query_gpu_info())
            self.msleep(1800)


class GpuPanel(QFrame):
    entered = Signal()
    leave_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint)
        self.setObjectName("gpuPanel")
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setMinimumWidth(270)
        layout = QVBoxLayout(self)
        self.title = QLabel("GPU")
        self.title.setObjectName("cardTitle")
        self.info = QLabel("Buscando informações…")
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.title)
        layout.addWidget(self.info)

    def enterEvent(self, event):
        self.entered.emit()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.leave_requested.emit()
        super().leaveEvent(event)

    def update_info(self, data: dict):
        if not data.get("available"):
            self.info.setText(f"{data.get('reason', 'Sem telemetria disponível.')}")
            return
        memory = f"{data['memory_used_mb']} / {data['memory_total_mb']} MB"
        rows = [
            f"{data['name']}",
            f"Uso: {data['utilization_percent']}%",
            f"Memória dedicada: {memory}",
            f"Temperatura: {data['temperature_c']} °C",
        ]
        if data.get("torch_allocated_mb") is not None:
            rows.extend([
                f"PyTorch alocada: {data['torch_allocated_mb']} MB",
                f"PyTorch reservada: {data['torch_reserved_mb']} MB",
            ])
        self.info.setText("\n".join(rows))


class MainWindow(QMainWindow):
    MEDIA_EXTENSIONS = {".mp3", ".wav", ".m4a", ".mp4", ".mpeg", ".mpga", ".webm", ".ogg"}

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Transkriptor")
        self.resize(1240, 860)
        self.settings = QSettings("Transkriptor", "Desktop")
        self.sources: list[str] = []
        self.jobs = []
        self.is_processing = False
        self.batch_thread: BatchThread | None = None
        self.states: dict[str, FileProgress] = {}
        self.current_job_id: str | None = None
        self.run_started_at: float | None = None
        self.gpu_data: dict = {}
        self.cpu_samples: list[float] = []
        self.ram_samples: list[float] = []
        self.file_resource_samples: dict[str, list[tuple[float, float]]] = {}
        self.last_status_write = 0.0
        self.status_write_errors: set[Path] = set()
        self.hide_gpu_timer = QTimer(self)
        self.hide_gpu_timer.setSingleShot(True)
        self.hide_gpu_timer.setInterval(250)
        self.hide_gpu_timer.timeout.connect(self._hide_gpu_if_pointer_left)
        self._build_ui()
        self._update_source_state()
        self._restore_theme()
        QApplication.instance().paletteChanged.connect(self._system_theme_changed)
        QApplication.styleHints().colorSchemeChanged.connect(self._system_theme_changed)
        self._load_system_stats()

        self.system_timer = QTimer(self)
        self.system_timer.timeout.connect(self._update_system_stats)
        self.system_timer.start(1000)
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._flush_status)
        self.status_timer.start(500)

        self.gpu_monitor = GpuMonitor(self)
        self.gpu_monitor.info_updated.connect(self._on_gpu_info)
        self.gpu_monitor.start()
        self._update_cuda_control()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main = QVBoxLayout(root)
        main.setContentsMargins(28, 22, 28, 24)
        main.setSpacing(16)

        header = QHBoxLayout()
        brand = QVBoxLayout()
        title = QLabel("Transkriptor")
        title.setObjectName("mainTitle")
        subtitle = QLabel("Transcrição local com progresso real e métricas claras")
        subtitle.setObjectName("muted")
        brand.addWidget(title)
        brand.addWidget(subtitle)
        header.addLayout(brand)
        header.addStretch(1)
        self.gpu_badge = HoverButton("GPU  ·  detalhes")
        self.gpu_badge.setObjectName("gpuBadge")
        self.gpu_badge.setMouseTracking(True)
        self.gpu_badge.entered.connect(self._show_gpu_panel)
        self.gpu_badge.left.connect(self.hide_gpu_timer.start)
        header.addWidget(self.gpu_badge)
        header.addWidget(QLabel("Tema"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItem("Sistema", "system")
        self.theme_combo.addItem("Claro", "light")
        self.theme_combo.addItem("Escuro", "dark")
        self.theme_combo.currentIndexChanged.connect(self._apply_theme)
        header.addWidget(self.theme_combo)
        main.addLayout(header)

        settings_card = self._card()
        settings = QGridLayout(settings_card)
        settings.setHorizontalSpacing(16)
        settings.setVerticalSpacing(8)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large", "turbo"])
        self.model_combo.setCurrentText("medium")
        self.device_combo = QComboBox()
        self.device_combo.addItem("GPU CUDA", "cuda")
        self.device_combo.addItem("CPU", "cpu")
        self.language_edit = QLineEdit("pt")
        self.output_edit = QLineEdit("transcricoes")
        self.output_edit.setPlaceholderText("transcricoes")
        settings.addWidget(self._field("Modelo", self.model_combo), 0, 0)
        settings.addWidget(self._field("Processar em", self.device_combo), 0, 1)
        settings.addWidget(self._field("Idioma", self.language_edit), 0, 2)
        settings.addWidget(self._field("Subpasta de saída em cada origem", self.output_edit), 1, 0, 1, 3)
        main.addWidget(settings_card)

        input_card = self._card()
        input_layout = QVBoxLayout(input_card)
        action_row = QHBoxLayout()
        action_row.addWidget(self._section_label("Arquivos de entrada"))
        action_row.addStretch(1)
        self.remove_button = QPushButton("Remover selecionados")
        self.remove_button.setEnabled(False)
        self.remove_button.clicked.connect(self._remove_selected_sources)
        action_row.addWidget(self.remove_button)
        input_layout.addLayout(action_row)
        self.drop_zone = DropZone()
        self.drop_zone.paths_dropped.connect(self._add_sources)
        self.drop_zone.choose_requested.connect(self._choose_sources)
        input_layout.addWidget(self.drop_zone)
        self.source_list = QListWidget()
        self.source_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.source_list.setMaximumHeight(96)
        self.source_list.itemSelectionChanged.connect(self._update_source_state)
        input_layout.addWidget(self.source_list)
        main.addWidget(input_card)

        body = QHBoxLayout()
        body.setSpacing(16)
        queue_card = self._card()
        queue_layout = QVBoxLayout(queue_card)
        queue_top = QHBoxLayout()
        queue_top.addWidget(self._section_label("Fila de transcrição"))
        queue_top.addStretch(1)
        self.queue_count = QLabel("Nenhum arquivo na fila")
        self.queue_count.setObjectName("muted")
        queue_top.addWidget(self.queue_count)
        queue_layout.addLayout(queue_top)
        self.queue_tree = QTreeWidget()
        self.queue_tree.setColumnCount(4)
        self.queue_tree.setHeaderLabels(["Arquivo", "Status", "Etapa", "Progresso"])
        self.queue_tree.setRootIsDecorated(False)
        self.queue_tree.setAlternatingRowColors(True)
        queue_layout.addWidget(self.queue_tree, 1)
        self.start_button = QPushButton("Iniciar transcrição")
        self.start_button.setObjectName("primaryButton")
        self.start_button.clicked.connect(self._start_batch)
        queue_layout.addWidget(self.start_button)
        body.addWidget(queue_card, 3)

        detail_card = self._card()
        detail_layout = QVBoxLayout(detail_card)
        detail_layout.addWidget(self._section_label("Agora processando"))
        self.active_name = QLabel("Adicione arquivos ou pastas para começar")
        self.active_name.setObjectName("activeName")
        self.active_name.setWordWrap(True)
        detail_layout.addWidget(self.active_name)
        self.stage_label = QLabel("Aguardando entrada")
        self.stage_label.setObjectName("muted")
        detail_layout.addWidget(self.stage_label)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        detail_layout.addWidget(self.progress)
        self.progress_detail = QLabel("O percentual acompanha os frames reais analisados pelo Whisper.")
        self.progress_detail.setObjectName("muted")
        self.progress_detail.setWordWrap(True)
        detail_layout.addWidget(self.progress_detail)
        detail_layout.addSpacing(6)
        self.metrics_grid = QGridLayout()
        self.metric_labels: dict[str, QLabel] = {}
        for index, (key, name) in enumerate([
            ("model", "Modelo / dispositivo"), ("load", "Carga do modelo"),
            ("processing", "Inferência"), ("media", "Duração da mídia"),
            ("rtf", "Velocidade relativa"), ("system", "CPU / memória"),
        ]):
            label = QLabel("—")
            label.setObjectName("metricValue")
            self.metric_labels[key] = label
            self.metrics_grid.addWidget(self._metric(name, label), index // 2, index % 2)
        detail_layout.addLayout(self.metrics_grid)
        self.run_summary = QLabel("Fila: —  ·  Concluídos: 0  ·  Falhas: 0")
        self.run_summary.setObjectName("muted")
        self.run_summary.setWordWrap(True)
        detail_layout.addWidget(self.run_summary)
        detail_layout.addStretch(1)
        body.addWidget(detail_card, 2)
        main.addLayout(body, 1)

        self.gpu_panel = GpuPanel(self)
        self.gpu_panel.entered.connect(self.hide_gpu_timer.stop)
        self.gpu_panel.leave_requested.connect(self.hide_gpu_timer.start)

    def _card(self):
        frame = QFrame()
        frame.setObjectName("card")
        return frame

    @staticmethod
    def _field(title: str, widget: QWidget):
        holder = QWidget()
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        label = QLabel(title)
        label.setObjectName("fieldLabel")
        layout.addWidget(label)
        layout.addWidget(widget)
        return holder

    @staticmethod
    def _section_label(text: str):
        label = QLabel(text)
        label.setObjectName("sectionTitle")
        return label

    @staticmethod
    def _metric(title: str, value: QLabel):
        frame = QFrame()
        frame.setObjectName("metricCard")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 8, 10, 8)
        caption = QLabel(title)
        caption.setObjectName("muted")
        layout.addWidget(caption)
        layout.addWidget(value)
        return frame

    def _choose_sources(self):
        dialog = SourcePickerDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._add_sources(dialog.selected_paths())

    def _add_sources(self, paths: list[str]):
        if self.is_processing:
            return
        added = 0
        invalid: list[str] = []
        for raw in paths:
            if not raw:
                continue
            try:
                path = Path(raw).expanduser().resolve()
                if not path.exists() or not (path.is_file() or path.is_dir()):
                    invalid.append(str(raw))
                    continue
            except (OSError, RuntimeError, ValueError):
                invalid.append(str(raw))
                continue
            normalized = str(path)
            if normalized not in self.sources:
                self.sources.append(normalized)
                self.source_list.addItem(QListWidgetItem(normalized))
                added += 1
        self._update_source_state()
        if invalid:
            preview = "\n".join(invalid[:10])
            remainder = f"\n… e mais {len(invalid) - 10} caminho(s)." if len(invalid) > 10 else ""
            QMessageBox.warning(self, "Entrada não adicionada",
                                f"Alguns caminhos não existem ou não podem ser acessados:\n{preview}{remainder}")
        elif added == 0 and paths:
            self.statusBar().showMessage("As entradas já estavam na lista.", 5000)

    def _remove_selected_sources(self):
        for item in self.source_list.selectedItems():
            row = self.source_list.row(item)
            self.sources.remove(item.text())
            self.source_list.takeItem(row)
        self._update_source_state()

    def _update_source_state(self):
        has_sources = bool(self.sources)
        running = self.is_processing
        self.start_button.setEnabled(has_sources and not running)
        self.drop_zone.setEnabled(not running)
        self.source_list.setEnabled(not running)
        self.remove_button.setEnabled(bool(self.source_list.selectedItems()) and not running)
        self.active_name.setText(f"{len(self.sources)} entrada(s) selecionada(s)" if has_sources else "Adicione arquivos ou pastas para começar")

    def _update_cuda_control(self):
        available, detail = cuda_status()
        self.device_combo.setItemData(0, detail, Qt.ItemDataRole.ToolTipRole)
        self.device_combo.model().item(0).setEnabled(available)
        self.device_combo.setEnabled(True)
        if available:
            self.device_combo.setCurrentIndex(0)
            self.gpu_badge.setText("GPU  ·  detalhes")
        else:
            self.device_combo.setCurrentIndex(1)
            self.gpu_badge.setText("GPU indisponível  ·  detalhes")
            self.gpu_badge.setToolTip(detail)

    def _start_batch(self):
        output_name = self.output_edit.text().strip()
        output_path = Path(output_name)
        invalid_windows_name = any(character in output_name for character in '<>:"/\\|?*')
        if (not output_name or output_name in {".", ".."} or output_path.is_absolute()
                or output_path.name != output_name or output_path.drive
                or invalid_windows_name or output_name.rstrip(" .") != output_name):
            QMessageBox.warning(self, "Subpasta de saída",
                                "Informe apenas o nome de uma subpasta, por exemplo: transcricoes.")
            return

        jobs, warnings = collect_transcription_jobs(
            self.sources, output_path, output_next_to_source=True)
        if warnings:
            QMessageBox.warning(self, "Entradas ignoradas", "\n".join(warnings[:15]))
        if not jobs:
            QMessageBox.information(self, "Fila vazia", "Nenhum arquivo de áudio ou vídeo compatível foi encontrado.")
            return

        self.jobs = jobs
        self.states = {
            job.key: FileProgress(status="Queued", stage="Na fila", model_name=self.model_combo.currentText(),
                                  device=self.device_combo.currentData(), media_duration_sec=job.media_duration_sec)
            for job in jobs
        }
        self.current_job_id = None
        self.run_started_at = time.time()
        self.cpu_samples.clear()
        self.ram_samples.clear()
        self.file_resource_samples.clear()
        self.status_write_errors.clear()
        self._render_queue()
        self.queue_count.setText(f"{len(jobs)} arquivo(s) · processados em sequência")
        self.is_processing = True
        self._update_source_state()
        self.device_combo.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.language_edit.setEnabled(False)
        self.output_edit.setEnabled(False)
        self.batch_thread = BatchThread(jobs, self.model_combo.currentText(), self.language_edit.text().strip() or "pt",
                                        self.device_combo.currentData())
        self.batch_thread.event_received.connect(self._handle_event)
        self.batch_thread.item_finished.connect(self._handle_item_finished)
        self.batch_thread.batch_finished.connect(self._batch_finished)
        self.batch_thread.start()

    def _render_queue(self):
        self.queue_tree.clear()
        for job in self.jobs:
            state = self.states[job.key]
            percent = f"{state.percent}%" if state.percent is not None else "—"
            item = QTreeWidgetItem([job.display_path or job.path.name, state.status, state.stage, percent])
            item.setData(0, Qt.ItemDataRole.UserRole, job.key)
            item.setToolTip(0, str(job.path))
            if state.error:
                item.setToolTip(1, state.error)
            self.queue_tree.addTopLevelItem(item)
        self.queue_tree.resizeColumnToContents(0)

    def _handle_event(self, event: dict):
        kind = event.get("type")
        job_id = event.get("job_id") or self.current_job_id
        state = self.states.get(job_id) if job_id else None
        if kind == "media_duration" and state:
            state.media_duration_sec = event.get("media_duration_sec")
        elif kind == "model_loading" and state:
            state.model_name = event.get("model_name", state.model_name)
            state.device = event.get("device", state.device)
            self.file_resource_samples[job_id] = []
        elif kind == "model_loaded" and state:
            state.model_load_sec = event.get("load_duration_sec")
        elif state and kind == "started":
            self.current_job_id = job_id
            state.status = "Processando"
            state.stage = "Carregando modelo"
            state.started_at = event.get("timestamp", time.time())
            state.stage_started_at = state.started_at
            state.model_name = event.get("model_name", state.model_name)
            state.device = event.get("device", state.device)
            self.active_name.setText(self._display_name(job_id))
            self.metric_labels["model"].setText(f"{state.model_name} · {state.device.upper()}")
        elif state and kind == "stage_changed":
            state.stage = event.get("stage", state.stage)
            state.stage_started_at = event.get("timestamp", time.time())
        elif state and kind == "transcription_progress":
            if state.stage != "Transcrevendo":
                state.stage_started_at = event.get("timestamp", time.time())
            state.stage = "Transcrevendo"
            state.current_frames = event.get("current_frames")
            state.total_frames = event.get("total_frames")
            if state.total_frames:
                state.percent = min(99, int(state.current_frames * 100 / state.total_frames))
                now = event.get("timestamp", time.time())
                if state.progress_started_at is None:
                    state.progress_started_at = now
                    state.progress_start_frames = state.current_frames
                advanced = state.current_frames - state.progress_start_frames
                elapsed = now - state.progress_started_at
                if advanced > 0 and elapsed > 0:
                    rate = advanced / elapsed
                    state.eta_sec = max(0, (state.total_frames - state.current_frames) / rate)
        elif state and kind == "finished":
            state.status = "Concluído"
            state.stage = "Concluído"
            state.percent = 100
            state.eta_sec = None
            state.duration_sec = event.get("duration_sec")
            state.model_load_sec = event.get("model_load_sec", state.model_load_sec)
            state.processing_duration_sec = event.get("processing_duration_sec")
        elif state and kind == "error":
            state.status = "Falha"
            state.stage = f"Falha: {event.get('stage', state.stage)}"
            state.error = event.get("error", "Erro desconhecido")
            state.duration_sec = event.get("duration_sec")
            state.eta_sec = None
        if state:
            self._show_progress(state)
        self._render_queue()

    def _show_progress(self, state: FileProgress):
        self.stage_label.setText(state.stage)
        self.progress.setValue(state.percent or 0)
        if state.stage == "Transcrevendo" and state.current_frames is not None and state.total_frames:
            eta = self._format_duration(state.eta_sec) if state.eta_sec is not None else "calculando"
            self.progress_detail.setText(f"Áudio analisado: {state.current_frames:,} / {state.total_frames:,} frames  ·  ETA {eta}")
        elif state.percent is not None:
            self.progress_detail.setText(f"{state.percent}% concluído")
        else:
            elapsed = self._format_duration(time.time() - state.stage_started_at) if state.stage_started_at else "0s"
            self.progress_detail.setText(f"{state.stage} há {elapsed}; progresso de frames ainda indisponível.")
        self.metric_labels["model"].setText(f"{state.model_name or '—'} · {state.device.upper() if state.device else '—'}")
        self.metric_labels["load"].setText(self._format_duration(state.model_load_sec))
        self.metric_labels["processing"].setText(self._format_duration(state.processing_duration_sec))
        self.metric_labels["media"].setText(self._format_duration(state.media_duration_sec))
        self.metric_labels["rtf"].setText(f"{state.realtime_factor:.2f}×" if state.realtime_factor is not None else "—")

    def _display_name(self, job_id: str):
        job = next((job for job in self.jobs if job.key == job_id), None)
        return job.display_path if job else job_id

    def _handle_item_finished(self, job_id: str, result: dict):
        state = self.states[job_id]
        if result.get("status") == "success":
            state.status = "Concluído"
            state.stage = "Concluído"
            state.percent = 100
            state.duration_sec = result.get("duration_sec")
            state.model_name = result.get("model_name", state.model_name)
            state.device = result.get("device", state.device)
            state.model_load_sec = result.get("model_load_sec", state.model_load_sec)
            state.processing_duration_sec = result.get("processing_duration_sec")
            if state.media_duration_sec and state.processing_duration_sec:
                state.realtime_factor = state.media_duration_sec / state.processing_duration_sec
        else:
            state.status = "Falha"
            state.stage = f"Falha: {result.get('stage', state.stage)}"
            state.error = result.get("error", "Erro desconhecido")
        self._render_queue()
        self._update_summary()

    def _batch_finished(self):
        self._render_queue()
        self._update_summary()
        self.is_processing = False
        self.source_list.setEnabled(True)
        self.device_combo.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.language_edit.setEnabled(True)
        self.output_edit.setEnabled(True)
        self._update_source_state()
        self.stage_label.setText("Fila finalizada")

    def _update_summary(self):
        done = sum(state.status == "Concluído" for state in self.states.values())
        failed = sum(state.status == "Falha" for state in self.states.values())
        details = f"Fila: {len(self.jobs)}  ·  Concluídos: {done}  ·  Falhas: {failed}"
        if self.cpu_samples:
            details += f"  ·  CPU média/pico: {sum(self.cpu_samples) / len(self.cpu_samples):.0f}% / {max(self.cpu_samples):.0f}%"
        if self.ram_samples:
            details += f"  ·  RAM pico: {max(self.ram_samples):.0f} MB"
        self.run_summary.setText(details)

    def _flush_status(self):
        if not self.jobs:
            return
        grouped: dict[Path, list] = {}
        for job in self.jobs:
            grouped.setdefault(job.output_dir, []).append(job)
        for output_dir, jobs in grouped.items():
            states = {job.key: self.states[job.key] for job in jobs}
            try:
                write_live_status_tsv(output_dir, jobs, states)
            except OSError as exc:
                if output_dir not in self.status_write_errors:
                    self.status_write_errors.add(output_dir)
                    self.statusBar().showMessage(
                        f"Não foi possível atualizar o status em {output_dir}: {exc}", 8000)

    def _load_system_stats(self):
        try:
            import psutil
            self.process = psutil.Process()
            self.process.cpu_percent(None)
        except ImportError:
            self.process = None

    def _update_system_stats(self):
        cpu = ram = None
        if self.process:
            try:
                cpu = self.process.cpu_percent(None)
                ram = self.process.memory_info().rss / 1048576
                if self.run_started_at:
                    self.cpu_samples.append(cpu)
                    self.ram_samples.append(ram)
                    if self.current_job_id in self.file_resource_samples:
                        samples = self.file_resource_samples[self.current_job_id]
                        samples.append((cpu, ram))
                        state = self.states[self.current_job_id]
                        state.cpu_avg = sum(sample[0] for sample in samples) / len(samples)
                        state.cpu_peak = max(sample[0] for sample in samples)
                        state.ram_avg_mb = sum(sample[1] for sample in samples) / len(samples)
                        state.ram_peak_mb = max(sample[1] for sample in samples)
                        self._show_progress(state)
            except Exception:
                pass
        if cpu is not None:
            self.metric_labels["system"].setText(f"CPU {cpu:.0f}% · RAM {ram:.0f} MB")

    def _on_gpu_info(self, data: dict):
        self.gpu_data = data
        self.gpu_panel.update_info(data)
        state = self.states.get(self.current_job_id) if self.current_job_id else None
        if state and data.get("available"):
            state.gpu_utilization_percent = data.get("utilization_percent")
            state.gpu_memory_used_mb = data.get("memory_used_mb")
            state.gpu_memory_total_mb = data.get("memory_total_mb")
        if data.get("available"):
            self.gpu_badge.setText(f"GPU  ·  {data['utilization_percent']}%")
        elif not cuda_status()[0]:
            self.gpu_badge.setText("GPU indisponível  ·  detalhes")

    def _show_gpu_panel(self, event=None):
        self.hide_gpu_timer.stop()
        self.gpu_panel.adjustSize()
        point = self.gpu_badge.mapToGlobal(self.gpu_badge.rect().bottomLeft())
        self.gpu_panel.move(point.x() - self.gpu_panel.width() + self.gpu_badge.width(), point.y() + 7)
        self.gpu_panel.show()

    def _hide_gpu_if_pointer_left(self):
        widget = QApplication.widgetAt(QCursor.pos())
        if widget is self.gpu_badge or widget is self.gpu_panel or (widget and self.gpu_panel.isAncestorOf(widget)):
            return
        self.gpu_panel.hide()

    def _restore_theme(self):
        value = self.settings.value("theme", "system")
        index = self.theme_combo.findData(value)
        self.theme_combo.setCurrentIndex(max(0, index))
        self._apply_theme()

    def _apply_theme(self, *_):
        theme = self.theme_combo.currentData() or "system"
        self.settings.setValue("theme", theme)
        system_dark = QApplication.palette().color(QPalette.ColorRole.Window).lightness() < 128
        dark = theme == "dark" or (theme == "system" and system_dark)
        background = "#111827" if dark else "#f3f6fb"
        card = "#1b2638" if dark else "#ffffff"
        text = "#edf2f7" if dark else "#1b2638"
        muted = "#9aa9bd" if dark else "#68778d"
        border = "#2d3b50" if dark else "#dce4ee"
        accent = "#6ea8fe" if dark else "#2563eb"
        QApplication.instance().setStyleSheet(f"""
            QWidget {{ background: {background}; color: {text}; font-size: 13px; }}
            QFrame#card, QFrame#gpuPanel {{ background: {card}; border: 1px solid {border}; border-radius: 14px; }}
            QFrame#metricCard {{ background: {background}; border: 1px solid {border}; border-radius: 10px; }}
            QFrame#dropZone {{ background: {background}; border: 2px dashed {accent}; border-radius: 12px; min-height: 78px; }}
            QLabel#mainTitle {{ font-size: 26px; font-weight: 700; }}
            QLabel#sectionTitle, QLabel#cardTitle {{ font-size: 16px; font-weight: 650; }}
            QLabel#activeName {{ font-size: 17px; font-weight: 600; }}
            QLabel#muted, QLabel#fieldLabel {{ color: {muted}; }}
            QLabel#metricValue {{ font-size: 15px; font-weight: 600; }}
            QLineEdit, QComboBox, QListWidget, QTreeWidget {{ background: {card}; border: 1px solid {border}; border-radius: 7px; padding: 7px; }}
            QTreeWidget::item {{ padding: 7px; }}
            QPushButton {{ background: {card}; border: 1px solid {border}; border-radius: 8px; padding: 8px 12px; }}
            QPushButton:hover {{ border-color: {accent}; }}
            QPushButton#primaryButton {{ background: {accent}; color: white; border: 0; font-weight: 700; padding: 11px; }}
            QPushButton#gpuBadge {{ color: {accent}; font-weight: 650; }}
            QProgressBar {{ background: {background}; border: 0; border-radius: 6px; height: 13px; text-align: center; }}
            QProgressBar::chunk {{ background: {accent}; border-radius: 6px; }}
        """)

    def _system_theme_changed(self, *_):
        if self.theme_combo.currentData() == "system":
            self._apply_theme()

    @staticmethod
    def _format_duration(seconds):
        if seconds is None:
            return "—"
        seconds = max(0, int(seconds))
        if seconds >= 3600:
            return f"{seconds // 3600}h {(seconds % 3600) // 60}m"
        if seconds >= 60:
            return f"{seconds // 60}m {seconds % 60}s"
        return f"{seconds}s"

    def closeEvent(self, event):
        if getattr(self, "batch_thread", None) and self.batch_thread.isRunning():
            QMessageBox.information(self, "Transcrição em andamento", "Aguarde a fila terminar antes de fechar a janela.")
            event.ignore()
            return
        if hasattr(self, "gpu_monitor"):
            self.gpu_monitor.requestInterruption()
            self.gpu_monitor.wait(2500)
        event.accept()


def main():
    app = QApplication([])
    app.setOrganizationName("Transkriptor")
    app.setApplicationName("Transkriptor Desktop")
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
