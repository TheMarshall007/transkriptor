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
    QMainWindow, QMessageBox, QPushButton,
    QTreeView, QVBoxLayout, QWidget,
)

from devices import cuda_status
from gpu_info import query_gpu_info, query_gpu_process_metrics
from models import FileProgress
from metrics_dialog import FileMetricsDialog
from queue_model import EyeButtonDelegate, QueueTableModel, QueueTableView
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


class DiscoveryThread(QThread):
    discovered = Signal(list, list)

    def __init__(self, sources, output_name):
        super().__init__()
        self.sources = list(sources)
        self.output_name = output_name

    def run(self):
        try:
            jobs, warnings = collect_transcription_jobs(
                self.sources, Path(self.output_name), output_next_to_source=True)
        except Exception as exc:
            jobs, warnings = [], [f"Falha ao percorrer as entradas: {exc}"]
        self.discovered.emit(jobs, warnings)


class GpuMonitor(QThread):
    info_updated = Signal(dict)

    def run(self):
        last_seen = None
        while not self.isInterruptionRequested():
            process = query_gpu_process_metrics(__import__("os").getpid(), last_seen)
            last_seen = process.get("last_seen_us", last_seen)
            self.info_updated.emit({"device": query_gpu_info(), "process": process})
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
        self.excluded_paths: set[str] = set()
        self.metric_dialogs: dict[str, FileMetricsDialog] = {}
        self.is_processing = False
        self.is_discovering = False
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

        queue_card = self._card()
        queue_layout = QVBoxLayout(queue_card)
        action_row = QHBoxLayout()
        action_row.addWidget(self._section_label("Fila de transcrição"))
        action_row.addStretch(1)
        self.queue_count = QLabel("Nenhum arquivo na fila")
        self.queue_count.setObjectName("muted")
        action_row.addWidget(self.queue_count)
        self.add_button = QPushButton("Adicionar arquivos/pastas")
        self.add_button.clicked.connect(self._choose_sources)
        action_row.addWidget(self.add_button)
        self.remove_button = QPushButton("Remover selecionados")
        self.remove_button.setEnabled(False)
        self.remove_button.clicked.connect(self._remove_selected_sources)
        action_row.addWidget(self.remove_button)
        queue_layout.addLayout(action_row)
        self.queue_model = QueueTableModel(self)
        self.queue_table = QueueTableView()
        self.queue_table.setModel(self.queue_model)
        self.queue_table.paths_dropped.connect(self._add_sources)
        self.queue_table.selectionModel().selectionChanged.connect(self._update_source_state)
        self.queue_table.metrics_requested.connect(self._show_file_metrics)
        self.eye_delegate = EyeButtonDelegate(self.queue_table)
        self.eye_delegate.activated.connect(self._show_file_metrics)
        self.queue_table.setItemDelegateForColumn(QueueTableModel.DETAILS_COLUMN, self.eye_delegate)
        self.queue_table.horizontalHeader().setSectionResizeMode(0, self.queue_table.horizontalHeader().ResizeMode.Stretch)
        self.queue_table.horizontalHeader().setSectionResizeMode(1, self.queue_table.horizontalHeader().ResizeMode.Stretch)
        for column, width in ((2, 95), (3, 105), (4, 150), (5, 95), (6, 95), (7, 48)):
            self.queue_table.setColumnWidth(column, width)
        queue_layout.addWidget(self.queue_table, 1)
        self.start_button = QPushButton("Iniciar transcrição")
        self.start_button.setObjectName("primaryButton")
        self.start_button.clicked.connect(self._start_batch)
        queue_layout.addWidget(self.start_button)
        main.addWidget(queue_card, 1)
        self.status_line = QLabel("Adicione arquivos ou pastas para começar")
        self.status_line.setObjectName("muted")
        main.addWidget(self.status_line)
        self.run_summary = QLabel("Fila: —  ·  Concluídos: 0  ·  Falhas: 0")
        self.run_summary.setObjectName("muted")
        main.addWidget(self.run_summary)
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
        if self.is_processing or self.is_discovering:
            return
        valid = []
        invalid = []
        for raw in paths:
            try:
                path = Path(raw).expanduser().resolve()
                if path.exists() and (path.is_file() or path.is_dir()):
                    valid.append(str(path))
                else:
                    invalid.append(str(raw))
            except (OSError, RuntimeError, ValueError):
                invalid.append(str(raw))
        known = {str(Path(item).resolve()).casefold() for item in self.sources}
        self.sources.extend(item for item in valid if item.casefold() not in known)
        if self.sources:
            self.is_discovering = True
            self.queue_table.setEnabled(False)
            self.add_button.setEnabled(False)
            self.status_line.setText("Procurando mídias nas entradas…")
            self.discovery_thread = DiscoveryThread(self.sources, self.output_edit.text().strip() or "transcricoes")
            self.discovery_thread.discovered.connect(self._discovery_finished)
            self.discovery_thread.start()
        if invalid:
            QMessageBox.warning(self, "Entrada não adicionada", "Não foi possível acessar:\n" + "\n".join(invalid[:10]))

    def _discovery_finished(self, jobs, warnings):
        existing = {str(job.path.resolve()).casefold(): job for job in self.jobs}
        excluded = getattr(self, "excluded_paths", set())
        for job in jobs:
            key_path = str(job.path.resolve()).casefold()
            if key_path not in existing and key_path not in excluded:
                self.jobs.append(job)
                self.states[job.key] = FileProgress(status="Waiting", stage="Pronto")
        self.jobs.sort(key=lambda job: str(job.path).casefold())
        self.queue_model.replace_rows(self.jobs, self.states)
        self.is_discovering = False
        self.queue_table.setEnabled(True)
        self._update_source_state()
        self.add_button.setEnabled(not self.is_processing)
        self.status_line.setText(f"{len(self.jobs)} arquivo(s) carregado(s)" if self.jobs else "Nenhuma mídia compatível encontrada")
        if warnings:
            QMessageBox.warning(self, "Entradas ignoradas", "\n".join(warnings[:15]))

    def _remove_selected_sources(self):
        if self.is_processing:
            return
        rows = sorted({index.row() for index in self.queue_table.selectionModel().selectedRows()})
        removed = self.queue_model.remove_rows(rows)
        for job in removed:
            self.states.pop(job.key, None)
            self.excluded_paths.add(str(job.path.resolve()).casefold())
        self.jobs = list(self.queue_model.jobs)
        self._update_source_state()

    def _update_source_state(self, *_):
        running = self.is_processing
        has_jobs = bool(self.jobs)
        self.start_button.setEnabled(has_jobs and not running and not self.is_discovering)
        if hasattr(self, "add_button"):
            self.add_button.setEnabled(not running and not self.is_discovering)
        selected_rows = self.queue_table.selectionModel().selectedRows() if hasattr(self, "queue_table") else []
        ready_selected = any(self.states.get(self.jobs[index.row()].key, FileProgress()).status in ("Waiting", "Pronto")
                             for index in selected_rows if index.row() < len(self.jobs))
        self.remove_button.setEnabled(ready_selected and not running)
        self.queue_count.setText(f"{len(self.jobs)} arquivo(s) na fila" if has_jobs else "Nenhum arquivo na fila")

    def _show_file_metrics(self, job_id: str):
        state = self.states.get(job_id)
        job = self.queue_model.job_for_key(job_id)
        if not state or not job:
            return
        current = self.metric_dialogs.get(job_id)
        if current:
            current.raise_()
            current.activateWindow()
            return
        dialog = FileMetricsDialog(job.display_path or job.path.name, self)
        dialog.setModal(True)
        dialog.update_metrics(state)
        self.metric_dialogs = getattr(self, "metric_dialogs", {})
        self.metric_dialogs[job_id] = dialog
        dialog.finished.connect(lambda *_args, key=job_id: self.metric_dialogs.pop(key, None))
        dialog.open()
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

        jobs = list(self.jobs)
        for job in jobs:
            job.output_dir = job.path.parent / output_name
        if not jobs:
            QMessageBox.information(self, "Fila vazia", "Nenhum arquivo de áudio ou vídeo compatível foi encontrado.")
            return

        self.jobs = jobs
        self.states = {
            job.key: FileProgress(status="Waiting", stage="Na fila", model_name=self.model_combo.currentText(),
                                  device=self.device_combo.currentData(), media_duration_sec=job.media_duration_sec)
            for job in jobs
        }
        for index, job in enumerate(jobs):
            self.states[job.key].status = "Queued"
            self.states[job.key].stage = "Na fila"
        self.current_job_id = None
        self.run_started_at = time.time()
        self.cpu_samples.clear()
        self.ram_samples.clear()
        self.file_resource_samples.clear()
        self.status_write_errors.clear()
        self.queue_model.replace_rows(jobs, self.states)
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
        for job in self.jobs:
            self.queue_model.notify_state(job.key)

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
            state.pid = event.get("pid")
            state.started_at = event.get("timestamp", time.time())
            state.stage_started_at = state.started_at
            state.model_name = event.get("model_name", state.model_name)
            state.device = event.get("device", state.device)
            self.status_line.setText(f"Processando: {self._display_name(job_id)}")
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
        if self.current_job_id:
            eta = f" · ETA {self._format_duration(state.eta_sec)}" if state.eta_sec is not None else ""
            progress = f" · {state.percent}%{eta}" if state.percent is not None else ""
            self.status_line.setText(f"{self._display_name(self.current_job_id)} — {state.stage}{progress}")
        for job_id, dialog in self.metric_dialogs.items():
            if self.states.get(job_id) is state:
                dialog.update_metrics(state)

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
        self._show_progress(state)
        if self.current_job_id == job_id:
            self.current_job_id = None
        self._render_queue()
        self._update_summary()

    def _batch_finished(self):
        self.current_job_id = None
        self._render_queue()
        self._update_summary()
        self.is_processing = False
        self.add_button.setEnabled(True)
        self.device_combo.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.language_edit.setEnabled(True)
        self.output_edit.setEnabled(True)
        self._update_source_state()
        self.status_line.setText("Fila finalizada")

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
        if not self.process:
            return
        try:
            cpu = self.process.cpu_percent(None)
            ram = self.process.memory_info().rss / 1048576
            if self.current_job_id and self.current_job_id in self.states:
                state = self.states[self.current_job_id]
                metrics = state.resource_metrics
                metrics.cpu_percent.add(cpu)
                metrics.ram_mb.add(ram)
                state.cpu_avg = metrics.cpu_percent.average
                state.cpu_peak = metrics.cpu_percent.maximum
                state.ram_avg_mb = metrics.ram_mb.average
                state.ram_peak_mb = metrics.ram_mb.maximum
                self._render_queue()
                for dialog in self.metric_dialogs.values():
                    dialog.update_metrics(state)
        except Exception:
            pass

    def _on_gpu_info(self, data: dict):
        device = data.get("device", {})
        process = data.get("process", {})
        self.gpu_data = device
        self.gpu_panel.update_info(device)
        state = self.states.get(self.current_job_id) if self.current_job_id else None
        if state:
            metrics = state.resource_metrics
            metrics.gpu_available = process.get("available", False)
            metrics.gpu_utilization_available = process.get("utilization_available", False)
            metrics.gpu_memory_available = process.get("memory_available", False)
            metrics.gpu_unavailable_reason = process.get("reason", "")
            metrics.gpu_utilization_percent.add(process.get("utilization_percent"))
            metrics.gpu_memory_mb.add(process.get("memory_mb"))
            # Legacy columns now contain PID-attributed data only; device totals remain in GPU panel.
            state.gpu_utilization_percent = metrics.gpu_utilization_percent.current
            state.gpu_memory_used_mb = int(metrics.gpu_memory_mb.current) if metrics.gpu_memory_mb.current is not None else None
            state.gpu_memory_total_mb = None
            self._render_queue()
            for dialog in self.metric_dialogs.values():
                dialog.update_metrics(state)
        if device.get("available"):
            self.gpu_badge.setText(f"GPU  ·  {device['utilization_percent']}%")
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
            QLineEdit, QComboBox, QTableView {{ background: {card}; border: 1px solid {border}; border-radius: 7px; padding: 7px; }}
            QTableView::item {{ padding: 7px; border-bottom: 1px solid {border}; }}
            QHeaderView::section {{ background: {background}; border: 0; border-bottom: 1px solid {border}; padding: 8px; }}
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
