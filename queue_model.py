"""Single-row-per-media model and interaction helpers for the desktop queue."""

from __future__ import annotations

from PySide6.QtCore import QAbstractTableModel, QEvent, QModelIndex, Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QApplication, QStyle, QStyleOptionButton, QStyledItemDelegate, QTableView,
)

from formatting import human_size
from models import FileJob, FileProgress


class QueueTableModel(QAbstractTableModel):
    FILE_KEY_ROLE = int(Qt.ItemDataRole.UserRole) + 1
    HEADERS = ["Arquivo", "Origem", "Tamanho", "Status", "Etapa", "Progresso", "ETA", ""]
    DETAILS_COLUMN = 7

    def __init__(self, parent=None):
        super().__init__(parent)
        self.jobs: list[FileJob] = []
        self.states: dict[str, FileProgress] = {}
        self.row_by_key: dict[str, int] = {}

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.jobs)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.HEADERS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.HEADERS[section]
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self.jobs):
            return None
        job = self.jobs[index.row()]
        key = job.key
        state = self.states.get(key)
        column = index.column()

        if role == self.FILE_KEY_ROLE:
            return key
        if role == Qt.ItemDataRole.ToolTipRole:
            if column == self.DETAILS_COLUMN:
                return "Abrir métricas deste arquivo"
            if column in (0, 1):
                return str(job.path)
            return state.error if state and state.error else None
        if role == Qt.ItemDataRole.AccessibleTextRole and column == self.DETAILS_COLUMN:
            return "Ver métricas do arquivo"
        if role == Qt.ItemDataRole.TextAlignmentRole and column in (2, 5, 6, self.DETAILS_COLUMN):
            return int(Qt.AlignmentFlag.AlignCenter)
        if role != Qt.ItemDataRole.DisplayRole:
            return None

        status_labels = {"Waiting": "Pronto", "Queued": "Na fila", "Processing": "Processando",
                         "Done": "Concluído", "Error": "Falha"}

        values = [
            job.path.name,
            str(job.path.parent),
            human_size(job.size_bytes),
            status_labels.get(state.status, state.status) if state else "Pronto",
            state.stage if state else "Pronto",
            f"{state.percent}%" if state and state.percent is not None else "—",
            _format_eta(state.eta_sec) if state and state.eta_sec is not None else "—",
            "",
        ]
        return values[column]

    def replace_rows(self, jobs: list[FileJob], states: dict[str, FileProgress]) -> None:
        self.beginResetModel()
        self.jobs = list(jobs)
        self.states = states
        self.row_by_key = {job.key: row for row, job in enumerate(self.jobs)}
        self.endResetModel()

    def remove_rows(self, rows: list[int]) -> list[FileJob]:
        removed = []
        for row in sorted(set(rows), reverse=True):
            if 0 <= row < len(self.jobs):
                self.beginRemoveRows(QModelIndex(), row, row)
                removed.append(self.jobs.pop(row))
                self.endRemoveRows()
        self.row_by_key = {job.key: row for row, job in enumerate(self.jobs)}
        return removed

    def notify_state(self, job_id: str) -> None:
        row = self.row_by_key.get(job_id)
        if row is None:
            return
        self.dataChanged.emit(
            self.index(row, 0), self.index(row, self.DETAILS_COLUMN),
            [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole],
        )

    def job_for_key(self, job_id: str) -> FileJob | None:
        row = self.row_by_key.get(job_id)
        return self.jobs[row] if row is not None else None

    @staticmethod
    def _format_eta(seconds: float) -> str:
        seconds = max(0, int(seconds))
        if seconds >= 3600:
            return f"{seconds // 3600}h {(seconds % 3600) // 60}m"
        if seconds >= 60:
            return f"{seconds // 60}m {seconds % 60}s"
        return f"{seconds}s"


def _format_eta(seconds: float) -> str:
    return QueueTableModel._format_eta(seconds)


class QueueTableView(QTableView):
    paths_dropped = Signal(list)
    metrics_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QTableView.SelectionMode.ExtendedSelection)
        self.setAlternatingRowColors(True)
        self.setShowGrid(False)
        self.setSortingEnabled(False)
        self.verticalHeader().hide()
        self.setWordWrap(False)
        self.setTabKeyNavigation(False)

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

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            index = self.currentIndex()
            if index.isValid():
                key = index.data(QueueTableModel.FILE_KEY_ROLE)
                if key:
                    self.metrics_requested.emit(key)
                    event.accept()
                    return
        super().keyPressEvent(event)


class EyeButtonDelegate(QStyledItemDelegate):
    activated = Signal(str)

    def paint(self, painter: QPainter, option, index):
        button = QStyleOptionButton()
        button.rect = option.rect.adjusted(5, 4, -5, -4)
        button.state = QStyle.StateFlag.State_Enabled
        if option.state & QStyle.StateFlag.State_MouseOver:
            button.state |= QStyle.StateFlag.State_MouseOver
        if option.state & QStyle.StateFlag.State_Selected:
            button.state |= QStyle.StateFlag.State_On
        QApplication.style().drawControl(QStyle.ControlElement.CE_PushButton, button, painter)

        eye = button.rect.adjusted(8, 7, -8, -7)
        center_x = eye.center().x()
        center_y = eye.center().y()
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(option.palette.buttonText().color(), 1.5))
        path = QPainterPath()
        path.moveTo(eye.left(), center_y)
        path.cubicTo(eye.left() + 7, eye.top(), eye.right() - 7, eye.top(), eye.right(), center_y)
        path.cubicTo(eye.right() - 7, eye.bottom(), eye.left() + 7, eye.bottom(), eye.left(), center_y)
        painter.drawPath(path)
        painter.drawEllipse(int(center_x - 2), int(center_y - 2), 4, 4)
        painter.restore()

    def editorEvent(self, event, model, option, index):
        if (event.type() == QEvent.Type.MouseButtonRelease
                and getattr(event, "button", lambda: None)() == Qt.MouseButton.LeftButton
                and option.rect.contains(event.pos())):
            key = index.data(QueueTableModel.FILE_KEY_ROLE)
            if key:
                self.activated.emit(key)
                return True
        return False
