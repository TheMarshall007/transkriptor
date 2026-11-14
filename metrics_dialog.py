"""Per-file resource statistics dialog for the desktop queue."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QGridLayout, QLabel, QVBoxLayout


class FileMetricsDialog(QDialog):
    METRICS = [
        ("CPU do processo (%)", "cpu_percent"),
        ("RAM do processo (MB)", "ram_mb"),
        ("GPU do processo (%)", "gpu_utilization_percent"),
        ("VRAM do processo (MB)", "gpu_memory_mb"),
    ]

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Métricas · {title}")
        self.setMinimumWidth(520)
        self.values = {}
        layout = QVBoxLayout(self)
        self.heading = QLabel(title)
        self.heading.setWordWrap(True)
        self.heading.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.heading)
        grid = QGridLayout()
        for row, (label, key) in enumerate(self.METRICS):
            grid.addWidget(QLabel(label), row, 0)
            self.values[key] = QLabel("—")
            grid.addWidget(self.values[key], row, 1)
        layout.addLayout(grid)
        self.gpu_note = QLabel("")
        self.gpu_note.setWordWrap(True)
        layout.addWidget(self.gpu_note)

    @staticmethod
    def _fmt(value):
        return f"{value:.1f}" if value is not None else "—"

    def update_metrics(self, state):
        metrics = state.resource_metrics
        for key, series in vars(metrics).items():
            if key not in self.values:
                continue
            current = self._fmt(series.current)
            minimum = self._fmt(series.minimum)
            average = self._fmt(series.average)
            maximum = self._fmt(series.maximum)
            self.values[key].setText(f"Atual {current} · Mín {minimum} · Méd {average} · Máx {maximum}")
        if metrics.gpu_available is False:
            self.gpu_note.setText(f"Telemetria por processo indisponível: {metrics.gpu_unavailable_reason or 'driver não fornece dados por PID'}")
        elif metrics.gpu_available is True:
            unavailable = []
            if metrics.gpu_utilization_available is False:
                unavailable.append("utilização da GPU indisponível por PID")
            if metrics.gpu_memory_available is False:
                unavailable.append("VRAM por PID indisponível")
            note = "; ".join(unavailable) if unavailable else "uso da GPU e VRAM atribuídos ao processo"
            self.gpu_note.setText(f"NVML: {note}.")
        else:
            self.gpu_note.setText("Aguardando amostras de recursos…")
