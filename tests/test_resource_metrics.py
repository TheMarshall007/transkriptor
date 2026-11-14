import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

import gpu_info
from metrics_dialog import FileMetricsDialog
from models import FileJob, FileProgress
from queue_model import QueueTableModel
from pathlib import Path


class ResourceMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_series_tracks_current_min_average_and_max_and_ignores_missing_values(self):
        from models import MetricSeries
        series = MetricSeries()
        for value in (10, None, 30, 20):
            series.add(value)
        self.assertEqual(series.current, 20)
        self.assertEqual(series.minimum, 10)
        self.assertEqual(series.average, 20)
        self.assertEqual(series.maximum, 30)
        self.assertEqual(series.samples, 3)

    def test_metrics_dialog_keeps_final_per_file_summary(self):
        state = FileProgress()
        state.resource_metrics.cpu_percent.add(5)
        state.resource_metrics.cpu_percent.add(25)
        state.resource_metrics.ram_mb.add(100)
        state.resource_metrics.ram_mb.add(200)
        state.resource_metrics.gpu_available = False
        state.resource_metrics.gpu_unavailable_reason = "sem amostras por PID"
        dialog = FileMetricsDialog("amostra.wav")
        dialog.update_metrics(state)
        self.assertIn("Atual 25.0 · Mín 5.0 · Méd 15.0 · Máx 25.0", dialog.values["cpu_percent"].text())
        self.assertIn("sem amostras por PID", dialog.gpu_note.text())
        dialog.close()

    def test_gpu_telemetry_reports_unavailable_when_nvml_is_missing(self):
        gpu_info._nvml_initialized = False
        with patch.dict("sys.modules", {"pynvml": None}):
            result = gpu_info.query_gpu_process_metrics(12345)
        self.assertFalse(result["available"])
        self.assertIsNone(result["utilization_percent"])
        self.assertIsNone(result["memory_mb"])
        self.assertIn("NVML", result["reason"])

    def test_gpu_telemetry_uses_only_samples_for_requested_pid(self):
        class FakeProcess:
            pid = 77
            usedGpuMemory = 64 * 1024 * 1024

        class FakeSample:
            pid = 77
            timestamp = 123456
            smUtil = 38

        class FakeNvml:
            @staticmethod
            def nvmlInit():
                return None
            @staticmethod
            def nvmlDeviceGetCount():
                return 1
            @staticmethod
            def nvmlDeviceGetHandleByIndex(_index):
                return object()
            @staticmethod
            def nvmlDeviceGetComputeRunningProcesses(_handle):
                return [FakeProcess()]
            @staticmethod
            def nvmlDeviceGetProcessUtilization(_handle, _last_seen):
                return [FakeSample()]

        gpu_info._nvml_initialized = False
        with patch.dict("sys.modules", {"pynvml": FakeNvml()}):
            result = gpu_info.query_gpu_process_metrics(77)
        self.assertTrue(result["available"])
        self.assertEqual(result["utilization_percent"], 38)
        self.assertEqual(result["memory_mb"], 64)
        self.assertEqual(result["last_seen_us"], 123456)
        self.assertTrue(result["utilization_available"])
        self.assertTrue(result["memory_available"])

    def test_one_table_row_updates_status_and_real_progress_in_place(self):
        model = QueueTableModel()
        job = FileJob(path=Path("audio.wav"), size_bytes=10, job_id="job-1")
        state = FileProgress(status="Waiting", stage="Pronto")
        model.replace_rows([job], {job.key: state})
        status = model.index(0, 3)
        progress = model.index(0, 5)
        self.assertEqual(model.rowCount(), 1)
        self.assertEqual(model.data(status), "Pronto")
        state.status = "Processando"
        state.stage = "Transcrevendo"
        state.percent = 42
        model.notify_state(job.key)
        self.assertEqual(model.rowCount(), 1)
        self.assertEqual(model.data(status), "Processando")
        self.assertEqual(model.data(model.index(0, 4)), "Transcrevendo")
        self.assertEqual(model.data(progress), "42%")


if __name__ == "__main__":
    unittest.main()
