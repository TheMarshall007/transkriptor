import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PySide6.QtCore import QMimeData, QPointF, QUrl, Qt
from PySide6.QtGui import QDropEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog

from gui import DropZone, MainWindow, SourcePickerDialog
from models import FileJob, FileProgress


class GuiInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls.app = QApplication.instance() or QApplication([])

    def test_drop_zone_click_and_drop_emit_paths(self):
        zone = DropZone()
        chosen = []
        dropped = []
        zone.choose_requested.connect(lambda: chosen.append(True))
        zone.paths_dropped.connect(dropped.extend)
        zone.show()
        self.app.processEvents()

        QTest.mouseClick(zone, Qt.MouseButton.LeftButton)
        self.assertEqual(chosen, [True])

        with tempfile.TemporaryDirectory() as temp:
            media = Path(temp) / "sample.wav"
            media.write_bytes(b"audio")
            mime = QMimeData()
            mime.setUrls([QUrl.fromLocalFile(str(media))])
            event = QDropEvent(QPointF(10, 10), Qt.DropAction.CopyAction, mime,
                               Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
            zone.dropEvent(event)
            self.assertEqual([Path(path) for path in dropped], [media])
        zone.close()

    def test_picker_returns_files_and_folders_together(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "media folder"
            folder.mkdir()
            media = root / "single.wav"
            media.write_bytes(b"audio")
            dialog = SourcePickerDialog()
            dialog._set_directory(root)
            QTest.qWait(150)
            selection = dialog.tree.selectionModel()
            for path in (media, folder):
                index = dialog.model.index(str(path))
                self.assertTrue(index.isValid(), str(path))
                selection.select(index, selection.SelectionFlag.Select | selection.SelectionFlag.Rows)

            self.assertEqual({Path(path) for path in dialog.selected_paths()}, {media, folder})
            dialog.close()

    def test_remove_button_tracks_selection_and_sources_are_added(self):
        class IdleGpuMonitor:
            def __init__(self, parent=None):
                from PySide6.QtCore import QObject, Signal

                class MonitorSignal(QObject):
                    info_updated = Signal(dict)

                self._signals = MonitorSignal(parent)
                self.info_updated = self._signals.info_updated

            def start(self):
                return None

            def requestInterruption(self):
                return None

            def wait(self, _timeout):
                return True

        with tempfile.TemporaryDirectory() as temp, \
                patch("gui.GpuMonitor", IdleGpuMonitor), \
                patch("gui.cuda_status", return_value=(False, "CUDA indisponível")):
            media = Path(temp) / "sample.wav"
            media.write_bytes(b"audio")
            folder = Path(temp) / "folder"
            (folder / "nested").mkdir(parents=True)
            (folder / "nested" / "inside.mp3").write_bytes(b"audio")
            window = MainWindow()
            with patch("gui.SourcePickerDialog") as picker_class:
                picker = picker_class.return_value
                picker.exec.return_value = QDialog.DialogCode.Accepted
                picker.selected_paths.return_value = [str(media)]
                QTest.mouseClick(window.drop_zone, Qt.MouseButton.LeftButton)
                picker_class.assert_called_once_with(window)
            self.assertEqual(window.source_list.count(), 1)
            self.assertFalse(window.remove_button.isEnabled())

            window.drop_zone.paths_dropped.emit([str(folder)])
            self.assertEqual(window.source_list.count(), 2)

            missing = Path(temp) / "missing.wav"
            with patch("gui.QMessageBox.warning") as warning:
                window._add_sources([str(missing)])
                warning.assert_called_once()
            self.assertEqual(window.source_list.count(), 2)

            for index in range(window.source_list.count()):
                window.source_list.item(index).setSelected(True)
            self.assertTrue(window.remove_button.isEnabled())
            window.remove_button.click()
            self.assertEqual(window.source_list.count(), 0)
            self.assertFalse(window.remove_button.isEnabled())
            window.close()

    def test_live_status_is_written_inside_each_source_output_folder(self):
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first_dir = root / "one" / "transcricoes"
            second_dir = root / "two" / "transcricoes"
            first_path = root / "one" / "one.wav"
            second_path = root / "two" / "two.wav"
            first = FileJob(path=first_path, size_bytes=1, job_id="one", output_dir=first_dir)
            second = FileJob(path=second_path, size_bytes=1, job_id="two", output_dir=second_dir)
            fake_window = SimpleNamespace(
                jobs=[first, second],
                states={first.key: FileProgress(), second.key: FileProgress()},
                status_write_errors=set(),
                statusBar=lambda: SimpleNamespace(showMessage=lambda *_args: None),
            )

            with patch("gui.write_live_status_tsv") as write_status:
                MainWindow._flush_status(fake_window)

            self.assertEqual({call.args[0] for call in write_status.call_args_list}, {first_dir, second_dir})


if __name__ == "__main__":
    unittest.main()
