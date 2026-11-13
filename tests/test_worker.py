import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import worker


class EventCollector:
    def __init__(self):
        self.events = []

    def put(self, event):
        self.events.append(event)


class WorkerTests(unittest.TestCase):
    def setUp(self):
        worker._MODELS.clear()
        worker._MODEL_LOAD_SECONDS.clear()

    def tearDown(self):
        worker._MODELS.clear()
        worker._MODEL_LOAD_SECONDS.clear()

    def test_cpu_worker_reports_model_device_and_writes_collision_safe_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            media = root / "meeting.mp3"
            media.write_bytes(b"placeholder")
            model = SimpleNamespace()
            events = EventCollector()

            def fake_transcribe(_model, _path, _language, event_queue, _stage, job_id, device):
                event_queue.put({
                    "type": "transcription_progress", "file_name": media.name,
                    "job_id": job_id, "current_frames": 25, "total_frames": 100,
                    "timestamp": 1.0,
                })
                self.assertEqual(device, "cpu")
                return {"text": "transcrição local", "segments": []}

            with patch.object(worker.whisper, "load_model", return_value=model) as load_model, \
                    patch.object(worker, "transcribe_with_progress", side_effect=fake_transcribe):
                result = worker.transcribe_file_worker(
                    str(media), str(root), str(root / "output"), "medium", "pt", events,
                    "cpu", "unique-job", "meeting__left",
                )

            load_model.assert_called_once_with("medium", device="cpu")
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["model_name"], "medium")
            self.assertEqual(result["device"], "cpu")
            self.assertTrue((root / "output" / "meeting__left.txt").exists())
            self.assertTrue((root / "output" / "meeting__left.srt").exists())
            self.assertTrue((root / "output" / "meeting__left.json").exists())
            self.assertTrue(any(event.get("type") == "transcription_progress"
                                and event.get("job_id") == "unique-job" for event in events.events))


if __name__ == "__main__":
    unittest.main()
