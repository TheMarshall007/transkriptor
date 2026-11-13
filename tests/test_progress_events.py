import logging
import queue
import threading
import time
import unittest

from event_handlers import queue_listener_loop
from models import FileProgress


class ProgressEventTests(unittest.TestCase):
    def _run_events(self, events, state):
        event_queue = queue.Queue()
        worker_metrics = {}
        stop = threading.Event()
        listener = threading.Thread(
            target=queue_listener_loop,
            args=(event_queue, state, worker_metrics, logging.getLogger("test-progress"), stop),
        )
        listener.start()
        for event in events:
            event_queue.put(event)
        deadline = time.monotonic() + 2
        final_key = events[-1].get("job_id")
        final_status = "Done" if events[-1]["type"] == "finished" else "Error"
        while state[final_key].status != final_status and time.monotonic() < deadline:
            time.sleep(0.01)
        stop.set()
        listener.join(timeout=2)
        self.assertFalse(listener.is_alive())
        return worker_metrics

    def test_model_device_and_actual_frames_reach_completed_state(self):
        state = {"job-1": FileProgress(status="Queued", media_duration_sec=60)}
        events = [
            {"type": "started", "job_id": "job-1", "file_name": "episode.mp4", "pid": 44,
             "model_name": "medium", "device": "cuda", "timestamp": 10.0},
            {"type": "transcription_progress", "job_id": "job-1", "current_frames": 10,
             "total_frames": 100, "timestamp": 11.0},
            {"type": "transcription_progress", "job_id": "job-1", "current_frames": 40,
             "total_frames": 100, "timestamp": 14.0},
            {"type": "finished", "job_id": "job-1", "file_name": "episode.mp4", "pid": 44,
             "model_name": "medium", "device": "cuda", "model_load_sec": 2.5,
             "processing_duration_sec": 8.0, "duration_sec": 10.5, "timestamp": 20.0},
        ]

        self._run_events(events, state)

        result = state["job-1"]
        self.assertEqual(result.status, "Done")
        self.assertEqual(result.percent, 100)
        self.assertEqual(result.current_frames, 40)
        self.assertEqual(result.total_frames, 100)
        self.assertEqual(result.model_name, "medium")
        self.assertEqual(result.device, "cuda")
        self.assertEqual(result.model_load_sec, 2.5)
        self.assertEqual(result.processing_duration_sec, 8.0)
        self.assertEqual(result.realtime_factor, 7.5)

    def test_error_preserves_last_real_percent_and_frame_count(self):
        state = {"job-2": FileProgress(status="Queued")}
        events = [
            {"type": "started", "job_id": "job-2", "file_name": "broken.wav", "pid": 45,
             "model_name": "medium", "device": "cpu", "timestamp": 10.0},
            {"type": "transcription_progress", "job_id": "job-2", "current_frames": 30,
             "total_frames": 100, "timestamp": 12.0},
            {"type": "error", "job_id": "job-2", "file_name": "broken.wav", "stage": "Transcrevendo",
             "error": "decoder error", "duration_sec": 3.0, "timestamp": 13.0},
        ]

        self._run_events(events, state)

        result = state["job-2"]
        self.assertEqual(result.status, "Error")
        self.assertEqual(result.percent, 30)
        self.assertEqual(result.current_frames, 30)
        self.assertEqual(result.stage, "Falha: Transcrevendo")
        self.assertIsNone(result.eta_sec)


if __name__ == "__main__":
    unittest.main()
