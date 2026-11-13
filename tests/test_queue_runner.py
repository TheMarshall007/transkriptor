import tempfile
import unittest
from pathlib import Path

from models import FileJob, FileProgress
from queue_runner import collect_transcription_jobs
from ui import write_live_status_tsv


class QueueRunnerTests(unittest.TestCase):
    def test_folder_tree_and_duplicate_folder_names_are_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = root / "course-a" / "Lessons"
            second = root / "course-b" / "Lessons"
            (first / "week-1").mkdir(parents=True)
            second.mkdir(parents=True)
            (first / "week-1" / "intro.wav").write_bytes(b"audio")
            (second / "intro.wav").write_bytes(b"audio")

            jobs, warnings = collect_transcription_jobs([first, second], root / "out")

            self.assertEqual(warnings, [])
            self.assertEqual([job.display_path for job in jobs], [
                "Lessons\\week-1\\intro.wav", "Lessons (2)\\intro.wav",
            ])
            self.assertNotEqual(jobs[0].key, jobs[1].key)
            self.assertEqual(jobs[0].input_root, first.resolve())
            self.assertEqual(jobs[0].output_dir.name, "Lessons")
            self.assertEqual(jobs[1].output_dir.name, "Lessons (2)")

    def test_individual_file_stem_collisions_receive_unique_names(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            left = root / "left"
            right = root / "right"
            left.mkdir()
            right.mkdir()
            first = left / "meeting.mp3"
            second = right / "meeting.wav"
            first.write_bytes(b"one")
            second.write_bytes(b"two")

            jobs, warnings = collect_transcription_jobs([first, second], root / "out")

            self.assertEqual(warnings, [])
            self.assertEqual([job.output_stem for job in jobs], ["meeting", "meeting__right"])
            self.assertEqual([job.output_dir for job in jobs], [(root / "out").resolve()] * 2)

    def test_tsv_preserves_legacy_columns_and_adds_performance_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            media = Path(temp) / "sample.mp3"
            media.write_bytes(b"media")
            job = FileJob(path=media, size_bytes=5, job_id="unique", display_path="sample.mp3")
            state = FileProgress(status="Processing", stage="Transcrevendo", percent=35,
                                 model_name="medium", device="cuda", current_frames=35,
                                 total_frames=100, eta_sec=12.5, model_load_sec=2.0,
                                 processing_duration_sec=8.0, media_duration_sec=20.0,
                                 realtime_factor=2.5)
            target = write_live_status_tsv(Path(temp) / "out", [job], {job.key: state})
            rows = target.read_text(encoding="utf-8").splitlines()
            columns = rows[0].split("\t")
            values = rows[1].split("\t")

            self.assertEqual(columns[:15], [
                "group_name", "file_name", "size_bytes", "size_human", "status", "percent",
                "pid", "queue_position", "discovered_index", "elapsed_or_duration", "stage",
                "current_frames", "total_frames", "eta_seconds", "stage_elapsed",
            ])
            self.assertEqual(values[15:17], ["medium", "cuda"])
            self.assertEqual(values[17:21], ["2.000", "8.000", "20.000", "2.500"])


if __name__ == "__main__":
    unittest.main()
