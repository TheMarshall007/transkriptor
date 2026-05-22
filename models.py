from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class FileJob:
    path: Path
    size_bytes: int
    group_name: str = ""
    discovered_index: int = 0


@dataclass
class FileProgress:
    status: str = "Waiting"  # Waiting | Queued | Processing | Done | Error
    percent: int = 0
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    pid: Optional[int] = None
    duration_sec: Optional[float] = None


@dataclass
class ResourceStats:
    samples: int = 0
    cpu_sum: float = 0.0
    cpu_peak: float = 0.0
    rss_sum_mb: float = 0.0
    rss_peak_mb: float = 0.0

    def add_sample(self, cpu_percent: float, rss_mb: float) -> None:
        self.samples += 1
        self.cpu_sum += cpu_percent
        self.cpu_peak = max(self.cpu_peak, cpu_percent)
        self.rss_sum_mb += rss_mb
        self.rss_peak_mb = max(self.rss_peak_mb, rss_mb)

    @property
    def cpu_avg(self) -> float:
        return self.cpu_sum / self.samples if self.samples else 0.0

    @property
    def rss_avg_mb(self) -> float:
        return self.rss_sum_mb / self.samples if self.samples else 0.0


@dataclass
class FolderGroup:
    folder: Path
    files: list[FileJob]