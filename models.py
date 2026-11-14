from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class FileJob:
    path: Path
    size_bytes: int
    group_name: str = ""
    discovered_index: int = 0
    job_id: str = ""
    input_root: Optional[Path] = None
    output_dir: Optional[Path] = None
    output_stem: Optional[str] = None
    display_path: str = ""
    media_duration_sec: Optional[float] = None

    @property
    def key(self) -> str:
        return self.job_id or str(self.path.resolve())


@dataclass
class MetricSeries:
    current: Optional[float] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    total: float = 0.0
    samples: int = 0

    def add(self, value: Optional[float]) -> None:
        if value is None:
            return
        value = float(value)
        self.current = value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)
        self.total += value
        self.samples += 1

    @property
    def average(self) -> Optional[float]:
        return self.total / self.samples if self.samples else None


@dataclass
class FileResourceMetrics:
    cpu_percent: MetricSeries = field(default_factory=MetricSeries)
    ram_mb: MetricSeries = field(default_factory=MetricSeries)
    gpu_utilization_percent: MetricSeries = field(default_factory=MetricSeries)
    gpu_memory_mb: MetricSeries = field(default_factory=MetricSeries)
    gpu_available: Optional[bool] = None
    gpu_utilization_available: Optional[bool] = None
    gpu_memory_available: Optional[bool] = None
    gpu_unavailable_reason: str = ""


@dataclass
class FileProgress:
    status: str = "Waiting"  # Waiting | Queued | Processing | Done | Error
    stage: str = "Na fila"
    percent: Optional[int] = None
    current_frames: Optional[int] = None
    total_frames: Optional[int] = None
    eta_sec: Optional[float] = None
    started_at: Optional[float] = None
    stage_started_at: Optional[float] = None
    progress_started_at: Optional[float] = None
    progress_start_frames: int = 0
    finished_at: Optional[float] = None
    error: Optional[str] = None
    pid: Optional[int] = None
    duration_sec: Optional[float] = None
    model_name: str = ""
    device: str = ""
    model_load_sec: Optional[float] = None
    processing_duration_sec: Optional[float] = None
    media_duration_sec: Optional[float] = None
    realtime_factor: Optional[float] = None
    cpu_avg: Optional[float] = None
    cpu_peak: Optional[float] = None
    ram_avg_mb: Optional[float] = None
    ram_peak_mb: Optional[float] = None
    gpu_utilization_percent: Optional[float] = None
    gpu_memory_used_mb: Optional[int] = None
    gpu_memory_total_mb: Optional[int] = None
    resource_metrics: FileResourceMetrics = field(default_factory=FileResourceMetrics)


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
