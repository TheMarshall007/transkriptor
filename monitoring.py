"""
Resource monitoring and metrics collection.
"""

import logging
import os
import time
from typing import Dict

from models import ResourceStats

try:
    import psutil
except ImportError:
    psutil = None


def resource_monitor_loop(
    stop_event,
    worker_metrics: Dict[int, dict],
    resource_metrics: Dict[str, dict],
    logger: logging.Logger,
    sample_interval: float = 1.0,
) -> None:
    """
    Monitor CPU and memory usage of the main process and workers.
    
    Args:
        stop_event: Threading event to signal loop shutdown.
        worker_metrics: Dictionary tracking metrics for each worker.
        resource_metrics: Dictionary to store collected resource metrics.
        logger: Logger instance.
        sample_interval: Sampling interval in seconds.
    """
    if psutil is None:
        return

    cpu_count = os.cpu_count() or 1
    run_stats = ResourceStats()
    worker_stats: Dict[str, ResourceStats] = {}

    parent = psutil.Process(os.getpid())

    prev_parent_total = None
    prev_parent_ts = None
    prev_worker_totals: Dict[int, float] = {}
    prev_worker_ts: Dict[int, float] = {}

    while not stop_event.is_set():
        loop_started = time.time()

        try:
            # Monitor main process
            try:
                now = time.time()
                times = parent.cpu_times()
                total_cpu_time = float(times.user + times.system)
                rss_mb = parent.memory_info().rss / (1024 * 1024)

                if prev_parent_total is not None and prev_parent_ts is not None:
                    wall_delta = now - prev_parent_ts
                    cpu_delta = total_cpu_time - prev_parent_total
                    cpu_percent = 0.0
                    if wall_delta > 0:
                        cpu_percent = (cpu_delta / wall_delta) * 100.0 / cpu_count
                    run_stats.add_sample(max(0.0, cpu_percent), rss_mb)

                prev_parent_total = total_cpu_time
                prev_parent_ts = now
            except Exception:
                pass

            # Monitor workers
            for pid in list(worker_metrics.keys()):
                pid_str = str(pid)
                try:
                    proc = psutil.Process(pid)

                    now = time.time()
                    times = proc.cpu_times()
                    total_cpu_time = float(times.user + times.system)
                    rss_mb = proc.memory_info().rss / (1024 * 1024)

                    if pid_str not in worker_stats:
                        worker_stats[pid_str] = ResourceStats()

                    if pid in prev_worker_totals and pid in prev_worker_ts:
                        wall_delta = now - prev_worker_ts[pid]
                        cpu_delta = total_cpu_time - prev_worker_totals[pid]
                        cpu_percent = 0.0
                        if wall_delta > 0:
                            cpu_percent = (cpu_delta / wall_delta) * 100.0 / cpu_count
                        worker_stats[pid_str].add_sample(max(0.0, cpu_percent), rss_mb)

                    prev_worker_totals[pid] = total_cpu_time
                    prev_worker_ts[pid] = now

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                except Exception:
                    continue

            resource_metrics["run"] = {
                "cpu_avg": run_stats.cpu_avg,
                "cpu_peak": run_stats.cpu_peak,
                "rss_avg_mb": run_stats.rss_avg_mb,
                "rss_peak_mb": run_stats.rss_peak_mb,
                "samples": run_stats.samples,
            }

            resource_metrics["workers"] = {
                pid_str: {
                    "cpu_avg": stats.cpu_avg,
                    "cpu_peak": stats.cpu_peak,
                    "rss_avg_mb": stats.rss_avg_mb,
                    "rss_peak_mb": stats.rss_peak_mb,
                    "samples": stats.samples,
                }
                for pid_str, stats in worker_stats.items()
            }

        except Exception as exc:
            logger.error(f"[RESOURCE MONITOR ERROR] {exc}")

        elapsed = time.time() - loop_started
        sleep_for = max(0.1, sample_interval - elapsed)
        time.sleep(sleep_for)

