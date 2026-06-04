# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Finelog stats schemas and counter-key constants for Zephyr pipelines.

Two namespaces are written:

- ``zephyr.stage`` — one row per stage at completion, emitted by the
  coordinator. Contains throughput and aggregated resource usage.
- ``zephyr.worker`` — one row per shard at START, each sample interval
  (RUNNING), and END, emitted directly by each runner.

Resource counters (cpu, memory, io) are sampled by runner background threads
via :func:`zephyr.runners._sample_process_stats` and stored with
:meth:`~zephyr.runners._InProcessWorkerContext.set_counter`. Counters are also
sent to the coordinator via heartbeats for aggregation into stage stats.
"""

from __future__ import annotations

import enum
import logging
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import ClassVar

from finelog.client import LogClient, Table
from iris.client import get_iris_ctx
from iris.cluster.endpoints import LOG_SERVER_ENDPOINT_NAME

logger = logging.getLogger(__name__)

ZEPHYR_STAGE_STATS_NAMESPACE = "zephyr.stage"
ZEPHYR_WORKER_STATS_NAMESPACE = "zephyr.worker"

ZEPHYR_STAGE_ITEM_COUNT_KEY = "zephyr/stage/{stage_name}/item_count"
"""Counter key template for items processed by a stage; format with ``stage_name``."""
ZEPHYR_STAGE_BYTES_PROCESSED_KEY = "zephyr/stage/{stage_name}/bytes_processed"
"""Counter key template for bytes processed by a stage; format with ``stage_name``."""

# Counter keys written by runner sampler threads using set_counter().
# Read by the coordinator from completed task snapshots for stage stat aggregation.
ZEPHYR_WORKER_CPU_MILLI_KEY = "zephyr/worker/{stage_name}/cpu_millipct"
"""cpu_percent * 1000, stored as an integer millipercent; format with ``stage_name``."""
ZEPHYR_WORKER_CPU_TIME_MS_KEY = "zephyr/worker/{stage_name}/cpu_time_ms"
"""Cumulative CPU time (user + system) since shard start, in milliseconds; format with ``stage_name``."""
ZEPHYR_WORKER_MEM_CURRENT_KEY = "zephyr/worker/{stage_name}/mem_current_bytes"
"""Current resident-set size of the runner process in bytes; format with ``stage_name``."""
ZEPHYR_WORKER_MEM_PEAK_KEY = "zephyr/worker/{stage_name}/mem_peak_bytes"
"""Monotonically increasing peak RSS seen across all sampling intervals; format with ``stage_name``."""
ZEPHYR_WORKER_IO_READ_KEY = "zephyr/worker/{stage_name}/io_read_bytes"
"""Cumulative bytes read by the runner process (best-effort; 0 if unavailable); format with ``stage_name``."""
ZEPHYR_WORKER_IO_WRITE_KEY = "zephyr/worker/{stage_name}/io_write_bytes"
"""Cumulative bytes written by the runner process (best-effort; 0 if unavailable); format with ``stage_name``."""


class ZephyrWorkerStatStatus(enum.StrEnum):
    """Lifecycle status of a ZephyrWorkerStat or ZephyrStageStat row."""

    START = "START"
    RUNNING = "RUNNING"
    END = "END"
    FAILED = "FAILED"


@dataclass
class ZephyrStageStat:
    """One row per stage at completion (or failure), written by the coordinator."""

    key_column: ClassVar[str] = "execution_id"

    execution_id: str
    stage_name: str
    status: str  # ZephyrWorkerStatStatus value; str because LogClient cannot serialize StrEnum
    ts: datetime
    elapsed: float  # seconds
    items: int
    bytes_processed: int
    item_rate: float
    byte_rate: float
    total_shards: int
    # Resource usage aggregated across all completed shard tasks for this stage.
    cpu_pct_avg: float
    cpu_time_total: float  # seconds
    mem_bytes_avg: int
    mem_bytes_max: int
    mem_peak_bytes_max: int
    io_read_bytes_total: int
    io_write_bytes_total: int


@dataclass
class ZephyrWorkerStat:
    """One row per shard per sample interval, written by each runner."""

    key_column: ClassVar[str] = "execution_id"

    execution_id: str
    stage_name: str
    shard_idx: int
    status: str  # ZephyrWorkerStatStatus value; str because LogClient cannot serialize StrEnum
    ts: datetime
    items: int
    bytes_processed: int
    item_rate: float
    byte_rate: float
    cpu_time_total: float  # seconds
    cpu_pct_avg: float
    mem_current_bytes: int
    mem_peak_bytes: int
    io_bytes_read_total: int
    io_bytes_written_total: int


class StatsWriter:
    """Manages finelog connections and emits Zephyr stat rows.

    Call ``connect()`` to get a live instance; pass a pre-resolved URL when
    an Iris context is not available (e.g. in a subprocess).  All emit
    methods are no-ops when the log client is unavailable.
    """

    def __init__(self, log_client: LogClient | None) -> None:
        self._log_client = log_client
        self._stage_table: Table | None = None
        self._worker_table: Table | None = None
        if log_client is not None:
            with suppress(Exception):
                self._stage_table = log_client.get_table(ZEPHYR_STAGE_STATS_NAMESPACE, ZephyrStageStat)
            with suppress(Exception):
                self._worker_table = log_client.get_table(ZEPHYR_WORKER_STATS_NAMESPACE, ZephyrWorkerStat)

    @classmethod
    def connect(cls, url: str | None = None) -> StatsWriter:
        """Connect to finelog; resolves the URL via Iris if not provided.

        Returns a no-op instance if the URL cannot be determined or the
        connection fails.
        """
        resolved = url or cls.resolve_url()
        if resolved is None:
            return cls(None)
        try:
            return cls(LogClient.connect(resolved))
        except Exception:
            logger.warning("Could not connect to finelog at %s; stats disabled", resolved, exc_info=True)
            return cls(None)

    @staticmethod
    def resolve_url() -> str | None:
        """Resolve the finelog endpoint URL via the Iris controller registry."""
        iris_ctx = get_iris_ctx()
        if iris_ctx is None or iris_ctx.client is None:
            return None
        try:
            return iris_ctx.client.resolve_endpoint(LOG_SERVER_ENDPOINT_NAME)
        except Exception:
            logger.warning("Could not resolve finelog endpoint", exc_info=True)
            return None

    def emit_stage_stat(
        self,
        stage_name: str,
        execution_id: str,
        elapsed: float,
        total_shards: int,
        completed_counters: list[dict[str, int]],
        inflight_counters: list[dict[str, int]],
        status: ZephyrWorkerStatStatus = ZephyrWorkerStatStatus.END,
    ) -> None:
        """Build and emit a ZephyrStageStat row from raw counter snapshots.

        ``completed_counters`` — one dict per finished shard (used for both
        throughput and resource aggregation).  ``inflight_counters`` — one
        dict per still-running shard (throughput only; resource stats are
        excluded because they haven't reached the END sample yet).
        Pass ``status=ZephyrWorkerStatStatus.FAILED`` when emitting for a failed stage.
        """
        if self._stage_table is None:
            return
        item_key = ZEPHYR_STAGE_ITEM_COUNT_KEY.format(stage_name=stage_name)
        byte_key = ZEPHYR_STAGE_BYTES_PROCESSED_KEY.format(stage_name=stage_name)
        cpu_milli_key = ZEPHYR_WORKER_CPU_MILLI_KEY.format(stage_name=stage_name)
        cpu_time_key = ZEPHYR_WORKER_CPU_TIME_MS_KEY.format(stage_name=stage_name)
        mem_current_key = ZEPHYR_WORKER_MEM_CURRENT_KEY.format(stage_name=stage_name)
        mem_peak_key = ZEPHYR_WORKER_MEM_PEAK_KEY.format(stage_name=stage_name)
        io_read_key = ZEPHYR_WORKER_IO_READ_KEY.format(stage_name=stage_name)
        io_write_key = ZEPHYR_WORKER_IO_WRITE_KEY.format(stage_name=stage_name)

        all_for_throughput = completed_counters + inflight_counters
        total_items = sum(s.get(item_key, 0) for s in all_for_throughput)
        total_bytes = sum(s.get(byte_key, 0) for s in all_for_throughput)
        item_rate = total_items / elapsed if elapsed > 0 else 0.0
        byte_rate = total_bytes / elapsed if elapsed > 0 else 0.0

        sampled = [s for s in completed_counters if cpu_milli_key in s]
        n_sampled = len(sampled)
        avg_cpu_pct = sum(s.get(cpu_milli_key, 0) / 1000.0 for s in sampled) / n_sampled if n_sampled > 0 else 0.0
        total_cpu_s = sum(s.get(cpu_time_key, 0) for s in sampled) / 1000.0
        mem_current_values = [s.get(mem_current_key, 0) for s in sampled]
        mem_avg_bytes = int(sum(mem_current_values) / n_sampled) if n_sampled > 0 else 0
        mem_max_bytes = max(mem_current_values, default=0)
        mem_peak_max = max((s.get(mem_peak_key, 0) for s in sampled), default=0)
        io_read_sum = sum(s.get(io_read_key, 0) for s in sampled)
        io_write_sum = sum(s.get(io_write_key, 0) for s in sampled)

        stat = ZephyrStageStat(
            execution_id=execution_id,
            stage_name=stage_name,
            status=status,
            ts=datetime.now(timezone.utc).replace(tzinfo=None),
            elapsed=elapsed,
            items=total_items,
            bytes_processed=total_bytes,
            item_rate=item_rate,
            byte_rate=byte_rate,
            total_shards=total_shards,
            cpu_pct_avg=avg_cpu_pct,
            cpu_time_total=total_cpu_s,
            mem_bytes_avg=mem_avg_bytes,
            mem_bytes_max=mem_max_bytes,
            mem_peak_bytes_max=mem_peak_max,
            io_read_bytes_total=io_read_sum,
            io_write_bytes_total=io_write_sum,
        )
        try:
            self._stage_table.write([stat])
        except Exception:
            logger.warning("Failed to write stage stat to finelog", exc_info=True)

    def emit_worker_stat(
        self,
        stage_name: str,
        shard_idx: int,
        execution_id: str,
        status: ZephyrWorkerStatStatus,
        start_time: float,
        counters: dict[str, int],
    ) -> None:
        """Build and emit a ZephyrWorkerStat row from the runner's counter dict."""
        if self._worker_table is None:
            return
        elapsed = time.monotonic() - start_time
        item_key = ZEPHYR_STAGE_ITEM_COUNT_KEY.format(stage_name=stage_name)
        byte_key = ZEPHYR_STAGE_BYTES_PROCESSED_KEY.format(stage_name=stage_name)
        items = counters.get(item_key, 0)
        bytes_processed = counters.get(byte_key, 0)
        item_rate = items / elapsed if elapsed > 0 else 0.0
        byte_rate = bytes_processed / elapsed if elapsed > 0 else 0.0
        cpu_time_total = counters.get(ZEPHYR_WORKER_CPU_TIME_MS_KEY.format(stage_name=stage_name), 0) / 1000.0
        cpu_pct_avg = (cpu_time_total / elapsed * 100) if elapsed > 0 else 0.0
        stat = ZephyrWorkerStat(
            execution_id=execution_id,
            stage_name=stage_name,
            shard_idx=shard_idx,
            status=status,
            ts=datetime.now(timezone.utc).replace(tzinfo=None),
            items=items,
            bytes_processed=bytes_processed,
            item_rate=item_rate,
            byte_rate=byte_rate,
            cpu_time_total=cpu_time_total,
            cpu_pct_avg=cpu_pct_avg,
            mem_current_bytes=counters.get(ZEPHYR_WORKER_MEM_CURRENT_KEY.format(stage_name=stage_name), 0),
            mem_peak_bytes=counters.get(ZEPHYR_WORKER_MEM_PEAK_KEY.format(stage_name=stage_name), 0),
            io_bytes_read_total=counters.get(ZEPHYR_WORKER_IO_READ_KEY.format(stage_name=stage_name), 0),
            io_bytes_written_total=counters.get(ZEPHYR_WORKER_IO_WRITE_KEY.format(stage_name=stage_name), 0),
        )
        try:
            self._worker_table.write([stat])
        except Exception:
            logger.warning("Failed to write worker stat to finelog", exc_info=True)

    def close(self) -> None:
        if self._log_client is not None:
            with suppress(Exception):
                self._log_client.close()
