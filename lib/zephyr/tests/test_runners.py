# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pluggable StageRunner strategies (zephyr.runners)."""

from __future__ import annotations

import os
import socket
import threading
import uuid
from contextlib import suppress

import pytest
import uvicorn
from finelog.client import LogClient
from finelog.server.asgi import build_log_server_asgi
from finelog.server.service import LogServiceImpl
from finelog.server.stats_service import StatsServiceImpl
from finelog.store import LogStore
from fray import ResourceConfig
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext, ZephyrWorkerError
from zephyr.runners import InlineRunner, SubprocessRunner
from zephyr.stats import (
    ZEPHYR_STAGE_STATS_NAMESPACE,
    ZEPHYR_WORKER_STATS_NAMESPACE,
    StatsWriter,
)


def _ctx(local_client, tmp_path, *, stage_runner_factory) -> ZephyrContext:
    return ZephyrContext(
        client=local_client,
        max_workers=2,
        resources=ResourceConfig(cpu=1, ram="512m"),
        chunk_storage_prefix=str(tmp_path / "chunks"),
        name=f"test-runner-{uuid.uuid4().hex[:8]}",
        stage_runner_factory=stage_runner_factory,
    )


@pytest.fixture(
    params=[
        pytest.param(lambda n: InlineRunner(num_workers=n), id="inline"),
        pytest.param(lambda n: SubprocessRunner(num_workers=n), id="subprocess"),
    ]
)
def runner_factory(request):
    """Run each test against both shipped runners."""
    return request.param


def test_simple_map(local_client, tmp_path, runner_factory):
    """Both runners produce correct results for a basic map pipeline."""
    ctx = _ctx(local_client, tmp_path, stage_runner_factory=runner_factory)
    try:
        ds = Dataset.from_list([1, 2, 3, 4, 5]).map(lambda x: x * 3)
        results = ctx.execute(ds).results
    finally:
        ctx.shutdown()
    assert sorted(results) == [3, 6, 9, 12, 15]


def test_user_counters_propagate(local_client, tmp_path, runner_factory):
    """User counters flow back from the worker (or its subprocess child) to the coordinator."""

    def increment(x: int) -> int:
        counters.increment("docs", 1)
        counters.increment("doubled_sum", x * 2)
        return x

    ctx = _ctx(local_client, tmp_path, stage_runner_factory=runner_factory)
    try:
        ds = Dataset.from_list([1, 2, 3, 4, 5]).map(increment)
        outcome = ctx.execute(ds)
    finally:
        ctx.shutdown()
    assert sorted(outcome.results) == [1, 2, 3, 4, 5]
    assert outcome.counters.get("docs") == 5
    assert outcome.counters.get("doubled_sum") == 30


def test_exception_preserves_user_frame(local_client, tmp_path, runner_factory):
    """A Python exception in user code surfaces with the user's frame visible.

    Inline path raises directly; subprocess path attaches a ``__notes__``
    breadcrumb so the user frame survives cloudpickling.
    """

    def buggy(_: int) -> int:
        empty: tuple = ()
        return empty[0]

    ctx = _ctx(local_client, tmp_path, stage_runner_factory=runner_factory)
    try:
        ds = Dataset.from_list([0]).map(buggy)
        with pytest.raises(ZephyrWorkerError) as exc_info:
            ctx.execute(ds)
    finally:
        ctx.shutdown()

    chained = ""
    cur: BaseException | None = exc_info.value
    while cur is not None:
        chained += str(cur) + "".join(getattr(cur, "__notes__", []))
        cur = cur.__cause__ or cur.__context__
    assert "buggy" in chained or "tuple index out of range" in chained, chained


@pytest.mark.parametrize(
    "runner_factory_fn",
    [
        pytest.param(lambda n: InlineRunner(num_workers=n), id="inline"),
        pytest.param(
            lambda n: SubprocessRunner(num_workers=n),
            id="subprocess",
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="subprocess gives each shard a unique PID; strict=True catches silent fallback to inline.",
            ),
        ),
    ],
)
def test_runner_parametrization_isolates_processes(local_client, tmp_path, runner_factory_fn):
    """Regression guard that subprocess parametrization actually spawns subprocesses.

    Inline reuses the worker actor (≤ max_workers PIDs); subprocess gets one PID per shard.
    """

    def record_pid(x: int) -> int:
        counters.increment(f"shard_pid_{os.getpid()}", 1)
        return x

    ctx = _ctx(local_client, tmp_path, stage_runner_factory=runner_factory_fn)
    try:
        ds = Dataset.from_list(list(range(5))).map(record_pid)
        outcome = ctx.execute(ds)
    finally:
        ctx.shutdown()

    pid_counters = {k: v for k, v in outcome.counters.items() if k.startswith("shard_pid_")}
    assert sum(pid_counters.values()) == 5, pid_counters
    assert len(pid_counters) <= 2, pid_counters


def test_subprocess_runner_isolates_native_crash(local_client, tmp_path):
    """A native abort in one shard surfaces as a deterministic TASK error.

    Forcibly terminate the child via ``os._exit(139)`` (the exit code SIGSEGV
    would produce); the inline runner has no way to recover from that, but
    the subprocess runner sees ``returncode != 0`` and routes to
    ``report_error`` so the pipeline aborts cleanly after MAX_SHARD_FAILURES.
    """

    def crash(_: int) -> int:
        os._exit(139)

    ctx = _ctx(local_client, tmp_path, stage_runner_factory=lambda n: SubprocessRunner(num_workers=n))
    try:
        ds = Dataset.from_list([0]).map(crash)
        with pytest.raises(ZephyrWorkerError) as exc_info:
            ctx.execute(ds)
    finally:
        ctx.shutdown()

    rendered = str(exc_info.value)
    assert "Shard 0" in rendered
    assert "exited with code 139" in rendered or "failed" in rendered


@pytest.fixture()
def finelog_server(tmp_path):
    """Start a real finelog server on a free port and yield its URL."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    store = LogStore(log_dir=tmp_path / "finelog")
    service = LogServiceImpl(log_store=store)
    stats_service = StatsServiceImpl(log_store=store)
    app = build_log_server_asgi(service, stats_service=stats_service)

    started_event = threading.Event()

    class _Server(uvicorn.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets=sockets)
            started_event.set()

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", log_config=None)
    server = _Server(config)
    t = threading.Thread(target=server.run, daemon=True, name="finelog-test-server")
    t.start()

    if not started_event.wait(timeout=5.0):
        raise RuntimeError("finelog test server did not start in time")

    yield f"http://127.0.0.1:{port}"

    server.should_exit = True
    t.join(timeout=5.0)
    store.close()


def test_finelog_stats_emitted(local_client, tmp_path, finelog_server, monkeypatch):
    """Pipeline emits rows to both zephyr.stage and zephyr.worker finelog tables."""
    writers: list[StatsWriter] = []

    def make_writer(url: str | None = None) -> StatsWriter:
        w = StatsWriter(LogClient.connect(finelog_server))
        writers.append(w)
        return w

    monkeypatch.setattr(StatsWriter, "connect", staticmethod(make_writer))

    ctx = _ctx(local_client, tmp_path, stage_runner_factory=lambda n: InlineRunner(num_workers=n))
    try:
        ds = Dataset.from_list(list(range(10))).map(lambda x: x)
        ctx.execute(ds)
    finally:
        ctx.shutdown()

    # ctx.shutdown() closes the coordinator's writer; close any runner writers too.
    for w in writers:
        with suppress(Exception):
            w.close()

    query_client = LogClient.connect(finelog_server)
    try:
        stage_rows = query_client.query(f'SELECT * FROM "{ZEPHYR_STAGE_STATS_NAMESPACE}"')
        worker_rows = query_client.query(f'SELECT * FROM "{ZEPHYR_WORKER_STATS_NAMESPACE}"')
    finally:
        query_client.close()

    assert stage_rows.num_rows >= 1, "Expected stage stat rows, got none"
    assert worker_rows.num_rows >= 1, "Expected worker stat rows, got none"

    stage_names = stage_rows.column("stage_name").to_pylist()
    assert any("map" in s.lower() for s in stage_names), f"No map stage in {stage_names}"

    # Stage stat correctness: items processed, throughput > 0, status = END.
    total_items = sum(stage_rows.column("items").to_pylist())
    assert total_items >= 10, f"Expected >= 10 items across stage rows, got {total_items}"
    elapsed_values = stage_rows.column("elapsed").to_pylist()
    assert all(e >= 0 for e in elapsed_values), f"Negative elapsed in stage rows: {elapsed_values}"
    item_rates = stage_rows.column("item_rate").to_pylist()
    assert all(r >= 0 for r in item_rates), f"Negative item_rate in stage rows: {item_rates}"
    statuses = stage_rows.column("status").to_pylist()
    assert all(s == "END" for s in statuses), f"Unexpected stage statuses: {statuses}"

    # Worker stat correctness: at least one START and one END row per shard.
    worker_statuses = worker_rows.column("status").to_pylist()
    assert "START" in worker_statuses, f"No START worker rows: {worker_statuses}"
    assert "END" in worker_statuses, f"No END worker rows: {worker_statuses}"
