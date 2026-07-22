"""Measured headless GymServer throughput and JSONL payload sizes."""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any


MAX_DECISIONS_PER_EPISODE = 10_000


class BenchmarkClient:
    """Reusable tactical GymServer process with wire-size accounting."""

    def __init__(self, server_cmd: Sequence[str]) -> None:
        self.bytes_sent = 0
        self.bytes_received = 0
        self.request_count = 0
        self.response_count = 0
        self.proc = subprocess.Popen(
            list(server_cmd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        try:
            self._rpc({"cmd": "spaces"})
        except BaseException:
            self.close()
            raise

    def _rpc(self, message: dict[str, Any]) -> dict[str, Any]:
        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("GymServer pipes are unavailable")
        payload = json.dumps(message, separators=(",", ":")) + "\n"
        self.bytes_sent += len(payload.encode("utf-8"))
        self.request_count += 1
        self.proc.stdin.write(payload)
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("GymServer closed unexpectedly")
        self.bytes_received += len(line.encode("utf-8"))
        self.response_count += 1
        response = json.loads(line)
        if not isinstance(response, dict):
            raise RuntimeError("GymServer response must be an object")
        return response

    def run_episode(self, seed: int) -> int:
        state = self._rpc({"cmd": "reset", "seed": seed})
        decisions = 0
        while decisions < MAX_DECISIONS_PER_EPISODE:
            mask = state.get("mask")
            if not isinstance(mask, list):
                raise RuntimeError("GymServer response omitted the action mask")
            try:
                action = next(index for index, legal in enumerate(mask) if legal)
            except StopIteration as error:
                raise RuntimeError("GymServer returned no legal action") from error
            state = self._rpc({"cmd": "step", "action": action})
            decisions += 1
            if bool(state.get("terminated")) or bool(state.get("truncated")):
                return decisions
        raise RuntimeError("benchmark episode exceeded the decision limit")

    def close(self) -> None:
        proc = getattr(self, "proc", None)
        if proc is None:
            return
        try:
            if proc.poll() is None and proc.stdin is not None:
                proc.stdin.write(json.dumps({"cmd": "close"}) + "\n")
                proc.stdin.flush()
        except Exception:
            pass
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)


def benchmark_gymserver(
    *,
    games: int,
    seed_start: int,
    workers: int,
    server_cmd: Sequence[str] | None = None,
    client_factory: Callable[[int], Any] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    cpu_count: Callable[[], int | None] = os.cpu_count,
) -> dict[str, Any]:
    """Benchmark deterministic episodes while reusing one process per worker."""
    if games <= 0:
        raise ValueError("benchmark games must be positive")
    if workers <= 0:
        raise ValueError("benchmark workers must be positive")
    if client_factory is None:
        if not server_cmd:
            raise ValueError("server command is required")
        client_factory = lambda _index: BenchmarkClient(server_cmd)

    def run_partition(worker_index: int) -> dict[str, int]:
        client = client_factory(worker_index)
        try:
            decisions = sum(
                int(client.run_episode(seed))
                for seed in range(seed_start + worker_index, seed_start + games, workers)
            )
            return {
                "decisions": decisions,
                "bytes_sent": int(client.bytes_sent),
                "bytes_received": int(client.bytes_received),
                "request_count": int(client.request_count),
                "response_count": int(client.response_count),
            }
        finally:
            client.close()

    started = clock()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_partition, index) for index in range(workers)]
        partitions = [future.result() for future in futures]
    elapsed = max(0.0, clock() - started)
    decisions = sum(partition["decisions"] for partition in partitions)
    bytes_sent = sum(partition["bytes_sent"] for partition in partitions)
    bytes_received = sum(partition["bytes_received"] for partition in partitions)
    request_count = sum(partition["request_count"] for partition in partitions)
    response_count = sum(partition["response_count"] for partition in partitions)
    return {
        "schema_version": 1,
        "elapsed_seconds": elapsed,
        "reset_count": games,
        "decision_count": decisions,
        "resets_per_second": games / elapsed if elapsed > 0 else 0.0,
        "decisions_per_second": decisions / elapsed if elapsed > 0 else 0.0,
        "cpu_count": int(cpu_count() or 0),
        "worker_count": workers,
        "protocol": {
            "bytes_sent": bytes_sent,
            "bytes_received": bytes_received,
            "total_bytes": bytes_sent + bytes_received,
            "request_count": request_count,
            "response_count": response_count,
            "mean_request_bytes": bytes_sent / request_count if request_count else 0.0,
            "mean_response_bytes": bytes_received / response_count if response_count else 0.0,
        },
    }
