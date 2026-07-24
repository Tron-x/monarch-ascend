#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Measure RDMA startup and transfers before and after pytokio removal.

Each run uses one host and two Monarch procs, with one actor in each proc. The
holder actor provides the memory being accessed, and the driver actor initiates
the transfers. A separate client process coordinates their endpoint calls but
does not move the data. The benchmark checks that all three processes have
different process IDs.

The holder proc backs 33 RDMA-accessible buffers: one 64-byte source containing
known data and 32 independent 64-byte targets. Each is an ordinary bytearray
exposed through an `RDMABuffer` handle. The client receives those handles from
`holder.make_buffers()` and passes the relevant handles to the driver actor.
The handles identify the registered memory; they are not extra buffers, and
the underlying bytearrays stay in the holder proc.

The driver creates separate local scratch memory: one buffer for serial work
or 32 buffers for concurrent work. Operations are named from the driver's
point of view. A read pulls data from the holder's source into driver memory. A
write pushes data from driver memory into a holder target. Concurrent reads
share the holder's source, while concurrent writes use its 32 distinct targets.
The benchmark checks the transferred bytes after each timed operation or batch.

In ibverbs mode, creating the holder's `RDMABuffer` handles registers its
memory with the NIC. The driver's scratch memory is registered when its
`memoryview` is first used. Warmups pay the driver's registration cost, and
steady-state measurements reuse the same memoryviews. Both procs' RDMA managers
use the same requested NIC. In TCP mode, both procs use TCP instead.

Cold samples use fresh procs and time creation of the holder's 33 RDMA buffers
plus the first read, including registration of the driver's first local buffer.
Steady-state samples warm both directions, then measure serial operations and
groups of 32 concurrent operations. This same-host fixture measures the
orchestration affected by the refactor, not network-fabric throughput.
"""

import argparse
import asyncio
import math
import os
import re
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from monarch.actor import Actor, endpoint, ProcMesh, this_host
from monarch.config import configured
from monarch.rdma import is_ibverbs_available, RDMABuffer


_PAYLOAD_BYTES = 64
_CONCURRENCY = 32
_COLD_SAMPLES = 5
_SERIAL_WARMUPS = 20
_SERIAL_SAMPLES = 400
_CONCURRENT_WARMUPS = 1
_CONCURRENT_BATCHES = 50
_OP_TIMEOUT_S = 60
_COLD_CHILD_TIMEOUT_S = 180
_TEARDOWN_TIMEOUT_S = 30
_COLD_PREFIX = "RDMA_COLD_NS "


def _source_pattern(size: int) -> bytes:
    return bytes(i & 0xFF for i in range(size))


def _write_payload(size: int, counter: int) -> bytes:
    body = bytes(0xC0 ^ (i & 0xFF) for i in range(max(size - 8, 0)))
    return (counter.to_bytes(8, "little") + body)[:size]


def _check_read(expected: bytes, observed: bytes) -> None:
    if observed != expected:
        raise AssertionError("read destination does not match the source")


def _check_write(expected: bytes, observed: bytes) -> None:
    if observed != expected:
        raise AssertionError("write target does not match the payload")


class _Holder(Actor):
    def __init__(self) -> None:
        self._source = bytearray(_source_pattern(_PAYLOAD_BYTES))
        self._targets = [bytearray(_PAYLOAD_BYTES) for _ in range(_CONCURRENCY)]

    @endpoint
    async def make_buffers(self) -> tuple[RDMABuffer, list[RDMABuffer]]:
        source = RDMABuffer(memoryview(self._source))
        targets = [RDMABuffer(memoryview(target)) for target in self._targets]
        return source, targets

    @endpoint
    async def source_bytes(self) -> bytes:
        return bytes(self._source)

    @endpoint
    async def target_bytes(self, index: int) -> bytes:
        return bytes(self._targets[index])

    @endpoint
    async def pid(self) -> int:
        return os.getpid()


class _Driver(Actor):
    """Runs transfers while retaining memoryviews across steady-state samples.

    Local-memory registration is cached by memoryview identity, so recreating a
    view inside a timed loop would measure registration instead of steady state.
    """

    @endpoint
    async def pid(self) -> int:
        return os.getpid()

    @endpoint
    async def initialize(
        self,
        holder: _Holder,
        source: RDMABuffer,
        target: RDMABuffer,
    ) -> None:
        expected = bytes(await holder.source_bytes.call_one())
        destination = bytearray(b"\xa5" * source.size())
        await source.read_into(memoryview(destination), timeout=_OP_TIMEOUT_S)
        _check_read(expected, bytes(destination))

        payload = _write_payload(target.size(), 0)
        await target.write_from(memoryview(payload), timeout=_OP_TIMEOUT_S)
        observed = bytes(await holder.target_bytes.call_one(0))
        _check_write(payload, observed)

    @endpoint
    async def cold_read(self, source: RDMABuffer) -> bytes:
        destination = bytearray(b"\xa5" * source.size())
        await source.read_into(memoryview(destination), timeout=_OP_TIMEOUT_S)
        return bytes(destination)

    @endpoint
    async def serial_reads(self, holder: _Holder, source: RDMABuffer) -> list[int]:
        expected = bytes(await holder.source_bytes.call_one())
        poison = b"\xa5" * source.size()
        destination = bytearray(source.size())
        destination_view = memoryview(destination)
        samples: list[int] = []
        for index in range(_SERIAL_WARMUPS + _SERIAL_SAMPLES):
            destination[:] = poison
            start = time.perf_counter_ns()
            await source.read_into(destination_view, timeout=_OP_TIMEOUT_S)
            elapsed = time.perf_counter_ns() - start
            _check_read(expected, bytes(destination))
            if index >= _SERIAL_WARMUPS:
                samples.append(elapsed)
        return samples

    @endpoint
    async def serial_writes(
        self,
        holder: _Holder,
        target: RDMABuffer,
    ) -> list[int]:
        payload = bytearray(target.size())
        payload_view = memoryview(payload)
        samples: list[int] = []
        for index in range(_SERIAL_WARMUPS + _SERIAL_SAMPLES):
            payload[:] = _write_payload(target.size(), index + 1)
            start = time.perf_counter_ns()
            await target.write_from(payload_view, timeout=_OP_TIMEOUT_S)
            elapsed = time.perf_counter_ns() - start
            observed = bytes(await holder.target_bytes.call_one(0))
            _check_write(bytes(payload), observed)
            if index >= _SERIAL_WARMUPS:
                samples.append(elapsed)
        return samples

    @endpoint
    async def concurrent_reads(
        self,
        holder: _Holder,
        source: RDMABuffer,
    ) -> list[int]:
        expected = bytes(await holder.source_bytes.call_one())
        poison = b"\xa5" * source.size()
        destinations = [bytearray(source.size()) for _ in range(_CONCURRENCY)]
        destination_views = [memoryview(destination) for destination in destinations]
        samples: list[int] = []
        for index in range(_CONCURRENT_WARMUPS + _CONCURRENT_BATCHES):
            for destination in destinations:
                destination[:] = poison
            start = time.perf_counter_ns()
            pending = [
                source.read_into(destination, timeout=_OP_TIMEOUT_S).as_asyncio()
                for destination in destination_views
            ]
            for operation in pending:
                await operation
            elapsed = time.perf_counter_ns() - start
            for destination in destinations:
                _check_read(expected, bytes(destination))
            if index >= _CONCURRENT_WARMUPS:
                samples.append(elapsed)
        return samples

    @endpoint
    async def concurrent_writes(
        self,
        holder: _Holder,
        targets: list[RDMABuffer],
    ) -> list[int]:
        payloads = [bytearray(target.size()) for target in targets]
        payload_views = [memoryview(payload) for payload in payloads]
        samples: list[int] = []
        for index in range(_CONCURRENT_WARMUPS + _CONCURRENT_BATCHES):
            for slot, (target, payload) in enumerate(zip(targets, payloads)):
                payload[:] = _write_payload(
                    target.size(), index * _CONCURRENCY + slot + 1
                )
            start = time.perf_counter_ns()
            pending = [
                target.write_from(payload, timeout=_OP_TIMEOUT_S).as_asyncio()
                for target, payload in zip(targets, payload_views)
            ]
            for operation in pending:
                await operation
            elapsed = time.perf_counter_ns() - start
            for slot, payload in enumerate(payloads):
                observed = bytes(await holder.target_bytes.call_one(slot))
                _check_write(bytes(payload), observed)
            if index >= _CONCURRENT_WARMUPS:
                samples.append(elapsed)
        return samples


@dataclass(frozen=True)
class _Results:
    cold_init: list[int]
    serial_read: list[int]
    serial_write: list[int]
    concurrent_read: list[int]
    concurrent_write: list[int]


async def _assert_distinct(holder: _Holder, driver: _Driver) -> None:
    holder_pid = int(await holder.pid.call_one())
    driver_pid = int(await driver.pid.call_one())
    if len({os.getpid(), holder_pid, driver_pid}) != 3:
        raise RuntimeError("client, holder, and driver must be separate processes")


async def _teardown(holder_procs: ProcMesh, driver_procs: ProcMesh) -> None:
    errors: list[Exception] = []
    for procs in (holder_procs, driver_procs):
        try:
            await asyncio.wait_for(procs.stop(), timeout=_TEARDOWN_TIMEOUT_S)
        except Exception as error:
            errors.append(error)
    if errors:
        raise errors[0]


async def _steady_body() -> tuple[list[int], list[int], list[int], list[int]]:
    holder_procs = this_host().spawn_procs(per_host={"processes": 1})
    driver_procs = this_host().spawn_procs(per_host={"processes": 1})
    holder = holder_procs.spawn("rdma_bench_holder", _Holder)
    driver = driver_procs.spawn("rdma_bench_driver", _Driver)
    try:
        await _assert_distinct(holder, driver)
        source, targets = await holder.make_buffers.call_one()
        await driver.initialize.call_one(holder, source, targets[0])
        serial_read = list(await driver.serial_reads.call_one(holder, source))
        serial_write = list(await driver.serial_writes.call_one(holder, targets[0]))
        concurrent_read = list(await driver.concurrent_reads.call_one(holder, source))
        concurrent_write = list(
            await driver.concurrent_writes.call_one(holder, targets)
        )
        return serial_read, serial_write, concurrent_read, concurrent_write
    finally:
        await _teardown(holder_procs, driver_procs)


async def _cold_body() -> int:
    holder_procs = this_host().spawn_procs(per_host={"processes": 1})
    driver_procs = this_host().spawn_procs(per_host={"processes": 1})
    holder = holder_procs.spawn("rdma_cold_holder", _Holder)
    driver = driver_procs.spawn("rdma_cold_driver", _Driver)
    try:
        await _assert_distinct(holder, driver)
        expected = bytes(await holder.source_bytes.call_one())
        start = time.perf_counter_ns()
        source, _targets = await holder.make_buffers.call_one()
        observed = bytes(await driver.cold_read.call_one(source))
        elapsed = time.perf_counter_ns() - start
        _check_read(expected, observed)
        return elapsed
    finally:
        await _teardown(holder_procs, driver_procs)


def _require_effective_config(
    config: Mapping[str, object], backend: str, target: str | None
) -> None:
    expected: dict[str, object] = {
        "rdma_allow_tcp_fallback": backend == "tcp",
        "rdma_disable_ibverbs": backend == "tcp",
        "rdma_ibverbs_target": "" if target is None else target,
    }
    mismatches = [
        f"{key}={config.get(key)!r} (expected {value!r})"
        for key, value in expected.items()
        if config.get(key) != value
    ]
    if mismatches:
        raise RuntimeError(
            f"effective configuration does not select {backend}: "
            f"{', '.join(mismatches)}; unset conflicting MONARCH_RDMA_* settings"
        )


async def _configured_steady(
    backend: str, target: str | None
) -> tuple[list[int], list[int], list[int], list[int]]:
    if backend == "tcp":
        with configured(
            rdma_allow_tcp_fallback=True,
            rdma_disable_ibverbs=True,
            rdma_ibverbs_target="",
        ) as config:
            _require_effective_config(config, backend, target)
            return await _steady_body()
    assert target is not None
    with configured(
        rdma_allow_tcp_fallback=False,
        rdma_disable_ibverbs=False,
        rdma_ibverbs_target=target,
    ) as config:
        _require_effective_config(config, backend, target)
        return await _steady_body()


async def _configured_cold(backend: str, target: str | None) -> int:
    if backend == "tcp":
        with configured(
            rdma_allow_tcp_fallback=True,
            rdma_disable_ibverbs=True,
            rdma_ibverbs_target="",
        ) as config:
            _require_effective_config(config, backend, target)
            return await _cold_body()
    assert target is not None
    with configured(
        rdma_allow_tcp_fallback=False,
        rdma_disable_ibverbs=False,
        rdma_ibverbs_target=target,
    ) as config:
        _require_effective_config(config, backend, target)
        return await _cold_body()


def _cold_command(backend: str, target: str | None) -> list[str]:
    command = [sys.argv[0], "--backend", backend, "--_cold-child"]
    if target is not None:
        command.extend(["--target", target])
    return command


def _collect_cold(backend: str, target: str | None) -> list[int]:
    samples: list[int] = []
    for _ in range(_COLD_SAMPLES):
        completed = subprocess.run(
            _cold_command(backend, target),
            capture_output=True,
            check=False,
            text=True,
            timeout=_COLD_CHILD_TIMEOUT_S,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip()[-1000:]
            raise RuntimeError(
                f"cold-init child failed with exit {completed.returncode}: {detail}"
            )
        sample = next(
            (
                int(line.removeprefix(_COLD_PREFIX))
                for line in completed.stdout.splitlines()
                if line.startswith(_COLD_PREFIX)
            ),
            None,
        )
        if sample is None:
            raise RuntimeError("cold-init child produced no timing")
        samples.append(sample)
    return samples


def _percentile(samples: Sequence[int], percentile: int) -> int:
    if not samples:
        raise ValueError("cannot summarize an empty sample set")
    ordered = sorted(samples)
    rank = math.ceil((percentile / 100.0) * len(ordered))
    return ordered[max(rank, 1) - 1]


def _microseconds(value_ns: int) -> str:
    return f"{value_ns / 1_000:,.2f} us"


def _print_results(backend: str, target: str | None, results: _Results) -> None:
    print(f"backend: {backend}")
    if target is not None:
        print(f"target: {target}")
    print(
        f"shape: {_PAYLOAD_BYTES} bytes, K={_CONCURRENCY}, "
        f"serial={_SERIAL_SAMPLES}, concurrent={_CONCURRENT_BATCHES}, "
        f"cold={_COLD_SAMPLES}"
    )
    cold = results.cold_init
    print(
        "cold_init: "
        f"min={_microseconds(min(cold))} "
        f"p50={_microseconds(_percentile(cold, 50))} "
        f"max={_microseconds(max(cold))}"
    )
    for name, samples in (
        ("serial_read", results.serial_read),
        ("serial_write", results.serial_write),
    ):
        print(
            f"{name}: "
            f"p50={_microseconds(_percentile(samples, 50))} "
            f"p90={_microseconds(_percentile(samples, 90))} "
            f"p99={_microseconds(_percentile(samples, 99))}"
        )
    for name, samples in (
        ("concurrent_read_makespan", results.concurrent_read),
        ("concurrent_write_makespan", results.concurrent_write),
    ):
        print(
            f"{name}: "
            f"p50={_microseconds(_percentile(samples, 50))} "
            f"p90={_microseconds(_percentile(samples, 90))}"
        )


def _validate_target(
    parser: argparse.ArgumentParser, backend: str, target: str | None
) -> None:
    if backend == "tcp":
        if target is not None:
            parser.error("--target is only valid with --backend ibverbs")
        return
    if target is None:
        parser.error("--backend ibverbs requires --target nic:<name>")
    if re.fullmatch(r"nic:[A-Za-z0-9_.-]+", target) is None:
        parser.error("--target must be an exact NIC target such as nic:mlx5_0")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("tcp", "ibverbs"), required=True)
    parser.add_argument(
        "--target", help="ibverbs device target, for example nic:mlx5_0"
    )
    parser.add_argument("--_cold-child", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    backend = str(args.backend)
    target = str(args.target) if args.target is not None else None
    _validate_target(parser, backend, target)

    if bool(args._cold_child):
        sample = asyncio.run(_configured_cold(backend, target))
        print(f"{_COLD_PREFIX}{sample}")
        return 0

    if backend == "ibverbs" and not is_ibverbs_available():
        parser.error("ibverbs is not available on this host")

    cold = _collect_cold(backend, target)
    serial_read, serial_write, concurrent_read, concurrent_write = asyncio.run(
        _configured_steady(backend, target)
    )
    _print_results(
        backend,
        target,
        _Results(
            cold,
            serial_read,
            serial_write,
            concurrent_read,
            concurrent_write,
        ),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
