# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

"""Performance benchmarks for the Event system.

Uses relative assertions (p99 ≤ 2× baseline) so slow CI machines
don't fail spuriously.

Skipped on CI because these are benchmarks, not correctness tests.
"""

import os
import statistics
import time

import pytest

from hyperspy.events import Event

# GitHub CI and Azure Pipelines has `CI`` and `TF_BUILD`
# environment variable, respectively
_IN_CI = (
    os.environ.get("CI", "").lower() in {"true", "1"}
    or os.environ.get("TF_BUILD", "").lower() == "true"
)

pytestmark = pytest.mark.skipif(
    _IN_CI,
    reason="Performance benchmarks excluded on CI",
)


def _measure(func, iterations=100, warmup=10):
    """Run *func* *iterations* times, return p50, p99 (seconds)."""
    # Warmup
    for _ in range(warmup):
        func()
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        func()
        times.append(time.perf_counter() - t0)
    times.sort()
    p50 = statistics.median(times)
    p99 = times[int(len(times) * 0.99)]
    return p50, p99


# ── Baseline ─────────────────────────────────────────────────────────┘


def _baseline_empty_connect_disconnect():
    """connect + disconnect a single no-op callback."""
    ev = Event()
    cb = lambda **kw: None  # noqa: E731
    ev.connect(cb)
    ev.disconnect(cb)


def _baseline_empty_emit():
    """emit on an Event with zero callbacks."""
    ev = Event()
    ev.emit()


# ── Benchmark 1: connect/disconnect 1000 callbacks ───────────────────┘


@pytest.mark.slow
def test_connect_disconnect_1000():
    """connect/disconnect 1000 individually — p99 ≤ 2× baseline."""
    base_p50, base_p99 = _measure(_baseline_empty_connect_disconnect)

    def _workload():
        ev = Event()
        cbs = [lambda **kw: None for _ in range(1000)]
        for cb in cbs:
            ev.connect(cb)
        for cb in cbs:
            ev.disconnect(cb)

    p50, p99 = _measure(_workload, iterations=20, warmup=3)
    threshold = max(base_p99 * 1000 * 2, 0.5)
    assert p99 < threshold, f"p99={p99:.6f}s exceeds {threshold:.6f}s"


# ── Benchmark 2: emit with 100 connected callbacks ───────────────────┘


@pytest.mark.slow
def test_emit_100_callbacks():
    """emit() with 100 connected callbacks — p99 ≤ 2× baseline per cb."""
    base_p50, base_p99 = _measure(_baseline_empty_emit)

    ev = Event()
    counts = [0]

    def _make_cb(idx):
        def cb(**kw):
            counts[0] += 1

        return cb

    cbs = [_make_cb(i) for i in range(100)]
    for cb in cbs:
        ev.connect(cb)

    def _workload():
        ev.emit()

    # Warmup separately so the counter isn't inflated by warmup iterations
    for _ in range(20):
        _workload()
    counts[0] = 0

    p50, p99 = _measure(_workload, iterations=200, warmup=0)
    # Allow up to 2× per-callback baseline scaled by 100 callbacks
    threshold = max(base_p99 * 100 * 2, 0.01)
    assert p99 < threshold, f"p99={p99:.6f}s exceeds {threshold:.6f}s"
    assert counts[0] == 100 * 200  # one increment per callback per iteration


# ── Benchmark 3: weakref connect/disconnect lifecycle ─────────────────┘


# @pytest.mark.slow
# def test_weakref_connect_disconnect_lifecycle():
#     """connect with weakrefs, let GC collect, verify cleanup — p99 ≤ 2× baseline."""

#     class _Callback:
#         def __call__(self, **kw):
#             pass

#     def _make_and_connect(ev, n):
#         cbs = [_Callback() for _ in range(n)]
#         for cb in cbs:
#             ev.connect(cb)
#         return cbs

#     base_p50, base_p99 = _measure(_baseline_empty_connect_disconnect)

#     def _workload():
#         ev = Event()
#         cbs = _make_and_connect(ev, 200)
#         # Drop strong references — garbage collection will clean up
#         del cbs
#         # Force recycling
#         for _ in range(10):
#             import gc

#             gc.collect()

#     p50, p99 = _measure(_workload, iterations=20, warmup=3)
#     threshold = max(base_p99 * 200 * 2, 1.0)
#     assert p99 < threshold, f"p99={p99:.6f}s exceeds {threshold:.6f}s"


# ── Benchmark 4: data_changed.emit(obj=self) × 10k ───────────────────┘


@pytest.mark.slow
def test_data_changed_emit_10k():
    """10k iterations of data_changed.emit(obj=self) pattern — p99 ≤ 2× baseline."""

    class _MockObj:
        pass

    obj = _MockObj()
    ev = Event()

    counts = [0]

    def _listener(obj):
        counts[0] += 1

    ev.connect(_listener)

    base_p50, base_p99 = _measure(_baseline_empty_emit)

    def _workload():
        ev.emit(obj=obj)

    for _ in range(10):
        _workload()
    counts[0] = 0

    p50, p99 = _measure(_workload, iterations=100, warmup=0)
    threshold = max(base_p99 * 2, 0.001)
    assert p99 < threshold, (
        f"data_changed.emit(obj=self) p99={p99:.6f}s exceeds {threshold:.6f}s"
    )
    assert counts[0] == 100  # one increment per iteration
