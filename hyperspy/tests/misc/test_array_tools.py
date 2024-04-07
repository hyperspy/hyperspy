# -*- coding: utf-8 -*-
# Copyright 2007-2024 The HyperSpy developers
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


import dask.array as da
import numpy as np
import pytest

from hyperspy.misc.array_tools import (
    get_array_memory_size_in_GiB,
    get_signal_chunk_slice,
    get_value_at_index,
    numba_histogram,
    round_half_away_from_zero,
    round_half_towards_zero,
)

dt = [("x", np.uint8), ("y", np.uint16), ("text", (bytes, 6))]


@pytest.mark.parametrize(
    "dtype, size",
    [
        ("int32", 4.470348e-7),
        ("float64", 8.940697e-7),
        ("uint8", 1.117587e-7),
        (np.dtype(np.int16), 2.235174e-7),
    ],
)
def test_get_memory_size(dtype, size):
    mem = get_array_memory_size_in_GiB((2, 3, 4, 5), dtype=dtype)
    print(mem)
    np.testing.assert_allclose(mem, size)


@pytest.mark.parametrize(
    "sig_chunks, index, expected",
    [
        ((5, 5), (1, 1), [slice(0, 5, None), slice(0, 5, None)]),
        ((5, 5), (7, 7), [slice(5, 10, None), slice(5, 10, None)]),
        ((5, 5), (1, 12), [slice(0, 5, None), slice(10, 15, None)]),
        (
            (5,),
            (1,),
            [
                slice(0, 5, None),
            ],
        ),
        (
            (20,),
            (1,),
            [
                slice(0, 20, None),
            ],
        ),
        (
            (5,),
            [1],
            [
                slice(0, 5, None),
            ],
        ),
        ((5,), (25,), "error"),
        ((20, 20), (25, 21), "error"),
    ],
)
def test_get_signal_chunk_slice(sig_chunks, index, expected):
    ndim = 1 + len(index)
    data = da.zeros([20] * ndim, chunks=(10, *sig_chunks[::-1]))
    if expected == "error":
        with pytest.raises(ValueError):
            chunk_slice = get_signal_chunk_slice(index, data.chunks)
    else:
        chunk_slice = get_signal_chunk_slice(index, data.chunks)
        assert chunk_slice == expected


@pytest.mark.parametrize(
    "sig_chunks, index, expected",
    [
        ((5, 5), (12, 7), [slice(10, 15, None), slice(5, 10, None)]),
        ((5, 5), (7, 12), "error"),
    ],
)
def test_get_signal_chunk_slice_not_square(sig_chunks, index, expected):
    data = da.zeros((2, 2, 10, 20), chunks=(2, 2, *sig_chunks[::-1]))
    if expected == "error":
        with pytest.raises(ValueError):
            chunk_slice = get_signal_chunk_slice(index, data.chunks)
    else:
        chunk_slice = get_signal_chunk_slice(index, data.chunks)
        assert chunk_slice == expected


@pytest.mark.parametrize("dtype", ["<u2", "u2", ">u2", "<f4", "f4", ">f4"])
def test_numba_histogram(dtype):
    arr = np.arange(100, dtype=dtype)
    np.testing.assert_array_equal(
        numba_histogram(arr, 5, (0, 100)), [20, 20, 20, 20, 20]
    )


def test_round_half_towards_zero_integer():
    a = np.array([-2.0, -1.7, -1.5, -0.2, 0.0, 0.2, 1.5, 1.7, 2.0])
    np.testing.assert_allclose(
        round_half_towards_zero(a, decimals=0),
        np.array([-2.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 2.0]),
    )
    np.testing.assert_allclose(
        round_half_towards_zero(a, decimals=0), round_half_towards_zero(a)
    )


def test_round_half_towards_zero():
    a = np.array([-2.01, -1.56, -1.55, -1.50, -0.22, 0.0, 0.22, 1.50, 1.55, 1.56, 2.01])
    np.testing.assert_allclose(
        round_half_towards_zero(a, decimals=1),
        np.array([-2.0, -1.6, -1.5, -1.5, -0.2, 0.0, 0.2, 1.5, 1.5, 1.6, 2.0]),
    )


def test_round_half_away_from_zero_integer():
    a = np.array([-2.0, -1.7, -1.5, -0.2, 0.0, 0.2, 1.5, 1.7, 2.0])
    np.testing.assert_allclose(
        round_half_away_from_zero(a, decimals=0),
        np.array([-2.0, -2.0, -2.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0]),
    )
    np.testing.assert_allclose(
        round_half_away_from_zero(a, decimals=0), round_half_away_from_zero(a)
    )


def test_round_half_away_from_zero():
    a = np.array([-2.01, -1.56, -1.55, -1.50, -0.22, 0.0, 0.22, 1.50, 1.55, 1.56, 2.01])
    np.testing.assert_allclose(
        round_half_away_from_zero(a, decimals=1),
        np.array([-2.0, -1.6, -1.6, -1.5, -0.2, 0.0, 0.2, 1.5, 1.6, 1.6, 2.0]),
    )


@pytest.mark.parametrize("start", [0, None])
@pytest.mark.parametrize("norm", [None, "log"])
@pytest.mark.parametrize("factor", [1.0, [0.1, 1.0]])
def test_get_value_at_index(start, norm, factor):
    x = np.arange(1, 11, 1)
    line_index = [3, 4]
    line_real_index = [2, 3]
    min_intensity = 0.1
    lines = get_value_at_index(
        x,
        indexes=line_index,
        real_index=line_real_index,
        factor=factor,
        start=start,
        stop=1.0,
        norm=norm,
        minimum_intensity=min_intensity,
    )
    if norm == "log":
        y_start_ans = np.array([0.1, 0.1])
    else:
        y_start_ans = np.array([0, 0])
    y_end_ans = np.array([4, 5]) * factor
    x_ans = np.array([2, 3])
    if start is None:
        np.testing.assert_array_equal(lines, np.stack([x_ans, y_end_ans], axis=1))
    else:
        ans = np.stack(
            (
                np.stack([x_ans, y_start_ans], axis=1),
                np.stack([x_ans, y_end_ans], axis=1),
            ),
            axis=1,
        )

        np.testing.assert_array_equal(lines, ans)


def test_get_value_at_index_fail():
    with pytest.raises(ValueError):
        _ = get_value_at_index(
            np.arange(1, 11, 1),
            indexes=[3, 4],
            real_index=[2, 3],
            factor=0.1,
            start=0,
            stop=1.0,
            norm="log",
            minimum_intensity=None,
        )
