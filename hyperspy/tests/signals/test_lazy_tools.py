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

import dask.array as da
import numpy as np
import pytest

from hyperspy.signals import Signal1D, Signal2D


def test_lazy_changetype_rechunk_True():
    ar = da.ones((50, 50, 512, 512), chunks=(25, 50, 128, 128), dtype=np.uint8)
    s = Signal2D(ar).as_lazy()
    assert s.data.dtype is np.dtype(np.uint8)
    chunks_old = s.data.chunks
    s.change_dtype(float, rechunk=True)
    assert s.data.dtype is np.dtype(float)
    chunks_new = s.data.chunks
    # We expect more chunks
    assert len(chunks_old[0]) * len(chunks_old[1]) < len(chunks_new[0]) * len(
        chunks_new[1]
    )
    s.change_dtype(np.uint8, rechunk=True)
    assert s.data.dtype is np.dtype(np.uint8)
    chunks_newest = s.data.chunks
    # We expect less chunks
    assert len(chunks_newest[0]) * len(chunks_newest[1]) < len(chunks_new[0]) * len(
        chunks_new[1]
    )


def test_lazy_changetype_rechunk_default():
    ar = da.ones((50, 50, 256, 256), chunks=(25, 50, 256, 256), dtype=np.uint8)
    s = Signal2D(ar).as_lazy()
    s.data = s._lazy_data(rechunk=True)
    assert s.data.dtype is np.dtype(np.uint8)
    chunks_old = s.data.chunks
    s.change_dtype(float)
    assert s.data.dtype is np.dtype(float)
    assert chunks_old == s.data.chunks


def test_lazy_changetype_rechunk_False():
    ar = da.ones((50, 50, 256, 256), chunks=(5, 5, 256, 256), dtype=np.uint8)
    s = Signal2D(ar).as_lazy()
    s.data = s._lazy_data(rechunk=True)
    assert s.data.dtype is np.dtype(np.uint8)
    chunks_old = s.data.chunks
    s.change_dtype(float, rechunk=False)
    assert s.data.dtype is np.dtype(float)
    assert chunks_old == s.data.chunks


def test_rechunk_error_parameter():
    s = Signal1D(da.ones((10, 100))).as_lazy()
    with pytest.raises(ValueError):
        s._lazy_data(rechunk=0)


def test_lazy_reduce_rechunk():
    s = Signal1D(da.ones((10, 100), chunks=(1, 2))).as_lazy()
    reduce_methods = (
        s.sum,
        s.mean,
        s.max,
        s.std,
        s.var,
        s.nansum,
        s.nanmax,
        s.nanmin,
        s.nanmean,
        s.nanstd,
        s.nanvar,
        s.indexmin,
        s.indexmax,
        s.valuemax,
        s.valuemin,
    )

    for rm in reduce_methods:
        assert rm(axis=0, rechunk=True).data.chunks == (
            (100,),
        )  # The data has been rechunked
        assert rm(axis=0, rechunk=False).data.chunks == (
            (2,) * 50,
        )  # The data has not been rechunked
        assert rm(axis=0).data.chunks == (
            (2,) * 50,
        )  # Default, which is not to rechunk


def test_lazy_reduce_rechunk_same_values():
    data = np.arange(40).reshape(2, 2, 10)
    s = Signal2D(da.from_array(data, chunks=(1, 1, 1))).as_lazy()

    reduce_methods = (
        s.sum,
        s.mean,
        s.max,
        s.std,
        s.var,
        s.nansum,
        s.nanmax,
        s.nanmin,
        s.nanmean,
        s.nanstd,
        s.nanvar,
        s.indexmin,
        s.indexmax,
        s.valuemax,
        s.valuemin,
    )

    for rm in reduce_methods:
        s_rm_rechunk = rm(axis=-2, rechunk=True)
        s_rm_not_rechunk = rm(axis=-2, rechunk=False)
        s_rm_rechunk.compute(scheduler="single-threaded")
        s_rm_not_rechunk.compute(scheduler="single-threaded")
        assert np.all(s_rm_rechunk.data == s_rm_not_rechunk.data)


def test_lazy_diff_rechunk():
    s = Signal1D(da.ones((10, 100), chunks=(1, 2))).as_lazy()
    # The data has been rechunked
    assert s.diff(axis=-1, rechunk=True).data.chunks == ((10,), (99,))
    # The data has not been rechunked
    assert s.diff(axis=-1, rechunk=False).data.chunks == ((1,) * 10, (1,) * 99)
    # The default, which is not to rechunk
    assert s.diff(axis=-1).data.chunks == ((1,) * 10, (1,) * 99)
