# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
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
from dask.threaded import get

import hyperspy.api as hs
from hyperspy import _lazy_signals
from hyperspy._signals.lazy import (
    _reshuffle_mixed_blocks,
    to_array,
    _get_navigation_dimension_chunk_slice
    )
from hyperspy.exceptions import VisibleDeprecationWarning


def _signal():
    ar = da.from_array(np.arange(6. * 9 * 7 * 11).reshape((6, 9, 7, 11)),
                       chunks=((2, 1, 3), (4, 5), (7,), (11,))
                       )
    return hs.signals.Signal2D(ar).as_lazy()

@pytest.fixture
def signal():
    return _signal()


@pytest.mark.parametrize("sl", [(0, 0),
                                (slice(None,), 0),
                                (slice(None), slice(None))
                                ]
                         )
def test_reshuffle(signal, sl):
    sig = signal.isig[sl]
    array = np.concatenate(
        [a for a in sig._block_iterator(flat_signal=True,
                                        navigation_mask=None,
                                        signal_mask=None)],
        axis=0
    )
    ndim = sig.axes_manager.navigation_dimension
    ans = _reshuffle_mixed_blocks(array,
                                  ndim,
                                  sig.data.shape[ndim:],
                                  sig.data.chunks[:ndim])
    np.testing.assert_allclose(ans, sig.data.compute())

nav_mask = np.zeros((6, 9), dtype=bool)
nav_mask[0, 0] = True
nav_mask[1, 1] = True
sig_mask = np.zeros((7, 11), dtype=bool)
sig_mask[0, :] = True


@pytest.mark.parametrize('nm', [None, nav_mask])
@pytest.mark.parametrize('sm', [None, sig_mask])
@pytest.mark.parametrize('flat', [True, False])
@pytest.mark.parametrize('dtype', ['float', 'int'])
def test_blockiter_bothmasks(signal, flat, dtype, nm, sm):
    real_first = get(signal.data.dask, (signal.data.name, 0, 0, 0, 0)).copy()
    real_second = get(signal.data.dask, (signal.data.name, 0, 1, 0, 0)).copy()
    # Don't want to rechunk, so change dtype manually
    signal.data = signal.data.astype(dtype)
    it = signal._block_iterator(flat_signal=flat,
                                navigation_mask=nm,
                                signal_mask=sm,
                                get=get)
    first_block = next(it)
    second_block = next(it)
    if nm is not None:
        nm = nm[:2, :4]
    real_first = real_first.astype(dtype)
    real_second = real_second.astype(dtype)
    if flat:
        if nm is not None:
            nm = ~nm
            navslice = np.where(nm.flat)[0]
        else:
            navslice = slice(None)
        sigslice = slice(11, None) if sm is not None else slice(None)
        slices1 = (navslice, sigslice)
        real_first = real_first.reshape((2 * 4, -1))[slices1]
        real_second = real_second.reshape((2 * 5, -1))[:, sigslice]
    else:
        value = np.nan if dtype == 'float' else 0
        if nm is not None:
            real_first[nm, ...] = value
        if sm is not None:
            real_first[..., sm] = value
            real_second[..., sm] = value
    np.testing.assert_allclose(first_block, real_first)
    np.testing.assert_allclose(second_block, real_second)


@pytest.mark.parametrize('sig', [_signal(),
                                 _signal().data,
                                 _signal().data.compute()])
def test_as_array_numpy(sig):
    thing = to_array(sig, chunks=None)
    assert isinstance(thing, np.ndarray)


@pytest.mark.parametrize('sig', [_signal(),
                                 _signal().data,
                                 _signal().data.compute()])
def test_as_array_dask(sig):
    chunks = ((6,), (9,), (7,), (11,))
    thing = to_array(sig, chunks=chunks)
    assert isinstance(thing, da.Array)
    assert thing.chunks == chunks


def test_as_array_fail():
    with pytest.raises(ValueError):
        to_array('asd', chunks=None)


def test_ma_lazify():
    s = hs.signals.BaseSignal(
        np.ma.masked_array(
            data=[
                1, 2, 3], mask=[
                0, 1, 0]))
    l = s.as_lazy()
    assert np.isnan(l.data[1].compute())
    ss = hs.stack([s, s])
    assert np.isnan(ss.data[:, 1]).all()


@pytest.mark.parametrize('nav_chunks', ["auto", -1, (3, -1)])
@pytest.mark.parametrize('sig_chunks', [-1, ("auto", "auto"), 4])
def test_rechunk(signal, nav_chunks, sig_chunks):
    signal.rechunk(nav_chunks=nav_chunks,
                   sig_chunks=sig_chunks)


def test_warning():
    sig = _signal()

    with pytest.warns(VisibleDeprecationWarning, match="progressbar"):
        sig.compute(progressbar=False)

    assert sig._lazy == False
    thing = to_array(sig, chunks=None)
    assert isinstance(thing, np.ndarray)


class TestGetNavigationDimensionHostChunkSlice:
    @pytest.mark.parametrize(
        "position, chunk_slice",
        [
            ((0, 0), np.s_[0:2, 0:2]),
            ((0, 19), np.s_[0:2, 18:20]),
            ((9, 9), np.s_[8:10, 8:10]),
            ((5, 14), np.s_[4:6, 14:16]),
        ],
    )
    def test_simple(self, position, chunk_slice):
        dask_array = da.zeros((10, 20, 50, 50), chunks=(2, 2, 25, 25))
        chunk_slice_output = _get_navigation_dimension_chunk_slice(
            position, dask_array.chunks
        )
        assert chunk_slice_output[:2] == chunk_slice

    @pytest.mark.parametrize(
        "position",
        [(12, 0), (0, 25), (25, 32)],
    )
    def test_out_of_range(self, position):
        dask_array = da.zeros((10, 20, 50, 50), chunks=(2, 2, 25, 25))
        chunk_slice_output = _get_navigation_dimension_chunk_slice(
            position, dask_array.chunks
        )
        assert not chunk_slice_output

    @pytest.mark.parametrize(
        "shape",
        [
            (10, 20),
            (6, 10, 20),
            (4, 6, 10, 20),
            (4, 4, 6, 10, 20),
            (2, 4, 4, 6, 10, 20),
        ],
    )
    def test_dimensions(self, shape):
        dask_array = da.zeros(shape, chunks=(2,) * len(shape))
        position = (1,) * len(shape)
        chunk_slice_output = _get_navigation_dimension_chunk_slice(
            position, dask_array.chunks
        )
        chunk_slice_compare = (slice(0, 2),) * len(shape)
        assert chunk_slice_output == chunk_slice_compare


class TestGetTemporaryDaskChunk:
    def test_correct_values(self):
        chunk_slice_list = [
            np.s_[0:5, 0:5],
            np.s_[5:10, 0:5],
            np.s_[0:5, 5:10],
            np.s_[5:10, 5:10],
            np.s_[0:5, 10:15],
            np.s_[5:10, 10:15],
        ]
        value_list = [1, 2, 3, 4, 5, 6]
        data = np.zeros((10, 15, 50, 50))
        for value, chunk_slice in zip(value_list, chunk_slice_list):
            data[chunk_slice] = value

        data = da.from_array(data, chunks=(5, 5, 25, 25))
        s = _lazy_signals.LazySignal2D(data)
        for value, chunk_slice in zip(value_list, chunk_slice_list):
            value_output = s._get_cache_dask_chunk(
                (chunk_slice[0].start, chunk_slice[1].start, slice(None), slice(None))
            )
            assert s._cache_dask_chunk.shape == (5, 5, 50, 50)
            assert np.all(s._cache_dask_chunk == value)
            assert chunk_slice == s._cache_dask_chunk_slice
            assert value == value_output.mean(dtype=np.uint16)

    def test_change_position(self):
        s = _lazy_signals.LazySignal2D(
            da.zeros((10, 10, 20, 20), chunks=(5, 5, 10, 10))
        )
        s._get_cache_dask_chunk((0, 0, slice(None), slice(None)))
        chunk_slice0 = s._cache_dask_chunk_slice

        s._cache_dask_chunk[:] = 2

        s._get_cache_dask_chunk((4, 4, slice(None), slice(None)))
        chunk_slice1 = s._cache_dask_chunk_slice
        assert chunk_slice0 == chunk_slice1
        assert np.all(s._cache_dask_chunk == 2)

        s._get_cache_dask_chunk((6, 4, slice(None), slice(None)))
        s._get_cache_dask_chunk((0, 0, slice(None), slice(None)))
        assert np.all(s._cache_dask_chunk == 0)

    @pytest.mark.parametrize(
        "shape",
        [
            (20, 30),
            (10, 20, 30),
            (6, 10, 20, 30),
            (4, 6, 10, 20, 30),
        ],
    )
    def test_dimensions(self, shape):
        chunks = (2,) * len(shape)
        s = _lazy_signals.LazySignal2D(da.zeros(shape), chunks=chunks)
        position = s.axes_manager._getitem_tuple
        s._get_cache_dask_chunk(position)
        assert len(position) == len(shape)

    def test_correct_value_within_chunk(self):
        data = np.zeros((10, 15, 50, 50))
        data[0, 0] = 1
        data[0, 1] = 2
        data[1, 0] = 3
        data[1, 1] = 4
        data = da.from_array(data, chunks=(2, 2, 25, 25))
        s = _lazy_signals.LazySignal2D(data)
        value = s._get_cache_dask_chunk(s.axes_manager._getitem_tuple)
        assert np.all(value == 1)

        s.axes_manager.indices = (1, 0)
        value = s._get_cache_dask_chunk(s.axes_manager._getitem_tuple)
        assert np.all(value == 2)

        s.axes_manager.indices = (0, 1)
        value = s._get_cache_dask_chunk(s.axes_manager._getitem_tuple)
        assert np.all(value == 3)

        s.axes_manager.indices = (1, 1)
        value = s._get_cache_dask_chunk(s.axes_manager._getitem_tuple)
        assert np.all(value == 4)

    def test_signal1d(self):
        data = np.zeros((10, 10, 20))
        data[5, 5] = 2
        data = da.from_array(data, chunks=(2, 2, 10))
        s = _lazy_signals.LazySignal1D(data)
        value = s._get_cache_dask_chunk(s.axes_manager._getitem_tuple)
        assert len(s._cache_dask_chunk_slice) == 2
        assert s._cache_dask_chunk.shape == (2, 2, 20)
        assert s._cache_dask_chunk_slice == np.s_[0:2, 0:2]
        assert len(value.shape) == 1
        assert len(value) == 20
        s.axes_manager.indices = (5, 5)
        value = s._get_cache_dask_chunk(s.axes_manager._getitem_tuple)
        assert np.all(value == 2)

    def test_changed_data_trigger(self):
        s = _lazy_signals.LazySignal2D(da.zeros((6, 6, 8, 8), chunks=(2, 2, 4, 4)))
        position = s.axes_manager._getitem_tuple
        s._get_cache_dask_chunk(position)
        assert s._cache_dask_chunk is not None
        assert s._cache_dask_chunk_slice is not None
        s.events.data_changed.trigger(None)
        assert s._cache_dask_chunk is None
        assert s._cache_dask_chunk_slice is None

    def test_map_inplace_data_changing(self):
        s = _lazy_signals.LazySignal2D(da.zeros((6, 6, 8, 8), chunks=(2, 2, 4, 4)))
        s.__call__()
        assert len(s._cache_dask_chunk.shape) == 4
        s.map(np.sum, axis=1, ragged=False, inplace=True)
        s.__call__()
        assert len(s._cache_dask_chunk.shape) == 3

    def test_clear_cache_dask_data_method(self):
        s = _lazy_signals.LazySignal2D(da.zeros((6, 6, 8, 8), chunks=(2, 2, 4, 4)))
        s.__call__()
        s._clear_cache_dask_data()
        assert s._cache_dask_chunk is None
        assert s._cache_dask_chunk_slice is None


class TestLazyPlot:
    def test_correct_value(self):
        chunk_slice_list = [
            np.s_[0:5, 0:5],
            np.s_[5:10, 0:5],
            np.s_[0:5, 5:10],
            np.s_[5:10, 5:10],
            np.s_[0:5, 10:15],
            np.s_[5:10, 10:15],
        ]
        value_list = [1, 2, 3, 4, 5, 6]
        data = np.zeros((10, 15, 50, 50))
        for value, chunk_slice in zip(value_list, chunk_slice_list):
            data[chunk_slice] = value

        data = da.from_array(data, chunks=(5, 5, 25, 25))
        s = _lazy_signals.LazySignal2D(data)
        for value, chunk_slice in zip(value_list, chunk_slice_list):
            s.plot()
            s.axes_manager.indices = (chunk_slice[1].start, chunk_slice[0].start)
            assert s._cache_dask_chunk.shape == (5, 5, 50, 50)
            assert np.all(s._cache_dask_chunk == value)
            assert chunk_slice == s._cache_dask_chunk_slice
            s._plot.close()

    @pytest.mark.parametrize(
        "shape",
        [
            (20, 30),
            (10, 20, 30),
            (6, 10, 20, 30),
            (4, 6, 10, 20, 30),
        ],
    )
    def test_dimensions(self, shape):
        chunks = (2,) * len(shape)
        s = _lazy_signals.LazySignal2D(da.zeros(shape), chunks=chunks)
        s.plot()
        s._plot.close()

    def test_plot_with_different_axes_manager(self):
        data0 = np.zeros((30, 40, 50, 50), dtype=np.uint16)
        data0[20, 20] = 100
        data0 = da.from_array(data0, chunks=(10, 10, 25, 25))
        s0 = _lazy_signals.LazySignal2D(data0)
        data1 = da.zeros((30, 40, 50, 50), chunks=(10, 10, 25, 25))
        s1 = _lazy_signals.LazySignal2D(data1)
        s0.plot(axes_manager=s1.axes_manager)
        s1.axes_manager.indices = (20, 20)
        assert np.all(s0._cache_dask_chunk[0, 0] == 100)
        s0._plot.close()

    def test_signal1d(self):
        s = _lazy_signals.LazySignal1D(da.zeros((10, 10, 20), chunks=(5, 5, 10)))
        s.plot()
        assert s._cache_dask_chunk.shape == (5, 5, 20)
        assert s._cache_dask_chunk_slice == np.s_[0:5, 0:5]
        s._plot.close()


class TestHTMLRep:

    def test_html_rep(self, signal):
        signal._repr_html_()

    def test_html_rep_zero_dim_nav(self):
        s = hs.signals.BaseSignal(da.random.random((500, 1000))).as_lazy()
        s._repr_html_()

    def test_html_rep_zero_dim_sig(self):
        s = hs.signals.BaseSignal(da.random.random((500, 1000))).as_lazy().T
        s._repr_html_()

    def test_get_chunk_string(self):
        s = hs.signals.BaseSignal(da.random.random((6, 6, 6, 6))).as_lazy()
        s = s.transpose(2)
        s.data = s.data.rechunk((3, 2, 6, 6))
        s_string = s._get_chunk_string()
        assert (s_string == "(2,3|<b>6</b>,<b>6</b>)")
        s.data = s.data.rechunk((6, 6, 2, 3))
        s_string = s._get_chunk_string()
        assert (s_string == "(<b>6</b>,<b>6</b>|3,2)")


def test_get_chunk_size(signal):
    sig = signal
    chunk_size = sig.get_chunk_size()
    assert chunk_size == ((2, 1, 3), (4, 5))
    assert sig.get_chunk_size(sig.axes_manager.navigation_axes) == chunk_size
    assert sig.get_chunk_size([0, 1]) == chunk_size

    sig = _signal()
    chunk_size = sig.get_chunk_size(axes=0)
    chunk_size == ((2, 1, 3), )
