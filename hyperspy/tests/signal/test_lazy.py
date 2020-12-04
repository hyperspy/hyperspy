# -*- coding: utf-8 -*-
# Copyright 2007-2020 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.

import dask.array as da
import numpy as np
import pytest
from dask.threaded import get

import hyperspy.api as hs
from hyperspy import _lazy_signals
from hyperspy._signals.lazy import _reshuffle_mixed_blocks, to_array
from hyperspy.exceptions import VisibleDeprecationWarning


def _signal():
    ar = da.from_array(np.arange(6. * 9 * 7 * 11).reshape((6, 9, 7, 11)),
                       chunks=((2, 1, 3), (4, 5), (7,), (11,))
                       )
    return _lazy_signals.LazySignal2D(ar)

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


def test_warning():
    sig = _signal()

    with pytest.warns(VisibleDeprecationWarning, match="progressbar"):
        sig.compute(progressbar=False)

    assert sig._lazy == False
    thing = to_array(sig, chunks=None)
    assert isinstance(thing, np.ndarray)
