# -*- coding: utf-8 -*-
# Copyright 2007-2021 The HyperSpy developers
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

from unittest import mock

import numpy as np
import pytest
import dask.array as da
from scipy.ndimage import gaussian_filter, gaussian_filter1d, rotate

import hyperspy.api as hs
from hyperspy.decorators import lazifyTestClass


@lazifyTestClass(ragged=False)
class TestSignal2D:

    def setup_method(self, method):
        self.im = hs.signals.Signal2D(np.arange(0., 18).reshape((2, 3, 3)))
        self.ragged = None

    @pytest.mark.parametrize('parallel', [True, False])
    def test_constant_sigma(self, parallel):
        s = self.im
        s.map(gaussian_filter, sigma=1, parallel=parallel, ragged=self.ragged)
        np.testing.assert_allclose(s.data, np.array(
            [[[1.68829507, 2.2662213, 2.84414753],
              [3.42207377, 4., 4.57792623],
              [5.15585247, 5.7337787, 6.31170493]],

             [[10.68829507, 11.2662213, 11.84414753],
              [12.42207377, 13., 13.57792623],
              [14.15585247, 14.7337787, 15.31170493]]]))

    @pytest.mark.parametrize('parallel', [True, False])
    def test_constant_sigma_navdim0(self, parallel):
        s = self.im.inav[0]
        s.map(gaussian_filter, sigma=1, parallel=parallel, ragged=self.ragged, inplace= not self.ragged)
        np.testing.assert_allclose(s.data, np.array(
            [[1.68829507, 2.2662213, 2.84414753],
             [3.42207377, 4., 4.57792623],
             [5.15585247, 5.7337787, 6.31170493]]))

    @pytest.mark.parametrize('parallel', [True, False])
    def test_variable_sigma(self, parallel):
        s = self.im

        sigmas = np.array([0, 1])

        s.map(gaussian_filter,
              sigma=sigmas, parallel=parallel, ragged=self.ragged)
        np.testing.assert_allclose(s.data, np.array(
            [[[0.42207377, 1., 1.57792623],
              [3.42207377, 4., 4.57792623],
              [6.42207377, 7., 7.57792623]],

             [[9.42207377, 10., 10.57792623],
              [12.42207377, 13., 13.57792623],
              [15.42207377, 16., 16.57792623]]]))

    @pytest.mark.parametrize('parallel', [True, False])
    def test_variable_sigma_navdim0(self, parallel):
        s = self.im

        sigma = 1
        s.map(gaussian_filter, sigma=sigma, parallel=parallel,
              ragged=self.ragged)
        np.testing.assert_allclose(s.data, np.array(
            [[[1.68829507, 2.2662213, 2.84414753],
              [3.42207377, 4., 4.57792623],
              [5.15585247, 5.7337787, 6.31170493]],

             [[10.68829507, 11.2662213, 11.84414753],
              [12.42207377, 13., 13.57792623],
              [14.15585247, 14.7337787, 15.31170493]]]))

    @pytest.mark.parametrize('parallel', [True, False])
    def test_axes_argument(self, parallel):
        s = self.im
        s.map(rotate, angle=45, reshape=False, parallel=parallel,
              ragged=self.ragged)
        np.testing.assert_allclose(s.data, np.array(
            [[[0., 2.23223305, 0.],
              [0.46446609, 4., 7.53553391],
              [0., 5.76776695, 0.]],

             [[0., 11.23223305, 0.],
              [9.46446609, 13., 16.53553391],
              [0., 14.76776695, 0.]]]))

    @pytest.mark.parametrize('parallel', [True, False])
    def test_different_shapes(self, parallel):
        s = self.im
        angles = hs.signals.BaseSignal([0, 45])
        if s._lazy:
            # inplace not compatible with ragged and lazy
            with pytest.raises(ValueError):
                s.map(rotate, angle=angles.T, reshape=True, inplace=True,
                      ragged=True)
            s = s.map(rotate, angle=angles.T, reshape=True, inplace=False,
                  ragged=True)
        else:
            s.map(rotate, angle=angles.T, reshape=True, show_progressbar=None,
                  parallel=parallel, ragged=True)
        # the dtype
        assert s.data.dtype is np.dtype('O')
        # the special slicing
        if not s._lazy:
            assert s.inav[0].data.base is s.data[0]
        # actual values
        np.testing.assert_allclose(s.data[0],
                                   np.arange(9.).reshape((3, 3)),
                                   atol=1e-7)
        np.testing.assert_allclose(s.data[1],
                                   np.array([[0., 0., 0., 0.],
                                             [0., 10.34834957,
                                                 13.88388348, 0.],
                                             [0., 12.11611652,
                                                 15.65165043, 0.],
                                             [0., 0., 0., 0.]]))

    @pytest.mark.parametrize('ragged', [True, False])
    def test_ragged(self, ragged):
        s = self.im
        out = s.map(lambda x: x, inplace=False, ragged=ragged)
        assert out.axes_manager.navigation_shape == s.axes_manager.navigation_shape
        if ragged:
            if s._lazy:
                with pytest.raises(ValueError):
                    s.map(lambda x: x, inplace=True, ragged=ragged)
            for i in range(s.axes_manager.navigation_size):
                np.testing.assert_allclose(s.data[i], out.data[i])
        else:
            np.testing.assert_allclose(s.data, out.data)

    @pytest.mark.parametrize('ragged', [True, False])
    def test_ragged_navigation_shape(self, ragged):
        s = hs.stack([self.im]*3)
        out = s.map(lambda x: x, inplace=False, ragged=ragged)
        assert out.axes_manager.navigation_shape == s.axes_manager.navigation_shape
        assert out.data.shape[:2] == s.axes_manager.navigation_shape[::-1]


@lazifyTestClass(ragged=False)
class TestSignal1D:

    def setup_method(self, method):
        self.s = hs.signals.Signal1D(np.arange(0., 6).reshape((2, 3)))
        self.ragged = None

    @pytest.mark.parametrize('parallel', [True, False])
    def test_constant_sigma(self, parallel):
        s = self.s
        m = mock.Mock()
        s.events.data_changed.connect(m.data_changed)
        s.map(gaussian_filter1d, sigma=1, parallel=parallel,
              ragged=self.ragged)
        np.testing.assert_allclose(s.data, np.array(
            ([[0.42207377, 1., 1.57792623],
              [3.42207377, 4., 4.57792623]])))
        assert m.data_changed.called

    @pytest.mark.parametrize('parallel', [True, False])
    def test_variable_signal_parameter(self, parallel):
        s = self.s
        m = mock.Mock()
        s.events.data_changed.connect(m.data_changed)
        s.map(lambda A, B: A - B, B=s, parallel=parallel, ragged=self.ragged)
        np.testing.assert_allclose(s.data, np.zeros_like(s.data))
        assert m.data_changed.called

    @pytest.mark.parametrize('parallel', [True, False])
    def test_constant_signal_parameter(self, parallel):
        s = self.s
        m = mock.Mock()
        s.events.data_changed.connect(m.data_changed)
        s.map(lambda A, B: A - B, B=s.inav[0], parallel=parallel,
              ragged=self.ragged)
        np.testing.assert_allclose(s.data, np.array(
            ([[0., 0., 0.],
              [3., 3., 3.]])))
        assert m.data_changed.called

    @pytest.mark.parametrize('parallel', [True, False])
    def test_dtype(self, parallel):
        s = self.s
        s.map(lambda data: np.sqrt(np.complex128(data)),
              parallel=parallel, ragged=self.ragged)
        assert s.data.dtype is np.dtype('complex128')

    @pytest.mark.parametrize('ragged', [True, False])
    def test_ragged(self, ragged):
        s = self.s
        out = s.map(lambda x: x, inplace=False, ragged=ragged)
        if ragged:
            for i in range(s.axes_manager.navigation_size):
                np.testing.assert_allclose(s.data[i], out.data[i])
        else:
            np.testing.assert_allclose(s.data, out.data)


@lazifyTestClass(ragged=False)
class TestSignal0D:

    def setup_method(self, method):
        self.s = hs.signals.BaseSignal(np.arange(0., 6).reshape((2, 3)))
        self.s.axes_manager.set_signal_dimension(0)
        self.ragged = None

    @pytest.mark.parametrize('parallel', [True, False])
    def test(self, parallel):
        s = self.s
        m = mock.Mock()
        s.events.data_changed.connect(m.data_changed)
        s.map(lambda x, e: x ** e, e=2, parallel=parallel, ragged=self.ragged)
        np.testing.assert_allclose(
            s.data, (np.arange(0., 6) ** 2).reshape((2, 3,)))
        assert m.data_changed.called

    @pytest.mark.parametrize('parallel', [True, False])
    def test_nav_dim_1(self, parallel):
        s = self.s.inav[1, 1]
        m = mock.Mock()
        s.events.data_changed.connect(m.data_changed)
        s.map(lambda x, e: x ** e, e=2, parallel=parallel, ragged=self.ragged)
        np.testing.assert_allclose(s.data, self.s.inav[1, 1].data ** 2)
        #assert m.data_changed.called


_alphabet = 'abcdefghijklmnopqrstuvwxyz'


@lazifyTestClass(ragged=False)
class TestChangingAxes:

    def setup_method(self, method):
        self.base = hs.signals.BaseSignal(np.empty((2, 3, 4, 5, 6, 7)))
        self.ragged = None
        for ax, name in zip(self.base.axes_manager._axes, _alphabet):
            ax.name = name

    @pytest.mark.parametrize('parallel', [True, False])
    def test_one_nav_reducing(self, parallel):
        s = self.base.transpose(signal_axes=4).inav[0, 0]
        s.map(np.mean, axis=1, parallel=parallel, ragged=self.ragged)
        assert list('def') == [ax.name for ax in
                               s.axes_manager._axes]
        assert 0 == len(s.axes_manager.navigation_axes)
        s.map(np.mean, axis=(1, 2), parallel=parallel, ragged=self.ragged)
        assert ['f'] == [ax.name for ax in s.axes_manager._axes]
        assert 0 == len(s.axes_manager.navigation_axes)

    @pytest.mark.parametrize('parallel', [True, False])
    def test_one_nav_increasing(self, parallel):
        s = self.base.transpose(signal_axes=4).inav[0, 0]
        s.map(np.tile, reps=(2, 1, 1, 1, 1),
              parallel=parallel, ragged=self.ragged)
        assert len(s.axes_manager.signal_axes) == 5
        assert set('cdef') <= {ax.name for ax in
                               s.axes_manager._axes}
        assert 0 == len(s.axes_manager.navigation_axes)
        assert s.data.shape == (2, 4, 5, 6, 7)

    @pytest.mark.parametrize('parallel', [True, False])
    def test_reducing(self, parallel):
        s = self.base.transpose(signal_axes=4)
        s.map(np.mean, axis=1, parallel=parallel, ragged=self.ragged)
        assert list('abdef') == [ax.name for ax in
                                 s.axes_manager._axes]
        assert 2 == len(s.axes_manager.navigation_axes)
        s.map(np.mean, axis=(1, 2), parallel=parallel, ragged=self.ragged)
        assert ['f'] == [ax.name for ax in
                         s.axes_manager.signal_axes]
        assert list('ba') == [ax.name for ax in
                              s.axes_manager.navigation_axes]
        assert 2 == len(s.axes_manager.navigation_axes)

    @pytest.mark.parametrize('parallel', [True, False])
    def test_increasing(self, parallel):
        s = self.base.transpose(signal_axes=4)
        s.map(np.tile, reps=(2, 1, 1, 1, 1),
              parallel=parallel, ragged=self.ragged)
        assert len(s.axes_manager.signal_axes) == 5
        assert set('cdef') <= {ax.name for ax in
                               s.axes_manager.signal_axes}
        assert list('ba') == [ax.name for ax in
                              s.axes_manager.navigation_axes]
        assert 2 == len(s.axes_manager.navigation_axes)
        assert s.data.shape == (2, 3, 2, 4, 5, 6, 7)


@pytest.mark.parametrize('parallel', [True, False])
def test_new_axes(parallel):
    s = hs.signals.Signal1D(np.empty((10, 10)))
    s.axes_manager.navigation_axes[0].name = 'a'
    s.axes_manager.signal_axes[0].name = 'b'

    def test_func(d, i):
        _slice = () + (None,) * i + (slice(None),)
        return d[_slice]
    res = s.map(test_func, inplace=False,
                i=hs.signals.BaseSignal(np.arange(10)).T,
                parallel=parallel, ragged=True)
    assert res is not None
    sl = res.inav[:2]
    assert sl.axes_manager._axes[-1].name == 'a'
    sl = res.inav[-1]
    assert isinstance(sl, hs.signals.BaseSignal)
    ax_names = {ax.name for ax in sl.axes_manager._axes}
    assert len(ax_names) == 1
    assert not 'a' in ax_names
    assert not 'b' in ax_names
    assert 0 == sl.axes_manager.navigation_dimension


class TestLazyMap:
    def setup_method(self, method):
        dask_array = da.zeros((10, 11, 12, 13), chunks=(3, 3, 3, 3))
        self.s = hs.signals.Signal2D(dask_array).as_lazy()

    @pytest.mark.parametrize('chunks', [(3, 2), (3, 3)])
    def test_map_iter(self,chunks):
        iter_array, _ = da.meshgrid(range(11), range(10))
        iter_array = iter_array.rechunk(chunks)
        s_iter = hs.signals.BaseSignal(iter_array).T
        s_iter = s_iter.as_lazy()
        f = lambda a, b: a + b
        s_out = self.s.map(function=f, b=s_iter, inplace=False)
        np.testing.assert_array_equal(s_out.mean(axis=(2, 3)).data, iter_array)

    def test_map_nav_size_error(self):
        iter_array, _ = da.meshgrid(range(12), range(10))
        s_iter = hs.signals.BaseSignal(iter_array).T
        f = lambda a, b: a + b
        with pytest.raises(ValueError):
            self.s.map(function=f, b=s_iter, inplace=False)

    def test_map_iterate_array(self):
        s = self.s
        iter_array, _ = np.meshgrid(range(11), range(10))
        f = lambda a, b: a + b
        iterating_kwargs = {'b':iter_array.T}
        s_out = s._map_iterate(function=f, iterating_kwargs=iterating_kwargs,
                               inplace=False)
        np.testing.assert_array_equal(s_out.mean(axis=(2, 3)).data, iter_array)


@pytest.mark.parametrize('ragged', [True, False, None])
def test_singleton(ragged):
    sig = hs.signals.Signal2D(np.empty((3, 2)))
    sig.axes_manager[0].name = 'x'
    sig.axes_manager[1].name = 'y'
    sig1 = sig.map(lambda x: 3, inplace=False, ragged=ragged)
    sig2 = sig.map(np.sum, inplace=False, ragged=ragged)
    sig.map(np.sum, inplace=True, ragged=ragged)
    sig_list = (sig, sig1, sig2)
    for _s in sig_list:
        assert len(_s.axes_manager._axes) == 1
        assert _s.axes_manager[0].name == 'Scalar'
        assert isinstance(_s, hs.signals.BaseSignal)
        assert not isinstance(_s, hs.signals.Signal1D)


def test_lazy_singleton():
    sig = hs.signals.Signal2D(np.empty((3, 2)))
    sig = sig.as_lazy()
    sig.axes_manager[0].name = 'x'
    sig.axes_manager[1].name = 'y'
    # One without arguments
    sig1 = sig.map(lambda x: 3, inplace=False, ragged=False)
    sig2 = sig.map(np.sum, inplace=False, ragged=False)
    # in place not supported for lazy signal and ragged
    sig.map(np.sum, ragged=False, inplace=True)
    sig_list = [sig1, sig2, sig]
    for _s in sig_list:
        assert len(_s.axes_manager._axes) == 1
        assert _s.axes_manager[0].name == 'Scalar'
        assert isinstance(_s, hs.signals.BaseSignal)
        assert not isinstance(_s, hs.signals.Signal1D)
        #assert isinstance(_s, LazySignal)


def test_lazy_singleton_ragged():
    sig = hs.signals.Signal2D(np.empty((3, 2)))
    sig = sig.as_lazy()
    sig.axes_manager[0].name = 'x'
    sig.axes_manager[1].name = 'y'
    # One without arguments
    sig1 = sig.map(lambda x: 3, inplace=False, ragged=True)
    sig2 = sig.map(np.sum, inplace=False, ragged=True)
    # in place not supported for lazy signal and ragged
    sig_list = (sig1, sig2)
    for _s in sig_list:
        assert isinstance(_s, hs.signals.BaseSignal)
        assert not isinstance(_s, hs.signals.Signal1D)
        #assert isinstance(_s, LazySignal)


def test_map_ufunc(caplog):
    data = np.arange(100, 200).reshape(10, 10)
    s = hs.signals.Signal1D(data)
    # check that it works and it raises a warning
    caplog.clear()
    # s.map(np.log)
    assert np.log(s) == s.map(np.log)
    np.testing.assert_allclose(s.data, np.log(data))
    assert "can direcly operate on hyperspy signals" in caplog.records[0].message


class TestLazyNavChunkSize1:
    @staticmethod
    def afunction(input_data):
        return np.array([1, 2, 3])

    def test_signal2d(self):
        dask_array = da.zeros((10, 15, 32, 32), chunks=(1, 1, 32, 32))
        s = hs.signals.Signal2D(dask_array).as_lazy()
        s_out = s.map(self.afunction, inplace=False, parallel=False, ragged=True)
        s_out.compute()

    def test_signal1d(self):
        dask_array = da.zeros((10, 15, 32), chunks=(1, 1, 32))
        s = hs.signals.Signal1D(dask_array).as_lazy()
        s_out = s.map(self.afunction, inplace=False, parallel=False, ragged=True)
        s_out.compute()
