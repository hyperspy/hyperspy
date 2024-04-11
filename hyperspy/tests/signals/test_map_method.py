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

from unittest import mock

import dask.array as da
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, gaussian_filter1d, rotate

import hyperspy.api as hs
from hyperspy._signals.lazy import LazySignal
from hyperspy.decorators import lazifyTestClass
from hyperspy.misc.utils import _get_block_pattern


def identify_function(x):
    return x


def substract_function(a, b):
    return a - b


def add_function(a, b):
    return a + b


def power_function(x, e):
    return x**e


def return_three_function(x):
    return 3


@lazifyTestClass(ragged=False)
class TestSignal2D:
    def setup_method(self, method):
        self.im = hs.signals.Signal2D(np.arange(0.0, 18).reshape((2, 3, 3)))
        self.ragged = None

    def test_constant_sigma(self):
        s = self.im
        s.map(gaussian_filter, sigma=1, ragged=self.ragged)
        np.testing.assert_allclose(
            s.data,
            np.array(
                [
                    [
                        [1.68829507, 2.2662213, 2.84414753],
                        [3.42207377, 4.0, 4.57792623],
                        [5.15585247, 5.7337787, 6.31170493],
                    ],
                    [
                        [10.68829507, 11.2662213, 11.84414753],
                        [12.42207377, 13.0, 13.57792623],
                        [14.15585247, 14.7337787, 15.31170493],
                    ],
                ]
            ),
        )

    def test_constant_sigma_signal_navdim0(self):
        # navigation dimension of signal is 0
        # sigma is constant
        s = self.im.inav[0]
        s.map(gaussian_filter, sigma=1, ragged=self.ragged, inplace=not self.ragged)
        np.testing.assert_allclose(
            s.data,
            np.array(
                [
                    [1.68829507, 2.2662213, 2.84414753],
                    [3.42207377, 4.0, 4.57792623],
                    [5.15585247, 5.7337787, 6.31170493],
                ]
            ),
        )

    def test_non_lazy_chunks(self):
        s = self.im
        s.map(
            gaussian_filter,
            sigma=1,
            ragged=self.ragged,
            inplace=not self.ragged,
            navigation_chunks=(1,),
        )
        np.testing.assert_allclose(
            s.data[0],
            np.array(
                [
                    [1.68829507, 2.2662213, 2.84414753],
                    [3.42207377, 4.0, 4.57792623],
                    [5.15585247, 5.7337787, 6.31170493],
                ]
            ),
        )

    def test_variable_sigma(self):
        s = self.im

        sigmas = hs.signals.BaseSignal(np.array([0, 1])).T
        s.map(gaussian_filter, sigma=sigmas, ragged=self.ragged)

        # For inav[0], sigma is 0, data unchanged
        np.testing.assert_allclose(s.inav[0].data, np.arange(9).reshape((3, 3)))
        # For inav[1], sigma is 1
        np.testing.assert_allclose(
            s.inav[1].data,
            np.array(
                [
                    [10.68829507, 11.2662213, 11.84414753],
                    [12.42207377, 13.0, 13.57792623],
                    [14.15585247, 14.7337787, 15.31170493],
                ]
            ),
        )

    def test_variable_sigma_navdim0(self):
        # navigation dimension of signal is 1, length 2
        # navigation dimension of sigmas is 1, length 1
        # raise ValueError because of navigation dimension mismatch
        s = self.im

        sigmas = hs.signals.BaseSignal(
            np.array(
                [
                    [0],
                ]
            )
        ).T
        with pytest.raises(ValueError):
            s.map(gaussian_filter, sigma=sigmas, ragged=self.ragged)

        # sigmas is a number
        sigmas = 1
        s.map(gaussian_filter, sigma=sigmas, ragged=self.ragged)
        np.testing.assert_allclose(
            s.data,
            np.array(
                [
                    [
                        [1.68829507, 2.2662213, 2.84414753],
                        [3.42207377, 4.0, 4.57792623],
                        [5.15585247, 5.7337787, 6.31170493],
                    ],
                    [
                        [10.68829507, 11.2662213, 11.84414753],
                        [12.42207377, 13.0, 13.57792623],
                        [14.15585247, 14.7337787, 15.31170493],
                    ],
                ]
            ),
        )

    def test_axes_argument(self):
        s = self.im
        s.map(rotate, angle=45, reshape=False, ragged=self.ragged)
        np.testing.assert_allclose(
            s.data,
            np.array(
                [
                    [
                        [0.0, 2.23223305, 0.0],
                        [0.46446609, 4.0, 7.53553391],
                        [0.0, 5.76776695, 0.0],
                    ],
                    [
                        [0.0, 11.23223305, 0.0],
                        [9.46446609, 13.0, 16.53553391],
                        [0.0, 14.76776695, 0.0],
                    ],
                ]
            ),
        )

    def test_different_shapes(self):
        s = self.im
        angles = hs.signals.BaseSignal([0, 45])
        if s._lazy:
            s = s.map(rotate, angle=angles.T, reshape=True, inplace=False, ragged=True)
            s.compute()
        else:
            s.map(
                rotate, angle=angles.T, reshape=True, show_progressbar=None, ragged=True
            )
        assert s.ragged
        # the dtype
        assert s.data.dtype is np.dtype("O")
        # Check slicing
        assert s.inav[0].data[0] is s.data[0]
        # actual values
        np.testing.assert_allclose(s.data[0], np.arange(9.0).reshape((3, 3)), atol=1e-7)
        np.testing.assert_allclose(
            s.data[1],
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 10.34834957, 13.88388348, 0.0],
                    [0.0, 12.11611652, 15.65165043, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
        )

    @pytest.mark.parametrize("ragged", [True, False])
    def test_ragged(self, ragged):
        s = self.im
        out = s.map(identify_function, inplace=False, ragged=ragged)
        assert out.axes_manager.navigation_shape == s.axes_manager.navigation_shape
        if ragged:
            s.map(identify_function, inplace=True, ragged=ragged)
            for i in range(s.axes_manager.navigation_size):
                np.testing.assert_allclose(s.data[i], out.data[i])
        else:
            np.testing.assert_allclose(s.data, out.data)
        assert out.ragged == ragged
        assert s.ragged == ragged

    @pytest.mark.parametrize("ragged", [True, False])
    def test_ragged_navigation_shape(self, ragged):
        s = hs.stack([self.im] * 3)
        out = s.map(identify_function, inplace=False, ragged=ragged)
        assert out.axes_manager.navigation_shape == s.axes_manager.navigation_shape
        assert out.data.shape[:2] == s.axes_manager.navigation_shape[::-1]
        assert out.ragged == ragged
        assert not s.ragged


@lazifyTestClass(ragged=False)
class TestSignal1D:
    def setup_method(self, method):
        self.s = hs.signals.Signal1D(np.arange(0.0, 6).reshape((2, 3)))
        self.ragged = None

    def test_constant_sigma(self):
        s = self.s
        m = mock.Mock()
        s.events.data_changed.connect(m.data_changed)
        s.map(gaussian_filter1d, sigma=1, ragged=self.ragged)
        np.testing.assert_allclose(
            s.data,
            np.array(([[0.42207377, 1.0, 1.57792623], [3.42207377, 4.0, 4.57792623]])),
        )
        assert m.data_changed.called

    def test_variable_signal_parameter(self):
        s = self.s
        m = mock.Mock()
        s.events.data_changed.connect(m.data_changed)
        s.map(substract_function, b=s, ragged=self.ragged)
        np.testing.assert_allclose(s.data, np.zeros_like(s.data))
        assert m.data_changed.called

    def test_constant_signal_parameter(self):
        s = self.s
        m = mock.Mock()
        s.events.data_changed.connect(m.data_changed)
        s.map(substract_function, b=s.inav[0], ragged=self.ragged)
        np.testing.assert_allclose(
            s.data, np.array(([[0.0, 0.0, 0.0], [3.0, 3.0, 3.0]]))
        )
        assert m.data_changed.called

    def test_dtype(
        self,
    ):
        s = self.s

        def sqrt_function(data):
            return np.sqrt(np.complex128(data))

        s.map(sqrt_function, ragged=self.ragged)
        assert s.data.dtype is np.dtype("complex128")

    @pytest.mark.parametrize("ragged", [True, False])
    def test_ragged(self, ragged):
        s = self.s
        out = s.map(identify_function, inplace=False, ragged=ragged)
        if ragged:
            for i in range(s.axes_manager.navigation_size):
                np.testing.assert_allclose(s.data[i], out.data[i])
        else:
            np.testing.assert_allclose(s.data, out.data)
        assert out.ragged == ragged
        assert not s.ragged


@lazifyTestClass(ragged=False)
class TestSignal0D:
    def setup_method(self, method):
        self.s = hs.signals.BaseSignal(np.arange(0.0, 6).reshape((2, 3))).T
        self.ragged = None

    def test(self):
        s = self.s
        m = mock.Mock()
        s.events.data_changed.connect(m.data_changed)
        s.map(power_function, e=2, ragged=self.ragged)
        np.testing.assert_allclose(
            s.data,
            (np.arange(0.0, 6) ** 2).reshape(
                (
                    2,
                    3,
                )
            ),
        )
        assert m.data_changed.called

    def test_nav_dim_1(self):
        s = self.s.inav[1, 1]
        m = mock.Mock()
        s.events.data_changed.connect(m.data_changed)
        s.map(power_function, e=2, ragged=self.ragged)
        np.testing.assert_allclose(s.data, self.s.inav[1, 1].data ** 2)
        # assert m.data_changed.called


_alphabet = "abcdefghijklmnopqrstuvwxyz"


@lazifyTestClass(ragged=False)
class TestChangingAxes:
    def setup_method(self, method):
        self.base = hs.signals.BaseSignal(np.empty((2, 3, 4, 5, 6, 7)))
        self.ragged = None
        for ax, name in zip(self.base.axes_manager._axes, _alphabet):
            ax.name = name

    # warning is platform dependent and reason unknown
    @pytest.mark.filterwarnings("ignore:invalid value encountered in reduce")
    def test_one_nav_reducing(self):
        s = self.base.transpose(signal_axes=4).inav[0, 0]
        s.map(np.mean, axis=1, ragged=self.ragged)
        assert list("def") == [ax.name for ax in s.axes_manager._axes]
        assert 0 == len(s.axes_manager.navigation_axes)
        s.map(np.mean, axis=(1, 2), ragged=self.ragged)
        assert ["f"] == [ax.name for ax in s.axes_manager._axes]
        assert 0 == len(s.axes_manager.navigation_axes)

    def test_one_nav_increasing(self):
        s = self.base.transpose(signal_axes=4).inav[0, 0]
        s.map(np.tile, reps=(2, 1, 1, 1, 1), ragged=self.ragged)
        assert len(s.axes_manager.signal_axes) == 5
        assert set("cdef") <= {ax.name for ax in s.axes_manager._axes}
        assert 0 == len(s.axes_manager.navigation_axes)
        assert s.data.shape == (2, 4, 5, 6, 7)

    def test_reducing(
        self,
    ):
        s = self.base.transpose(signal_axes=4)
        s.map(np.mean, axis=1, ragged=self.ragged)
        assert list("abdef") == [ax.name for ax in s.axes_manager._axes]
        assert 2 == len(s.axes_manager.navigation_axes)
        s.map(np.mean, axis=(1, 2), ragged=self.ragged)
        assert ["f"] == [ax.name for ax in s.axes_manager.signal_axes]
        assert list("ba") == [ax.name for ax in s.axes_manager.navigation_axes]
        assert 2 == len(s.axes_manager.navigation_axes)

    def test_increasing(self):
        s = self.base.transpose(signal_axes=4)
        s.map(np.tile, reps=(2, 1, 1, 1, 1), ragged=self.ragged)
        assert len(s.axes_manager.signal_axes) == 5
        assert set("cdef") <= {ax.name for ax in s.axes_manager.signal_axes}
        assert list("ba") == [ax.name for ax in s.axes_manager.navigation_axes]
        assert 2 == len(s.axes_manager.navigation_axes)
        assert s.data.shape == (2, 3, 2, 4, 5, 6, 7)


def test_new_axes():
    s = hs.signals.Signal1D(np.empty((10, 10)))
    s.axes_manager.navigation_axes[0].name = "a"
    s.axes_manager.signal_axes[0].name = "b"

    def test_func(d, i):
        i = int(i)
        _slice = () + (None,) * i + (slice(None),)
        return d[_slice]

    res = s.map(
        test_func, inplace=False, i=hs.signals.BaseSignal(np.arange(10)).T, ragged=True
    )
    assert res is not None
    sl = res.inav[:2]
    assert sl.axes_manager._axes[-1].name == "a"
    sl = res.inav[-1]
    assert isinstance(sl, hs.signals.BaseSignal)
    ax_names = {ax.name for ax in sl.axes_manager._axes}
    assert len(ax_names) == 1
    assert "b" not in ax_names
    assert sl.axes_manager.navigation_dimension == 1


class TestLazyMap:
    def setup_method(self, method):
        dask_array = da.zeros((10, 11, 12, 13), chunks=(3, 3, 3, 3))
        self.s = hs.signals.Signal2D(dask_array).as_lazy()

    @pytest.mark.parametrize("chunks", [(3, 2), (3, 3)])
    def test_map_iter(self, chunks):
        iter_array, _ = da.meshgrid(range(11), range(10))
        iter_array = iter_array.rechunk(chunks)
        s_iter = hs.signals.BaseSignal(iter_array).T
        s_iter = s_iter.as_lazy()
        s_out = self.s.map(function=add_function, b=s_iter, inplace=False)
        np.testing.assert_array_equal(s_out.mean(axis=(2, 3)).data, iter_array)

    def test_map_nav_size_error(self):
        iter_array, _ = da.meshgrid(range(12), range(10))
        s_iter = hs.signals.BaseSignal(iter_array).T
        with pytest.raises(ValueError):
            self.s.map(function=add_function, b=s_iter, inplace=False)

    def test_keep_navigation_chunks(self):
        s = self.s
        s_out = s.map(identify_function, inplace=False, lazy_output=True)
        assert s.get_chunk_size(s.axes_manager.navigation_axes) == s_out.get_chunk_size(
            s_out.axes_manager.navigation_axes
        )

    def test_keep_navigation_chunks_cropping(self):
        s = self.s
        s1 = s.inav[1:-2, 2:-1]
        s_out = s1.map(identify_function, inplace=False, lazy_output=True)
        assert s1.get_chunk_size(
            s1.axes_manager.navigation_axes
        ) == s_out.get_chunk_size(s_out.axes_manager.navigation_axes)

    @pytest.mark.parametrize("output_signal_size", [(3,), (3, 4), (3, 4, 5)])
    def test_map_output_signal_size(self, output_signal_size):
        def f(data):
            return np.ones(output_signal_size)

        s_out = self.s.map(function=f, inplace=False)
        assert s_out.data.shape[2:] == output_signal_size
        assert s_out.axes_manager.signal_shape == output_signal_size[::-1]


def a_function(image, add=4):
    return image + add


class TestLazyResultInplace:
    def setup_method(self):
        data = np.zeros((32, 40, 64, 64), dtype=np.uint16)
        data[:, :, 32 - 10 : 32 + 10, 32 - 10 : 32 + 10] = 100
        s = hs.signals.Signal2D(data)
        dask_array = da.from_array(data, chunks=(32, 32, 32, 32))
        s_lazy = hs.signals.Signal2D(dask_array).as_lazy()
        self.s_signal_image = data[0, 0].copy()
        self.s = s
        self.s_lazy = s_lazy

    def test_lazy_input_not_lazy_output_not_inplace(self):
        s = self.s_lazy
        add = 1
        s_out = s.map(a_function, add=add, inplace=False, lazy_output=False)
        assert not s_out._lazy
        s.compute()
        for ix, iy in np.ndindex(s_out.axes_manager.navigation_shape):
            np.testing.assert_allclose(s_out.data[iy, ix], self.s_signal_image + add)
            np.testing.assert_allclose(s.data[iy, ix], self.s_signal_image)

    def test_not_lazy_input_not_lazy_output_not_inplace(self):
        s = self.s
        add = 1
        s_out = s.map(a_function, add=add, inplace=False, lazy_output=False)
        assert not s_out._lazy
        for ix, iy in np.ndindex(s_out.axes_manager.navigation_shape):
            np.testing.assert_allclose(s_out.data[iy, ix], self.s_signal_image + add)
            np.testing.assert_allclose(s.data[iy, ix], self.s_signal_image)

    def test_lazy_input_lazy_output_not_inplace(self):
        s = self.s_lazy
        add = 1
        s_out = s.map(a_function, add=add, inplace=False, lazy_output=True)
        assert s_out._lazy
        s_out.compute()
        s.compute()
        for ix, iy in np.ndindex(s_out.axes_manager.navigation_shape):
            np.testing.assert_allclose(s_out.data[iy, ix], self.s_signal_image + add)
            np.testing.assert_allclose(s.data[iy, ix], self.s_signal_image)

    def test_not_lazy_input_lazy_output_not_inplace(self):
        s = self.s
        add = 1
        s_out = s.map(a_function, add=add, inplace=False, lazy_output=True)
        assert s_out._lazy
        s_out.compute()
        for ix, iy in np.ndindex(s_out.axes_manager.navigation_shape):
            np.testing.assert_allclose(s_out.data[iy, ix], self.s_signal_image + add)
            np.testing.assert_allclose(s.data[iy, ix], self.s_signal_image)

    def test_lazy_input_not_lazy_output_inplace(self):
        s = self.s_lazy
        add = 1
        s.map(a_function, add=add, inplace=True, lazy_output=False)
        assert not s._lazy
        for ix, iy in np.ndindex(s.axes_manager.navigation_shape):
            np.testing.assert_allclose(s.data[iy, ix], self.s_signal_image + add)

    def test_not_lazy_input_not_lazy_output_inplace(self):
        s = self.s
        add = 1
        s.map(a_function, add=add, inplace=True, lazy_output=False)
        assert not s._lazy
        for ix, iy in np.ndindex(s.axes_manager.navigation_shape):
            np.testing.assert_allclose(s.data[iy, ix], self.s_signal_image + add)

    def test_lazy_input_lazy_output_inplace(self):
        s = self.s_lazy
        add = 1
        s.map(a_function, add=add, inplace=True, lazy_output=True)
        assert s._lazy
        s.compute()
        for ix, iy in np.ndindex(s.axes_manager.navigation_shape):
            np.testing.assert_allclose(s.data[iy, ix], self.s_signal_image + add)

    def test_not_lazy_input_lazy_output_inplace(self):
        s = self.s
        add = 1
        s.map(a_function, add=add, inplace=True, lazy_output=True)
        assert s._lazy
        s.compute()
        for ix, iy in np.ndindex(s.axes_manager.navigation_shape):
            np.testing.assert_allclose(s.data[iy, ix], self.s_signal_image + add)


class TestOutputDtype:
    @pytest.mark.parametrize(
        "dtype", [np.uint16, np.uint32, np.uint64, np.int32, np.float32]
    )
    def test_output_dtype_specified_not_inplace(self, dtype):
        def a_function_dtype(data):
            return data.astype("float32")

        s = hs.signals.Signal1D(np.zeros((10, 100)), dtype=np.int16)
        s_out = s.map(
            a_function_dtype, inplace=False, output_dtype=dtype, lazy_output=True
        )
        assert s_out.data.dtype == dtype
        s_out.compute()
        assert s_out.data.dtype == dtype

    @pytest.mark.parametrize(
        "dtype", [np.uint16, np.uint32, np.uint64, np.int32, np.float32]
    )
    def test_output_dtype_specified_inplace(self, dtype):
        def a_function_dtype(data):
            return data.astype("float32")

        s = hs.signals.Signal1D(np.zeros((10, 100)), dtype=np.int16)
        s.map(a_function_dtype, inplace=True, output_dtype=dtype, lazy_output=True)
        assert s.data.dtype == dtype
        s.compute()
        assert s.data.dtype == dtype

    @pytest.mark.parametrize(
        "dtype", [np.uint16, np.uint32, np.uint64, np.int32, np.float32]
    )
    def test_output_dtype_auto(self, dtype):
        def a_function_dtype(data, dtype_to_function):
            return data.astype(dtype_to_function)

        s = hs.signals.Signal1D(np.zeros((10, 100)), dtype=np.int16)
        s_out = s.map(
            a_function_dtype, inplace=False, dtype_to_function=dtype, lazy_output=True
        )
        assert s_out.data.dtype == dtype
        s_out.compute()
        assert s_out.data.dtype == dtype

    @pytest.mark.parametrize("output_signal_size", [(10,), (10, 20), (10, 20, 30)])
    def test_output_signal_size(self, output_signal_size):
        def a_function_signal_size(data, output_signal_size_for_function):
            return np.zeros(output_signal_size_for_function)

        s = hs.signals.Signal1D(np.zeros((10, 100)), dtype=np.int16)
        s_out = s.map(
            a_function_signal_size,
            inplace=False,
            output_signal_size=output_signal_size,
            lazy_output=True,
            output_signal_size_for_function=output_signal_size,
        )
        assert s_out.data[0].shape == output_signal_size
        s_out.compute()
        assert s_out.data[0].shape == output_signal_size

    def test_output_signal_size_wrong_size(self):
        def a_function(data):
            return np.zeros(10)

        s = hs.signals.Signal1D(np.zeros((10, 100)), dtype=np.int16)
        s_out = s.map(
            a_function, inplace=False, output_signal_size=(11,), lazy_output=True
        )
        with pytest.raises(ValueError):
            s_out.compute()


class TestOutputSignalSizeScalarWithNavigationDimensions:
    @pytest.mark.parametrize("nav_shape", ((9,), (8, 7), (6, 5, 4)))
    def test_not_lazy_output(self, nav_shape):
        def a_function(image):
            return 10

        data_shape = nav_shape + (20, 30)
        data = np.zeros(data_shape)
        s = hs.signals.Signal2D(data)
        s_out = s.map(a_function, inplace=False, lazy_output=False)
        assert s_out.data.shape == nav_shape
        assert s_out.axes_manager.navigation_shape == nav_shape[::-1]
        assert (s_out.data == np.ones(nav_shape, dtype=float) * 10).all()
        assert s.data.shape == data_shape
        assert s.axes_manager.shape == nav_shape[::-1] + (30, 20)

        s.map(a_function, inplace=True, lazy_output=False)
        assert s.data.shape == nav_shape
        assert s.axes_manager.navigation_shape == nav_shape[::-1]

    @pytest.mark.parametrize("nav_shape", ((9,), (8, 7), (6, 5, 4)))
    def test_lazy_output(self, nav_shape):
        def a_function(image):
            return 10

        data_shape = nav_shape + (20, 30)
        data = np.zeros(data_shape)
        s = hs.signals.Signal2D(data)
        s_out = s.map(a_function, inplace=False, lazy_output=True)
        assert s_out.data.shape == nav_shape
        assert s_out.axes_manager.navigation_shape == nav_shape[::-1]
        assert s.data.shape == data_shape
        assert s.axes_manager.shape == nav_shape[::-1] + (30, 20)

        s.map(a_function, inplace=True, lazy_output=True)
        assert s.data.shape == nav_shape
        assert s.axes_manager.navigation_shape == nav_shape[::-1]


class TestGetIteratingKwargsSignal2D:
    def setup_method(self):
        dask_array = da.zeros((10, 20, 100, 100), chunks=(5, 10, 50, 50))
        s = hs.signals.Signal2D(dask_array).as_lazy()
        self.s = s

    def test_empty(self):
        s = self.s
        iterating_kwargs = {}
        args, arg_keys = s._get_iterating_kwargs(iterating_kwargs)
        assert len(arg_keys) == 0
        assert len(args) == 0

    def test_one_iterating_kwarg(self):
        s = self.s
        nav_chunks = s.get_chunk_size(axes=s.axes_manager.navigation_axes)
        nav_dim = len(nav_chunks)
        s_iter0 = hs.signals.Signal1D(np.random.random((10, 20, 2)))
        iterating_kwargs = {"iter0": s_iter0}
        args, arg_keys = s._get_iterating_kwargs(iterating_kwargs)
        assert arg_keys == ("iter0",)
        for arg in args:
            iter_nav_chunks = arg.chunks[: len(nav_chunks)]
            assert nav_chunks == iter_nav_chunks
            assert np.all(np.squeeze(arg.chunks[nav_dim:]) == arg.shape[nav_dim:])
            assert np.all(s_iter0.data == np.squeeze(arg.compute()))

    def test_many_iterating_kwarg(self):
        s = self.s
        nav_chunks = s.get_chunk_size(axes=s.axes_manager.navigation_axes)
        nav_dim = len(nav_chunks)
        s_iter0 = hs.signals.Signal1D(np.random.random((10, 20, 2)))
        s_iter1 = hs.signals.Signal2D(np.random.random((10, 20, 200, 200)))
        s_iter2 = hs.signals.BaseSignal(np.random.random((10, 20, 100, 100, 4)))
        s_iter2 = s_iter2.transpose(navigation_axes=(-2, -1))
        s_iter_list = [s_iter0, s_iter1, s_iter2]
        iterating_kwargs = {"iter0": s_iter0, "iter1": s_iter1, "iter2": s_iter2}
        args, arg_keys = s._get_iterating_kwargs(iterating_kwargs)
        assert arg_keys == ("iter0", "iter1", "iter2")
        for iarg, arg in enumerate(args):
            iter_nav_chunks = arg.chunks[:nav_dim]
            assert nav_chunks == iter_nav_chunks
            assert np.all(np.squeeze(arg.chunks[nav_dim:]) == arg.shape[nav_dim:])
            assert np.all(s_iter_list[iarg].data == np.squeeze(arg.compute()))

    def test_lazy_iterating_kwarg(self):
        s = self.s
        nav_chunks = s.get_chunk_size(axes=s.axes_manager.navigation_axes)
        nav_dim = len(nav_chunks)
        dask_array_iter0 = da.zeros((10, 20, 2), chunks=(5, 10, 2))
        dask_array_iter1 = da.zeros((10, 20, 2), chunks=(5, 5, 2))
        s_iter0 = hs.signals.Signal1D(dask_array_iter0).as_lazy()
        s_iter1 = hs.signals.Signal1D(dask_array_iter1).as_lazy()
        iterating_kwargs = {"iter0": s_iter0, "iter1": s_iter1}
        args, arg_keys = s._get_iterating_kwargs(iterating_kwargs)
        assert arg_keys == ("iter0", "iter1")
        for arg in args:
            iter_nav_chunks = arg.chunks[: len(nav_chunks)]
            assert nav_chunks == iter_nav_chunks
            assert np.all(np.squeeze(arg.chunks[nav_dim:]) == arg.shape[nav_dim:])

    def test_cropping_iterating_kwarg(self):
        s = self.s.inav[1:]
        nav_chunks = s.get_chunk_size(axes=s.axes_manager.navigation_axes)
        nav_dim = len(nav_chunks)
        s_iter0 = hs.signals.Signal1D(np.random.random((10, 19, 2)))
        iterating_kwargs = {"iter0": s_iter0}
        args, arg_keys = s._get_iterating_kwargs(iterating_kwargs)
        assert arg_keys == ("iter0",)
        for arg in args:
            iter_nav_chunks = arg.chunks[: len(nav_chunks)]
            assert nav_chunks == iter_nav_chunks
            assert np.all(np.squeeze(arg.chunks[nav_dim:]) == arg.shape[nav_dim:])
            assert np.all(s_iter0.data == np.squeeze(arg.compute()))

    def test_iterating_kwarg_non_array(self):
        def apply_func(data, f):
            return f(data, data)

        s = self.s.inav[0:2, 0:2]
        iter_add = hs.signals.BaseSignal([[np.add, np.add], [np.add, np.add]]).T
        out = s.map(apply_func, f=iter_add, inplace=False)
        np.testing.assert_array_equal(out.data, s.data)


class TestGetBlockPattern:
    @pytest.mark.parametrize(
        "input_shape",
        [(50, 40), (10, 50, 40), (100, 10, 40, 70), (150, 100, 20, 65, 13)],
    )
    def test_no_change_2d_signal(self, input_shape):
        chunks = (10,) * len(input_shape)
        dask_array = da.random.random(input_shape, chunks=chunks)
        s = hs.signals.Signal2D(dask_array).as_lazy()
        arg_pairs, adjust_chunks, new_axis, output_pattern = _get_block_pattern(
            (s.data,), input_shape
        )
        assert new_axis == {}
        assert adjust_chunks == {}

    @pytest.mark.parametrize(
        "input_shape",
        [(20,), (50, 40), (10, 50, 40), (100, 10, 40, 70), (150, 100, 20, 65, 13)],
    )
    def test_no_change_1d_signal(self, input_shape):
        chunks = (10,) * len(input_shape)
        dask_array = da.random.random(input_shape, chunks=chunks)
        s = hs.signals.Signal1D(dask_array).as_lazy()
        arg_pairs, adjust_chunks, new_axis, output_pattern = _get_block_pattern(
            (s.data,), input_shape
        )
        assert new_axis == {}
        assert adjust_chunks == {}

    def test_different_output_signal_size_signal2d(self):
        s = hs.signals.Signal2D(np.zeros((4, 5)))
        arg_pairs, adjust_chunks, new_axis, output_pattern = _get_block_pattern(
            (s.data,), (1,)
        )
        assert new_axis == {}
        assert adjust_chunks == {0: 1, 1: 0}

    def test_different_output_signal_size_signal2d_2(self):
        s = hs.signals.Signal2D(np.zeros((7, 10, 5)))
        arg_pairs, adjust_chunks, new_axis, output_pattern = _get_block_pattern(
            (s.data,), (7, 2)
        )
        assert new_axis == {}
        assert adjust_chunks == {1: 2, 2: 0}

    def test_different_output_signal_size_signal2d_3(self):
        s = hs.signals.Signal2D(np.zeros((3, 2, 7, 10, 5)))
        arg_pairs, adjust_chunks, new_axis, output_pattern = _get_block_pattern(
            (s.data,),
            (
                3,
                2,
                5,
            ),
        )
        assert new_axis == {}
        assert adjust_chunks == {2: 5, 3: 0, 4: 0}


def test_dask_array_store():
    def a_function(image):
        image = image * 101
        return image

    s = hs.signals.Signal2D(np.ones((10, 12, 20, 24)), dtype=np.int16)
    s.map(a_function, inplace=True, lazy_output=False)
    assert (s.data == 101).all()


class TestOutputShape:
    @pytest.mark.parametrize(
        "shape", [(2, 2, 10, 10), (3, 30, 50, 20), (40, 50, 100, 120)]
    )
    def test_2d_input_1d_output(self, shape):
        dask_array = da.zeros(shape, chunks=(10, 10, 20, 20))
        s = hs.signals.Signal2D(dask_array).as_lazy()

        def a_function(image):
            return np.zeros((2,))

        s_out = s.map(a_function, inplace=False, lazy_output=True)
        assert s.data.shape[:-2] + (2,) == s_out.data.shape

    @pytest.mark.parametrize(
        "shape", [(2, 2, 10, 10), (3, 30, 50, 20), (40, 50, 100, 120)]
    )
    def test_2d_input_2d_output(self, shape):
        dask_array = da.zeros(shape, chunks=(10, 10, 20, 20))
        s = hs.signals.Signal2D(dask_array).as_lazy()

        def a_function(image):
            return np.zeros((2, 3))

        s_out = s.map(a_function, inplace=False, lazy_output=True)
        assert s.data.shape[:-2] + (2, 3) == s_out.data.shape


@pytest.mark.parametrize("ragged", [True, False, None])
def test_singleton(ragged):
    sig = hs.signals.Signal2D(np.empty((3, 2)))
    sig.axes_manager[0].name = "x"
    sig.axes_manager[1].name = "y"
    sig1 = sig.map(return_three_function, inplace=False, ragged=ragged)
    sig2 = sig.map(np.sum, inplace=False, ragged=ragged)
    sig.map(np.sum, inplace=True, ragged=ragged)
    sig_list = (sig, sig1, sig2)
    for _s in sig_list:
        assert len(_s.axes_manager._axes) == 0 if ragged else 1
        ragged2 = ragged if ragged is not None else False
        if not ragged:
            assert _s.axes_manager[0].name == "Scalar"
        assert _s.axes_manager.ragged == ragged2
        assert _s.ragged == ragged2
        assert isinstance(_s, hs.signals.BaseSignal)
        assert not isinstance(_s, hs.signals.Signal1D)


def test_lazy_singleton():
    sig = hs.signals.Signal2D(np.empty((3, 2)))
    sig = sig.as_lazy()
    sig.axes_manager[0].name = "x"
    sig.axes_manager[1].name = "y"
    sig1 = sig.map(return_three_function, inplace=False, ragged=False)
    sig2 = sig.map(np.sum, inplace=False, ragged=False)
    sig.map(np.sum, ragged=False, inplace=True)
    sig_list = [sig1, sig2, sig]
    for _s in sig_list:
        assert len(_s.axes_manager._axes) == 1
        assert _s.axes_manager[0].name == "Scalar"
        assert isinstance(_s, hs.signals.BaseSignal)
        assert not isinstance(_s, hs.signals.Signal1D)
        assert not _s.ragged
        assert isinstance(_s, LazySignal)


def test_lazy_singleton_ragged():
    sig = hs.signals.Signal2D(np.empty((3, 2)))
    sig = sig.as_lazy()
    sig.axes_manager[0].name = "x"
    sig.axes_manager[1].name = "y"
    sig1 = sig.map(return_three_function, inplace=False, ragged=True)
    sig2 = sig.map(np.sum, inplace=False, ragged=True)
    sig.map(np.sum, inplace=True, ragged=True)
    sig_list = (sig1, sig2, sig)
    for _s in sig_list:
        assert isinstance(_s, hs.signals.BaseSignal)
        assert not isinstance(_s, hs.signals.Signal1D)
        assert _s.ragged
        assert isinstance(_s, LazySignal)


def test_map_ufunc(caplog):
    data = np.arange(100, 200).reshape(10, 10)
    s = hs.signals.Signal1D(data)
    # check that it works and it raises a warning
    caplog.clear()
    # s.map(np.log)
    assert np.log(s) == s.map(np.log)
    np.testing.assert_allclose(s.data, np.log(data))
    assert "can directly operate on hyperspy signals" in caplog.records[0].message


def shift_intensity_function(image, shift, intensity, crop):
    x, y = shift
    crop_x0, crop_x1 = crop[0]
    crop_y0, crop_y1 = crop[1]
    image = image[crop_x0:crop_x1, crop_y0:crop_y1]
    image_out = np.roll(image, (-x + crop_x0, -y + crop_y0), axis=(0, 1)) / intensity
    return image_out


class TestMapIterate:
    def setup_method(self):
        px, py, dx, dy = 20, 10, 200, 100
        self.s = hs.signals.Signal2D(np.ones((py, px, dy, dx)))
        self.px, self.py, self.dx, self.dy = px, py, dx, dy

    def test_lazy_output_none(self):
        s = self.s
        s_out = s._map_iterate(np.sum, lazy_output=None, inplace=False)
        assert (s_out.data == self.dx * self.dy).all()

    def test_lazy_output_false(self):
        s = self.s
        s_out = s._map_iterate(np.sum, lazy_output=False, inplace=False)
        assert (s_out.data == self.dx * self.dy).all()

    def test_lazy_output_true(self):
        s = self.s
        s_out = s._map_iterate(np.sum, lazy_output=True, inplace=False)
        s_out.compute()
        assert (s_out.data == self.dx * self.dy).all()

    def test_iterating_kwargs_none(self):
        s = self.s
        s_out = s._map_iterate(np.sum, iterating_kwargs=None)
        assert (s_out.data == self.dx * self.dy).all()

    def test_iterating_kwargs_dict(self):
        def add_sum(image, add):
            out = np.sum(image) + add
            return out

        s = self.s
        s_add = hs.signals.BaseSignal(2 * np.ones((10, 20))).T
        s_out = s._map_iterate(add_sum, inplace=False, iterating_kwargs={"add": s_add})
        assert ((s_out.data == self.dx * self.dy) + 2).all()

    def test_iter_kwarg_larger_shape_ragged(self):
        def return_img(image, add):
            return image

        x = np.empty((2,), dtype=object)
        x[0] = np.ones((4, 2))
        x[1] = np.ones((4, 2))

        s = hs.signals.BaseSignal(x, ragged=True)

        s_add = hs.signals.BaseSignal(2 * np.ones((2, 201, 101))).transpose(2)
        s_out = s.map(return_img, inplace=False, add=s_add, ragged=True)
        np.testing.assert_array_equal(s_out.data[0], x[0])


class TestFullProcessing:
    def setup_method(self):
        data_array = np.zeros((30, 40, 50, 60), dtype=np.uint16)
        shift_array = np.random.randint(20, 40, size=(30, 40, 2))
        intensity_array = np.random.randint(1, 2000, size=(30, 40))
        crop_array = np.zeros((30, 40, 2, 2), dtype=np.int16)
        crop_array[:, :, 0] = 5, -5
        crop_array[:, :, 1] = 8, -8
        for ix, iy in np.ndindex(data_array.shape[:-2]):
            shift_x, shift_y = shift_array[ix, iy]
            data_array[ix, iy, shift_x, shift_y] = intensity_array[ix, iy]

        self.s = hs.signals.Signal2D(data_array)
        self.s_shift = hs.signals.Signal1D(shift_array)
        s_intensity = hs.signals.BaseSignal(intensity_array)
        self.s_intensity = s_intensity.transpose(navigation_axes=(0, 1))
        s_crop = hs.signals.BaseSignal(crop_array)
        self.s_crop = s_crop.transpose(navigation_axes=(-2, -1))

    def test_signal2d_all_nonlazy(self):
        s = self.s
        s_crop, s_shift, s_intensity = self.s_crop, self.s_shift, self.s_intensity
        s_out = s.map(
            function=shift_intensity_function,
            shift=s_shift,
            intensity=s_intensity,
            crop=s_crop,
            inplace=False,
        )
        assert np.all(s_out.data[:, :, 0, 0] == 1.0)
        s_out.data[:, :, 0, 0] = 0.0
        assert not np.any(s_out.data)
        assert s_out.axes_manager.shape == (40, 30, 44, 40)

    def test_signal2d_lazy_signal_input(self):
        s = self.s
        s_crop, s_shift, s_intensity = self.s_crop, self.s_shift, self.s_intensity
        s.data = da.from_array(s.data, chunks=(5, 10, 20, 20))
        s = s.as_lazy()
        s_out = s.map(
            function=shift_intensity_function,
            shift=s_shift,
            intensity=s_intensity,
            crop=s_crop,
            inplace=False,
            lazy_output=False,
        )
        assert np.all(s_out.data[:, :, 0, 0] == 1.0)
        s_out.data[:, :, 0, 0] = 0.0
        assert not np.any(s_out.data)
        assert s_out.axes_manager.shape == (40, 30, 44, 40)

    def test_signal2d_lazy_all_input(self):
        s = self.s
        s_crop, s_shift, s_intensity = self.s_crop, self.s_shift, self.s_intensity
        s.data = da.from_array(s.data, chunks=(5, 10, 20, 20))
        s_crop.data = da.from_array(s_crop.data, chunks=(5, 10, 2, 2))
        s_shift.data = da.from_array(s_shift.data, chunks=(5, 10, 2))
        s, s_crop = s.as_lazy(), s_crop.as_lazy()
        s_shift, s_intensity = s_shift.as_lazy(), s_intensity.as_lazy()
        s_out = s.map(
            function=shift_intensity_function,
            shift=s_shift,
            intensity=s_intensity,
            crop=s_crop,
            inplace=False,
            lazy_output=False,
        )
        assert np.all(s_out.data[:, :, 0, 0] == 1.0)
        s_out.data[:, :, 0, 0] = 0.0
        assert not np.any(s_out.data)
        assert s_out.axes_manager.shape == (40, 30, 44, 40)

    def test_crop_signal2d_lazy_all_input(self):
        s = self.s
        s_crop, s_shift, s_intensity = self.s_crop, self.s_shift, self.s_intensity
        s.data = da.from_array(s.data, chunks=(5, 10, 20, 20))
        s_crop.data = da.from_array(s_crop.data, chunks=(5, 10, 2, 2))
        s_shift.data = da.from_array(s_shift.data, chunks=(5, 10, 2))
        s_intensity.data = da.from_array(s_intensity.data, chunks=(5, 10))
        s, s_crop = s.as_lazy(), s_crop.as_lazy()
        s_shift, s_intensity = s_shift.as_lazy(), s_intensity.as_lazy()
        s = s.inav[1:, 2:]
        s_crop = s_crop.inav[1:, 2:]
        s_shift = s_shift.inav[1:, 2:]
        s_intensity = s_intensity.inav[1:, 2:]
        s_out = s.map(
            function=shift_intensity_function,
            shift=s_shift,
            intensity=s_intensity,
            crop=s_crop,
            inplace=False,
            lazy_output=False,
        )
        assert np.all(s_out.data[:, :, 0, 0] == 1.0)
        s_out.data[:, :, 0, 0] = 0.0
        assert not np.any(s_out.data)
        assert s_out.axes_manager.shape == (39, 28, 44, 40)

    def test_rechunk_arguments(self):
        chunk_shape = (2, 2, 2, 2, 2)

        def add_sum(image, add1, add2):
            temp_add = add1.sum(-1) + add2
            out = image + np.sum(temp_add)
            return out

        x = np.ones((4, 5, 10, 11))
        s = hs.signals.Signal2D(x)
        s_add1 = hs.signals.BaseSignal(2 * np.ones((4, 5, 2, 3, 2))).transpose(3)
        s_add2 = hs.signals.BaseSignal(3 * np.ones((4, 5, 2, 3))).transpose(2)

        s = hs.signals.Signal2D(da.from_array(s.data, chunks=(2, 2, 2, 2))).as_lazy()
        s_add1 = (
            hs.signals.Signal2D(da.from_array(s_add1.data, chunks=chunk_shape))
            .as_lazy()
            .transpose(navigation_axes=(1, 2))
        )
        s_out = s.map(
            add_sum, inplace=False, add1=s_add1, add2=s_add2, lazy_output=False
        )
        assert s_out.axes_manager.shape == s.axes_manager.shape


class TestLazyNavChunkSize1:
    @staticmethod
    def afunction(input_data):
        return np.array([1, 2, 3])

    def test_signal2d(self):
        dask_array = da.zeros((10, 15, 32, 32), chunks=(1, 1, 32, 32))
        s = hs.signals.Signal2D(dask_array).as_lazy()
        s_out = s.map(self.afunction, inplace=False, ragged=True, lazy_output=True)
        s_out.compute()

    def test_signal1d(self):
        dask_array = da.zeros((10, 15, 32), chunks=(1, 1, 32))
        s = hs.signals.Signal1D(dask_array).as_lazy()
        s_out = s.map(self.afunction, inplace=False, ragged=True, lazy_output=True)
        s_out.compute()


class TestLazyInputMapAll:
    def test_not_inplace(self):
        dask_array = da.random.random((500, 500)) + 2.0
        s = hs.signals.Signal2D(dask_array).as_lazy()
        s_rot = s.map(
            function=rotate,
            angle=31,
            inplace=False,
            reshape=False,
            lazy_output=False,
        )
        assert not s_rot._lazy
        assert not hasattr(s_rot.data, "compute")
        assert s._lazy
        assert hasattr(s.data, "compute")
        assert s_rot.data[0, 0] == 0.0
        assert s_rot.data[0, -1] == 0.0
        assert s_rot.data[-1, 0] == 0.0
        assert s_rot.data[-1, -1] == 0.0
        assert s.data[0, 0] != 0.0
        assert s.data[0, -1] != 0.0
        assert s.data[-1, 0] != 0.0
        assert s.data[-1, -1] != 0.0

    def test_inplace(self):
        dask_array = da.random.random((500, 500)) + 2.0
        s = hs.signals.Signal2D(dask_array).as_lazy()
        s_rot = s.map(
            function=rotate,
            angle=31,
            inplace=True,
            reshape=False,
            lazy_output=False,
        )
        assert not s._lazy
        assert not hasattr(s.data, "compute")
        assert s.data[0, 0] == 0.0
        assert s.data[0, -1] == 0.0
        assert s.data[-1, 0] == 0.0
        assert s.data[-1, -1] == 0.0
        assert s_rot is None


class TestCompareMapAllvsMapIterate:
    @pytest.mark.parametrize(
        "shape", [(50, 50), (5, 50, 50), (3, 4, 50, 50), (3, 4, 5, 50, 50)]
    )
    def test_same_output_size(self, shape):
        data = np.random.randint(1, 99, shape)
        s = hs.signals.Signal2D(data)
        kwargs = {
            "function": rotate,
            "angle": 31,
            "inplace": False,
            "reshape": False,
            "lazy_output": False,
        }
        _ = s.map(**kwargs)

    @pytest.mark.parametrize(
        "shape", [(50, 50), (5, 50, 50), (3, 4, 50, 50), (3, 4, 5, 50, 50)]
    )
    def test_new_output_size(self, shape):
        data = np.random.randint(1, 99, (2, 2, 50, 50))
        s = hs.signals.Signal2D(data)
        kwargs = {
            "function": rotate,
            "angle": 31,
            "inplace": False,
            "reshape": True,
            "lazy_output": False,
        }
        s_rot_not_par = s.map(**kwargs)
        s_rot_par = s.map(**kwargs)
        assert (s_rot_par.data == s_rot_not_par.data).all()
        assert s_rot_not_par.axes_manager.signal_shape != (50, 50)
        assert s_rot_par.axes_manager.signal_shape != (50, 50)


def test_ragged():
    def afunction(image):
        output = np.arange(0, np.random.randint(1, 100))
        return output

    s = hs.signals.Signal1D(np.ones((10, 8, 100)))
    s_out = s.map(afunction, inplace=False, ragged=True)
    assert s_out.axes_manager.shape == s.axes_manager.navigation_shape
    assert s_out.data.dtype == object
    with pytest.raises(ValueError):
        s.map(afunction, inplace=False, ragged=False)


class TestRaggedInputSignal:
    def setup_method(self):
        data = np.empty((6, 4), dtype=object)
        for iy, ix in np.ndindex(data.shape):
            data[iy, ix] = np.arange(2, np.random.randint(4, 10))
        self.data = data

    def test_ragged_output(self):
        def test_function(image):
            return image[:-1]

        s = hs.signals.BaseSignal(self.data, ragged=True)
        s_out = s.map(test_function, inplace=False)
        assert s.axes_manager.shape == s_out.axes_manager.shape
        for iy, ix in np.ndindex(s.data.shape):
            assert np.all(s.data[iy, ix][:-1] == s_out.data[iy, ix])

    def test_lazy_input_ragged_output(self):
        def test_function(image):
            return image[:-1]

        dask_array = da.from_array(self.data, chunks=(2, 2))
        s = hs.signals.BaseSignal(dask_array, ragged=True).as_lazy()
        s_out = s.map(test_function, inplace=False)
        s.compute(show_progressbar=False)
        s_out.compute(show_progressbar=False)
        assert s.axes_manager.shape == s_out.axes_manager.shape
        for iy, ix in np.ndindex(s.data.shape):
            assert np.all(s.data[iy, ix][:-1] == s_out.data[iy, ix])

    def test_lazy_input_ragged_output_lazy_output_false(self):
        def test_function(image):
            return image[:-1]

        dask_array = da.from_array(self.data, chunks=(2, 2))
        s = hs.signals.BaseSignal(dask_array, ragged=True).as_lazy()
        s_out = s.map(test_function, inplace=False, lazy_output=False)
        s.compute(show_progressbar=False)
        assert s.axes_manager.shape == s_out.axes_manager.shape
        for iy, ix in np.ndindex(s.data.shape):
            assert np.all(s.data[iy, ix][:-1] == s_out.data[iy, ix])

    def test_not_ragged_output(self):
        def test_function(image):
            return np.sum(image)

        s = hs.signals.BaseSignal(self.data, ragged=True)
        s_out = s.map(test_function, inplace=False, ragged=False)
        assert s.axes_manager.shape == s_out.axes_manager.shape
        s_out_t = s_out.T
        assert s_out_t.axes_manager.signal_shape == s.axes_manager.navigation_shape
        assert s_out_t.axes_manager.navigation_dimension == 0


def test_0d_numpy_array_input():
    im = hs.signals.Signal2D(np.random.random((10, 64, 64)))
    sigmas = hs.signals.BaseSignal(np.linspace(2, 5, 10)).T
    im.map(gaussian_filter, sigma=sigmas)


class TestMapAll:
    def setup_method(self, method):
        im = hs.signals.Signal2D(np.random.random((10, 64, 64)))
        self.im = im

    @pytest.mark.parametrize("inplace", (True, False))
    def test_map_reduce(self, inplace):
        sig = self.im.map(np.sum, inplace=inplace)
        if inplace:
            sig = self.im

        assert sig.axes_manager.signal_shape == ()
        assert sig.axes_manager.navigation_shape == (10,)
        assert sig.data.shape == (10,)

    @pytest.mark.parametrize("inplace", (True, False))
    def test_map(self, inplace):
        sig = self.im.map(gaussian_filter, inplace=inplace, sigma=2)
        if inplace:
            sig = self.im

        assert sig.axes_manager.signal_shape == (64, 64)
        assert sig.axes_manager.navigation_shape == (10,)
        assert sig.data.shape == (10, 64, 64)
