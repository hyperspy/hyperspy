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

import logging
from unittest import mock

import numpy as np
import pytest

from hyperspy.decorators import lazifyTestClass
from hyperspy.signals import Signal1D, Signal2D


def _verify_test_sum_x_E(self, s):
    np.testing.assert_array_equal(self.signal.data.sum(), s.data)
    assert s.data.ndim == 1
    # Check that there is still one signal axis.
    assert s.axes_manager.signal_dimension == 1


@lazifyTestClass
class Test2D:
    def setup_method(self, method):
        self.signal = Signal1D(np.arange(5 * 10).reshape(5, 10))  # dtype int
        self.signal.axes_manager[0].name = "x"
        self.signal.axes_manager[1].name = "E"
        self.signal.axes_manager[0].scale = 0.5
        self.data = self.signal.data.copy()

    def test_sum_x(self):
        s = self.signal.sum("x")
        np.testing.assert_array_equal(self.signal.data.sum(0), s.data)
        assert s.data.ndim == 1
        assert s.axes_manager.navigation_dimension == 0

    def test_sum_x_E(self):
        s = self.signal.sum(("x", "E"))
        _verify_test_sum_x_E(self, s)
        s = self.signal.sum((0, "E"))
        _verify_test_sum_x_E(self, s)
        s = self.signal.sum((self.signal.axes_manager[0], "E"))
        _verify_test_sum_x_E(self, s)
        s = self.signal.sum("x").sum("E")
        _verify_test_sum_x_E(self, s)

    def test_axis_by_str(self):
        m = mock.Mock()
        s1 = self.signal.deepcopy()
        s1.events.data_changed.connect(m.data_changed)
        s2 = self.signal.deepcopy()
        s1.crop(0, 2, 4)
        assert m.data_changed.called
        s2.crop("x", 2, 4)
        np.testing.assert_array_almost_equal(s1.data, s2.data)

    def test_crop_int(self):
        s = self.signal
        d = self.data
        s.crop(0, 2, 4)
        np.testing.assert_array_almost_equal(s.data, d[2:4, :])

    def test_crop_float(self):
        s = self.signal
        d = self.data
        s.crop(0, 2, 2.0)
        np.testing.assert_array_almost_equal(s.data, d[2:4, :])

    def test_crop_start_end_equal(self):
        s = self.signal
        with pytest.raises(ValueError):
            s.crop(0, 2, 2)
        with pytest.raises(ValueError):
            s.crop(0, 2.0, 2.0)

    def test_crop_float_no_unit_convertion_signal1D(self):
        # Should convert the unit to eV
        d = np.arange(5 * 10 * 2000).reshape(5, 10, 2000)
        s = Signal1D(d)
        s.axes_manager.signal_axes[0].name = "E"
        s.axes_manager.signal_axes[0].scale = 0.05
        s.axes_manager.signal_axes[0].units = "keV"
        s.crop("E", 0.0, 1.0, convert_units=False)
        np.testing.assert_almost_equal(s.axes_manager.signal_axes[0].scale, 0.05)
        assert s.axes_manager.signal_axes[0].units == "keV"
        np.testing.assert_allclose(s.data, d[:, :, :20])

        # Should keep the unit to keV
        s = Signal1D(d)
        s.axes_manager.signal_axes[0].name = "E"
        s.axes_manager.signal_axes[0].scale = 0.05
        s.axes_manager.signal_axes[0].units = "keV"
        s.crop("E", 0.0, 50.0, convert_units=False)
        np.testing.assert_almost_equal(s.axes_manager.signal_axes[0].scale, 0.05)
        assert s.axes_manager.signal_axes[0].units == "keV"
        np.testing.assert_allclose(s.data, d[:, :, :1000])

    def test_crop_float_unit_convertion_signal1D(self):
        # Should convert the unit to eV
        d = np.arange(5 * 10 * 2000).reshape(5, 10, 2000)
        s = Signal1D(d)
        s.axes_manager.signal_axes[0].name = "E"
        s.axes_manager.signal_axes[0].scale = 0.05
        s.axes_manager.signal_axes[0].units = "keV"
        s.crop("E", 0.0, 1.0, convert_units=True)
        np.testing.assert_almost_equal(s.axes_manager.signal_axes[0].scale, 50.0)
        assert s.axes_manager.signal_axes[0].units == "eV"
        np.testing.assert_allclose(s.data, d[:, :, :20])

        # Should keep the unit to keV
        s = Signal1D(d)
        s.axes_manager.signal_axes[0].name = "E"
        s.axes_manager.signal_axes[0].scale = 0.05
        s.axes_manager.signal_axes[0].units = "keV"
        s.crop("E", 0.0, 50.0, convert_units=True)
        np.testing.assert_almost_equal(s.axes_manager.signal_axes[0].scale, 0.05)
        assert s.axes_manager.signal_axes[0].units == "keV"
        np.testing.assert_allclose(s.data, d[:, :, :1000])

    def test_crop_float_no_unit_convertion_signal2D(self):
        # Should convert the unit to nm
        d = np.arange(512 * 512).reshape(512, 512)
        s = Signal2D(d)
        s.axes_manager[0].name = "x"
        s.axes_manager[0].scale = 0.01
        s.axes_manager[0].units = "µm"
        s.axes_manager[1].name = "y"
        s.axes_manager[1].scale = 0.01
        s.axes_manager[1].units = "µm"
        s.crop(0, 0.0, 0.5, convert_units=False)
        s.crop(1, 0.0, 0.5, convert_units=False)
        np.testing.assert_almost_equal(s.axes_manager[0].scale, 0.01)
        assert s.axes_manager[0].units == "µm"
        np.testing.assert_allclose(s.data, d[:50, :50])

        # Should keep the unit to µm
        d = np.arange(512 * 512).reshape(512, 512)
        s = Signal2D(d)
        s.axes_manager[0].name = "x"
        s.axes_manager[0].scale = 0.01
        s.axes_manager[0].units = "µm"
        s.axes_manager[1].name = "y"
        s.axes_manager[1].scale = 0.01
        s.axes_manager[1].units = "µm"
        s.crop(0, 0.0, 5.0, convert_units=False)
        s.crop(1, 0.0, 5.0, convert_units=False)
        np.testing.assert_almost_equal(s.axes_manager[0].scale, 0.01)
        assert s.axes_manager[0].units == "µm"
        np.testing.assert_allclose(s.data, d[:500, :500])

    def test_crop_float_unit_convertion_signal2D(self):
        # Should convert the unit to nm
        d = np.arange(512 * 512).reshape(512, 512)
        s = Signal2D(d)
        s.axes_manager[0].name = "x"
        s.axes_manager[0].scale = 0.01
        s.axes_manager[0].units = "µm"
        s.axes_manager[1].name = "y"
        s.axes_manager[1].scale = 0.01
        s.axes_manager[1].units = "µm"
        s.crop(0, 0.0, 0.5, convert_units=True)  # also convert the other axis
        s.crop(1, 0.0, 500.0, convert_units=True)
        np.testing.assert_almost_equal(s.axes_manager[0].scale, 10.0)
        np.testing.assert_almost_equal(s.axes_manager[1].scale, 10.0)
        assert s.axes_manager[0].units == "nm"
        assert s.axes_manager[1].units == "nm"
        np.testing.assert_allclose(s.data, d[:50, :50])

        # Should keep the unit to µm
        d = np.arange(512 * 512).reshape(512, 512)
        s = Signal2D(d)
        s.axes_manager[0].name = "x"
        s.axes_manager[0].scale = 0.01
        s.axes_manager[0].units = "µm"
        s.axes_manager[1].name = "y"
        s.axes_manager[1].scale = 0.01
        s.axes_manager[1].units = "µm"
        s.crop(0, 0.0, 5.0, convert_units=True)
        s.crop(1, 0.0, 5.0, convert_units=True)
        np.testing.assert_almost_equal(s.axes_manager[0].scale, 0.01)
        np.testing.assert_almost_equal(s.axes_manager[1].scale, 0.01)
        assert s.axes_manager[0].units == "µm"
        assert s.axes_manager[1].units == "µm"
        np.testing.assert_allclose(s.data, d[:500, :500])

    def test_crop_image_unit_convertion_signal2D(self):
        # Should not convert the unit
        d = np.arange(512 * 512).reshape(512, 512)
        s = Signal2D(d)
        s.axes_manager[0].name = "x"
        s.axes_manager[0].scale = 0.01
        s.axes_manager[0].units = "µm"
        s.axes_manager[1].name = "y"
        s.axes_manager[1].scale = 0.01
        s.axes_manager[1].units = "µm"
        s.crop_image(0, 0.5, 0.0, 0.5)
        np.testing.assert_almost_equal(s.axes_manager[0].scale, 0.01)
        np.testing.assert_almost_equal(s.axes_manager[1].scale, 0.01)
        assert s.axes_manager[0].units == "µm"
        assert s.axes_manager[1].units == "µm"
        np.testing.assert_allclose(s.data, d[:50, :50])

        # Should convert the unit to nm
        d = np.arange(512 * 512).reshape(512, 512)
        s = Signal2D(d)
        s.axes_manager[0].name = "x"
        s.axes_manager[0].scale = 0.01
        s.axes_manager[0].units = "µm"
        s.axes_manager[1].name = "y"
        s.axes_manager[1].scale = 0.01
        s.axes_manager[1].units = "µm"
        s.crop_image(0, 0.5, 0.0, 0.5, convert_units=True)
        np.testing.assert_almost_equal(s.axes_manager[0].scale, 10.0)
        np.testing.assert_almost_equal(s.axes_manager[1].scale, 10.0)
        assert s.axes_manager[0].units == "nm"
        assert s.axes_manager[1].units == "nm"
        np.testing.assert_allclose(s.data, d[:50, :50])

        # Should keep the unit to µm
        d = np.arange(512 * 512).reshape(512, 512)
        s = Signal2D(d)
        s.axes_manager[0].name = "x"
        s.axes_manager[0].scale = 0.01
        s.axes_manager[0].units = "µm"
        s.axes_manager[1].name = "y"
        s.axes_manager[1].scale = 0.01
        s.axes_manager[1].units = "µm"
        s.crop_image(0, 5.0, 0.0, 5.0, convert_units=True)
        np.testing.assert_almost_equal(s.axes_manager[0].scale, 0.01)
        np.testing.assert_almost_equal(s.axes_manager[1].scale, 0.01)
        assert s.axes_manager[0].units == "µm"
        assert s.axes_manager[1].units == "µm"
        np.testing.assert_allclose(s.data, d[:500, :500])

    def test_split_axis0(self):
        result = self.signal.split(0, 2)
        assert len(result) == 2
        np.testing.assert_array_almost_equal(result[0].data, self.data[:2, :])
        np.testing.assert_array_almost_equal(result[1].data, self.data[2:4, :])

    def test_split_axis1(self):
        result = self.signal.split(1, 2)
        assert len(result) == 2
        np.testing.assert_array_almost_equal(result[0].data, self.data[:, :5])
        np.testing.assert_array_almost_equal(result[1].data, self.data[:, 5:])

    def test_split_axisE(self):
        result = self.signal.split("E", 2)
        assert len(result) == 2
        np.testing.assert_array_almost_equal(result[0].data, self.data[:, :5])
        np.testing.assert_array_almost_equal(result[1].data, self.data[:, 5:])

    def test_split_default(self):
        result = self.signal.split()
        assert len(result) == 5
        np.testing.assert_array_almost_equal(result[0].data, self.data[0])

    def test_split_step_size_list(self):
        result = self.signal.split(step_sizes=[1, 2])
        assert len(result) == 2
        np.testing.assert_array_almost_equal(result[0].data, self.data[:1, :10])
        np.testing.assert_array_almost_equal(result[1].data, self.data[1:3, :10])

    def test_split_error(self):
        with pytest.raises(
            ValueError,
            match="You can define step_sizes or number_of_parts but not both",
        ):
            _ = self.signal.split(number_of_parts=2, step_sizes=2)

        with pytest.raises(
            ValueError, match="The number of parts is greater than the axis size.",
        ):
            _ = self.signal.split(number_of_parts=1e9)

    def test_histogram(self):
        result = self.signal.get_histogram(3)
        assert isinstance(result, Signal1D)
        np.testing.assert_array_equal(result.data, np.array([17, 16, 17]))
        assert result.metadata.Signal.binned

    def test_noise_variance_helpers(self):
        assert self.signal.get_noise_variance() is None
        self.signal.set_noise_variance(2)
        assert self.signal.get_noise_variance() == 2
        self.signal.set_noise_variance(self.signal)
        variance = self.signal.get_noise_variance()
        np.testing.assert_array_equal(variance.data, self.signal.data)
        self.signal.set_noise_variance(None)
        assert self.signal.get_noise_variance() is None

        with pytest.raises(ValueError, match="`variance` must be one of"):
            self.signal.set_noise_variance(np.array([0, 1, 2]))

    def test_estimate_poissonian_noise_copy_data(self):
        self.signal.estimate_poissonian_noise_variance()
        variance = self.signal.metadata.Signal.Noise_properties.variance
        assert variance.data is not self.signal.data

    def test_estimate_poissonian_noise_copy_data_helper_function(self):
        self.signal.estimate_poissonian_noise_variance()
        variance = self.signal.get_noise_variance()
        assert variance.data is not self.signal.data

    def test_estimate_poissonian_noise_noarg(self):
        self.signal.estimate_poissonian_noise_variance()
        variance = self.signal.metadata.Signal.Noise_properties.variance
        np.testing.assert_array_equal(variance.data, self.signal.data)

    def test_estimate_poissonian_noise_with_args(self):
        self.signal.estimate_poissonian_noise_variance(
            expected_value=self.signal,
            gain_factor=2,
            gain_offset=1,
            correlation_factor=0.5,
        )
        variance = self.signal.metadata.Signal.Noise_properties.variance
        np.testing.assert_array_equal(variance.data, (self.signal.data * 2 + 1) * 0.5)

    def test_unfold_image(self):
        s = self.signal
        if s._lazy:
            pytest.skip("LazySignals do not support folding")
        s = s.transpose(signal_axes=2)
        s.unfold()
        assert s.data.shape == (50,)

    def test_unfold_image_returns_true(self):
        s = self.signal
        if s._lazy:
            pytest.skip("LazySignals do not support folding")
        s = s.transpose(signal_axes=2)
        assert s.unfold()

    def test_print_summary(self):
        # Just test if it doesn't raise an exception
        self.signal._print_summary()

    def test_print_summary_statistics(self):
        # Just test if it doesn't raise an exception
        self.signal.print_summary_statistics()
        if self.signal._lazy:
            self.signal.print_summary_statistics(rechunk=False)

    def test_numpy_unfunc_one_arg_titled(self):
        self.signal.metadata.General.title = "yes"
        result = np.exp(self.signal)
        assert isinstance(result, Signal1D)
        np.testing.assert_array_equal(result.data, np.exp(self.signal.data))
        assert result.metadata.General.title == "exp(yes)"

    def test_numpy_unfunc_one_arg_untitled(self):
        result = np.exp(self.signal)
        assert result.metadata.General.title == "exp(Untitled Signal 1)"

    def test_numpy_unfunc_two_arg_titled(self):
        s1, s2 = self.signal.deepcopy(), self.signal.deepcopy()
        s1.metadata.General.title = "A"
        s2.metadata.General.title = "B"
        result = np.add(s1, s2)
        assert isinstance(result, Signal1D)
        np.testing.assert_array_equal(result.data, np.add(s1.data, s2.data))
        assert result.metadata.General.title == "add(A, B)"

    def test_numpy_unfunc_two_arg_untitled(self):
        s1, s2 = self.signal.deepcopy(), self.signal.deepcopy()
        result = np.add(s1, s2)
        assert (
            result.metadata.General.title == "add(Untitled Signal 1, Untitled Signal 2)"
        )

    def test_numpy_func(self):
        result = np.angle(self.signal)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.angle(self.signal.data))

    def test_add_gaussian_noise(self):
        s = self.signal
        s.change_dtype("float64")
        kwargs = {}
        if s._lazy:
            data = s.data.compute()
            from dask.array.random import seed, normal

            kwargs["chunks"] = s.data.chunks
        else:
            data = s.data.copy()
            from numpy.random import seed, normal
        seed(1)
        s.add_gaussian_noise(std=1.0)
        seed(1)
        if s._lazy:
            s.compute()
        np.testing.assert_array_almost_equal(
            s.data - data, normal(scale=1.0, size=data.shape, **kwargs)
        )

    def test_add_gaussian_noise_seed(self):
        s = self.signal
        s.change_dtype("float64")
        kwargs = {}
        if s._lazy:
            data = s.data.compute()
            from dask.array.random import RandomState

            kwargs["chunks"] = s.data.chunks
            rng1 = RandomState(123)
            rng2 = RandomState(123)
        else:
            data = s.data.copy()
            rng1 = np.random.RandomState(123)
            rng2 = np.random.RandomState(123)

        s.add_gaussian_noise(std=1.0, random_state=rng1)
        if s._lazy:
            s.compute()

        np.testing.assert_array_almost_equal(
            s.data - data, rng2.normal(scale=1.0, size=data.shape, **kwargs)
        )

    def test_gaussian_noise_error(self):
        s = self.signal
        s.change_dtype("int64")
        with pytest.raises(TypeError, match="float datatype"):
            s.add_gaussian_noise(std=1.0)

    def test_add_poisson_noise(self):
        s = self.signal
        kwargs = {}
        if s._lazy:
            data = s.data.compute()
            from dask.array.random import seed, poisson

            kwargs["chunks"] = s.data.chunks
        else:
            data = s.data.copy()
            from numpy.random import seed, poisson
        seed(1)

        s.add_poissonian_noise(keep_dtype=False)

        if s._lazy:
            s.compute()
        seed(1)
        np.testing.assert_array_almost_equal(s.data, poisson(lam=data, **kwargs))
        s.change_dtype("float64")
        seed(1)

        s.add_poissonian_noise(keep_dtype=True)
        if s._lazy:
            s.compute()
        assert s.data.dtype == np.dtype("float64")

    def test_add_poisson_noise_seed(self):
        s = self.signal
        kwargs = {}
        if s._lazy:
            data = s.data.compute()
            from dask.array.random import RandomState

            kwargs["chunks"] = s.data.chunks
            rng1 = RandomState(123)
            rng2 = RandomState(123)
        else:
            data = s.data.copy()
            rng1 = np.random.RandomState(123)
            rng2 = np.random.RandomState(123)

        s.add_poissonian_noise(keep_dtype=False, random_state=rng1)

        if s._lazy:
            s.compute()

        np.testing.assert_array_almost_equal(s.data, rng2.poisson(lam=data, **kwargs))
        s.change_dtype("float64")

        s.add_poissonian_noise(keep_dtype=True, random_state=rng1)
        if s._lazy:
            s.compute()

        assert s.data.dtype == np.dtype("float64")

    def test_add_poisson_noise_warning(self, caplog):
        s = self.signal
        s.change_dtype("float64")

        with caplog.at_level(logging.WARNING):
            s.add_poissonian_noise(keep_dtype=True)

        assert "Changing data type from" in caplog.text

        with caplog.at_level(logging.WARNING):
            s.add_poissonian_noise(keep_dtype=False)

        assert "The data type changed from" in caplog.text
