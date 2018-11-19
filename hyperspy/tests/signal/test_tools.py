from unittest import mock
import sys

import numpy as np
import dask.array as da
from numpy.testing import assert_array_equal, assert_almost_equal
import pytest
import numpy.testing as nt

from hyperspy import signals
from hyperspy.decorators import lazifyTestClass
from hyperspy.signal_tools import SpikesRemoval
from hyperspy.components1d import Gaussian


def _verify_test_sum_x_E(self, s):
    nt.assert_array_equal(self.signal.data.sum(), s.data)
    assert s.data.ndim == 1
    # Check that there is still one signal axis.
    assert s.axes_manager.signal_dimension == 1


@lazifyTestClass
class Test1D:

    def setup_method(self, method):
        gaussian = Gaussian()
        gaussian.A.value = 20
        gaussian.sigma.value = 10
        gaussian.centre.value = 50
        self.signal = signals.Signal1D(
            gaussian.function(np.arange(0, 100, 0.01)))
        self.signal.axes_manager[0].scale = 0.01

    def test_integrate1D(self):
        integrated_signal = self.signal.integrate1D(axis=0)
        assert np.allclose(integrated_signal.data, 20,)


@lazifyTestClass
class Test2D:

    def setup_method(self, method):
        self.signal = signals.Signal1D(
            np.arange(
                5 *
                10).reshape(
                5,
                10))  # dtype int
        self.signal.axes_manager[0].name = "x"
        self.signal.axes_manager[1].name = "E"
        self.signal.axes_manager[0].scale = 0.5
        self.data = self.signal.data.copy()

    def test_sum_x(self):
        s = self.signal.sum("x")
        nt.assert_array_equal(self.signal.data.sum(0), s.data)
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
        nt.assert_array_almost_equal(s1.data, s2.data)

    def test_crop_int(self):
        s = self.signal
        d = self.data
        s.crop(0, 2, 4)
        nt.assert_array_almost_equal(s.data, d[2:4, :])

    def test_crop_float(self):
        s = self.signal
        d = self.data
        s.crop(0, 2, 2.)
        nt.assert_array_almost_equal(s.data, d[2:4, :])

    def test_crop_float_no_unit_convertion_signal1D(self):
        # Should convert the unit to eV
        d = np.arange(5 * 10 * 2000).reshape(5, 10, 2000)
        s = signals.Signal1D(d)
        s.axes_manager.signal_axes[0].name = "E"
        s.axes_manager.signal_axes[0].scale = 0.05
        s.axes_manager.signal_axes[0].units = "keV"
        s.crop('E', 0.0, 1.0, convert_units=False)
        nt.assert_almost_equal(s.axes_manager.signal_axes[0].scale, 0.05)
        assert s.axes_manager.signal_axes[0].units == "keV"
        nt.assert_allclose(s.data, d[:, :, :20])

        # Should keep the unit to keV
        s = signals.Signal1D(d)
        s.axes_manager.signal_axes[0].name = "E"
        s.axes_manager.signal_axes[0].scale = 0.05
        s.axes_manager.signal_axes[0].units = "keV"
        s.crop('E', 0.0, 50.0, convert_units=False)
        nt.assert_almost_equal(s.axes_manager.signal_axes[0].scale, 0.05)
        assert s.axes_manager.signal_axes[0].units == "keV"
        nt.assert_allclose(s.data, d[:, :, :1000])

    def test_crop_float_unit_convertion_signal1D(self):
        # Should convert the unit to eV
        d = np.arange(5 * 10 * 2000).reshape(5, 10, 2000)
        s = signals.Signal1D(d)
        s.axes_manager.signal_axes[0].name = "E"
        s.axes_manager.signal_axes[0].scale = 0.05
        s.axes_manager.signal_axes[0].units = "keV"
        s.crop('E', 0.0, 1.0, convert_units=True)
        nt.assert_almost_equal(s.axes_manager.signal_axes[0].scale, 50.0)
        assert s.axes_manager.signal_axes[0].units == "eV"
        nt.assert_allclose(s.data, d[:, :, :20])

        # Should keep the unit to keV
        s = signals.Signal1D(d)
        s.axes_manager.signal_axes[0].name = "E"
        s.axes_manager.signal_axes[0].scale = 0.05
        s.axes_manager.signal_axes[0].units = "keV"
        s.crop('E', 0.0, 50.0, convert_units=True)
        nt.assert_almost_equal(s.axes_manager.signal_axes[0].scale, 0.05)
        assert s.axes_manager.signal_axes[0].units == "keV"
        nt.assert_allclose(s.data, d[:, :, :1000])

    def test_crop_float_no_unit_convertion_signal2D(self):
        # Should convert the unit to nm
        d = np.arange(512 * 512).reshape(512, 512)
        s = signals.Signal2D(d)
        s.axes_manager[0].name = 'x'
        s.axes_manager[0].scale = 0.01
        s.axes_manager[0].units = 'µm'
        s.axes_manager[1].name = 'y'
        s.axes_manager[1].scale = 0.01
        s.axes_manager[1].units = 'µm'
        s.crop(0, 0.0, 0.5, convert_units=False)
        s.crop(1, 0.0, 0.5, convert_units=False)
        nt.assert_almost_equal(s.axes_manager[0].scale, 0.01)
        assert s.axes_manager[0].units == "µm"
        nt.assert_allclose(s.data, d[:50, :50])

        # Should keep the unit to µm
        d = np.arange(512 * 512).reshape(512, 512)
        s = signals.Signal2D(d)
        s.axes_manager[0].name = 'x'
        s.axes_manager[0].scale = 0.01
        s.axes_manager[0].units = 'µm'
        s.axes_manager[1].name = 'y'
        s.axes_manager[1].scale = 0.01
        s.axes_manager[1].units = 'µm'
        s.crop(0, 0.0, 5.0, convert_units=False)
        s.crop(1, 0.0, 5.0, convert_units=False)
        nt.assert_almost_equal(s.axes_manager[0].scale, 0.01)
        assert s.axes_manager[0].units == "µm"
        nt.assert_allclose(s.data, d[:500, :500])

    def test_crop_float_unit_convertion_signal2D(self):
        # Should convert the unit to nm
        d = np.arange(512 * 512).reshape(512, 512)
        s = signals.Signal2D(d)
        s.axes_manager[0].name = 'x'
        s.axes_manager[0].scale = 0.01
        s.axes_manager[0].units = 'µm'
        s.axes_manager[1].name = 'y'
        s.axes_manager[1].scale = 0.01
        s.axes_manager[1].units = 'µm'
        s.crop(0, 0.0, 0.5, convert_units=True)  # also convert the other axis
        s.crop(1, 0.0, 500.0, convert_units=True)
        nt.assert_almost_equal(s.axes_manager[0].scale, 10.0)
        nt.assert_almost_equal(s.axes_manager[1].scale, 10.0)
        assert s.axes_manager[0].units == 'nm'
        assert s.axes_manager[1].units == 'nm'
        nt.assert_allclose(s.data, d[:50, :50])

        # Should keep the unit to µm
        d = np.arange(512 * 512).reshape(512, 512)
        s = signals.Signal2D(d)
        s.axes_manager[0].name = 'x'
        s.axes_manager[0].scale = 0.01
        s.axes_manager[0].units = 'µm'
        s.axes_manager[1].name = 'y'
        s.axes_manager[1].scale = 0.01
        s.axes_manager[1].units = 'µm'
        s.crop(0, 0.0, 5.0, convert_units=True)
        s.crop(1, 0.0, 5.0, convert_units=True)
        nt.assert_almost_equal(s.axes_manager[0].scale, 0.01)
        nt.assert_almost_equal(s.axes_manager[1].scale, 0.01)
        assert s.axes_manager[0].units == "um"
        assert s.axes_manager[1].units == "um"
        nt.assert_allclose(s.data, d[:500, :500])

    def test_crop_image_unit_convertion_signal2D(self):
        # Should not convert the unit
        d = np.arange(512 * 512).reshape(512, 512)
        s = signals.Signal2D(d)
        s.axes_manager[0].name = 'x'
        s.axes_manager[0].scale = 0.01
        s.axes_manager[0].units = 'µm'
        s.axes_manager[1].name = 'y'
        s.axes_manager[1].scale = 0.01
        s.axes_manager[1].units = 'µm'
        s.crop_image(0, 0.5, 0.0, 0.5)
        nt.assert_almost_equal(s.axes_manager[0].scale, 0.01)
        nt.assert_almost_equal(s.axes_manager[1].scale, 0.01)
        assert s.axes_manager[0].units == 'µm'
        assert s.axes_manager[1].units == 'µm'
        nt.assert_allclose(s.data, d[:50, :50])

        # Should convert the unit to nm
        d = np.arange(512 * 512).reshape(512, 512)
        s = signals.Signal2D(d)
        s.axes_manager[0].name = 'x'
        s.axes_manager[0].scale = 0.01
        s.axes_manager[0].units = 'µm'
        s.axes_manager[1].name = 'y'
        s.axes_manager[1].scale = 0.01
        s.axes_manager[1].units = 'µm'
        s.crop_image(0, 0.5, 0.0, 0.5, convert_units=True)
        nt.assert_almost_equal(s.axes_manager[0].scale, 10.0)
        nt.assert_almost_equal(s.axes_manager[1].scale, 10.0)
        assert s.axes_manager[0].units == 'nm'
        assert s.axes_manager[1].units == 'nm'
        nt.assert_allclose(s.data, d[:50, :50])

        # Should keep the unit to µm
        d = np.arange(512 * 512).reshape(512, 512)
        s = signals.Signal2D(d)
        s.axes_manager[0].name = 'x'
        s.axes_manager[0].scale = 0.01
        s.axes_manager[0].units = 'µm'
        s.axes_manager[1].name = 'y'
        s.axes_manager[1].scale = 0.01
        s.axes_manager[1].units = 'µm'
        s.crop_image(0, 5.0, 0.0, 5.0, convert_units=True)
        nt.assert_almost_equal(s.axes_manager[0].scale, 0.01)
        nt.assert_almost_equal(s.axes_manager[1].scale, 0.01)
        assert s.axes_manager[0].units == "um"
        assert s.axes_manager[1].units == "um"
        nt.assert_allclose(s.data, d[:500, :500])

    def test_split_axis0(self):
        result = self.signal.split(0, 2)
        assert len(result) == 2
        nt.assert_array_almost_equal(result[0].data, self.data[:2, :])
        nt.assert_array_almost_equal(result[1].data, self.data[2:4, :])

    def test_split_axis1(self):
        result = self.signal.split(1, 2)
        assert len(result) == 2
        nt.assert_array_almost_equal(result[0].data, self.data[:, :5])
        nt.assert_array_almost_equal(result[1].data, self.data[:, 5:])

    def test_split_axisE(self):
        result = self.signal.split("E", 2)
        assert len(result) == 2
        nt.assert_array_almost_equal(result[0].data, self.data[:, :5])
        nt.assert_array_almost_equal(result[1].data, self.data[:, 5:])

    def test_split_default(self):
        result = self.signal.split()
        assert len(result) == 5
        nt.assert_array_almost_equal(result[0].data, self.data[0])

    def test_histogram(self):
        result = self.signal.get_histogram(3)
        assert isinstance(result, signals.Signal1D)
        nt.assert_array_equal(result.data, np.array([17, 16, 17]))
        assert result.metadata.Signal.binned

    def test_estimate_poissonian_noise_copy_data(self):
        self.signal.estimate_poissonian_noise_variance()
        variance = self.signal.metadata.Signal.Noise_properties.variance
        assert variance.data is not self.signal.data

    def test_estimate_poissonian_noise_noarg(self):
        self.signal.estimate_poissonian_noise_variance()
        variance = self.signal.metadata.Signal.Noise_properties.variance
        nt.assert_array_equal(variance.data, self.signal.data)

    def test_estimate_poissonian_noise_with_args(self):
        self.signal.estimate_poissonian_noise_variance(
            expected_value=self.signal,
            gain_factor=2,
            gain_offset=1,
            correlation_factor=0.5)
        variance = self.signal.metadata.Signal.Noise_properties.variance
        nt.assert_array_equal(variance.data,
                              (self.signal.data * 2 + 1) * 0.5)

    def test_unfold_image(self):
        s = self.signal
        if s._lazy:
            pytest.skip("LazyS do not support folding")
        s = s.transpose(signal_axes=2)
        s.unfold()
        assert s.data.shape == (50,)

    def test_unfold_image_returns_true(self):
        s = self.signal
        if s._lazy:
            pytest.skip("LazyS do not support folding")
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
        assert isinstance(result, signals.Signal1D)
        nt.assert_array_equal(result.data, np.exp(self.signal.data))
        assert result.metadata.General.title == "exp(yes)"

    def test_numpy_unfunc_one_arg_untitled(self):
        result = np.exp(self.signal)
        assert (result.metadata.General.title ==
                "exp(Untitled Signal 1)")

    def test_numpy_unfunc_two_arg_titled(self):
        s1, s2 = self.signal.deepcopy(), self.signal.deepcopy()
        s1.metadata.General.title = "A"
        s2.metadata.General.title = "B"
        result = np.add(s1, s2)
        assert isinstance(result, signals.Signal1D)
        nt.assert_array_equal(result.data, np.add(s1.data, s2.data))
        assert result.metadata.General.title == "add(A, B)"

    def test_numpy_unfunc_two_arg_untitled(self):
        s1, s2 = self.signal.deepcopy(), self.signal.deepcopy()
        result = np.add(s1, s2)
        assert (result.metadata.General.title ==
                "add(Untitled Signal 1, Untitled Signal 2)")

    def test_numpy_func(self):
        result = np.angle(self.signal)
        assert isinstance(result, np.ndarray)
        nt.assert_array_equal(result, np.angle(self.signal.data))

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
            s.data - data, normal(scale=1.0, size=data.shape, **kwargs))

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
        np.testing.assert_array_almost_equal(
            s.data, poisson(lam=data, **kwargs))
        s.change_dtype("float64")
        seed(1)
        s.add_poissonian_noise(keep_dtype=True)
        if s._lazy:
            s.compute()
        assert s.data.dtype == np.dtype("float64")


def _test_default_navigation_signal_operations_over_many_axes(self, op):
    s = getattr(self.signal, op)()
    ar = getattr(self.data, op)(axis=(0, 1))
    nt.assert_array_equal(ar, s.data)
    assert s.data.ndim == 1
    assert s.axes_manager.signal_dimension == 1
    assert s.axes_manager.navigation_dimension == 0


@lazifyTestClass
class Test3D:

    def setup_method(self, method):
        self.signal = signals.Signal1D(np.arange(2 * 4 * 6).reshape(2, 4, 6))
        self.signal.axes_manager[0].name = "x"
        self.signal.axes_manager[1].name = "y"
        self.signal.axes_manager[2].name = "E"
        self.signal.axes_manager[0].scale = 0.5
        self.data = self.signal.data.copy()

    def test_indexmin(self):
        s = self.signal.indexmin('E')
        ar = self.data.argmin(2)
        nt.assert_array_equal(ar, s.data)
        assert s.data.ndim == 2
        assert s.axes_manager.signal_dimension == 0
        assert s.axes_manager.navigation_dimension == 2

    def test_indexmax(self):
        s = self.signal.indexmax('E')
        ar = self.data.argmax(2)
        nt.assert_array_equal(ar, s.data)
        assert s.data.ndim == 2
        assert s.axes_manager.signal_dimension == 0
        assert s.axes_manager.navigation_dimension == 2

    def test_valuemin(self):
        s = self.signal.valuemin('x')
        ar = self.signal.axes_manager['x'].index2value(self.data.argmin(1))
        nt.assert_array_equal(ar, s.data)
        assert s.data.ndim == 2
        assert s.axes_manager.signal_dimension == 1
        assert s.axes_manager.navigation_dimension == 1

    def test_valuemax(self):
        s = self.signal.valuemax('x')
        ar = self.signal.axes_manager['x'].index2value(self.data.argmax(1))
        nt.assert_array_equal(ar, s.data)
        assert s.data.ndim == 2
        assert s.axes_manager.signal_dimension == 1
        assert s.axes_manager.navigation_dimension == 1

    def test_default_navigation_sum(self):
        _test_default_navigation_signal_operations_over_many_axes(self, 'sum')

    def test_default_navigation_max(self):
        _test_default_navigation_signal_operations_over_many_axes(self, 'max')

    def test_default_navigation_min(self):
        _test_default_navigation_signal_operations_over_many_axes(self, 'min')

    def test_default_navigation_mean(self):
        _test_default_navigation_signal_operations_over_many_axes(self, 'mean')

    def test_default_navigation_std(self):
        _test_default_navigation_signal_operations_over_many_axes(self, 'std')

    def test_default_navigation_var(self):
        _test_default_navigation_signal_operations_over_many_axes(self, 'var')

    def test_rebin(self):
        self.signal.estimate_poissonian_noise_variance()
        new_s = self.signal.rebin(scale=(2, 2, 1))
        var = new_s.metadata.Signal.Noise_properties.variance
        assert new_s.data.shape == (1, 2, 6)
        assert var.data.shape == (1, 2, 6)
        from hyperspy.misc.array_tools import rebin

        np.testing.assert_array_equal(rebin(self.signal.data, scale=(2, 2, 1)),
                                      var.data)
        np.testing.assert_array_equal(rebin(self.signal.data, scale=(2, 2, 1)),
                                      new_s.data)
        if self.signal._lazy:
            new_s = self.signal.rebin(scale=(2, 2, 1), rechunk=False)
            np.testing.assert_array_equal(rebin(self.signal.data, scale=(2, 2, 1)),
                                          var.data)
            np.testing.assert_array_equal(rebin(self.signal.data, scale=(2, 2, 1)),
                                          new_s.data)

    def test_rebin_no_variance(self):
        new_s = self.signal.rebin(scale=(2, 2, 1))
        with pytest.raises(AttributeError):
            _ = new_s.metadata.Signal.Noise_properties

    def test_rebin_const_variance(self):
        self.signal.metadata.set_item(
            'Signal.Noise_properties.variance', 0.3)
        new_s = self.signal.rebin(scale=(2, 2, 1))
        assert new_s.metadata.Signal.Noise_properties.variance == 0.3

    def test_rebin_dtype(self):
        s = signals.Signal1D(np.arange(1000).reshape(10, 10, 10))
        s.change_dtype(np.uint8)
        s2 = s.rebin(scale=(3, 3, 1), crop=False)
        assert s.sum() == s2.sum()

    def test_swap_axes_simple(self):
        s = self.signal
        if s._lazy:
            chunks = s.data.chunks
        assert s.swap_axes(0, 1).data.shape == (4, 2, 6)
        assert s.swap_axes(0, 2).axes_manager.shape == (6, 2, 4)
        if not s._lazy:
            assert not s.swap_axes(0, 2).data.flags['C_CONTIGUOUS']
            assert s.swap_axes(0, 2, optimize=True).data.flags['C_CONTIGUOUS']
        else:
            cks = s.data.chunks
            assert s.swap_axes(0, 1).data.chunks == (cks[1], cks[0], cks[2])
            # This data shape does not require rechunking
            assert s.swap_axes(
                0, 1, optimize=True).data.chunks == (
                cks[1], cks[0], cks[2])

    def test_swap_axes_iteration(self):
        s = self.signal
        s = s.swap_axes(0, 2)
        assert s.axes_manager._getitem_tuple[:2] == (0, 0)
        s.axes_manager.indices = (2, 1)
        assert s.axes_manager._getitem_tuple[:2] == (1, 2)

    def test_get_navigation_signal_nav_dim0(self):
        s = self.signal
        s = s.transpose(signal_axes=3)
        ns = s._get_navigation_signal()
        assert ns.axes_manager.signal_dimension == 1
        assert ns.axes_manager.signal_size == 1
        assert ns.axes_manager.navigation_dimension == 0

    def test_get_navigation_signal_nav_dim1(self):
        s = self.signal
        s = s.transpose(signal_axes=2)
        ns = s._get_navigation_signal()
        assert (ns.axes_manager.signal_shape ==
                s.axes_manager.navigation_shape)
        assert ns.axes_manager.navigation_dimension == 0

    def test_get_navigation_signal_nav_dim2(self):
        s = self.signal
        s = s.transpose(signal_axes=1)
        ns = s._get_navigation_signal()
        assert (ns.axes_manager.signal_shape ==
                s.axes_manager.navigation_shape)
        assert ns.axes_manager.navigation_dimension == 0

    def test_get_navigation_signal_nav_dim3(self):
        s = self.signal
        s = s.transpose(signal_axes=0)
        ns = s._get_navigation_signal()
        assert (ns.axes_manager.signal_shape ==
                s.axes_manager.navigation_shape)
        assert ns.axes_manager.navigation_dimension == 0

    def test_get_navigation_signal_wrong_data_shape(self):
        s = self.signal
        s = s.transpose(signal_axes=1)
        with pytest.raises(ValueError):
            s._get_navigation_signal(data=np.zeros((3, 2)))

    def test_get_navigation_signal_wrong_data_shape_dim0(self):
        s = self.signal
        s = s.transpose(signal_axes=3)
        with pytest.raises(ValueError):
            s._get_navigation_signal(data=np.asarray(0))

    def test_get_navigation_signal_given_data(self):
        s = self.signal
        s = s.transpose(signal_axes=1)
        data = np.zeros(s.axes_manager._navigation_shape_in_array)
        ns = s._get_navigation_signal(data=data)
        assert ns.data is data

    def test_get_signal_signal_nav_dim0(self):
        s = self.signal
        s = s.transpose(signal_axes=0)
        ns = s._get_signal_signal()
        assert ns.axes_manager.navigation_dimension == 0
        assert ns.axes_manager.navigation_size == 0
        assert ns.axes_manager.signal_dimension == 1

    def test_get_signal_signal_nav_dim1(self):
        s = self.signal
        s = s.transpose(signal_axes=1)
        ns = s._get_signal_signal()
        assert (ns.axes_manager.signal_shape ==
                s.axes_manager.signal_shape)
        assert ns.axes_manager.navigation_dimension == 0

    def test_get_signal_signal_nav_dim2(self):
        s = self.signal
        s = s.transpose(signal_axes=2)
        s._assign_subclass()
        ns = s._get_signal_signal()
        assert (ns.axes_manager.signal_shape ==
                s.axes_manager.signal_shape)
        assert ns.axes_manager.navigation_dimension == 0

    def test_get_signal_signal_nav_dim3(self):
        s = self.signal
        s = s.transpose(signal_axes=3)
        s._assign_subclass()
        ns = s._get_signal_signal()
        assert (ns.axes_manager.signal_shape ==
                s.axes_manager.signal_shape)
        assert ns.axes_manager.navigation_dimension == 0

    def test_get_signal_signal_wrong_data_shape(self):
        s = self.signal
        s = s.transpose(signal_axes=1)
        with pytest.raises(ValueError):
            s._get_signal_signal(data=np.zeros((3, 2)))

    def test_get_signal_signal_wrong_data_shape_dim0(self):
        s = self.signal
        s = s.transpose(signal_axes=0)
        with pytest.raises(ValueError):
            s._get_signal_signal(data=np.asarray(0))

    def test_get_signal_signal_given_data(self):
        s = self.signal
        s = s.transpose(signal_axes=2)
        data = np.zeros(s.axes_manager._signal_shape_in_array)
        ns = s._get_signal_signal(data=data)
        assert ns.data is data

    def test_get_navigation_signal_dtype(self):
        s = self.signal
        assert (s._get_navigation_signal().data.dtype.name ==
                s.data.dtype.name)

    def test_get_signal_signal_dtype(self):
        s = self.signal
        assert (s._get_signal_signal().data.dtype.name ==
                s.data.dtype.name)

    def test_get_navigation_signal_given_dtype(self):
        s = self.signal
        assert (
            s._get_navigation_signal(dtype="bool").data.dtype.name == "bool")

    def test_get_signal_signal_given_dtype(self):
        s = self.signal
        assert (
            s._get_signal_signal(dtype="bool").data.dtype.name == "bool")


@lazifyTestClass
class Test4D:

    def setup_method(self, method):
        s = signals.Signal1D(np.ones((5, 4, 3, 6)))
        for axis, name in zip(
                s.axes_manager._get_axes_in_natural_order(),
                ['x', 'y', 'z', 'E']):
            axis.name = name
        self.s = s

    def test_diff_data(self):
        s = self.s
        diff = s.diff(axis=2, order=2)
        diff_data = np.diff(s.data, n=2, axis=0)
        nt.assert_array_equal(diff.data, diff_data)

    def test_diff_axis(self):
        s = self.s
        diff = s.diff(axis=2, order=2)
        assert (
            diff.axes_manager[2].offset ==
            s.axes_manager[2].offset + s.axes_manager[2].scale)

    def test_rollaxis_int(self):
        assert self.s.rollaxis(2, 0).data.shape == (4, 3, 5, 6)

    def test_rollaxis_str(self):
        assert self.s.rollaxis("z", "x").data.shape == (4, 3, 5, 6)

    def test_unfold_spectrum(self):
        self.s.unfold()
        assert self.s.data.shape == (60, 6)

    def test_unfold_spectrum_returns_true(self):
        assert self.s.unfold()

    def test_unfold_spectrum_signal_returns_false(self):
        assert not self.s.unfold_signal_space()

    def test_unfold_image(self):
        im = self.s.to_signal2D()
        im.unfold()
        assert im.data.shape == (30, 12)

    def test_image_signal_unfolded_deepcopy(self):
        im = self.s.to_signal2D()
        im.unfold()
        # The following could fail if the constructor was not taking the fact
        # that the signal is unfolded into account when setting the signal
        # dimension.
        im.deepcopy()

    def test_image_signal_unfolded_false(self):
        im = self.s.to_signal2D()
        assert not im.metadata._HyperSpy.Folding.signal_unfolded

    def test_image_signal_unfolded_true(self):
        im = self.s.to_signal2D()
        im.unfold()
        assert im.metadata._HyperSpy.Folding.signal_unfolded

    def test_image_signal_unfolded_back_to_false(self):
        im = self.s.to_signal2D()
        im.unfold()
        im.fold()
        assert not im.metadata._HyperSpy.Folding.signal_unfolded


def test_signal_iterator():
    sig = signals.Signal1D(np.arange(3).reshape((3, 1)))
    for s in (sig, sig.as_lazy()):
        assert next(s).data[0] == 0
        # If the following fails it can be because the iteration index was not
        # restarted
        for i, signal in enumerate(s):
            assert i == signal.data[0]


@lazifyTestClass
class TestDerivative:

    def setup_method(self, method):
        offset = 3
        scale = 0.1
        x = np.arange(-offset, offset, scale)
        s = signals.Signal1D(np.sin(x))
        s.axes_manager[0].offset = x[0]
        s.axes_manager[0].scale = scale
        self.s = s

    def test_derivative_data(self):
        der = self.s.derivative(axis=0, order=4)
        nt.assert_allclose(der.data, np.sin(
            der.axes_manager[0].axis), atol=1e-2)


@lazifyTestClass
class TestOutArg:

    def setup_method(self, method):
        # Some test require consistent random data for reference to be correct
        np.random.seed(0)
        s = signals.Signal1D(np.random.rand(5, 4, 3, 6))
        for axis, name in zip(
                s.axes_manager._get_axes_in_natural_order(),
                ['x', 'y', 'z', 'E']):
            axis.name = name
        self.s = s

    def _run_single(self, f, s, kwargs):
        m = mock.Mock()
        s1 = f(**kwargs)
        s1.events.data_changed.connect(m.data_changed)
        s.data = s.data + 2
        s2 = f(**kwargs)
        r = f(out=s1, **kwargs)
        m.data_changed.assert_called_with(obj=s1)
        assert r is None
        assert_array_equal(s1.data, s2.data)

    def test_get_histogram(self):
        self._run_single(self.s.get_histogram, self.s, {})
        if self.s._lazy:
            self._run_single(self.s.get_histogram, self.s, {"rechunk": False})

    def test_sum(self):
        self._run_single(self.s.sum, self.s, dict(axis=('x', 'z')))
        self._run_single(self.s.sum, self.s.get_current_signal(),
                         dict(axis=0))

    def test_sum_return_1d_signal(self):
        self._run_single(self.s.sum, self.s, dict(
            axis=self.s.axes_manager._axes))
        self._run_single(self.s.sum, self.s.get_current_signal(),
                         dict(axis=0))

    def test_mean(self):
        self._run_single(self.s.mean, self.s, dict(axis=('x', 'z')))

    def test_max(self):
        self._run_single(self.s.max, self.s, dict(axis=('x', 'z')))

    def test_min(self):
        self._run_single(self.s.min, self.s, dict(axis=('x', 'z')))

    def test_std(self):
        self._run_single(self.s.std, self.s, dict(axis=('x', 'z')))

    def test_var(self):
        self._run_single(self.s.var, self.s, dict(axis=('x', 'z')))

    def test_diff(self):
        self._run_single(self.s.diff, self.s, dict(axis=0))

    def test_derivative(self):
        self._run_single(self.s.derivative, self.s, dict(axis=0))

    def test_integrate_simpson(self):
        self._run_single(self.s.integrate_simpson, self.s, dict(axis=0))

    def test_integrate1D(self):
        self._run_single(self.s.integrate1D, self.s, dict(axis=0))

    def test_indexmax(self):
        self._run_single(self.s.indexmax, self.s, dict(axis=0))

    def test_valuemax(self):
        self._run_single(self.s.valuemax, self.s, dict(axis=0))

    @pytest.mark.xfail(sys.platform == 'win32',
                       reason="sometimes it does not run lazily on windows")
    def test_rebin(self):
        s = self.s
        scale = (1, 2, 1, 2)
        self._run_single(s.rebin, s, dict(scale=scale))

    def test_as_spectrum(self):
        s = self.s
        self._run_single(s.as_signal1D, s, dict(spectral_axis=1))

    def test_as_image(self):
        s = self.s
        self._run_single(s.as_signal2D, s, dict(image_axes=(
            s.axes_manager.navigation_axes[0:2])))

    def test_inav(self):
        s = self.s
        self._run_single(s.inav.__getitem__, s, {
            "slices": (slice(2, 4, None), slice(None), slice(0, 2, None))})

    def test_isig(self):
        s = self.s
        self._run_single(s.isig.__getitem__, s, {
            "slices": (slice(2, 4, None),)})

    def test_inav_variance(self):
        s = self.s
        s.metadata.set_item("Signal.Noise_properties.variance",
                            s.deepcopy())
        s1 = s.inav[2:4, 0:2]
        s2 = s.inav[2:4, 1:3]
        s.inav.__getitem__(slices=(slice(2, 4, None), slice(1, 3, None),
                                   slice(None)), out=s1)
        assert_array_equal(s1.metadata.Signal.Noise_properties.variance.data,
                           s2.metadata.Signal.Noise_properties.variance.data,)

    def test_isig_variance(self):
        s = self.s
        s.metadata.set_item("Signal.Noise_properties.variance",
                            s.deepcopy())
        s1 = s.isig[2:4]
        s2 = s.isig[1:5]
        s.isig.__getitem__(slices=(slice(1, 5, None)), out=s1)
        assert_array_equal(s1.metadata.Signal.Noise_properties.variance.data,
                           s2.metadata.Signal.Noise_properties.variance.data,)

    def test_histogram_axis_changes(self):
        s = self.s
        h1 = s.get_histogram(bins=4)
        h2 = s.get_histogram(bins=5)
        s.get_histogram(bins=5, out=h1)
        assert_array_equal(h1.data, h2.data)
        assert (h1.axes_manager[-1].size ==
                h2.axes_manager[-1].size)

    def test_masked_array_mean(self):
        s = self.s
        if s._lazy:
            pytest.skip("LazyS do not support masked arrays")
        mask = (s.data > 0.5)
        s.data = np.arange(s.data.size).reshape(s.data.shape)
        s.data = np.ma.masked_array(s.data, mask=mask)
        sr = s.mean(axis=('x', 'z',))
        nt.assert_array_equal(
            sr.data.shape, [ax.size for ax in s.axes_manager[('y', 'E')]])
        print(sr.data.tolist())
        ref = [[202.28571428571428, 203.28571428571428, 182.0,
                197.66666666666666, 187.0, 177.8],
               [134.0, 190.0, 191.27272727272728, 170.14285714285714, 172.0,
                209.85714285714286],
               [168.0, 161.8, 162.8, 185.4, 197.71428571428572,
                178.14285714285714],
               [240.0, 184.33333333333334, 260.0, 229.0, 173.2, 167.0]]
        nt.assert_array_equal(sr.data, ref)

    def test_masked_array_sum(self):
        s = self.s
        if s._lazy:
            pytest.skip("LazyS do not support masked arrays")
        mask = (s.data > 0.5)
        s.data = np.ma.masked_array(np.ones_like(s.data), mask=mask)
        sr = s.sum(axis=('x', 'z',))
        nt.assert_array_equal(sr.data.sum(), (~mask).sum())

    @pytest.mark.parametrize('mask', (True, False))
    def test_sum_no_navigation_axis(self, mask):
        s = signals.Signal1D(np.arange(100))
        if mask:
            s.data = np.ma.masked_array(s.data, mask=(s < 50))
        # Since s haven't any navigation axis, it returns the same signal as
        # default
        np.testing.assert_array_equal(s, s.sum())
        # When we specify an axis, it actually takes the sum.
        np.testing.assert_array_equal(s.data.sum(), s.sum(axis=0))

    def test_masked_arrays_out(self):
        s = self.s
        if s._lazy:
            pytest.skip("LazyS do not support masked arrays")
        mask = (s.data > 0.5)
        s.data = np.ones_like(s.data)
        s.data = np.ma.masked_array(s.data, mask=mask)
        self._run_single(s.sum, s, dict(axis=('x', 'z')))

    def test_wrong_out_shape(self):
        s = self.s
        ss = s.sum()  # Sum over navigation, data shape (6,)
        with pytest.raises(ValueError):
            s.sum(axis=s.axes_manager._axes, out=ss)

    def test_wrong_out_shape_masked(self):
        s = self.s
        s.data = np.ma.array(s.data)
        ss = s.sum()  # Sum over navigation, data shape (6,)
        with pytest.raises(ValueError):
            s.sum(axis=s.axes_manager._axes, out=ss)


@lazifyTestClass
class TestTranspose:

    def setup_method(self, method):
        self.s = signals.BaseSignal(np.random.rand(1, 2, 3, 4, 5, 6))
        for ax, name in zip(self.s.axes_manager._axes, 'abcdef'):
            ax.name = name
        # just to make sure in case default changes
        self.s.axes_manager.set_signal_dimension(6)
        self.s.estimate_poissonian_noise_variance()

    def test_signal_int_transpose(self):
        t = self.s.transpose(signal_axes=2)
        var = t.metadata.Signal.Noise_properties.variance
        assert t.axes_manager.signal_shape == (6, 5)
        assert var.axes_manager.signal_shape == (6, 5)
        assert ([ax.name for ax in t.axes_manager.signal_axes] ==
                ['f', 'e'])
        assert isinstance(t, signals.Signal2D)
        assert isinstance(t.metadata.Signal.Noise_properties.variance,
                          signals.Signal2D)

    def test_signal_iterable_int_transpose(self):
        t = self.s.transpose(signal_axes=[0, 5, 4])
        var = t.metadata.Signal.Noise_properties.variance
        assert t.axes_manager.signal_shape == (6, 1, 2)
        assert var.axes_manager.signal_shape == (6, 1, 2)
        assert ([ax.name for ax in t.axes_manager.signal_axes] ==
                ['f', 'a', 'b'])

    def test_signal_iterable_names_transpose(self):
        t = self.s.transpose(signal_axes=['f', 'a', 'b'])
        var = t.metadata.Signal.Noise_properties.variance
        assert t.axes_manager.signal_shape == (6, 1, 2)
        assert var.axes_manager.signal_shape == (6, 1, 2)
        assert ([ax.name for ax in t.axes_manager.signal_axes] ==
                ['f', 'a', 'b'])

    def test_signal_iterable_axes_transpose(self):
        t = self.s.transpose(signal_axes=self.s.axes_manager.signal_axes[:2])
        var = t.metadata.Signal.Noise_properties.variance
        assert t.axes_manager.signal_shape == (6, 5)
        assert var.axes_manager.signal_shape == (6, 5)
        assert ([ax.name for ax in t.axes_manager.signal_axes] ==
                ['f', 'e'])

    def test_signal_one_name(self):
        with pytest.raises(ValueError):
            self.s.transpose(signal_axes='a')

    def test_too_many_signal_axes(self):
        with pytest.raises(ValueError):
            self.s.transpose(signal_axes=10)

    def test_navigation_int_transpose(self):
        t = self.s.transpose(navigation_axes=2)
        var = t.metadata.Signal.Noise_properties.variance
        assert t.axes_manager.navigation_shape == (2, 1)
        assert var.axes_manager.navigation_shape == (2, 1)
        assert ([ax.name for ax in t.axes_manager.navigation_axes] ==
                ['b', 'a'])

    def test_navigation_iterable_int_transpose(self):
        t = self.s.transpose(navigation_axes=[0, 5, 4])
        var = t.metadata.Signal.Noise_properties.variance
        assert t.axes_manager.navigation_shape == (6, 1, 2)
        assert var.axes_manager.navigation_shape == (6, 1, 2)
        assert ([ax.name for ax in t.axes_manager.navigation_axes] ==
                ['f', 'a', 'b'])

    def test_navigation_iterable_names_transpose(self):
        t = self.s.transpose(navigation_axes=['f', 'a', 'b'])
        var = t.metadata.Signal.Noise_properties.variance
        assert var.axes_manager.navigation_shape == (6, 1, 2)
        assert t.axes_manager.navigation_shape == (6, 1, 2)
        assert ([ax.name for ax in t.axes_manager.navigation_axes] ==
                ['f', 'a', 'b'])

    def test_navigation_iterable_axes_transpose(self):
        t = self.s.transpose(
            navigation_axes=self.s.axes_manager.signal_axes[
                :2])
        var = t.metadata.Signal.Noise_properties.variance
        assert t.axes_manager.navigation_shape == (6, 5)
        assert var.axes_manager.navigation_shape == (6, 5)
        assert ([ax.name for ax in t.axes_manager.navigation_axes] ==
                ['f', 'e'])

    def test_navigation_one_name(self):
        with pytest.raises(ValueError):
            self.s.transpose(navigation_axes='a')

    def test_too_many_navigation_axes(self):
        with pytest.raises(ValueError):
            self.s.transpose(navigation_axes=10)

    def test_transpose_shortcut(self):
        s = self.s.transpose(signal_axes=2)
        t = s.T
        assert t.axes_manager.navigation_shape == (6, 5)
        assert ([ax.name for ax in t.axes_manager.navigation_axes] ==
                ['f', 'e'])

    def test_optimize(self):
        if self.s._lazy:
            pytest.skip(
                "LazyS optimization is tested in test_lazy_tranpose_rechunk")
        t = self.s.transpose(signal_axes=['f', 'a', 'b'], optimize=False)
        assert t.data.base is self.s.data

        t = self.s.transpose(signal_axes=['f', 'a', 'b'], optimize=True)
        assert t.data.base is not self.s.data


def test_lazy_transpose_rechunks():
    ar = da.ones((50, 50, 256, 256), chunks=(5, 5, 256, 256))
    s = signals.Signal2D(ar).as_lazy()
    s1 = s.T  # By default it does not rechunk
    cks = s.data.chunks
    assert s1.data.chunks == (cks[2], cks[3], cks[0], cks[1])
    s2 = s.transpose(optimize=True)
    assert s2.data.chunks != s1.data.chunks


def test_lazy_changetype_rechunk():
    ar = da.ones((50, 50, 256, 256), chunks=(5, 5, 256, 256), dtype='uint8')
    s = signals.Signal2D(ar).as_lazy()
    s._make_lazy(rechunk=True)
    assert s.data.dtype is np.dtype('uint8')
    chunks_old = s.data.chunks
    s.change_dtype('float')
    assert s.data.dtype is np.dtype('float')
    chunks_new = s.data.chunks
    assert (len(chunks_old[0]) * len(chunks_old[1]) <
            len(chunks_new[0]) * len(chunks_new[1]))
    s.change_dtype('uint8')
    assert s.data.dtype is np.dtype('uint8')
    chunks_newest = s.data.chunks
    assert chunks_newest == chunks_new


def test_lazy_changetype_rechunk_False():
    ar = da.ones((50, 50, 256, 256), chunks=(5, 5, 256, 256), dtype='uint8')
    s = signals.Signal2D(ar).as_lazy()
    s._make_lazy(rechunk=True)
    assert s.data.dtype is np.dtype('uint8')
    chunks_old = s.data.chunks
    s.change_dtype('float', rechunk=False)
    assert s.data.dtype is np.dtype('float')
    assert chunks_old == s.data.chunks


def test_lazy_reduce_rechunk():
    s = signals.Signal1D(da.ones((10, 100), chunks=(1, 2))).as_lazy()
    reduce_methods = (s.sum, s.mean, s.max, s.std, s.var, s.nansum, s.nanmax, s.nanmin,
                      s.nanmean, s.nanstd, s.nanvar, s.indexmin, s.indexmax, s.valuemax,
                      s.valuemin)
    for rm in reduce_methods:
        assert rm(
            axis=0).data.chunks == (
            (100,),)  # The data has been rechunked
        assert rm(
            axis=0, rechunk=False).data.chunks == (
            (2,) * 50,)  # The data has not been rechunked


def test_lazy_diff_rechunk():
    s = signals.Signal1D(da.ones((10, 100), chunks=(1, 2))).as_lazy()
    for rm in (s.derivative, s.diff):
        # The data has been rechunked
        assert rm(axis=-1).data.chunks == ((10,), (99,))
        assert rm(axis=-1, rechunk=False).data.chunks == ((1,) *
                                                          10, (1,) * 99)  # The data has not been rechunked


def test_spikes_removal_tool(mpl_cleanup):
    s = signals.Signal1D(np.ones((2, 3, 30)))
    np.random.seed(1)
    s.add_gaussian_noise(1E-5)
    # Add three spikes
    s.data[1, 0, 1] += 2
    s.data[0, 2, 29] += 1
    s.data[1, 2, 14] += 1

    sr = SpikesRemoval(s)
    sr.threshold = 1.5
    sr.find()
    assert s.axes_manager.indices == (0, 1)
    sr.threshold = 0.5
    assert s.axes_manager.indices == (0, 0)
    sr.find()
    assert s.axes_manager.indices == (2, 0)
    sr.find()
    assert s.axes_manager.indices == (0, 1)
    sr.find(back=True)
    assert s.axes_manager.indices == (2, 0)
    sr.add_noise = False
    sr.apply()
    assert_almost_equal(s.data[0, 2, 29], 1, decimal=5)
    assert s.axes_manager.indices == (0, 1)
    sr.apply()
    assert_almost_equal(s.data[1, 0, 1], 1, decimal=5)
    assert s.axes_manager.indices == (2, 1)
    np.random.seed(1)
    sr.add_noise = True
    sr.default_spike_width = 3
    sr.interpolator_kind = "Spline"
    sr.spline_order = 3
    sr.apply()
    assert_almost_equal(s.data[1, 2, 14], 1, decimal=5)
    assert s.axes_manager.indices == (0, 0)


class TestLinearRebin:

    def test_linear_downsize(self):
        spectrum = signals.EDSTEMSpectrum(np.ones([3, 5, 1]))
        scale = (1.5, 2.5, 1)
        res = spectrum.rebin(scale=scale, crop=True)
        nt.assert_allclose(res.data, 3.75 * np.ones((1, 3, 1)))
        for axis in res.axes_manager._axes:
            assert scale[axis.index_in_axes_manager] == axis.scale
        res = spectrum.rebin(scale=scale, crop=False)
        nt.assert_allclose(res.data.sum(), spectrum.data.sum())

    def test_linear_upsize(self):
        spectrum = signals.EDSTEMSpectrum(np.ones([4, 5, 10]))
        scale = [0.3, 0.2, .5]
        res = spectrum.rebin(scale=scale)
        nt.assert_allclose(res.data, 0.03 * np.ones((20, 16, 20)))
        for axis in res.axes_manager._axes:
            assert scale[axis.index_in_axes_manager] == axis.scale
        res = spectrum.rebin(scale=scale, crop=False)
        nt.assert_allclose(res.data.sum(), spectrum.data.sum())

    def test_linear_downscale_out(self):
        spectrum = signals.EDSTEMSpectrum(np.ones([4, 1, 1]))
        scale = [1, 0.4, 1]
        res = spectrum.rebin(scale=scale)
        spectrum.data[2][0] = 5
        spectrum.rebin(scale=scale, out=res)
        nt.assert_allclose(res.data, [[[0.4]],
                                      [[0.4]], [[0.4]], [
            [0.4]], [[0.4]], [[2.]],
            [[2.]], [[1.2]], [[0.4]], [[0.4]]])
        for axis in res.axes_manager._axes:
            assert scale[axis.index_in_axes_manager] == axis.scale

    def test_linear_upscale_out(self):
        spectrum = signals.EDSTEMSpectrum(np.ones([4, 1, 1]))
        scale = [1, 0.4, 1]
        res = spectrum.rebin(scale=scale)
        spectrum.data[2][0] = 5
        spectrum.rebin(scale=scale, out=res)
        nt.assert_allclose(res.data, [[[0.4]],
                                      [[0.4]], [[0.4]], [
            [0.4]], [[0.4]], [[2.]],
            [[2.]], [[1.2]], [[0.4]], [[0.4]]], atol=1e-3)
        for axis in res.axes_manager._axes:
            assert scale[axis.index_in_axes_manager] == axis.scale
