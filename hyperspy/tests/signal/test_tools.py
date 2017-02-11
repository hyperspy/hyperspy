from unittest import mock

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

import pytest

from hyperspy import signals
from hyperspy.decorators import lazifyTestClass


def _verify_test_sum_x_E(self, s):
    np.testing.assert_array_equal(self.signal.data.sum(), s.data)
    assert s.data.ndim == 1
    # Check that there is still one signal axis.
    assert s.axes_manager.signal_dimension == 1


@lazifyTestClass
class Test2D:

    def setup_method(self, method):
        self.signal = signals.Signal1D(np.arange(5 * 10).reshape(5, 10))
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
        s.crop(0, 2, 2.)
        np.testing.assert_array_almost_equal(s.data, d[2:4, :])

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

    def test_histogram(self):
        result = self.signal.get_histogram(3)
        assert isinstance(result, signals.Signal1D)
        assert_array_equal(result.data, np.array([17, 16, 17]))
        assert result.metadata.Signal.binned

    def test_estimate_poissonian_noise_copy_data(self):
        self.signal.estimate_poissonian_noise_variance()
        variance = self.signal.metadata.Signal.Noise_properties.variance
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
            correlation_factor=0.5)
        variance = self.signal.metadata.Signal.Noise_properties.variance
        np.testing.assert_array_equal(variance.data,
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

    def test_numpy_unfunc_one_arg_titled(self):
        self.signal.metadata.General.title = "yes"
        result = np.exp(self.signal)
        assert isinstance(result, signals.Signal1D)
        np.testing.assert_array_equal(result.data, np.exp(self.signal.data))
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
        np.testing.assert_array_equal(result.data, np.add(s1.data, s2.data))
        assert result.metadata.General.title == "add(A, B)"

    def test_numpy_unfunc_two_arg_untitled(self):
        s1, s2 = self.signal.deepcopy(), self.signal.deepcopy()
        result = np.add(s1, s2)
        assert (result.metadata.General.title ==
                "add(Untitled Signal 1, Untitled Signal 2)")

    def test_numpy_func(self):
        result = np.angle(self.signal)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.angle(self.signal.data))


def _test_default_navigation_signal_operations_over_many_axes(self, op):
    s = getattr(self.signal, op)()
    ar = getattr(self.data, op)(axis=(0, 1))
    np.testing.assert_array_equal(ar, s.data)
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

    def test_indexmax(self):
        s = self.signal.indexmax('E')
        ar = self.data.argmax(2)
        np.testing.assert_array_equal(ar, s.data)
        assert s.data.ndim == 2
        assert s.axes_manager.signal_dimension == 0
        assert s.axes_manager.navigation_dimension == 2

    def test_valuemax(self):
        s = self.signal.valuemax('x')
        ar = self.signal.axes_manager['x'].index2value(self.data.argmax(1))
        np.testing.assert_array_equal(ar, s.data)
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
        new_s = self.signal.rebin((2, 1, 6))
        var = new_s.metadata.Signal.Noise_properties.variance
        assert new_s.data.shape == (1, 2, 6)
        assert var.data.shape == (1, 2, 6)
        from hyperspy.misc.array_tools import rebin

        if self.signal._lazy:
            from distutils.version import LooseVersion
            import dask
            if LooseVersion(np.__version__) >= "1.12.0" and \
               LooseVersion(dask.__version__) <= "0.13.0":
                pytest.skip("Dask not up to date with new numpy")

        np.testing.assert_array_equal(rebin(self.signal.data, (1, 2, 6)),
                                      var.data)
        np.testing.assert_array_equal(rebin(self.signal.data, (1, 2, 6)),
                                      new_s.data)

    def test_rebin_no_variance(self):
        new_s = self.signal.rebin((2, 1, 6))
        with pytest.raises(AttributeError):
            _ = new_s.metadata.Signal.Noise_properties

    def test_rebin_const_variance(self):
        self.signal.metadata.set_item(
            'Signal.Noise_properties.variance', 0.3)
        new_s = self.signal.rebin((2, 1, 6))
        assert new_s.metadata.Signal.Noise_properties.variance == 0.3

    def test_swap_axes_simple(self):
        s = self.signal
        if s._lazy:
            pytest.skip("LazyS do not support axes swapping")
        assert s.swap_axes(0, 1).data.shape == (4, 2, 6)
        assert s.swap_axes(0, 2).axes_manager.shape == (6, 2, 4)
        assert s.swap_axes(0, 2).data.flags['C_CONTIGUOUS']

    def test_swap_axes_iteration(self):
        s = self.signal
        if s._lazy:
            pytest.skip("LazyS do not support axes swapping")
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
        np.testing.assert_array_equal(diff.data, diff_data)

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
        if self.s._lazy:
            pytest.skip("LazyS do not support folding")
        self.s.unfold()
        assert self.s.data.shape == (60, 6)

    def test_unfold_spectrum_returns_true(self):
        if self.s._lazy:
            pytest.skip("LazyS do not support folding")
        assert self.s.unfold()

    def test_unfold_spectrum_signal_returns_false(self):
        if self.s._lazy:
            pytest.skip("LazyS do not support folding")
        assert not self.s.unfold_signal_space()

    def test_unfold_image(self):
        if self.s._lazy:
            pytest.skip("LazyS do not support folding")
        im = self.s.to_signal2D()
        im.unfold()
        assert im.data.shape == (30, 12)

    def test_image_signal_unfolded_deepcopy(self):
        if self.s._lazy:
            pytest.skip("LazyS do not support folding")
        im = self.s.to_signal2D()
        im.unfold()
        # The following could fail if the constructor was not taking the fact
        # that the signal is unfolded into account when setting the signal
        # dimension.
        im.deepcopy()

    def test_image_signal_unfolded_false(self):
        if self.s._lazy:
            pytest.skip("LazyS do not support folding")
        im = self.s.to_signal2D()
        assert not im.metadata._HyperSpy.Folding.signal_unfolded

    def test_image_signal_unfolded_true(self):
        if self.s._lazy:
            pytest.skip("LazyS do not support folding")
        im = self.s.to_signal2D()
        im.unfold()
        assert im.metadata._HyperSpy.Folding.signal_unfolded

    def test_image_signal_unfolded_back_to_false(self):
        if self.s._lazy:
            pytest.skip("LazyS do not support folding")
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
        assert_allclose(der.data, np.sin(der.axes_manager[0].axis), atol=1e-2)


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

    def test_rebin(self):
        s = self.s
        new_shape = (3, 2, 1, 3)
        if self.s._lazy:
            from distutils.version import LooseVersion
            import dask
            if LooseVersion(np.__version__) >= "1.12.0" and \
               LooseVersion(dask.__version__) <= "0.13.0":
                pytest.skip("Dask not up to date with new numpy")
        self._run_single(s.rebin, s, dict(new_shape=new_shape))

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
        np.testing.assert_array_equal(
            sr.data.shape, [ax.size for ax in s.axes_manager[('y', 'E')]])
        print(sr.data.tolist())
        ref = [[202.28571428571428, 203.28571428571428, 182.0,
                197.66666666666666, 187.0, 177.8],
               [134.0, 190.0, 191.27272727272728, 170.14285714285714, 172.0,
                209.85714285714286],
               [168.0, 161.8, 162.8, 185.4, 197.71428571428572,
                178.14285714285714],
               [240.0, 184.33333333333334, 260.0, 229.0, 173.2, 167.0]]
        np.testing.assert_array_equal(sr.data, ref)

    def test_masked_array_sum(self):
        s = self.s
        if s._lazy:
            pytest.skip("LazyS do not support masked arrays")
        mask = (s.data > 0.5)
        s.data = np.ma.masked_array(np.ones_like(s.data), mask=mask)
        sr = s.sum(axis=('x', 'z',))
        np.testing.assert_array_equal(sr.data.sum(), (~mask).sum())

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
        assert t.axes_manager.signal_shape == (6, 5)
        assert ([ax.name for ax in t.axes_manager.signal_axes] ==
                ['f', 'e'])
        assert isinstance(t, signals.Signal2D)
        assert isinstance(t.metadata.Signal.Noise_properties.variance,
                          signals.Signal2D)

    def test_signal_iterable_int_transpose(self):
        t = self.s.transpose(signal_axes=[0, 5, 4])
        assert t.axes_manager.signal_shape == (6, 1, 2)
        assert ([ax.name for ax in t.axes_manager.signal_axes] ==
                ['f', 'a', 'b'])

    def test_signal_iterable_names_transpose(self):
        t = self.s.transpose(signal_axes=['f', 'a', 'b'])
        assert t.axes_manager.signal_shape == (6, 1, 2)
        assert ([ax.name for ax in t.axes_manager.signal_axes] ==
                ['f', 'a', 'b'])

    def test_signal_iterable_axes_transpose(self):
        t = self.s.transpose(signal_axes=self.s.axes_manager.signal_axes[:2])
        assert t.axes_manager.signal_shape == (6, 5)
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
        assert t.axes_manager.navigation_shape == (2, 1)
        assert ([ax.name for ax in t.axes_manager.navigation_axes] ==
                ['b', 'a'])

    def test_navigation_iterable_int_transpose(self):
        t = self.s.transpose(navigation_axes=[0, 5, 4])
        assert t.axes_manager.navigation_shape == (6, 1, 2)
        assert ([ax.name for ax in t.axes_manager.navigation_axes] ==
                ['f', 'a', 'b'])

    def test_navigation_iterable_names_transpose(self):
        t = self.s.transpose(navigation_axes=['f', 'a', 'b'])
        assert t.axes_manager.navigation_shape == (6, 1, 2)
        assert ([ax.name for ax in t.axes_manager.navigation_axes] ==
                ['f', 'a', 'b'])

    def test_navigation_iterable_axes_transpose(self):
        t = self.s.transpose(
            navigation_axes=self.s.axes_manager.signal_axes[
                :2])
        assert t.axes_manager.navigation_shape == (6, 5)
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
            pytest.skip("LazyS do not support optimizations")
        t = self.s.transpose(signal_axes=['f', 'a', 'b'], optimize=False)
        assert t.data.base is self.s.data

        t = self.s.transpose(signal_axes=['f', 'a', 'b'], optimize=True)
        assert t.data.base is not self.s.data
