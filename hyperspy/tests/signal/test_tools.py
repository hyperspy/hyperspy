import numpy as np
import nose.tools as nt

from hyperspy.signals import BaseSignal
from hyperspy import signals


class Test2D:

    def setUp(self):
        self.signal = BaseSignal(np.arange(5 * 10).reshape(5, 10))
        self.signal.axes_manager[0].name = "x"
        self.signal.axes_manager[1].name = "E"
        self.signal.axes_manager[0].scale = 0.5
        self.data = self.signal.data.copy()

    def test_sum_x(self):
        s = self.signal.sum("x")
        np.testing.assert_array_equal(self.signal.data.sum(0), s.data)
        nt.assert_equal(s.data.ndim, 1)
        nt.assert_equal(s.axes_manager.navigation_dimension, 0)

    def test_sum_x_E(self):
        s = self.signal.sum("x").sum("E")
        np.testing.assert_array_equal(self.signal.data.sum(), s.data)
        nt.assert_equal(s.data.ndim, 1)
        # Check that there is still one signal axis.
        nt.assert_equal(s.axes_manager.signal_dimension, 1)

    def test_axis_by_str(self):
        s1 = self.signal.deepcopy()
        s2 = self.signal.deepcopy()
        s1.crop(0, 2, 4)
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
        nt.assert_equal(len(result), 2)
        np.testing.assert_array_almost_equal(result[0].data, self.data[:2, :])
        np.testing.assert_array_almost_equal(result[1].data, self.data[2:4, :])

    def test_split_axis1(self):
        result = self.signal.split(1, 2)
        nt.assert_equal(len(result), 2)
        np.testing.assert_array_almost_equal(result[0].data, self.data[:, :5])
        np.testing.assert_array_almost_equal(result[1].data, self.data[:, 5:])

    def test_split_axisE(self):
        result = self.signal.split("E", 2)
        nt.assert_equal(len(result), 2)
        np.testing.assert_array_almost_equal(result[0].data, self.data[:, :5])
        np.testing.assert_array_almost_equal(result[1].data, self.data[:, 5:])

    def test_split_default(self):
        result = self.signal.split()
        nt.assert_equal(len(result), 5)
        np.testing.assert_array_almost_equal(result[0].data, self.data[0])

    def test_histogram(self):
        result = self.signal.get_histogram(3)
        nt.assert_true(isinstance(result, signals.Signal1D))
        np.testing.assert_equal(result.data, [17, 16, 17])
        nt.assert_true(result.metadata.Signal.binned)

    def test_estimate_poissonian_noise_copy_data(self):
        self.signal.estimate_poissonian_noise_variance()
        variance = self.signal.metadata.Signal.Noise_properties.variance
        nt.assert_true(
            variance.data is not self.signal.data)

    def test_estimate_poissonian_noise_noarg(self):
        self.signal.estimate_poissonian_noise_variance()
        variance = self.signal.metadata.Signal.Noise_properties.variance
        np.testing.assert_array_equal(variance.data, self.signal.data)
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
        s.axes_manager.set_signal_dimension(2)
        s.unfold()
        nt.assert_equal(s.data.shape, (50,))

    def test_unfold_image_returns_true(self):
        s = self.signal
        s.axes_manager.set_signal_dimension(2)
        nt.assert_true(s.unfold())

    def test_print_summary(self):
        # Just test if it doesn't raise an exception
        self.signal._print_summary()


class Test3D:

    def setUp(self):
        self.signal = BaseSignal(np.arange(2 * 4 * 6).reshape(2, 4, 6))
        self.signal.axes_manager[0].name = "x"
        self.signal.axes_manager[1].name = "y"
        self.signal.axes_manager[2].name = "E"
        self.signal.axes_manager[0].scale = 0.5
        self.data = self.signal.data.copy()

    def test_rebin(self):
        self.signal.estimate_poissonian_noise_variance()
        new_s = self.signal.rebin((2, 1, 6))
        var = new_s.metadata.Signal.Noise_properties.variance
        nt.assert_equal(new_s.data.shape, (1, 2, 6))
        nt.assert_equal(var.data.shape, (1, 2, 6))
        from hyperspy.misc.array_tools import rebin
        np.testing.assert_array_equal(rebin(self.signal.data, (1, 2, 6)),
                                      var.data)
        np.testing.assert_array_equal(rebin(self.signal.data, (1, 2, 6)),
                                      new_s.data)

    @nt.raises(AttributeError)
    def test_rebin_no_variance(self):
        new_s = self.signal.rebin((2, 1, 6))
        _ = new_s.metadata.Signal.Noise_properties

    def test_rebin_const_variance(self):
        self.signal.metadata.set_item('Signal.Noise_properties.variance', 0.3)
        new_s = self.signal.rebin((2, 1, 6))
        nt.assert_equal(new_s.metadata.Signal.Noise_properties.variance, 0.3)

    def test_swap_axes_simple(self):
        s = self.signal
        nt.assert_equal(s.swap_axes(0, 1).data.shape, (4, 2, 6))
        nt.assert_equal(s.swap_axes(0, 2).axes_manager.shape, (6, 2, 4))
        nt.assert_true(s.swap_axes(0, 2).data.flags['C_CONTIGUOUS'])

    def test_swap_axes_iteration(self):
        s = self.signal
        s = s.swap_axes(0, 2)
        nt.assert_equal(s.axes_manager._getitem_tuple[:2], (0, 0))
        s.axes_manager.indices = (2, 1)
        nt.assert_equal(s.axes_manager._getitem_tuple[:2], (1, 2))

    def test_get_navigation_signal_nav_dim0(self):
        s = self.signal
        s.axes_manager.set_signal_dimension(3)
        ns = s._get_navigation_signal()
        nt.assert_equal(ns.axes_manager.signal_dimension, 1)
        nt.assert_equal(ns.axes_manager.signal_size, 1)
        nt.assert_equal(ns.axes_manager.navigation_dimension, 0)

    def test_get_navigation_signal_nav_dim1(self):
        s = self.signal
        s.axes_manager.set_signal_dimension(2)
        ns = s._get_navigation_signal()
        nt.assert_equal(ns.axes_manager.signal_shape,
                        s.axes_manager.navigation_shape)
        nt.assert_equal(ns.axes_manager.navigation_dimension, 0)

    def test_get_navigation_signal_nav_dim2(self):
        s = self.signal
        s.axes_manager.set_signal_dimension(1)
        ns = s._get_navigation_signal()
        nt.assert_equal(ns.axes_manager.signal_shape,
                        s.axes_manager.navigation_shape)
        nt.assert_equal(ns.axes_manager.navigation_dimension, 0)

    def test_get_navigation_signal_nav_dim3(self):
        s = self.signal
        s.axes_manager.set_signal_dimension(0)
        ns = s._get_navigation_signal()
        nt.assert_equal(ns.axes_manager.signal_shape,
                        s.axes_manager.navigation_shape)
        nt.assert_equal(ns.axes_manager.navigation_dimension, 0)

    @nt.raises(ValueError)
    def test_get_navigation_signal_wrong_data_shape(self):
        s = self.signal
        s.axes_manager.set_signal_dimension(1)
        s._get_navigation_signal(data=np.zeros((3, 2)))

    @nt.raises(ValueError)
    def test_get_navigation_signal_wrong_data_shape_dim0(self):
        s = self.signal
        s.axes_manager.set_signal_dimension(3)
        s._get_navigation_signal(data=np.asarray(0))

    def test_get_navigation_signal_given_data(self):
        s = self.signal
        s.axes_manager.set_signal_dimension(1)
        data = np.zeros(s.axes_manager._navigation_shape_in_array)
        ns = s._get_navigation_signal(data=data)
        nt.assert_is(ns.data, data)

    def test_get_signal_signal_nav_dim0(self):
        s = self.signal
        s.axes_manager.set_signal_dimension(0)
        ns = s._get_signal_signal()
        nt.assert_equal(ns.axes_manager.navigation_dimension, 0)
        nt.assert_equal(ns.axes_manager.navigation_size, 0)
        nt.assert_equal(ns.axes_manager.signal_dimension, 1)

    def test_get_signal_signal_nav_dim1(self):
        s = self.signal
        s.axes_manager.set_signal_dimension(1)
        ns = s._get_signal_signal()
        nt.assert_equal(ns.axes_manager.signal_shape,
                        s.axes_manager.signal_shape)
        nt.assert_equal(ns.axes_manager.navigation_dimension, 0)

    def test_get_signal_signal_nav_dim2(self):
        s = self.signal
        s.axes_manager.set_signal_dimension(2)
        ns = s._get_signal_signal()
        nt.assert_equal(ns.axes_manager.signal_shape,
                        s.axes_manager.signal_shape)
        nt.assert_equal(ns.axes_manager.navigation_dimension, 0)

    def test_get_signal_signal_nav_dim3(self):
        s = self.signal
        s.axes_manager.set_signal_dimension(3)
        ns = s._get_signal_signal()
        nt.assert_equal(ns.axes_manager.signal_shape,
                        s.axes_manager.signal_shape)
        nt.assert_equal(ns.axes_manager.navigation_dimension, 0)

    @nt.raises(ValueError)
    def test_get_signal_signal_wrong_data_shape(self):
        s = self.signal
        s.axes_manager.set_signal_dimension(1)
        s._get_signal_signal(data=np.zeros((3, 2)))

    @nt.raises(ValueError)
    def test_get_signal_signal_wrong_data_shape_dim0(self):
        s = self.signal
        s.axes_manager.set_signal_dimension(0)
        s._get_signal_signal(data=np.asarray(0))

    def test_get_signal_signal_given_data(self):
        s = self.signal
        s.axes_manager.set_signal_dimension(2)
        data = np.zeros(s.axes_manager._signal_shape_in_array)
        ns = s._get_signal_signal(data=data)
        nt.assert_is(ns.data, data)

    def test_get_navigation_signal_dtype(self):
        s = self.signal
        nt.assert_equal(s._get_navigation_signal().data.dtype.name,
                        s.data.dtype.name)

    def test_get_signal_signal_dtype(self):
        s = self.signal
        nt.assert_equal(s._get_signal_signal().data.dtype.name,
                        s.data.dtype.name)

    def test_get_navigation_signal_given_dtype(self):
        s = self.signal
        nt.assert_equal(
            s._get_navigation_signal(dtype="bool").data.dtype.name, "bool")

    def test_get_signal_signal_given_dtype(self):
        s = self.signal
        nt.assert_equal(
            s._get_signal_signal(dtype="bool").data.dtype.name, "bool")


class Test4D:

    def setUp(self):
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
        nt.assert_equal(
            diff.axes_manager[2].offset,
            s.axes_manager[2].offset + s.axes_manager[2].scale)

    def test_rollaxis_int(self):
        nt.assert_equal(self.s.rollaxis(2, 0).data.shape, (4, 3, 5, 6))

    def test_rollaxis_str(self):
        nt.assert_equal(self.s.rollaxis("z", "x").data.shape, (4, 3, 5, 6))

    def test_unfold_spectrum(self):
        self.s.unfold()
        nt.assert_equal(self.s.data.shape, (60, 6))

    def test_unfold_spectrum_returns_true(self):
        nt.assert_true(self.s.unfold())

    def test_unfold_spectrum_signal_returns_false(self):
        nt.assert_false(self.s.unfold_signal_space())

    def test_unfold_image(self):
        im = self.s.to_signal2D()
        im.unfold()
        nt.assert_equal(im.data.shape, (30, 12))

    def test_image_signal_unfolded_deepcopy(self):
        im = self.s.to_signal2D()
        im.unfold()
        # The following could fail if the constructor was not taking the fact
        # that the signal is unfolded into account when setting the signal
        # dimension.
        im.deepcopy()

    def test_image_signal_unfolded_false(self):
        im = self.s.to_signal2D()
        nt.assert_false(im.metadata._HyperSpy.Folding.signal_unfolded)

    def test_image_signal_unfolded_true(self):
        im = self.s.to_signal2D()
        im.unfold()
        nt.assert_true(im.metadata._HyperSpy.Folding.signal_unfolded)

    def test_image_signal_unfolded_back_to_false(self):
        im = self.s.to_signal2D()
        im.unfold()
        im.fold()
        nt.assert_false(im.metadata._HyperSpy.Folding.signal_unfolded)


def test_signal_iterator():
    s = BaseSignal(np.arange(3).reshape((3, 1)))
    nt.assert_equal(next(s).data[0], 0)
    # If the following fails it can be because the iteration index was not
    # restarted
    for i, signal in enumerate(s):
        nt.assert_equal(i, signal.data[0])


class TestDerivative:

    def setup(self):
        offset = 3
        scale = 0.1
        x = np.arange(-offset, offset, scale)
        s = signals.Signal1D(np.sin(x))
        s.axes_manager[0].offset = x[0]
        s.axes_manager[0].scale = scale
        self.s = s

    def test_derivative_data(self):
        der = self.s.derivative(axis=0, order=4)
        nt.assert_true(np.allclose(der.data,
                                   np.sin(der.axes_manager[0].axis),
                                   atol=1e-2),)
