import numpy as np
from nose.tools import (
    assert_true,
    assert_equal,)

from hyperspy.signal import Signal
from hyperspy import signals


class Test2D:

    def setUp(self):
        self.signal = Signal(np.arange(5 * 10).reshape(5, 10))
        self.signal.axes_manager[0].name = "x"
        self.signal.axes_manager[1].name = "E"
        self.signal.axes_manager[0].scale = 0.5
        self.data = self.signal.data.copy()

    def test_axis_by_str(self):
        s1 = self.signal.deepcopy()
        s2 = self.signal.deepcopy()
        s1.crop(0, 2, 4)
        s2.crop("x", 2, 4)
        assert_true((s1.data == s2.data).all())

    def test_crop_int(self):
        s = self.signal
        d = self.data
        s.crop(0, 2, 4)
        assert_true((s.data == d[2:4, :]).all())

    def test_crop_float(self):
        s = self.signal
        d = self.data
        s.crop(0, 2, 2.)
        assert_true((s.data == d[2:4, :]).all())

    def test_split_axis0(self):
        result = self.signal.split(0, 2)
        assert_true(len(result) == 2)
        assert_true((result[0].data == self.data[:2, :]).all())
        assert_true((result[1].data == self.data[2:4, :]).all())

    def test_split_axis1(self):
        result = self.signal.split(1, 2)
        assert_true(len(result) == 2)
        assert_true((result[0].data == self.data[:, :5]).all())
        assert_true((result[1].data == self.data[:, 5:]).all())

    def test_split_axisE(self):
        result = self.signal.split("E", 2)
        assert_true(len(result) == 2)
        assert_true((result[0].data == self.data[:, :5]).all())
        assert_true((result[1].data == self.data[:, 5:]).all())

    def test_split_default(self):
        result = self.signal.split()
        assert_true(len(result) == 5)
        assert_true((result[0].data == self.data[0]).all())

    def test_histogram(self):
        result = self.signal.get_histogram(3)
        assert_true(isinstance(result, signals.Spectrum))
        assert_true((result.data == np.array([17, 16, 17])).all())
        assert_true(result.metadata.Signal.binned)

    def test_estimate_poissonian_noise_copy_data(self):
        self.signal.estimate_poissonian_noise_variance()
        assert_true(self.signal.metadata.Signal.Noise_properties.variance.data
                    is not self.signal.data)

    def test_estimate_poissonian_noise_noarg(self):
        self.signal.estimate_poissonian_noise_variance()
        assert_true(
            (self.signal.metadata.Signal.Noise_properties.variance.data ==
             self.signal.data).all())

    def test_estimate_poissonian_noise_with_args(self):
        self.signal.estimate_poissonian_noise_variance(
            expected_value=self.signal,
            gain_factor=2,
            gain_offset=1,
            correlation_factor=0.5)
        assert_true(
            (self.signal.metadata.Signal.Noise_properties.variance.data ==
             (self.signal.data * 2 + 1) * 0.5).all())


class Test3D:

    def setUp(self):
        self.signal = Signal(np.arange(2 * 4 * 6).reshape(2, 4, 6))
        self.signal.axes_manager[0].name = "x"
        self.signal.axes_manager[1].name = "y"
        self.signal.axes_manager[2].name = "E"
        self.signal.axes_manager[0].scale = 0.5
        self.data = self.signal.data.copy()

    def test_rebin(self):
        assert_true(self.signal.rebin((2, 1, 6)).data.shape == (1, 2, 6))

    def test_swap_axes(self):
        s = self.signal
        assert_equal(s.swap_axes(0, 1).data.shape, (4, 2, 6))
        assert_true(s.swap_axes(0, 2).data.flags['C_CONTIGUOUS'])


class Test4D:

    def setUp(self):
        s = signals.Spectrum(np.ones((5, 4, 3, 6)))
        for axis, name in zip(
                s.axes_manager._get_axes_in_natural_order(),
                ['x', 'y', 'z', 'E']):
            axis.name = name
        self.s = s

    def test_rollaxis_int(self):
        assert_equal(self.s.rollaxis(2, 0).data.shape, (4, 3, 5, 6))

    def test_rollaxis_str(self):
        assert_equal(self.s.rollaxis("z", "x").data.shape, (4, 3, 5, 6))

    def test_unfold_spectrum(self):
        self.s.unfold()
        assert_equal(self.s.data.shape, (60, 6))

    def test_unfold_image(self):
        im = self.s.to_image()
        im.unfold()
        assert_equal(im.data.shape, (30, 12))


def test_signal_iterator():
    s = Signal(np.arange(3).reshape((3, 1)))
    assert_equal(s.next().data[0], 0)
    # If the following fails it can be because the iteration index was not
    # restarted
    for i, signal in enumerate(s):
        assert_equal(i, signal.data[0])
