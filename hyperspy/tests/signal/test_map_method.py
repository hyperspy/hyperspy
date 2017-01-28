from unittest import mock

import numpy as np
from scipy.ndimage import rotate, gaussian_filter, gaussian_filter1d
import nose.tools as nt

import hyperspy.api as hs


class TestImage:

    def setup_method(self, method):
        self.im = hs.signals.Signal2D(np.arange(0., 18).reshape((2, 3, 3)))

    def test_constant_sigma(self):
        im = self.im
        imt = im.deepcopy()
        for s, t in zip([im, imt], [False, True]):
            s.map(gaussian_filter, sigma=1, show_progressbar=None, parallel=t)
            np.testing.assert_allclose(s.data, np.array(
                [[[1.68829507, 2.2662213, 2.84414753],
                  [3.42207377, 4., 4.57792623],
                  [5.15585247, 5.7337787, 6.31170493]],

                 [[10.68829507, 11.2662213, 11.84414753],
                  [12.42207377, 13., 13.57792623],
                  [14.15585247, 14.7337787, 15.31170493]]]))

    def test_constant_sigma_navdim0(self):
        im = self.im.inav[0]
        imt = im.deepcopy()
        for s, t in zip([im, imt], [False, True]):
            s.map(gaussian_filter, sigma=1, show_progressbar=None,
                  parallel=t)
            np.testing.assert_allclose(s.data, np.array(
                [[1.68829507, 2.2662213, 2.84414753],
                 [3.42207377, 4., 4.57792623],
                 [5.15585247, 5.7337787, 6.31170493]]))

    def test_variable_sigma(self):
        im = self.im
        imt = im.deepcopy()

        sigmas = hs.signals.BaseSignal(np.array([0, 1]))
        sigmas.axes_manager.set_signal_dimension(0)

        for s, t in zip([im, imt], [False, True]):
            s.map(gaussian_filter,
                  sigma=sigmas, show_progressbar=None,
                  parallel=t)
            np.testing.assert_allclose(s.data, np.array(
                [[[0., 1., 2.],
                    [3., 4., 5.],
                    [6., 7., 8.]],

                 [[10.68829507, 11.2662213, 11.84414753],
                  [12.42207377, 13., 13.57792623],
                  [14.15585247, 14.7337787, 15.31170493]]]))

    def test_axes_argument(self):
        im = self.im
        imt = im.deepcopy()
        for s, t in zip([im, imt], [False, True]):
            s.map(rotate, angle=45, reshape=False, show_progressbar=None,
                  parallel=t)
            np.testing.assert_allclose(s.data, np.array(
                [[[0., 2.23223305, 0.],
                  [0.46446609, 4., 7.53553391],
                  [0., 5.76776695, 0.]],

                 [[0., 11.23223305, 0.],
                  [9.46446609, 13., 16.53553391],
                  [0., 14.76776695, 0.]]]))

    def test_different_shapes(self):
        im = self.im
        imt = im.deepcopy()
        angles = hs.signals.BaseSignal([0, 45])
        for s, t in zip([im, imt], [False, True]):
            s.map(rotate, angle=angles.T, reshape=True, show_progressbar=None,
                  parallel=t)
            # the dtype
            assert_is(s.data.dtype, np.dtype('O'))
            # the special slicing
            assert_is(s.inav[0].data.base, s.data[0])
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


class TestSignal1D:

    def setup_method(self, method):
        self.s = hs.signals.Signal1D(np.arange(0., 6).reshape((2, 3)))

    def test_constant_sigma(self):
        ss = self.s.deepcopy()
        for s, t in zip([self.s, ss], [False, True]):
            m = mock.Mock()
            s.events.data_changed.connect(m.data_changed)
            s.map(gaussian_filter1d, sigma=1, show_progressbar=None,
                  parallel=t)
            np.testing.assert_allclose(s.data, np.array(
                ([[0.42207377, 1., 1.57792623],
                  [3.42207377, 4., 4.57792623]])))
            assert_true(m.data_changed.called)

    def test_dtype(self):
        ss = self.s.deepcopy()
        for s, t in zip([self.s, ss], [False, True]):
            s.map(lambda data: np.sqrt(np.complex128(data)),
                  show_progressbar=None,
                  parallel=t)
            assert_is(s.data.dtype, np.dtype('complex128'))


class TestSignal0D:

    def setup_method(self, method):
        self.s = hs.signals.BaseSignal(np.arange(0., 6).reshape((2, 3)))
        self.s.axes_manager.set_signal_dimension(0)

    def test(self):
        ss = self.s.deepcopy()
        for s, t in zip([self.s, ss], [False, True]):
            m = mock.Mock()
            s.events.data_changed.connect(m.data_changed)
            s.map(lambda x, e: x ** e, e=2, show_progressbar=None,
                  parallel=t)
            np.testing.assert_allclose(
                s.data, (np.arange(0., 6) ** 2).reshape((2, 3)))
            assert_true(m.data_changed.called)

    def test_nav_dim_1(self):
        s1 = self.s.inav[1, 1]
        ss = s1.deepcopy()
        for s, t in zip([s1, ss], [False, True]):
            m = mock.Mock()
            s.events.data_changed.connect(m.data_changed)
            s.map(lambda x, e: x ** e, e=2, show_progressbar=None,
                  parallel=t)
            np.testing.assert_allclose(s.data, self.s.inav[1, 1].data ** 2)
            assert_true(m.data_changed.called)


_alphabet = 'abcdefghijklmnopqrstuvwxyz'


class TestChangingAxes:

    def setup_method(self, method):
        self.base = hs.signals.BaseSignal(np.empty((2, 3, 4, 5, 6, 7)))
        for ax, name in zip(self.base.axes_manager._axes, _alphabet):
            ax.name = name

    def test_one_nav_reducing(self):
        s = self.base.transpose(signal_axes=4).inav[0, 0]
        s.map(np.mean, axis=1)
        assert_equal(list('def'), [ax.name for ax in
                                   s.axes_manager._axes])
        assert_equal(0, len(s.axes_manager.navigation_axes))
        s.map(np.mean, axis=(1, 2))
        assert_equal(['f'], [ax.name for ax in s.axes_manager._axes])
        assert_equal(0, len(s.axes_manager.navigation_axes))

    def test_one_nav_increasing(self):
        s = self.base.transpose(signal_axes=4).inav[0, 0]
        s.map(np.tile, reps=(2, 1, 1, 1, 1))
        assert_equal(len(s.axes_manager.signal_axes), 5)
        assert_true(set('cdef') <= {ax.name for ax in
                                    s.axes_manager._axes})
        assert_equal(0, len(s.axes_manager.navigation_axes))
        assert_equal(s.data.shape, (2, 4, 5, 6, 7))

    def test_reducing(self):
        s = self.base.transpose(signal_axes=4)
        s.map(np.mean, axis=1)
        assert_equal(list('abdef'), [ax.name for ax in
                                     s.axes_manager._axes])
        assert_equal(2, len(s.axes_manager.navigation_axes))
        s.map(np.mean, axis=(1, 2))
        assert_equal(['f'], [ax.name for ax in
                             s.axes_manager.signal_axes])
        assert_equal(list('ba'), [ax.name for ax in
                                  s.axes_manager.navigation_axes])
        assert_equal(2, len(s.axes_manager.navigation_axes))

    def test_increasing(self):
        s = self.base.transpose(signal_axes=4)
        s.map(np.tile, reps=(2, 1, 1, 1, 1))
        assert_equal(len(s.axes_manager.signal_axes), 5)
        assert_true(set('cdef') <= {ax.name for ax in
                                    s.axes_manager.signal_axes})
        assert_equal(list('ba'), [ax.name for ax in
                                  s.axes_manager.navigation_axes])
        assert_equal(2, len(s.axes_manager.navigation_axes))
        assert_equal(s.data.shape, (2, 3, 2, 4, 5, 6, 7))


def test_new_axes():
    s = hs.signals.Signal1D(np.empty((10, 10)))
    s.axes_manager.navigation_axes[0].name = 'a'
    s.axes_manager.signal_axes[0].name = 'b'

    def test_func(d, i):
        _slice = () + (None,) * i + (slice(None),)
        return d[_slice]
    res = s.map(test_func, inplace=False,
                i=hs.signals.BaseSignal(np.arange(10)).T)
    assert_is_not_none(res)
    sl = res.inav[:2]
    assert_equal(sl.axes_manager._axes[-1].name, 'a')
    sl = res.inav[-1]
    assert_is_instance(sl, hs.signals.BaseSignal)
    ax_names = {ax.name for ax in sl.axes_manager._axes}
    assert_equal(len(ax_names), 1)
    assert_false('a' in ax_names)
    assert_false('b' in ax_names)
    assert_equal(0, sl.axes_manager.navigation_dimension)
