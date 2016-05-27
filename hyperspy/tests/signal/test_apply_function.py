import numpy as np
from scipy.ndimage import rotate, gaussian_filter, gaussian_filter1d
import nose.tools as nt

import hyperspy.api as hs


class TestImage:

    def setup(self):
        self.im = hs.signals.Signal2D(np.arange(0., 18).reshape((2, 3, 3)))

    def test_constant_sigma(self):
        im = self.im
        im.map(gaussian_filter, sigma=1, show_progressbar=None)
        nt.assert_true(np.allclose(im.data, np.array(
            [[[1.68829507, 2.2662213, 2.84414753],
              [3.42207377, 4., 4.57792623],
              [5.15585247, 5.7337787, 6.31170493]],

             [[10.68829507, 11.2662213, 11.84414753],
              [12.42207377, 13., 13.57792623],
              [14.15585247, 14.7337787, 15.31170493]]])))

    def test_constant_sigma_navdim0(self):
        im = self.im.inav[0]
        im.map(gaussian_filter, sigma=1, show_progressbar=None)
        nt.assert_true(np.allclose(im.data, np.array(
            [[1.68829507, 2.2662213, 2.84414753],
             [3.42207377, 4., 4.57792623],
             [5.15585247, 5.7337787, 6.31170493]])))

    def test_variable_sigma(self):
        im = self.im
        sigmas = hs.signals.BaseSignal(np.array([0., 1.]))
        sigmas.axes_manager.set_signal_dimension(0)
        im.map(gaussian_filter,
               sigma=sigmas, show_progressbar=None)
        nt.assert_true(np.allclose(im.data, np.array(
            [[[0., 1., 2.],
                [3., 4., 5.],
                [6., 7., 8.]],

             [[10.68829507, 11.2662213, 11.84414753],
              [12.42207377, 13., 13.57792623],
              [14.15585247, 14.7337787, 15.31170493]]])))

    def test_axes_argument(self):
        im = self.im
        im.map(rotate, angle=45, reshape=False, show_progressbar=None)
        nt.assert_true(np.allclose(im.data, np.array(
            [[[0., 2.23223305, 0.],
              [0.46446609, 4., 7.53553391],
              [0., 5.76776695, 0.]],

             [[0., 11.23223305, 0.],
              [9.46446609, 13., 16.53553391],
              [0., 14.76776695, 0.]]])))


class TestSignal1D:

    def setup(self):
        self.s = hs.signals.Signal1D(np.arange(0., 6).reshape((2, 3)))

    def test_constant_sigma(self):
        s = self.s
        s.map(gaussian_filter1d, sigma=1, show_progressbar=None)
        nt.assert_true(np.allclose(s.data, np.array(
            ([[0.42207377, 1., 1.57792623],
              [3.42207377, 4., 4.57792623]]))))
