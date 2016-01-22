import nose.tools as nt

import numpy as np

from hyperspy.signals import Image, Spectrum
from hyperspy.roi import (Point1DROI, Point2DROI, SpanROI, RectangularROI,
                          Line2DROI, CircleROI)


class TestROIs():

    def setUp(self):
        np.random.seed(0)  # Same random every time, Line2DROi test requires it
        self.s_s = Spectrum(np.random.rand(50, 60, 4))
        self.s_s.axes_manager[0].scale = 5
        self.s_s.axes_manager[0].units = 'nm'
        self.s_s.axes_manager[1].scale = 5
        self.s_s.axes_manager[1].units = 'nm'

        # 4D dataset
        self.s_i = Image(np.random.rand(100, 100, 4, 4))

    def test_point1d_spectrum(self):
        s = self.s_s
        r = Point1DROI(35)
        sr = r(s)
        scale = s.axes_manager[0].scale
        nt.assert_equal(sr.axes_manager.navigation_shape,
                        s.axes_manager.navigation_shape[1:])
        np.testing.assert_equal(
            sr.data, s.data[:, 35/scale, ...])

    def test_point1d_spectrum_ronded_coord(self):
        s = self.s_s
        r = Point1DROI(37.)
        sr = r(s)
        scale = s.axes_manager[0].scale
        np.testing.assert_equal(
            sr.data, s.data[:, round(37/scale), ...])
        r = Point1DROI(39.)
        sr = r(s)
        np.testing.assert_equal(
            sr.data, s.data[:, round(39/scale), ...])

    def test_point1d_image(self):
        s = self.s_i
        r = Point1DROI(35)
        sr = r(s)
        scale = s.axes_manager[0].scale
        nt.assert_equal(sr.axes_manager.navigation_shape,
                        s.axes_manager.navigation_shape[1:])
        np.testing.assert_equal(
            sr.data, s.data[:, 35/scale, ...])

    def test_point2d_image(self):
        s = self.s_i
        r = Point2DROI(35, 40)
        sr = r(s)
        scale = s.axes_manager[0].scale
        nt.assert_equal(sr.axes_manager.navigation_shape,
                        s.axes_manager.navigation_shape[2:])
        np.testing.assert_equal(
            sr.data, s.data[40/scale, 35/scale, ...])

    def test_point2d_image_sig(self):
        s = self.s_i
        r = Point2DROI(1, 2)
        sr = r(s, axes=s.axes_manager.signal_axes)
        scale = s.axes_manager.signal_axes[0].scale
        nt.assert_equal(sr.axes_manager.signal_shape,
                        s.axes_manager.signal_shape[2:])
        np.testing.assert_equal(
            sr.data, s.data[..., 2/scale, 1/scale])

    def test_span_spectrum_nav(self):
        s = self.s_s
        r = SpanROI(15, 30)
        sr = r(s)
        scale = s.axes_manager[0].scale
        n = (30 - 15) / scale
        nt.assert_equal(sr.axes_manager.navigation_shape,
                        (n, ) + s.axes_manager.navigation_shape[1:])
        np.testing.assert_equal(
            sr.data, s.data[:, 15/scale:30/scale, ...])

    def test_span_spectrum_sig(self):
        s = self.s_s
        r = SpanROI(1, 3)
        sr = r(s, axes=s.axes_manager.signal_axes)
        scale = s.axes_manager.signal_axes[0].scale
        n = (3 - 1) / scale
        nt.assert_equal(sr.axes_manager.signal_shape, (n, ))
        np.testing.assert_equal(sr.data, s.data[..., 1/scale:3/scale])
