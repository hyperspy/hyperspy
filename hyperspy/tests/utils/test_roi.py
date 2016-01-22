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

    def test_rect_image(self):
        s = self.s_i
        s.axes_manager[0].scale = 0.2
        s.axes_manager[1].scale = 0.8
        r = RectangularROI(left=2.3, top=5.6, right=3.5, bottom=12.2)
        sr = r(s)
        scale0 = s.axes_manager[0].scale
        scale1 = s.axes_manager[1].scale
        n = ((round(2.3 / scale0), round(3.5 / scale0),),
             (round(5.6 / scale1), round(12.2 / scale1),))
        nt.assert_equal(sr.axes_manager.navigation_shape,
                        (n[0][1] - n[0][0], n[1][1] - n[1][0]))
        np.testing.assert_equal(
            sr.data, s.data[n[1][0]:n[1][1], n[0][0]:n[0][1], ...])

    def test_circle_spec(self):
        s = self.s_s
        s.data = np.ones_like(s.data)
        r = CircleROI(20, 25, 20)
        sr = r(s)
        scale = s.axes_manager[0].scale
        n = int(round(40 / scale))
        nt.assert_equal(sr.axes_manager.navigation_shape, (n, n))
        # Check that mask is same for all images:
        for i in xrange(n):
            for j in xrange(n):
                nt.assert_true(np.all(sr.data.mask[j, i, :] == True) or
                               np.all(sr.data.mask[j, i, :] == False))
        # Check that the correct elements has been masked out:
        mask = sr.data.mask[:, :, 0]
        print mask   # To help debugging, this shows the shape of the mask
        np.testing.assert_array_equal(
            np.where(mask.flatten())[0],
            [0,  1,  6,  7,  8, 15, 48, 55, 56, 57, 62, 63])
        # Check that mask works for sum
        nt.assert_equal(np.sum(sr.data), (n**2 - 3*4)*4)
