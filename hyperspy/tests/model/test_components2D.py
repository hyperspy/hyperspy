import numpy as np
import nose.tools as nt

import hyperspy.api as hs


class TestGaussian2D:

    def setUp(self):
        g = hs.model.components2d.Gaussian2D(
            centre_x=-5.,
            centre_y=-5.,
            sigma_x=1.,
            sigma_y=2.)
        x = np.arange(-10, 10, 0.01)
        y = np.arange(-10, 10, 0.01)
        X, Y = np.meshgrid(x, y)
        im = hs.signals.Image(g.function(X, Y))
        im.axes_manager[0].scale = 0.01
        im.axes_manager[0].offset = -10
        im.axes_manager[1].scale = 0.01
        im.axes_manager[1].offset = -10
        self.im = im

    def test_fitting(self):
        im = self.im
        m = im.create_model()
        gt = hs.model.components2d.Gaussian2D(centre_x=-4.5,
                                              centre_y=-4.5,
                                              sigma_x=0.5,
                                              sigma_y=1.5)
        m.append(gt)
        m.fit()
        nt.assert_almost_equal(gt.centre_x.value, -5.)
        nt.assert_almost_equal(gt.centre_y.value, -5.)
        nt.assert_almost_equal(gt.sigma_x.value, 1.)
        nt.assert_almost_equal(gt.sigma_y.value, 2.)
