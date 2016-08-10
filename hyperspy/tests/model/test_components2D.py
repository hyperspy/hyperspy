import numpy as np
import nose.tools as nt

import hyperspy.api as hs


class TestGaussian2D:

    def setUp(self):
        g = hs.model.components2D.Gaussian2D(
            centre_x=-5.,
            centre_y=-5.,
            sigma_x=1.,
            sigma_y=2.)
        x = np.arange(-10, 10, 0.01)
        y = np.arange(-10, 10, 0.01)
        X, Y = np.meshgrid(x, y)
        gt = g.function(X, Y)
        self.g = g
        self.gt = gt

    def test_values(self):
        gt = self.gt
        g = self.g
        nt.assert_almost_equal(g.fwhm_x, 2.35482004503)
        nt.assert_almost_equal(g.fwhm_y, 4.70964009006)
        nt.assert_almost_equal(gt.max(), 0.0795774715459)
        nt.assert_almost_equal(gt.argmax(axis=0)[0], 500)
        nt.assert_almost_equal(gt.argmax(axis=1)[0], 500)
