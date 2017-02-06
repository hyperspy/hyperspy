import numpy as np

from numpy.testing import assert_allclose

import hyperspy.api as hs


class TestGaussian2D:

    def setup_method(self, method):
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
        assert_allclose(g.fwhm_x, 2.35482004503)
        assert_allclose(g.fwhm_y, 4.70964009006)
        assert_allclose(gt.max(), 0.0795774715459)
        assert_allclose(gt.argmax(axis=0)[0], 500)
        assert_allclose(gt.argmax(axis=1)[0], 500)
