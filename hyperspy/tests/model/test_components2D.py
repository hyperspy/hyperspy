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
        im.axes_manager[1].scale = 0.02
        im.axes_manager[1].offset = -10
        im = hs.utils.stack([im]*2)
        self.im = im

    def test_fitting(self):
        im = self.im
        m = im.create_model()
        m.append(hs.model.components2d.Gaussian2D)
        m.fit()
        nt.assert_almost_equal(im)


class TestScalableFixedPattern2D:

    def setUp(self):
        im = hs.signals.Image((np.linspace(0., 100., 10), np.linspace(0., 1., 10)) )
        im1 = hs.signals.Image(np.linspace(0., 1., 10))
        im.axes_manager[0].scale = 0.1
        im.axes_manager[0].scale = 0.1
        self.im = im
        self.pattern = im1

    def test_fitting(self):
        im = self.im
        im1 = self.pattern
        m = im.create_model()
        fp = hs.model.components2d.ScalableFixedPattern2d(im1)
        m.append(fp)
        m.fit()
        nt.assert_almost_equal(fp.yscale.value, 100, delta=0.1)
