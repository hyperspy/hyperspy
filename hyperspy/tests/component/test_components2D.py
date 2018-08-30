import numpy as np

from numpy.testing import assert_allclose

import hyperspy.api as hs

GAUSSIAN2D_EXPR = \
    "exp(-((x-x0)**2 / (2 * sx ** 2) + (y-y0)**2 / (2 * sy ** 2)))"


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


class TestExpression2D:

    def setup_method(self, method):
        self.array0 = np.array([[1.69189792e-10, 3.72665317e-06, 1.50343919e-03,
                                 1.11089965e-02, 1.50343919e-03],
                                [2.06115362e-09, 4.53999298e-05, 1.83156389e-02,
                                 1.35335283e-01, 1.83156389e-02],
                                [9.23744966e-09, 2.03468369e-04, 8.20849986e-02,
                                 6.06530660e-01, 8.20849986e-02],
                                [1.52299797e-08, 3.35462628e-04, 1.35335283e-01,
                                 1.00000000e+00, 1.35335283e-01],
                                [9.23744966e-09, 2.03468369e-04, 8.20849986e-02,
                                 6.06530660e-01, 8.20849986e-02]])

    def test_with_rotation(self):
        g = hs.model.components2D.Expression(
            GAUSSIAN2D_EXPR, name="gaussian2d", add_rotation=True,
            position=("x0", "y0"),)
        g.rotation_angle.value = np.radians(20)
        g.sy.value = .1
        g.sx.value = 0.5
        g.sy.value = 1
        g.x0.value = 1
        g.y0.value = 1
        l = np.linspace(-2, 2, 5)
        x, y = np.meshgrid(l, l)
        assert_allclose(
            g.function(x, y),
            np.array([[6.68025544e-06, 8.55249949e-04, 6.49777231e-03,
                       2.92959352e-03, 7.83829650e-05],
                      [1.87435095e-05, 5.01081110e-03, 7.94944111e-02,
                       7.48404874e-02, 4.18126855e-03],
                      [1.43872256e-05, 8.03140097e-03, 2.66058420e-01,
                       5.23039095e-01, 6.10186705e-02],
                      [3.02114453e-06, 3.52162323e-03, 2.43604733e-01,
                       1.00000000e+00, 2.43604733e-01],
                      [1.73553850e-07, 4.22437802e-04, 6.10186705e-02,
                       5.23039095e-01, 2.66058420e-01]])
        )
        assert_allclose(
            g.grad_sx(x, y),
            np.array([[1.20880828e-04, 3.17536657e-03, 1.04030848e-03,
                       2.17878745e-02, 2.00221335e-03],
                      [4.99616174e-04, 4.02985917e-02, 2.05882951e-02,
                       2.47378292e-01, 7.18407981e-02],
                      [5.30432684e-04, 1.12636930e-01, 5.34932340e-01,
                       4.32214309e-01, 6.38979998e-01],
                      [1.47232153e-04, 7.62766363e-02, 1.31908984e+00,
                       0.00000000e+00, 1.31908984e+00],
                      [1.08041078e-05, 1.30732490e-02, 6.38979998e-01,
                       4.32214309e-01, 5.34932340e-01]])

        )

    def test_with_rotation_center_tuple(self):
        g = hs.model.components2D.Expression(
            GAUSSIAN2D_EXPR, "gaussian2d", add_rotation=True,
            position=("x0", "y0"), module="numpy", rotation_center=(1, 2))
        g.rotation_angle.value = np.radians(30)
        g.sx.value = .1
        g.sy.value = .1
        g.x0.value = 0
        g.y0.value = 0
        l = np.linspace(-2, 2, 5)
        x, y = np.meshgrid(l, l)
        assert_allclose(
            g.function(x, y),
            np.array([[1.77220718e-208, 1.97005871e-181, 1.00498099e-181,
                       2.35261319e-209, 2.52730239e-264],
                      [9.94006394e-101, 6.64536056e-081, 2.03874037e-088,
                       2.87024715e-123, 1.85434555e-185],
                      [1.07437572e-033, 4.31966683e-021, 7.96999992e-036,
                       6.74808249e-078, 2.62190196e-147],
                      [2.23777036e-007, 5.41095940e-002, 6.00408779e-024,
                       3.05726999e-073, 7.14388660e-150],
                      [8.98187837e-022, 1.30614268e-023, 8.71621777e-053,
                       2.66919022e-109, 3.75098197e-193]])
        )

    def test_with_rotation_no_position(self):
        g = hs.model.components2D.Expression(
            GAUSSIAN2D_EXPR, "gaussian2d", add_rotation=True, module="numpy")
        g.rotation_angle.value = np.radians(45)
        g.sx.value = .5
        g.sy.value = .1
        g.x0.value = 0
        g.y0.value = 0
        l = np.linspace(-2, 2, 5)
        x, y = np.meshgrid(l, l)
        assert_allclose(
            g.function(x, y),
            np.array([[9.64172248e-175, 5.46609733e-099, 5.03457536e-045,
                       7.53374790e-013, 1.83156389e-002],
                      [1.89386646e-098, 3.13356463e-044, 8.42346375e-012,
                       3.67879441e-001, 2.61025584e-012],
                      [2.63952332e-044, 1.27462190e-011, 1.00000000e+000,
                       1.27462190e-011, 2.63952332e-044],
                      [2.61025584e-012, 3.67879441e-001, 8.42346375e-012,
                       3.13356463e-044, 1.89386646e-098],
                      [1.83156389e-002, 7.53374790e-013, 5.03457536e-045,
                       5.46609733e-099, 9.64172248e-175]])
        )

    def test_with_rotation_no_position_init_values(self):
        g = hs.model.components2D.Expression(
            GAUSSIAN2D_EXPR, "gaussian2d", add_rotation=True, module="numpy",
            sx=.5, sy=.1, x0=0, y0=0, rotation_angle=np.radians(45))
        l = np.linspace(-2, 2, 5)
        x, y = np.meshgrid(l, l)
        assert_allclose(
            g.function(x, y),
            np.array([[9.64172248e-175, 5.46609733e-099, 5.03457536e-045,
                       7.53374790e-013, 1.83156389e-002],
                      [1.89386646e-098, 3.13356463e-044, 8.42346375e-012,
                       3.67879441e-001, 2.61025584e-012],
                      [2.63952332e-044, 1.27462190e-011, 1.00000000e+000,
                       1.27462190e-011, 2.63952332e-044],
                      [2.61025584e-012, 3.67879441e-001, 8.42346375e-012,
                       3.13356463e-044, 1.89386646e-098],
                      [1.83156389e-002, 7.53374790e-013, 5.03457536e-045,
                       5.46609733e-099, 9.64172248e-175]])
        )

    def test_no_rotation(self):
        g = hs.model.components2D.Expression(
            GAUSSIAN2D_EXPR, name="gaussian2d", add_rotation=False,
            position=("x0", "y0"),)
        g.sy.value = .1
        g.sx.value = 0.5
        g.sy.value = 1
        g.x0.value = 1
        g.y0.value = 1
        l = np.linspace(-2, 2, 5)
        x, y = np.meshgrid(l, l)
        assert_allclose(g.function(x, y), self.array0)
        assert_allclose(
            g.grad_sx(x, y),
            np.array([[1.21816650e-08, 1.19252902e-04, 1.20275135e-02,
                       0.00000000e+00, 1.20275135e-02],
                      [1.48403061e-07, 1.45279775e-03, 1.46525111e-01,
                       0.00000000e+00, 1.46525111e-01],
                      [6.65096376e-07, 6.51098781e-03, 6.56679989e-01,
                       0.00000000e+00, 6.56679989e-01],
                      [1.09655854e-06, 1.07348041e-02, 1.08268227e+00,
                       0.00000000e+00, 1.08268227e+00],
                      [6.65096376e-07, 6.51098781e-03, 6.56679989e-01,
                       0.00000000e+00, 6.56679989e-01]])

        )

    def test_no_function_nd(self):
        g = hs.model.components2D.Expression(
            GAUSSIAN2D_EXPR, name="gaussian2d", add_rotation=False,
            position=("x0", "y0"),)
        g.sy.value = .1
        g.sx.value = 0.5
        g.sy.value = 1
        g.x0.value = 1
        g.y0.value = 1
        l = np.linspace(-2, 2, 5)
        x, y = np.meshgrid(l, l)
        assert_allclose(g.function_nd(x, y), self.array0)

    def test_no_function_nd_signal(self):
        g = hs.model.components2D.Expression(
            GAUSSIAN2D_EXPR, name="gaussian2d", add_rotation=False,
            position=("x0", "y0"),)
        g.sy.value = .1
        g.sx.value = 0.5
        g.sy.value = 1
        g.x0.value = 1
        g.y0.value = 1
        l = np.arange(0, 3)
        x, y = np.meshgrid(l, l)
        s = hs.signals.Signal2D(g.function(x, y))
        s2 = hs.stack([s]*2)
        m = s2.create_model()
        m.append(g)
        m.multifit()
        res = g.function_nd(x, y)
        assert res.shape == (2, 3, 3)
        assert_allclose(res, s2.data)
