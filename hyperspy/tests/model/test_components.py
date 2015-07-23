import numpy as np
import nose.tools as nt

import hyperspy.hspy as hs
from hyperspy.model import Model


class TestPowerLaw:

    def setUp(self):
        s = hs.signals.Spectrum(np.empty((1024)))
        s.axes_manager[0].offset = 100
        s.axes_manager[0].scale = 0.01
        m = hs.create_model(s)
        m.append(hs.components.PowerLaw())
        m[0].A.value = 10
        m[0].r.value = 4
        self.m = m

    def test_estimate_parameters_binned_only_current(self):
        self.m.spectrum.metadata.Signal.binned = True
        s = self.m.as_signal()
        s.metadata.Signal.binned = True
        g = hs.components.PowerLaw()
        g.estimate_parameters(s,
                              None,
                              None,
                              only_current=True)
        nt.assert_almost_equal(g.A.value, 10.084913947965161)
        nt.assert_almost_equal(g.r.value, 4.0017676988807409)

    def test_estimate_parameters_unbinned_only_current(self):
        self.m.spectrum.metadata.Signal.binned = False
        s = self.m.as_signal()
        s.metadata.Signal.binned = False
        g = hs.components.PowerLaw()
        g.estimate_parameters(s,
                              None,
                              None,
                              only_current=True)
        nt.assert_almost_equal(g.A.value, 10.064378823244837)
        nt.assert_almost_equal(g.r.value, 4.0017522876514304)

    def test_estimate_parameters_binned(self):
        self.m.spectrum.metadata.Signal.binned = True
        s = self.m.as_signal()
        s.metadata.Signal.binned = True
        g = hs.components.PowerLaw()
        g._axes_manager = self.m.spectrum.axes_manager
        g.estimate_parameters(s,
                              None,
                              None,
                              only_current=False)
        nt.assert_almost_equal(g.A.value, 10.084913947965161)
        nt.assert_almost_equal(g.r.value, 4.0017676988807409)

    def test_estimate_parameters_unbinned(self):
        self.m.spectrum.metadata.Signal.binned = False
        s = self.m.as_signal()
        s.metadata.Signal.binned = False
        g = hs.components.PowerLaw()
        g._axes_manager = self.m.spectrum.axes_manager
        g.estimate_parameters(s,
                              None,
                              None,
                              only_current=False)
        nt.assert_almost_equal(g.A.value, 10.064378823244837)
        nt.assert_almost_equal(g.r.value, 4.0017522876514304)


class TestOffset:

    def setUp(self):
        s = hs.signals.Spectrum(np.empty((10)))
        s.axes_manager[0].scale = 0.01
        m = hs.create_model(s)
        m.append(hs.components.Offset())
        m[0].offset.value = 10
        self.m = m

    def test_estimate_parameters_binned(self):
        self.m.spectrum.metadata.Signal.binned = True
        s = self.m.as_signal()
        s.metadata.Signal.binned = True
        g = hs.components.Offset()
        g.estimate_parameters(s,
                              None,
                              None,
                              only_current=True)
        nt.assert_almost_equal(g.offset.value, 10)

    def test_estimate_parameters_unbinned(self):
        self.m.spectrum.metadata.Signal.binned = False
        s = self.m.as_signal()
        s.metadata.Signal.binned = False
        g = hs.components.Offset()
        g.estimate_parameters(s,
                              None,
                              None,
                              only_current=True)
        nt.assert_almost_equal(g.offset.value, 10)


class TestPolynomial:

    def setUp(self):
        s = hs.signals.Spectrum(np.empty((1024)))
        s.axes_manager[0].offset = -5
        s.axes_manager[0].scale = 0.01
        m = hs.create_model(s)
        m.append(hs.components.Polynomial(order=2))
        m[0].coefficients.value = (0.5, 2, 3)
        self.m = m
        s_2d = hs.signals.Spectrum(np.arange(1000).reshape(10, 100))
        self.m_2d = hs.create_model(s_2d)
        self.m_2d.append(m[0])
        s_3d = hs.signals.Spectrum(np.arange(1000).reshape(2, 5, 100))
        self.m_3d = hs.create_model(s_3d)
        self.m_3d.append(m[0])

    def test_estimate_parameters_binned(self):
        self.m.spectrum.metadata.Signal.binned = True
        s = self.m.as_signal()
        s.metadata.Signal.binned = True
        g = hs.components.Polynomial(order=2)
        g.estimate_parameters(s,
                              None,
                              None,
                              only_current=True)
        nt.assert_almost_equal(g.coefficients.value[0], 0.5)
        nt.assert_almost_equal(g.coefficients.value[1], 2)
        nt.assert_almost_equal(g.coefficients.value[2], 3)

    def test_estimate_parameters_unbinned(self):
        self.m.spectrum.metadata.Signal.binned = False
        s = self.m.as_signal()
        s.metadata.Signal.binned = False
        g = hs.components.Polynomial(order=2)
        g.estimate_parameters(s,
                              None,
                              None,
                              only_current=True)
        nt.assert_almost_equal(g.coefficients.value[0], 0.5)
        nt.assert_almost_equal(g.coefficients.value[1], 2)
        nt.assert_almost_equal(g.coefficients.value[2], 3)

    def test_2d_signal(self):
        # This code should run smoothly, any exceptions should trigger failure
        s = self.m_2d.as_signal()
        model = Model(s)
        p = hs.components.Polynomial(order=2)
        model.append(p)
        p.estimate_parameters(s, 0, 100, only_current=False)
        np.testing.assert_allclose(p.coefficients.map['values'],
                                   np.tile([0.5, 2, 3], (10, 1)))

    def test_3d_signal(self):
        # This code should run smoothly, any exceptions should trigger failure
        s = self.m_3d.as_signal()
        model = Model(s)
        p = hs.components.Polynomial(order=2)
        model.append(p)
        p.estimate_parameters(s, 0, 100, only_current=False)
        np.testing.assert_allclose(p.coefficients.map['values'],
                                   np.tile([0.5, 2, 3], (2, 5, 1)))


class TestGaussian:

    def setUp(self):
        s = hs.signals.Spectrum(np.empty((1024)))
        s.axes_manager[0].offset = -5
        s.axes_manager[0].scale = 0.01
        m = hs.create_model(s)
        m.append(hs.components.Gaussian())
        m[0].sigma.value = 0.5
        m[0].centre.value = 1
        m[0].A.value = 2
        self.m = m

    def test_estimate_parameters_binned(self):
        self.m.spectrum.metadata.Signal.binned = True
        s = self.m.as_signal()
        s.metadata.Signal.binned = True
        g = hs.components.Gaussian()
        g.estimate_parameters(s,
                              None,
                              None,
                              only_current=True)
        nt.assert_almost_equal(g.sigma.value, 0.5)
        nt.assert_almost_equal(g.A.value, 2)
        nt.assert_almost_equal(g.centre.value, 1)

    def test_estimate_parameters_unbinned(self):
        self.m.spectrum.metadata.Signal.binned = False
        s = self.m.as_signal()
        s.metadata.Signal.binned = False
        g = hs.components.Gaussian()
        g.estimate_parameters(s,
                              None,
                              None,
                              only_current=True)
        nt.assert_almost_equal(g.sigma.value, 0.5)
        nt.assert_almost_equal(g.A.value, 2)
        nt.assert_almost_equal(g.centre.value, 1)
