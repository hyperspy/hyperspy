import nose.tools

import hyperspy.hspy as hs


class TestlineFit:

    def setUp(self):
        s = hs.signals.EDSSEMSpectrum(range(200))
        s.set_microscope_parameters(beam_energy=100)
        s.axes_manager.signal_axes[0].units = "keV"
        s.axes_manager[-1].offset = 0.150
        s.add_elements(("Al", "Zn"))
        self.m = hs.create_model(s, auto_background=False)

    def test_param(self):
        m = self.m
        nose.tools.assert_equal(len(m), 9)
        nose.tools.assert_equal(len(m.xray_lines), 3)

    def test_fit(self):
        m = self.m
        m.fit()

    def test_get_intensity(self):
        m = self.m


class TestbackgroundFit:

    def setUp(self):
        s = hs.signals.EDSSEMSpectrum(range(200))
        s.set_microscope_parameters(beam_energy=100)
        s.axes_manager.signal_axes[0].units = "keV"
        s.axes_manager[-1].offset = 0.150
        s.add_elements(("Al", "Zn"))
        self.m = hs.create_model(s)

    def test_fit(self):
        m = self.m
        # m.fit()
