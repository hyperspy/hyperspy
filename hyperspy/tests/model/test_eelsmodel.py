import numpy as np
import nose.tools

import hyperspy.hspy as hs


class TestEELSModel:

    def setUp(self):
        s = hs.signals.EELSSpectrum(np.empty(200))
        s.set_microscope_parameters(100, 10, 10)
        s.axes_manager[-1].offset = 150
        s.add_elements(("B", "C"))
        self.m = hs.create_model(s)

    def test_suspend_auto_fsw(self):
        m = self.m
        m["B_K"].fine_structure_width = 140.
        m.suspend_auto_fine_structure_width()
        m.enable_fine_structure()
        m.resolve_fine_structure()
        nose.tools.assert_equal(140, m["B_K"].fine_structure_width)

    def test_resume_fsw(self):
        m = self.m
        m["B_K"].fine_structure_width = 140.
        m.suspend_auto_fine_structure_width()
        m.resume_auto_fine_structure_width()
        window = (m["C_K"].onset_energy.value -
                  m["B_K"].onset_energy.value -
                  hs.preferences.EELS.preedge_safe_window_width)
        m.enable_fine_structure()
        m.resolve_fine_structure()
        nose.tools.assert_equal(window, m["B_K"].fine_structure_width)
