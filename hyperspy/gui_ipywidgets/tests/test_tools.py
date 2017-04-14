import numpy as np
import numpy.testing
import numpy.random

import hyperspy.api as hs
from hyperspy.gui_ipywidgets.tests.utils import KWARGS
from hyperspy.signal_tools import Signal1DCalibration



class TestTools:

    def setup_method(self, method):
        self.s = hs.signals.Signal1D(1 + np.arange(100)**2)
        self.s.axes_manager[0].offset = 10
        self.s.axes_manager[0].scale = 2
        self.s.axes_manager[0].units = "m"

    def test_calibrate(self):
        s = self.s
        cal = Signal1DCalibration(s)
        cal.ss_left_value = 10
        cal.ss_right_value = 30
        wd = cal.gui(**KWARGS)["ipywidgets"]["wdict"]
        wd["new_left"].value = 0
        wd["new_right"].value = 10
        wd["units"].value = "nm"
        assert wd["offset"].value == 0
        assert wd["scale"].value == 1
        wd["apply_button"]._click_handlers(wd["apply_button"])    # Trigger it
        assert s.axes_manager[0].scale == 1
        assert s.axes_manager[0].offset == 0
        assert s.axes_manager[0].units == "nm"

    def test_calibrate_from_s(self):
        s = self.s
        wd = s.calibrate(**KWARGS)["ipywidgets"]["wdict"]
        wd["left"].value = 10
        wd["right"].value = 30
        wd["new_left"].value = 0
        wd["new_right"].value = 10
        wd["units"].value = "nm"
        assert wd["offset"].value == 0
        assert wd["scale"].value == 1
        wd["apply_button"]._click_handlers(wd["apply_button"])    # Trigger it
        assert s.axes_manager[0].scale == 1
        assert s.axes_manager[0].offset == 0
        assert s.axes_manager[0].units == "nm"

    def test_smooth_sg(self):
        s = self.s
        s.add_gaussian_noise(0.1)
        s2 = s.deepcopy()
        wd = s.smooth_savitzky_golay(**KWARGS)["ipywidgets"]["wdict"]
        wd["window_length"].value = 11
        wd["polynomial_order"].value = 5
        wd["differential_order"].value = 1
        wd["color"].value = "red"
        wd["apply_button"]._click_handlers(wd["apply_button"])    # Trigger it
        s2.smooth_savitzky_golay(polynomial_order=5, window_length=11,
                                 differential_order=1)
        np.testing.assert_allclose(s.data, s2.data)

    def test_smooth_lowess(self):
        s = self.s
        s.add_gaussian_noise(0.1)
        s2 = s.deepcopy()
        wd = s.smooth_lowess(**KWARGS)["ipywidgets"]["wdict"]
        wd["smoothing_parameter"].value = 0.9
        wd["number_of_iterations"].value = 3
        wd["color"].value = "red"
        wd["apply_button"]._click_handlers(wd["apply_button"])    # Trigger it
        s2.smooth_lowess(smoothing_parameter=0.9, number_of_iterations=3)
        np.testing.assert_allclose(s.data, s2.data)

    def test_smooth_tv(self):
        s = self.s
        s.add_gaussian_noise(0.1)
        s2 = s.deepcopy()
        wd = s.smooth_tv(**KWARGS)["ipywidgets"]["wdict"]
        wd["smoothing_parameter"].value = 300
        wd["color"].value = "red"
        wd["apply_button"]._click_handlers(wd["apply_button"])    # Trigger it
        s2.smooth_tv(smoothing_parameter=300)
        np.testing.assert_allclose(s.data, s2.data)

    def test_filter_butterworth(self):
        s = self.s
        s.add_gaussian_noise(0.1)
        s2 = s.deepcopy()
        wd = s.filter_butterworth(**KWARGS)["ipywidgets"]["wdict"]
        wd["cutoff"].value = 0.5
        wd["order"].value = 3
        wd["type"].value = "high"
        wd["color"].value = "red"
        wd["apply_button"]._click_handlers(wd["apply_button"])    # Trigger it
        s2.filter_butterworth(
            cutoff_frequency_ratio=0.5,
            order=3,
            type="high")
        np.testing.assert_allclose(s.data, s2.data)

    def test_remove_background(self):
        s = self.s
        s.add_gaussian_noise(0.1)
        s2 = s.remove_background(
            signal_range=(15., 50.),
            background_type='Polynomial',
            polynomial_order=2,
            fast=False,)
        wd = s.remove_background(**KWARGS)["ipywidgets"]["wdict"]
        assert wd["polynomial_order"].layout.display == "none" # not visible
        wd["background_type"].value = "Polynomial"
        assert wd["polynomial_order"].layout.display == "" # visible
        wd["polynomial_order"].value = 2
        wd["fast"].value = False
        wd["left"].value = 15.
        wd["right"].value = 50.
        wd["apply_button"]._click_handlers(wd["apply_button"])    # Trigger it
        np.testing.assert_allclose(s.data, s2.data)
