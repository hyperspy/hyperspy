import numpy as np
import numpy.random

import hyperspy.api as hs
from hyperspy.gui_ipywidgets.tests.utils import KWARGS
from hyperspy.signal_tools import Signal1DCalibration


class TestTools:

    def setup_method(self, method):
        self.s = hs.signals.Signal1D(np.arange(100))
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
