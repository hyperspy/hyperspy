from ipywidgets.widgets.tests import setup_test_comm, teardown_test_comm
from numpy.random import random

import hyperspy.api as hs
from hyperspy.gui_ipywidgets.tests.utils import KWARGS
from hyperspy._signals.eels import EELSTEMParametersUI
from hyperspy._signals.eds_sem import EDSSEMParametersUI
from hyperspy._signals.eds_tem import EDSTEMParametersUI


def setup():
    setup_test_comm()


def teardown():
    teardown_test_comm()


class TestSetMicroscopeParameters:

    def setup_method(self, method):
        self.s = hs.signals.Signal1D((2, 3, 4))

    def _perform_t(self, signal_type):
        s = self.s
        s.set_signal_type(signal_type)
        md = s.metadata
        wd = s.set_microscope_parameters(**KWARGS)["ipywidgets"]["wdict"]
        if signal_type == "EELS":
            mapping = EELSTEMParametersUI.mapping
        elif signal_type == "EDS_SEM":
            mapping = EDSSEMParametersUI.mapping
        elif signal_type == "EDS_TEM":
            mapping = EDSTEMParametersUI.mapping
        for key, widget in wd.items():
            if "button" not in key:
                widget.value = random()
        button = wd["store_button"]
        button._click_handlers(button)    # Trigger it
        for item, name in mapping.items():
            assert md.get_item(item) == wd[name].value

    def test_eels(self):
        self._perform_t(signal_type="EELS")

    def test_eds_tem(self):
        self._perform_t(signal_type="EDS_TEM")

    def test_eds_sem(self):
        self._perform_t(signal_type="EDS_SEM")
