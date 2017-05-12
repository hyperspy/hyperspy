
from numpy.random import random, uniform
import ipywidgets

import hyperspy.api as hs
from hyperspy.gui_ipywidgets.tests.utils import KWARGS


def test_preferences():
    wd = hs.preferences.gui(**KWARGS)["ipywidgets"]["wdict"]
    for tabkey, tabvalue in wd.items():
        if tabkey.startswith("tab_"):
            for key, value in tabvalue.items():
                assert getattr(
                    getattr(hs.preferences, tabkey[4:]), key) == value.value
                if isinstance(value, ipywidgets.Checkbox):
                    value.value = not value
                elif isinstance(value, ipywidgets.FloatText):
                    value.value = random()
                elif isinstance(value, ipywidgets.Text):
                    value.value = "qwerty"
                elif isinstance(value, ipywidgets.FloatSlider):
                    value.value = uniform(low=value.min, high=value.max)
                elif isinstance(value, ipywidgets.Dropdown):
                    options = set(value.options) - set(value.value)
                    value.value = options.pop()
                assert getattr(
                    getattr(hs.preferences, tabkey[4:]), key) == value.value
