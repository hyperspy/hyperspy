
from ipywidgets.widgets.tests import setup_test_comm, teardown_test_comm
from numpy.random import random

import hyperspy.api as hs
from hyperspy.component import Component, Parameter
from hyperspy.gui_ipywidgets.tests.utils import KWARGS


def test_parameter():
    p = Parameter()
    p.bmin = -5
    p.bmax = 10
    p.value = 1.5
    wd = p.gui(**KWARGS)["ipywidgets"]["wdict"]
    assert wd["value"].value == p.value
    assert wd["min"].value == p.bmin
    assert wd["max"].value == p.bmax
    wd["value"].value = -4
    p.bmin = -10
    p.bmax = 0
    assert wd["value"].value == p.value
    assert wd["min"].value == p.bmin
    assert wd["max"].value == p.bmax
