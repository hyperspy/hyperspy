
from ipywidgets.widgets.tests import setup_test_comm, teardown_test_comm
from numpy.random import random

import hyperspy.api as hs
from hyperspy.component import Component, Parameter
from hyperspy.gui_ipywidgets.tests.utils import KWARGS


def test_parameter():
    p = Parameter()
    p.bmin = None
    p.bmax = 10
    p.value = 1.5
    wd = p.gui(**KWARGS)["ipywidgets"]["wdict"]
    assert wd["value"].value == p.value
    assert wd["max"].value == p.bmax
    wd["value"].value = -4
    p.bmin = -10
    p.bmax = 0
    assert wd["value"].value == p.value
    assert wd["min"].value == p.bmin
    assert wd["max"].value == p.bmax


def test_multivalue_parameter():
    p = Parameter()
    p._number_of_elements = 2
    p.value = (1.5, 3)
    wd = p.gui(**KWARGS)["ipywidgets"]["wdict"]
    assert wd["element0"]["value"].value == p.value[0]
    assert wd["element1"]["value"].value == p.value[1]
    wd["element0"]["value"].value = -4
    wd["element1"]["value"].value = -3
    assert wd["element0"]["value"].value == p.value[0]
    assert wd["element1"]["value"].value == p.value[1]


def test_component():
    c = Component(["a", "b"])
    c.a.value = 3
    c.b.value = 2
    c.active = False
    wd = c.gui(**KWARGS)["ipywidgets"]["wdict"]
    assert wd["active"].value == c.active
    assert wd["parameter_a"]["value"].value == c.a.value
    assert wd["parameter_b"]["value"].value == c.b.value
    wd["active"].value = True
    wd["parameter_b"]["value"].value = 34
    wd["parameter_a"]["value"].value = 31
    assert wd["active"].value == c.active
    assert wd["parameter_a"]["value"].value == c.a.value
    assert wd["parameter_b"]["value"].value == c.b.value


def test_model():
    s = hs.signals.Signal1D([0])
    m = s.create_model()
    c = Component(["a", "b"])
    d = Component(["a", "b"])
    m.extend((c, d))
    c.name = "c"
    d.name = "d"
    c.active = False
    d.active = True
    wd = m.gui(**KWARGS)["ipywidgets"]["wdict"]
    assert wd["component_c"]["active"].value == c.active
    assert wd["component_d"]["active"].value == d.active
