from ipywidgets.widgets.tests import setup_test_comm, teardown_test_comm
import numpy as np

import hyperspy.api as hs


KWARGS = {
    "toolkit": "ipywidgets",
    "display": False,
}


def setup():
    setup_test_comm()


def teardown():
    teardown_test_comm()


def check_axis_attributes(axes_manager, widgets_dict, index, attributes):
    for attribute in attributes:
        assert (widgets_dict["axis{}".format(index)][attribute].value ==
                getattr(axes_manager[index], attribute))


class TestAxes:

    def setup_method(self, method):
        self.s = hs.signals.Signal1D(np.empty((2, 3, 4)))
        am = self.s.axes_manager
        am[0].scale = 0.5
        am[0].name = "a"
        am[0].units = "eV"
        am[1].scale = 1000
        am[1].name = "b"
        am[1].units = "meters"
        am[2].scale = 5
        am[2].name = "c"
        am[2].units = "e"
        am.indices = (2, 1)

    def test_navigation_sliders(self):
        s = self.s
        am = self.s.axes_manager
        wd = s.axes_manager.navigation_sliders(**KWARGS)["ipywidgets"]["wdict"]
        check_axis_attributes(axes_manager=am, widgets_dict=wd, index=0,
                              attributes=("value", "index", "units"))
        check_axis_attributes(axes_manager=am, widgets_dict=wd, index=1,
                              attributes=("value", "index", "units"))
        wd["axis0"]["value"].value = 1.5
        am[0].units = "cm"
        check_axis_attributes(axes_manager=am, widgets_dict=wd, index=0,
                              attributes=("value", "index", "units"))

    def test_axes_manager_gui(self):
        s = self.s
        am = self.s.axes_manager
        wd = s.axes_manager.gui(**KWARGS)["ipywidgets"]["wdict"]
        check_axis_attributes(axes_manager=am, widgets_dict=wd, index=0,
                              attributes=("value", "index", "units",
                                          "index_in_array", "name",
                                          "size", "scale", "offset"))
        check_axis_attributes(axes_manager=am, widgets_dict=wd, index=1,
                              attributes=("value", "index", "units",
                                          "index_in_array", "name", "size",
                                          "scale", "offset"))
        check_axis_attributes(axes_manager=am, widgets_dict=wd, index=2,
                              attributes=("units", "index_in_array",
                                          "name", "size", "scale",
                                          "offset"))
        wd["axis0"]["value"].value = 1.5
        wd["axis0"]["name"].name = "parrot"
        wd["axis0"]["offset"].name = -1
        wd["axis0"]["scale"].name = 1e-10
        wd["axis0"]["units"].value = "cm"
        check_axis_attributes(axes_manager=am, widgets_dict=wd, index=0,
                              attributes=("value", "index", "units",
                                          "index_in_array", "name",
                                          "size", "scale", "offset"))
