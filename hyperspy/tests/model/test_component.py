import numpy as np
import nose.tools
from hyperspy.component import Component
from hyperspy.axes import AxesManager


class TestMultidimensionalActive:

    def setUp(self):
        self.c = Component(["parameter"])
        self.c._axes_manager = AxesManager([{"size": 3,
                                             "navigate": True},
                                            {"size": 2,
                                             "navigate": True}])

    def test_enable_pixel_switching_current_on(self):
        c = self.c
        c._axes_manager.indices = (1, 1)
        c.active = True
        c.active_is_multidimensional = True
        nose.tools.assert_true(np.all(c._active_array))

    def test_enable_pixel_switching_current_off(self):
        c = self.c
        c._axes_manager.indices = (1, 1)
        c.active = False
        c.active_is_multidimensional = True
        nose.tools.assert_false(self.c.active)

    def test_disable_pixel_switching(self):
        c = self.c
        c.active = True
        c.active_is_multidimensional = True
        c.active_is_multidimensional = False
        nose.tools.assert_is(c._active_array, None)

    def test_disable_pixel_switching_current_on(self):
        c = self.c
        c._axes_manager.indices = (1, 1)
        c.active = True
        c.active_is_multidimensional = True
        c.active_is_multidimensional = False
        nose.tools.assert_true(c.active)

    def test_disable_pixel_switching_current_off(self):
        c = self.c
        c._axes_manager.indices = (1, 1)
        c.active = False
        c.active_is_multidimensional = True
        c.active_is_multidimensional = False
        nose.tools.assert_false(c.active)
