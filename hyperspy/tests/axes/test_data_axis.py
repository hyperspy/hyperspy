import copy
import math
from unittest import mock

import nose.tools as nt
import numpy as np

from hyperspy.axes import DataAxis


class TestDataAxis:

    def setUp(self):
        self.axis = DataAxis(size=10, scale=0.1, offset=10)

    def test_value_range_to_indices_in_range(self):
        nt.assert_equal(
            self.axis.value_range_to_indices(
                10.1, 10.8), (1, 8))

    def test_value_range_to_indices_endpoints(self):
        nt.assert_equal(
            self.axis.value_range_to_indices(
                10, 10.9), (0, 9))

    def test_value_range_to_indices_out(self):
        nt.assert_equal(
            self.axis.value_range_to_indices(
                9, 11), (0, 9))

    def test_value_range_to_indices_None(self):
        nt.assert_equal(
            self.axis.value_range_to_indices(
                None, None), (0, 9))

    @nt.raises(ValueError)
    def test_value_range_to_indices_v1_greater_than_v2(self):
        self.axis.value_range_to_indices(2, 1)

    def test_deepcopy(self):
        ac = copy.deepcopy(self.axis)
        ac.offset = 100
        nt.assert_not_equal(self.axis.offset, ac.offset)

    def test_deepcopy_on_trait_change(self):
        ac = copy.deepcopy(self.axis)
        ac.offset = 100
        nt.assert_equal(ac.axis[0], ac.offset)

    def test_value2index_float_in(self):
        nt.assert_equal(self.axis.value2index(10.15), 2)

    def test_value2index_float_end_point_left(self):
        nt.assert_equal(self.axis.value2index(10.), 0)

    def test_value2index_float_end_point_right(self):
        nt.assert_equal(self.axis.value2index(10.9), 9)

    @nt.raises(ValueError)
    def test_value2index_float_out(self):
        self.axis.value2index(11)

    def test_value2index_array_in(self):
        nt.assert_equal(
            self.axis.value2index(np.array([10.15, 10.15])).tolist(),
            [2, 2])

    def test_value2index_array_in_ceil(self):
        nt.assert_equal(
            self.axis.value2index(np.array([10.14, 10.14]),
                                  rounding=math.ceil).tolist(),
            [2, 2])

    def test_value2index_array_in_floor(self):
        nt.assert_equal(
            self.axis.value2index(np.array([10.15, 10.15]),
                                  rounding=math.floor).tolist(),
            [1, 1])

    @nt.raises(ValueError)
    def test_value2index_array_out(self):
        self.axis.value2index(np.array([10, 11]))

    def test_slice_me(self):
        nt.assert_equal(
            self.axis._slice_me(slice(np.float32(10.2), 10.4, 2)),
            slice(2, 4, 2))

    def test_update_from(self):
        ax2 = DataAxis(size=2, units="nm", scale=0.5)
        self.axis.update_from(ax2, attributes=("units", "scale"))
        nt.assert_equal((ax2.units, ax2.scale),
                        (self.axis.units, self.axis.scale))

    def test_value_changed_event(self):
        ax = self.axis
        m = mock.Mock()
        ax.events.value_changed.connect(m.trigger_me)
        ax.value = ax.value
        nt.assert_false(m.trigger_me.called)
        ax.value = ax.value + ax.scale * 0.3
        nt.assert_false(m.trigger_me.called)
        ax.value = ax.value + ax.scale
        nt.assert_true(m.trigger_me.called)

    def test_value_changed_event_continuous(self):
        ax = self.axis
        ax.continuous_value = True
        m = mock.Mock()
        ax.events.value_changed.connect(m.trigger_me_value)
        ax.events.index_changed.connect(m.trigger_me_index)
        ax.value = ax.value
        nt.assert_false(m.trigger_me_value.called)
        ax.value = ax.value + ax.scale * 0.3
        nt.assert_true(m.trigger_me_value.called)
        nt.assert_false(m.trigger_me_index.called)
        ax.value = ax.value + ax.scale
        nt.assert_true(m.trigger_me_index.called)

    def test_index_changed_event(self):
        ax = self.axis
        m = mock.Mock()
        ax.events.index_changed.connect(m.trigger_me)
        ax.index = ax.index
        nt.assert_false(m.trigger_me.called)
        ax.index += 1
        nt.assert_true(m.trigger_me.called)
