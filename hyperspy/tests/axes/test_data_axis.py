import copy
import math
from unittest import mock

import numpy as np
from numpy.testing import assert_allclose
import traits.api as t
import pytest

from hyperspy.axes import BaseDataAxis, DataAxis, LinearDataAxis
from hyperspy.misc.test_utils import assert_deep_almost_equal


class TestBaseDataAxis:

    def setup_method(self, method):
        self.axis = BaseDataAxis()

    def test_initialisation_BaseDataAxis_default(self):
        with pytest.raises(AttributeError):
            assert self.axis.index_in_array is None
        assert self.axis.name is t.Undefined
        assert self.axis.units is t.Undefined
        assert self.axis.navigate is t.Undefined
        assert not self.axis.is_linear

    def test_initialisation_BaseDataAxis(self):
        axis = BaseDataAxis(name='named axis', units='s', navigate=True)
        assert axis.name == 'named axis'
        assert axis.units == 's'
        assert axis.navigate
        assert not self.axis.is_linear
        assert_deep_almost_equal(axis.get_axis_dictionary(),
                                 {'name': 'named axis',
                                  'units': 's',
                                  'navigate': True})


class TestDataAxis:

    def setup_method(self, method):
        self.axis = DataAxis(axis=np.arange(16)**2)

    def test_axis_value(self):
        assert_allclose(self.axis.axis, np.arange(16)**2)
        assert self.axis.size == 16
        assert not self.axis.is_linear

    def test_update_axes(self):
        values = np.arange(20)**2
        self.axis.update_axis(values.tolist())
        assert self.axis.size == 20
        assert_allclose(self.axis.axis, values)

    def test_update_axes2(self):
        values = np.array([3, 4, 10, 40])
        self.axis.update_axis(values)
        assert_allclose(self.axis.axis, values)

    def test_update_axis_from_list(self):
        values = np.arange(16)**2
        self.axis.update_axis(values.tolist())
        assert_allclose(self.axis.axis, values)

    def test_unsorted_axis(self):
        with pytest.raises(ValueError):
            DataAxis(axis=np.array([10, 40, 1, 30, 20]))

    def test_index_changed_event(self):
        ax = self.axis
        m = mock.Mock()
        ax.events.index_changed.connect(m.trigger_me)
        ax.index = ax.index
        assert not m.trigger_me.called
        ax.index += 1
        assert m.trigger_me.called

    def test_value_changed_event(self):
        ax = self.axis
        m = mock.Mock()
        ax.events.value_changed.connect(m.trigger_me)
        ax.value = ax.value
        assert not m.trigger_me.called
        ax.value = ax.value + (ax.axis[1] - ax.axis[0]) / 2
        assert not m.trigger_me.called
        ax.value = ax.axis[1]
        assert m.trigger_me.called

    def test_deepcopy(self):
        ac = copy.deepcopy(self.axis)
        np.testing.assert_allclose(ac.axis, np.arange(16)**2)
        ac.name = 'name changed'
        assert ac.name == 'name changed'
        assert self.axis.name != ac.name

    def test_slice_me(self):
        assert self.axis._slice_me(slice(1, 5)) == slice(1, 5)
        assert self.axis.size == 4
        np.testing.assert_allclose(self.axis.axis, np.arange(1, 5)**2)

    def test_slice_me_step(self):
        assert self.axis._slice_me(slice(0, 10, 2)) == slice(0, 10, 2)
        assert self.axis.size == 5
        np.testing.assert_allclose(self.axis.axis, np.arange(0, 10, 2)**2)


class TestLinearDataAxis:

    def setup_method(self, method):
        self.axis = LinearDataAxis(size=10, scale=0.1, offset=10)

    def test_value_range_to_indices_in_range(self):
        assert self.axis.is_linear
        assert (
            self.axis.value_range_to_indices(
                10.1, 10.8) == (1, 8))

    def test_value_range_to_indices_endpoints(self):
        assert (
            self.axis.value_range_to_indices(
                10, 10.9) == (0, 9))

    def test_value_range_to_indices_out(self):
        assert (
            self.axis.value_range_to_indices(
                9, 11) == (0, 9))

    def test_value_range_to_indices_None(self):
        assert (
            self.axis.value_range_to_indices(
                None, None) == (0, 9))

    def test_value_range_to_indices_v1_greater_than_v2(self):
        with pytest.raises(ValueError):
            self.axis.value_range_to_indices(2, 1)

    def test_deepcopy(self):
        ac = copy.deepcopy(self.axis)
        ac.offset = 100
        assert self.axis.offset != ac.offset

    def test_deepcopy_on_trait_change(self):
        ac = copy.deepcopy(self.axis)
        ac.offset = 100
        assert ac.axis[0] == ac.offset

    def test_value2index_float_in(self):
        assert self.axis.value2index(10.15) == 2

    def test_value2index_float_end_point_left(self):
        assert self.axis.value2index(10.) == 0

    def test_value2index_float_end_point_right(self):
        assert self.axis.value2index(10.9) == 9

    def test_value2index_float_out(self):
        with pytest.raises(ValueError):
            self.axis.value2index(11)

    def test_value2index_array_in(self):
        assert (
            self.axis.value2index(np.array([10.15, 10.15])).tolist() ==
            [2, 2])

    def test_value2index_array_in_ceil(self):
        assert (
            self.axis.value2index(np.array([10.14, 10.14]),
                                  rounding=math.ceil).tolist() ==
            [2, 2])

    def test_value2index_array_in_floor(self):
        assert (
            self.axis.value2index(np.array([10.15, 10.15]),
                                  rounding=math.floor).tolist() ==
            [1, 1])

    def test_value2index_array_out(self):
        with pytest.raises(ValueError):
            self.axis.value2index(np.array([10, 11]))

    def test_slice_me(self):
        assert (
            self.axis._slice_me(slice(np.float32(10.2), 10.4, 2)) ==
            slice(2, 4, 2))

    def test_update_from(self):
        ax2 = LinearDataAxis(size=2, units="nm", scale=0.5)
        self.axis.update_from(ax2, attributes=("units", "scale"))
        assert ((ax2.units, ax2.scale) ==
                (self.axis.units, self.axis.scale))

    def test_value_changed_event(self):
        ax = self.axis
        m = mock.Mock()
        ax.events.value_changed.connect(m.trigger_me)
        ax.value = ax.value
        assert not m.trigger_me.called
        ax.value = ax.value + ax.scale * 0.3
        assert not m.trigger_me.called
        ax.value = ax.value + ax.scale
        assert m.trigger_me.called

    def test_index_changed_event(self):
        ax = self.axis
        m = mock.Mock()
        ax.events.index_changed.connect(m.trigger_me)
        ax.index = ax.index
        assert not m.trigger_me.called
        ax.index += 1
        assert m.trigger_me.called
