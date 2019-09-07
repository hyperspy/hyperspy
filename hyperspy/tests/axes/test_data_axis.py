import copy
import math
from unittest import mock

import numpy as np
from numpy.testing import assert_allclose
import traits.api as t
import pytest

from hyperspy.axes import (BaseDataAxis, DataAxis, FunctionalDataAxis,
                           LinearDataAxis, create_axis)
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
        self._axis = np.arange(16)**2
        self.axis = DataAxis(axis=self._axis)

    def _test_initialisation_parameters(self, axis):
        np.testing.assert_allclose(axis.axis, self._axis)

    def test_initialisation_parameters(self):
        self._test_initialisation_parameters(self.axis)

    def test_create_axis(self):
        axis = create_axis(**self.axis.get_axis_dictionary())
        assert isinstance(axis, DataAxis)
        self._test_initialisation_parameters(axis)

    def test_axis_value(self):
        assert_allclose(self.axis.axis, np.arange(16)**2)
        assert self.axis.size == 16
        assert not self.axis.is_linear

    def test_update_axes(self):
        values = np.arange(20)**2
        self.axis.axis = values.tolist()
        self.axis.update_axis()
        assert self.axis.size == 20
        assert_allclose(self.axis.axis, values)

    def test_update_axes2(self):
        values = np.array([3, 4, 10, 40])
        self.axis.axis = values
        self.axis.update_axis()
        assert_allclose(self.axis.axis, values)

    def test_update_axis_from_list(self):
        values = np.arange(16)**2
        self.axis.axis = values.tolist()
        self.axis.update_axis()
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

    def test_convert_to_linear_axis(self):
        scale = (self.axis.high_value - self.axis.low_value) / self.axis.size
        self.axis.convert_to_linear_axis()
        assert isinstance(self.axis, LinearDataAxis)
        assert self.axis.size == 16
        assert self.axis.scale == scale
        assert self.axis.offset == 0
        assert self.axis.low_value == 0
        assert self.axis.high_value == 15 * scale

    def test_value2index(self):
        assert self.axis.value2index(10.15) == 3
        assert self.axis.value2index(60) == 8


class TestFunctionalDataAxis:

    def setup_method(self, method):
        expression = "a * x + b"
        self.axis = FunctionalDataAxis(size=10, expression=expression,
                                       a=0.1, b=10)

    def _test_initialisation_parameters(self, axis):
        assert axis.a == 0.1
        assert axis.b == 10
        assert hasattr(axis, 'function')
        np.testing.assert_allclose(axis.axis, np.linspace(10, 10.9, 10))

    def test_initialisation_parameters(self):
        self._test_initialisation_parameters(self.axis)

    def test_create_axis(self):
        axis = create_axis(**self.axis.get_axis_dictionary())
        assert isinstance(axis, FunctionalDataAxis)
        self._test_initialisation_parameters(axis)


class TestReciprocalDataAxis:

    def setup_method(self, method):
        expression = "a / (x + 1) + b"
        self.axis = FunctionalDataAxis(size=10, expression=expression,
                                       a=0.1, b=10)

    def _test_initialisation_parameters(self, axis):
        assert axis.a == 0.1
        assert axis.b == 10
        assert hasattr(axis, 'function')
        assert hasattr(axis, '_expression')
        def func(x): return 0.1 / (x + 1) + 10
        np.testing.assert_allclose(axis.axis, func(np.arange(10)))

    def test_initialisation_parameters(self):
        self._test_initialisation_parameters(self.axis)

    def test_create_axis(self):
        axis = create_axis(**self.axis.get_axis_dictionary())
        assert isinstance(axis, FunctionalDataAxis)
        self._test_initialisation_parameters(axis)        
        

class TestLinearDataAxis:

    def setup_method(self, method):
        self.axis = LinearDataAxis(size=10, scale=0.1, offset=10)

    def _test_initialisation_parameters(self, axis):
        assert axis.scale == 0.1
        assert axis.offset == 10
        assert hasattr(axis, 'function')
        assert hasattr(axis, '_expression')
        def func(x): return axis.scale* x + axis.offset
        np.testing.assert_allclose(axis.axis, func(np.arange(10)))

    def test_initialisation_parameters(self):
        self._test_initialisation_parameters(self.axis)

    def test_create_axis(self):
        axis = create_axis(**self.axis.get_axis_dictionary())
        assert isinstance(axis, LinearDataAxis)
        self._test_initialisation_parameters(axis)  

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

    def test_convert_to_non_linear_axis(self):
        axis = np.copy(self.axis.axis)
        self.axis.convert_to_non_linear_axis()
        assert isinstance(self.axis, DataAxis)
        assert self.axis.size == 10
        assert self.axis.low_value == 10
        assert self.axis.high_value == 10 + 0.1 * 9
        np.testing.assert_allclose(self.axis.axis, axis)

    @pytest.mark.parametrize("use_indices", (False, True))
    def test_crop(self, use_indices):
        axis = LinearDataAxis(size=10, scale=0.1, offset=10)
        start = 10.2
        if use_indices:
            start = axis.value2index(start)
        axis.crop(start)
        assert axis.size == 8
        np.testing.assert_almost_equal(axis.axis[0], 10.2)
        np.testing.assert_almost_equal(axis.axis[-1], 10.9)
        np.testing.assert_almost_equal(axis.offset, 10.2)
        np.testing.assert_almost_equal(axis.scale, 0.1)

        axis = LinearDataAxis(size=10, scale=0.1, offset=10)
        end = 10.4
        if use_indices:
            end = axis.value2index(end)
        axis.crop(start, end)
        assert axis.size == 2
        np.testing.assert_almost_equal(axis.axis[0], 10.2)
        np.testing.assert_almost_equal(axis.axis[-1], 10.3)
        np.testing.assert_almost_equal(axis.offset, 10.2)
        np.testing.assert_almost_equal(axis.scale, 0.1)

        axis = LinearDataAxis(size=10, scale=0.1, offset=10)
        axis.crop(None, end)
        assert axis.size == 4
        np.testing.assert_almost_equal(axis.axis[0], 10.0)
        np.testing.assert_almost_equal(axis.axis[-1], 10.3)
        np.testing.assert_almost_equal(axis.offset, 10.0)
        np.testing.assert_almost_equal(axis.scale, 0.1)

    def test_crop_values(self):
        axis = LinearDataAxis(size=10, scale=0.1, offset=10)
        axis.crop(10.2)
        assert axis.size == 8
        np.testing.assert_almost_equal(axis.axis[0], 10.2)
        np.testing.assert_almost_equal(axis.axis[-1], 10.9)
        np.testing.assert_almost_equal(axis.offset, 10.2)
        np.testing.assert_almost_equal(axis.scale, 0.1)
