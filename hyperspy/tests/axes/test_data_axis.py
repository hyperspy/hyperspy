# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

import copy
import math
import platform
from unittest import mock

import numpy as np
from numpy.testing import assert_allclose
import traits.api as t
import pytest

from hyperspy.axes import (BaseDataAxis, DataAxis, FunctionalDataAxis,
                           UniformDataAxis, create_axis)
from hyperspy.signals import Signal1D
from hyperspy.misc.test_utils import assert_deep_almost_equal


class TestBaseDataAxis:

    def setup_method(self, method):
        self.axis = BaseDataAxis()

    def test_initialisation_BaseDataAxis_default(self):
        with pytest.raises(AttributeError):
            assert self.axis.index_in_array is None
        assert self.axis.name is t.Undefined
        assert self.axis.units is t.Undefined
        assert not self.axis.navigate
        assert not self.axis.is_binned
        assert not self.axis.is_uniform

    def test_initialisation_BaseDataAxis(self):
        axis = BaseDataAxis(name='named axis', units='s', navigate=True)
        assert axis.name == 'named axis'
        assert axis.units == 's'
        assert axis.navigate
        assert not self.axis.is_uniform
        assert_deep_almost_equal(axis.get_axis_dictionary(),
                                 {'_type': 'BaseDataAxis',
                                  'name': 'named axis',
                                  'units': 's',
                                  'navigate': True,
                                  'is_binned': False})

    def test_error_BaseDataAxis(self):
        with pytest.raises(NotImplementedError):
            self.axis._slice_me(1)
        with pytest.raises(ValueError):
            self.axis._parse_value_from_string('')
        with pytest.raises(ValueError):
            self.axis._parse_value_from_string('spam')

    #Note: The following methods from BaseDataAxis rely on the self.axis.axis
    #numpy array to be initialized, and are tested in the subclasses:
    #BaseDataAxis.value2index --> tested in FunctionalDataAxis
    #BaseDataAxis.index2value --> NOT EXPLICITLY TESTED
    #BaseDataAxis.value_range_to_indices --> tested in UniformDataAxis
    #BaseDataAxis.update_from --> tested in DataAxis and FunctionalDataAxis

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
        assert not self.axis.is_uniform

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
        ax.value = ax.value + (ax.axis[1] - ax.axis[0]) * 0.4
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

    def test_convert_to_uniform_axis(self):
        scale = (self.axis.high_value - self.axis.low_value) / self.axis.size
        is_binned = self.axis.is_binned
        navigate = self.axis.navigate
        self.axis.name = "parrot"
        self.axis.units = "plumage"
        s = Signal1D(np.arange(10), axes=[self.axis])
        index_in_array = s.axes_manager[0].index_in_array
        s.axes_manager[0].convert_to_uniform_axis()
        assert isinstance(s.axes_manager[0], UniformDataAxis)
        assert s.axes_manager[0].name == "parrot"
        assert s.axes_manager[0].units == "plumage"
        assert s.axes_manager[0].size == 16
        assert s.axes_manager[0].scale == scale
        assert s.axes_manager[0].offset == 0
        assert s.axes_manager[0].low_value == 0
        assert s.axes_manager[0].high_value == 15 * scale
        assert index_in_array == s.axes_manager[0].index_in_array
        assert is_binned == s.axes_manager[0].is_binned
        assert navigate == s.axes_manager[0].navigate

    def test_value2index(self):
        assert self.axis.value2index(10.15) == 3
        assert self.axis.value2index(60) == 8
        assert self.axis.value2index(2.5, rounding=round) == 1
        assert self.axis.value2index(2.5, rounding=math.ceil) == 2
        assert self.axis.value2index(2.5, rounding=math.floor) == 1
        # Test that output is integer
        assert isinstance(self.axis.value2index(60), (int, np.integer))
        self.axis.axis = self.axis.axis - 2
        # test rounding on negative value
        assert self.axis.value2index(-1.5, rounding=round) == 1


    def test_value2index_error(self):
        with pytest.raises(ValueError):
            self.axis.value2index(226)

    def test_parse_value_from_relative_string(self):
        ax = self.axis
        assert ax._parse_value_from_string('rel0.0') == 0.0
        assert ax._parse_value_from_string('rel0.5') == 112.5
        assert ax._parse_value_from_string('rel1.0') == 225.0
        with pytest.raises(ValueError):
            ax._parse_value_from_string('rela0.5')
        with pytest.raises(ValueError):
            ax._parse_value_from_string('rel1.5')
        with pytest.raises(ValueError):
            ax._parse_value_from_string('abcd')

    def test_parse_value_from_string_with_units(self):
        ax = self.axis
        ax.units = 'nm'
        with pytest.raises(ValueError):
            ax._parse_value_from_string('0.02 um')

    @pytest.mark.parametrize("use_indices", (False, True))
    def test_crop(self, use_indices):
        axis = DataAxis(axis=self._axis)
        start, end = 4., 196.
        if use_indices:
            start = axis.value2index(start)
            end = axis.value2index(end)
        axis.crop(start, end)
        assert axis.size == 12
        np.testing.assert_almost_equal(axis.axis[0], 4)
        np.testing.assert_almost_equal(axis.axis[-1], 169)

    def test_crop_reverses_indexing(self):
        # reverse indexing
        axis = DataAxis(axis=self._axis)
        axis.crop(-14, -2)
        assert axis.size == 12
        np.testing.assert_almost_equal(axis.axis[0], 4)
        np.testing.assert_almost_equal(axis.axis[-1], 169)

        # mixed reverses indexing
        axis = DataAxis(axis=self._axis)
        axis.crop(2, -2)
        assert axis.size == 12
        np.testing.assert_almost_equal(axis.axis[0], 4)
        np.testing.assert_almost_equal(axis.axis[-1], 169)

    def test_error_DataAxis(self):
        with pytest.raises(ValueError):
            _ = DataAxis(axis=np.arange(16)**2, _type='UniformDataAxis')
        with pytest.raises(AttributeError):
            self.axis.index_in_axes_manager()
        with pytest.raises(IndexError):
            self.axis._get_positive_index(-17)
        with pytest.raises(ValueError):
            self.axis._get_array_slices(slice_=slice(1,2,1.5))
        with pytest.raises(IndexError):
            self.axis._get_array_slices(slice_=slice(225,-1.1,1))
        with pytest.raises(IndexError):
            self.axis._get_array_slices(slice_=slice(225.1,0,1))

    def test_update_from(self):
        ax2 = DataAxis(units="plumage", name="parrot", axis=np.arange(16))
        self.axis.update_from(ax2, attributes=("units", "name"))
        assert ((ax2.units, ax2.name) ==
                (self.axis.units, self.axis.name))

    def test_calibrate(self):
        with pytest.raises(TypeError, match="only for uniform axes"):
            self.axis.calibrate(value_tuple=(11,12), index_tuple=(0,5))


class TestFunctionalDataAxis:

    def setup_method(self, method):
        expression = "x ** power"
        self.axis = FunctionalDataAxis(
            size=10,
            expression=expression,
            power=2,)

    def test_initialisation_parameters(self):
        axis = self.axis
        assert axis.power == 2
        np.testing.assert_allclose(
            axis.axis,
            np.arange(10)**2)

    def test_create_axis(self):
        axis = create_axis(**self.axis.get_axis_dictionary())
        assert isinstance(axis, FunctionalDataAxis)

    def test_initialisation_errors(self):
        expression = "x ** power"
        with pytest.raises(ValueError, match="Please provide"):
            self.axis = FunctionalDataAxis(
                expression=expression,
                power=2,)
        with pytest.raises(ValueError, match="The values of"):
            self.axis = FunctionalDataAxis(
                size=10,
                expression=expression,)

    @pytest.mark.parametrize("use_indices", (True, False))
    def test_crop(self, use_indices):
        axis = self.axis
        start, end = 3.9, 72.6
        if use_indices:
            start = 2
            end = -1
        axis.crop(start, end)
        assert axis.size == 7
        np.testing.assert_almost_equal(axis.axis[0], 4.)
        np.testing.assert_almost_equal(axis.axis[-1], 64.)

    def test_convert_to_non_uniform_axis(self):
        axis = np.copy(self.axis.axis)
        is_binned = self.axis.is_binned
        navigate = self.axis.navigate
        self.axis.name = "parrot"
        self.axis.units = "plumage"
        s = Signal1D(np.arange(10), axes=[self.axis])
        index_in_array = s.axes_manager[0].index_in_array
        s.axes_manager[0].convert_to_non_uniform_axis()
        assert isinstance(s.axes_manager[0], DataAxis)
        assert s.axes_manager[0].name == "parrot"
        assert s.axes_manager[0].units == "plumage"
        assert s.axes_manager[0].size == 10
        assert s.axes_manager[0].low_value == 0
        assert s.axes_manager[0].high_value == 81
        np.testing.assert_allclose(s.axes_manager[0].axis, axis)
        with pytest.raises(AttributeError):
            s.axes_manager[0]._expression
        with pytest.raises(AttributeError):
            s.axes_manager[0]._function
        with pytest.raises(AttributeError):
            s.axes_manager[0].x
        assert index_in_array == s.axes_manager[0].index_in_array
        assert is_binned == s.axes_manager[0].is_binned
        assert navigate == s.axes_manager[0].navigate

    def test_update_from(self):
        ax2 = FunctionalDataAxis(size=2, units="nm", expression="x ** power", power=3)
        self.axis.update_from(ax2, attributes=("units", "power"))
        assert ((ax2.units, ax2.power) ==
                (self.axis.units, self.axis.power))

    def test_slice_me(self):
        assert self.axis._slice_me(slice(1, 5)) == slice(1, 5)
        assert self.axis.size == 4
        np.testing.assert_allclose(self.axis.axis, np.arange(1, 5)**2)

    def test_calibrate(self):
        with pytest.raises(TypeError, match="only for uniform axes"):
            self.axis.calibrate(value_tuple=(11,12), index_tuple=(0,5))

    def test_functional_value2index(self):
        #Tests for value2index
        #Works as intended
        assert self.axis.value2index(44.7) == 7
        assert self.axis.value2index(2.5, rounding=round) == 1
        assert self.axis.value2index(2.5, rounding=math.ceil) == 2
        assert self.axis.value2index(2.5, rounding=math.floor) == 1
        # Returns integer
        assert isinstance(self.axis.value2index(45), (int, np.integer))
        #Input None --> output None
        assert self.axis.value2index(None) == None
        #NaN in --> error out
        with pytest.raises(ValueError):
            self.axis.value2index(np.nan)
        #Values in out of bounds --> error out (both sides of axis)
        with pytest.raises(ValueError):
            self.axis.value2index(-2)
        with pytest.raises(ValueError):
            self.axis.value2index(111)
        #str in --> error out
        with pytest.raises(TypeError):
            self.axis.value2index("69")
        #Empty str in --> error out
        with pytest.raises(TypeError):
            self.axis.value2index("")

        #Tests with array Input
        #Array in --> array out
        arval = np.array([[0,4],[16.,36.]])
        assert np.all(self.axis.value2index(arval) == np.array([[0,2],[4,6]]))
        #One value out of bound in array in --> error out (both sides)
        arval[1,1] = 111
        with pytest.raises(ValueError):
            self.axis.value2index(arval)
        arval[1,1] = -0.3
        with pytest.raises(ValueError):
            self.axis.value2index(arval)
        #One NaN in array in --> error out
        arval[1,1] = np.nan
        with pytest.raises(ValueError):
            self.axis.value2index(arval)
        #Single-value-array-in --> scalar out
        arval = np.array([1.0])
        assert np.isscalar(self.axis.value2index(arval))


class TestReciprocalDataAxis:

    def setup_method(self, method):
        expression = "a / (x + 1) + b"
        self.axis = FunctionalDataAxis(size=10, expression=expression,
                                       a=0.1, b=10)

    def _test_initialisation_parameters(self, axis):
        assert axis.a == 0.1
        assert axis.b == 10
        def func(x): return 0.1 / (x + 1) + 10
        np.testing.assert_allclose(axis.axis, func(np.arange(10)))

    def test_initialisation_parameters(self):
        self._test_initialisation_parameters(self.axis)

    def test_create_axis(self):
        axis = create_axis(**self.axis.get_axis_dictionary())
        assert isinstance(axis, FunctionalDataAxis)
        self._test_initialisation_parameters(axis)


    @pytest.mark.parametrize("use_indices", (True, False))
    def test_crop(self, use_indices):
        axis = self.axis
        start, end = 10.05, 10.02
        if use_indices:
            start = axis.value2index(start)
            end = axis.value2index(end)
        axis.crop(start, end)
        assert axis.size == 3
        np.testing.assert_almost_equal(axis.axis[0], 10.05)
        np.testing.assert_almost_equal(axis.axis[-1], 10.025)


class TestUniformDataAxis:

    def setup_method(self, method):
        self.axis = UniformDataAxis(size=10, scale=0.1, offset=10)

    def _test_initialisation_parameters(self, axis):
        assert axis.scale == 0.1
        assert axis.offset == 10
        def func(x): return axis.scale * x + axis.offset
        np.testing.assert_allclose(axis.axis, func(np.arange(10)))

    def test_initialisation_parameters(self):
        self._test_initialisation_parameters(self.axis)

    def test_create_axis(self):
        axis = create_axis(**self.axis.get_axis_dictionary())
        assert isinstance(axis, UniformDataAxis)
        self._test_initialisation_parameters(axis)

    def test_value_range_to_indices_in_range(self):
        assert self.axis.is_uniform
        assert (
            self.axis.value_range_to_indices(
                10.1, 10.8) == (1, 8))

    def test_value_range_to_indices_endpoints(self):
        assert self.axis.value_range_to_indices(10, 10.9) == (0, 9)

    def test_value_range_to_indices_out(self):
        assert self.axis.value_range_to_indices(9, 11) == (0, 9)

    def test_value_range_to_indices_None(self):
        assert self.axis.value_range_to_indices(None, None) == (0, 9)

    def test_value_range_to_indices_v1_greater_than_v2(self):
        with pytest.raises(ValueError):
            self.axis.value_range_to_indices(2, 1)

    def test_deepcopy(self):
        ac = copy.deepcopy(self.axis)
        ac.offset = 100
        assert self.axis.offset != ac.offset
        assert self.axis.navigate == ac.navigate
        assert self.axis.is_binned == ac.is_binned

    def test_deepcopy_on_trait_change(self):
        ac = copy.deepcopy(self.axis)
        ac.offset = 100
        assert ac.axis[0] == ac.offset

    def test_uniform_value2index(self):
        #Tests for value2index
        #Works as intended
        assert self.axis.value2index(10.15) == 1
        assert self.axis.value2index(10.17, rounding=math.floor) == 1
        assert self.axis.value2index(10.13, rounding=math.ceil) == 2
        # Test that output is integer
        assert isinstance(self.axis.value2index(10.15), (int, np.integer))
        #Endpoint left
        assert self.axis.value2index(10.) == 0
        #Endpoint right
        assert self.axis.value2index(10.9) == 9
        #Input None --> output None
        assert self.axis.value2index(None) == None
        #NaN in --> error out
        with pytest.raises(ValueError):
            self.axis.value2index(np.nan)
        #Values in out of bounds --> error out (both sides of axis)
        with pytest.raises(ValueError):
            self.axis.value2index(-2)
        with pytest.raises(ValueError):
            self.axis.value2index(111)
        #str without unit in --> error out
        with pytest.raises(ValueError):
            self.axis.value2index("69")
        #Empty str in --> error out
        with pytest.raises(ValueError):
            self.axis.value2index("")
        #Value with unit when axis is unitless --> Error out
        with pytest.raises(ValueError):
            self.axis.value2index("0.0101um")

        #Tests with array Input
        #Arrays work as intended
        arval = np.array([[10.15, 10.15], [10.24, 10.28]])
        assert np.all(self.axis.value2index(arval) \
                        == np.array([[1, 1], [2, 3]]))
        assert np.all(self.axis.value2index(arval, rounding=math.floor) \
                        == np.array([[1, 1], [2, 2]]))
        assert np.all(self.axis.value2index(arval, rounding=math.ceil)\
                        == np.array([[2, 2], [3, 3]]))
        #List in --> array out
        assert np.all(self.axis.value2index(arval.tolist()) \
                                            == np.array([[1, 1], [2, 3]]))
        #One value out of bound in array in --> error out (both sides)
        arval[1, 1] = 111
        with pytest.raises(ValueError):
            self.axis.value2index(arval)
        arval[1, 1] = -0.3
        with pytest.raises(ValueError):
            self.axis.value2index(arval)
        #One NaN in array in --> error out
        if platform.machine() != 'aarch64':
            # Skip aarch64 platform because it doesn't raise error
            arval[1, 1] = np.nan
            with pytest.raises(ValueError):
                self.axis.value2index(arval)

        #Copy of axis with units
        axis = copy.deepcopy(self.axis)
        axis.units = 'nm'

        #Value with unit in --> OK out
        assert axis.value2index("0.0101um") == 1
        #Value with relative units in --> OK out
        assert self.axis.value2index("rel0.5") == 4

        #Works with arrays of values with units in
        np.testing.assert_allclose(
            axis.value2index(['0.01um', '0.0101um', '0.0103um']),
            np.array([0, 1, 3])
            )
        #Raises errors if a weird unit is passed in
        with pytest.raises(BaseException):
            axis.value2index(["0.01uma", '0.0101uma', '0.0103uma'])
        #Values
        np.testing.assert_allclose(
            self.axis.value2index(["rel0.0", "rel0.5", "rel1.0"]),
            np.array([0, 4, 9])
            )

    def test_slice_me(self):
        assert (
            self.axis._slice_me(slice(np.float32(10.2), 10.4, 2)) ==
            slice(2, 4, 2)
            )

    def test_update_from(self):
        ax2 = UniformDataAxis(size=2, units="nm", scale=0.5)
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

    def test_convert_to_non_uniform_axis(self):
        axis = np.copy(self.axis.axis)
        is_binned = self.axis.is_binned
        navigate = self.axis.navigate
        self.axis.name = "parrot"
        self.axis.units = "plumage"
        s = Signal1D(np.arange(10), axes=[self.axis])
        index_in_array = s.axes_manager[0].index_in_array
        s.axes_manager[0].convert_to_non_uniform_axis()
        assert isinstance(s.axes_manager[0], DataAxis)
        assert s.axes_manager[0].name == "parrot"
        assert s.axes_manager[0].units == "plumage"
        assert s.axes_manager[0].size == 10
        assert s.axes_manager[0].low_value == 10
        assert s.axes_manager[0].high_value == 10 + 0.1 * 9
        np.testing.assert_allclose(s.axes_manager[0].axis, axis)
        with pytest.raises(AttributeError):
            s.axes_manager[0].offset
        with pytest.raises(AttributeError):
            s.axes_manager[0].scale
        assert index_in_array == s.axes_manager[0].index_in_array
        assert is_binned == s.axes_manager[0].is_binned
        assert navigate == s.axes_manager[0].navigate

    def test_convert_to_functional_data_axis(self):
        axis = np.copy(self.axis.axis)
        is_binned = self.axis.is_binned
        navigate = self.axis.navigate
        self.axis.name = "parrot"
        self.axis.units = "plumage"
        s = Signal1D(np.arange(10), axes=[self.axis])
        index_in_array = s.axes_manager[0].index_in_array
        s.axes_manager[0].convert_to_functional_data_axis(expression = 'x**2')
        assert isinstance(s.axes_manager[0], FunctionalDataAxis)
        assert s.axes_manager[0].name == "parrot"
        assert s.axes_manager[0].units == "plumage"
        assert s.axes_manager[0].size == 10
        assert s.axes_manager[0].low_value == 10**2
        assert s.axes_manager[0].high_value == (10 + 0.1 * 9)**2
        assert s.axes_manager[0]._expression == 'x**2'
        assert isinstance(s.axes_manager[0].x, UniformDataAxis)
        np.testing.assert_allclose(s.axes_manager[0].axis, axis**2)
        with pytest.raises(AttributeError):
            s.axes_manager[0].offset
        with pytest.raises(AttributeError):
            s.axes_manager[0].scale
        assert index_in_array == s.axes_manager[0].index_in_array
        assert is_binned == s.axes_manager[0].is_binned
        assert navigate == s.axes_manager[0].navigate

    @pytest.mark.parametrize("use_indices", (False, True))
    def test_crop(self, use_indices):
        axis = UniformDataAxis(size=10, scale=0.1, offset=10)
        start = 10.2
        if use_indices:
            start = axis.value2index(start)
        axis.crop(start)
        assert axis.size == 8
        np.testing.assert_almost_equal(axis.axis[0], 10.2)
        np.testing.assert_almost_equal(axis.axis[-1], 10.9)
        np.testing.assert_almost_equal(axis.offset, 10.2)
        np.testing.assert_almost_equal(axis.scale, 0.1)

        axis = UniformDataAxis(size=10, scale=0.1, offset=10)
        end = 10.4
        if use_indices:
            end = axis.value2index(end)
        axis.crop(start, end)
        assert axis.size == 2
        np.testing.assert_almost_equal(axis.axis[0], 10.2)
        np.testing.assert_almost_equal(axis.axis[-1], 10.3)
        np.testing.assert_almost_equal(axis.offset, 10.2)
        np.testing.assert_almost_equal(axis.scale, 0.1)

        axis = UniformDataAxis(size=10, scale=0.1, offset=10)
        axis.crop(None, end)
        assert axis.size == 4
        np.testing.assert_almost_equal(axis.axis[0], 10.0)
        np.testing.assert_almost_equal(axis.axis[-1], 10.3)
        np.testing.assert_almost_equal(axis.offset, 10.0)
        np.testing.assert_almost_equal(axis.scale, 0.1)

    @pytest.mark.parametrize("mixed", (False, True))
    def test_crop_reverses_indexing(self, mixed):
        axis = UniformDataAxis(size=10, scale=0.1, offset=10)
        if mixed:
            i1, i2 = 2, -6
        else:
            i1, i2 = -8, -6
        axis.crop(i1, i2)
        assert axis.size == 2
        np.testing.assert_almost_equal(axis.axis[0], 10.2)
        np.testing.assert_almost_equal(axis.axis[-1], 10.3)
        np.testing.assert_almost_equal(axis.offset, 10.2)
        np.testing.assert_almost_equal(axis.scale, 0.1)

    def test_parse_value(self):
        ax = copy.deepcopy(self.axis)
        ax.units = 'nm'
        # slicing by index
        assert ax._parse_value(5) == 5
        assert type(ax._parse_value(5)) is int
        # slicing by calibrated value
        assert ax._parse_value(10.5) == 10.5
        assert type(ax._parse_value(10.5)) is float
        # slicing by unit
        assert ax._parse_value('10.5nm') == 10.5
        np.testing.assert_almost_equal(ax._parse_value('10500pm'), 10.5)

    def test_parse_value_from_relative_string(self):
        ax = self.axis
        assert ax._parse_value_from_string('rel0.0') == 10.0
        assert ax._parse_value_from_string('rel0.5') == 10.45
        assert ax._parse_value_from_string('rel1.0') == 10.9

    def test_slice_empty_string(self):
        ax = self.axis
        with pytest.raises(ValueError):
            ax._parse_value("")

    def test_calibrate(self):
        offset, scale = self.axis.calibrate(value_tuple=(11,12), \
                                index_tuple=(0,5), modify_calibration=False)
        assert scale == 0.2
        assert offset == 11
        self.axis.calibrate(value_tuple=(11,12), index_tuple=(0,5))
        assert self.axis.scale == 0.2
        assert self.axis.offset == 11


class TestUniformDataAxisValueRangeToIndicesNegativeScale:

    def setup_method(self, method):
        self.axis = UniformDataAxis(size=10, scale=-0.1, offset=10)

    def test_value_range_to_indices_in_range(self):
        assert self.axis.value_range_to_indices(9.9, 9.2) == (1, 8)

    def test_value_range_to_indices_endpoints(self):
        assert self.axis.value_range_to_indices(10, 9.1) == (0, 9)

    def test_value_range_to_indices_out(self):
        assert self.axis.value_range_to_indices(11, 9) == (0, 9)

    def test_value_range_to_indices_None(self):
        assert self.axis.value_range_to_indices(None, None) == (0, 9)

    def test_value_range_to_indices_v1_greater_than_v2(self):
        with pytest.raises(ValueError):
            self.axis.value_range_to_indices(1, 2)


def test_rounding_consistency_axis_type():
    scales = [0.1, -0.1, 0.1, -0.1]
    offsets = [-11.0, -10.9, 10.9, 11.0]
    values = [-10.95, -10.95, 10.95, 10.95]

    for i, (scale, offset, value) in enumerate(zip(scales, offsets, values)):
        ax = UniformDataAxis(scale=scale, offset=offset, size=3)
        ua_idx = ax.value2index(value)
        nua_idx = super(type(ax), ax).value2index(value)
        print('scale', scale)
        print('offset', offset)
        print('Axis values:', ax.axis)
        print(f"value: {value} --> uniform: {ua_idx}, non-uniform: {nua_idx}")
        print("\n")
        assert nua_idx == ua_idx


@pytest.mark.parametrize('shift', (0.05, 0.025))
def test_rounding_consistency_axis_type_half(shift):

    axis = UniformDataAxis(size=20, scale=0.1, offset=-1.0);
    test_vals = axis.axis[:-1] + shift

    uaxis_indices = axis.value2index(test_vals)
    nuaxis_indices = super(type(axis), axis).value2index(test_vals)

    np.testing.assert_allclose(uaxis_indices, nuaxis_indices)
