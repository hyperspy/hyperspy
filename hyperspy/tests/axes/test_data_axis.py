# -*- coding: utf-8 -*-
# Copyright 2007-2020 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.

import copy
import math
from unittest import mock

import numpy as np
import pytest

from hyperspy.axes import DataAxis


class TestDataAxis:

    def setup_method(self, method):
        self.axis = DataAxis(size=10, scale=0.1, offset=10, units='nm')

    def test_value_range_to_indices_in_range(self):
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

    def test_value2index_None(self):
        assert self.axis.value2index(None) is None

    def test_value2index_fail_string_in(self):
        with pytest.raises(ValueError):
            self.axis.value2index("10.15")

    def test_value2index_fail_empty_string_in(self):
        with pytest.raises(ValueError):
            self.axis.value2index("")

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

    def test_value2index_list_in(self):
        assert (
            self.axis.value2index([10.15, 10.15]).tolist() ==
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

    def test_calibrated_value2index_list_in(self):
        np.testing.assert_allclose(
            self.axis.value2index(['0.01um', '0.0101um', '0.0103um']),
            np.array([0, 1, 3])
            )
        with pytest.raises(BaseException):
            self.axis.value2index(["0.01uma", '0.0101uma', '0.0103uma'])

    def test_calibrated_value2index_in(self):
        assert self.axis.value2index("0.0101um") == 1

    def test_relative_value2index_in(self):
        assert self.axis.value2index("rel0.5") == 4

    def test_relative_value2index_list_in(self):
        np.testing.assert_allclose(
            self.axis.value2index(["rel0.0", "rel0.5", "rel1.0"]),
            np.array([0, 4, 9])
            )

    def test_value2index_array_out(self):
        with pytest.raises(ValueError):
            self.axis.value2index(np.array([10, 11]))

    def test_slice_me(self):
        assert (
            self.axis._slice_me(slice(np.float32(10.2), 10.4, 2)) ==
            slice(2, 4, 2)
            )

    def test_update_from(self):
        ax2 = DataAxis(size=2, units="nm", scale=0.5)
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

    def test_value_changed_event_continuous(self):
        ax = self.axis
        ax.continuous_value = True
        m = mock.Mock()
        ax.events.value_changed.connect(m.trigger_me_value)
        ax.events.index_changed.connect(m.trigger_me_index)
        ax.value = ax.value
        assert not m.trigger_me_value.called
        ax.value = ax.value + ax.scale * 0.3
        assert m.trigger_me_value.called
        assert not m.trigger_me_index.called
        ax.value = ax.value + ax.scale
        assert m.trigger_me_index.called

    def test_index_changed_event(self):
        ax = self.axis
        m = mock.Mock()
        ax.events.index_changed.connect(m.trigger_me)
        ax.index = ax.index
        assert not m.trigger_me.called
        ax.index += 1
        assert m.trigger_me.called

    def test_parse_value(self):
        ax = self.axis
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
        with pytest.raises(ValueError):
            ax._parse_value_from_string('rela0.5')
        with pytest.raises(ValueError):
            ax._parse_value_from_string('rela1.5')
        with pytest.raises(ValueError):
            ax._parse_value_from_string('abcd')

    def test_slice_empty_string(self):
        ax = self.axis
        with pytest.raises(ValueError):
            ax._parse_value("")