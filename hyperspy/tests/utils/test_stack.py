# -*- coding: utf-8 -*-
# Copyright 2007-2021 The HyperSpy developers
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

import numpy as np
import pytest

from hyperspy import utils
from hyperspy.signal import BaseSignal
from hyperspy.exceptions import VisibleDeprecationWarning
from hyperspy.misc.utils import DictionaryTreeBrowser



def test_stack_warning():
    with pytest.warns(VisibleDeprecationWarning, match="deprecated"):
        _ = utils.stack([BaseSignal([1]), BaseSignal([2])], mmap=True)


class TestUtilsStack:

    def setup_method(self, method):
        s = BaseSignal(np.random.random((3, 2, 5)))
        s.axes_manager.set_signal_dimension(1)
        s.axes_manager[0].name = "x"
        s.axes_manager[1].name = "y"
        s.axes_manager[2].name = "E"
        s.axes_manager[2].scale = 0.5
        s.metadata.General.title = 'test'
        s.original_metadata = DictionaryTreeBrowser({'om': 'some metadata'})
        self.signal = s

    @pytest.mark.parametrize('stack_metadata', [True, False, 0, 1])
    def test_stack_stack_metadata(self, stack_metadata):
        s = self.signal
        s1 = s.deepcopy() + 1
        s2 = s.deepcopy() * 4
        test_axis = s.axes_manager[0].index_in_array
        result_signal = utils.stack([s, s1, s2], stack_metadata=stack_metadata)
        result_list = result_signal.split()
        assert test_axis == s.axes_manager[0].index_in_array
        assert len(result_list) == 3
        np.testing.assert_array_almost_equal(
            result_list[0].data, result_signal.inav[:, :, 0].data)
        if stack_metadata is True:
            om = result_signal.original_metadata.stack_elements.element0.original_metadata
        elif stack_metadata in [0, 1]:
            om = result_signal.original_metadata
        if stack_metadata is False:
            assert om.as_dictionary() == {}
        else:
            assert om.as_dictionary() == s.original_metadata.as_dictionary()

    def test_stack_stack_metadata_value(self):
        s = BaseSignal(1)
        s.metadata.General.title = 'title 1'
        s.original_metadata.set_item('a', 1)

        s2 = BaseSignal(2)
        s2.metadata.General.title = 'title 2'
        s2.original_metadata.set_item('a', 2)

        stack_out = utils.stack([s, s2], stack_metadata=True)
        elem0 = stack_out.original_metadata.stack_elements.element0
        elem1 = stack_out.original_metadata.stack_elements.element1

        for el, _s in zip([elem0, elem1], [s, s2]):
            assert el.original_metadata.as_dictionary() == \
                _s.original_metadata.as_dictionary()
            assert el.metadata.as_dictionary() == _s.metadata.as_dictionary()

    def test_stack_stack_metadata_error(self):
        s = self.signal
        s2 = s.deepcopy()
        with pytest.raises(ValueError):
            utils.stack([s, s2], stack_metadata='not supported argument')

    def test_stack_stack_metadata_index(self):
        s = self.signal
        s1 = s.deepcopy() + 1
        s1.metadata.General.title = 'first signal'
        s1.original_metadata.om_title = 'first signal om'
        s2 = s.deepcopy() * 4
        s2.metadata.General.title = 'second_signal'
        s2.original_metadata.om_title = 'second signal om'

        res = utils.stack([s1, s2, s], stack_metadata=0)
        assert res.metadata.General.title == s1.metadata.General.title

        res2 = utils.stack([s1, s2, s], stack_metadata=2)
        assert res2.metadata.General.title == s.metadata.General.title

    def test_stack_number_of_parts(self):
        s = self.signal
        s1 = s.deepcopy() + 1
        s2 = s.deepcopy() * 4
        test_axis = s.axes_manager[0].index_in_array
        result_signal = utils.stack([s, s1, s2])
        result_list = result_signal.split(number_of_parts=3)
        assert test_axis == s.axes_manager[0].index_in_array
        assert len(result_list) == 3
        np.testing.assert_array_almost_equal(
            result_list[0].data, result_signal.inav[:, :, 0].data)

    def test_stack_of_stack(self):
        s = self.signal
        s1 = utils.stack([s] * 2)
        s2 = utils.stack([s1] * 3)
        s3 = s2.split()[0]
        s4 = s3.split()[0]
        np.testing.assert_array_almost_equal(s4.data, s.data)
        assert not hasattr(s4.original_metadata, 'stack_elements')
        assert s4.metadata.General.title == 'test'

    def test_stack_not_default(self):
        s = self.signal
        s1 = s.inav[:, :-1] + 1
        s2 = s.inav[:, ::2] * 4
        result_signal = utils.stack([s, s1, s2], axis=1)
        axis_size = s.axes_manager[1].size
        axs1 = s1.axes_manager[1].size
        result_list = result_signal.split()
        assert len(result_list) == 3
        for rs in [result_signal, utils.stack([s, s1, s2], axis='y')]:
            np.testing.assert_array_almost_equal(
                result_list[0].data, rs.inav[:, :axis_size].data)
            np.testing.assert_array_almost_equal(
                s.data, rs.inav[:, :axis_size].data)
            np.testing.assert_array_almost_equal(
                s1.data, rs.inav[:, axis_size:axis_size + axs1].data)
            np.testing.assert_array_almost_equal(
                s2.data, rs.inav[:, axis_size + axs1:].data)

    def test_stack_bigger_than_ten(self):
        s = self.signal
        list_s = [s] * 12
        list_s.append(s.deepcopy() * 3)
        s1 = utils.stack(list_s)
        res = s1.split()
        np.testing.assert_array_almost_equal(list_s[-1].data, res[-1].data)
        assert res[-1].metadata.General.title == 'test'

    def test_stack_broadcast_number(self):
        s = self.signal
        rs = utils.stack([5, s])
        np.testing.assert_array_equal(
            rs.inav[..., 0].data, 5 * np.ones((3, 2, 5)))

    def test_stack_broadcast_number_not_default(self):
        s = self.signal
        rs = utils.stack([5, s], axis='E')
        np.testing.assert_array_equal(rs.isig[0].data, 5 * np.ones((3, 2)))
