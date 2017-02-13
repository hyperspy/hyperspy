
import numpy as np

from hyperspy.signal import BaseSignal
from hyperspy import utils


class TestUtilsStack:

    def setup_method(self, method):
        s = BaseSignal(np.random.random((3, 2, 5)))
        s.axes_manager.set_signal_dimension(1)
        s.axes_manager[0].name = "x"
        s.axes_manager[1].name = "y"
        s.axes_manager[2].name = "E"
        s.axes_manager[2].scale = 0.5
        s.metadata.General.title = 'test'
        self.signal = s

    def test_stack_default(self):
        s = self.signal
        s1 = s.deepcopy() + 1
        s2 = s.deepcopy() * 4
        test_axis = s.axes_manager[0].index_in_array
        result_signal = utils.stack([s, s1, s2])
        result_list = result_signal.split()
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
        s1 = s.deepcopy() + 1
        s2 = s.deepcopy() * 4
        result_signal = utils.stack([s, s1, s2], axis=1)
        axis_size = s.axes_manager[1].size
        result_list = result_signal.split()
        assert len(result_list) == 3
        np.testing.assert_array_almost_equal(
            result_list[0].data, result_signal.inav[:, :axis_size].data)
        result_signal = utils.stack([s, s1, s2], axis='y')
        np.testing.assert_array_almost_equal(
            result_list[0].data, result_signal.inav[:, :axis_size].data)

    def test_stack_bigger_than_ten(self):
        s = self.signal
        list_s = [s] * 12
        list_s.append(s.deepcopy() * 3)
        list_s[-1].metadata.General.title = 'test'
        s1 = utils.stack(list_s)
        res = s1.split()
        np.testing.assert_array_almost_equal(list_s[-1].data, res[-1].data)
        assert res[-1].metadata.General.title == 'test'

    def test_stack_broadcast_number(self):
        s = self.signal
        rs = utils.stack([5, s])
        np.testing.assert_array_equal(rs.inav[...,0].data, 5*np.ones((3,2,5)))

    def test_stack_broadcast_number_not_default(self):
        s = self.signal
        rs = utils.stack([5, s], axis='E')
        np.testing.assert_array_equal(rs.isig[0].data, 5*np.ones((3,2)))
