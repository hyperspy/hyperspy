import nose.tools as nt
import h5py
import numpy as np

from hyperspy.signal import Signal
from hyperspy import utils


class TestUtilsStack:

    def setUp(self):
        s = Signal(np.arange(3 * 2 * 5).reshape((3, 2, 5)))
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
        result_signal = utils.stack([s, s1, s2])
        result_list = result_signal.split()
        nt.assert_true(len(result_list) == 3)
        nt.assert_true(
            (result_list[0].data == result_signal.data[0, ...]).all())

    def test_stack_of_stack(self):
        s = self.signal
        s1 = utils.stack([s] * 2)
        s2 = utils.stack([s1] * 3)
        s3 = s2.split()[0]
        s4 = s3.split()[0]
        nt.assert_true((s4.data == s.data).all())
        nt.assert_true(
            (hasattr(
                s4.original_metadata,
                'stack_elements')is False))
        nt.assert_true((s4.metadata.General.title == 'test'))

    def test_stack_not_default(self):
        s = self.signal
        s1 = s.deepcopy() + 1
        s2 = s.deepcopy() * 4
        result_signal = utils.stack([s, s1, s2], axis=1)
        result_list = result_signal.split()
        nt.assert_true(len(result_list) == 3)
        np.testing.assert_array_equal(result_list[0].data, result_signal.data[:3, :,:])
        result_signal = utils.stack([s, s1, s2], axis='y')
        np.testing.assert_array_equal(result_list[0].data, result_signal.data[:3, :,:])

    def test_stack_bigger_than_ten(self):
        s = self.signal
        list_s = [s] * 12
        list_s.append(s.deepcopy() * 3)
        list_s[-1].metadata.General.title = 'test'
        s1 = utils.stack(list_s)
        res = s1.split()
        nt.assert_true((list_s[-1].data == res[-1].data).all())
        nt.assert_true((res[-1].metadata.General.title == 'test'))

    def test_stack_oom_default(self):
        s = self.signal
        s1 = s.deepcopy() + 1
        s2 = s.deepcopy() * 4
        result_signal = utils.stack([s, s1, s2], load_to_memory=False)
        nt.assert_is_instance(result_signal.data, h5py.Dataset)
        nt.assert_true(hasattr(result_signal, '_tempfile'))
        nt.assert_equal(result_signal.data.shape, (3, 3, 2, 5))
        for res_d, _s in zip(result_signal.data, [s, s1, s2]):
            np.testing.assert_array_equal(res_d, _s.data)

    def test_stack_oom_not_default(self):
        s = self.signal
        s1 = s.deepcopy() + 1
        s2 = s.deepcopy() * 4
        result_signal = utils.stack([s, s1, s2], load_to_memory=False, axis=1)
        nt.assert_is_instance(result_signal.data, h5py.Dataset)
        nt.assert_true(hasattr(result_signal, '_tempfile'))
        nt.assert_equal(result_signal.data.shape, (9, 2, 5))
        for res_d, _s in zip(np.split(result_signal.data.value, 3), [s, s1, s2]):
            np.testing.assert_array_equal(res_d, _s.data)
