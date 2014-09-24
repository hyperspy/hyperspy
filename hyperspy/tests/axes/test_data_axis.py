import copy
import math

import nose.tools
import numpy as np

from hyperspy.axes import DataAxis


class TestDataAxis:

    def setUp(self):
        self.axis = DataAxis(size=10, scale=0.1, offset=10)

    def test_value_range_to_indices_in_range(self):
        nose.tools.assert_equal(
            self.axis.value_range_to_indices(
                10.1, 10.8), (1, 8))

    def test_value_range_to_indices_endpoints(self):
        nose.tools.assert_equal(
            self.axis.value_range_to_indices(
                10, 10.9), (0, 9))

    def test_value_range_to_indices_out(self):
        nose.tools.assert_equal(
            self.axis.value_range_to_indices(
                9, 11), (0, 9))

    def test_value_range_to_indices_None(self):
        nose.tools.assert_equal(
            self.axis.value_range_to_indices(
                None, None), (0, 9))

    @nose.tools.raises(ValueError)
    def test_value_range_to_indices_v1_greater_than_v2(self):
        self.axis.value_range_to_indices(2, 1)

    def test_deepcopy(self):
        ac = copy.deepcopy(self.axis)
        ac.offset = 100
        nose.tools.assert_not_equal(self.axis.offset, ac.offset)

    def test_deepcopy_on_trait_change(self):
        ac = copy.deepcopy(self.axis)
        ac.offset = 100
        nose.tools.assert_equal(ac.axis[0], ac.offset)

    def test_value2index_float_in(self):
        nose.tools.assert_equal(self.axis.value2index(10.15), 2)

    def test_value2index_float_end_point_left(self):
        nose.tools.assert_equal(self.axis.value2index(10.), 0)

    def test_value2index_float_end_point_right(self):
        nose.tools.assert_equal(self.axis.value2index(10.9), 9)

    @nose.tools.raises(ValueError)
    def test_value2index_float_out(self):
        self.axis.value2index(11)

    def test_value2index_array_in(self):
        nose.tools.assert_equal(
            self.axis.value2index(np.array([10.15, 10.15])).tolist(),
            [2, 2])

    def test_value2index_array_in_ceil(self):
        nose.tools.assert_equal(
            self.axis.value2index(np.array([10.14, 10.14]),
                                  rounding=math.ceil).tolist(),
            [2, 2])

    def test_value2index_array_in_floor(self):
        nose.tools.assert_equal(
            self.axis.value2index(np.array([10.15, 10.15]),
                                  rounding=math.floor).tolist(),
            [1, 1])

    @nose.tools.raises(ValueError)
    def test_value2index_array_out(self):
        self.axis.value2index(np.array([10, 11]))

    def test_slice_me(self):
        nose.tools.assert_equal(
            self.axis._slice_me(slice(np.float32(10.2), 10.4, 2)),
            slice(2, 4, 2))
