import nose.tools
import numpy as np
from hyperspy.drawing.utils import contrast_stretching


class TestImageStretching:

    def setUp(self):
        self.data = np.arange(11).astype("float")
        self.data[-1] = np.nan

    def test_no_nans(self):
        data = self.data[:-1]
        bounds = contrast_stretching(data, 1)
        nose.tools.assert_tuple_equal(bounds, (
            np.percentile(data, 0.5),
            np.percentile(data, 99.5)))

    def test_nans(self):
        data = self.data[:-1]
        bounds = contrast_stretching(self.data, 1)
        nose.tools.assert_tuple_equal(bounds, (
            np.percentile(data, 0.5),
            np.percentile(data, 99.5)))

    @nose.tools.raises(ValueError)
    def test_out_of_range(self):
        contrast_stretching(self.data, -1)
