
import pytest
import numpy as np

from hyperspy.drawing.utils import contrast_stretching


class TestImageStretching:

    def setup_method(self, method):
        self.data = np.arange(11).astype("float")
        self.data[-1] = np.nan


    @pytest.mark.parametrize("value", (("0.5th", "99.5th"), (0.045, 8.955)))
    def test_no_nans(self, value):
        data = self.data[:-1]
        bounds = contrast_stretching(data, value[0], value[1])
        assert bounds == (
            np.percentile(data, 0.5),
            np.percentile(data, 99.5))

    @pytest.mark.parametrize("value", (("0.5th", "99.5th"), (0.045, 8.955)))
    def test_nans(self, value):
        data = self.data[:-1]
        bounds = contrast_stretching(self.data, value[0], value[1])
        assert bounds == (
            np.percentile(data, 0.5),
            np.percentile(data, 99.5))

    def test_out_of_range(self):
        with pytest.raises(ValueError):
            contrast_stretching(self.data, "-0.5th", "100.5th")