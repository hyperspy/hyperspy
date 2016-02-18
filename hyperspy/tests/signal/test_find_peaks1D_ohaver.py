import os
from nose.tools import assert_not_equal, assert_equal
from hyperspy.api import load

my_path = os.path.dirname(__file__)


class TestFindPeaks1DOhaver():

    def setUp(self):
        self.spectrum = load(
            my_path +
            "/test_find_peaks1D_ohaver/test_find_peaks1D_ohaver.hdf5")

    def test_find_peaks1D_ohaver_high_amp_thres(self):
        spectrum = self.spectrum
        peak_list = spectrum.find_peaks1D_ohaver(amp_thresh=10.)[0]
        assert_equal(len(peak_list), 0)

    def test_find_peaks1D_ohaver_zero_value_bug(self):
        spectrum = self.spectrum
        peak_list = spectrum.find_peaks1D_ohaver()[0]
        assert_equal(len(peak_list), 48)
