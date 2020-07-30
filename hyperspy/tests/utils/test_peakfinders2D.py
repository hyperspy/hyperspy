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

import pytest
import numpy as np
from hyperspy.utils.peakfinders2D import (
    find_peaks_zaefferer,
    find_peaks_stat,
    find_peaks_dog,
    find_peaks_log,
    find_peaks_xc,
    _fast_mean,
    _fast_std,
)


def test_mean_std():
    x = np.array([1, 2, 3, 4, 5, 6])
    np.testing.assert_allclose(_fast_mean(x), _fast_mean.py_func(x))
    np.testing.assert_allclose(_fast_mean(x), np.mean(x))
    np.testing.assert_allclose(_fast_mean(x), 3.5)
    np.testing.assert_allclose(_fast_std(x), _fast_std.py_func(x))
    np.testing.assert_allclose(_fast_std(x), np.std(x))
    np.testing.assert_allclose(_fast_std(x), 1.707825)


# see https://stackoverflow.com/questions/9205081/
dispatcher = {
    "log": find_peaks_log,
    "dog": find_peaks_dog,
    "zaf": find_peaks_zaefferer,
    "stat": find_peaks_stat,
}


@pytest.fixture
def single_peak():
    pattern = np.zeros((128, 128))
    pattern[40:43, 40:43] = 1  # index 40,41,42 are greater than zero
    return pattern


@pytest.fixture
def many_peak():
    pattern = np.zeros((128, 128))
    pattern[40:43, 40:43] = 1  # index 40,41,42 are greater than zero
    pattern[70, 21] = 1  # index 70 and 21 are greater than zero
    pattern[10:13, 41:43] = 1
    pattern[100:113, 100:105] = 2
    return pattern


@pytest.fixture
def no_peak():
    pattern = np.ones((128, 128)) * 0.5
    return pattern


methods = ["zaf"]
# dog and log have no safe way of returning for an empty peak array
# stat throws an error while running


@pytest.mark.parametrize("method", methods)
def test_no_peak_case(no_peak, method):
    peaks = dispatcher[method](no_peak)
    assert np.isnan(peaks[0, 0])
    assert np.isnan(peaks[0, 1])


methods = ["zaf", "log", "dog"]


@pytest.mark.parametrize("method", methods)
@pytest.mark.filterwarnings("ignore::DeprecationWarning")  # skimage internals
def test_one_peak_case(single_peak, method):
    peaks = dispatcher[method](single_peak)
    assert peaks[0, 0] > 39.5
    assert peaks[0, 0] < 42.5
    assert peaks[0, 0] == peaks[0, 1]


methods = ["zaf", "log", "stat"]


@pytest.mark.parametrize("method", methods)
@pytest.mark.filterwarnings("ignore::DeprecationWarning")  # skimage internals
def test_many_peak_case(many_peak, method):
    peaks = dispatcher[method](many_peak)
    assert np.max(peaks) > 2


class TestXCmethods:
    @pytest.fixture
    def peaks(self):
        pattern = np.zeros((128, 128))
        pattern[40:43, 40:43] = 1  # index 40,41,42 are greater than zero
        pattern[50:52, 80:82] = 0.75
        return pattern

    @pytest.mark.filterwarnings("ignore::FutureWarning")  # skimage not us
    def test_peaks_xc(self, peaks):
        disc = np.zeros((4, 4))
        disc[1:3, 1:3] = 1
        peaks = find_peaks_xc(peaks, disc, 3)
        assert peaks.shape == (2, 2)


class TestUncoveredCodePaths:
    def test_zaf_continue(self, many_peak):
        peaks = find_peaks_zaefferer(many_peak, distance_cutoff=1e-5)
