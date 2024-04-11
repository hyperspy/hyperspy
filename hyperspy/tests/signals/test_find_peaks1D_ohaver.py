# -*- coding: utf-8 -*-
# Copyright 2007-2024 The HyperSpy developers
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

from pathlib import Path

import pytest

from hyperspy.api import load
from hyperspy.decorators import lazifyTestClass
from hyperspy.exceptions import VisibleDeprecationWarning


@lazifyTestClass
class TestFindPeaks1DOhaver:
    def setup_method(self, method):
        with pytest.warns(VisibleDeprecationWarning):
            self.signal = load(
                Path(__file__).parent / "data" / "test_find_peaks1D_ohaver.hdf5",
                reader="hspy",
            )

    def test_find_peaks1D_ohaver_high_amp_thres(self):
        signal1D = self.signal
        # Disable medium filter to avoid warning; reason unknown
        # UserWarning: kernel_size exceeds volume extent: the volume will be zero-padded.
        # d = np.gradient(medfilt(y, medfilt_radius))
        peak_list = signal1D.find_peaks1D_ohaver(amp_thresh=10.0, medfilt_radius=0)[0]
        assert len(peak_list) == 0

    def test_find_peaks1D_ohaver_zero_value_bug(self):
        signal1D = self.signal
        peak_list = signal1D.find_peaks1D_ohaver()[0]
        assert len(peak_list) == 48
