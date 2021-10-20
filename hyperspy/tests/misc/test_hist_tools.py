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

import logging

import numpy as np
import pytest

import hyperspy.api as hs
from hyperspy.decorators import lazifyTestClass
from hyperspy.exceptions import VisibleDeprecationWarning


def generate_bad_toy_data():
    """
    Use a deliberately bad dataset here, as per
    https://github.com/hyperspy/hyperspy/issues/784,
    which previously caused a MemoryError when
    using the Freedman-Diaconis rule.
    """
    ax1 = np.exp(-np.abs(np.arange(-30, 100, 0.02)))
    ax2 = np.exp(-np.abs(np.arange(-40, 90, 0.02)))
    s1 = hs.signals.EELSSpectrum(ax1)
    s2 = hs.signals.EELSSpectrum(ax2)
    s1 = hs.stack([s1] * 5)
    s2 = hs.stack([s2] * 5)
    s1.align_zero_loss_peak(also_align=[s2])
    return s1

@pytest.mark.parametrize("bins",[10,np.linspace(1,20,num=11)])
def test_types_of_bins(bins):
    s1 = generate_bad_toy_data()
    out = s1.get_histogram(bins)
    assert out.data.shape == (10,)
    s2 = generate_bad_toy_data().as_lazy()
    out = s2.get_histogram(bins)
    assert out.data.shape == (10,)

def test_knuth_bad_data_set(caplog):
    s1 = generate_bad_toy_data()
    with caplog.at_level(logging.WARNING):
        out = s1.get_histogram("knuth")

    assert out.data.shape == (250,)
    assert "Initial estimation of number of bins using Freedman-Diaconis" in caplog.text
    assert "Capping the number of bins" in caplog.text


def test_bayesian_blocks_warning():
    s1 = generate_bad_toy_data()
    with np.errstate(divide="ignore"):  # Required here due to dataset
        with pytest.warns(
            UserWarning, match="is not fully supported in this version of HyperSpy"
        ):
            s1.get_histogram(bins="blocks")


def test_unsupported_lazy():
    s1 = generate_bad_toy_data().as_lazy()
    with pytest.raises(ValueError, match="Unrecognized 'bins' argument"):
        s1.get_histogram(bins="sturges")


@lazifyTestClass
class TestHistogramBinMethodsBadDataset:
    def setup_method(self, method):
        self.s1 = generate_bad_toy_data()

    def test_fd_logger_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            out = self.s1.get_histogram()

        assert out.data.shape == (250,)
        assert "Capping the number of bins" in caplog.text

    def test_int_bins_logger_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            out = self.s1.get_histogram(bins=251)

        assert out.data.shape == (250,)
        assert "Capping the number of bins" in caplog.text

    @pytest.mark.parametrize("bins, size", [("scott", (106,)), (10, (10,))])
    def test_working_bins(self, bins, size):
        out = self.s1.get_histogram(bins=bins)
        assert out.data.shape == size

    @pytest.mark.parametrize("bins", ["scotts", "freedman"])
    def test_deprecation_warnings(self, bins):
        with pytest.warns(
            VisibleDeprecationWarning, match="has been deprecated and will be removed"
        ):
            _ = self.s1.get_histogram(bins=bins)
