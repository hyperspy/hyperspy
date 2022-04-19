# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
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

import pytest
import numpy as np
from scipy.stats import norm

from hyperspy.signals import Signal2D, BaseSignal, Signal1D
from hyperspy._signals.lazy import LazySignal
from hyperspy.decorators import lazifyTestClass
from hyperspy.signal_tools import PeaksFinder2D
from hyperspy.ui_registry import TOOLKIT_REGISTRY


def _generate_dataset():
    coefficients = np.array(
        [350949.04890400 + 0.j, -22003.98742841 + 51494.56650429j,
         37292.52741553 + 38067.97686711j, 37292.52741553 - 38067.97686711j,
         -22003.98742841 - 51494.56650429j]
    )
    coordinates = np.array([[0, 26, 30, 994, 998],
                            [0, 1003, 39, 985, 21]]
    )
    dense = np.zeros((1024, 1024), dtype=complex)
    dense[coordinates[0], coordinates[1]] = coefficients
    dense = Signal2D(np.real(np.fft.ifft2(dense)))
    dense = dense.isig[500:550, 500:550]

    coefficients = np.array(
        [10, 5, 86, 221, 6, 95, 70, 12, 255, 5, 255, 3, 23,
         24, 77, 255, 11, 255, 8, 35, 195, 165, 27, 255, 8, 14,
         255, 21, 53, 107, 255, 18, 255, 4, 26, 255, 39, 27, 255,
         6, 255, 7, 13, 37, 35, 9, 83]
    )
    coordinates = np.array(
        [[3, 40],    [3, 138],  [9, 67],   [14, 95],   [20, 23],
         [20, 122],  [26, 51],  [26, 100], [31, 78],   [31, 128],
         [37, 107],  [38, 7],   [43, 34],  [43, 84],   [43, 134],
         [49, 62],   [49, 112], [54, 90],  [60, 17],   [60, 67],
         [60, 118],  [66, 45],  [66, 96],  [72, 73],   [72, 124],
         [77, 51],   [77, 101], [83, 28],  [83, 79],   [83, 130],
         [89, 57],   [89, 107], [95, 85],  [101, 12],  [101, 62],
         [101, 113], [106, 40], [107, 91], [112, 68],  [113, 119],
         [119, 97],  [124, 23], [124, 74], [124, 125], [130, 51],
         [130, 103], [136, 80]])
    sparse = np.zeros((144, 144))
    xs, ys = np.ogrid[:144, :144]
    for (x0, y0), a in zip(coordinates, coefficients):
        sparse += a * norm.pdf(xs, x0)*norm.pdf(ys, y0)
    sparse = sparse[50:100, 50:100]
    sparse_nav0d = Signal2D(sparse)
    sparse_nav1d = Signal2D(np.stack([sparse]*2))
    sparse_nav2d = Signal2D(np.stack([[sparse]*2]*3))
    shifts = np.array([[2*i, 2*i] for i in range(sparse_nav2d.axes_manager.navigation_size)])
    shifts = shifts.reshape(3, 2, 2)
    shifts = Signal1D(-shifts)
    sparse_nav2d_shifted = sparse_nav2d.deepcopy()
    sparse_nav2d_shifted.align2D(shifts=shifts, fill_value=0)

    return dense, sparse_nav0d, sparse_nav1d, sparse_nav2d, sparse_nav2d_shifted


def _generate_reference():
    xref, yref = 72, 72
    ref = np.zeros((144, 144))
    xs, ys = np.ogrid[:144, :144]
    ref += 100 * norm.pdf(xs, xref)*norm.pdf(ys, yref)
    return Signal2D(ref), xref, yref


def _get_disc():
    disc = np.zeros((11, 11))
    disc[2:9, 2:9] = 0.5
    disc[4:7, 4:7] = 0.75
    disc[5, 5] = 1
    return disc


PEAK_METHODS = ['local_max', 'max', 'minmax', 'zaefferer', 'stat',
                'laplacian_of_gaussian', 'difference_of_gaussian',
                'template_matching']
DATASETS = _generate_dataset()
DATASETS_NAME = ["dense", "sparse_nav0d", "sparse_nav1d", "sparse_nav2d"]
DISC = _get_disc()
GUI_INSTALLED = len(TOOLKIT_REGISTRY) > 0


@lazifyTestClass
class TestFindPeaks2D:

    def setup_method(self, method):
        # All these signal needs to be in the `setup_method` to get "lazified"
        self.dense = DATASETS[0]
        self.sparse_nav0d = DATASETS[1]
        self.sparse_nav1d = DATASETS[2]
        self.sparse_nav2d = DATASETS[3]
        self.sparse_nav2d_shifted = DATASETS[4]
        self.ref, self.xref, self.yref = _generate_reference()

    @pytest.mark.parametrize('method', PEAK_METHODS)
    @pytest.mark.parametrize('dataset_name', DATASETS_NAME)
    @pytest.mark.parametrize('parallel', [True, False])
    def test_find_peaks(self, method, dataset_name, parallel):
        if method == 'stat':
            pytest.importorskip("sklearn")
        dataset = getattr(self, dataset_name)
        # Parallel is not used in `map` for lazy signal
        if parallel and dataset._lazy:
            pytest.skip("Parallel=True is ignored for lazy signal.")

        if method == 'template_matching':
            peaks = dataset.find_peaks(method=method, parallel=parallel,
                                       interactive=False, template=DISC)
        else:
            peaks = dataset.find_peaks(method=method, parallel=parallel,
                                       interactive=False)
        assert isinstance(peaks, BaseSignal)
        assert not isinstance(peaks, LazySignal)

        # Check navigation shape
        np.testing.assert_equal(dataset.axes_manager.navigation_shape,
                        peaks.axes_manager.navigation_shape)
        if dataset.axes_manager.navigation_size == 0:
            shape = (1,)
        else:
            shape = dataset.axes_manager.navigation_shape[::-1]
        assert peaks.data.shape == shape
        assert peaks.data[0].shape[-1] == 2

    @pytest.mark.parametrize('parallel', [True, False])
    def test_ordering_results(self, parallel):
        peaks = self.sparse_nav2d_shifted.find_peaks(parallel=parallel,
                                                     interactive=False)

        peaks0 = peaks.inav[0]
        if peaks0._lazy:
            peaks0.data.compute()
        np.testing.assert_equal(peaks0.data[0],
                                np.array([[27,  1],
                                          [10, 17],
                                          [22, 23],
                                          [33, 29]]))
        np.testing.assert_equal(peaks0.data[1],
                                np.array([[35,  3],
                                          [ 6, 13],
                                          [18, 19],
                                          [29, 25]]))

    @pytest.mark.parametrize('method', PEAK_METHODS)
    @pytest.mark.parametrize('parallel', [True, False])
    def test_gets_right_answer(self, method, parallel):
        if method == 'stat':
            pytest.importorskip("sklearn")
        ans = np.empty((1,), dtype=object)
        ans[0] = np.array([[self.xref, self.yref]])
        if method == 'template_matching':
            disc = np.zeros((5, 5))
            disc[1:4, 1:4] = 0.5
            disc[2,2] = 1
            peaks = self.ref.find_peaks(method=method, parallel=parallel,
                                        interactive=False, template=disc)
        else:
            peaks = self.ref.find_peaks(method=method, parallel=parallel,
                                        interactive=False)
        np.testing.assert_allclose(peaks.data[0], ans[0])

    def test_return_peaks(self):
        sig = self.sparse_nav2d_shifted
        sig.axes_manager.indices = (0, 0)
        axes_dict = sig.axes_manager._get_axes_dicts(
            sig.axes_manager.navigation_axes)
        peaks = BaseSignal(np.empty(sig.axes_manager.navigation_shape),
                           axes=axes_dict)
        pf2D = PeaksFinder2D(sig, method='local_max', peaks=peaks)
        np.testing.assert_allclose(peaks.data, np.array([[22, 23]]))

        pf2D.local_max_threshold = 2
        pf2D._update_peak_finding()
        result_index0 = np.array([[10, 17], [22, 23], [33, 29]])
        np.testing.assert_allclose(peaks.data, result_index0)
        pf2D.compute_navigation()
        pf2D.close()

        assert peaks.data.shape == (3, 2)


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
@pytest.mark.parametrize('method', ['local_max', 'minmax', 'zaefferer', 'stat',
                                    'laplacian_of_gaussian',
                                    'difference_of_gaussian'])
def test_find_peaks_zeros(method):
    if method == 'stat':
        pytest.importorskip("sklearn")
    n = 64
    s = Signal2D(np.zeros([n]*2))
    s.find_peaks(method=method, interactive=False)


@pytest.mark.skipif(not GUI_INSTALLED, reason="no GUI available")
@pytest.mark.parametrize('method', PEAK_METHODS)
def test_find_peaks_interactive(method):
    if method == 'stat':
        pytest.importorskip("sklearn")
    s = DATASETS[0]
    s.find_peaks(method=method, template=DISC)
