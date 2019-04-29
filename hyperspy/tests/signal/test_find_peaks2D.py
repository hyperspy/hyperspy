# Copyright 2007-2016 The HyperSpy developers
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
# along with HyperSpy. If not, see <http://www.gnu.org/licenses/>.

import pytest
import numpy as np
import numpy.testing as nt
from scipy.stats import norm

from hyperspy.signals import Signal2D, BaseSignal
from hyperspy.decorators import lazifyTestClass

   
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
    sparse0d = Signal2D(sparse)
    sparse1d = Signal2D(np.array([sparse for i in range(2)]))
    sparse2d = Signal2D(np.array([[sparse for i in range(2)] for j in range(2)]))

    return dense, sparse0d, sparse1d, sparse2d


PEAK_METHODS = ['skimage', 'max', 'minmax', 'zaefferer', 'stat', 
                'laplacian_of_gaussians', 'difference_of_gaussians']
DATASETS = _generate_dataset()
DATASETS_NAME = ["dense", "sparse0d", "sparse1d", "sparse2d"]


@lazifyTestClass
class TestFindPeaks2D:

    def setup_method(self, method):
        self.dense = DATASETS[0]
        self.sparse0d = DATASETS[1]
        self.sparse1d = DATASETS[2]
        self.sparse2d = DATASETS[3]

    @pytest.mark.parametrize('method', PEAK_METHODS)
    @pytest.mark.parametrize('dataset_name', DATASETS_NAME)
    @pytest.mark.parametrize('parallel', [True, False])
    def test_peaks_match_input(self, method, dataset_name, parallel):
        if method=='stat':
            pytest.importorskip("sklearn")
        dataset = getattr(self, dataset_name)
        peaks = dataset.find_peaks2D(method=method, parallel=parallel)
        assert isinstance(peaks, BaseSignal)

        if dataset.axes_manager.navigation_size > 0:
            signal_shape = dataset.axes_manager.navigation_shape[::-1]
            peaks_shape = peaks.axes_manager.navigation_shape[::-1]
        else:
            signal_shape = peaks_shape = (1,)   
        nt.assert_equal(peaks_shape, signal_shape)

    @pytest.mark.parametrize('method', PEAK_METHODS)
    @pytest.mark.parametrize('dataset_name', DATASETS_NAME)
    @pytest.mark.parametrize('parallel', [True, False])
    def test_peaks_are_coordinates(self, method, dataset_name, parallel):
        if method=='stat':
            pytest.importorskip("sklearn")
        dataset = getattr(self, dataset_name)
        peaks = dataset.find_peaks2D(method=method, parallel=parallel)
        peak_shapes = np.array([peak.shape for peak in peaks.data.flatten()])
        assert np.all(peak_shapes[:, 1] == 2) 

    @pytest.mark.parametrize('method', PEAK_METHODS)
    @pytest.mark.parametrize('parallel', [True, False])
    def test_gets_right_answer(self, method, parallel):
        if method=='stat':
            pytest.importorskip("sklearn")
        xref, yref = 72, 72
        ref = np.zeros((144, 144))
        xs, ys = np.ogrid[:144, :144]
        ref += 100 * norm.pdf(xs, xref)*norm.pdf(ys, yref)
        ref = Signal2D(ref)
        ans = np.empty((1,), dtype=object)
        ans[0] = np.array([[xref, yref]])

        peaks = ref.find_peaks2D(parallel=parallel)
        assert np.all(peaks.data[0] == ans[0])
