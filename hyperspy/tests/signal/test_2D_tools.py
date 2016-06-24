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


from unittest import mock

import nose.tools as nt
import numpy as np
import scipy as sp
from scipy.misc import face, ascent
from scipy.ndimage import fourier_shift

import hyperspy.api as hs


class TestSubPixelAlign:

    def setUp(self):
        ref_image = ascent()
        center = np.array((256, 256))
        shifts = np.array([(0.0, 0.0), (4.3, 2.13), (1.65, 3.58),
                           (-2.3, 2.9), (5.2, -2.1), (2.7, 2.9),
                           (5.0, 6.8), (-9.1, -9.5), (-9.0, -9.9),
                           (-6.3, -9.2)])
        s = hs.signals.Signal2D(np.zeros((10, 100, 100)))
        for i in range(10):
            # Apply each sup-pixel shift using FFT and InverseFFT
            offset_image = fourier_shift(np.fft.fftn(ref_image), shifts[i])
            offset_image = np.fft.ifftn(offset_image).real

            # Crop central regions of shifted images to avoid wrap around
            s.data[i, ...] = offset_image[center[0]:center[0] + 100,
                                          center[1]:center[1] + 100]

            self.signal = s
            self.shifts = shifts

    def test_align_subpix(self):
        # Align signal
        s = self.signal
        shifts = self.shifts
        s.align2D(shifts=shifts)
        # Compare by broadcasting
        np.testing.assert_allclose(s.data[4], s.data[0], rtol=1)


class TestAlignTools:

    def setUp(self):
        im = face(gray=True)
        self.ascent_offset = np.array((256, 256))
        s = hs.signals.Signal2D(np.zeros((10, 100, 100)))
        self.scales = np.array((0.1, 0.3))
        self.offsets = np.array((-2, -3))
        izlp = []
        for ax, offset, scale in zip(
                s.axes_manager.signal_axes, self.offsets, self.scales):
            ax.scale = scale
            ax.offset = offset
            izlp.append(ax.value2index(0))
        self.izlp = izlp
        self.ishifts = np.array([(0, 0), (4, 2), (1, 3), (-2, 2), (5, -2),
                                 (2, 2), (5, 6), (-9, -9), (-9, -9), (-6, -9)])
        self.new_offsets = self.offsets - self.ishifts.min(0) * self.scales
        zlp_pos = self.ishifts + self.izlp
        for i in range(10):
            slices = self.ascent_offset - zlp_pos[i, ...]
            s.data[i, ...] = im[slices[0]:slices[0] + 100,
                                slices[1]:slices[1] + 100]
        self.signal = s

        # How image should be after successfull alignment
        smin = self.ishifts.min(0)
        smax = self.ishifts.max(0)
        offsets = self.ascent_offset + self.offsets / self.scales - smin
        size = np.array((100, 100)) - (smax - smin)
        self.aligned = im[int(offsets[0]):int(offsets[0] + size[0]),
                          int(offsets[1]):int(offsets[1] + size[1])]

    def test_estimate_shift(self):
        s = self.signal
        shifts = s.estimate_shift2D()
        print(shifts)
        print(self.ishifts)
        nt.assert_true(np.allclose(shifts, self.ishifts))

    def test_align(self):
        # Align signal
        m = mock.Mock()
        s = self.signal
        s.events.data_changed.connect(m.data_changed)
        s.align2D()
        # Compare by broadcasting
        nt.assert_true(np.all(s.data == self.aligned))
        nt.assert_true(m.data_changed.called)

    def test_align_expand(self):
        s = self.signal
        s.align2D(expand=True)

        # Check the numbers of NaNs to make sure expansion happened properly
        ds = self.ishifts.max(0) - self.ishifts.min(0)
        Nnan = np.sum(ds) * 100 + np.prod(ds)
        Nnan_data = np.sum(1 * np.isnan(s.data), axis=(1, 2))
        # Due to interpolation, the number of NaNs in the data might
        # be 2 higher (left and right side) than expected
        nt.assert_true(np.all(Nnan_data - Nnan <= 2))

        # Check alignment is correct
        d_al = s.data[:, ds[0]:-ds[0], ds[1]:-ds[1]]
        nt.assert_true(np.all(d_al == self.aligned))


class TestFindPeaks2D:

    def setUp(self):
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
        dense = hs.signals.Signal2D(np.real(np.fft.ifft2(dense)))
        self.dense = dense.isig[500:550, 500:550]

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
            sparse += a * sp.stats.norm.pdf(xs, x0)*sp.stats.norm.pdf(ys, y0)
        sparse = sparse[50:100, 50:100]
        self.sparse0d = hs.signals.Signal2D(sparse)
        self.sparse1d = hs.signals.Signal2D(np.array([sparse for i in range(2)]))
        self.sparse2d = hs.signals.Signal2D(np.array([[sparse for i in range(2)] for j in range(2)]))
        xref, yref = 72, 72
        ref = np.zeros((144, 144))
        ref += 100 * sp.stats.norm.pdf(xs, xref)*sp.stats.norm.pdf(ys, yref)

        self.ref = hs.signals.Signal2D(ref)
        ans = np.empty((1,), dtype=object)
        ans[0] = np.array([[xref, yref]])
        self.ans = ans

        self.methods = ['skimage', 'max', 'minmax', 'zaefferer', 'stat', 'laplacian_of_gaussians', 'difference_of_gaussians']
        self.datasets = {'dense': self.dense, 'sparse0d': self.sparse0d, 'sparse1d': self.sparse1d, 'sparse2d': self.sparse2d}
        self.thetests = {'creates_array': self.creates_array, 'peaks_match_input': self.peaks_match_input, 'peaks_are_coordinates': self.peaks_are_coordinates}

    def test_properties(self):
        self.setUp()  # Test generation does not explicitly call setUp()
        for method in self.methods:
            for dataset in self.datasets:
                for each_test in self.thetests:
                    yield self.thetests[each_test], method, self.datasets[dataset]

    def test_answers(self):
        self.setUp()  # Test generation does not explicitly call setUp()
        for method in self.methods:
            yield self.gets_right_answer, method, self.ref, self.ans

    def creates_array(self, method, dataset):
        peaks = dataset.find_peaks2D(method=method)
        nt.assert_is_instance(peaks, np.ndarray)

    def peaks_match_input(self, method, dataset):
        peaks = dataset.find_peaks2D(method=method)
        signal_shape = dataset.axes_manager.navigation_shape[::-1] if dataset.axes_manager.navigation_size > 0 else (1,)
        nt.assert_equal(peaks.shape, signal_shape)

    def peaks_are_coordinates(self, method, dataset):
        peaks = dataset.find_peaks2D(method=method)
        peak_shapes = np.array([peak.shape for peak in peaks.flatten()])
        nt.assert_true(np.all(peak_shapes[:, 1] == 2))

    def gets_right_answer(self, method, dataset, answer):
        peaks = dataset.find_peaks2D()
        nt.assert_true(np.all(peaks[0] == answer[0]))


class TestFindPeaks2DInteractive:

    def setUp(self):
        from hyperspy._signals.signal2d import PeakFinder2D, Max, MinMax, Zaefferer, Stat, DifferenceOfGaussians, LaplacianOfGaussians
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
            sparse += a * sp.stats.norm.pdf(xs, x0)*sp.stats.norm.pdf(ys, y0)
        sparse = sparse[50:100, 50:100]
        sparse0d = hs.signals.Signal2D(sparse)
        sparse1d = hs.signals.Signal2D(np.array([sparse for i in range(2)]))
        sparse2d = hs.signals.Signal2D(np.array([[sparse for i in range(2)] for j in range(2)]))
        self.datasets = [sparse0d, sparse1d, sparse2d]
        self.methods = [Max, MinMax, Zaefferer, Stat, DifferenceOfGaussians, LaplacianOfGaussians]
        self.peakfinder = PeakFinder2D()

    def test_current_method_property(self):
        nt.assert_in(self.peakfinder.current_method.__class__, self.methods)







