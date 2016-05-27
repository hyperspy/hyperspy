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

import numpy as np
import nose.tools as nt
from nose.plugins.skip import SkipTest
from scipy.signal import savgol_filter
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    skip_lowess = False
except:
    skip_lowess = True

from hyperspy.misc.tv_denoise import _tv_denoise_1d
import hyperspy.api as hs


class TestAlignTools:

    def setUp(self):
        s = hs.signals.Signal1D(np.zeros((10, 100)))
        self.scale = 0.1
        self.offset = -2
        eaxis = s.axes_manager.signal_axes[0]
        eaxis.scale = self.scale
        eaxis.offset = self.offset
        self.izlp = eaxis.value2index(0)
        self.bg = 2
        self.ishifts = np.array([0, 4, 2, -2, 5, -2, -5, -9, -9, -8])
        self.new_offset = self.offset - self.ishifts.min() * self.scale
        s.data[np.arange(10), self.ishifts + self.izlp] = 10
        s.data += self.bg
        self.signal = s

    def test_estimate_shift(self):
        s = self.signal
        eshifts = -1 * s.estimate_shift1D(show_progressbar=None)
        np.testing.assert_allclose(
            eshifts, self.ishifts * self.scale, atol=1e-3)

    def test_shift1D(self):
        s = self.signal
        s.shift1D(-
                  1 *
                  self.ishifts[:, np.newaxis] *
                  self.scale, show_progressbar=None)
        i_zlp = s.axes_manager.signal_axes[0].value2index(0)
        nt.assert_true(np.allclose(s.data[:, i_zlp], 12))
        # Check that at the edges of the spectrum the value == to the
        # background value. If it wasn't it'll mean that the cropping
        # code is buggy
        nt.assert_true((s.data[:, -1] == 2).all())
        nt.assert_true((s.data[:, 0] == 2).all())
        # Check that the calibration is correct
        nt.assert_equal(s.axes_manager._axes[1].offset, self.new_offset)
        nt.assert_equal(s.axes_manager._axes[1].scale, self.scale)

    def test_align(self):
        s = self.signal
        s.align1D(show_progressbar=None)
        i_zlp = s.axes_manager.signal_axes[0].value2index(0)
        nt.assert_true(np.allclose(s.data[:, i_zlp], 12))
        # Check that at the edges of the spectrum the value == to the
        # background value. If it wasn't it'll mean that the cropping
        # code is buggy
        nt.assert_true((s.data[:, -1] == 2).all())
        nt.assert_true((s.data[:, 0] == 2).all())
        # Check that the calibration is correct
        nt.assert_equal(
            s.axes_manager._axes[1].offset, self.new_offset)
        nt.assert_equal(s.axes_manager._axes[1].scale, self.scale)


class TestShift1D:

    def setUp(self):
        self.s = hs.signals.Signal1D(np.arange(10))
        self.s.axes_manager[0].scale = 0.2

    def test_crop_left(self):
        s = self.s
        s.shift1D(np.array((0.01)), crop=True, show_progressbar=None)
        nt.assert_equal(
            tuple(
                s.axes_manager[0].axis), tuple(
                np.arange(
                    0.2, 2., 0.2)))

    def test_crop_right(self):
        s = self.s
        s.shift1D(np.array((-0.01)), crop=True, show_progressbar=None)
        nt.assert_equal(
            tuple(
                s.axes_manager[0].axis), tuple(
                np.arange(
                    0., 1.8, 0.2)))


class TestFindPeaks1D:

    def setUp(self):
        x = np.arange(0, 50, 0.01)
        s = hs.signals.Signal1D(np.vstack((np.cos(x), np.sin(x))))
        s.axes_manager.signal_axes[0].scale = 0.01
        self.peak_positions0 = np.arange(8) * 2 * np.pi
        self.peak_positions1 = np.arange(8) * 2 * np.pi + np.pi / 2
        self.signal = s

    def test_single_spectrum(self):
        peaks = self.signal.inav[0].find_peaks1D_ohaver()
        nt.assert_true(np.allclose(
            peaks[0]['position'], self.peak_positions0, rtol=1e-5, atol=1e-4))

    def test_two_spectra(self):
        peaks = self.signal.find_peaks1D_ohaver()
        nt.assert_true(np.allclose(
            peaks[1]['position'], self.peak_positions1, rtol=1e-5, atol=1e-4))

    def test_height(self):
        peaks = self.signal.find_peaks1D_ohaver()
        nt.assert_true(np.allclose(
            peaks[1]['height'], 1.0, rtol=1e-5, atol=1e-4))

    def test_width(self):
        peaks = self.signal.find_peaks1D_ohaver()
        nt.assert_true(np.allclose(
            peaks[1]['width'], 3.5758, rtol=1e-4, atol=1e-4),
            msg="One or several widths are not close enough to expected " +
            "value (3.5758): " + str(peaks[1]['width']))

    def test_n_peaks(self):
        peaks = self.signal.find_peaks1D_ohaver()
        nt.assert_equal(len(peaks[1]), 8)

    def test_maxpeaksn(self):
        for n in range(1, 10):
            peaks = self.signal.find_peaks1D_ohaver(maxpeakn=n)
            nt.assert_equal(len(peaks[1]), min((8, n)))


class TestInterpolateInBetween:

    def setUp(self):
        s = hs.signals.Signal1D(np.arange(40).reshape((2, 20)))
        s.axes_manager.signal_axes[0].scale = 0.1
        s.isig[8:12] = 0
        self.s = s

    def test_single_spectrum(self):
        s = self.s.inav[0]
        s.interpolate_in_between(8, 12, show_progressbar=None)
        np.testing.assert_array_equal(s.data, np.arange(20))

    def test_single_spectrum_in_units(self):
        s = self.s.inav[0]
        s.interpolate_in_between(0.8, 1.2, show_progressbar=None)
        np.testing.assert_array_equal(s.data, np.arange(20))

    def test_two_spectra(self):
        s = self.s
        s.interpolate_in_between(8, 12, show_progressbar=None)
        np.testing.assert_array_equal(s.data, np.arange(40).reshape(2, 20))

    def test_delta_int(self):
        s = self.s.inav[0]
        s.change_dtype('float')
        s.data[12] *= 10
        s.interpolate_in_between(8, 12, delta=2, kind='cubic')
        print(s.data[8:12])
        np.testing.assert_allclose(
            s.data[8:12], np.array([44., 95.4, 139.6, 155.]))

    def test_delta_float(self):
        s = self.s.inav[0]
        s.change_dtype('float')
        s.data[12] *= 10.
        s.interpolate_in_between(8, 12, delta=0.31, kind='cubic')
        print(s.data[8:12])
        np.testing.assert_allclose(
            s.data[8:12], np.array([45.09388598, 104.16170809,
                                    155.48258721, 170.33564422]))


class TestEstimatePeakWidth:

    def setUp(self):
        scale = 0.1
        window = 2
        x = np.arange(-window, window, scale)
        g = hs.model.components.Gaussian()
        s = hs.signals.Signal1D(g.function(x))
        s.axes_manager[-1].scale = scale
        self.s = s

    def test_full_range(self):
        width, left, right = self.s.estimate_peak_width(
            window=None,
            return_interval=True,
            show_progressbar=None)
        nt.assert_equal(width, 2.35482074)
        nt.assert_equal(left, 0.82258963)
        nt.assert_equal(right, 3.17741037)

    def test_too_narrow_range(self):
        width, left, right = self.s.estimate_peak_width(
            window=2.2,
            return_interval=True,
            show_progressbar=None)
        nt.assert_equal(width, np.nan)
        nt.assert_equal(left, np.nan)
        nt.assert_equal(right, np.nan)

    def test_two_peaks(self):
        s = self.s.deepcopy()
        s.shift1D(np.array([0.5]), show_progressbar=None)
        self.s.isig[:-5] += s
        width, left, right = self.s.estimate_peak_width(
            window=None,
            return_interval=True,
            show_progressbar=None)
        nt.assert_equal(width, np.nan)
        nt.assert_equal(left, np.nan)
        nt.assert_equal(right, np.nan)


class TestSmoothing:

    def setUp(self):
        n, m = 2, 100
        self.s = hs.signals.SpectrumSimulation(np.arange(n * m).reshape(n, m))
        np.random.seed(1)
        self.s.add_gaussian_noise(0.1)

    def test_lowess(self):
        if skip_lowess:
            raise SkipTest
        frac = 0.5
        it = 1
        data = self.s.data.copy()
        for i in range(data.shape[0]):
            data[i, :] = lowess(
                endog=data[i, :],
                exog=self.s.axes_manager[-1].axis,
                frac=frac,
                it=it,
                is_sorted=True,
                return_sorted=False,)
        self.s.smooth_lowess(smoothing_parameter=frac,
                             number_of_iterations=it,
                             show_progressbar=None)
        nt.assert_true(np.allclose(data, self.s.data))

    def test_tv(self):
        weight = 1
        data = self.s.data.copy()
        for i in range(data.shape[0]):
            data[i, :] = _tv_denoise_1d(
                im=data[i, :],
                weight=weight,)
        self.s.smooth_tv(smoothing_parameter=weight,
                         show_progressbar=None)
        nt.assert_true(np.allclose(data, self.s.data))

    def test_savgol(self):
        window_length = 13
        polyorder = 1
        deriv = 1
        data = savgol_filter(
            x=self.s.data,
            window_length=window_length,
            polyorder=polyorder,
            deriv=deriv,
            delta=self.s.axes_manager[-1].scale,
            axis=-1,)
        self.s.smooth_savitzky_golay(
            window_length=window_length,
            polynomial_order=polyorder,
            differential_order=deriv,)
        nt.assert_true(np.allclose(data, self.s.data))
