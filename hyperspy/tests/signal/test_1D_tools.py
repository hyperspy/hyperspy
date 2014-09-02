# Copyright 2007-2012 The HyperSpy developers
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


import numpy as np
import nose.tools
from nose.plugins.skip import SkipTest
from scipy.signal import savgol_filter
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    skip_lowess = False
except:
    skip_lowess = True

from hyperspy.misc.tv_denoise import _tv_denoise_1d
import hyperspy.hspy as hs


class TestAlignTools:

    def setUp(self):
        s = hs.signals.Spectrum(np.zeros((10, 100)))
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
        self.spectrum = s

    def test_estimate_shift(self):
        s = self.spectrum
        eshifts = -1 * s.estimate_shift1D()
        nose.tools.assert_true(np.allclose(eshifts, self.ishifts * self.scale))

    def test_shift1D(self):
        s = self.spectrum
        s.shift1D(-1 * self.ishifts[:, np.newaxis] * self.scale)
        i_zlp = s.axes_manager.signal_axes[0].value2index(0)
        nose.tools.assert_true(np.allclose(s.data[:, i_zlp], 12))
        # Check that at the edges of the spectrum the value == to the
        # background value. If it wasn't it'll mean that the cropping
        # code is buggy
        nose.tools.assert_true((s.data[:, -1] == 2).all())
        nose.tools.assert_true((s.data[:, 0] == 2).all())
        # Check that the calibration is correct
        nose.tools.assert_equal(
            s.axes_manager._axes[1].offset, self.new_offset)
        nose.tools.assert_equal(s.axes_manager._axes[1].scale, self.scale)

    def test_align(self):
        s = self.spectrum
        s.align1D()
        i_zlp = s.axes_manager.signal_axes[0].value2index(0)
        nose.tools.assert_true(np.allclose(s.data[:, i_zlp], 12))
        # Check that at the edges of the spectrum the value == to the
        # background value. If it wasn't it'll mean that the cropping
        # code is buggy
        nose.tools.assert_true((s.data[:, -1] == 2).all())
        nose.tools.assert_true((s.data[:, 0] == 2).all())
        # Check that the calibration is correct
        nose.tools.assert_equal(
            s.axes_manager._axes[1].offset, self.new_offset)
        nose.tools.assert_equal(s.axes_manager._axes[1].scale, self.scale)

    def test_align_axis0(self):
        s = self.spectrum
        s = s.swap_axes(0, 1)
        s.align1D()
        s = s.swap_axes(0, 1)
        i_zlp = s.axes_manager.signal_axes[0].value2index(0)
        nose.tools.assert_true(np.allclose(s.data[:, i_zlp], 12))
        # Check that at the edges of the spectrum the value == to the
        # background value. If it wasn't it'll mean that the cropping
        # code is buggy
        nose.tools.assert_true((s.data[:, -1] == 2).all())
        nose.tools.assert_true((s.data[:, 0] == 2).all())
        # Check that the calibration is correct
        nose.tools.assert_equal(
            s.axes_manager._axes[1].offset, self.new_offset)
        nose.tools.assert_equal(s.axes_manager._axes[1].scale, self.scale)


class TestShift1D():

    def setUp(self):
        self.s = hs.signals.Spectrum(np.arange(10))
        self.s.axes_manager[0].scale = 0.2

    def test_crop_left(self):
        s = self.s
        s.shift1D(np.array((0.01)), crop=True)
        nose.tools.assert_equal(
            tuple(
                s.axes_manager[0].axis), tuple(
                np.arange(
                    0.2, 2., 0.2)))

    def test_crop_right(self):
        s = self.s
        s.shift1D(np.array((-0.01)), crop=True)
        nose.tools.assert_equal(
            tuple(
                s.axes_manager[0].axis), tuple(
                np.arange(
                    0., 1.8, 0.2)))


class TestFindPeaks1D:

    def setUp(self):
        x = np.arange(0, 50, 0.01)
        s = hs.signals.Spectrum(np.vstack((np.cos(x), np.sin(x))))
        s.axes_manager.signal_axes[0].scale = 0.01
        self.peak_positions0 = np.arange(8) * 2 * np.pi
        self.peak_positions1 = np.arange(8) * 2 * np.pi + np.pi / 2
        self.spectrum = s

    def test_single_spectrum(self):
        peaks = self.spectrum[0].find_peaks1D_ohaver()
        nose.tools.assert_true(np.allclose(peaks[0]['position'],
                                           self.peak_positions0, rtol=1e-5, atol=1e-4))

    def test_two_spectra(self):
        peaks = self.spectrum.find_peaks1D_ohaver()
        peaks = self.spectrum.find_peaks1D_ohaver()
        nose.tools.assert_true(np.allclose(peaks[1]['position'],
                                           self.peak_positions1, rtol=1e-5, atol=1e-4))


class TestInterpolateInBetween:

    def setUp(self):
        s = hs.signals.Spectrum(np.arange(40).reshape((2, 20)))
        s.axes_manager.signal_axes[0].scale = 0.1
        s[:, 8:12] = 0
        self.s = s

    def test_single_spectrum(self):
        s = self.s[0]
        s.interpolate_in_between(8, 12)
        nose.tools.assert_true((s.data == np.arange(20)).all())

    def test_single_spectrum_in_units(self):
        s = self.s[0]
        s.interpolate_in_between(0.8, 1.2)
        nose.tools.assert_true((s.data == np.arange(20)).all())

    def test_two_spectra(self):
        s = self.s
        s.interpolate_in_between(8, 12)
        nose.tools.assert_true((s.data == np.arange(40).reshape(2, 20)).all())


class TestEstimatePeakWidth():

    def setUp(self):
        scale = 0.1
        window = 2
        x = np.arange(-window, window, scale)
        g = hs.components.Gaussian()
        s = hs.signals.Spectrum(g.function(x))
        s.axes_manager[-1].scale = scale
        self.s = s

    def test_full_range(self):
        width, left, right = self.s.estimate_peak_width(
            window=None,
            return_interval=True)
        nose.tools.assert_equal(width, 2.35482074)
        nose.tools.assert_equal(left, 0.82258963)
        nose.tools.assert_equal(right, 3.17741037)

    def test_too_narrow_range(self):
        width, left, right = self.s.estimate_peak_width(
            window=2.2,
            return_interval=True)
        nose.tools.assert_equal(width, np.nan)
        nose.tools.assert_equal(left, np.nan)
        nose.tools.assert_equal(right, np.nan)

    def test_two_peaks(self):
        s = self.s.deepcopy()
        s.shift1D(np.array([0.5]))
        self.s += s
        width, left, right = self.s.estimate_peak_width(
            window=None,
            return_interval=True)
        nose.tools.assert_equal(width, np.nan)
        nose.tools.assert_equal(left, np.nan)
        nose.tools.assert_equal(right, np.nan)


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
                             number_of_iterations=it,)
        nose.tools.assert_true(np.allclose(data, self.s.data))

    def test_tv(self):
        weight = 1
        data = self.s.data.copy()
        for i in range(data.shape[0]):
            data[i, :] = _tv_denoise_1d(
                im=data[i, :],
                weight=weight,)
        self.s.smooth_tv(smoothing_parameter=weight,)
        nose.tools.assert_true(np.allclose(data, self.s.data))

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
        nose.tools.assert_true(np.allclose(data, self.s.data))
