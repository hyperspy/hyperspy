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
from scipy.signal import savgol_filter
import pytest

from hyperspy.misc.tv_denoise import _tv_denoise_1d
from hyperspy.decorators import lazifyTestClass
import hyperspy.api as hs


@lazifyTestClass
class TestAlignTools:

    def setup_method(self, method):
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
        m = mock.Mock()
        s.events.data_changed.connect(m.data_changed)
        s.shift1D(-
                  1 *
                  self.ishifts[:, np.newaxis] *
                  self.scale, show_progressbar=None)
        assert m.data_changed.called
        i_zlp = s.axes_manager.signal_axes[0].value2index(0)
        assert np.allclose(s.data[:, i_zlp], 12)
        # Check that at the edges of the spectrum the value == to the
        # background value. If it wasn't it'll mean that the cropping
        # code is buggy
        assert (s.data[:, -1] == 2).all()
        assert (s.data[:, 0] == 2).all()
        # Check that the calibration is correct
        assert s.axes_manager._axes[1].offset == self.new_offset
        assert s.axes_manager._axes[1].scale == self.scale

    def test_align(self):
        s = self.signal
        s.align1D(show_progressbar=None)
        i_zlp = s.axes_manager.signal_axes[0].value2index(0)
        assert np.allclose(s.data[:, i_zlp], 12)
        # Check that at the edges of the spectrum the value == to the
        # background value. If it wasn't it'll mean that the cropping
        # code is buggy
        assert (s.data[:, -1] == 2).all()
        assert (s.data[:, 0] == 2).all()
        # Check that the calibration is correct
        assert (
            s.axes_manager._axes[1].offset == self.new_offset)
        assert s.axes_manager._axes[1].scale == self.scale

    def test_align_expand(self):
        s = self.signal
        s.align1D(expand=True)

        # Check the numbers of NaNs to make sure expansion happened properly
        Nnan = self.ishifts.max() - self.ishifts.min()
        Nnan_data = np.sum(np.isnan(s.data), axis=1)
        # Due to interpolation, the number of NaNs in the data might
        # be 2 higher (left and right side) than expected
        assert np.all(Nnan_data - Nnan <= 2)

        # Check actual alignment of zlp
        i_zlp = s.axes_manager.signal_axes[0].value2index(0)
        assert np.allclose(s.data[:, i_zlp], 12)


@lazifyTestClass
class TestShift1D:

    def setup_method(self, method):
        self.s = hs.signals.Signal1D(np.arange(10))
        self.s.axes_manager[0].scale = 0.2

    def test_crop_left(self):
        s = self.s
        s.shift1D(np.array((0.01)), crop=True, show_progressbar=None)
        assert (
            tuple(
                s.axes_manager[0].axis) == tuple(
                np.arange(
                    0.2, 2., 0.2)))

    def test_crop_right(self):
        s = self.s
        s.shift1D(np.array((-0.01)), crop=True, show_progressbar=None)
        assert (
            tuple(
                s.axes_manager[0].axis) == tuple(
                np.arange(
                    0., 1.8, 0.2)))


@lazifyTestClass
class TestFindPeaks1D:

    def setup_method(self, method):
        x = np.arange(0, 50, 0.01)
        s = hs.signals.Signal1D(np.vstack((np.cos(x), np.sin(x))))
        s.axes_manager.signal_axes[0].scale = 0.01
        self.peak_positions0 = np.arange(8) * 2 * np.pi
        self.peak_positions1 = np.arange(8) * 2 * np.pi + np.pi / 2
        self.signal = s

    def test_single_spectrum(self):
        peaks = self.signal.inav[0].find_peaks1D_ohaver()[0]
        if self.signal._lazy:
            peaks = peaks.compute()
        assert np.allclose(
            peaks['position'], self.peak_positions0, rtol=1e-5, atol=1e-4)

    def test_two_spectra(self):
        peaks = self.signal.find_peaks1D_ohaver()[1]
        if self.signal._lazy:
            peaks = peaks.compute()
        assert np.allclose(
            peaks['position'], self.peak_positions1, rtol=1e-5, atol=1e-4)

    def test_height(self):
        peaks = self.signal.find_peaks1D_ohaver()[1]
        if self.signal._lazy:
            peaks = peaks.compute()
        assert np.allclose(
            peaks['height'], 1.0, rtol=1e-5, atol=1e-4)

    def test_width(self):
        peaks = self.signal.find_peaks1D_ohaver()[1]
        if self.signal._lazy:
            peaks = peaks.compute()
        assert np.allclose(peaks['width'], 3.5758, rtol=1e-4, atol=1e-4)

    def test_n_peaks(self):
        peaks = self.signal.find_peaks1D_ohaver()[1]
        if self.signal._lazy:
            peaks = peaks.compute()
        assert len(peaks) == 8

    def test_maxpeaksn(self):
        for n in range(1, 10):
            peaks = self.signal.find_peaks1D_ohaver(maxpeakn=n)[1]
            if self.signal._lazy:
                peaks = peaks.compute()
            assert len(peaks) == min((8, n))


@lazifyTestClass
class TestInterpolateInBetween:

    def setup_method(self, method):
        s = hs.signals.Signal1D(np.arange(40).reshape((2, 20)))
        s.axes_manager.signal_axes[0].scale = 0.1
        s.isig[8:12] = 0
        self.s = s

    def test_single_spectrum(self):
        s = self.s.inav[0]
        m = mock.Mock()
        s.events.data_changed.connect(m.data_changed)
        s.interpolate_in_between(8, 12, show_progressbar=None)
        np.testing.assert_array_equal(s.data, np.arange(20))
        assert m.data_changed.called

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
        tmp = np.zeros_like(s.data)
        tmp[12] = s.data[12]
        s.data += tmp * 9.
        s.interpolate_in_between(8, 12, delta=2, kind='cubic')
        print(s.data[8:12])
        np.testing.assert_allclose(
            s.data[8:12], np.array([44., 95.4, 139.6, 155.]))

    def test_delta_float(self):
        s = self.s.inav[0]
        s.change_dtype('float')
        tmp = np.zeros_like(s.data)
        tmp[12] = s.data[12]
        s.data += tmp * 9.
        s.interpolate_in_between(8, 12, delta=0.31, kind='cubic')
        print(s.data[8:12])
        np.testing.assert_allclose(
            s.data[8:12], np.array([45.09388598, 104.16170809,
                                    155.48258721, 170.33564422]),
            atol=1,
        )


@lazifyTestClass
class TestEstimatePeakWidth:

    def setup_method(self, method):
        scale = 0.1
        window = 2
        x = np.arange(-window, window, scale)
        g = hs.model.components1D.Gaussian(sigma=0.3)
        s = hs.signals.Signal1D(g.function(x))
        s.axes_manager[-1].scale = scale
        self.s = s
        self.rtol = 1e-7
        self.atol = 0

    def test_full_range(self):
        width, left, right = self.s.estimate_peak_width(
            window=None,
            return_interval=True,
            show_progressbar=None)
        np.testing.assert_allclose(width.data, 0.7065102,
                                   rtol=self.rtol, atol=self.atol)
        np.testing.assert_allclose(left.data, 1.6467449,
                                   rtol=self.rtol, atol=self.atol)
        np.testing.assert_allclose(right.data, 2.353255,
                                   rtol=self.rtol, atol=self.atol)
        for t in (width, left, right):
            assert t.metadata.Signal.signal_type == ""
            assert t.axes_manager.signal_dimension == 0

    def test_too_narrow_range(self):
        width, left, right = self.s.estimate_peak_width(
            window=0.5,
            return_interval=True,
            show_progressbar=None)
        assert np.isnan(width.data).all()
        assert np.isnan(left.data).all()
        assert np.isnan(right.data).all()

    def test_two_peaks(self):
        s = self.s.deepcopy()
        s.shift1D(np.array([1.0]), show_progressbar=None)
        self.s = self.s.isig[10:] + s
        width, left, right = self.s.estimate_peak_width(
            window=None,
            return_interval=True,
            show_progressbar=None)
        assert np.isnan(width.data).all()
        assert np.isnan(left.data).all()
        assert np.isnan(right.data).all()


@lazifyTestClass(rtol=1e-4, atol=0.4)
class TestSmoothing:

    def setup_method(self, method):
        n, m = 2, 100
        self.s = hs.signals.Signal1D(
            np.arange(
                n * m,
                dtype='float').reshape(
                n,
                m))
        np.random.seed(1)
        self.s.add_gaussian_noise(0.1)
        self.rtol = 1e-7
        self.atol = 0

    @pytest.mark.parametrize('parallel',
                             [pytest.param(True, marks=pytest.mark.parallel), False])
    def test_lowess(self, parallel):
        pytest.importorskip("statsmodels")
        from statsmodels.nonparametric.smoothers_lowess import lowess
        frac = 0.5
        it = 1
        data = np.asanyarray(self.s.data, dtype='float')
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
                             show_progressbar=None,
                             parallel=parallel)
        np.testing.assert_allclose(self.s.data, data,
                                   rtol=self.rtol, atol=self.atol)

    @pytest.mark.parametrize('parallel',
                             [pytest.param(True, marks=pytest.mark.parallel), False])
    def test_tv(self, parallel):
        weight = 1
        data = np.asanyarray(self.s.data, dtype='float')
        for i in range(data.shape[0]):
            data[i, :] = _tv_denoise_1d(
                im=data[i, :],
                weight=weight,)
        self.s.smooth_tv(smoothing_parameter=weight,
                         show_progressbar=None,
                         parallel=parallel)
        np.testing.assert_allclose(data, self.s.data,
                                   rtol=self.rtol, atol=self.atol)

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
        np.testing.assert_allclose(data, self.s.data)


@pytest.mark.parametrize('lazy', [True, False])
@pytest.mark.parametrize('offset', [3, 0])
def test_hanning(lazy, offset):
    sig = hs.signals.Signal1D(np.random.rand(5, 20))
    if lazy:
        sig = sig.as_lazy()
    data = np.array(sig.data)
    channels = 5
    hanning = np.hanning(channels * 2)
    data[..., :offset] = 0
    data[..., offset:offset + channels] *= hanning[:channels]
    rl = None if offset == 0 else -offset
    data[..., -offset - channels:rl] *= hanning[-channels:]
    if offset != 0:
        data[..., -offset:] = 0

    assert channels == sig.hanning_taper(side='both', channels=channels,
                                         offset=offset)
    np.testing.assert_allclose(data, sig.data)


@pytest.mark.parametrize('float_data', [True, False])
def test_hanning_wrong_type(float_data):
    sig = hs.signals.Signal1D(np.arange(100).reshape(5, 20))
    if float_data:
        sig.change_dtype('float')

    if float_data:
        sig.hanning_taper()
    else:
        with pytest.raises(TypeError):
            sig.hanning_taper()
