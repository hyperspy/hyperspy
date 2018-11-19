# Copyright 2007-2016 The HyperSpy developers
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


import numpy as np
import pytest

from hyperspy import signals, model
from hyperspy.decorators import lazifyTestClass


@lazifyTestClass
class Test_Estimate_Elastic_Scattering_Threshold:

    def setup_method(self, method):
        # Create an empty spectrum
        s = signals.EELSSpectrum(np.zeros((3, 2, 1024)))
        energy_axis = s.axes_manager.signal_axes[0]
        energy_axis.scale = 0.02
        energy_axis.offset = -5

        gauss = model.components1d.Gaussian()
        gauss.centre.value = 0
        gauss.A.value = 5000
        gauss.sigma.value = 0.5
        gauss2 = model.components1d.Gaussian()
        gauss2.sigma.value = 0.5
        # Inflexion point 1.5
        gauss2.A.value = 5000
        gauss2.centre.value = 5
        s.data[:] = (gauss.function(energy_axis.axis) +
                     gauss2.function(energy_axis.axis))
        self.signal = s

    def test_min_in_window_with_smoothing(self):
        s = self.signal
        thr = s.estimate_elastic_scattering_threshold(
            window=5,
            window_length=5,
            tol=0.00001,
        )
        np.testing.assert_allclose(thr.data, 2.5, atol=10e-3)
        assert thr.metadata.Signal.signal_type == ""
        assert thr.axes_manager.signal_dimension == 0

    def test_min_in_window_without_smoothing_single_spectrum(self):
        s = self.signal.inav[0, 0]
        thr = s.estimate_elastic_scattering_threshold(
            window=5,
            window_length=0,
            tol=0.001,
        )
        np.testing.assert_allclose(thr.data, 2.49, atol=10e-3)

    def test_min_in_window_without_smoothing(self):
        s = self.signal
        thr = s.estimate_elastic_scattering_threshold(
            window=5,
            window_length=0,
            tol=0.001,
        )
        np.testing.assert_allclose(thr.data, 2.49, atol=10e-3)

    def test_min_not_in_window(self):
        # If I use a much lower window, this is the value that has to be
        # returned as threshold.
        s = self.signal
        data = s.estimate_elastic_scattering_threshold(window=1.5,
                                                       tol=0.001,
                                                       ).data
        assert np.all(np.isnan(data))

    def test_estimate_elastic_scattering_intensity(self):
        s = self.signal
        threshold = s.estimate_elastic_scattering_threshold(window=4.)
        # Threshold is nd signal
        t = s.estimate_elastic_scattering_intensity(threshold=threshold)
        assert t.metadata.Signal.signal_type == ""
        assert t.axes_manager.signal_dimension == 0
        np.testing.assert_array_almost_equal(t.data, 249999.985133)
        # Threshold is signal, 1 spectrum
        s0 = s.inav[0]
        t0 = s0.estimate_elastic_scattering_threshold(window=4.)
        t = s0.estimate_elastic_scattering_intensity(threshold=t0)
        np.testing.assert_array_almost_equal(t.data, 249999.985133)
        # Threshold is value
        t = s.estimate_elastic_scattering_intensity(threshold=2.5)
        np.testing.assert_array_almost_equal(t.data, 249999.985133)


@lazifyTestClass
class TestEstimateZLPCentre:

    def setup_method(self, method):
        s = signals.EELSSpectrum(np.diag(np.arange(1, 11)))
        s.axes_manager[-1].scale = 0.1
        s.axes_manager[-1].offset = 100
        self.signal = s

    def test_estimate_zero_loss_peak_centre(self):
        s = self.signal
        zlpc = s.estimate_zero_loss_peak_centre()
        np.testing.assert_allclose(zlpc.data,
                                   np.arange(100,
                                             101,
                                             0.1))
        assert zlpc.metadata.Signal.signal_type == ""
        assert zlpc.axes_manager.signal_dimension == 0


@lazifyTestClass
class TestAlignZLP:

    def setup_method(self, method):
        s = signals.EELSSpectrum(np.zeros((10, 100)))
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
        s.axes_manager[-1].offset += 100
        self.signal = s

    def test_align_zero_loss_peak_calibrate_true(self):
        s = self.signal
        s.align_zero_loss_peak(
            calibrate=True,
            print_stats=False,
            show_progressbar=None)
        zlpc = s.estimate_zero_loss_peak_centre()
        np.testing.assert_allclose(zlpc.data.mean(), 0)
        np.testing.assert_allclose(zlpc.data.std(), 0)

    def test_align_zero_loss_peak_calibrate_true_with_mask(self):
        s = self.signal
        mask = s._get_navigation_signal(dtype="bool").T
        mask.data[[3, 5]] = (True, True)
        s.align_zero_loss_peak(
            calibrate=True,
            print_stats=False,
            show_progressbar=None,
            mask=mask)
        zlpc = s.estimate_zero_loss_peak_centre(mask=mask)
        np.testing.assert_allclose(np.nanmean(zlpc.data), 0,
                                   atol=np.finfo(float).eps)
        np.testing.assert_allclose(np.nanstd(zlpc.data), 0,
                                   atol=np.finfo(float).eps)

    def test_align_zero_loss_peak_calibrate_false(self):
        s = self.signal
        s.align_zero_loss_peak(
            calibrate=False,
            print_stats=False,
            show_progressbar=None)
        zlpc = s.estimate_zero_loss_peak_centre()
        np.testing.assert_allclose(zlpc.data.std(), 0, atol=10e-3)

    def test_also_aligns(self):
        s = self.signal
        s2 = s.deepcopy()
        s.align_zero_loss_peak(calibrate=True,
                               print_stats=False,
                               also_align=[s2],
                               show_progressbar=None)
        zlpc = s2.estimate_zero_loss_peak_centre()
        assert zlpc.data.mean() == 0
        assert zlpc.data.std() == 0

    def test_align_zero_loss_peak_with_spike_signal_range(self):
        s = self.signal
        spike = np.zeros((10, 100))
        spike_amplitude = 20
        spike[:, 75] = spike_amplitude
        s.data += spike
        s.align_zero_loss_peak(
            print_stats=False, subpixel=False, signal_range=(98., 102.))
        zlp_max = s.isig[-0.5:0.5].max(-1).data
        # Max value in the original spectrum is 12, but due to the aligning
        # the peak is split between two different channels. So 8 is the
        # maximum value for the aligned spectrum
        assert np.allclose(zlp_max, 8)

    def test_align_zero_loss_peak_crop_false(self):
        s = self.signal
        original_size = s.axes_manager.signal_axes[0].size
        s.align_zero_loss_peak(
            crop=False,
            print_stats=False,
            show_progressbar=None)
        assert original_size == s.axes_manager.signal_axes[0].size


@lazifyTestClass
class TestPowerLawExtrapolation:

    def setup_method(self, method):
        s = signals.EELSSpectrum(0.1 * np.arange(50, 250, 0.5) ** -3.)
        s.metadata.Signal.binned = False
        s.axes_manager[-1].offset = 50
        s.axes_manager[-1].scale = 0.5
        self.s = s

    def test_unbinned(self):
        sc = self.s.isig[:300]
        s = sc.power_law_extrapolation(extrapolation_size=100)
        np.testing.assert_allclose(s.data, self.s.data, atol=10e-3)

    def test_binned(self):
        self.s.data *= self.s.axes_manager[-1].scale
        self.s.metadata.Signal.binned = True
        sc = self.s.isig[:300]
        s = sc.power_law_extrapolation(extrapolation_size=100)
        np.testing.assert_allclose(s.data, self.s.data, atol=10e-3)


@lazifyTestClass
class TestFourierRatioDeconvolution:

    @pytest.mark.parametrize(('extrapolate_lowloss'), [True, False])
    def test_running(self, extrapolate_lowloss):
        s = signals.EELSSpectrum(np.arange(200))
        gaussian = model.components1d.Gaussian()
        gaussian.A.value = 50
        gaussian.sigma.value = 10
        gaussian.centre.value = 20
        s_ll = signals.EELSSpectrum(gaussian.function(np.arange(0, 200, 1)))
        s_ll.axes_manager[0].offset = -50
        s.fourier_ratio_deconvolution(s_ll,
                                      extrapolate_lowloss=extrapolate_lowloss)


class TestRebin:
    def setup_method(self, method):
        # Create an empty spectrum
        s = signals.EELSSpectrum(np.ones((4, 2, 1024)))
        self.signal = s

    def test_rebin_without_dwell_time(self):
        s = self.signal
        s.rebin(scale=(2, 2, 1))

    def test_rebin_dwell_time(self):
        s = self.signal
        s.metadata.add_node("Acquisition_instrument.TEM.Detector.EELS")
        s_mdEELS = s.metadata.Acquisition_instrument.TEM.Detector.EELS
        s_mdEELS.dwell_time = 0.1
        s_mdEELS.exposure = 0.5
        s2 = s.rebin(scale=(2, 2, 8))
        s2_mdEELS = s2.metadata.Acquisition_instrument.TEM.Detector.EELS
        assert s2_mdEELS.dwell_time == (0.1 * 2 * 2)
        assert s2_mdEELS.exposure == (0.5 * 2 * 2)

        def test_rebin_exposure(self):
            s = self.signal
            s.metadata.exposure = 10.2
            s2 = s.rebin(scale=(2, 2, 8))
            assert s2.metadata.exposure == (10.2 * 2 * 2)

    def test_offset_after_rebin(self):
        s = self.signal
        s.axes_manager[0].offset = 1
        s.axes_manager[1].offset = 2
        s.axes_manager[2].offset = 3
        s2 = s.rebin(scale=(2, 2, 1))
        assert s2.axes_manager[0].offset == 1.5
        assert s2.axes_manager[1].offset == 2.5
        assert s2.axes_manager[2].offset == s.axes_manager[2].offset
