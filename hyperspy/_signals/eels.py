# -*- coding: utf-8 -*-
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

import numbers
import logging

import numpy as np
import dask.array as da
import traits.api as t
from scipy import constants

from hyperspy._signals.signal1d import (Signal1D, LazySignal1D)
from hyperspy.misc.elements import elements as elements_db
import hyperspy.axes
from hyperspy.decorators import only_interactive
from hyperspy.gui.eels import TEMParametersUI
from hyperspy.defaults_parser import preferences
import hyperspy.gui.messages as messagesui
from hyperspy.external.progressbar import progressbar
from hyperspy.components1d import PowerLaw
from hyperspy.misc.utils import isiterable, closest_power_of_two, underline
from hyperspy.misc.utils import without_nans

_logger = logging.getLogger(__name__)


class EELSSpectrum_mixin:

    _signal_type = "EELS"
    _alias_signal_types = ["TEM EELS"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Attributes defaults
        self.subshells = set()
        self.elements = set()
        self.edges = list()
        if hasattr(self.metadata, 'Sample') and \
                hasattr(self.metadata.Sample, 'elements'):
            self.add_elements(self.metadata.Sample.elements)
        self.metadata.Signal.binned = True

    def add_elements(self, elements, include_pre_edges=False):
        """Declare the elemental composition of the sample.

        The ionisation edges of the elements present in the current
        energy range will be added automatically.

        Parameters
        ----------
        elements : tuple of strings
            The symbol of the elements. Note this input must always be
            in the form of a tuple. Meaning: add_elements(('C',)) will
            work, while add_elements(('C')) will NOT work.
        include_pre_edges : bool
            If True, the ionization edges with an onset below the lower
            energy limit of the SI will be incluided

        Examples
        --------

        >>> s = hs.signals.EELSSpectrum(np.arange(1024))
        >>> s.add_elements(('C', 'O'))

        Raises
        ------
        ValueError

        """
        if not isiterable(elements) or isinstance(elements, str):
            raise ValueError(
                "Input must be in the form of a tuple. For example, "
                "if `s` is the variable containing this EELS spectrum:\n "
                ">>> s.add_elements(('C',))\n"
                "See the docstring for more information.")

        for element in elements:
            if isinstance(element, bytes):
                element = element.decode()
            if element in elements_db:
                self.elements.add(element)
            else:
                raise ValueError(
                    "%s is not a valid symbol of a chemical element"
                    % element)
        if not hasattr(self.metadata, 'Sample'):
            self.metadata.add_node('Sample')
        self.metadata.Sample.elements = list(self.elements)
        if self.elements:
            self.generate_subshells(include_pre_edges)

    def generate_subshells(self, include_pre_edges=False):
        """Calculate the subshells for the current energy range for the
        elements present in self.elements

        Parameters
        ----------
        include_pre_edges : bool
            If True, the ionization edges with an onset below the lower
            energy limit of the SI will be incluided

        """
        Eaxis = self.axes_manager.signal_axes[0].axis
        if not include_pre_edges:
            start_energy = Eaxis[0]
        else:
            start_energy = 0.
        end_energy = Eaxis[-1]
        for element in self.elements:
            e_shells = list()
            for shell in elements_db[element][
                    'Atomic_properties']['Binding_energies']:
                if shell[-1] != 'a':
                    energy = (elements_db[element]['Atomic_properties']
                              ['Binding_energies'][shell]['onset_energy (eV)'])
                    if start_energy <= energy <= end_energy:
                        subshell = '%s_%s' % (element, shell)
                        if subshell not in self.subshells:
                            self.subshells.add(
                                '%s_%s' % (element, shell))
                            e_shells.append(subshell)

    def estimate_zero_loss_peak_centre(self, mask=None):
        """Estimate the posision of the zero-loss peak.

        This function provides just a coarse estimation of the position
        of the zero-loss peak centre by computing the position of the maximum
        of the spectra. For subpixel accuracy use `estimate_shift1D`.

        Parameters
        ----------
        mask : Signal1D of bool data type.
            It must have signal_dimension = 0 and navigation_shape equal to the
            current signal. Where mask is True the shift is not computed
            and set to nan.

        Returns
        -------
        zlpc : Signal1D subclass
            The estimated position of the maximum of the ZLP peak.

        Notes
        -----
        This function only works when the zero-loss peak is the most
        intense feature in the spectrum. If it is not in most cases
        the spectrum can be cropped to meet this criterium.
        Alternatively use `estimate_shift1D`.

        See Also
        --------
        estimate_shift1D, align_zero_loss_peak

        """
        self._check_signal_dimension_equals_one()
        self._check_navigation_mask(mask)
        zlpc = self.valuemax(-1)
        if mask is not None:
            if zlpc._lazy:
                zlpc.data = da.where(mask.data, np.nan, zlpc.data)
            else:
                zlpc.data[mask.data] = np.nan
        zlpc.set_signal_type("")
        title = self.metadata.General.title
        zlpc.metadata.General.title = "ZLP(%s)" % title
        return zlpc

    def align_zero_loss_peak(
            self,
            calibrate=True,
            also_align=[],
            print_stats=True,
            subpixel=True,
            mask=None,
            signal_range=None,
            show_progressbar=None,
            **kwargs):
        """Align the zero-loss peak.

        This function first aligns the spectra using the result of
        `estimate_zero_loss_peak_centre` and afterward, if subpixel is True,
        proceeds to align with subpixel accuracy using `align1D`. The offset
        is automatically correct if `calibrate` is True.

        Parameters
        ----------
        calibrate : bool
            If True, set the offset of the spectral axis so that the
            zero-loss peak is at position zero.
        also_align : list of signals
            A list containing other spectra of identical dimensions to
            align using the shifts applied to the current spectrum.
            If `calibrate` is True, the calibration is also applied to
            the spectra in the list.
        print_stats : bool
            If True, print summary statistics of the ZLP maximum before
            the aligment.
        subpixel : bool
            If True, perform the alignment with subpixel accuracy
            using cross-correlation.
        mask : Signal1D of bool data type.
            It must have signal_dimension = 0 and navigation_shape equal to the
            current signal. Where mask is True the shift is not computed
            and set to nan.
        signal_range : tuple of integers, tuple of floats. Optional
            Will only search for the ZLP within the signal_range. If given
            in integers, the range will be in index values. If given floats,
            the range will be in spectrum values. Useful if there are features
            in the spectrum which are more intense than the ZLP.
            Default is searching in the whole signal.
        show_progressbar : None or bool
            If True, display a progress bar. If None the default is set in
            `preferences`.

        Examples
        --------
        >>> s_ll = hs.signals.EELSSpectrum(np.zeros(1000))
        >>> s_ll.data[100] = 100
        >>> s_ll.align_zero_loss_peak()

        Aligning both the lowloss signal and another signal
        >>> s = hs.signals.EELSSpectrum(np.range(1000))
        >>> s_ll.align_zero_loss_peak(also_align=[s])

        Aligning within a narrow range of the lowloss signal
        >>> s_ll.align_zero_loss_peak(signal_range=(-10.,10.))


        See Also
        --------
        estimate_zero_loss_peak_centre, align1D, estimate_shift1D.

        Notes
        -----
        Any extra keyword arguments are passed to `align1D`. For
        more information read its docstring.

        """
        def substract_from_offset(value, signals):
            if isinstance(value, da.Array):
                value = value.compute()
            for signal in signals:
                signal.axes_manager[-1].offset -= value

        def estimate_zero_loss_peak_centre(s, mask, signal_range):
            if signal_range:
                zlpc = s.isig[signal_range[0]:signal_range[1]].\
                    estimate_zero_loss_peak_centre(mask=mask)
            else:
                zlpc = s.estimate_zero_loss_peak_centre(mask=mask)
            return zlpc

        zlpc = estimate_zero_loss_peak_centre(self, mask, signal_range)
        mean_ = without_nans(zlpc.data).mean()
        if print_stats is True:
            print()
            print(underline("Initial ZLP position statistics"))
            zlpc.print_summary_statistics()

        for signal in also_align + [self]:
            signal.shift1D(-
                           zlpc.data +
                           mean_, show_progressbar=show_progressbar)

        if calibrate is True:
            zlpc = estimate_zero_loss_peak_centre(self, mask, signal_range)
            substract_from_offset(without_nans(zlpc.data).mean(),
                                  also_align + [self])

        if subpixel is False:
            return
        left, right = -3., 3.
        if calibrate is False:
            mean_ = without_nans(estimate_zero_loss_peak_centre(
                self, mask, signal_range).data).mean()
            left += mean_
            right += mean_

        left = (left if left > self.axes_manager[-1].axis[0]
                else self.axes_manager[-1].axis[0])
        right = (right if right < self.axes_manager[-1].axis[-1]
                 else self.axes_manager[-1].axis[-1])
        if self.axes_manager.navigation_size > 1:
            self.align1D(
                left,
                right,
                also_align=also_align,
                show_progressbar=show_progressbar,
                **kwargs)
        zlpc = self.estimate_zero_loss_peak_centre(mask=mask)
        if calibrate is True:
            substract_from_offset(without_nans(zlpc.data).mean(),
                                  also_align + [self])

    def estimate_elastic_scattering_intensity(
            self, threshold, show_progressbar=None):
        """Rough estimation of the elastic scattering intensity by
        truncation of a EELS low-loss spectrum.

        Parameters
        ----------
        threshold : {Signal1D, float, int}
            Truncation energy to estimate the intensity of the elastic
            scattering. The threshold can be provided as a signal of the same
            dimension as the input spectrum navigation space containing the
            threshold value in the energy units. Alternatively a constant
            threshold can be specified in energy/index units by passing
            float/int.
        show_progressbar : None or bool
            If True, display a progress bar. If None the default is set in
            `preferences`.


        Returns
        -------
        I0: Signal1D
            The elastic scattering intensity.

        See Also
        --------
        estimate_elastic_scattering_threshold

        """
        # TODO: Write units tests
        self._check_signal_dimension_equals_one()

        if show_progressbar is None:
            show_progressbar = preferences.General.show_progressbar

        if isinstance(threshold, numbers.Number):
            I0 = self.isig[:threshold].integrate1D(-1)
        else:
            ax = self.axes_manager.signal_axes[0]
            # I0 = self._get_navigation_signal()
            # I0.axes_manager.set_signal_dimension(0)
            threshold.axes_manager.set_signal_dimension(0)
            binned = self.metadata.Signal.binned

            def estimating_function(data, threshold=None):
                if np.isnan(threshold):
                    return np.nan
                else:
                    # the object is just an array, so have to reimplement
                    # integrate1D. However can make certain assumptions, for
                    # example 1D signal and pretty much always binned. Should
                    # probably at some point be joint
                    ind = ax.value2index(threshold)
                    data = data[:ind]
                    if binned:
                        return data.sum()
                    else:
                        from scipy.integrate import simps
                        axis = ax.axis[:ind]
                        return simps(y=data, x=axis)

            I0 = self.map(estimating_function, threshold=threshold,
                          ragged=False, show_progressbar=show_progressbar,
                          inplace=False)
        I0.metadata.General.title = (
            self.metadata.General.title + ' elastic intensity')
        I0.set_signal_type("")
        if self.tmp_parameters.has_item('filename'):
            I0.tmp_parameters.filename = (
                self.tmp_parameters.filename +
                '_elastic_intensity')
            I0.tmp_parameters.folder = self.tmp_parameters.folder
            I0.tmp_parameters.extension = \
                self.tmp_parameters.extension
        return I0

    def estimate_elastic_scattering_threshold(self,
                                              window=10.,
                                              tol=None,
                                              window_length=5,
                                              polynomial_order=3,
                                              start=1.):
        """Calculate the first inflexion point of the spectrum derivative
        within a window.

        This method assumes that the zero-loss peak is located at position zero
        in all the spectra. Currently it looks for an inflexion point, that can
        be a local maximum or minimum. Therefore, to estimate the elastic
        scattering threshold `start` + `window` must be less than the first
        maximum for all spectra (often the bulk plasmon maximum). If there is
        more than one inflexion point in energy the window it selects the
        smoother one what, often, but not always, is a good choice in this
        case.

        Parameters
        ----------
        window : {None, float}
            If None, the search for the local inflexion point is performed
            using the full energy range. A positive float will restrict
            the search to the (0,window] energy window, where window is given
            in the axis units. If no inflexion point is found in this
            spectral range the window value is returned instead.
        tol : {None, float}
            The threshold tolerance for the derivative. If "auto" it is
            automatically calculated as the minimum value that guarantees
            finding an inflexion point in all the spectra in given energy
            range.
        window_length : int
            If non zero performs order three Savitzky-Golay smoothing
            to the data to avoid falling in local minima caused by
            the noise. It must be an odd interger.
        polynomial_order : int
            Savitzky-Golay filter polynomial order.
        start : float
            Position from the zero-loss peak centre from where to start
            looking for the inflexion point.


        Returns
        -------

        threshold : Signal1D
            A Signal1D of the same dimension as the input spectrum
            navigation space containing the estimated threshold. Where the
            threshold couldn't be estimated the value is set to nan.

        See Also
        --------

        estimate_elastic_scattering_intensity,align_zero_loss_peak,
        find_peaks1D_ohaver, fourier_ratio_deconvolution.

        Notes
        -----

        The main purpose of this method is to be used as input for
        `estimate_elastic_scattering_intensity`. Indeed, for currently
        achievable energy resolutions, there is not such a thing as a elastic
        scattering threshold. Therefore, please be aware of the limitations of
        this method when using it.

        """
        self._check_signal_dimension_equals_one()
        # Create threshold with the same shape as the navigation dims.
        threshold = self._get_navigation_signal().transpose(signal_axes=0)

        # Progress Bar
        axis = self.axes_manager.signal_axes[0]
        min_index, max_index = axis.value_range_to_indices(start,
                                                           start + window)
        if max_index < min_index + 10:
            raise ValueError("Please select a bigger window")
        s = self.isig[min_index:max_index].deepcopy()
        if window_length:
            s.smooth_savitzky_golay(polynomial_order=polynomial_order,
                                    window_length=window_length,
                                    differential_order=1)
        else:
            s = s.diff(-1)
        if tol is None:
            tol = np.max(np.abs(s.data).min(axis.index_in_array))
        saxis = s.axes_manager[-1]
        inflexion = (np.abs(s.data) <= tol).argmax(saxis.index_in_array)
        threshold.data[:] = saxis.index2value(inflexion)
        if isinstance(inflexion, np.ndarray):
            threshold.data[inflexion == 0] = np.nan
        else:  # Single spectrum
            if inflexion == 0:
                threshold.data[:] = np.nan
        del s
        if np.isnan(threshold.data).any():
            _logger.warning(
                "No inflexion point could be found in some positions "
                "that have been marked with nans.")
        # Create spectrum image, stop and return value
        threshold.metadata.General.title = (
            self.metadata.General.title +
            ' elastic scattering threshold')
        if self.tmp_parameters.has_item('filename'):
            threshold.tmp_parameters.filename = (
                self.tmp_parameters.filename +
                '_elastic_scattering_threshold')
            threshold.tmp_parameters.folder = self.tmp_parameters.folder
            threshold.tmp_parameters.extension = \
                self.tmp_parameters.extension
        threshold.set_signal_type("")
        return threshold

    def estimate_thickness(self,
                           threshold,
                           zlp=None,):
        """Estimates the thickness (relative to the mean free path)
        of a sample using the log-ratio method.

        The current EELS spectrum must be a low-loss spectrum containing
        the zero-loss peak. The hyperspectrum must be well calibrated
        and aligned.

        Parameters
        ----------
        threshold : {Signal1D, float, int}
            Truncation energy to estimate the intensity of the
            elastic scattering. The threshold can be provided as a signal of
            the same dimension as the input spectrum navigation space
            containing the threshold value in the energy units. Alternatively a
            constant threshold can be specified in energy/index units by
            passing float/int.
        zlp : {None, EELSSpectrum}
            If not None the zero-loss peak intensity is calculated from the ZLP
            spectrum supplied by integration using Simpson's rule. If None
            estimates the zero-loss peak intensity using
            `estimate_elastic_scattering_intensity` by truncation.

        Returns
        -------
        s : Signal1D
            The thickness relative to the MFP. It returns a Signal1D,
            Signal2D or a BaseSignal, depending on the current navigation
            dimensions.

        Notes
        -----
        For details see: Egerton, R. Electron Energy-Loss
        Spectroscopy in the Electron Microscope. Springer-Verlag, 2011.

        """
        # TODO: Write units tests
        self._check_signal_dimension_equals_one()
        axis = self.axes_manager.signal_axes[0]
        total_intensity = self.integrate1D(axis.index_in_array).data
        if zlp is not None:
            I0 = zlp.integrate1D(axis.index_in_array).data
        else:
            I0 = self.estimate_elastic_scattering_intensity(
                threshold=threshold,).data
        if self._lazy:
            t_over_lambda = da.log(total_intensity / I0)
        else:
            t_over_lambda = np.log(total_intensity / I0)
        s = self._get_navigation_signal(data=t_over_lambda)
        s.metadata.General.title = (self.metadata.General.title +
                                    ' $\\frac{t}{\\lambda}$')
        if self.tmp_parameters.has_item('filename'):
            s.tmp_parameters.filename = (
                self.tmp_parameters.filename +
                '_relative_thickness')
            s.tmp_parameters.folder = self.tmp_parameters.folder
            s.tmp_parameters.extension = \
                self.tmp_parameters.extension
        s.axes_manager.set_signal_dimension(0)
        s.set_signal_type("")
        return s

    def fourier_log_deconvolution(self,
                                  zlp,
                                  add_zlp=False,
                                  crop=False):
        """Performs fourier-log deconvolution.

        Parameters
        ----------
        zlp : EELSSpectrum
            The corresponding zero-loss peak.

        add_zlp : bool
            If True, adds the ZLP to the deconvolved spectrum
        crop : bool
            If True crop the spectrum to leave out the channels that
            have been modified to decay smoothly to zero at the sides
            of the spectrum.

        Returns
        -------
        An EELSSpectrum containing the current data deconvolved.

        Notes
        -----
        For details see: Egerton, R. Electron Energy-Loss
        Spectroscopy in the Electron Microscope. Springer-Verlag, 2011.

        """
        self._check_signal_dimension_equals_one()
        s = self.deepcopy()
        zlp_size = zlp.axes_manager.signal_axes[0].size
        self_size = self.axes_manager.signal_axes[0].size
        tapped_channels = s.hanning_taper()
        # Conservative new size to solve the wrap-around problem
        size = zlp_size + self_size - 1
        # Increase to the closest power of two to enhance the FFT
        # performance
        size = closest_power_of_two(size)

        axis = self.axes_manager.signal_axes[0]
        if self._lazy or zlp._lazy:

            z = da.fft.rfft(zlp.data, n=size, axis=axis.index_in_array)
            j = da.fft.rfft(s.data, n=size, axis=axis.index_in_array)
            j1 = z * da.log(j / z).map_blocks(np.nan_to_num)
            sdata = da.fft.irfft(j1, axis=axis.index_in_array)
        else:
            z = np.fft.rfft(zlp.data, n=size, axis=axis.index_in_array)
            j = np.fft.rfft(s.data, n=size, axis=axis.index_in_array)
            j1 = z * np.nan_to_num(np.log(j / z))
            sdata = np.fft.irfft(j1, axis=axis.index_in_array)

        s.data = sdata[s.axes_manager._get_data_slice(
            [(axis.index_in_array, slice(None, self_size)), ])]
        if add_zlp is True:
            if self_size >= zlp_size:
                if self._lazy:
                    _slices_before = s.axes_manager._get_data_slice(
                        [(axis.index_in_array, slice(None, zlp_size)), ])
                    _slices_after = s.axes_manager._get_data_slice(
                        [(axis.index_in_array, slice(zlp_size, None)), ])
                    s.data = da.stack((s.data[_slices_before] + zlp.data,
                                       s.data[_slices_after]),
                                      axis=axis.index_in_array)
                else:
                    s.data[s.axes_manager._get_data_slice(
                        [(axis.index_in_array, slice(None, zlp_size)), ])
                    ] += zlp.data
            else:
                s.data += zlp.data[s.axes_manager._get_data_slice(
                    [(axis.index_in_array, slice(None, self_size)), ])]

        s.metadata.General.title = (s.metadata.General.title +
                                    ' after Fourier-log deconvolution')
        if s.tmp_parameters.has_item('filename'):
            s.tmp_parameters.filename = (
                self.tmp_parameters.filename +
                '_after_fourier_log_deconvolution')
        if crop is True:
            s.crop(axis.index_in_axes_manager,
                   None, int(-tapped_channels))
        return s

    def fourier_ratio_deconvolution(self, ll,
                                    fwhm=None,
                                    threshold=None,
                                    extrapolate_lowloss=True,
                                    extrapolate_coreloss=True):
        """Performs Fourier-ratio deconvolution.

        The core-loss should have the background removed. To reduce
         the noise amplication the result is convolved with a
        Gaussian function.

        Parameters
        ----------
        ll: EELSSpectrum
            The corresponding low-loss (ll) EELSSpectrum.

        fwhm : float or None
            Full-width half-maximum of the Gaussian function by which
            the result of the deconvolution is convolved. It can be
            used to select the final SNR and spectral resolution. If
            None, the FWHM of the zero-loss peak of the low-loss is
            estimated and used.
        threshold : {None, float}
            Truncation energy to estimate the intensity of the
            elastic scattering. If None the threshold is taken as the
             first minimum after the ZLP centre.
        extrapolate_lowloss, extrapolate_coreloss : bool
            If True the signals are extrapolated using a power law,

        Notes
        -----
        For details see: Egerton, R. Electron Energy-Loss
        Spectroscopy in the Electron Microscope. Springer-Verlag, 2011.

        """
        self._check_signal_dimension_equals_one()
        orig_cl_size = self.axes_manager.signal_axes[0].size

        if threshold is None:
            threshold = ll.estimate_elastic_scattering_threshold()

        if extrapolate_coreloss is True:
            cl = self.power_law_extrapolation(
                window_size=20,
                extrapolation_size=100)
        else:
            cl = self.deepcopy()

        if extrapolate_lowloss is True:
            ll = ll.power_law_extrapolation(
                window_size=100,
                extrapolation_size=100)
        else:
            ll = ll.deepcopy()

        ll.hanning_taper()
        cl.hanning_taper()
        if self._lazy or zlp._lazy:
            rfft = da.fft.rfft
            irfft = da.fft.irfft
        else:
            rfft = np.fft.rfft
            irfft = np.fft.irfft

        ll_size = ll.axes_manager.signal_axes[0].size
        cl_size = self.axes_manager.signal_axes[0].size
        # Conservative new size to solve the wrap-around problem
        size = ll_size + cl_size - 1
        # Increase to the closest multiple of two to enhance the FFT
        # performance
        size = int(2 ** np.ceil(np.log2(size)))

        axis = ll.axes_manager.signal_axes[0]
        if fwhm is None:
            fwhm = float(ll.get_current_signal().estimate_peak_width()())
            _logger.info("FWHM = %1.2f" % fwhm)

        I0 = ll.estimate_elastic_scattering_intensity(threshold=threshold)
        I0 = I0.data
        if ll.axes_manager.navigation_size > 0:
            I0_shape = list(I0.shape)
            I0_shape.insert(axis.index_in_array, 1)
            I0 = I0.reshape(I0_shape)

        from hyperspy.components1d import Gaussian
        g = Gaussian()
        g.sigma.value = fwhm / 2.3548
        g.A.value = 1
        g.centre.value = 0
        zl = g.function(
            np.linspace(axis.offset,
                        axis.offset + axis.scale * (size - 1),
                        size))
        z = np.fft.rfft(zl)
        jk = rfft(cl.data, n=size, axis=axis.index_in_array)
        jl = rfft(ll.data, n=size, axis=axis.index_in_array)
        zshape = [1, ] * len(cl.data.shape)
        zshape[axis.index_in_array] = jk.shape[axis.index_in_array]
        cl.data = irfft(z.reshape(zshape) * jk / jl,
                        axis=axis.index_in_array)
        cl.data *= I0
        cl.crop(-1, None, int(orig_cl_size))
        cl.metadata.General.title = (self.metadata.General.title +
                                     ' after Fourier-ratio deconvolution')
        if cl.tmp_parameters.has_item('filename'):
            cl.tmp_parameters.filename = (
                self.tmp_parameters.filename +
                'after_fourier_ratio_deconvolution')
        return cl

    def richardson_lucy_deconvolution(self, psf, iterations=15, mask=None,
                                      show_progressbar=None,
                                      parallel=None):
        """1D Richardson-Lucy Poissonian deconvolution of
        the spectrum by the given kernel.

        Parameters
        ----------
        iterations: int
            Number of iterations of the deconvolution. Note that
            increasing the value will increase the noise amplification.
        psf: EELSSpectrum
            It must have the same signal dimension as the current
            spectrum and a spatial dimension of 0 or the same as the
            current spectrum.
        show_progressbar : None or bool
            If True, display a progress bar. If None the default is set in
            `preferences`.
        parallel : {None,bool,int}
            if True, the deconvolution will be performed in a threaded (parallel)
            manner.

        Notes:
        -----
        For details on the algorithm see Gloter, A., A. Douiri,
        M. Tence, and C. Colliex. “Improving Energy Resolution of
        EELS Spectra: An Alternative to the Monochromator Solution.”
        Ultramicroscopy 96, no. 3–4 (September 2003): 385–400.

        """
        if show_progressbar is None:
            show_progressbar = preferences.General.show_progressbar
        self._check_signal_dimension_equals_one()
        psf_size = psf.axes_manager.signal_axes[0].size
        kernel = psf()
        imax = kernel.argmax()
        maxval = self.axes_manager.navigation_size
        show_progressbar = show_progressbar and (maxval > 0)

        def deconv_function(signal, kernel=None,
                            iterations=15, psf_size=None):
            imax = kernel.argmax()
            result = np.array(signal).copy()
            mimax = psf_size - 1 - imax
            for _ in range(iterations):
                first = np.convolve(kernel, result)[imax: imax + psf_size]
                result *= np.convolve(kernel[::-1], signal /
                                      first)[mimax:mimax + psf_size]
            return result
        ds = self.map(deconv_function, kernel=psf, iterations=iterations,
                      psf_size=psf_size, show_progressbar=show_progressbar,
                      parallel=parallel, ragged=False, inplace=False)

        ds.metadata.General.title += (
            ' after Richardson-Lucy deconvolution %i iterations' %
            iterations)
        if ds.tmp_parameters.has_item('filename'):
            ds.tmp_parameters.filename += (
                '_after_R-L_deconvolution_%iiter' % iterations)
        return ds

    def _are_microscope_parameters_missing(self):
        """Check if the EELS parameters necessary to calculate the GOS
        are defined in metadata. If not, in interactive mode
        raises an UI item to fill the values"""
        must_exist = (
            'Acquisition_instrument.TEM.convergence_angle',
            'Acquisition_instrument.TEM.beam_energy',
            'Acquisition_instrument.TEM.Detector.EELS.collection_angle',)
        missing_parameters = []
        for item in must_exist:
            exists = self.metadata.has_item(item)
            if exists is False:
                missing_parameters.append(item)
        if missing_parameters:
            if preferences.General.interactive is True:
                par_str = "The following parameters are missing:\n"
                for par in missing_parameters:
                    par_str += '%s\n' % par
                par_str += 'Please set them in the following wizard'
                is_ok = messagesui.information(par_str)
                if is_ok:
                    self._set_microscope_parameters()
                else:
                    return True
            else:
                return True
        else:
            return False

    def set_microscope_parameters(self,
                                  beam_energy=None,
                                  convergence_angle=None,
                                  collection_angle=None):
        """Set the microscope parameters that are necessary to calculate
        the GOS.

        If not all of them are defined, raises in interactive mode
        raises an UI item to fill the values

        beam_energy: float
            The energy of the electron beam in keV
        convengence_angle : float
            The microscope convergence semi-angle in mrad.
        collection_angle : float
            The collection semi-angle in mrad.
        """

        mp = self.metadata
        if beam_energy is not None:
            mp.set_item("Acquisition_instrument.TEM.beam_energy", beam_energy)
        if convergence_angle is not None:
            mp.set_item(
                "Acquisition_instrument.TEM.convergence_angle",
                convergence_angle)
        if collection_angle is not None:
            mp.set_item(
                "Acquisition_instrument.TEM.Detector.EELS.collection_angle",
                collection_angle)

        self._are_microscope_parameters_missing()

    @only_interactive
    def _set_microscope_parameters(self):
        tem_par = TEMParametersUI()
        mapping = {
            'Acquisition_instrument.TEM.convergence_angle':
            'tem_par.convergence_angle',
            'Acquisition_instrument.TEM.beam_energy':
            'tem_par.beam_energy',
            'Acquisition_instrument.TEM.Detector.EELS.collection_angle':
            'tem_par.collection_angle',
        }
        for key, value in mapping.items():
            if self.metadata.has_item(key):
                exec('%s = self.metadata.%s' % (value, key))
        tem_par.edit_traits()
        mapping = {
            'Acquisition_instrument.TEM.convergence_angle':
            tem_par.convergence_angle,
            'Acquisition_instrument.TEM.beam_energy':
            tem_par.beam_energy,
            'Acquisition_instrument.TEM.Detector.EELS.collection_angle':
            tem_par.collection_angle,
        }
        for key, value in mapping.items():
            if value != t.Undefined:
                self.metadata.set_item(key, value)
        self._are_microscope_parameters_missing()

    def power_law_extrapolation(self,
                                window_size=20,
                                extrapolation_size=1024,
                                add_noise=False,
                                fix_neg_r=False):
        """Extrapolate the spectrum to the right using a powerlaw


        Parameters
        ----------
        window_size : int
            The number of channels from the right side of the
            spectrum that are used to estimate the power law
            parameters.
        extrapolation_size : int
            Size of the extrapolation in number of channels
        add_noise : bool
            If True, add poissonian noise to the extrapolated spectrum.
        fix_neg_r : bool
            If True, the negative values for the "components.PowerLaw"
            parameter r will be flagged and the extrapolation will be
            done with a constant zero-value.

        Returns
        -------
        A new spectrum, with the extrapolation.

        """
        self._check_signal_dimension_equals_one()
        axis = self.axes_manager.signal_axes[0]
        s = self.deepcopy()
        s.metadata.General.title += (
            ' %i channels extrapolated' %
            extrapolation_size)
        if s.tmp_parameters.has_item('filename'):
            s.tmp_parameters.filename += (
                '_%i_channels_extrapolated' % extrapolation_size)
        new_shape = list(self.data.shape)
        new_shape[axis.index_in_array] += extrapolation_size
        if self._lazy:
            left_data = s.data
            right_shape = list(self.data.shape)
            right_shape[axis.index_in_array] = extrapolation_size
            right_chunks = list(self.data.chunks)
            right_chunks[axis.index_in_array] = (extrapolation_size, )
            right_data = da.zeros(
                shape=tuple(right_shape),
                chunks=tuple(right_chunks),
                dtype=self.data.dtype)
            s.data = da.concatenate(
                [left_data, right_data], axis=axis.index_in_array)
        else:
            # just old code
            s.data = np.zeros(new_shape)
            s.data[..., :axis.size] = self.data
        s.get_dimensions_from_data()
        pl = PowerLaw()
        pl._axes_manager = self.axes_manager
        A, r = pl.estimate_parameters(
            s,
            axis.index2value(axis.size - window_size),
            axis.index2value(axis.size - 1),
            out=True)
        if fix_neg_r is True:
            if s._lazy:
                _where = da.where
            else:
                _where = np.where
            A = _where(r <= 0, 0, A)
        # If the signal is binned we need to bin the extrapolated power law
        # what, in a first approximation, can be done by multiplying by the
        # axis step size.
        if self.metadata.Signal.binned is True:
            factor = s.axes_manager[-1].scale
        else:
            factor = 1
        if self._lazy:
            # only need new axes if the navigation dimension is not 0
            if s.axes_manager.navigation_dimension:
                rightslice = (..., None)
                axisslice = (None, slice(axis.size, None))
            else:
                rightslice = (..., )
                axisslice = (slice(axis.size, None), )
            right_chunks[axis.index_in_array] = 1
            x = da.from_array(
                s.axes_manager.signal_axes[0].axis[axisslice],
                chunks=(extrapolation_size, ))
            A = A[rightslice]
            r = r[rightslice]
            right_data = factor * A * x**(-r)
            s.data = da.concatenate(
                [left_data, right_data], axis=axis.index_in_array)
        else:
            s.data[..., axis.size:] = (
                factor * A[..., np.newaxis] *
                s.axes_manager.signal_axes[0].axis[np.newaxis, axis.size:]**(
                    -r[..., np.newaxis]))
        return s

    def kramers_kronig_analysis(self,
                                zlp=None,
                                iterations=1,
                                n=None,
                                t=None,
                                delta=0.5,
                                full_output=False):
        """Calculate the complex
        dielectric function from a single scattering distribution (SSD) using
        the Kramers-Kronig relations.

        It uses the FFT method as in [Egerton2011]_.  The SSD is an
        EELSSpectrum instance containing SSD low-loss EELS with no zero-loss
        peak. The internal loop is devised to approximately subtract the
        surface plasmon contribution supposing an unoxidized planar surface and
        neglecting coupling between the surfaces. This method does not account
        for retardation effects, instrumental broading and surface plasmon
        excitation in particles.

        Note that either refractive index or thickness are required.
        If both are None or if both are provided an exception is raised.

        Parameters
        ----------
        zlp: {None, number, Signal1D}
            ZLP intensity. It is optional (can be None) if `t` is None and `n`
            is not None and the thickness estimation is not required. If `t`
            is not None, the ZLP is required to perform the normalization and
            if `t` is not None, the ZLP is required to calculate the thickness.
            If the ZLP is the same for all spectra, the integral of the ZLP
            can be provided as a number. Otherwise, if the ZLP intensity is not
            the same for all spectra, it can be provided as i) a Signal1D
            of the same dimensions as the current signal containing the ZLP
            spectra for each location ii) a BaseSignal of signal dimension 0
            and navigation_dimension equal to the current signal containing the
            integrated ZLP intensity.
        iterations: int
            Number of the iterations for the internal loop to remove the
            surface plasmon contribution. If 1 the surface plasmon contribution
            is not estimated and subtracted (the default is 1).
        n: {None, float}
            The medium refractive index. Used for normalization of the
            SSD to obtain the energy loss function. If given the thickness
            is estimated and returned. It is only required when `t` is None.
        t: {None, number, Signal1D}
            The sample thickness in nm. Used for normalization of the
            SSD to obtain the energy loss function. It is only required when
            `n` is None. If the thickness is the same for all spectra it can be
            given by a number. Otherwise, it can be provided as a BaseSignal with
            signal dimension 0 and navigation_dimension equal to the current
            signal.
        delta : float
            A small number (0.1-0.5 eV) added to the energy axis in
            specific steps of the calculation the surface loss correction to
            improve stability.
        full_output : bool
            If True, return a dictionary that contains the estimated
            thickness if `t` is None and the estimated surface plasmon
            excitation and the spectrum corrected from surface plasmon
            excitations if `iterations` > 1.

        Returns
        -------
        eps: DielectricFunction instance
            The complex dielectric function results,
                $\epsilon = \epsilon_1 + i*\epsilon_2$,
            contained in an DielectricFunction instance.
        output: Dictionary (optional)
            A dictionary of optional outputs with the following keys:

            ``thickness``
                The estimated  thickness in nm calculated by normalization of
                the SSD (only when `t` is None)

            ``surface plasmon estimation``
               The estimated surface plasmon excitation (only if
               `iterations` > 1.)

        Raises
        ------
        ValuerError
            If both `n` and `t` are undefined (None).
        AttribureError
            If the beam_energy or the collection semi-angle are not defined in
            metadata.

        Notes
        -----
        This method is based in Egerton's Matlab code [Egerton2011]_ with some
        minor differences:

        * The integrals are performed using the simpsom rule instead of using
          a summation.
        * The wrap-around problem when computing the ffts is workarounded by
          padding the signal instead of substracting the reflected tail.

        .. [Egerton2011] Ray Egerton, "Electron Energy-Loss
           Spectroscopy in the Electron Microscope", Springer-Verlag, 2011.

        """
        output = {}
        if iterations == 1:
            # In this case s.data is not modified so there is no need to make
            # a deep copy.
            s = self.isig[0.:]
        else:
            s = self.isig[0.:].deepcopy()

        sorig = self.isig[0.:]
        # Avoid singularity at 0
        if s.axes_manager.signal_axes[0].axis[0] == 0:
            s = s.isig[1:]
            sorig = self.isig[1:]

        # Constants and units
        me = constants.value(
            'electron mass energy equivalent in MeV') * 1e3  # keV

        # Mapped parameters
        try:
            e0 = s.metadata.Acquisition_instrument.TEM.beam_energy
        except:
            raise AttributeError("Please define the beam energy."
                                 "You can do this e.g. by using the "
                                 "set_microscope_parameters method")
        try:
            beta = s.metadata.Acquisition_instrument.TEM.Detector.\
                EELS.collection_angle
        except:
            raise AttributeError("Please define the collection semi-angle. "
                                 "You can do this e.g. by using the "
                                 "set_microscope_parameters method")

        axis = s.axes_manager.signal_axes[0]
        eaxis = axis.axis.copy()

        if isinstance(zlp, hyperspy.signal.BaseSignal):
            if (zlp.axes_manager.navigation_dimension ==
                    self.axes_manager.navigation_dimension):
                if zlp.axes_manager.signal_dimension == 0:
                    i0 = zlp.data
                else:
                    i0 = zlp.integrate1D(axis.index_in_axes_manager).data
            else:
                raise ValueError('The ZLP signal dimensions are not '
                                 'compatible with the dimensions of the '
                                 'low-loss signal')
            i0 = i0.reshape(
                np.insert(i0.shape, axis.index_in_array, 1))
        elif isinstance(zlp, numbers.Number):
            i0 = zlp
        else:
            raise ValueError('The zero-loss peak input is not valid, it must be\
                             in the BaseSignal class or a Number.')

        if isinstance(t, hyperspy.signal.BaseSignal):
            if (t.axes_manager.navigation_dimension ==
                    self.axes_manager.navigation_dimension) and (
                    t.axes_manager.signal_dimension == 0):
                t = t.data
                t = t.reshape(
                    np.insert(t.shape, axis.index_in_array, 1))
            else:
                raise ValueError('The thickness signal dimensions are not '
                                 'compatible with the dimensions of the '
                                 'low-loss signal')
        elif isinstance(t, np.ndarray) and t.shape and t.shape != (1,):
            raise ValueError("thickness must be a HyperSpy signal or a number,"
                             " not a numpy array.")

        # Slicer to get the signal data from 0 to axis.size
        slicer = s.axes_manager._get_data_slice(
            [(axis.index_in_array, slice(None, axis.size)), ])

        # Kinetic definitions
        ke = e0 * (1 + e0 / 2. / me) / (1 + e0 / me) ** 2
        tgt = e0 * (2 * me + e0) / (me + e0)
        rk0 = 2590 * (1 + e0 / me) * np.sqrt(2 * ke / me)

        for io in range(iterations):
            # Calculation of the ELF by normalization of the SSD
            # Norm(SSD) = Imag(-1/epsilon) (Energy Loss Funtion, ELF)

            # We start by the "angular corrections"
            Im = s.data / (np.log(1 + (beta * tgt / eaxis) ** 2)) / axis.scale
            if n is None and t is None:
                raise ValueError("The thickness and the refractive index are "
                                 "not defined. Please provide one of them.")
            elif n is not None and t is not None:
                raise ValueError("Please provide the refractive index OR the "
                                 "thickness information, not both")
            elif n is not None:
                # normalize using the refractive index.
                K = (Im / eaxis).sum(axis=axis.index_in_array) * axis.scale
                K = (K / (np.pi / 2) / (1 - 1. / n ** 2)).reshape(
                    np.insert(K.shape, axis.index_in_array, 1))
                # Calculate the thickness only if possible and required
                if zlp is not None and (full_output is True or
                                        iterations > 1):
                    te = (332.5 * K * ke / i0)
                    if full_output is True:
                        output['thickness'] = te
            elif t is not None:
                if zlp is None:
                    raise ValueError("The ZLP must be provided when the  "
                                     "thickness is used for normalization.")
                # normalize using the thickness
                K = t * i0 / (332.5 * ke)
                te = t
            Im = Im / K

            # Kramers Kronig Transform:
            # We calculate KKT(Im(-1/epsilon))=1+Re(1/epsilon) with FFT
            # Follows: D W Johnson 1975 J. Phys. A: Math. Gen. 8 490
            # Use a size that is a power of two to speed up the fft and
            # make it double the closest upper value to workaround the
            # wrap-around problem.
            esize = 2 * closest_power_of_two(axis.size)
            q = -2 * np.fft.fft(Im, esize,
                                axis.index_in_array).imag / esize

            q[slicer] *= -1
            q = np.fft.fft(q, axis=axis.index_in_array)
            # Final touch, we have Re(1/eps)
            Re = q[slicer].real + 1

            # Egerton does this to correct the wrap-around problem, but in our
            # case this is not necessary because we compute the fft on an
            # extended and padded spectrum to avoid this problem.
            # Re=real(q)
            # Tail correction
            # vm=Re[axis.size-1]
            # Re[:(axis.size-1)]=Re[:(axis.size-1)]+1-(0.5*vm*((axis.size-1) /
            #  (axis.size*2-arange(0,axis.size-1)))**2)
            # Re[axis.size:]=1+(0.5*vm*((axis.size-1) /
            #  (axis.size+arange(0,axis.size)))**2)

            # Epsilon appears:
            #  We calculate the real and imaginary parts of the CDF
            e1 = Re / (Re ** 2 + Im ** 2)
            e2 = Im / (Re ** 2 + Im ** 2)

            if iterations > 1 and zlp is not None:
                # Surface losses correction:
                #  Calculates the surface ELF from a vaccumm border effect
                #  A simulated surface plasmon is subtracted from the ELF
                Srfelf = 4 * e2 / ((e1 + 1) ** 2 + e2 ** 2) - Im
                adep = (tgt / (eaxis + delta) *
                        np.arctan(beta * tgt / axis.axis) -
                        beta / 1000. /
                        (beta ** 2 + axis.axis ** 2. / tgt ** 2))
                Srfint = 2000 * K * adep * Srfelf / rk0 / te * axis.scale
                s.data = sorig.data - Srfint
                _logger.debug('Iteration number: %d / %d', io + 1, iterations)
                if iterations == io + 1 and full_output is True:
                    sp = sorig._deepcopy_with_new_data(Srfint)
                    sp.metadata.General.title += (
                        " estimated surface plasmon excitation.")
                    output['surface plasmon estimation'] = sp
                    del sp
                del Srfint

        eps = s._deepcopy_with_new_data(e1 + e2 * 1j)
        del s
        eps.set_signal_type("DielectricFunction")
        eps.metadata.General.title = (self.metadata.General.title +
                                      'dielectric function '
                                      '(from Kramers-Kronig analysis)')
        if eps.tmp_parameters.has_item('filename'):
            eps.tmp_parameters.filename = (
                self.tmp_parameters.filename +
                '_CDF_after_Kramers_Kronig_transform')
        if 'thickness' in output:
            thickness = eps._get_navigation_signal(
                data=te[self.axes_manager._get_data_slice(
                    [(axis.index_in_array, 0)])])
            thickness.metadata.General.title = (
                self.metadata.General.title + ' thickness '
                '(calculated using Kramers-Kronig analysis)')
            output['thickness'] = thickness
        if full_output is False:
            return eps
        else:
            return eps, output

    def create_model(self, ll=None, auto_background=True, auto_add_edges=True,
                     GOS=None, dictionary=None):
        """Create a model for the current EELS data.

        Parameters
        ----------
        ll : EELSSpectrum, optional
            If an EELSSpectrum is provided, it will be assumed that it is
            a low-loss EELS spectrum, and it will be used to simulate the
            effect of multiple scattering by convolving it with the EELS
            spectrum.
        auto_background : boolean, default True
            If True, and if spectrum is an EELS instance adds automatically
            a powerlaw to the model and estimate the parameters by the
            two-area method.
        auto_add_edges : boolean, default True
            If True, and if spectrum is an EELS instance, it will
            automatically add the ionization edges as defined in the
            Signal1D instance. Adding a new element to the spectrum using
            the components.EELSSpectrum.add_elements method automatically
            add the corresponding ionisation edges to the model.
        GOS : {'hydrogenic' | 'Hartree-Slater'}, optional
            The generalized oscillation strenght calculations to use for the
            core-loss EELS edges. If None the Hartree-Slater GOS are used if
            available, otherwise it uses the hydrogenic GOS.
        dictionary : {None | dict}, optional
            A dictionary to be used to recreate a model. Usually generated
            using :meth:`hyperspy.model.as_dictionary`

        Returns
        -------

        model : `EELSModel` instance.

        """
        from hyperspy.models.eelsmodel import EELSModel
        model = EELSModel(self,
                          ll=ll,
                          auto_background=auto_background,
                          auto_add_edges=auto_add_edges,
                          GOS=GOS,
                          dictionary=dictionary)
        return model


class LazyEELSSpectrum(EELSSpectrum_mixin, LazySignal1D):

    pass


class EELSSpectrum(EELSSpectrum_mixin, Signal1D):

    pass
