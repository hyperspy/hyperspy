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

import numbers
import logging

import numpy as np
import dask.array as da
import traits.api as t
from scipy import constants
from prettytable import PrettyTable

from hyperspy.signal import BaseSetMetadataItems, BaseSignal
from hyperspy._signals.signal1d import (Signal1D, LazySignal1D)
from hyperspy.signal_tools import EdgesRange
from hyperspy.misc.elements import elements as elements_db
from hyperspy.misc.label_position import SpectrumLabelPosition
import hyperspy.axes
from hyperspy.defaults_parser import preferences
from hyperspy.components1d import PowerLaw
from hyperspy.misc.utils import isiterable, underline, print_html
from hyperspy.misc.utils import is_binned # remove in v2.0
from hyperspy.misc.math_tools import optimal_fft_size
from hyperspy.misc.eels.tools import get_edges_near_energy
from hyperspy.misc.eels.electron_inelastic_mean_free_path import iMFP_Iakoubovskii, iMFP_angular_correction
from hyperspy.ui_registry import add_gui_method, DISPLAY_DT, TOOLKIT_DT
from hyperspy.docstrings.signal1d import (
    CROP_PARAMETER_DOC, SPIKES_DIAGNOSIS_DOCSTRING, MASK_ZERO_LOSS_PEAK_WIDTH,
    SPIKES_REMOVAL_TOOL_DOCSTRING)
from hyperspy.docstrings.signal import (
    SHOW_PROGRESSBAR_ARG, PARALLEL_ARG, MAX_WORKERS_ARG, SIGNAL_MASK_ARG,
    NAVIGATION_MASK_ARG)



_logger = logging.getLogger(__name__)


@add_gui_method(toolkey="hyperspy.microscope_parameters_EELS")
class EELSTEMParametersUI(BaseSetMetadataItems):
    convergence_angle = t.Float(t.Undefined,
                                label='Convergence semi-angle (mrad)')
    beam_energy = t.Float(t.Undefined,
                          label='Beam energy (keV)')
    collection_angle = t.Float(t.Undefined,
                               label='Collection semi-angle (mrad)')
    mapping = {
        'Acquisition_instrument.TEM.convergence_angle':
        'convergence_angle',
        'Acquisition_instrument.TEM.beam_energy':
        'beam_energy',
        'Acquisition_instrument.TEM.Detector.EELS.collection_angle':
        'collection_angle',
    }


class EELSSpectrum(Signal1D):

    """1D signal class for EELS spectra."""

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
        self.axes_manager.signal_axes[0].is_binned = True
        self._edge_markers = {}

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
            energy limit of the SI will be included

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
            energy limit of the SI will be included

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

    def edges_at_energy(self, energy='interactive', width=10, only_major=False,
                        order='closest', display=True, toolkit=None):
        """Show EELS edges according to an energy range selected from the
        spectrum or within a provided energy window

        Parameters
        ----------
        energy : 'interactive' or float
            If it is 'interactive', a table with edges are shown and it depends
            on the energy range selected in the spectrum. If it is a float, a
            table with edges are shown and it depends on the energy window
            defined by energy +/- (width/2). The default is 'interactive'.
        width : float
            Width of window, in eV, around energy in which to find nearby
            energies, i.e. a value of 10 eV (the default) means to
            search +/- 5 eV. The default is 10.
        only_major : bool
            Whether to show only the major edges. The default is False.
        order : str
            Sort the edges, if 'closest', return in the order of energy
            difference, if 'ascending', return in ascending order, similarly
            for 'descending'. The default is 'closest'.

        Returns
        -------
        An interactive widget if energy is 'interactive', or a html-format
        table or ASCII table, depends on the environment.
        """

        if energy == 'interactive':
            er = EdgesRange(self)
            return er.gui(display=display, toolkit=toolkit)
        else:
            return self.print_edges_near_energy(energy, width, only_major,
                                                order)

    @staticmethod
    def print_edges_near_energy(energy=None, width=10, only_major=False,
                                order='closest', edges=None):
        """Find and print a table of edges near a given energy that are within
        the given energy window.

        Parameters
        ----------
        energy : float
            Energy to search, in eV
        width : float
            Width of window, in eV, around energy in which to find nearby
            energies, i.e. a value of 10 eV (the default) means to
            search +/- 5 eV. The default is 10.
        only_major : bool
            Whether to show only the major edges. The default is False.
        order : str
            Sort the edges, if 'closest', return in the order of energy
            difference, if 'ascending', return in ascending order, similarly
            for 'descending'. The default is 'closest'.
        edges : iterable
            A sequence of edges, if provided, it overrides energy, width,
            only_major and order.

        Returns
        -------
        A PrettyText object where its representation is ASCII in terminal and
        html-formatted in Jupyter notebook
        """

        if edges is None and energy is not None:
            edges = get_edges_near_energy(energy=energy, width=width,
                                          only_major=only_major, order=order)
        elif edges is None and energy is None:
            raise ValueError('Either energy or edges should be provided.')

        table = PrettyTable()
        table.field_names = [
        'edge',
        'onset energy (eV)',
        'relevance',
        'description']

        for edge in edges:
            element, shell = edge.split('_')
            shell_dict = elements_db[element]['Atomic_properties'][
                         'Binding_energies'][shell]

            onset = shell_dict['onset_energy (eV)']
            relevance = shell_dict['relevance']
            threshold = shell_dict['threshold']
            edge_ = shell_dict['edge']
            description = threshold + '. '*(threshold !='' and edge_ !='') + edge_

            table.add_row([edge, onset, relevance, description])

        # this ensures the html version try its best to mimick the ASCII one
        table.format = True

        return print_html(f_text=table.get_string,
                          f_html=table.get_html_string)

    def estimate_zero_loss_peak_centre(self, mask=None):
        """Estimate the position of the zero-loss peak.

        This function provides just a coarse estimation of the position
        of the zero-loss peak centre by computing the position of the maximum
        of the spectra. For subpixel accuracy use `estimate_shift1D`.

        Parameters
        ----------
        mask : Signal1D of bool data type or bool array
            It must have signal_dimension = 0 and navigation_shape equal to the
            navigation shape of the current signal. Where mask is True the
            shift is not computed and set to nan.

        Returns
        -------
        zlpc : Signal1D subclass
            The estimated position of the maximum of the ZLP peak.

        Notes
        -----
        This function only works when the zero-loss peak is the most
        intense feature in the spectrum. If it is not in most cases
        the spectrum can be cropped to meet this criterion.
        Alternatively use `estimate_shift1D`.

        See Also
        --------
        estimate_shift1D, align_zero_loss_peak

        """
        self._check_signal_dimension_equals_one()
        self._check_navigation_mask(mask)
        if isinstance(mask, BaseSignal):
            mask = mask.data
        zlpc = self.valuemax(-1)
        if mask is not None:
            zlpc.data = np.where(mask, np.nan, zlpc.data)
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
            crop=True,
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
            the alignment.
        subpixel : bool
            If True, perform the alignment with subpixel accuracy
            using cross-correlation.
        mask : Signal1D of bool data type or bool array.
            It must have signal_dimension = 0 and navigation_shape equal to
            the shape of the current signal. Where mask is True the shift is
            not computed and set to nan.
        signal_range : tuple of integers, tuple of floats. Optional
            Will only search for the ZLP within the signal_range. If given
            in integers, the range will be in index values. If given floats,
            the range will be in spectrum values. Useful if there are features
            in the spectrum which are more intense than the ZLP.
            Default is searching in the whole signal. Note that ROIs can be used
            in place of a tuple.
        %s
        %s

        Raises
        ------
        NotImplementedError
            If the signal axis is a non-uniform axis.

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
            # Test that axes is uniform
            if not self.axes_manager[-1].is_uniform:
                raise NotImplementedError("Support for EELS signals with "
                            "non-uniform signal axes is not yet implemented.")
            if isinstance(value, da.Array):
                value = value.compute()
            for signal in signals:
                signal.axes_manager[-1].offset -= value
                signal.events.data_changed.trigger(signal)

        def estimate_zero_loss_peak_centre(s, mask, signal_range):
            if signal_range:
                zlpc = s.isig[signal_range[0]:signal_range[1]].\
                    estimate_zero_loss_peak_centre(mask=mask)
            else:
                zlpc = s.estimate_zero_loss_peak_centre(mask=mask)
            return zlpc

        zlpc = estimate_zero_loss_peak_centre(
            self, mask=mask, signal_range=signal_range)

        mean_ = np.nanmean(zlpc.data)

        if print_stats is True:
            print(underline("Initial ZLP position statistics"))
            zlpc.print_summary_statistics()

        for signal in also_align + [self]:
            shift_array = -zlpc.data + mean_
            if zlpc._lazy:
                # We must compute right now because otherwise any changes to the
                # axes_manager of the signal later in the workflow may result in
                # a wrong shift_array
                shift_array = shift_array.compute()
            signal.shift1D(
                shift_array, crop=crop, show_progressbar=show_progressbar)

        if calibrate is True:
            zlpc = estimate_zero_loss_peak_centre(
                self, mask=mask, signal_range=signal_range)
            substract_from_offset(np.nanmean(zlpc.data),
                                  also_align + [self])

        if subpixel is False:
            return
        left, right = -3., 3.
        if calibrate is False:
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
                mask=mask,
                crop=crop,
                **kwargs)
        if calibrate is True:
            zlpc = estimate_zero_loss_peak_centre(
                self, mask=mask, signal_range=signal_range)
            substract_from_offset(np.nanmean(zlpc.data),
                                  also_align + [self])
    align_zero_loss_peak.__doc__ %= (SHOW_PROGRESSBAR_ARG, CROP_PARAMETER_DOC)

    def get_zero_loss_peak_mask(self, zero_loss_peak_mask_width=5.0,
                                signal_mask=None):
        """Return boolean array with True value at the position of the zero
        loss peak. This mask can be used to restrict operation to the signal
        locations not marked as True (masked).

        Parameters
        ----------
        zero_loss_peak_mask_width: float
            Width of the zero loss peak mask.
        %s

        Returns
        -------
        bool array
        """
        zlpc = self.estimate_zero_loss_peak_centre()
        (signal_axis, ) = self.axes_manager[self.axes_manager.signal_axes]
        axis = signal_axis.axis
        mini_value = zlpc.data.mean() - zero_loss_peak_mask_width / 2
        maxi_value = zlpc.data.mean() + zero_loss_peak_mask_width / 2
        mask = np.logical_and(mini_value <= axis, axis <= maxi_value)
        if signal_mask is not None:
            signal_mask = np.logical_or(mask, signal_mask)
        else:
            signal_mask = mask
        return signal_mask

    get_zero_loss_peak_mask.__doc__ %= (SIGNAL_MASK_ARG)

    def spikes_diagnosis(self, signal_mask=None, navigation_mask=None,
                         zero_loss_peak_mask_width=None, **kwargs):
        if zero_loss_peak_mask_width is not None:
            signal_mask = self.get_zero_loss_peak_mask(zero_loss_peak_mask_width,
                                                       signal_mask)
        super().spikes_diagnosis(signal_mask=signal_mask, navigation_mask=None,
                                 **kwargs)

    spikes_diagnosis.__doc__ = SPIKES_DIAGNOSIS_DOCSTRING % MASK_ZERO_LOSS_PEAK_WIDTH

    def spikes_removal_tool(self, signal_mask=None,
                            navigation_mask=None,
                            threshold='auto',
                            zero_loss_peak_mask_width=None,
                            interactive=True,
                            display=True,
                            toolkit=None):
        if zero_loss_peak_mask_width is not None:
            axis = self.axes_manager.signal_axes[0].axis
            # check the zero_loss is in the signal
            if (axis[0] - zero_loss_peak_mask_width / 2 > 0 or
                axis[-1] + zero_loss_peak_mask_width / 2 < 0):
                raise ValueError("The zero loss peaks isn't in the energy range.")
            signal_mask = self.get_zero_loss_peak_mask(zero_loss_peak_mask_width,
                                                       signal_mask)
        super().spikes_removal_tool(signal_mask=signal_mask,
                                    navigation_mask=navigation_mask,
                                    threshold=threshold,
                                    interactive=interactive,
                                    display=display, toolkit=toolkit)
    spikes_removal_tool.__doc__ = SPIKES_REMOVAL_TOOL_DOCSTRING % (
        SIGNAL_MASK_ARG, NAVIGATION_MASK_ARG, MASK_ZERO_LOSS_PEAK_WIDTH, DISPLAY_DT, TOOLKIT_DT,)

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
        %s

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
            # I0 = I0.transpose(signal_axes=[])
            threshold = threshold.transpose(signal_axes=[])
            binned = ax.is_binned

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
    estimate_elastic_scattering_intensity.__doc__ %= SHOW_PROGRESSBAR_ARG

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
            the noise. It must be an odd integer.
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
            s = s.derivative(-1)
        if tol is None:
            tol = np.max(abs(s.data).min(axis.index_in_array))
        saxis = s.axes_manager[-1]
        inflexion = (abs(s.data) <= tol).argmax(saxis.index_in_array)
        if isinstance(inflexion, da.Array):
            inflexion = inflexion.compute()
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
                           threshold=None,
                           zlp=None,
                           density=None,
                           mean_free_path=None,):
        """Estimates the thickness (relative and absolute)
        of a sample using the log-ratio method.

        The current EELS spectrum must be a low-loss spectrum containing
        the zero-loss peak. The hyperspectrum must be well calibrated
        and aligned. To obtain the thickness relative to the mean free path
        don't set the `density` and the `mean_free_path`.

        Parameters
        ----------
        threshold : {BaseSignal, float}, optional
            If the zero-loss-peak is not provided, use this energy threshold
            to roughly estimate its intensity by truncation.
            If the threshold is constant across the dataset use a float. Otherwise,
            provide a signal of
            the same dimension as the input spectrum navigation space
            containing the threshold value in the energy units.
        zlp : BaseSignal, optional
            If not None the zero-loss peak intensity is calculated from the ZLP
            spectrum supplied by integration.
        mean_free_path : float, optional
            The mean free path of the material in nanometers.
            If not provided, the thickness
            is given relative to the mean free path.
        density : float, optional
            The density of the material in g/cm**3. This is used to estimate the mean
            free path when the mean free path is not known and to perform the
            angular corrections.

        Returns
        -------
        s : BaseSignal
            The thickness relative to the MFP. It returns a Signal1D,
            Signal2D or a BaseSignal, depending on the current navigation
            dimensions.

        Notes
        -----
        For details see Egerton, R. Electron Energy-Loss Spectroscopy in the Electron
        Microscope.  Springer-Verlag, 2011.
        """
        axis = self.axes_manager.signal_axes[0]
        total_intensity = self.integrate1D(axis.index_in_array).data
        if threshold is None and zlp is None:
            raise ValueError("Please provide one of the following keywords: "
                             "`threshold`, `zlp`")
        if zlp is not None:
            I0 = zlp.integrate1D(axis.index_in_array).data
        else:
            I0 = self.estimate_elastic_scattering_intensity(
                threshold=threshold,).data
        
        t_over_lambda = np.log(total_intensity / I0)
        
        if density is not None:
            if self._are_microscope_parameters_missing():
                raise RuntimeError(
                    "Some microscope parameters are missing. Please use the "
                    "`set_microscope_parameters()` method to set them. "
                    "If you don't know them, don't set the `density` keyword."
                )
            else:
                md = self.metadata.Acquisition_instrument.TEM
                t_over_lambda *= iMFP_angular_correction(
                    beam_energy=md.beam_energy,
                    alpha=md.convergence_angle,
                    beta=md.Detector.EELS.collection_angle,
                    density=density,
                )
                if mean_free_path is None:
                    mean_free_path = iMFP_Iakoubovskii(
                        electron_energy=self.metadata.Acquisition_instrument.TEM.beam_energy,
                        density=density)
                    _logger.info(f"The estimated iMFP is {mean_free_path} nm")
        else:
            _logger.warning(
                "Computing the thickness without taking into account the effect of "
                "the limited collection angle, what usually leads to underestimating "
                "the thickness. To perform the angular corrections you must provide "
                "the density of the material.")

        s = self._get_navigation_signal(data=t_over_lambda)
        if mean_free_path is not None:
            s.data *= mean_free_path
            s.metadata.General.title = (
                self.metadata.General.title +
                ' thickness (nm)')
            s.metadata.Signal.quantity = "thickness (nm)"
        else:
            _logger.warning(
                "Computing the relative thickness. To compute the absolute "
                "thickness provide the `mean_free_path` and/or the `density`")
            s.metadata.General.title = (self.metadata.General.title +
                                        ' $\\frac{t}{\\lambda}$')
            s.metadata.Signal.quantity = "$\\frac{t}{\\lambda}$"
        if self.tmp_parameters.has_item('filename'):
            s.tmp_parameters.filename = (
                self.tmp_parameters.filename +
                '_thickness')
            s.tmp_parameters.folder = self.tmp_parameters.folder
            s.tmp_parameters.extension = \
                self.tmp_parameters.extension
        s = s.transpose(signal_axes=[])
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

        Raises
        ------
        NotImplementedError
            If the signal axis is a non-uniform axis.

        Notes
        -----
        For details see: Egerton, R. Electron Energy-Loss
        Spectroscopy in the Electron Microscope. Springer-Verlag, 2011.

        """
        self._check_signal_dimension_equals_one()
        if not self.axes_manager.signal_axes[0].is_uniform:
            raise NotImplementedError(
                "This operation is not yet implemented for non-uniform energy axes")
        s = self.deepcopy()
        zlp_size = zlp.axes_manager.signal_axes[0].size
        self_size = self.axes_manager.signal_axes[0].size
        tapped_channels = s.hanning_taper()
        # Conservative new size to solve the wrap-around problem
        size = zlp_size + self_size - 1
        # Calculate optimal FFT padding for performance
        complex_result = (zlp.data.dtype.kind == 'c' or s.data.dtype.kind == 'c')
        size = optimal_fft_size(size, not complex_result)

        axis = self.axes_manager.signal_axes[0]

        z = np.fft.rfft(zlp.data, n=size, axis=axis.index_in_array)
        j = np.fft.rfft(s.data, n=size, axis=axis.index_in_array)
        if self._lazy or zlp._lazy:
            j1 = z * da.log(j / z).map_blocks(np.nan_to_num)
        else:

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

        The core-loss should have the background removed. To reduce the noise
        amplification the result is convolved with a Gaussian function.

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

        Raises
        ------
        NotImplementedError
            If the signal axis is a non-uniform axis.

        Notes
        -----
        For details see: Egerton, R. Electron Energy-Loss
        Spectroscopy in the Electron Microscope. Springer-Verlag, 2011.

        """
        self._check_signal_dimension_equals_one()
        if not self.axes_manager.signal_axes[0].is_uniform:
            raise NotImplementedError(
                "This operation is not yet implemented for non-uniform energy axes.")
        if not ll.axes_manager.signal_axes[0].is_uniform:
            raise NotImplementedError(
                "The low-loss energy axis is non-uniform. "
                "This operation is not yet implemented for non-uniform energy axes")
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

        ll_size = ll.axes_manager.signal_axes[0].size
        cl_size = self.axes_manager.signal_axes[0].size
        # Conservative new size to solve the wrap-around problem
        size = ll_size + cl_size - 1
        # Calculate the optimal FFT size
        size = optimal_fft_size(size)

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
        jk = np.fft.rfft(cl.data, n=size, axis=axis.index_in_array)
        jl = np.fft.rfft(ll.data, n=size, axis=axis.index_in_array)
        zshape = [1, ] * len(cl.data.shape)
        zshape[axis.index_in_array] = jk.shape[axis.index_in_array]
        cl.data = np.fft.irfft(z.reshape(zshape) * jk / jl,
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

    def richardson_lucy_deconvolution(self, psf, iterations=15,
                                      show_progressbar=None,
                                      parallel=None, max_workers=None):
        """1D Richardson-Lucy Poissonian deconvolution of
        the spectrum by the given kernel.

        Parameters
        ----------
        psf : EELSSpectrum
            It must have the same signal dimension as the current
            spectrum and a spatial dimension of 0 or the same as the
            current spectrum.
        iterations : int
            Number of iterations of the deconvolution. Note that
            increasing the value will increase the noise amplification.
        %s
        %s
        %s

        Raises
        ------
        NotImplementedError
            If the signal axis is a non-uniform axis.

        Notes
        -----
        For details on the algorithm see Gloter, A., A. Douiri,
        M. Tence, and C. Colliex. “Improving Energy Resolution of
        EELS Spectra: An Alternative to the Monochromator Solution.”
        Ultramicroscopy 96, no. 3–4 (September 2003): 385–400.

        """
        if not self.axes_manager.signal_axes[0].is_uniform:
            raise NotImplementedError(
                "This operation is not yet implemented for non-uniform energy axes.")
        if show_progressbar is None:
            show_progressbar = preferences.General.show_progressbar
        self._check_signal_dimension_equals_one()
        psf_size = psf.axes_manager.signal_axes[0].size
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
                      parallel=parallel, max_workers=max_workers,
                      ragged=False, inplace=False)

        ds.metadata.General.title += (
            ' after Richardson-Lucy deconvolution %i iterations' %
            iterations)
        if ds.tmp_parameters.has_item('filename'):
            ds.tmp_parameters.filename += (
                '_after_R-L_deconvolution_%iiter' % iterations)
        return ds

    richardson_lucy_deconvolution.__doc__ %= (SHOW_PROGRESSBAR_ARG, PARALLEL_ARG, MAX_WORKERS_ARG)

    def _are_microscope_parameters_missing(self, ignore_parameters=[]):
        """
        Check if the EELS parameters necessary to calculate the GOS
        are defined in metadata. If not, in interactive mode
        raises an UI item to fill the values.
        The `ignore_parameters` list can be to ignore parameters.
        """
        must_exist = (
            'Acquisition_instrument.TEM.convergence_angle',
            'Acquisition_instrument.TEM.beam_energy',
            'Acquisition_instrument.TEM.Detector.EELS.collection_angle',)
        missing_parameters = []
        for item in must_exist:
            exists = self.metadata.has_item(item)
            if exists is False and item.split(
                    '.')[-1] not in ignore_parameters:
                missing_parameters.append(item)
        if missing_parameters:
            _logger.info("Missing parameters {}".format(missing_parameters))
            return True
        else:
            return False

    def set_microscope_parameters(self,
                                  beam_energy=None,
                                  convergence_angle=None,
                                  collection_angle=None,
                                  toolkit=None,
                                  display=True):
        if set((beam_energy, convergence_angle, collection_angle)) == {None}:
            tem_par = EELSTEMParametersUI(self)
            return tem_par.gui(toolkit=toolkit, display=display)
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
    set_microscope_parameters.__doc__ = \
        """
        Set the microscope parameters that are necessary to calculate
        the GOS.

        If not all of them are defined, in interactive mode
        raises an UI item to fill the values.

        beam_energy: float
            The energy of the electron beam in keV.
        convergence_angle : float
            The microscope convergence semi-angle in mrad.
        collection_angle : float
            The collection semi-angle in mrad.
        {}
        {}
        """.format(TOOLKIT_DT, DISPLAY_DT)

    def power_law_extrapolation(self,
                                window_size=20,
                                extrapolation_size=1024,
                                add_noise=False,
                                fix_neg_r=False):
        """Extrapolate the spectrum to the right using a powerlaw.


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
            A = np.where(r <= 0, 0, A)
        # If the signal is binned we need to bin the extrapolated power law
        # what, in a first approximation, can be done by multiplying by the
        # axis step size.
        if is_binned(self):
        # in v2 replace by
        # if self.axes_manager[-1].is_binned:
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
        r"""Calculate the complex dielectric function from a single scattering
        distribution (SSD) using the Kramers-Kronig relations.

        It uses the FFT method as in [1]_.  The SSD is an
        EELSSpectrum instance containing SSD low-loss EELS with no zero-loss
        peak. The internal loop is devised to approximately subtract the
        surface plasmon contribution supposing an unoxidized planar surface and
        neglecting coupling between the surfaces. This method does not account
        for retardation effects, instrumental broadening and surface plasmon
        excitation in particles.

        Note that either refractive index or thickness are required.
        If both are None or if both are provided an exception is raised.

        Parameters
        ----------
        zlp : {None, number, Signal1D}
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
        iterations : int
            Number of the iterations for the internal loop to remove the
            surface plasmon contribution. If 1 the surface plasmon contribution
            is not estimated and subtracted (the default is 1).
        n : {None, float}
            The medium refractive index. Used for normalization of the
            SSD to obtain the energy loss function. If given the thickness
            is estimated and returned. It is only required when `t` is None.
        t : {None, number, Signal1D}
            The sample thickness in nm. Used for normalization of the SSD
            to obtain the energy loss function. It is only required when
            `n` is None. If the thickness is the same for all spectra it can be
            given by a number. Otherwise, it can be provided as a BaseSignal
            with signal dimension 0 and navigation_dimension equal to the
            current signal.
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

                .. math::
                    \epsilon = \epsilon_1 + i*\epsilon_2,

            contained in an DielectricFunction instance.
        output: Dictionary (optional)
            A dictionary of optional outputs with the following keys

            * ``thickness``: the estimated  thickness in nm calculated by
              normalization of the SSD (only when ``t`` is None)
            * ``surface plasmon estimation``: the estimated surface plasmon
              excitation (only if ``iterations`` > 1.)

        Raises
        ------
        ValueError
            If both `n` and `t` are undefined (None).
        AttributeError
            If the beam_energy or the collection semi-angle are not defined in
            metadata.
        NotImplementedError
            If the signal axis is a non-uniform axis.

        Notes
        -----
        This method is based in Egerton's Matlab code [1]_ with a
        minor difference: the wrap-around problem when computing the FFTs is
        workarounded by padding the signal instead of subtracting the
        reflected tail.

        .. [1] Ray Egerton, "Electron Energy-Loss Spectroscopy in the Electron
           Microscope", Springer-Verlag, 2011.

        """
        if not self.axes_manager.signal_axes[0].is_uniform:
            raise NotImplementedError(
                "This operation is not yet implemented for non-uniform energy axes.")
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
        self._are_microscope_parameters_missing(
            ignore_parameters=['convergence_angle'])
        e0 = s.metadata.Acquisition_instrument.TEM.beam_energy
        beta = s.metadata.Acquisition_instrument.TEM.Detector.EELS.\
            collection_angle

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
            # The following prevents errors if the signal is a single spectrum
            if len(i0) != 1:
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
                             " not a NumPy array.")

        # Slicer to get the signal data from 0 to axis.size
        slicer = s.axes_manager._get_data_slice(
            [(axis.index_in_array, slice(None, axis.size)), ])

        # Kinetic definitions
        ke = e0 * (1 + e0 / 2. / me) / (1 + e0 / me) ** 2
        tgt = e0 * (2 * me + e0) / (me + e0)
        rk0 = 2590 * (1 + e0 / me) * np.sqrt(2 * ke / me)

        for io in range(iterations):
            # Calculation of the ELF by normalization of the SSD
            # Norm(SSD) = Imag(-1/epsilon) (Energy Loss Function, ELF)

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
                K = (Im / eaxis).sum(axis=axis.index_in_array, keepdims=True) \
                    * axis.scale
                K = (K / (np.pi / 2) / (1 - 1. / n ** 2))
                # K = (K / (np.pi / 2) / (1 - 1. / n ** 2)).reshape(
                #    np.insert(K.shape, axis.index_in_array, 1))
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
            # Use an optimal FFT size to speed up the calculation, and
            # make it double the closest upper value to workaround the
            # wrap-around problem.
            esize = optimal_fft_size(2 * axis.size)
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
                #  Calculates the surface ELF from a vacuum border effect
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
            # As above,prevent errors if the signal is a single spectrum
            if len(te) != 1:
                te = te[self.axes_manager._get_data_slice(
                        [(axis.index_in_array, 0)])]
            thickness = eps._get_navigation_signal(data=te)
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
        auto_background : bool, default True
            If True, and if spectrum is an EELS instance adds automatically
            a powerlaw to the model and estimate the parameters by the
            two-area method.
        auto_add_edges : bool, default True
            If True, and if spectrum is an EELS instance, it will
            automatically add the ionization edges as defined in the
            Signal1D instance. Adding a new element to the spectrum using
            the components.EELSSpectrum.add_elements method automatically
            add the corresponding ionisation edges to the model.
        GOS : {'hydrogenic' | 'Hartree-Slater'}, optional
            The generalized oscillation strength calculations to use for the
            core-loss EELS edges. If None the Hartree-Slater GOS are used if
            available, otherwise it uses the hydrogenic GOS.
        dictionary : {None | dict}, optional
            A dictionary to be used to recreate a model. Usually generated
            using :meth:`hyperspy.model.as_dictionary`

        Returns
        -------
        model : `EELSModel` instance.

        Raises
        ------
        NotImplementedError
            If the signal axis is a non-uniform axis.
        """
        from hyperspy.models.eelsmodel import EELSModel
        if ll is not None and not self.axes_manager.signal_axes[0].is_uniform:
            raise NotImplementedError(
                "Multiple scattering is not implemented for spectra with a non-uniform energy axis. "
                "To create a model that does not account for multiple-scattering do not set "
                "the `ll` keyword.")
        model = EELSModel(self,
                          ll=ll,
                          auto_background=auto_background,
                          auto_add_edges=auto_add_edges,
                          GOS=GOS,
                          dictionary=dictionary)
        return model

    def plot(self, plot_edges=False, only_edges=('Major', 'Minor'),
             **kwargs):
        """Plot the EELS spectrum. Markers indicating the position of the
        EELS edges can be added.

        Parameters
        ----------
        plot_edges : {False, True, list of string or string}
            If True, draws on s.metadata.Sample.elements for edges.
            Alternatively, provide a string of a single edge, or an iterable
            containing a list of valid elements, EELS families or edges. For
            example, an element should be 'Zr', an element edge family should
            be 'Zr_L' or an EELS edge 'Zr_L3'.
        only_edges : tuple of string
            Either 'Major' or 'Minor'. Defaults to both.
        kwargs
            The extra keyword arguments for plot()
        """

        super().plot(**kwargs)

        if plot_edges is not False:
            edges = self._get_edges_to_plot(plot_edges, only_edges)
            self.plot_edges_label(edges)

    def plot_edges_label(self, edges, vertical_line_marker=None,
                            text_marker=None):
        """Put the EELS edge label (vertical line segment and text box) on
        the signal

        Parameters
        ----------
        edges : dictionary
            A dictionary with the labels as keys and their energies as values.
            For example, {'Fe_L2': 721.0, 'O_K': 532.0}
        vertical_line_marker :  list
            A list contains HyperSpy's vertical line segment marker, if None,
            determine from the given edges
        text_marker :  list
            A list contains HyperSpy's text box marker, if None,
            determine from the given edges

        Raises
        ------
        ValueError
            If the size of edges, vertical_line_marker and text_marker do not
            match.
        """

        if vertical_line_marker is None or text_marker is None:
            # get position of markers for edges if no marker is provided
            # no marker provided, implies non-interactive mode
            slp = SpectrumLabelPosition(self)
            vertical_line_marker, text_marker = slp.get_markers(edges)
            # the object is needed to connect replot method when axes_manager
            # indices changed
            _ = EdgesRange(self, active=list(edges.keys()))
        if len(vertical_line_marker) != len(text_marker) or \
            len(edges) != len(vertical_line_marker):
            raise ValueError('The size of edges, vertical_line_marker and '
                             'text_marker needs to be the same.')

        # add the markers to the signal and store them
        self.add_marker(vertical_line_marker + text_marker, render_figure=False)
        added = dict(zip(edges, map(list, zip(vertical_line_marker, text_marker))))
        self._edge_markers.update(added)

    def _get_edges_to_plot(self, plot_edges, only_edges):
        # get the dictionary of the edge to be shown
        extra_element_edge_family = []
        if plot_edges is True:
            try:
                elements = self.metadata.Sample.elements
            except AttributeError:
                raise ValueError("No elements defined. Add them with "
                                 "s.add_elements, or specify elements, edge "
                                 "families or edges directly")
        else:
            extra_element_edge_family.extend(np.atleast_1d(plot_edges))
            try:
                elements = self.metadata.Sample.elements
            except:
                elements = []

        element_edge_family = elements + extra_element_edge_family
        edges_dict = self._get_edges(element_edge_family, only_edges)

        return edges_dict

    def _get_edges(self, element_edge_family, only_edges):
        # get corresponding information depending on whether it is an element
        # a particular edge or a family of edge
        axis_min = self.axes_manager[-1].low_value
        axis_max = self.axes_manager[-1].high_value

        names_and_energies = {}
        shells = ["K", "L", "M", "N", "O"]

        errmsg = ("Edge family '{}' is not supported. Supported edge family "
                  "is {}.")
        for member in element_edge_family:
            try:
                element, ss = member.split("_")

                if len(ss) == 1:
                    memtype = 'family'
                    if ss not in shells:
                        raise AttributeError(errmsg.format(ss, shells))
                if len(ss) == 2:
                    memtype = 'edge'
                    if ss[0] not in shells:
                        raise AttributeError(errmsg.format(ss[0], shells))
            except ValueError:
                element = member
                ss = ''
                memtype = 'element'

            try:
                Binding_energies = elements_db[element]["Atomic_properties"]["Binding_energies"]
            except KeyError as err:
                raise ValueError("'{}' is not a valid element".format(element)) from err

            for edge in Binding_energies.keys():
                relevance = Binding_energies[edge]["relevance"]
                energy = Binding_energies[edge]["onset_energy (eV)"]

                isInRel = relevance in only_edges
                isInRng = axis_min < energy < axis_max
                isSameFamily = ss in edge

                if memtype == 'element':
                    flag = isInRel & isInRng
                    edge_key = element + "_" + edge
                elif memtype == 'edge':
                    flag = isInRng & (edge == ss)
                    edge_key = member
                elif memtype == 'family':
                    flag = isInRel & isInRng & isSameFamily
                    edge_key = element + "_" + edge

                if flag:
                    names_and_energies[edge_key] = energy

        return names_and_energies

    def _edge_marker_closed(self, obj):
        marker = obj
        for EELS_edge, line_markers in reversed(list(
                self._edge_markers.items())):
            if marker in line_markers:
                line_markers.remove(marker)
            if not line_markers:
                self._edge_markers.pop(EELS_edge)

    def remove_EELS_edges_markers(self, EELS_edges):
        for EELS_edge in EELS_edges:
            if EELS_edge in self._edge_markers:
                line_markers = self._edge_markers[EELS_edge]
                while line_markers:
                    m = line_markers.pop()
                    m.close(render_figure=False)

    def get_complementary_edges(self, edges, only_major=False):
        ''' Get other edges of the same element present within the energy
        range of the axis

        Parameters
        ----------
        edges : iterable
            A sequence of strings contains edges in the format of
            element_subshell for EELS. For example, ['Fe_L2', 'O_K']
        only_major : bool
            Whether to show only the major edges. The default is False.

        Returns
        -------
        complmt_edges : list
            A list containing all the complementary edges of the same element
            present within the energy range of the axis
        '''

        emin = self.axes_manager[-1].low_value
        emax = self.axes_manager[-1].high_value
        complmt_edges = []

        elements = set()
        for edge in edges:
            element, _ = edge.split('_')
            elements.update([element])

        for element in elements:
            ss_info = elements_db[element]['Atomic_properties'][
                        'Binding_energies']

            for subshell in ss_info:
                sse = ss_info[subshell]['onset_energy (eV)']
                ssr = ss_info[subshell]['relevance']

                if only_major:
                    if ssr != 'Major':
                        continue

                edge = element + '_' + subshell
                if (emin <= sse <= emax) and (subshell[-1] != 'a') and \
                    (edge not in edges):
                    complmt_edges.append(edge)

        return complmt_edges

    def rebin(self, new_shape=None, scale=None, crop=True, dtype=None,
              out=None):
        factors = self._validate_rebin_args_and_get_factors(
            new_shape=new_shape,
            scale=scale)
        m = super().rebin(new_shape=new_shape, scale=scale, crop=crop,
                          dtype=dtype, out=out)
        m = out or m
        time_factor = np.prod([factors[axis.index_in_array]
                               for axis in m.axes_manager.navigation_axes])
        mdeels = m.metadata
        m.get_dimensions_from_data()
        if m.metadata.get_item("Acquisition_instrument.TEM.Detector.EELS"):
            mdeels = m.metadata.Acquisition_instrument.TEM.Detector.EELS
            if "dwell_time" in mdeels:
                mdeels.dwell_time *= time_factor
            if "exposure" in mdeels:
                mdeels.exposure *= time_factor
        else:
            _logger.info('No dwell_time could be found in the metadata so '
                         'this has not been updated.')
        if out is None:
            return m
        else:
            out.events.data_changed.trigger(obj=out)
        return m
    rebin.__doc__ = hyperspy.signal.BaseSignal.rebin.__doc__

    def vacuum_mask(self, threshold=10.0, start_energy=None,
                    closing=True, opening=False):
        """
        Generate mask of the vacuum region

        Parameters
        ----------
        threshold: float
            For a given navigation coordinate, mean value in the energy axis
            below which the pixel is considered as vacuum.
        start_energy: float, None
            Minimum energy included in the calculation of the mean intensity.
            If None, consider only the last quarter of the spectrum to
            calculate the mask.
        closing: bool
            If True, a morphological closing is applied to the mask.
        opening: bool
            If True, a morphological opening is applied to the mask.

        Returns
        -------
        mask: signal
            The mask of the region.
        """
        if self.axes_manager.navigation_dimension == 0:
            raise RuntimeError('Navigation dimenstion must be higher than 0 '
                               'to estimate a vacuum mask.')
        signal_axis = self.axes_manager.signal_axes[0]
        if start_energy is None:
            start_energy = 0.75 * signal_axis.high_value

        mask = (self.isig[start_energy:].mean(-1) <= threshold)

        from scipy.ndimage import binary_dilation, binary_erosion
        if closing:
            mask.data = binary_dilation(mask.data, border_value=0)
            mask.data = binary_erosion(mask.data, border_value=1)
        if opening:
            mask.data = binary_erosion(mask.data, border_value=1)
            mask.data = binary_dilation(mask.data, border_value=0)
        return mask


class LazyEELSSpectrum(EELSSpectrum, LazySignal1D):

    pass
