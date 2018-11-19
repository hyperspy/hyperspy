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

import logging
from collections import OrderedDict
import scipy.constants as constants
import numpy as np
from dask.array import Array as daArray
from pint import UnitRegistry, UndefinedUnitError

from hyperspy._signals.signal2d import Signal2D
from hyperspy.signal import BaseSignal
from hyperspy._signals.signal1d import Signal1D
from hyperspy._signals.lazy import LazySignal
from hyperspy.misc.holography.reconstruct import (
    reconstruct, estimate_sideband_position, estimate_sideband_size)
from hyperspy.misc.holography.tools import calculate_carrier_frequency, estimate_fringe_contrast_fourier

_logger = logging.getLogger(__name__)


def _first_nav_pixel_data(s):
    return s._data_aligned_with_axes[(0, ) *
                                     s.axes_manager.navigation_dimension]


def _parse_sb_position(s, reference, sb_position, sb, high_cf, parallel):

    if sb_position is None:
        _logger.warning('Sideband position is not specified. The sideband '
                        'will be found automatically which may cause '
                        'wrong results.')
        if reference is None:
            sb_position = s.estimate_sideband_position(
                sb=sb, high_cf=high_cf, parallel=parallel)
        else:
            sb_position = reference.estimate_sideband_position(
                sb=sb, high_cf=high_cf, parallel=parallel)

    else:
        if isinstance(sb_position, BaseSignal) and \
                not sb_position._signal_dimension == 1:
            raise ValueError('sb_position dimension has to be 1.')

        if not isinstance(sb_position, Signal1D):
            sb_position = Signal1D(sb_position)
            if isinstance(sb_position.data, daArray):
                sb_position = sb_position.as_lazy()

        if not sb_position.axes_manager.signal_size == 2:
            raise ValueError('sb_position should to have signal size of 2.')

    if sb_position.axes_manager.navigation_size != s.axes_manager.navigation_size:
        if sb_position.axes_manager.navigation_size:
            raise ValueError('Sideband position dimensions do not match'
                             ' neither reference nor hologram dimensions.')
        # sb_position navdim=0, therefore map function should not iterate:
        else:
            sb_position_temp = sb_position.data
    else:
        sb_position_temp = sb_position.deepcopy()
    return sb_position, sb_position_temp


def _parse_sb_size(s, reference, sb_position, sb_size, parallel):
    # Default value is 1/2 distance between sideband and central band
    if sb_size is None:
        if reference is None:
            sb_size = s.estimate_sideband_size(
                sb_position, parallel=parallel)
        else:
            sb_size = reference.estimate_sideband_size(
                sb_position, parallel=parallel)
    else:
        if not isinstance(sb_size, BaseSignal):
            if isinstance(sb_size,
                          (np.ndarray, daArray)) and sb_size.size > 1:
                # transpose if np.array of multiple instances
                sb_size = BaseSignal(sb_size).T
            else:
                sb_size = BaseSignal(sb_size)
            if isinstance(sb_size.data, daArray):
                sb_size = sb_size.as_lazy()

    if sb_size.axes_manager.navigation_size != s.axes_manager.navigation_size:
        if sb_size.axes_manager.navigation_size:
            raise ValueError('Sideband size dimensions do not match '
                             'neither reference nor hologram dimensions.')
        # sb_position navdim=0, therefore map function should not iterate:
        else:
            sb_size_temp = np.float64(sb_size.data)
    else:
        sb_size_temp = sb_size.deepcopy()
    return sb_size, sb_size_temp


def _estimate_fringe_contrast_statistical(holo):
    """
    Estimates average fringe contrast of a hologram using statistical definition:
    V = STD / MEAN.

    Parameters
    ----------
    holo_data: ndarray
        The data of the hologram.

    Returns
    -------
    Fringe contrast as a float
    """

    axes = holo.axes_manager.signal_axes

    return holo.std(axes) / holo.mean(axes)


class HologramImage(Signal2D):
    """Image subclass for holograms acquired via off-axis electron holography."""

    _signal_type = 'hologram'

    def set_microscope_parameters(self,
                                  beam_energy=None,
                                  biprism_voltage=None,
                                  tilt_stage=None):
        """Set the microscope parameters.

        If no arguments are given, raises an interactive mode to fill
        the values.

        Parameters
        ----------
        beam_energy: float
            The energy of the electron beam in keV
        biprism_voltage : float
            In volts
        tilt_stage : float
            In degrees

        Examples
        --------

        >>> s.set_microscope_parameters(beam_energy=300.)
        >>> print('Now set to %s keV' %
        >>>       s.metadata.Acquisition_instrument.
        >>>       TEM.beam_energy)

        Now set to 300.0 keV

        """
        md = self.metadata

        if beam_energy is not None:
            md.set_item("Acquisition_instrument.TEM.beam_energy", beam_energy)
        if biprism_voltage is not None:
            md.set_item("Acquisition_instrument.TEM.Biprism.voltage",
                        biprism_voltage)
        if tilt_stage is not None:
            md.set_item(
                "Acquisition_instrument.TEM.Stage.tilt_alpha",
                tilt_stage)

    def estimate_sideband_position(self,
                                   ap_cb_radius=None,
                                   sb='lower',
                                   high_cf=True,
                                   show_progressbar=False,
                                   parallel=None):
        """
        Estimates the position of the sideband and returns its position.

        Parameters
        ----------
        ap_cb_radius: float, None
            The aperture radius used to mask out the centerband.
        sb : str, optional
            Chooses which sideband is taken. 'lower' or 'upper'
        high_cf : bool, optional
            If False, the highest carrier frequency allowed for the sideband location is equal to
            half of the Nyquist frequency (Default: True).
        show_progressbar : boolean
            Shows progressbar while iterating over different slices of the signal (passes the
            parameter to map method).
        parallel : bool
            Estimate the positions in parallel

        Returns
        -------
        Signal1D instance of sideband positions (y, x), referred to the unshifted FFT.

        Examples
        --------

        >>> import hyperspy.api as hs
        >>> s = hs.datasets.example_signals.object_hologram()
        >>> sb_position = s.estimate_sideband_position()
        >>> sb_position.data

        array([124, 452])
        """

        sb_position = self.map(
            estimate_sideband_position,
            holo_sampling=(self.axes_manager.signal_axes[0].scale,
                           self.axes_manager.signal_axes[1].scale),
            central_band_mask_radius=ap_cb_radius,
            sb=sb,
            high_cf=high_cf,
            show_progressbar=show_progressbar,
            inplace=False,
            parallel=parallel,
            ragged=False)

        # Workaround to a map disfunctionality:
        sb_position.set_signal_type('signal1d')

        return sb_position

    def estimate_sideband_size(self,
                               sb_position,
                               show_progressbar=False,
                               parallel=None):
        """
        Estimates the size of the sideband and returns its size.

        Parameters
        ----------
        sb_position : :class:`~hyperspy.signals.BaseSignal
            The sideband position (y, x), referred to the non-shifted FFT.
        show_progressbar: boolean
            Shows progressbar while iterating over different slices of the signal (passes the parameter to map method).
        parallel : bool
            Estimate the sizes in parallel

        Returns
        -------
        Signal 1D instance with sideband size, referred to the unshifted FFT.

        Examples
        --------
        >>> import hyperspy.api as hs
        >>> s = hs.datasets.example_signals.object_hologram()
        >>> sb_position = s.estimate_sideband_position()
        >>> sb_size = s.estimate_sideband_size(sb_position)
        >>> sb_size.data

        array([ 68.87670143])
        """

        sb_size = sb_position.map(
            estimate_sideband_size,
            holo_shape=self.axes_manager.signal_shape[::-1],
            show_progressbar=show_progressbar,
            inplace=False,
            parallel=parallel,
            ragged=False)

        return sb_size

    def reconstruct_phase(self,
                          reference=None,
                          sb_size=None,
                          sb_smoothness=None,
                          sb_unit=None,
                          sb='lower',
                          sb_position=None,
                          high_cf=True,
                          output_shape=None,
                          plotting=False,
                          show_progressbar=False,
                          store_parameters=True,
                          parallel=None):
        """Reconstruct electron holograms. Operates on multidimensional
        hyperspy signals. There are several usage schemes:
         1. Reconstruct 1d or Nd hologram without reference
         2. Reconstruct 1d or Nd hologram using single reference hologram
         3. Reconstruct Nd hologram using Nd reference hologram (applies each
         reference to each hologram in Nd stack)

         The reconstruction parameters (sb_position, sb_size, sb_smoothness)
         have to be 1d or to have same dimensionality as the hologram.

        Parameters
        ----------
        reference : ndarray, :class:`~hyperspy.signals.Signal2D, None
            Vacuum reference hologram.
        sb_size : float, ndarray, :class:`~hyperspy.signals.BaseSignal, None
            Sideband radius of the aperture in corresponding unit (see
            'sb_unit'). If None, the radius of the aperture is set to 1/3 of
            the distance between sideband and center band.
        sb_smoothness : float, ndarray, :class:`~hyperspy.signals.BaseSignal, None
            Smoothness of the aperture in the same unit as sb_size.
        sb_unit : str, None
            Unit of the two sideband parameters 'sb_size' and 'sb_smoothness'.
            Default: None - Sideband size given in pixels
            'nm': Size and smoothness of the aperture are given in 1/nm.
            'mrad': Size and smoothness of the aperture are given in mrad.
        sb : str, None
            Select which sideband is selected. 'upper' or 'lower'.
        sb_position : tuple, :class:`~hyperspy.signals.Signal1D, None
            The sideband position (y, x), referred to the non-shifted FFT. If
            None, sideband is determined automatically from FFT.
        high_cf : bool, optional
            If False, the highest carrier frequency allowed for the sideband location is equal to
            half of the Nyquist frequency (Default: True).
        output_shape: tuple, None
            Choose a new output shape. Default is the shape of the input
            hologram. The output shape should not be larger than the input
            shape.
        plotting : boolean
            Shows details of the reconstruction (i.e. SB selection).
        show_progressbar : boolean
            Shows progressbar while iterating over different slices of the
            signal (passes the parameter to map method).
        parallel : bool
            Run the reconstruction in parallel
        store_parameters : boolean
            Store reconstruction parameters in metadata

        Returns
        -------
        wave : :class:`~hyperspy.signals.WaveImage
            Reconstructed electron wave. By default object wave is devided by
            reference wave

        Examples
        --------
        >>> import hyperspy.api as hs
        >>> s = hs.datasets.example_signals.object_hologram()
        >>> sb_position = s.estimate_sideband_position()
        >>> sb_size = s.estimate_sideband_size(sb_position)
        >>> sb_size.data
        >>> wave = s.reconstruct_phase(sb_position=sb_position, sb_size=sb_size)

        """

        # TODO: Use defaults for choosing sideband, smoothness, relative filter
        # size and output shape if not provided
        # TODO: Plot FFT with marked SB and SB filter if plotting is enabled

        # Parsing reference:
        if not isinstance(reference, HologramImage):
            if isinstance(reference, Signal2D):
                if (not reference.axes_manager.navigation_shape ==
                        self.axes_manager.navigation_shape and
                        reference.axes_manager.navigation_size):

                    raise ValueError('The navigation dimensions of object and'
                                     'reference holograms do not match')

                _logger.warning('The reference image signal type is not '
                                'HologramImage. It will be converted to '
                                'HologramImage automatically.')
                reference.set_signal_type('hologram')
            elif reference is not None:
                reference = HologramImage(reference)
                if isinstance(reference.data, daArray):
                    reference = reference.as_lazy()

        # Testing match of navigation axes of reference and self
        # (exception: reference nav_dim=1):
        if (reference and not reference.axes_manager.navigation_shape ==
                self.axes_manager.navigation_shape and
                reference.axes_manager.navigation_size):

            raise ValueError('The navigation dimensions of object and '
                             'reference holograms do not match')

        if reference and not reference.axes_manager.signal_shape == self.axes_manager.signal_shape:

            raise ValueError('The signal dimensions of object and reference'
                             ' holograms do not match')

        # Parsing sideband position:
        (sb_position, sb_position_temp) = _parse_sb_position(self,
                                                             reference,
                                                             sb_position,
                                                             sb,
                                                             high_cf,
                                                             parallel)

        # Parsing sideband size:
        (sb_size, sb_size_temp) = _parse_sb_size(self,
                                                 reference,
                                                 sb_position,
                                                 sb_size,
                                                 parallel)

        # Standard edge smoothness of sideband aperture 5% of sb_size
        if sb_smoothness is None:
            sb_smoothness = sb_size * 0.05
        else:
            if not isinstance(sb_smoothness, BaseSignal):
                if isinstance(
                    sb_smoothness,
                        (np.ndarray, daArray)) and sb_smoothness.size > 1:
                    sb_smoothness = BaseSignal(sb_smoothness).T
                else:
                    sb_smoothness = BaseSignal(sb_smoothness)
                if isinstance(sb_smoothness.data, daArray):
                    sb_smoothness = sb_smoothness.as_lazy()

        if sb_smoothness.axes_manager.navigation_size != self.axes_manager.navigation_size:
            if sb_smoothness.axes_manager.navigation_size:
                raise ValueError('Sideband smoothness dimensions do not match'
                                 ' neither reference nor hologram '
                                 'dimensions.')
            # sb_position navdim=0, therefore map function should not iterate
            # it:
            else:
                sb_smoothness_temp = np.float64(sb_smoothness.data)
        else:
            sb_smoothness_temp = sb_smoothness.deepcopy()

        # Convert sideband size from 1/nm or mrad to pixels
        if sb_unit == 'nm':
            f_sampling = np.divide(
                1,
                [a * b for a, b in
                 zip(self.axes_manager.signal_shape,
                     (self.axes_manager.signal_axes[0].scale,
                      self.axes_manager.signal_axes[1].scale))]
            )
            sb_size_temp = sb_size_temp / np.mean(f_sampling)
            sb_smoothness_temp = sb_smoothness_temp / np.mean(f_sampling)
        elif sb_unit == 'mrad':
            f_sampling = np.divide(
                1,
                [a * b for a, b in
                 zip(self.axes_manager.signal_shape,
                     (self.axes_manager.signal_axes[0].scale,
                      self.axes_manager.signal_axes[1].scale))]
            )
            try:
                ht = self.metadata.Acquisition_instrument.TEM.beam_energy
            except BaseException:
                raise AttributeError("Please define the beam energy."
                                     "You can do this e.g. by using the "
                                     "set_microscope_parameters method")

            momentum = 2 * constants.m_e * constants.elementary_charge * ht * \
                1000 * (1 + constants.elementary_charge * ht *
                        1000 / (2 * constants.m_e * constants.c ** 2))
            wavelength = constants.h / np.sqrt(momentum) * 1e9  # in nm
            sb_size_temp = sb_size_temp / (1000 * wavelength *
                                           np.mean(f_sampling))
            sb_smoothness_temp = sb_smoothness_temp / (1000 * wavelength *
                                                       np.mean(f_sampling))

        # Find output shape:
        if output_shape is None:
            # Future improvement will give a possibility to choose
            # if sb_size.axes_manager.navigation_size > 0:
            #     output_shape = (np.int(sb_size.inav[0].data*2), np.int(sb_size.inav[0].data*2))
            # else:
            #     output_shape = (np.int(sb_size.data*2), np.int(sb_size.data*2))
            output_shape = self.axes_manager.signal_shape
            output_shape = output_shape[::-1]

        # Logging the reconstruction parameters if appropriate:
        _logger.info('Sideband position in pixels: {}'.format(sb_position))
        _logger.info('Sideband aperture radius in pixels: {}'.format(sb_size))
        _logger.info('Sideband aperture smoothness in pixels: {}'.format(
            sb_smoothness))

        # Reconstructing object electron wave:

        # Checking if reference is a single image, which requires sideband
        # parameters as a nparray to avoid iteration trough those:
        wave_object = self.map(
            reconstruct,
            holo_sampling=(self.axes_manager.signal_axes[0].scale,
                           self.axes_manager.signal_axes[1].scale),
            sb_size=sb_size_temp,
            sb_position=sb_position_temp,
            sb_smoothness=sb_smoothness_temp,
            output_shape=output_shape,
            plotting=plotting,
            show_progressbar=show_progressbar,
            inplace=False,
            parallel=parallel,
            ragged=False)

        # Reconstructing reference wave and applying it (division):
        if reference is None:
            wave_reference = 1
        # case when reference is 1d
        elif reference.axes_manager.navigation_size != self.axes_manager.navigation_size:

            # Prepare parameters for reconstruction of the reference wave:

            if reference.axes_manager.navigation_size == 0 and \
               sb_position.axes_manager.navigation_size > 0:
                # 1d reference, but parameters are multidimensional
                sb_position_ref = _first_nav_pixel_data(sb_position_temp)
            else:
                sb_position_ref = sb_position_temp

            if reference.axes_manager.navigation_size == 0 and \
               sb_size.axes_manager.navigation_size > 0:
                # 1d reference, but parameters are multidimensional
                sb_size_ref = _first_nav_pixel_data(sb_size_temp)
            else:
                sb_size_ref = sb_size_temp

            if reference.axes_manager.navigation_size == 0 and \
               sb_smoothness.axes_manager.navigation_size > 0:
                # 1d reference, but parameters are multidimensional
                sb_smoothness_ref = np.float64(
                    _first_nav_pixel_data(sb_smoothness_temp))
            else:
                sb_smoothness_ref = sb_smoothness_temp
            #

            wave_reference = reference.map(
                reconstruct,
                holo_sampling=(self.axes_manager.signal_axes[0].scale,
                               self.axes_manager.signal_axes[1].scale),
                sb_size=sb_size_ref,
                sb_position=sb_position_ref,
                sb_smoothness=sb_smoothness_ref,
                output_shape=output_shape,
                plotting=plotting,
                show_progressbar=show_progressbar,
                inplace=False,
                parallel=parallel,
                ragged=False)

        else:
            wave_reference = reference.map(
                reconstruct,
                holo_sampling=(self.axes_manager.signal_axes[0].scale,
                               self.axes_manager.signal_axes[1].scale),
                sb_size=sb_size_temp,
                sb_position=sb_position_temp,
                sb_smoothness=sb_smoothness_temp,
                output_shape=output_shape,
                plotting=plotting,
                show_progressbar=show_progressbar,
                inplace=False,
                parallel=parallel,
                ragged=False)

        wave_image = wave_object / wave_reference

        # New signal is a complex
        wave_image.set_signal_type('complex_signal2d')

        wave_image.axes_manager.signal_axes[0].scale = \
            self.axes_manager.signal_axes[0].scale * \
            self.axes_manager.signal_shape[0] / output_shape[1]
        wave_image.axes_manager.signal_axes[1].scale = \
            self.axes_manager.signal_axes[1].scale * \
            self.axes_manager.signal_shape[1] / output_shape[0]

        # Reconstruction parameters are stored in
        # holo_reconstruction_parameters:

        if store_parameters:
            rec_param_dict = OrderedDict(
                [('sb_position', sb_position_temp), ('sb_size', sb_size_temp),
                 ('sb_units', sb_unit), ('sb_smoothness', sb_smoothness_temp)])
            wave_image.metadata.Signal.add_node('Holography')
            wave_image.metadata.Signal.Holography.add_node(
                'Reconstruction_parameters')
            wave_image.metadata.Signal.Holography.Reconstruction_parameters.add_dictionary(
                rec_param_dict)
            _logger.info('Reconstruction parameters stored in metadata')

        return wave_image

    def statistics(self,
                   sb_position=None,
                   sb='lower',
                   high_cf=False,
                   fringe_contrast_algorithm='statistical',
                   apodization='hanning',
                   single_values=True,
                   show_progressbar=False,
                   parallel=None):
        """
        Calculates following statistics for off-axis electron holograms:

        1. Fringe contrast using either statistical definition or
        Fourier space approach (see description of `fringe_contrast_algorithm` parameter)
        2. Fringe sampling (in pixels)
        3. Fringe spacing (in calibrated units)
        4. Carrier frequency (in calibrated units, radians and 1/px)

        Parameters
        ----------
        sb_position : tuple, :class:`~hyperspy.signals.Signal1D, None
            The sideband position (y, x), referred to the non-shifted FFT.
            It has to be tuple or to have the same dimensionality as the hologram.
            If None, sideband is determined automatically from FFT.
        sb : str, None
            Select which sideband is selected. 'upper', 'lower', 'left' or 'right'.
        high_cf : bool, optional
            If False, the highest carrier frequency allowed for the sideband location is equal to
            half of the Nyquist frequency (Default: False).
        fringe_contrast_algorithm : str
            Select fringe contrast algorithm between:

            'fourier'
                fringe contrast is estimated as:
                2 * <I(k_0)> / <I(0)>,
                where I(k_0) is intensity of sideband and I(0) is the intensity of central band (FFT origin).
                This method delivers also reasonable estimation if
                interference pattern do not cover full field of view.
            'statistical'
                fringe contrast is estimated by dividing standard deviation by mean
                of the hologram intensity in real space. This algorithm relays on that the fringes are regular and
                covering entire field of view.

            (Default: 'statistical')
        apodization: str or None, optional
            Used with `fringe_contrast_algorithm='fourier'`. If 'hanning' or 'hamming' apodization window
            will be applied in real space before FFT for estimation of fringe contrast.
            Apodization is typically needed to suppress striking  due to sharp edges of the image,
            which often results in underestimation of the fringe contrast. (Default: 'hanning')
        single_values : bool, optional
            If True calculates statistics only for the first navigation pixels and
            returns the values as single floats (Default: True)
        show_progressbar : bool, optional
            Shows progressbar while iterating over different slices of the
            signal (passes the parameter to map method). (Default: False)
        parallel : bool, None, optional
            Run the reconstruction in parallel

        Returns
        -------
        statistics_dict :
            Dictionary with the statistics

        Examples
        --------
        >>> import hyperspy.api as hs
        >>> s = hs.datasets.example_signals.reference_hologram()
        >>> sb_position = s.estimate_sideband_position(high_cf=True)
        >>> s.statistics(sb_position=sb_position)
        {'Fringe spacing (nm)': 3.4860442674236256,
        'Carrier frequency (1/px)': 0.26383819985575441,
        'Carrier frequency (mrad)': 0.56475154609203482,
        'Fringe contrast': 0.071298357213623778,
        'Fringe sampling (px)': 3.7902017241882331,
        'Carrier frequency (1 / nm)': 0.28685808994016415}
        """

        # Testing match of navigation axes of reference and self
        # (exception: reference nav_dim=1):

        # Parsing sideband position:
        (sb_position, sb_position_temp) = _parse_sb_position(
            self, None, sb_position, sb, high_cf, parallel)

        # Calculate carrier frequency in 1/px and fringe sampling:
        fourier_sampling = 1. / np.array(self.axes_manager.signal_shape)
        if single_values:
            carrier_freq_px = calculate_carrier_frequency(_first_nav_pixel_data(self),
                                                          sb_position=_first_nav_pixel_data(
                                                              sb_position),
                                                          scale=fourier_sampling)
        else:
            carrier_freq_px = self.map(calculate_carrier_frequency,
                                       sb_position=sb_position,
                                       scale=fourier_sampling,
                                       inplace=False,
                                       ragged=False,
                                       show_progressbar=show_progressbar,
                                       parallel=parallel)
        fringe_sampling = np.divide(1., carrier_freq_px)

        ureg = UnitRegistry()
        try:
            units = ureg.parse_expression(
                str(self.axes_manager.signal_axes[0].units))
        except UndefinedUnitError:
            raise ValueError('Signal axes units should be defined.')

        # Calculate carrier frequency in 1/units and fringe spacing in units:
        f_sampling_units = np.divide(
            1.,
            [a * b for a, b in
             zip(self.axes_manager.signal_shape,
                 (self.axes_manager.signal_axes[0].scale,
                  self.axes_manager.signal_axes[1].scale))]
        )
        if single_values:
            carrier_freq_units = calculate_carrier_frequency(_first_nav_pixel_data(self),
                                                             sb_position=_first_nav_pixel_data(
                                                                 sb_position),
                                                             scale=f_sampling_units)
        else:
            carrier_freq_units = self.map(calculate_carrier_frequency,
                                          sb_position=sb_position,
                                          scale=f_sampling_units,
                                          inplace=False,
                                          ragged=False,
                                          show_progressbar=show_progressbar,
                                          parallel=parallel)
        fringe_spacing = np.divide(1., carrier_freq_units)

        # Calculate carrier frequency in mrad:
        try:
            ht = self.metadata.Acquisition_instrument.TEM.beam_energy
        except BaseException:
            raise AttributeError("Please define the beam energy."
                                 "You can do this e.g. by using the "
                                 "set_microscope_parameters method.")

        momentum = 2 * constants.m_e * constants.elementary_charge * ht * \
            1000 * (1 + constants.elementary_charge * ht *
                    1000 / (2 * constants.m_e * constants.c ** 2))
        wavelength = constants.h / np.sqrt(momentum) * 1e9  # in nm
        carrier_freq_quantity = wavelength * \
            ureg('nm') * carrier_freq_units / units * ureg('rad')
        carrier_freq_mrad = carrier_freq_quantity.to('mrad').magnitude

        # Calculate fringe contrast:
        if fringe_contrast_algorithm == 'fourier':
            if single_values:
                fringe_contrast = estimate_fringe_contrast_fourier(_first_nav_pixel_data(self),
                                                                   sb_position=_first_nav_pixel_data(
                                                                       sb_position),
                                                                   apodization=apodization)
            else:
                fringe_contrast = self.map(estimate_fringe_contrast_fourier,
                                           sb_position=sb_position,
                                           apodization=apodization,
                                           inplace=False,
                                           ragged=False,
                                           show_progressbar=show_progressbar,
                                           parallel=parallel)
        elif fringe_contrast_algorithm == 'statistical':
            if single_values:
                fringe_contrast = _first_nav_pixel_data(
                    self).std() / _first_nav_pixel_data(self).mean()
            else:
                fringe_contrast = _estimate_fringe_contrast_statistical(self)
        else:
            raise ValueError(
                "fringe_contrast_algorithm can only be set to fourier or statistical.")

        return {'Fringe contrast': fringe_contrast,
                'Fringe sampling (px)': fringe_sampling,
                'Fringe spacing ({:~})'.format(units.units): fringe_spacing,
                'Carrier frequency (1/px)': carrier_freq_px,
                'Carrier frequency ({:~})'.format((1. / units).units): carrier_freq_units,
                'Carrier frequency (mrad)': carrier_freq_mrad}


class LazyHologramImage(LazySignal, HologramImage):

    _lazy = True
