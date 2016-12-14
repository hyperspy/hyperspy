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

import numpy as np
from hyperspy.signals import Signal2D, BaseSignal, Signal1D
from collections import OrderedDict
from hyperspy.misc.holography.reconstruct import reconstruct, find_sideband_position, find_sideband_size
import logging
import warnings
import scipy.constants as constants

_logger = logging.getLogger(__name__)


class HologramImage(Signal2D):
    """Image subclass for holograms acquired via off-axis electron holography."""

    _signal_type = 'hologram'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampling = (self.axes_manager[0].scale, self.axes_manager[1].scale)
        self.f_sampling = np.divide(1, [a * b for a, b in zip(self.data.shape, self.sampling)])

    def find_sideband_position(self, ap_cb_radius=None, sb='lower', show_progressbar=False):
        """
        Finds the position of the sideband and returns its position.

        Parameters
        ----------
        ap_cb_radius: float, None
            The aperture radius used to mask out the centerband.
        sb : str, optional
            Chooses which sideband is taken. 'lower' or 'upper'
        show_progressbar : boolean
            Shows progressbar while iterating over different slices of the signal (passes the parameter to map method).

        Returns
        -------
        Signal1D instance of sideband positions (y, x), referred to the unshifted FFT.
        """

        sb_position = self.deepcopy()
        sb_position.map(find_sideband_position, holo_sampling=self.sampling, central_band_mask_radius=ap_cb_radius,
                        sb=sb, show_progressbar=show_progressbar)

        # Workaround to a map disfunctionality:
        sb_position.set_signal_type('signal1d')

        return sb_position

    def find_sideband_size(self, sb_position, show_progressbar=False):
        """
        Finds the size of the sideband and returns its position.

        Parameters
        ----------
        sb_position : :class:`~hyperspy.signals.BaseSignal
            The sideband position (y, x), referred to the non-shifted FFT.
        show_progressbar: boolean
            Shows progressbar while iterating over different slices of the signal (passes the parameter to map method).

        Returns
        -------
        Signal 1D instance with sideband size, referred to the unshifted FFT.
        """
        sb_size = sb_position.deepcopy()
        sb_size.map(find_sideband_size, holo_shape=self.axes_manager.signal_shape, show_progressbar=show_progressbar)

        return sb_size

    def reconstruct_phase(self, reference=None, sb_size=None, sb_smoothness=None, sb_unit=None,
                          sb='lower', sb_position=None, output_shape=None, plotting=False, show_progressbar=False,
                          log_parameters=True):
        """Reconstruct electron holograms. Operates on multidimensional hyperspy signals. There are several usage
        schemes:
         1. Reconstruct 1d or Nd hologram without reference
         2. Reconstruct 1d or Nd hologram using single reference hologram
         3. Reconstruct Nd hologram using Nd reference hologram (applies each reference to each hologram in Nd stack)

         The reconstruction parameters (sb_position, sb_size, sb_smoothness) have to be 1d or to have same
         dimensionality as the hologram.

        Parameters
        ----------
        reference : ndarray, :class:`~hyperspy.signals.Signal2D, None
            Vacuum reference hologram.
        sb_size : float, :class:`~hyperspy.signals.BaseSignal, None
            Sideband radius of the aperture in corresponding unit (see 'sb_unit'). If None,
            the radius of the aperture is set to 1/3 of the distance between sideband and
            centreband.
        sb_smoothness : float, :class:`~hyperspy.signals.BaseSignal, None
            Smoothness of the aperture in the same unit as sb_size.
        sb_unit : str, None
            Unit of the two sideband parameters 'sb_size' and 'sb_smoothness'.
            Default: None - Sideband size given in pixels
            'nm': Size and smoothness of the aperture are given in 1/nm.
            'mrad': Size and smoothness of the aperture are given in mrad.
        sb : str, None
            Select which sideband is selected. 'upper' or 'lower'.
        sb_position : tuple, :class:`~hyperspy.signals.Signal1D, None
            Sideband position in pixel. If None, sideband is determined automatically from FFT.
        output_shape: tuple, None
            Choose a new output shape. Default is the shape of the input hologram. The output
            shape should not be larger than the input shape.
        plotting : boolean
            Shows details of the reconstruction (i.e. SB selection).
        show_progressbar : boolean
            Shows progressbar while iterating over different slices of the signal (passes the parameter to map method).
        log_parameters : boolean
            Logs reconstruction parameters

        Returns
        -------
        wave : :class:`~hyperspy.signals.WaveImage
            Reconstructed electron wave. By default object wave is devided by reference wave

        Notes
        -----

        """

        # Parsing reference:
        if not isinstance(reference, HologramImage):
            if isinstance(reference, Signal2D):
                _logger.warning('The reference image signal type is not HologramImage. It will '
                                'be converted to HologramImage automatically.')
                reference.set_signal_type('hologram')
            elif reference is not None:
                reference = HologramImage(reference)

        # Testing match of navigation axes of reference and self (exception: reference nav_dim=1):
        if (reference and not reference.axes_manager.navigation_shape == self.axes_manager.navigation_shape
            and reference.axes_manager.navigation_size):

            raise ValueError('The navigation dimensions of object and reference holograms do not match')

        # Parsing sideband position:
        if sb_position is None:
            warnings.warn('Sideband position is not specified. The sideband will be found automatically which may '
                          'cause wrong results.')
            if reference is None:
                sb_position = self.find_sideband_position(sb=sb)
            else:
                sb_position = reference.find_sideband_position(sb=sb)

        else:
            if not isinstance(sb_position, Signal1D):
                sb_position = Signal1D(sb_position)

        if sb_position.axes_manager.navigation_size != self.axes_manager.navigation_size:
            if sb_position.axes_manager.navigation_size:
                warnings.warn('Sideband position dimensions do not match neither reference nor hologram dimensions.'
                              'The reconstruction will be performed with the first values')
                sb_position_temp = sb_position.inav[0].data
            else:  # sb_position navdim=0, therefore map function should not iterate it:
                sb_position_temp = sb_position.data
        else:
            sb_position_temp = sb_position.deepcopy()
        #

        # Parsing sideband size
        if sb_size is None:  # Default value is 1/2 distance between sideband and central band
            if reference is None:
                sb_size = self.find_sideband_size(sb_position)
            else:
                sb_size = reference.find_sideband_size(sb_position)
        else:
            if not isinstance(sb_size, BaseSignal):
                if isinstance(sb_size, np.ndarray) and sb_size.size > 1:  # transpose if np.array of multiple instances
                    sb_size = BaseSignal(sb_size).T
                else:
                    sb_size = BaseSignal(sb_size)

        if sb_size.axes_manager.navigation_size != self.axes_manager.navigation_size:
            if sb_size.axes_manager.navigation_size:
                warnings.warn('Sideband size dimensions do not match neither reference nor hologram dimensions.'
                              'The reconstruction will be performed with the first value.')
                sb_size_temp = np.float64(sb_size.inav[0].data)
            else:  # sb_position navdim=0, therefore map function should not iterate it:
                sb_size_temp = np.float64(sb_size.data)
        else:
            sb_size_temp = sb_size.deepcopy()
        #

        # Standard edge smoothness of sideband aperture 5% of sb_size
        if sb_smoothness is None:
            sb_smoothness = sb_size * 0.05
        else:
            if not isinstance(sb_smoothness, BaseSignal):
                if isinstance(sb_smoothness, np.ndarray) and sb_smoothness.size > 1:
                    sb_smoothness = BaseSignal(sb_smoothness).T
                else:
                    sb_smoothness = BaseSignal(sb_smoothness)

        if sb_smoothness.axes_manager.navigation_size != self.axes_manager.navigation_size:
            if sb_smoothness.axes_manager.navigation_size:
                warnings.warn('Sideband smoothness dimensions do not match neither reference nor hologram dimensions.'
                              'The reconstruction will be performed with the first value.')
                sb_smoothness_temp = np.float64(sb_smoothness.inav[0].data)
            else:  # sb_position navdim=0, therefore map function should not iterate it:
                sb_smoothness_temp = np.float64(sb_smoothness.data)
        else:
            sb_smoothness_temp = sb_smoothness.deepcopy()

        # Convert sideband size from 1/nm or mrad to pixels
        if sb_unit == 'nm':
            sb_size_temp = sb_size_temp / np.mean(self.f_sampling)
            sb_smoothness = sb_smoothness / np.mean(self.f_sampling)
        elif sb_unit == 'mrad':
            try:
                ht = self.metadata.Acquisition_instrument.TEM.beam_energy
            except AttributeError:
                ht = int(input('Enter beam energy in kV: '))
                self.metadata.add_node('Acquisition_instrument')
                self.metadata.Acquisition_instrument.add_node('TEM')
                self.metadata.Acquisition_instrument.TEM.add_node('beam_energy')
                self.metadata.Acquisition_instrument.TEM.beam_energy = ht
            momentum = 2 * constants.m_e * constants.elementary_charge * ht * 1000 *\
                       (1 + constants.elementary_charge * ht * 1000 / (2 * constants.m_e *
                                                                       constants.c ** 2))
            wavelength = constants.h / np.sqrt(momentum) * 1e9 # in nm
            sb_size_temp = sb_size_temp / (1000 * wavelength * np.mean(self.f_sampling))
            sb_smoothness_temp = sb_smoothness_temp / (1000 * wavelength * np.mean(
                self.f_sampling))

        # Find output shape:
        if output_shape is None:
            # if sb_size.axes_manager.navigation_size > 0: #  Future improvement will give a possibility to choose
            #     output_shape = (np.int(sb_size.inav[0].data*2), np.int(sb_size.inav[0].data*2))
            # else:
            #     output_shape = (np.int(sb_size.data*2), np.int(sb_size.data*2))
            output_shape = self.axes_manager.signal_shape

        # Logging the reconstruction parameters if appropriate:
        if log_parameters:
            _logger.info('Sideband position in pixels: {}'.format(sb_position))
            _logger.info('Sideband aperture radius in pixels: {}'.format(sb_size))
            _logger.info('Sideband aperture smoothness in pixels: {}'.format(sb_smoothness))

        # Shows the selected sideband and the position of the sideband
        # if plotting:
        #     fft_holo = fft2(self.data) / np.prod(self.data.shape)
        #     fig, axs = plt.subplots(1, 1, figsize=(4, 4))
        #     axs.imshow(np.abs(fft_holo), clim=(0, 2.2))
        #     axs.scatter(sb_position[1], sb_position[0], s=20, color='red', marker='x')
        #     axs.set_xlim(sb_position[1]-sb_size, sb_position[1]+sb_size)
        #     axs.set_ylim(sb_position[0]-sb_size, sb_position[0]+sb_size)
        #     plt.show()

        # Reconstructing object electron wave:
        wave_object = self.deepcopy()

        # Checking if reference is a single image, which requires sideband parameters as a nparray to avoid iteration
        # trough those:
        wave_object.map(reconstruct, holo_sampling=self.sampling, sb_size=sb_size_temp,
                        sb_position=sb_position_temp, sb_smoothness=sb_smoothness_temp,
                        output_shape=output_shape, plotting=plotting, show_progressbar=show_progressbar)

        # Reconstructing reference wave and applying it (division):
        if reference is None:
            wave_reference = 1
        elif reference.axes_manager.navigation_size != self.axes_manager.navigation_size:  # case when reference is 1d
            wave_reference = reference.deepcopy()

            # Prepare parameters for reconstruction of the reference wave:

            if reference.axes_manager.navigation_size != sb_position.axes_manager.navigation_size:  # 1d reference, but
                # parameters are multidimensional
                sb_position_ref = sb_position.inav[0].data
            else:
                sb_position_ref = sb_position_temp

            if reference.axes_manager.navigation_size != sb_size.axes_manager.navigation_size:  # 1d reference, but
                # parameters are multidimensional
                sb_size_ref = np.float64(sb_size.inav[0].data)
            else:
                sb_size_ref = sb_size_temp

            if reference.axes_manager.navigation_size != sb_smoothness.axes_manager.navigation_size:  # 1d reference, but
                # parameters are multidimensional
                sb_smoothness_ref = np.float64(sb_smoothness.inav[0].data)
            else:
                sb_smoothness_ref = sb_smoothness_temp
            #

            wave_reference.map(reconstruct, holo_sampling=self.sampling, sb_size=sb_size_ref,
                               sb_position=sb_position_ref, sb_smoothness=sb_smoothness_ref, output_shape=output_shape,
                               plotting=plotting, show_progressbar=show_progressbar)

        else:
            wave_reference = reference.deepcopy()
            wave_reference.map(reconstruct, holo_sampling=self.sampling, sb_size=sb_size_temp,
                               sb_position=sb_position_temp, sb_smoothness=sb_smoothness_temp,
                               output_shape=output_shape, plotting=plotting, show_progressbar=show_progressbar)

        wave_image = wave_object / wave_reference

        wave_image.set_signal_type('electron_wave')  # New signal is a wave image!

        wave_image.axes_manager.signal_axes[0].scale = self.sampling[0] * self.axes_manager.signal_shape[0] / \
                                                       output_shape[0]
        wave_image.axes_manager.signal_axes[1].scale = self.sampling[1] * self.axes_manager.signal_shape[1] / \
                                                       output_shape[1]

        # Reconstruction parameters are stored in holo_reconstruction_parameters:
        rec_param_dict = OrderedDict([('sb_position', sb_position_temp), ('sb_size', sb_size_temp),
                                      ('sb_units', sb_unit), ('sb_smoothness',
                                                              sb_smoothness_temp)])

        wave_image.metadata.Signal.add_node('holo_reconstruction_parameters')
        wave_image.metadata.Signal.holo_reconstruction_parameters.add_dictionary(rec_param_dict)

        return wave_image
