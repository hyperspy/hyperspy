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

import traits.api as t

from hyperspy._signals.eds import (EDSSpectrum, LazyEDSSpectrum)
from hyperspy.defaults_parser import preferences
from hyperspy.ui_registry import add_gui_method, DISPLAY_DT, TOOLKIT_DT
from hyperspy.signal import BaseSetMetadataItems


_logger = logging.getLogger(__name__)


@add_gui_method(toolkey="microscope_parameters_EDS_SEM")
class EDSSEMParametersUI(BaseSetMetadataItems):

    beam_energy = t.Float(t.Undefined,
                          label='Beam energy (keV)')
    live_time = t.Float(t.Undefined,
                        label='Live time (s)')
    tilt_stage = t.Float(t.Undefined,
                         label='Stage tilt (degree)')
    azimuth_angle = t.Float(t.Undefined,
                            label='Azimuth angle (degree)')
    elevation_angle = t.Float(t.Undefined,
                              label='Elevation angle (degree)')
    energy_resolution_MnKa = t.Float(t.Undefined,
                                     label='Energy resolution MnKa (eV)')
    mapping = {
        'Acquisition_instrument.SEM.beam_energy': 'beam_energy',
        'Acquisition_instrument.TEM.Stage.tilt_alpha': 'tilt_stage',
        'Acquisition_instrument.SEM.Detector.EDS.live_time':
        'live_time',
        'Acquisition_instrument.SEM.Detector.EDS.azimuth_angle':
        'azimuth_angle',
        'Acquisition_instrument.SEM.Detector.EDS.elevation_angle':
        'elevation_angle',
        'Acquisition_instrument.SEM.Detector.EDS.energy_resolution_MnKa':
        'energy_resolution_MnKa', }


class EDSSEM_mixin:

    _signal_type = "EDS_SEM"

    def __init__(self, *args, **kwards):
        super().__init__(*args, **kwards)
        # Attributes defaults
        if 'Acquisition_instrument.SEM.Detector.EDS' not in self.metadata:
            if 'Acquisition_instrument.TEM' in self.metadata:
                self.metadata.set_item(
                    "Acquisition_instrument.SEM",
                    self.metadata.Acquisition_instrument.TEM)
                del self.metadata.Acquisition_instrument.TEM
        self._set_default_param()

    def get_calibration_from(self, ref, nb_pix=1):
        """Copy the calibration and all metadata of a reference.

        Primary use: To add a calibration to ripple file from INCA
        software

        Parameters
        ----------
        ref : signal
            The reference contains the calibration in its
            metadata
        nb_pix : int
            The live time (real time corrected from the "dead time")
            is divided by the number of pixel (spectrums), giving an
            average live time.

        Examples
        --------
        >>> ref = hs.datasets.example_signals.EDS_SEM_Spectrum()
        >>> s = hs.signals.EDSSEMSpectrum(
        >>>     hs.datasets.example_signals.EDS_SEM_Spectrum().data)
        >>> print(s.axes_manager[0].scale)
        >>> s.get_calibration_from(ref)
        >>> print(s.axes_manager[0].scale)
        1.0
        0.01

        """

        self.original_metadata = ref.original_metadata.deepcopy()
        # Setup the axes_manager
        ax_m = self.axes_manager.signal_axes[0]
        ax_ref = ref.axes_manager.signal_axes[0]
        ax_m.scale = ax_ref.scale
        ax_m.units = ax_ref.units
        ax_m.offset = ax_ref.offset

        # Setup metadata
        if 'Acquisition_instrument.SEM' in ref.metadata:
            mp_ref = ref.metadata.Acquisition_instrument.SEM
        elif 'Acquisition_instrument.TEM' in ref.metadata:
            mp_ref = ref.metadata.Acquisition_instrument.TEM
        else:
            raise ValueError(
                "The reference has no metadata.Acquisition_instrument.TEM"
                "\n nor metadata.Acquisition_instrument.SEM ")

        mp = self.metadata

        mp.Acquisition_instrument.SEM = mp_ref.deepcopy()

        if hasattr(mp_ref.Detector.EDS, 'live_time'):
            mp.Acquisition_instrument.SEM.Detector.EDS.live_time = \
                mp_ref.Detector.EDS.live_time / nb_pix

    def _load_from_TEM_param(self):
        """Transfer metadata.Acquisition_instrument.TEM to
        metadata.Acquisition_instrument.SEM

        """

        mp = self.metadata
        if mp.has_item('Acquisition_instrument.SEM') is False:
            mp.add_node('Acquisition_instrument.SEM')
        if mp.has_item('Acquisition_instrument.SEM.Detector.EDS') is False:
            mp.Acquisition_instrument.SEM.add_node('EDS')
        mp.Signal.signal_type = "EDS_SEM"

        # Transfer
        if 'Acquisition_instrument.TEM' in mp:
            mp.Acquisition_instrument.SEM = mp.Acquisition_instrument.TEM
            del mp.Acquisition_instrument.TEM

    def _set_default_param(self):
        """Set to value to default (defined in preferences)

        """
        mp = self.metadata
        if "Acquisition_instrument.SEM.Stage.tilt_alpha" not in mp:
            mp.set_item(
                "Acquisition_instrument.SEM.Stage.tilt_alpha",
                preferences.EDS.eds_tilt_stage)
        if "Acquisition_instrument.SEM.Detector.EDS.elevation_angle" not in mp:
            mp.set_item(
                "Acquisition_instrument.SEM.Detector.EDS.elevation_angle",
                preferences.EDS.eds_detector_elevation)
        if "Acquisition_instrument.SEM.Detector.EDS.energy_resolution_MnKa" \
                not in mp:
            mp.set_item(
                "Acquisition_instrument.SEM.Detector.EDS."
                "energy_resolution_MnKa",
                preferences.EDS.eds_mn_ka)
        if "Acquisition_instrument.SEM.Detector.EDS.azimuth_angle" not in mp:
            mp.set_item(
                "Acquisition_instrument.SEM.Detector.EDS.azimuth_angle",
                preferences.EDS.eds_detector_azimuth)

    def set_microscope_parameters(self,
                                  beam_energy=None,
                                  live_time=None,
                                  tilt_stage=None,
                                  azimuth_angle=None,
                                  elevation_angle=None,
                                  energy_resolution_MnKa=None,
                                  display=True, toolkit=None):
        if set([beam_energy, live_time, tilt_stage, azimuth_angle,
                elevation_angle, energy_resolution_MnKa]) == {None}:
            tem_par = EDSSEMParametersUI(self)
            return tem_par.gui(toolkit=toolkit, display=display)
        md = self.metadata

        if beam_energy is not None:
            md.set_item("Acquisition_instrument.SEM.beam_energy", beam_energy)
        if live_time is not None:
            md.set_item(
                "Acquisition_instrument.SEM.Detector.EDS.live_time",
                live_time)
        if tilt_stage is not None:
            md.set_item(
                "Acquisition_instrument.SEM.Stage.tilt_alpha",
                tilt_stage)
        if azimuth_angle is not None:
            md.set_item(
                "Acquisition_instrument.SEM.Detector.EDS.azimuth_angle",
                azimuth_angle)
        if elevation_angle is not None:
            md.set_item(
                "Acquisition_instrument.SEM.Detector.EDS.elevation_angle",
                elevation_angle)
        if energy_resolution_MnKa is not None:
            md.set_item(
                "Acquisition_instrument.SEM.Detector.EDS."
                "energy_resolution_MnKa",
                energy_resolution_MnKa)
    set_microscope_parameters.__doc__ = \
        """
        Set the microscope parameters.

        If no arguments are given, raises an interactive mode to fill
        the values.

        Parameters
        ----------
        beam_energy: float
            The energy of the electron beam in keV
        live_time : float
            In second
        tilt_stage : float
            In degree
        azimuth_angle : float
            In degree
        elevation_angle : float
            In degree
        energy_resolution_MnKa : float
            In eV
        {}
        {}

        Examples
        --------
        >>> s = hs.datasets.example_signals.EDS_SEM_Spectrum()
        >>> print('Default value %s eV' %
        >>>       s.metadata.Acquisition_instrument.
        >>>       SEM.Detector.EDS.energy_resolution_MnKa)
        >>> s.set_microscope_parameters(energy_resolution_MnKa=135.)
        >>> print('Now set to %s eV' %
        >>>       s.metadata.Acquisition_instrument.
        >>>       SEM.Detector.EDS.energy_resolution_MnKa)
        Default value 130.0 eV
        Now set to 135.0 eV

        """.format(DISPLAY_DT, TOOLKIT_DT)

    def _are_microscope_parameters_missing(self):
        """Check if the EDS parameters necessary for quantification
        are defined in metadata. If not, in interactive mode
        raises an UI item to fill the values

        """
        must_exist = (
            'Acquisition_instrument.SEM.beam_energy',
            'Acquisition_instrument.SEM.Detector.EDS.live_time', )

        missing_parameters = []
        for item in must_exist:
            exists = self.metadata.has_item(item)
            if exists is False:
                missing_parameters.append(item)
        if missing_parameters:
            _logger.info("Missing parameters {}".format(missing_parameters))
            return True
        else:
            return False

    def create_model(self, auto_background=True, auto_add_lines=True,
                     *args, **kwargs):
        """Create a model for the current SEM EDS data.

        Parameters
        ----------
        auto_background : boolean, default True
            If True, adds automatically a polynomial order 6 to the model,
            using the edsmodel.add_polynomial_background method.
        auto_add_lines : boolean, default True
            If True, automatically add Gaussians for all X-rays generated in
            the energy range by an element using the edsmodel.add_family_lines
            method.
        dictionary : {None, dict}, optional
            A dictionary to be used to recreate a model. Usually generated
            using :meth:`hyperspy.model.as_dictionary`

        Returns
        -------

        model : `EDSSEMModel` instance.

        """
        from hyperspy.models.edssemmodel import EDSSEMModel
        model = EDSSEMModel(self,
                            auto_background=auto_background,
                            auto_add_lines=auto_add_lines,
                            *args, **kwargs)
        return model


class EDSSEMSpectrum(EDSSEM_mixin, EDSSpectrum):
    pass


class LazyEDSSEMSpectrum(EDSSEMSpectrum, LazyEDSSpectrum):
    pass
