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


@add_gui_method(toolkey="microscope_parameters_EDS_XRF")
class EDSXRFParametersUI(BaseSetMetadataItems):

    xray_source = t.String(t.Undefined, label='X-ray source material')
    xray_source_energy = t.Float(t.Undefined,
                                 label='X-ray source energy (keV)')
    xray_source_current = t.Float(t.Undefined,
                                  label='X-ray source current (uA)')
    live_time = t.Float(t.Undefined,
                        label='Live time (s)')
    tilt_stage = t.Float(t.Undefined,
                         label='Stage tilt (degree)')
    azimuth_angle = t.Float(t.Undefined,
                            label='Azimuth angle (degree)')
    elevation_angle = t.Float(t.Undefined,
                              label='Elevation angle (degree)')

    mapping = {
        'Acquisition_instrument.XRF.xray_source': 'xray_source',
        'Acquisition_instrument.XRF.xray_source_energy': 'xray_source_energy',
        'Acquisition_instrument.XRF.xray_source_current':
            'xray_source_current',
        'Acquisition_instrument.XRF.Stage.tilt_alpha': 'tilt_stage',
        'Acquisition_instrument.XRF.Detector.EDS.live_time':
            'live_time',
        'Acquisition_instrument.XRF.Detector.EDS.azimuth_angle':
            'azimuth_angle',
        'Acquisition_instrument.XRF.Detector.EDS.elevation_angle':
            'elevation_angle'}


class EDSXRF_mixin:

    _signal_type = "EDS_XRF"

    def __init__(self, *args, **kwards):
        super().__init__(*args, **kwards)
        # Attributes defaults
        self._set_default_param()


    def _set_default_param(self):
        """Set to value to default (defined in preferences)

        """
        mp = self.metadata
        if "Acquisition_instrument.XRF.Stage.tilt_alpha" not in mp:
            mp.set_item(
                "Acquisition_instrument.XRF.Stage.tilt_alpha",
                preferences.EDS.eds_tilt_stage)
        if "Acquisition_instrument.XRF.Detector.EDS.elevation_angle" not in mp:
            mp.set_item(
                "Acquisition_instrument.XRF.Detector.EDS.elevation_angle",
                preferences.EDS.eds_detector_elevation)
        if "Acquisition_instrument.XRF.Detector.EDS.azimuth_angle" not in mp:
            mp.set_item(
                "Acquisition_instrument.XRF.Detector.EDS.azimuth_angle",
                preferences.EDS.eds_detector_azimuth)

    def set_microscope_parameters(self,
                                  xray_source=None,
                                  xray_source_energy=None,
                                  xray_source_current=None,
                                  live_time=None,
                                  tilt_stage=None,
                                  azimuth_angle=None,
                                  elevation_angle=None,
                                  display=True, toolkit=None):
        if set([xray_source, xray_source_energy, xray_source_current,
                live_time, tilt_stage, azimuth_angle, elevation_angle]) == \
                {None}:
            tem_par = EDSXRFParametersUI(self)
            return tem_par.gui(toolkit=toolkit, display=display)
        md = self.metadata

        if xray_source is not None:
            md.set_item("Acquisition_instrument.XRF.xray_source", xray_source)
        if xray_source_energy is not None:
            md.set_item("Acquisition_instrument.XRF.xray_source_energy",
                        xray_source_energy)
        if xray_source_current is not None:
            md.set_item("Acquisition_instrument.XRF.xray_source_current",
                        xray_source_current)
        if live_time is not None:
            md.set_item(
                "Acquisition_instrument.XRF.Detector.EDS.live_time",
                live_time)
        if tilt_stage is not None:
            md.set_item(
                "Acquisition_instrument.XRF.Stage.tilt_alpha",
                tilt_stage)
        if azimuth_angle is not None:
            md.set_item(
                "Acquisition_instrument.XRF.Detector.EDS.azimuth_angle",
                azimuth_angle)
        if elevation_angle is not None:
            md.set_item(
                "Acquisition_instrument.XRF.Detector.EDS.elevation_angle",
                elevation_angle)
    set_microscope_parameters.__doc__ = \
        """
        Set the microscope parameters.

        If no arguments are given, raises an interactive mode to fill
        the values.

        Parameters
        ----------
        xray_source : String
            The material used as the x-ray source
        xray_source_energy : float
            The energy of the x-ray source in keV
        xray_source_current : float
            The current of the x-ray source in uA
        live_time : float
            In seconds
        tilt_stage : float
            In degrees
        azimuth_angle : float
            In degrees
        elevation_angle : float
            In degrees


        Examples
        --------
        >>> s = hs.datasets.example_signals.EDS_XRF_Spectrum()
        >>> print('Default value %s keV' %
        >>>       s.metadata.Acquisition_instrument.XRF.xray_source_energy)
        >>> s.set_microscope_parameters(xray_source_energy=30)
        >>> print('Now set to %s keV' %
        >>>       s.metadata.Acquisition_instrument.XRF.xray_source_energy)
        Default value 20.0 keV
        Now set to 30.0 keV

        """.format(DISPLAY_DT, TOOLKIT_DT)

    def create_model(self, auto_add_lines=True,
                     *args, **kwargs):
        """Create a model for the current XRF data.

        Parameters
        ----------
        auto_add_lines : boolean, default True
            If True, automatically add Gaussians for all X-rays generated in
            the energy range by an element using the edsmodel.add_family_lines
            method.
        dictionary : {None, dict}, optional
            A dictionary to be used to recreate a model. Usually generated
            using :meth:`hyperspy.model.as_dictionary`

        Returns
        -------

        model : `EDSXRFModel` instance.

        """
        from hyperspy.models.edsxrfmodel import EDSXRFModel
        model = EDSXRFModel(self,
                            auto_add_lines=auto_add_lines,
                            *args, **kwargs)
        return model


class EDSXRFSpectrum(EDSXRF_mixin, EDSSpectrum):
    pass


class LazyEDSXRFSpectrum(EDSXRFSpectrum, LazyEDSSpectrum):
    pass
