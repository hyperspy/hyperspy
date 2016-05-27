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


import traits.api as t
import numpy as np

from hyperspy import utils
from hyperspy._signals.eds import EDSSpectrum
from hyperspy.decorators import only_interactive
from hyperspy.gui.eds import TEMParametersUI
from hyperspy.defaults_parser import preferences
import hyperspy.gui.messages as messagesui
from hyperspy.misc.eds import utils as utils_eds


class EDSTEMSpectrum(EDSSpectrum):
    _signal_type = "EDS_TEM"

    def __init__(self, *args, **kwards):
        EDSSpectrum.__init__(self, *args, **kwards)
        # Attributes defaults
        if 'Acquisition_instrument.TEM.Detector.EDS' not in self.metadata:
            if 'Acquisition_instrument.SEM.Detector.EDS' in self.metadata:
                self.metadata.set_item(
                    "Acquisition_instrument.TEM",
                    self.metadata.Acquisition_instrument.SEM)
                del self.metadata.Acquisition_instrument.SEM
        self._set_default_param()

    def _set_default_param(self):
        """Set to value to default (defined in preferences)
        """

        mp = self.metadata
        mp.Signal.signal_type = 'EDS_TEM'

        mp = self.metadata
        if "Acquisition_instrument.TEM.tilt_stage" not in mp:
            mp.set_item(
                "Acquisition_instrument.TEM.tilt_stage",
                preferences.EDS.eds_tilt_stage)
        if "Acquisition_instrument.TEM.Detector.EDS.elevation_angle" not in mp:
            mp.set_item(
                "Acquisition_instrument.TEM.Detector.EDS.elevation_angle",
                preferences.EDS.eds_detector_elevation)
        if "Acquisition_instrument.TEM.Detector.EDS.energy_resolution_MnKa"\
                not in mp:
            mp.set_item("Acquisition_instrument.TEM.Detector.EDS." +
                        "energy_resolution_MnKa",
                        preferences.EDS.eds_mn_ka)
        if "Acquisition_instrument.TEM.Detector.EDS.azimuth_angle" not in mp:
            mp.set_item(
                "Acquisition_instrument.TEM.Detector.EDS.azimuth_angle",
                preferences.EDS.eds_detector_azimuth)

    def set_microscope_parameters(self,
                                  beam_energy=None,
                                  live_time=None,
                                  tilt_stage=None,
                                  azimuth_angle=None,
                                  elevation_angle=None,
                                  energy_resolution_MnKa=None):
        """Set the microscope parameters.

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

        Examples
        --------
        >>> s = hs.datasets.example_signals.EDS_TEM_Spectrum()
        >>> print(s.metadata.Acquisition_instrument.
        >>>       TEM.Detector.EDS.energy_resolution_MnKa)
        >>> s.set_microscope_parameters(energy_resolution_MnKa=135.)
        >>> print(s.metadata.Acquisition_instrument.
        >>>       TEM.Detector.EDS.energy_resolution_MnKa)
        133.312296
        135.0

        """
        md = self.metadata

        if beam_energy is not None:
            md.set_item("Acquisition_instrument.TEM.beam_energy ", beam_energy)
        if live_time is not None:
            md.set_item(
                "Acquisition_instrument.TEM.Detector.EDS.live_time",
                live_time)
        if tilt_stage is not None:
            md.set_item("Acquisition_instrument.TEM.tilt_stage", tilt_stage)
        if azimuth_angle is not None:
            md.set_item(
                "Acquisition_instrument.TEM.Detector.EDS.azimuth_angle",
                azimuth_angle)
        if elevation_angle is not None:
            md.set_item(
                "Acquisition_instrument.TEM.Detector.EDS.elevation_angle",
                elevation_angle)
        if energy_resolution_MnKa is not None:
            md.set_item(
                "Acquisition_instrument.TEM.Detector.EDS." +
                "energy_resolution_MnKa",
                energy_resolution_MnKa)

        if set([beam_energy, live_time, tilt_stage, azimuth_angle,
                elevation_angle, energy_resolution_MnKa]) == {None}:
            self._are_microscope_parameters_missing()

    @only_interactive
    def _set_microscope_parameters(self):
        tem_par = TEMParametersUI()
        mapping = {
            'Acquisition_instrument.TEM.beam_energy':
            'tem_par.beam_energy',
            'Acquisition_instrument.TEM.tilt_stage':
            'tem_par.tilt_stage',
            'Acquisition_instrument.TEM.Detector.EDS.live_time':
            'tem_par.live_time',
            'Acquisition_instrument.TEM.Detector.EDS.azimuth_angle':
            'tem_par.azimuth_angle',
            'Acquisition_instrument.TEM.Detector.EDS.elevation_angle':
            'tem_par.elevation_angle',
            'Acquisition_instrument.TEM.Detector.EDS.energy_resolution_MnKa':
            'tem_par.energy_resolution_MnKa', }
        for key, value in mapping.items():
            if self.metadata.has_item(key):
                exec('%s = self.metadata.%s' % (value, key))
        tem_par.edit_traits()

        mapping = {
            'Acquisition_instrument.TEM.beam_energy':
            tem_par.beam_energy,
            'Acquisition_instrument.TEM.tilt_stage':
            tem_par.tilt_stage,
            'Acquisition_instrument.TEM.Detector.EDS.live_time':
            tem_par.live_time,
            'Acquisition_instrument.TEM.Detector.EDS.azimuth_angle':
            tem_par.azimuth_angle,
            'Acquisition_instrument.TEM.Detector.EDS.elevation_angle':
            tem_par.elevation_angle,
            'Acquisition_instrument.TEM.Detector.EDS.energy_resolution_MnKa':
            tem_par.energy_resolution_MnKa, }

        for key, value in mapping.items():
            if value != t.Undefined:
                self.metadata.set_item(key, value)
        self._are_microscope_parameters_missing()

    def _are_microscope_parameters_missing(self):
        """Check if the EDS parameters necessary for quantification
        are defined in metadata. Raise in interactive mode
         an UI item to fill or cahnge the values"""
        must_exist = (
            'Acquisition_instrument.TEM.beam_energy',
            'Acquisition_instrument.TEM.Detector.EDS.live_time',)

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
        >>> ref = hs.datasets.example_signals.EDS_TEM_Spectrum()
        >>> s = hs.signals.EDSTEMSpectrum(
        >>>     hs.datasets.example_signals.EDS_TEM_Spectrum().data)
        >>> print s.axes_manager[0].scale
        >>> s.get_calibration_from(ref)
        >>> print s.axes_manager[0].scale
        1.0
        0.020028

        """

        self.original_metadata = ref.original_metadata.deepcopy()
        # Setup the axes_manager
        ax_m = self.axes_manager.signal_axes[0]
        ax_ref = ref.axes_manager.signal_axes[0]
        ax_m.scale = ax_ref.scale
        ax_m.units = ax_ref.units
        ax_m.offset = ax_ref.offset

        # Setup metadata
        if 'Acquisition_instrument.TEM' in ref.metadata:
            mp_ref = ref.metadata.Acquisition_instrument.TEM
        elif 'Acquisition_instrument.SEM' in ref.metadata:
            mp_ref = ref.metadata.Acquisition_instrument.SEM
        else:
            raise ValueError("The reference has no metadata." +
                             "Acquisition_instrument.TEM" +
                             "\n nor metadata.Acquisition_instrument.SEM ")

        mp = self.metadata
        mp.Acquisition_instrument.TEM = mp_ref.deepcopy()
        if mp_ref.has_item("Detector.EDS.live_time"):
            mp.Acquisition_instrument.TEM.Detector.EDS.live_time = \
                mp_ref.Detector.EDS.live_time / nb_pix

    def quantification(self,
                       intensities,
                       kfactors,
                       composition_units='weight',
                       navigation_mask=1.0,
                       closing=True,
                       plot_result=False,
                       **kwargs):
        """
        Quantification of intensities to return elemental composition

        Method: Cliff-Lorimer

        Parameters
        ----------
        intensities: list of signal
            the intensitiy for each X-ray lines.
        kfactors: list of float
            The list of kfactor in same order as intensities. Note that
            intensities provided by hyperspy are sorted by the aplhabetical
            order of the X-ray lines. eg. kfactors =[0.982, 1.32, 1.60] for
            ['Al_Ka','Cr_Ka', 'Ni_Ka'].
        composition_units: 'weight' or 'atomic'
            Quantification returns weight percent. By choosing 'atomic', the
            return composition is in atomic percent.
        navigation_mask : None or float or signal
            The navigation locations marked as True are not used in the
            quantification. If int is given the vacuum_mask method is used to
            generate a mask with the int value as threhsold.
            Else provides a signal with the navigation shape.
        closing: bool
            If true, applied a morphologic closing to the mask obtained by
            vacuum_mask.
        plot_result : bool
            If True, plot the calculated composition. If the current
            object is a single spectrum it prints the result instead.
        kwargs
            The extra keyword arguments are passed to plot.

        Return
        ------
        A list of quantified elemental maps (signal) giving the composition of
        the sample in weight or atomic percent.

        Examples
        --------
        >>> s = hs.datasets.example_signals.EDS_TEM_Spectrum()
        >>> s.add_lines()
        >>> kfactors = [1.450226, 5.075602] #For Fe Ka and Pt La
        >>> bw = s.estimate_background_windows(line_width=[5.0, 2.0])
        >>> s.plot(background_windows=bw)
        >>> intensities = s.get_lines_intensity(background_windows=bw)
        >>> res = s.quantification(intensities, kfactors, plot_result=True,
        >>>                        composition_units='atomic')
        Fe (Fe_Ka): Composition = 15.41 atomic percent
        Pt (Pt_La): Composition = 84.59 atomic percent

        See also
        --------
        vacuum_mask
        """
        if isinstance(navigation_mask, float):
            navigation_mask = self.vacuum_mask(navigation_mask, closing).data
        elif navigation_mask is not None:
            navigation_mask = navigation_mask.data
        xray_lines = self.metadata.Sample.xray_lines
        composition = utils.stack(intensities)
        composition.data = utils_eds.quantification_cliff_lorimer(
            composition.data, kfactors=kfactors,
            mask=navigation_mask) * 100.
        composition = composition.split()
        if composition_units == 'atomic':
            composition = utils.material.weight_to_atomic(composition)
        for i, xray_line in enumerate(xray_lines):
            element, line = utils_eds._get_element_and_line(xray_line)
            composition[i].metadata.General.title = composition_units + \
                ' percent of ' + element
            composition[i].metadata.set_item("Sample.elements", ([element]))
            composition[i].metadata.set_item(
                "Sample.xray_lines", ([xray_line]))
            if plot_result and \
                    composition[i].axes_manager.signal_size == 1:
                print("%s (%s): Composition = %.2f %s percent"
                      % (element, xray_line, composition[i].data,
                         composition_units))
        if plot_result and composition[i].axes_manager.signal_size != 1:
            utils.plot.plot_signals(composition, **kwargs)
        return composition

    def vacuum_mask(self, threshold=1.0, closing=True, opening=False):
        """
        Generate mask of the vacuum region

        Parameters
        ----------
        threshold: float
            For a given pixel, maximum value in the energy axis below which the
            pixel is considered as vacuum.
        closing: bool
            If true, applied a morphologic closing to the mask
        opnening: bool
            If true, applied a morphologic opening to the mask

        Examples
        --------
        >>> # Simulate a spectrum image with vacuum region
        >>> s = hs.datasets.example_signals.EDS_TEM_Spectrum()
        >>> s_vac = hs.signals.Simulation(np.ones_like(s.data, dtype=float))*0.005
        >>> s_vac.add_poissonian_noise()
        >>> si = hs.stack([s]*3 + [s_vac])
        >>> si.vacuum_mask().data
        array([False, False, False,  True], dtype=bool)

        Return
        ------
        mask: signal
            The mask of the region
        """
        from scipy.ndimage.morphology import binary_dilation, binary_erosion
        mask = (self.max(-1) <= threshold)
        if closing:
            mask.data = binary_dilation(mask.data, border_value=0)
            mask.data = binary_erosion(mask.data, border_value=1)
        if opening:
            mask.data = binary_erosion(mask.data, border_value=1)
            mask.data = binary_dilation(mask.data, border_value=0)
        return mask

    def decomposition(self,
                      normalize_poissonian_noise=True,
                      navigation_mask=1.0,
                      closing=True,
                      *args,
                      **kwargs):
        """
        Decomposition with a choice of algorithms

        The results are stored in self.learning_results

        Parameters
        ----------
        normalize_poissonian_noise : bool
            If True, scale the SI to normalize Poissonian noise
        navigation_mask : None or float or boolean numpy array
            The navigation locations marked as True are not used in the
            decompostion. If float is given the vacuum_mask method is used to
            generate a mask with the float value as threshold.
        closing: bool
            If true, applied a morphologic closing to the maks obtained by
            vacuum_mask.
        algorithm : 'svd' | 'fast_svd' | 'mlpca' | 'fast_mlpca' | 'nmf' |
            'sparse_pca' | 'mini_batch_sparse_pca'
        output_dimension : None or int
            number of components to keep/calculate
        centre : None | 'variables' | 'trials'
            If None no centring is applied. If 'variable' the centring will be
            performed in the variable axis. If 'trials', the centring will be
            performed in the 'trials' axis. It only has effect when using the
            svd or fast_svd algorithms
        auto_transpose : bool
            If True, automatically transposes the data to boost performance.
            Only has effect when using the svd of fast_svd algorithms.
        signal_mask : boolean numpy array
            The signal locations marked as True are not used in the
            decomposition.
        var_array : numpy array
            Array of variance for the maximum likelihood PCA algorithm
        var_func : function or numpy array
            If function, it will apply it to the dataset to obtain the
            var_array. Alternatively, it can a an array with the coefficients
            of a polynomial.
        polyfit :
        reproject : None | signal | navigation | both
            If not None, the results of the decomposition will be projected in
            the selected masked area.

        Examples
        --------
        >>> s = hs.datasets.example_signals.EDS_TEM_Spectrum()
        >>> si = hs.stack([s]*3)
        >>> si.change_dtype(float)
        >>> si.decomposition()

        See also
        --------
        vacuum_mask
        """
        if isinstance(navigation_mask, float):
            navigation_mask = self.vacuum_mask(navigation_mask, closing).data
        super(EDSSpectrum, self).decomposition(
            normalize_poissonian_noise=normalize_poissonian_noise,
            navigation_mask=navigation_mask, *args, **kwargs)
        self.learning_results.loadings = np.nan_to_num(
            self.learning_results.loadings)
