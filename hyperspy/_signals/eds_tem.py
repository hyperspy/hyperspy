# -*- coding: utf-8 -*-
# Copyright 2007-2020 The HyperSpy developers
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


import warnings
import logging

import traits.api as t
import numpy as np
from scipy import constants
import pint

from hyperspy.signal import BaseSetMetadataItems
from hyperspy import utils
from hyperspy._signals.eds import (EDSSpectrum, LazyEDSSpectrum)
from hyperspy.defaults_parser import preferences
from hyperspy.ui_registry import add_gui_method, DISPLAY_DT, TOOLKIT_DT
from hyperspy.misc.eds import utils as utils_eds
from hyperspy.misc.elements import elements as elements_db
from hyperspy.misc.utils import isiterable
from hyperspy.external.progressbar import progressbar
from hyperspy.axes import DataAxis

_logger = logging.getLogger(__name__)


@add_gui_method(toolkey="hyperspy.microscope_parameters_EDS_TEM")
class EDSTEMParametersUI(BaseSetMetadataItems):
    beam_energy = t.Float(t.Undefined,
                          label='Beam energy (keV)')
    real_time = t.Float(t.Undefined,
                        label='Real time (s)')
    tilt_stage = t.Float(t.Undefined,
                         label='Stage tilt (degree)')
    live_time = t.Float(t.Undefined,
                        label='Live time (s)')
    probe_area = t.Float(t.Undefined,
                         label='Beam/probe area (nm²)')
    azimuth_angle = t.Float(t.Undefined,
                            label='Azimuth angle (degree)')
    elevation_angle = t.Float(t.Undefined,
                              label='Elevation angle (degree)')
    energy_resolution_MnKa = t.Float(t.Undefined,
                                     label='Energy resolution MnKa (eV)')
    beam_current = t.Float(t.Undefined,
                           label='Beam current (nA)')
    mapping = {
        'Acquisition_instrument.TEM.beam_energy': 'beam_energy',
        'Acquisition_instrument.TEM.Stage.tilt_alpha': 'tilt_stage',
        'Acquisition_instrument.TEM.Detector.EDS.live_time': 'live_time',
        'Acquisition_instrument.TEM.Detector.EDS.azimuth_angle':
        'azimuth_angle',
        'Acquisition_instrument.TEM.Detector.EDS.elevation_angle':
        'elevation_angle',
        'Acquisition_instrument.TEM.Detector.EDS.energy_resolution_MnKa':
        'energy_resolution_MnKa',
        'Acquisition_instrument.TEM.beam_current':
        'beam_current',
        'Acquisition_instrument.TEM.probe_area':
        'probe_area',
        'Acquisition_instrument.TEM.Detector.EDS.real_time':
        'real_time', }


class EDSTEM_mixin:

    _signal_type = "EDS_TEM"

    def __init__(self, *args, **kwards):
        super().__init__(*args, **kwards)
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
        mp.Signal.signal_type = "EDS_TEM"

        mp = self.metadata
        if "Acquisition_instrument.TEM.Stage.tilt_alpha" not in mp:
            mp.set_item(
                "Acquisition_instrument.TEM.Stage.tilt_alpha",
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
                                  energy_resolution_MnKa=None,
                                  beam_current=None,
                                  probe_area=None,
                                  real_time=None,
                                  display=True,
                                  toolkit=None):
        if set([beam_energy, live_time, tilt_stage, azimuth_angle,
                elevation_angle, energy_resolution_MnKa, beam_current,
                probe_area, real_time]) == {None}:
            tem_par = EDSTEMParametersUI(self)
            return tem_par.gui(display=display, toolkit=toolkit)
        md = self.metadata

        if beam_energy is not None:
            md.set_item("Acquisition_instrument.TEM.beam_energy ", beam_energy)
        if live_time is not None:
            md.set_item(
                "Acquisition_instrument.TEM.Detector.EDS.live_time",
                live_time)
        if tilt_stage is not None:
            md.set_item(
                "Acquisition_instrument.TEM.Stage.tilt_alpha",
                tilt_stage)
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
        if beam_current is not None:
            md.set_item(
                "Acquisition_instrument.TEM.beam_current",
                beam_current)
        if probe_area is not None:
            md.set_item(
                "Acquisition_instrument.TEM.probe_area",
                probe_area)
        if real_time is not None:
            md.set_item(
                "Acquisition_instrument.TEM.Detector.EDS.real_time",
                real_time)

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
            In seconds
        tilt_stage : float
            In degree
        azimuth_angle : float
            In degree
        elevation_angle : float
            In degree
        energy_resolution_MnKa : float
            In eV
        beam_current: float
            In nA
        probe_area: float
            In nm²
        real_time: float
            In seconds
        {}
        {}

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

        """.format(DISPLAY_DT, TOOLKIT_DT)

    def _are_microscope_parameters_missing(self):
        """Check if the EDS parameters necessary for quantification are
        defined in metadata."""
        must_exist = (
            'Acquisition_instrument.TEM.beam_energy',
            'Acquisition_instrument.TEM.Detector.EDS.live_time',)

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
        >>> print(s.axes_manager[0].scale)
        >>> s.get_calibration_from(ref)
        >>> print(s.axes_manager[0].scale)
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
            raise ValueError("The reference has no metadata "
                             "'Acquisition_instrument.TEM '"
                             "or 'metadata.Acquisition_instrument.SEM'.")

        mp = self.metadata
        mp.Acquisition_instrument.TEM = mp_ref.deepcopy()
        if mp_ref.has_item("Detector.EDS.live_time"):
            mp.Acquisition_instrument.TEM.Detector.EDS.live_time = \
                mp_ref.Detector.EDS.live_time / nb_pix

    def quantification(self,
                       intensities,
                       method,
                       factors,
                       composition_units='atomic',
                       absorption_correction=False,
                       take_off_angle='auto',
                       thickness='auto',
                       convergence_criterion=0.5,
                       navigation_mask=1.0,
                       closing=True,
                       plot_result=False,
                       probe_area='auto',
                       max_iterations=30,
                       show_progressbar=None,
                       **kwargs):
        """
        Absorption corrected quantification using Cliff-Lorimer, the zeta-factor
        method, or ionization cross sections. The function iterates through
        quantification function until two successive interations don't change
        the final composition by a defined percentage critera (0.5% by default).

        Parameters
        ----------
        intensities: list of signal
            the intensitiy for each X-ray lines.
        method: {'CL', 'zeta', 'cross_section'}
            Set the quantification method: Cliff-Lorimer, zeta-factor, or
            ionization cross sections.
        factors: list of float
            The list of kfactors, zeta-factors or cross sections in same order
            as intensities. Note that intensities provided by Hyperspy are
            sorted by the alphabetical order of the X-ray lines.
            eg. factors =[0.982, 1.32, 1.60] for ['Al_Ka', 'Cr_Ka', 'Ni_Ka'].
        composition_units: {'atomic', 'weight'}
            The quantification returns the composition in 'atomic' percent by
            default, but can also return weight percent if specified.
        absorption_correction: bool
            Specify whether or not an absorption correction should be applied.
            'False' by default so absorption will not be applied unless
            specfied.
        take_off_angle : {'auto'}
            The angle between the sample surface and the vector along which
            X-rays travel to reach the centre of the detector.
        thickness: {'auto'}
            thickness in nm (can be a single value or
            have the same navigation dimension as the signal).
            NB: Must be specified for 'CL' method. For 'zeta' or 'cross_section'
            methods, first quantification step provides a mass_thickness
            internally during quantification.
        convergence_criterion: The convergence criterium defined as the percentage
            difference between 2 successive iterations. 0.5% by default.
        navigation_mask : None or float or signal
            The navigation locations marked as True are not used in the
            quantification. If float is given the vacuum_mask method is used to
            generate a mask with the float value as threhsold.
            Else provides a signal with the navigation shape. Only for the
            'Cliff-Lorimer' method.
        closing: bool
            If true, applied a morphologic closing to the mask obtained by
            vacuum_mask.
        plot_result : bool
            If True, plot the calculated composition. If the current
            object is a single spectrum it prints the result instead.
        probe_area = {'auto'}
            This allows the user to specify the probe_area for interaction with
            the sample needed specifically for the cross_section method of
            quantification. When left as 'auto' the pixel area is used,
            calculated from the navigation axes information.
        max_iterations : int
            An upper limit to the number of calculations for absorption correction.
        kwargs
            The extra keyword arguments are passed to plot.

        Returns
        -------
        A list of quantified elemental maps (signal) giving the composition of
        the sample in weight or atomic percent with absorption correciton taken
        into account based on the sample thickness estimate provided.

        If the method is 'zeta' this function also returns the mass thickness
        profile for the data.

        If the method is 'cross_section' this function also returns the atom
        counts for each element.

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
            navigation_mask = self.vacuum_mask(navigation_mask, closing)

        xray_lines = [intensity.metadata.Sample.xray_lines[0] for intensity in intensities]
        it = 0
        if absorption_correction:
            if show_progressbar is None:  # pragma: no cover
                show_progressbar = preferences.General.show_progressbar
            if show_progressbar:
                pbar = progressbar(total=None,
                                   desc='Absorption correction calculation')

        composition = utils.stack(intensities, lazy=False,
                                  show_progressbar=False)

        if take_off_angle == 'auto':
            toa = self.get_take_off_angle()
        else:
            toa = take_off_angle

        #determining illumination area for cross sections quantification.
        if method == 'cross_section':
            if probe_area == 'auto':
                parameters = self.metadata.Acquisition_instrument.TEM
                if probe_area in parameters:
                    probe_area = parameters.TEM.probe_area
                else:
                    probe_area = self.get_probe_area(
                        navigation_axes=self.axes_manager.navigation_axes)

        int_stack = utils.stack(intensities, lazy=False,
                                show_progressbar=False)
        comp_old = np.zeros_like(int_stack.data)

        abs_corr_factor = None # initial

        if method == 'CL':
            quantification_method = utils_eds.quantification_cliff_lorimer
            kwargs = {"intensities" : int_stack.data,
                      "kfactors" : factors,
                      "absorption_correction" : abs_corr_factor,
                      "mask": navigation_mask}

        elif method == 'zeta':
            quantification_method = utils_eds.quantification_zeta_factor
            kwargs = {"intensities" : int_stack.data,
                      "zfactors" : factors,
                      "dose" : self._get_dose(method),
                      "absorption_correction" : abs_corr_factor}

        elif method =='cross_section':
            quantification_method = utils_eds.quantification_cross_section
            kwargs = {"intensities" : int_stack.data,
                      "cross_sections" : factors,
                      "dose" : self._get_dose(method, **kwargs),
                      "absorption_correction" : abs_corr_factor}

        else:
            raise ValueError('Please specify method for quantification, '
                             'as "CL", "zeta" or "cross_section".')

        while True:
            results = quantification_method(**kwargs)

            if method == 'CL':
                composition.data = results * 100.
                if absorption_correction:
                    if thickness is not None:
                        mass_thickness = intensities[0].deepcopy()
                        mass_thickness.data = self.CL_get_mass_thickness(
                            composition.split(),
                            thickness
                            )
                        mass_thickness.metadata.General.title = 'Mass thickness'
                    else:
                        raise ValueError(
                            'Thickness is required for absorption correction '
                            'with k-factor method. Results will contain no '
                            'correction for absorption.'
                        )

            elif method == 'zeta':
                composition.data = results[0] * 100
                mass_thickness = intensities[0].deepcopy()
                mass_thickness.data = results[1]

            else:
                composition.data = results[0] * 100.
                number_of_atoms = composition._deepcopy_with_new_data(results[1])

            if method == 'cross_section':
                abs_corr_factor = utils_eds.get_abs_corr_cross_section(composition.split(),
                                                       number_of_atoms.split(),
                                                       toa,
                                                       probe_area)
                kwargs["absorption_correction"] = abs_corr_factor
            else:
                if absorption_correction:
                    abs_corr_factor = utils_eds.get_abs_corr_zeta(composition.split(),
                                                       mass_thickness,
                                                       toa)
                    kwargs["absorption_correction"] = abs_corr_factor

            res_max = np.max(composition.data - comp_old)
            comp_old = composition.data

            if absorption_correction and show_progressbar:
                pbar.update(1)
            it += 1
            if not absorption_correction or abs(res_max) < convergence_criterion:
                break
            elif it >= max_iterations:
                raise Exception('Absorption correction failed as solution '
                                f'did not converge after {max_iterations} '
                                'iterations')

        if method == 'cross_section':
            number_of_atoms = composition._deepcopy_with_new_data(results[1])
            number_of_atoms = number_of_atoms.split()
            composition = composition.split()
        else:
            composition = composition.split()

        #convert ouput units to selection as required.
        if composition_units == 'atomic':
            if method != 'cross_section':
                composition = utils.material.weight_to_atomic(composition)
        else:
            if method == 'cross_section':
                composition = utils.material.atomic_to_weight(composition)

        #Label each of the elemental maps in the image stacks for composition.
        for i, xray_line in enumerate(xray_lines):
            element, line = utils_eds._get_element_and_line(xray_line)
            composition[i].metadata.General.title = composition_units + \
                ' percent of ' + element
            composition[i].metadata.set_item("Sample.elements", ([element]))
            composition[i].metadata.set_item(
                "Sample.xray_lines", ([xray_line]))
            if plot_result and composition[i].axes_manager.navigation_size == 1:
                c = composition[i].data
                print(f"{element} ({xray_line}): Composition = {c:.2f} percent")
        #For the cross section method this is repeated for the number of atom maps
        if method == 'cross_section':
            for i, xray_line in enumerate(xray_lines):
                element, line = utils_eds._get_element_and_line(xray_line)
                number_of_atoms[i].metadata.General.title = \
                    'atom counts of ' + element
                number_of_atoms[i].metadata.set_item("Sample.elements",
                                                     ([element]))
                number_of_atoms[i].metadata.set_item(
                    "Sample.xray_lines", ([xray_line]))
        if plot_result and composition[i].axes_manager.navigation_size != 1:
            utils.plot.plot_signals(composition, **kwargs)

        if absorption_correction:
            _logger.info(f'Conversion found after {it} interations.')

        if method == 'zeta':
            mass_thickness.metadata.General.title = 'Mass thickness'
            self.metadata.set_item("Sample.mass_thickness", mass_thickness)
            return composition, mass_thickness
        elif method == 'cross_section':
            return composition, number_of_atoms
        elif method == 'CL':
            if absorption_correction:
                mass_thickness.metadata.General.title = 'Mass thickness'
                return composition, mass_thickness
            else:
                return composition
        else:
            raise ValueError('Please specify method for quantification, as '
                             '"CL", "zeta" or "cross_section"')


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

        Returns
        -------
        mask: signal
            The mask of the region

        Examples
        --------
        >>> # Simulate a spectrum image with vacuum region
        >>> s = hs.datasets.example_signals.EDS_TEM_Spectrum()
        >>> s_vac = hs.signals.BaseSignal(
                np.ones_like(s.data, dtype=float))*0.005
        >>> s_vac.add_poissonian_noise()
        >>> si = hs.stack([s]*3 + [s_vac])
        >>> si.vacuum_mask().data
        array([False, False, False,  True], dtype=bool)
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
        """Apply a decomposition to a dataset with a choice of algorithms.

        The results are stored in ``self.learning_results``.

        Read more in the :ref:`User Guide <mva.decomposition>`.

        Parameters
        ----------
        normalize_poissonian_noise : bool, default True
            If True, scale the signal to normalize Poissonian noise using
            the approach described in [Keenan2004]_.
        navigation_mask : None or float or boolean numpy array, default 1.0
            The navigation locations marked as True are not used in the
            decomposition. If float is given the vacuum_mask method is used to
            generate a mask with the float value as threshold.
        closing: bool, default True
            If true, applied a morphologic closing to the mask obtained by
            vacuum_mask.
        algorithm : {"SVD", "MLPCA", "sklearn_pca", "NMF", "sparse_pca", "mini_batch_sparse_pca", "RPCA", "ORPCA", "ORNMF", custom object}, default "SVD"
            The decomposition algorithm to use. If algorithm is an object,
            it must implement a ``fit_transform()`` method or ``fit()`` and
            ``transform()`` methods, in the same manner as a scikit-learn estimator.
        output_dimension : None or int
            Number of components to keep/calculate.
            Default is None, i.e. ``min(data.shape)``.
        centre : {None, "navigation", "signal"}, default None
            * If None, the data is not centered prior to decomposition.
            * If "navigation", the data is centered along the navigation axis.
              Only used by the "SVD" algorithm.
            * If "signal", the data is centered along the signal axis.
              Only used by the "SVD" algorithm.
        auto_transpose : bool, default True
            If True, automatically transposes the data to boost performance.
            Only used by the "SVD" algorithm.
        signal_mask : boolean numpy array
            The signal locations marked as True are not used in the
            decomposition.
        var_array : numpy array
            Array of variance for the maximum likelihood PCA algorithm.
            Only used by the "MLPCA" algorithm.
        var_func : None or function or numpy array, default None
            * If None, ignored
            * If function, applies the function to the data to obtain ``var_array``.
              Only used by the "MLPCA" algorithm.
            * If numpy array, creates ``var_array`` by applying a polynomial function
              defined by the array of coefficients to the data. Only used by
              the "MLPCA" algorithm.
        reproject : {None, "signal", "navigation", "both"}, default None
            If not None, the results of the decomposition will be projected in
            the selected masked area.
        return_info: bool, default False
            The result of the decomposition is stored internally. However,
            some algorithms generate some extra information that is not
            stored. If True, return any extra information if available.
            In the case of sklearn.decomposition objects, this includes the
            sklearn Estimator object.
        print_info : bool, default True
            If True, print information about the decomposition being performed.
            In the case of sklearn.decomposition objects, this includes the
            values of all arguments of the chosen sklearn algorithm.
        svd_solver : {"auto", "full", "arpack", "randomized"}, default "auto"
            If auto:
                The solver is selected by a default policy based on `data.shape` and
                `output_dimension`: if the input data is larger than 500x500 and the
                number of components to extract is lower than 80% of the smallest
                dimension of the data, then the more efficient "randomized"
                method is enabled. Otherwise the exact full SVD is computed and
                optionally truncated afterwards.
            If full:
                run exact SVD, calling the standard LAPACK solver via
                :py:func:`scipy.linalg.svd`, and select the components by postprocessing
            If arpack:
                use truncated SVD, calling ARPACK solver via
                :py:func:`scipy.sparse.linalg.svds`. It requires strictly
                `0 < output_dimension < min(data.shape)`
            If randomized:
                use truncated SVD, calling :py:func:`sklearn.utils.extmath.randomized_svd`
                to estimate a limited number of components
        copy : bool, default True
            * If True, stores a copy of the data before any pre-treatments
              such as normalization in ``s._data_before_treatments``. The original
              data can then be restored by calling ``s.undo_treatments()``.
            * If False, no copy is made. This can be beneficial for memory
              usage, but care must be taken since data will be overwritten.
        **kwargs : extra keyword arguments
            Any keyword arguments are passed to the decomposition algorithm.


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
            navigation_mask = self.vacuum_mask(navigation_mask, closing)
        super().decomposition(
            normalize_poissonian_noise=normalize_poissonian_noise,
            navigation_mask=navigation_mask, *args, **kwargs)
        self.learning_results.loadings = np.nan_to_num(
            self.learning_results.loadings)

    def create_model(self, auto_background=True, auto_add_lines=True,
                     *args, **kwargs):
        """Create a model for the current TEM EDS data.

        Parameters
        ----------
        auto_background : bool, default True
            If True, adds automatically a polynomial order 6 to the model,
            using the edsmodel.add_polynomial_background method.
        auto_add_lines : bool, default True
            If True, automatically add Gaussians for all X-rays generated in
            the energy range by an element using the edsmodel.add_family_lines
            method.
        dictionary : {None, dict}, optional
            A dictionary to be used to recreate a model. Usually generated
            using :meth:`hyperspy.model.as_dictionary`

        Returns
        -------
        model : `EDSTEMModel` instance.

        """
        from hyperspy.models.edstemmodel import EDSTEMModel
        model = EDSTEMModel(self,
                            auto_background=auto_background,
                            auto_add_lines=auto_add_lines,
                            *args, **kwargs)
        return model

    def get_probe_area(self, navigation_axes=None):
        """
        Calculates a pixel area which can be approximated to probe area,
        when the beam is larger than or equal to pixel size.
        The probe area can be calculated only when the number of navigation
        dimension are less than 2 and all the units have the dimensions of
        length.

        Parameters
        ----------
        navigation_axes : DataAxis, string or integer (or list of)
            Navigation axes corresponding to the probe area. If string or
            integer, the provided value is used to index the ``axes_manager``.

        Returns
        -------
        probe area in nm².

        Examples
        --------
        >>> s = hs.datasets.example_signals.EDS_TEM_Spectrum()
        >>> si = hs.stack([s]*3)
        >>> si.axes_manager.navigation_axes[0].scale = 0.01
        >>> si.axes_manager.navigation_axes[0].units = 'μm'
        >>> si.get_probe_area()
        100.0

        """
        if navigation_axes is None:
            navigation_axes = self.axes_manager.navigation_axes
        elif not isiterable(navigation_axes):
            navigation_axes = [navigation_axes]
        if len(navigation_axes) == 0:
            raise ValueError("The navigation dimension is zero, the probe "
                             "area can not be calculated automatically.")
        elif len(navigation_axes) > 2:
            raise ValueError("The navigation axes corresponding to the probe "
                             "are ambiguous and the probe area can not be "
                             "calculated automatically.")
        scales = []

        for axis in navigation_axes:
            try:
                if not isinstance(navigation_axes, DataAxis):
                    axis = self.axes_manager[axis]
                scales.append(axis.convert_to_units('nm', inplace=False)[0])
            except pint.DimensionalityError:
                raise ValueError(f"The unit of the axis {axis} has not the "
                                 "dimension of length.")

        if len(scales) == 1:
            probe_area = scales[0] ** 2
        else:
            probe_area = scales[0] * scales[1]

        if probe_area == 1:
            warnings.warn("Please note that the probe area has been "
                          "calculated to be 1 nm², meaning that it is highly "
                          "likley that the scale of the navigation axes have not "
                          "been set correctly. Please read the user "
                          "guide for how to set this.")
        return probe_area


    def _get_dose(self, method, beam_current='auto', live_time='auto',
                  probe_area='auto'):
        """
        Calculates the total electron dose for the zeta-factor or cross section
        methods of quantification.

        Input given by i*t*N, i the current, t the
        acquisition time, and N the number of electron by unit electric charge.

        Parameters
        ----------
        method : 'zeta' or 'cross_section'
            If 'zeta', the dose is given by i*t*N
            If 'cross section', the dose is given by i*t*N/A
            where i is the beam current, t is the acquistion time,
            N is the number of electrons per unit charge (1/e) and
            A is the illuminated beam area or pixel area.
        beam_current: float
            Probe current in nA
        live_time: float
            Acquisiton time in s, compensated for the dead time of the detector.
        probe_area: float or 'auto'
            The illumination area of the electron beam in nm².
            If 'auto' the value is extracted from the scale axes_manager.
            Therefore we assume the probe is oversampling such that
            the illumination area can be approximated to the pixel area of the
            spectrum image.

        Returns
        --------
        Dose in electrons (zeta factor) or electrons per nm² (cross_section)

        See also
        --------
        set_microscope_parameters
        """

        parameters = self.metadata.Acquisition_instrument.TEM

        if beam_current == 'auto':
            beam_current = parameters.get_item('beam_current')
            if beam_current is None:
                raise Exception('Electron dose could not be calculated as the '
                                'beam current is not set. It can set using '
                                '`set_microscope_parameters()`.')

        if live_time == 'auto':
            live_time = parameters.get_item('Detector.EDS.live_time')
            if live_time is None:
                raise Exception('Electron dose could not be calculated as '
                                'live time is not set. It can set using '
                                '`set_microscope_parameters()`.')

        if method == 'cross_section':
            if probe_area == 'auto':
                probe_area = parameters.get_item('probe_area')
                if probe_area is None:
                    probe_area = self.get_probe_area(
                        navigation_axes=self.axes_manager.navigation_axes)
            return (live_time * beam_current * 1e-9) / (constants.e * probe_area)
            # 1e-9 is included here because the beam_current is in nA.
        elif method == 'zeta':
            return live_time * beam_current * 1e-9 / constants.e
        else:
            raise Exception("Method need to be 'zeta' or 'cross_section'.")


    @staticmethod
    def CL_get_mass_thickness(weight_percent, thickness):
        """
        Creates a array of mass_thickness based on a known material composition
        and measured thickness. Required for absorption correction calcultions
        using the Cliff Lorimer method.

        Parameters
        ----------
        weight_percent : :py:class:`~hyperspy.signal.BaseSignal` (or subclass)
            Stack of compositions as determined from an initial k_factor
            quantification.
        thickness : float or :py:class:`numpy.ndarray`
            Either a float value for thickness in nm or an array equal to the
            size of the EDX map with thickness at each position of the sample.

        Returns
        -------
        mass_thickness : :py:class:`numpy.ndarray`
            Mass thickness in kg/m².
        """
        if isinstance(thickness, (float, int)):
            thickness_map = np.ones_like(weight_percent[0]) * thickness
        else:
            thickness_map = thickness

        elements = [intensity.metadata.Sample.elements[0] for intensity in weight_percent]
        mass_thickness = np.zeros_like(weight_percent[0])
        densities = np.array(
            [elements_db[element]['Physical_properties']['density (g/cm^3)']
                    for element in elements])
        for density, element_composition in zip(densities, weight_percent):
            # convert composition from % to fraction: factor of 1E-2
            # convert thickness from nm to m: factor of 1E-9
            # convert density from g/cm3 to kg/m2: factor of 1E3
            elemental_mt = element_composition * thickness_map * density * 1E-8
            mass_thickness += elemental_mt
        return mass_thickness


class EDSTEMSpectrum(EDSTEM_mixin, EDSSpectrum):
    pass


class LazyEDSTEMSpectrum(EDSTEMSpectrum, LazyEDSSpectrum):
    pass
