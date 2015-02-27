# -*- coding: utf-8 -*-
# Copyright 2007-2011 The HyperSpy developers
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

from __future__ import division

import numpy as np
import math

from hyperspy.model import Model
from hyperspy._signals.eds import EDSSpectrum
from hyperspy.misc.elements import elements as elements_db
from hyperspy.misc.eds import utils as utils_eds
import hyperspy.components as create_component
# from hyperspy import utils


def _get_weight(element, line, weight_line=None):
    if weight_line is None:
        weight_line = elements_db[
            element]['Atomic_properties']['Xray_lines'][line]['weight']
    return lambda x: x * weight_line


def _get_iweight(element, line, weight_line=None):
    if weight_line is None:
        weight_line = elements_db[
            element]['Atomic_properties']['Xray_lines'][line]['weight']
    return lambda x: x / weight_line


def _get_sigma(E, E_ref, units_factor):
    # 2.5 from Goldstein, / 1000 eV->keV, / 2.355^2 for FWHM -> sigma
    return lambda sig_ref: math.sqrt(abs(
        4.5077 * 1e-4 * (E - E_ref) * units_factor + np.power(sig_ref, 2)))


class EDSModel(Model):

    """Build a fit a model

    Parameters
    ----------
    spectrum : an Spectrum (or any Spectrum subclass) instance
    auto_add_lines : boolean
        If True, and if spectrum is an eds instance adds automatically
        gaussians to model the X-ray lines.

    """

    def __init__(self, spectrum,
                 auto_add_lines=True,
                 *args, **kwargs):
        Model.__init__(self, spectrum, *args, **kwargs)
        self.xray_lines = list()
        self.background_components = list()
        end_energy = self.axes_manager.signal_axes[0].high_value
        if self.spectrum._get_beam_energy() < end_energy:
            self.end_energy = self.spectrum._get_beam_energy()
        else:
            self.end_energy = end_energy
        self.start_energy = self.axes_manager.signal_axes[0].low_value
        units_name = self.axes_manager.signal_axes[0].units
        if units_name == 'eV':
            self.units_factor = 1000.
        elif units_name == 'keV':
            self.units_factor = 1.
        else:
            raise ValueError("Energy units, %s, not supported" %
                             str(units_name))
        if auto_add_lines is True:
            self.add_family_lines()

    @property
    def spectrum(self):
        return self._spectrum

    @spectrum.setter
    def spectrum(self, value):
        if isinstance(value, EDSSpectrum):
            self._spectrum = value
        else:
            raise ValueError(
                "This attribute can only contain an EDSSpectrum "
                "but an object of type %s was provided" %
                str(type(value)))

    @property
    def _active_xray_lines(self):
        return [xray_line for xray_line
                in self.xray_lines if xray_line.active]

    def add_family_lines(self, xray_lines='from_elements'):
        """Create the Xray-lines instances and configure them appropiately

        If a X-ray line is given, all the the lines of the familiy is added

        Parameters
        -----------
        xray_lines: {None, 'from_elements', list of string}
            If None, if `metadata` contains `xray_lines` list of lines use
            those. Else, add all lines from the elements contains in `metadata`
            Alternatively, provide an iterable containing
            a list of valid X-ray lines symbols. (eg. ('Al_Ka','Zn_Ka')).
        """

        only_one = False
        only_lines = ("Ka", "La", "Ma")

        if xray_lines is None or xray_lines == 'from_elements':
            if 'Sample.xray_lines' in self.spectrum.metadata \
                    and xray_lines != 'from_elements':
                xray_lines = self.spectrum.metadata.Sample.xray_lines
            elif 'Sample.elements' in self.spectrum.metadata:
                xray_lines = self.spectrum._get_lines_from_elements(
                    self.spectrum.metadata.Sample.elements,
                    only_one=only_one,
                    only_lines=only_lines)
            else:
                raise ValueError(
                    "No elements defined, set them with `add_elements`")

        components_names = [xr.name for xr in self.xray_lines]
        xray_lines = filter(lambda x: x not in components_names, xray_lines)
        xray_lines, xray_not_here = self.spectrum.\
            _get_xray_lines_in_spectral_range(xray_lines)
        for xray in xray_not_here:
            print("Warning: %s is not in the data energy range." % (xray))

        for i, xray_line in enumerate(xray_lines):
            element, line = utils_eds._get_element_and_line(xray_line)
            line_energy, line_FWHM = self.spectrum._get_line_energy(
                xray_line,
                FWHM_MnKa='auto')
            component = create_component.Gaussian()
            component.centre.value = line_energy
            component.sigma.value = line_FWHM / 2.355
            component.centre.free = False
            component.sigma.free = False
            component.name = xray_line
            self.append(component)
            self.xray_lines.append(component)
            init = True
            if init:
                self[xray_line].A.map[
                    'values'] = self.spectrum[..., line_energy].data * \
                    line_FWHM / self.spectrum.axes_manager[-1].scale
                self[xray_line].A.map['is_set'] = (
                    np.ones(self.spectrum[..., line_energy].data.shape) == 1)

            component.A.ext_force_positive = True
            for li in elements_db[element]['Atomic_properties']['Xray_lines']:
                if line[0] in li and line != li:
                    xray_sub = element + '_' + li
                    if self.spectrum.\
                            _get_xray_lines_in_spectral_range(
                                [xray_sub])[0] != []:
                        line_energy, line_FWHM = self.spectrum.\
                            _get_line_energy(
                                xray_sub, FWHM_MnKa='auto')
                        component_sub = create_component.Gaussian()
                        component_sub.centre.value = line_energy
                        component_sub.name = xray_sub
                        component_sub.sigma.value = line_FWHM / 2.355
                        component_sub.centre.free = False
                        component_sub.sigma.free = False
                        component_sub.A.twin_function = _get_weight(
                            element, li)
                        component_sub.A.twin_inverse_function = _get_iweight(
                            element, li)
                        component_sub.A.twin = component.A
                        self.append(component_sub)
            self.fetch_stored_values()

    @property
    def _active_background_components(self):
        return [bc for bc in self.background_components
                if bc.coefficients.free]

    def add_background(self, order=3):
        """
        Add a quadratic background
        """
        background = create_component.Polynomial(order=order)
        background.name = 'background'
        background.isbackground = True
        self.append(background)
        self.background_components.append(background)

    def free_background(self):
        """Free the yscale of the background components.

        """
        for component in self.background_components:
            component.coefficients.free = True

    def fix_background(self):
        """Fix the background components.

        """
        for component in self._active_background_components:
            component.coefficients.free = False

    def enable_xray_lines(self):
        """Enable the X-ray lines components.

        """
        for component in self.xray_lines:
            component.active = True

    def disable_xray_lines(self):
        """Disable the X-ray lines components.

        """
        for component in self._active_xray_lines:
            component.active = False

    def fit_background(self,
                       start_energy=None,
                       end_energy=None,
                       windows_sigma=[4, 3],
                       kind='single',
                       **kwargs):
        """
        Fit the background to energy range containing no X-ray line.

        Parameters
        ----------
        start_energy : {float, None}
            If float, limit the range of energies from the left to the
            given value.
        kind : {'single', 'multi'}
            If 'single' fit only the current location. If 'multi'
            use multifit.
        **kwargs : extra key word arguments
            All extra key word arguments are passed to fit or
        """

        if end_energy is None:
            end_energy = self.end_energy
        if start_energy is None:
            start_energy = self.start_energy

        # desactivate line
        self.free_background()
        self.disable_xray_lines()
        self.set_signal_range(start_energy, end_energy)
        for component in self:
            if component.isbackground is False:
                try:
                    self.remove_signal_range(
                        component.centre.value -
                        windows_sigma[0] * component.sigma.value,
                        component.centre.value +
                        windows_sigma[1] * component.sigma.value)
                except:
                    pass

        if kind == 'single':
            self.fit(**kwargs)
        if kind == 'multi':
            self.multifit(**kwargs)
        self.reset_signal_range()
        self.enable_xray_lines()
        self.fix_background()
        
    def free_energy_resolution(self, xray_lines):
        """
        Free the energy resolution of the main X-ray lines

        Resolutions of the different peak are twinned

        """
        if xray_lines == 'all_alpha':
            xray_lines = [compo.name for compo in self.xray_lines]

        for i, xray_line in enumerate(xray_lines):
            component = self[xray_line]
            if i == 0:
                component_ref = component
                component_ref.sigma.free = True
                E_ref = component_ref.centre.value
            else:
                component.sigma.free = True
                E = component.centre.value
                component.sigma.twin_function = _get_sigma(
                    E, E_ref, self.units_factor)
                component.sigma.twin_inverse_function = _get_sigma(
                    E_ref, E, self.units_factor)
                component.sigma.twin = component_ref.sigma

    def fix_energy_resolution(self, xray_lines):
        """
        Fix and remove twin of X-ray lines sigma
        """

        if xray_lines == 'all_alpha':
            xray_lines = [compo.name for compo in self.xray_lines]
        for i, xray_line in enumerate(xray_lines):
            component = self[xray_line]
            component.sigma.twin = None
            component.sigma.free = False

    def set_energy_resolution(self, xray_lines):
        """
        Set the fitted energy resolution to the spectrum and
        adjust the FHWM for all lines
        """
        if xray_lines == 'all_alpha':
            xray_lines = [compo.name for compo in self.xray_lines]
        energy_Mn_Ka = self.spectrum._get_line_energy('Mn_Ka')
        get_sigma_Mn_Ka = _get_sigma(
            energy_Mn_Ka, self[xray_lines[0]].centre.value, self.units_factor)
        FWHM_MnKa = get_sigma_Mn_Ka(self[xray_lines[0]].sigma.value
                                    ) * 1000. / self.units_factor * 2.355
        if FWHM_MnKa < 110:
            print "FWHM_MnKa of " + str(FWHM_MnKa) + " smaller than " + \
                "physically possible"
        else:

            self.spectrum.set_microscope_parameters(
                energy_resolution_MnKa=FWHM_MnKa)
            print 'FWHM_MnKa ' + str(FWHM_MnKa)

            for component in self:
                if component.isbackground is False:
                    line_energy, line_FWHM = self.spectrum._get_line_energy(
                        component.name, FWHM_MnKa='auto')
                    component.sigma.value = line_FWHM / 2.355

    def calibrate_energy_axis(self,
                              calibrate='resolution',
                              xray_lines='all_alpha',
                              kind='single',
                              spread_to_all_lines=True,
                              **kwargs):
        """
        Fit the calibration

        energy scaling of the spectrum

        Parameters
        ----------
        calibrate: 'resolution' or 'scale' or 'offset'
            If 'resolution', calibrate by fitting the peak width given by
            `energy_resolution_MnKa` in `metdata`
            If 'scale', calibrate the scale of the energy axes
            If 'scale', calibrate the offset of the energy axes
        xray_lines: {list of str | 'all_alpha'}
            The Xray lines. If 'all_alpha', fit all using all alpha lines
        kind : {'single', 'multi'}
            If 'single' fit only the current location. If 'multi'
            use multifit.
        spread_to_all_lines: bool
            if True, change the calibration value of the spectrum
        **kwargs : extra key word arguments
            All extra key word arguments are passed to fit or
        multifit, depending on the value of kind.

        """
        if calibrate == 'resolution':
            self.free_energy_resolution(xray_lines=xray_lines)
            if kind == 'single':
                self.fit(**kwargs)
            if kind == 'multi':
                self.multifit(**kwargs)
            self.fix_energy_resolution(xray_lines=xray_lines)
            if spread_to_all_lines:
                self.set_energy_resolution(xray_lines=xray_lines)
        elif calibrate == 'scale':
            print 'not done yet'
        elif calibrate == 'offset':
            print 'not done yet'
