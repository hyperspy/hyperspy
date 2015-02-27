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
# import math

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

    def add_family_lines(self, xray_lines='from_elements'):
        """Create the Xray-lines instances and configure them appropiately

        If a X-ray line is given, all the the lines of the familiy is added

        Parameters
        -----------
        xray_lines: {None, 'from_elements', list of string}
            If None,
            if `metadata`  contains `xray_lines` list of lines use those.
            If `mapped.parameters.Sample.elements.xray_lines` is undefined
            or empty or if xray_lines equals 'from_elements' and
            `mapped.parameters.Sample.elements` is defined,
            use the same syntax as `add_line` to select a subset of lines
            for the operation.
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

    def add_background(self, order=3):
        """
        Add a quadratic background
        """
        background = create_component.Polynomial(order=order)
        background.name = 'background'
        background.isbackground = True
        self.append(background)
        self.background_components.append(background)
