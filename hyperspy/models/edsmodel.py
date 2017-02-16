# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
#
# This file is part of HyperSpy.
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

import warnings
import numpy as np
import math
import logging

from hyperspy.misc.utils import stash_active_state
from hyperspy.misc.eds.utils import _get_element_and_line

from hyperspy.models.model1d import Model1D
from hyperspy._signals.eds import EDSSpectrum
from hyperspy.misc.elements import elements as elements_db
from hyperspy.misc.eds import utils as utils_eds
import hyperspy.components1d as create_component

_logger = logging.getLogger(__name__)

eV2keV = 1000.
sigma2fwhm = 2 * math.sqrt(2 * math.log(2))


def _get_weight(element, line, weight_line=None):
    if weight_line is None:
        weight_line = elements_db[
            element]['Atomic_properties']['Xray_lines'][line]['weight']
    return "x * {}".format(weight_line)


def _get_sigma(E, E_ref, units_factor, return_f=False):
    """
    Calculates an approximate sigma value, accounting for peak broadening due
    to the detector, for a peak at energy E given a known width at a reference
    energy.

    The factor 2.5 is a constant derived by Fiori & Newbury as references
    below.

    Parameters
    ----------
    energy_resolution_MnKa : float
        Energy resolution of Mn Ka in eV
    E : float
        Energy of the peak in keV

    Returns
    -------
    float : FWHM of the peak in keV

    Notes
    -----
    This method implements the equation derived by Fiori and Newbury as is
    documented in the following:

        Fiori, C. E., and Newbury, D. E. (1978). In SEM/1978/I, SEM, Inc.,
        AFM O'Hare, Illinois, p. 401.

        Goldstein et al. (2003). "Scanning Electron Microscopy & X-ray
        Microanalysis", Plenum, third edition, p 315.
    """
    energy2sigma_factor = 2.5 / (eV2keV * (sigma2fwhm**2))
    if return_f:
        return lambda sig_ref: math.sqrt(abs(
            energy2sigma_factor * (E - E_ref) * units_factor +
            np.power(sig_ref, 2)))
    else:
        return "sqrt(abs({} * ({} - {}) * {} + sig_ref ** 2))".format(
            energy2sigma_factor, E, E_ref, units_factor)


def _get_offset(diff):
    return "x + {}".format(diff)


def _get_scale(E1, E_ref1, fact):
    return "{} + {} * (x - {})".format(E1, fact, E_ref1)


class EDSModel(Model1D):

    """Build and fit a model of an EDS Signal1D.

    Parameters
    ----------
    spectrum : an EDSSpectrum (or any EDSSpectrum subclass) instance.

    auto_add_lines : boolean
        If True, automatically add Gaussians for all X-rays generated
        in the energy range by an element, using the edsmodel.add_family_lines
        method.

    auto_background : boolean
        If True, adds automatically a polynomial order 6 to the model,
        using the edsmodel.add_polynomial_background method.

    Any extra arguments are passed to the Model creator.

    Example
    -------
    >>> m = s.create_model()
    >>> m.fit()
    >>> m.fit_background()
    >>> m.calibrate_energy_axis('resolution')
    >>> m.calibrate_xray_lines('energy', ['Au_Ma'])
    >>> m.calibrate_xray_lines('sub_weight',['Mn_La'], bound=10)
    """

    def __init__(self, spectrum,
                 auto_background=True,
                 auto_add_lines=True,
                 *args, **kwargs):
        Model1D.__init__(self, spectrum, *args, **kwargs)
        self.xray_lines = list()
        self.family_lines = list()
        end_energy = self.axes_manager.signal_axes[0].high_value
        self.end_energy = min(end_energy, self.signal._get_beam_energy())
        self.start_energy = self.axes_manager.signal_axes[0].low_value
        self.background_components = list()
        if 'dictionary' in kwargs or len(args) > 1:
            d = args[1] if len(args) > 1 else kwargs['dictionary']
            if len(d['xray_lines']) > 0:
                self.xray_lines.extend(
                    [self[name] for name in d['xray_lines']])
                auto_add_lines = False
            if len(d['background_components']) > 0:
                self.background_components.extend(
                    [self[name] for name in d['background_components']])
                auto_background = False
        if auto_background is True:
            self.add_polynomial_background()
        if auto_add_lines is True:
            # Will raise an error if no elements are specified, so check:
            if 'Sample.elements' in self.signal.metadata:
                self.add_family_lines()

    def as_dictionary(self, fullcopy=True):
        dic = super(EDSModel, self).as_dictionary(fullcopy)
        dic['xray_lines'] = [c.name for c in self.xray_lines]
        dic['background_components'] = [c.name for c in
                                        self.background_components]
        return dic

    @property
    def units_factor(self):
        units_name = self.axes_manager.signal_axes[0].units
        if units_name == 'eV':
            return 1000.
        elif units_name == 'keV':
            return 1.
        else:
            raise ValueError("Energy units, %s, not supported" %
                             str(units_name))

    @property
    def spectrum(self):
        return self._signal

    @spectrum.setter
    def spectrum(self, value):
        if isinstance(value, EDSSpectrum):
            self._signal = value
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

        If a X-ray line is given, all the the lines of the familiy is added.
        For instance if Zn Ka is given, Zn Kb is added too. The main lines
        (alpha) is added to self.xray_lines

        Parameters
        -----------
        xray_lines: {None, 'from_elements', list of string}
            If None, if `metadata` contains `xray_lines` list of lines use
            those. If 'from_elements', add all lines from the elements contains
            in `metadata`. Alternatively, provide an iterable containing
            a list of valid X-ray lines symbols. (eg. ('Al_Ka','Zn_Ka')).
        """

        only_one = False
        only_lines = ("Ka", "La", "Ma")

        if xray_lines is None or xray_lines == 'from_elements':
            if 'Sample.xray_lines' in self.signal.metadata \
                    and xray_lines != 'from_elements':
                xray_lines = self.signal.metadata.Sample.xray_lines
            elif 'Sample.elements' in self.signal.metadata:
                xray_lines = self.signal._get_lines_from_elements(
                    self.signal.metadata.Sample.elements,
                    only_one=only_one,
                    only_lines=only_lines)
            else:
                raise ValueError(
                    "No elements defined, set them with `add_elements`")

        components_names = [xr.name for xr in self.xray_lines]
        xray_lines = filter(lambda x: x not in components_names, xray_lines)
        xray_lines, xray_not_here = self.signal.\
            _get_xray_lines_in_spectral_range(xray_lines)
        for xray in xray_not_here:
            warnings.warn("%s is not in the data energy range." % (xray))

        for xray_line in xray_lines:
            element, line = utils_eds._get_element_and_line(xray_line)
            line_energy, line_FWHM = self.signal._get_line_energy(
                xray_line,
                FWHM_MnKa='auto')
            component = create_component.Gaussian()
            component.centre.value = line_energy
            component.fwhm = line_FWHM
            component.centre.free = False
            component.sigma.free = False
            component.name = xray_line
            self.append(component)
            self.xray_lines.append(component)
            self[xray_line].A.map[
                'values'] = self.signal.isig[line_energy].data * \
                line_FWHM / self.signal.axes_manager[-1].scale
            self[xray_line].A.map['is_set'] = (
                np.ones(self.signal.isig[line_energy].data.shape) == 1)
            component.A.ext_force_positive = True
            for li in elements_db[element]['Atomic_properties']['Xray_lines']:
                if line[0] in li and line != li:
                    xray_sub = element + '_' + li
                    if self.signal.\
                            _get_xray_lines_in_spectral_range(
                                [xray_sub])[0] != []:
                        line_energy, line_FWHM = self.signal.\
                            _get_line_energy(
                                xray_sub, FWHM_MnKa='auto')
                        component_sub = create_component.Gaussian()
                        component_sub.centre.value = line_energy
                        component_sub.fwhm = line_FWHM
                        component_sub.centre.free = False
                        component_sub.sigma.free = False
                        component_sub.name = xray_sub
                        component_sub.A.twin_function_expr = _get_weight(
                            element, li)
                        component_sub.A.twin = component.A
                        self.append(component_sub)
                        self.family_lines.append(component_sub)
            self.fetch_stored_values()

    @property
    def _active_background_components(self):
        return [bc for bc in self.background_components
                if bc.coefficients.free]

    def add_polynomial_background(self, order=6):
        """
        Add a polynomial background.

        the background is added to self.background_components

        Parameters
        ----------
        order: int
            The order of the polynomial
        """
        background = create_component.Polynomial(order=order)
        background.name = 'background_order_' + str(order)
        background.isbackground = True
        self.append(background)
        self.background_components.append(background)

    def free_background(self):
        """
        Free the yscale of the background components.
        """
        for component in self.background_components:
            component.coefficients.free = True

    def fix_background(self):
        """
        Fix the background components.
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

    def _make_position_adjuster(self, component, fix_it, show_label):
        # Override to ensure formatting of labels of xray lines
        super(EDSModel, self)._make_position_adjuster(
            component, fix_it, show_label)
        if show_label and component in (self.xray_lines + self.family_lines):
            label = self._position_widgets[component._position][1]
            label.string = (r"$\mathrm{%s}_{\mathrm{%s}}$" %
                            _get_element_and_line(component.name))

    def fit_background(self,
                       start_energy=None,
                       end_energy=None,
                       windows_sigma=(4., 3.),
                       kind='single',
                       **kwargs):
        """
        Fit the background in the energy range containing no X-ray line.

        After the fit, the background is fixed.

        Parameters
        ----------
        start_energy : {float, None}
            If float, limit the range of energies from the left to the
            given value.
        end_energy : {float, None}
            If float, limit the range of energies from the right to the
            given value.
        windows_sigma: tuple of two float
            The (lower, upper) bounds around each X-ray line, each as a float,
            to define the energy range free of X-ray lines.
        kind : {'single', 'multi'}
            If 'single' fit only the current location. If 'multi'
            use multifit.
        **kwargs : extra key word arguments
            All extra key word arguments are passed to fit or

        See also
        --------
        free_background
        """

        if end_energy is None:
            end_energy = self.end_energy
        if start_energy is None:
            start_energy = self.start_energy

        # disactivate line
        self.free_background()
        with stash_active_state(self):
            self.disable_xray_lines()
            self.set_signal_range(start_energy, end_energy)
            for component in self:
                if component.isbackground is False:
                    self.remove_signal_range(
                        component.centre.value -
                        windows_sigma[0] * component.sigma.value,
                        component.centre.value +
                        windows_sigma[1] * component.sigma.value)
            if kind == 'single':
                self.fit(**kwargs)
            if kind == 'multi':
                self.multifit(**kwargs)
            self.reset_signal_range()
        self.fix_background()

    def _twin_xray_lines_width(self, xray_lines):
        """
        Twin the width of the peaks

        The twinning models the energy resolution of the detector

        Parameters
        ----------
        xray_lines: list of str or 'all_alpha'
            The Xray lines. If 'all_alpha', fit all using all alpha lines
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
                component.sigma.twin_function_expr = _get_sigma(
                    E, E_ref, self.units_factor)
                component.sigma.twin_inverse_function_expr = _get_sigma(
                    E_ref, E, self.units_factor)

    def _set_energy_resolution(self, xray_lines, *args, **kwargs):
        """
        Adjust the width of all lines and set the fitted energy resolution
        to the spectrum

        Parameters
        ----------
        xray_lines: list of str or 'all_alpha'
            The Xray lines. If 'all_alpha', fit all using all alpha lines
        """
        if xray_lines == 'all_alpha':
            xray_lines = [compo.name for compo in self.xray_lines]
        energy_Mn_Ka, FWHM_MnKa_old = self.signal._get_line_energy('Mn_Ka',
                                                                   'auto')
        FWHM_MnKa_old *= eV2keV / self.units_factor
        get_sigma_Mn_Ka = _get_sigma(
            energy_Mn_Ka, self[xray_lines[0]].centre.value, self.units_factor,
            return_f=True)
        FWHM_MnKa = get_sigma_Mn_Ka(self[xray_lines[0]].sigma.value
                                    ) * eV2keV / self.units_factor * sigma2fwhm
        if FWHM_MnKa < 110:
            raise ValueError("FWHM_MnKa of " + str(FWHM_MnKa) +
                             " smaller than" + "physically possible")
        else:
            self.signal.set_microscope_parameters(
                energy_resolution_MnKa=FWHM_MnKa)
            _logger.info("Energy resolution (FWHM at Mn Ka) changed from " +
                         "{:.2f} to {:.2f} eV".format(
                             FWHM_MnKa_old, FWHM_MnKa))
            for component in self:
                if component.isbackground is False:
                    line_FWHM = self.signal._get_line_energy(
                        component.name, FWHM_MnKa='auto')[1]
                    component.fwhm = line_FWHM

    def _twin_xray_lines_scale(self, xray_lines):
        """
        Twin the scale of the peaks

        Parameters
        ----------
        xray_lines: list of str or 'all_alpha'
            The Xray lines. If 'all_alpha', fit all using all alpha lines
        """
        if xray_lines == 'all_alpha':
            xray_lines = [compo.name for compo in self.xray_lines]
        ax = self.signal.axes_manager[-1]
        ref = []
        for i, xray_line in enumerate(xray_lines):
            component = self[xray_line]
            if i == 0:
                component_ref = component
                component_ref.centre.free = True
                E_ref = component_ref.centre.value
                ref.append(E_ref)
            else:
                component.centre.free = True
                E = component.centre.value
                fact = float(ax.value2index(E)) / ax.value2index(E_ref)
                component.centre.twin_function_expr = _get_scale(
                    E, E_ref, fact)
                component.centre.twin = component_ref.centre
                ref.append(E)
        return ref

    def _set_energy_scale(self, xray_lines, ref):
        """
        Adjust the width of all lines and set the fitted energy resolution
        to the spectrum

        Parameters
        ----------
        xray_lines: list of str or 'all_alpha'
            The X-ray lines. If 'all_alpha', fit all using all alpha lines
        ref: list of float
            The centres, before fitting, of the X-ray lines included
        """
        if xray_lines == 'all_alpha':
            xray_lines = [compo.name for compo in self.xray_lines]
        ax = self.signal.axes_manager[-1]
        scale_old = self.signal.axes_manager[-1].scale
        ind = np.argsort(np.array(
            [compo.centre.value for compo in self.xray_lines]))[-1]
        E = self[xray_lines[ind]].centre.value
        scale = (ref[ind] - ax.offset) / ax.value2index(E)
        ax.scale = scale
        for i, xray_line in enumerate(xray_lines):
            component = self[xray_line]
            component.centre.value = ref[i]
        _logger.info("Scale changed from  %lf to %lf", scale_old, scale)

    def _twin_xray_lines_offset(self, xray_lines):
        """
        Twin the offset of the peaks

        Parameters
        ----------
        xray_lines: list of str or 'all_alpha'
            The Xray lines. If 'all_alpha', fit all using all alpha lines
        """
        if xray_lines == 'all_alpha':
            xray_lines = [compo.name for compo in self.xray_lines]
        ref = []
        for i, xray_line in enumerate(xray_lines):
            component = self[xray_line]
            if i == 0:
                component_ref = component
                component_ref.centre.free = True
                E_ref = component_ref.centre.value
                ref.append(E_ref)
            else:
                component.centre.free = True
                E = component.centre.value
                diff = E_ref - E
                component.centre.twin_function_expr = _get_offset(-diff)
                component.centre.twin = component_ref.centre
                ref.append(E)
        return ref

    def _set_energy_offset(self, xray_lines, ref):
        """
        Adjust the width of all lines and set the fitted energy resolution
        to the spectrum

        Parameters
        ----------
        xray_lines: list of str or 'all_alpha'
            The Xray lines. If 'all_alpha', fit all using all alpha lines
        ref: list of float
            The centres, before fitting, of the X-ray lines included
        """
        if xray_lines == 'all_alpha':
            xray_lines = [compo.name for compo in self.xray_lines]
        diff = self[xray_lines[0]].centre.value - ref[0]
        offset_old = self.signal.axes_manager[-1].offset
        self.signal.axes_manager[-1].offset -= diff
        offset = self.signal.axes_manager[-1].offset
        _logger.info("Offset changed from  %lf to %lf", offset_old, offset)
        for i, xray_line in enumerate(xray_lines):
            component = self[xray_line]
            component.centre.value = ref[i]

    def calibrate_energy_axis(self,
                              calibrate='resolution',
                              xray_lines='all_alpha',
                              **kwargs):
        """
        Calibrate the resolution, the scale or the offset of the energy axis
        by fitting.

        Parameters
        ----------
        calibrate: 'resolution' or 'scale' or 'offset'
            If 'resolution', fits the width of Gaussians place at all x-ray
            lines. The width is given by a model of the detector resolution,
            obtained by extrapolating the `energy_resolution_MnKa` in `metadata`
            `metadata`.
            This method will update the value of `energy_resolution_MnKa`.
            If 'scale', calibrate the scale of the energy axis
            If 'offset', calibrate the offset of the energy axis
        xray_lines: list of str or 'all_alpha'
            The Xray lines. If 'all_alpha', fit all using all alpha lines
        **kwargs : extra key word arguments
            All extra key word arguments are passed to fit or
            multifit, depending on the value of kind.

        """

        if calibrate == 'resolution':
            free = self._twin_xray_lines_width
            fix = self.fix_xray_lines_width
            scale = self._set_energy_resolution
        elif calibrate == 'scale':
            free = self._twin_xray_lines_scale
            fix = self.fix_xray_lines_energy
            scale = self._set_energy_scale
        elif calibrate == 'offset':
            free = self._twin_xray_lines_offset
            fix = self.fix_xray_lines_energy
            scale = self._set_energy_offset
        ref = free(xray_lines=xray_lines)
        self.fit(**kwargs)
        fix(xray_lines=xray_lines)
        scale(xray_lines=xray_lines, ref=ref)
        self.update_plot()

    def free_sub_xray_lines_weight(self, xray_lines='all', bound=0.01):
        """
        Free the weight of a sub X-ray lines

        Remove the twin on the height of sub-Xray lines (non alpha)

        Parameters
        ----------
        xray_lines: list of str or 'all'
            The Xray lines. If 'all', fit all lines
        bounds: float
            Bound the height of the peak to a fraction of
            its height
        """
        def free_twin(component):
            component.A.twin = None
            component.A.free = True
            if component.A.value - bound * component.A.value <= 0:
                component.A.bmin = 1e-10
            else:
                component.A.bmin = component.A.value - \
                    bound * component.A.value
            component.A.bmax = component.A.value + \
                bound * component.A.value
            component.A.ext_force_positive = True
        xray_families = [
            utils_eds._get_xray_lines_family(line) for line in xray_lines]
        for component in self:
            if component.isbackground is False:
                if xray_lines == 'all':
                    free_twin(component)
                elif utils_eds._get_xray_lines_family(
                        component.name) in xray_families:
                    free_twin(component)

    def fix_sub_xray_lines_weight(self, xray_lines='all'):
        """
        Fix the weight of a sub X-ray lines to the main X-ray lines

        Establish the twin on the height of sub-Xray lines (non alpha)
        """
        def fix_twin(component):
            component.A.bmin = 0.0
            component.A.bmax = None
            element, line = utils_eds._get_element_and_line(component.name)
            for li in elements_db[element]['Atomic_properties']['Xray_lines']:
                if line[0] in li and line != li:
                    xray_sub = element + '_' + li
                    if xray_sub in self:
                        component_sub = self[xray_sub]
                        component_sub.A.bmin = 1e-10
                        component_sub.A.bmax = None
                        weight_line = component_sub.A.value / component.A.value
                        component_sub.A.twin_function_expr = _get_weight(
                            element, li, weight_line)
                        component_sub.A.twin = component.A
                    else:
                        warnings.warn("The X-ray line expected to be in the "
                                      "model was not found")
        for component in self.xray_lines:
            if xray_lines == 'all' or component.name in xray_lines:
                fix_twin(component)
        self.fetch_stored_values()

    def free_xray_lines_energy(self, xray_lines='all', bound=0.001):
        """
        Free the X-ray line energy (shift or centre of the Gaussian)

        Parameters
        ----------
        xray_lines: list of str or 'all'
            The Xray lines. If 'all', fit all lines
        bound: float
            the bound around the actual energy, in keV or eV
        """

        for component in self:
            if component.isbackground is False:
                if xray_lines == 'all' or component.name in xray_lines:
                    component.centre.free = True
                    component.centre.bmin = component.centre.value - bound
                    component.centre.bmax = component.centre.value + bound

    def fix_xray_lines_energy(self, xray_lines='all'):
        """
        Fix the X-ray line energy (shift or centre of the Gaussian)

        Parameters
        ----------
        xray_lines: list of str, 'all', or 'all_alpha'
            The Xray lines. If 'all', fit all lines. If 'all_alpha' fit all
            using all alpha lines.
        bound: float
            the bound around the actual energy, in keV or eV
        """
        if xray_lines == 'all_alpha':
            xray_lines = [compo.name for compo in self.xray_lines]
        for component in self:
            if component.isbackground is False:
                if xray_lines == 'all' or component.name in xray_lines:
                    component.centre.twin = None
                    component.centre.free = False
                    component.centre.bmin = None
                    component.centre.bmax = None

    def free_xray_lines_width(self, xray_lines='all', bound=0.01):
        """
        Free the X-ray line width (sigma of the Gaussian)

        Parameters
        ----------
        xray_lines: list of str or 'all'
            The Xray lines. If 'all', fit all lines
        bound: float
            the bound around the actual energy, in keV or eV
        """

        for component in self:
            if component.isbackground is False:
                if xray_lines == 'all' or component.name in xray_lines:
                    component.sigma.free = True
                    component.sigma.bmin = component.sigma.value - bound
                    component.sigma.bmax = component.sigma.value + bound

    def fix_xray_lines_width(self, xray_lines='all'):
        """
        Fix the X-ray line width (sigma of the Gaussian)

        Parameters
        ----------
        xray_lines: list of str, 'all', or 'all_alpha'
            The Xray lines. If 'all', fit all lines. If 'all_alpha' fit all
            using all alpha lines.
        bound: float
            the bound around the actual energy, in keV or eV
        """
        if xray_lines == 'all_alpha':
            xray_lines = [compo.name for compo in self.xray_lines]
        for component in self:
            if component.isbackground is False:
                if xray_lines == 'all' or component.name in xray_lines:
                    component.sigma.twin = None
                    component.sigma.free = False
                    component.sigma.bmin = None
                    component.sigma.bmax = None

    def calibrate_xray_lines(self,
                             calibrate='energy',
                             xray_lines='all',
                             bound=1,
                             kind='single',
                             **kwargs):
        """
        Calibrate individually the X-ray line parameters.

        The X-ray line energy, the weight of the sub-lines and the X-ray line
        width can be calibrated.

        Parameters
        ----------
        calibrate: 'energy' or 'sub_weight' or 'width'
            If 'energy', calibrate the X-ray line energy.
            If 'sub_weight', calibrate the ratio between the main line
            alpha and the other sub-lines of the family
            If 'width', calibrate the X-ray line width.
        xray_lines: list of str or 'all'
            The Xray lines. If 'all', fit all lines
        bounds: float
            for 'energy' and 'width' the bound in energy, in eV
            for 'sub_weight' Bound the height of the peak to fraction of
            its height
        kind : {'single', 'multi'}
            If 'single' fit only the current location. If 'multi'
            use multifit.
        **kwargs : extra key word arguments
            All extra key word arguments are passed to fit or
            multifit, depending on the value of kind.
        """

        if calibrate == 'energy':
            bound = (bound / eV2keV) * self.units_factor
            free = self.free_xray_lines_energy
            fix = self.fix_xray_lines_energy
        elif calibrate == 'sub_weight':
            free = self.free_sub_xray_lines_weight
            fix = self.fix_sub_xray_lines_weight
        elif calibrate == 'width':
            bound = (bound / eV2keV) * self.units_factor
            free = self.free_xray_lines_width
            fix = self.fix_xray_lines_width

        free(xray_lines=xray_lines, bound=bound)
        if kind == 'single':
            self.fit(bounded=True, fitter='mpfit', **kwargs)
        elif kind == 'multi':
            self.multifit(bounded=True, fitter='mpfit', **kwargs)
        fix(xray_lines=xray_lines)

    def get_lines_intensity(self,
                            xray_lines=None,
                            plot_result=False,
                            **kwargs):
        """
        Return the fitted intensity of the X-ray lines.

        Return the area under the gaussian corresping to the X-ray lines

        Parameters
        ----------
        xray_lines: list of str or None or 'from_metadata'
            If None, all main X-ray lines (alpha)
            If 'from_metadata', take the Xray_lines stored in the `metadata`
            of the spectrum. Alternatively, provide an iterable containing
            a list of valid X-ray lines symbols.
        plot_result : bool
            If True, plot the calculated line intensities. If the current
            object is a single spectrum it prints the result instead.
        kwargs
            The extra keyword arguments for plotting. See
            `utils.plot.plot_signals`

        Returns
        -------
        intensities : list
            A list containing the intensities as Signal subclasses.

        Examples
        --------
        >>> m.multifit()
        >>> m.get_lines_intensity(["C_Ka", "Ta_Ma"])
        """
        from hyperspy import utils
        intensities = []
        if xray_lines is None:
            xray_lines = [component.name for component in self.xray_lines]
        else:
            if xray_lines == 'from_metadata':
                xray_lines = self.signal.metadata.Sample.xray_lines
            xray_lines = filter(lambda x: x in [a.name for a in
                                                self], xray_lines)
        if not xray_lines:
            raise ValueError("These X-ray lines are not part of the model.")
        for xray_line in xray_lines:
            element, line = utils_eds._get_element_and_line(xray_line)
            line_energy = self.signal._get_line_energy(xray_line)
            data_res = self[xray_line].A.map['values']
            if self.axes_manager.navigation_dimension == 0:
                data_res = data_res[0]
            img = self.signal.isig[0:1].integrate1D(-1)
            img.data = data_res
            img.metadata.General.title = (
                'Intensity of %s at %.2f %s from %s' %
                (xray_line,
                 line_energy,
                 self.signal.axes_manager.signal_axes[0].units,
                 self.signal.metadata.General.title))
            if plot_result and img.axes_manager.signal_dimension == 0:
                print("%s at %s %s : Intensity = %.2f"
                      % (xray_line,
                         line_energy,
                         self.signal.axes_manager.signal_axes[0].units,
                         img.data))
            img.metadata.set_item("Sample.elements", ([element]))
            img.metadata.set_item("Sample.xray_lines", ([xray_line]))
            intensities.append(img)
        if plot_result and img.axes_manager.signal_dimension != 0:
            utils.plot.plot_signals(intensities, **kwargs)
        return intensities
