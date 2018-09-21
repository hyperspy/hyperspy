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


import math
import logging

import numpy as np
from scipy.interpolate import splev

from hyperspy.component import Component
from hyperspy.misc.eels.hartree_slater_gos import HartreeSlaterGOS
from hyperspy.misc.eels.hydrogenic_gos import HydrogenicGOS
from hyperspy.misc.eels.effective_angle import effective_angle
from hyperspy.ui_registry import add_gui_method


_logger = logging.getLogger(__name__)


@add_gui_method(toolkey="EELSCLEdge_Component")
class EELSCLEdge(Component):

    """EELS core loss ionisation edge from hydrogenic or tabulated
    Hartree-Slater GOS with splines for fine structure fitting.

    Hydrogenic GOS are limited to K and L shells.

    Currently it only supports Peter Rez's Hartree Slater cross sections
    parametrised as distributed by Gatan in their Digital Micrograph (DM)
    software. If Digital Micrograph is installed in the system HyperSpy in the
    standard location HyperSpy should find the path to the HS GOS folder.
    Otherwise, the location of the folder can be defined in HyperSpy
    preferences, which can be done through hs.preferences.gui() or the
    hs.preferences.EELS.eels_gos_files_path variable.

    Parameters
    ----------
    element_subshell : {str, dict}
        Usually a string, for example, 'Ti_L3' for the GOS of the titanium L3
        subshell. If a dictionary is passed, it is assumed that Hartree Slater
        GOS was exported using `GOS.as_dictionary`, and will be reconstructed.

    GOS : {'hydrogenic', 'Hartree-Slater', None}
        The GOS to use. If None it will use the Hartree-Slater GOS if
        they are available, otherwise it will use the hydrogenic GOS.

    Attributes
    ----------
    onset_energy : Parameter
        The edge onset position
    intensity : Parameter
        The factor by which the cross section is multiplied, what in
        favourable cases is proportional to the number of atoms of
        the element. It is a component.Parameter instance.
        It is fixed by default.
    fine_structure_coeff : Parameter
        The coefficients of the spline that fits the fine structure.
        Fix this parameter to fix the fine structure. It is a
        component.Parameter instance.
    effective_angle : Parameter
        The effective collection semi-angle. It is automatically
        calculated by set_microscope_parameters. It is a
        component.Parameter instance. It is fixed by default.
    fine_structure_smoothing : float between 0 and 1
        Controls the level of smoothing of the fine structure model.
        Decreasing the value increases the level of smoothing.

    fine_structure_active : bool
        Activates/deactivates the fine structure feature.

    """
    _fine_structure_smoothing = 0.3

    def __init__(self, element_subshell, GOS=None):
        # Declare the parameters
        Component.__init__(self,
                           ['intensity',
                            'fine_structure_coeff',
                            'effective_angle',
                            'onset_energy'])
        if isinstance(element_subshell, dict):
            self.element = element_subshell['element']
            self.subshell = element_subshell['subshell']
        else:
            self.element, self.subshell = element_subshell.split('_')
        self.name = "_".join([self.element, self.subshell])
        self.energy_scale = None
        self.effective_angle.free = False
        self.fine_structure_active = False
        self.fine_structure_width = 30.
        self.fine_structure_coeff.ext_force_positive = False
        self.GOS = None
        # Set initial actions
        if GOS is None:
            try:
                self.GOS = HartreeSlaterGOS(element_subshell)
                GOS = 'Hartree-Slater'
            except IOError:
                GOS = 'hydrogenic'
                _logger.info(
                    'Hartree-Slater GOS not available. '
                    'Using hydrogenic GOS')
        if self.GOS is None:
            if GOS == 'Hartree-Slater':
                self.GOS = HartreeSlaterGOS(element_subshell)
            elif GOS == 'hydrogenic':
                self.GOS = HydrogenicGOS(element_subshell)
            else:
                raise ValueError(
                    'gos must be one of: None, \'hydrogenic\''
                    ' or \'Hartree-Slater\'')
        self.onset_energy.value = self.GOS.onset_energy
        self.onset_energy.free = False
        self._position = self.onset_energy
        self.free_onset_energy = False
        self.intensity.grad = self.grad_intensity
        self.intensity.value = 1
        self.intensity.bmin = 0.
        self.intensity.bmax = None

        self._whitelist['GOS'] = ('init', GOS)
        if GOS == 'Hartree-Slater':
            self._whitelist['element_subshell'] = (
                'init',
                self.GOS.as_dictionary(True))
        elif GOS == 'hydrogenic':
            self._whitelist['element_subshell'] = ('init', element_subshell)
        self._whitelist['fine_structure_active'] = None
        self._whitelist['fine_structure_width'] = None
        self._whitelist['fine_structure_smoothing'] = None
        self.effective_angle.events.value_changed.connect(
            self._integrate_GOS, [])
        self.onset_energy.events.value_changed.connect(self._integrate_GOS, [])
        self.onset_energy.events.value_changed.connect(
            self._calculate_knots, [])

    # Automatically fix the fine structure when the fine structure is
    # disable.
    # In this way we avoid a common source of problems when fitting
    # However the fine structure must be *manually* freed when we
    # reactivate the fine structure.
    def _get_fine_structure_active(self):
        return self.__fine_structure_active

    def _set_fine_structure_active(self, arg):
        if arg is False:
            self.fine_structure_coeff.free = False
        self.__fine_structure_active = arg
        # Force replot
        self.intensity.value = self.intensity.value
    fine_structure_active = property(_get_fine_structure_active,
                                     _set_fine_structure_active)

    def _get_fine_structure_width(self):
        return self.__fine_structure_width

    def _set_fine_structure_width(self, arg):
        self.__fine_structure_width = arg
        self._set_fine_structure_coeff()
    fine_structure_width = property(_get_fine_structure_width,
                                    _set_fine_structure_width)

    # E0
    def _get_E0(self):
        return self.__E0

    def _set_E0(self, arg):
        self.__E0 = arg
        self._calculate_effective_angle()
    E0 = property(_get_E0, _set_E0)

    # Collection semi-angle
    def _get_collection_angle(self):
        return self.__collection_angle

    def _set_collection_angle(self, arg):
        self.__collection_angle = arg
        self._calculate_effective_angle()
    collection_angle = property(_get_collection_angle,
                                _set_collection_angle)
    # Convergence semi-angle

    def _get_convergence_angle(self):
        return self.__convergence_angle

    def _set_convergence_angle(self, arg):
        self.__convergence_angle = arg
        self._calculate_effective_angle()
    convergence_angle = property(_get_convergence_angle,
                                 _set_convergence_angle)

    def _calculate_effective_angle(self):
        try:
            self.effective_angle.value = effective_angle(
                self.E0,
                self.GOS.onset_energy,
                self.convergence_angle,
                self.collection_angle)
        except BaseException:
            # All the parameters may not be defined yet...
            pass

    @property
    def fine_structure_smoothing(self):
        """Controls the level of the smoothing of the fine structure.

        It must a real number between 0 and 1. The higher close to 0
        the higher the smoothing.

        """
        return self._fine_structure_smoothing

    @fine_structure_smoothing.setter
    def fine_structure_smoothing(self, value):
        if 0 <= value <= 1:
            self._fine_structure_smoothing = value
            self._set_fine_structure_coeff()
        else:
            raise ValueError(
                "The value must be a number between 0 and 1")

    # It is needed because the property cannot be used to sort the edges
    def _onset_energy(self):
        return self.onset_energy.value

    def _set_fine_structure_coeff(self):
        if self.energy_scale is None:
            return
        self.fine_structure_coeff._number_of_elements = int(
            round(self.fine_structure_smoothing *
                  self.fine_structure_width /
                  self.energy_scale)) + 4
        self.fine_structure_coeff.bmin = None
        self.fine_structure_coeff.bmax = None
        self._calculate_knots()
        if self.fine_structure_coeff.map is not None:
            self.fine_structure_coeff._create_array()

    def set_microscope_parameters(self, E0, alpha, beta, energy_scale):
        """
        Parameters
        ----------
        E0 : float
            Electron beam energy in keV.
        alpha: float
            Convergence semi-angle in mrad.
        beta: float
            Collection semi-angle in mrad.
        energy_scale : float
            The energy step in eV.
        """
        # Relativistic correction factors
        old = self.effective_angle.value
        with self.effective_angle.events.value_changed.suppress_callback(
                self._integrate_GOS):
            self.convergence_angle = alpha
            self.collection_angle = beta
            self.energy_scale = energy_scale
            self.E0 = E0
        if self.effective_angle.value != old:
            self._integrate_GOS()

    def _integrate_GOS(self):
        # Integration over q using splines
        angle = self.effective_angle.value * 1e-3  # in rad
        self.tab_xsection = self.GOS.integrateq(
            self.onset_energy.value, angle, self.E0)
        # Calculate extrapolation powerlaw extrapolation parameters
        E1 = self.GOS.energy_axis[-2] + self.GOS.energy_shift
        E2 = self.GOS.energy_axis[-1] + self.GOS.energy_shift
        y1 = self.GOS.qint[-2]  # in m**2/bin */
        y2 = self.GOS.qint[-1]  # in m**2/bin */
        self.r = math.log(y2 / y1) / math.log(E1 / E2)
        self.A = y1 / E1 ** -self.r

    def _calculate_knots(self):
        start = self.onset_energy.value
        stop = start + self.fine_structure_width
        self.__knots = np.r_[
            [start] * 4,
            np.linspace(
                start,
                stop,
                self.fine_structure_coeff._number_of_elements)[
                2:-2],
            [stop] * 4]

    def function(self, E):
        """Returns the number of counts in barns

        """
        shift = self.onset_energy.value - self.GOS.onset_energy
        if shift != self.GOS.energy_shift:
            # Because hspy Events are not executed in any given order,
            # an external function could be in the same event execution list
            # as _integrate_GOS and be executed first. That can potentially
            # cause an error that enforcing _integrate_GOS here prevents. Note
            # that this is suboptimal because _integrate_GOS is computed twice
            # unnecessarily.
            self._integrate_GOS()
        Emax = self.GOS.energy_axis[-1] + self.GOS.energy_shift
        cts = np.zeros((len(E)))
        bsignal = (E >= self.onset_energy.value)
        if self.fine_structure_active is True:
            bfs = bsignal * (
                E < (self.onset_energy.value + self.fine_structure_width))
            cts[bfs] = splev(
                E[bfs], (
                    self.__knots,
                    self.fine_structure_coeff.value + (0,) * 4,
                    3))
            bsignal[bfs] = False
        itab = bsignal * (E <= Emax)
        cts[itab] = self.tab_xsection(E[itab])
        bsignal[itab] = False
        cts[bsignal] = self.A * E[bsignal] ** -self.r
        return cts * self.intensity.value

    def grad_intensity(self, E):
        return self.function(E) / self.intensity.value

    def fine_structure_coeff_to_txt(self, filename):
        np.savetxt(filename + '.dat', self.fine_structure_coeff.value,
                   fmt="%12.6G")

    def txt_to_fine_structure_coeff(self, filename):
        fs = np.loadtxt(filename)
        self._calculate_knots()
        if len(fs) == len(self.__knots):
            self.fine_structure_coeff.value = fs
        else:
            raise ValueError(
                "The provided fine structure file "
                "doesn't match the size of the current fine structure")

    def get_fine_structure_as_signal1D(self):
        """Returns a spectrum containing the fine structure.

        Notes
        -----
        The fine structure is corrected from multiple scattering if
        the model was convolved with a low-loss spectrum

        """
        from hyperspy._signals.eels import EELSSpectrum
        channels = int(np.floor(
            self.fine_structure_width / self.energy_scale))
        data = np.zeros(self.fine_structure_coeff.map.shape +
                        (channels,))
        s = EELSSpectrum(
            data,
            axes=self.intensity._axes_manager._get_axes_dicts())
        s.get_dimensions_from_data()
        s.axes_manager.signal_axes[0].offset = self.onset_energy.value
        # Backup the axes_manager
        original_axes_manager = self._axes_manager
        self._axes_manager = s.axes_manager
        for spectrum in s:
            self.fetch_stored_values()
            spectrum.data[:] = self.function(
                s.axes_manager.signal_axes[0].axis)
        # Restore the axes_manager and the values
        self._axes_manager = original_axes_manager
        self.fetch_stored_values()

        s.metadata.General.title = self.name.replace(
            '_', ' ') + ' fine structure'

        return s
