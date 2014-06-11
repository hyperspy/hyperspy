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

from hyperspy import utils
from hyperspy._signals.spectrum import Spectrum
from hyperspy.misc.elements import elements as elements_db
from hyperspy.misc.eds import utils as utils_eds
from hyperspy.misc.utils import isiterable


class EDSSpectrum(Spectrum):
    _signal_type = "EDS"

    def __init__(self, *args, **kwards):
        Spectrum.__init__(self, *args, **kwards)
        if self.metadata.Signal.signal_type == 'EDS':
            print('The microscope type is not set. Use '
                  'set_signal_type(\'EDS_TEM\') or set_signal_type(\'EDS_SEM\')')
        self.metadata.Signal.binned = True

    def _get_line_energy(self, Xray_line, FWHM_MnKa=None):
        """
        Get the line energy and the energy resolution of a Xray line.

        The return values are in the same units than the signal axis

        Parameters
        ----------

        Xray_line : strings
            Valid element X-ray lines e.g. Fe_Kb.

        FWHM_MnKa: {None, float, 'auto'}
            The energy resolution of the detector in eV
            if 'auto', used the one in
            'self.metadata.Acquisition_instrument.SEM.Detector.EDS.energy_resolution_MnKa'

        Returns
        ------

        float: the line energy, if FWHM_MnKa is None
        (float,float): the line energy and the energy resolution, if FWHM_MnKa is not None
        """

        units_name = self.axes_manager.signal_axes[0].units

        if FWHM_MnKa == 'auto':
            if self.metadata.Signal.signal_type == 'EDS_SEM':
                FWHM_MnKa = self.metadata.Acquisition_instrument.SEM.Detector.EDS.energy_resolution_MnKa
            elif self.metadata.Signal.signal_type == 'EDS_TEM':
                FWHM_MnKa = self.metadata.Acquisition_instrument.TEM.Detector.EDS.energy_resolution_MnKa
            else:
                raise NotImplementedError(
                    "This method only works for EDS_TEM or EDS_SEM signals. "
                    "You can use `set_signal_type(\"EDS_TEM\")` or"
                    "`set_signal_type(\"EDS_SEM\")` to convert to one of these"
                    "signal types.")
        line_energy = utils_eds._get_energy_xray_line(Xray_line)
        if units_name == 'eV':
            line_energy *= 1000
            if FWHM_MnKa is not None:
                line_FWHM = utils_eds.get_FWHM_at_Energy(FWHM_MnKa,
                                                         line_energy / 1000) * 1000
        elif units_name == 'keV':
            if FWHM_MnKa is not None:
                line_FWHM = utils_eds.get_FWHM_at_Energy(FWHM_MnKa,
                                                         line_energy)
        else:
            raise ValueError(
                "%s is not a valid units for the energy axis. "
                "Only `eV` and `keV` are supported. "
                "If `s` is the variable containing this EDS spectrum:\n "
                ">>> s.axes_manager.signal_axes[0].units = \'keV\' \n"
                % (units_name))
        if FWHM_MnKa is None:
            return line_energy
        else:
            return line_energy, line_FWHM

    def _get_beam_energy(self):
        """
        Get the beam energy.

        The return value is in the same units than the signal axis
        """

        if "Acquisition_instrument.SEM.beam_energy" in self.metadata:
            beam_energy = self.metadata.Acquisition_instrument.SEM.beam_energy
        elif "Acquisition_instrument.TEM.beam_energy" in self.metadata:
            beam_energy = self.metadata.Acquisition_instrument.TEM.beam_energy
        else:
            raise AttributeError(
                "To use this method the beam energy `Acquisition_instrument.TEM.beam_energy` "
                "or `Acquisition_instrument.SEM.beam_energy` must be defined in "
                "`metadata`.")

        units_name = self.axes_manager.signal_axes[0].units

        if units_name == 'eV':
            beam_energy = beam_energy * 1000
        return beam_energy

    def sum(self, axis):
        """Sum the data over the given axis.

        Parameters
        ----------
        axis : {int, string}
           The axis can be specified using the index of the axis in
           `axes_manager` or the axis name.

        Returns
        -------
        s : Signal

        See also
        --------
        sum_in_mask, mean

        Examples
        --------
        >>> import numpy as np
        >>> s = Signal(np.random.random((64,64,1024)))
        >>> s.data.shape
        (64,64,1024)
        >>> s.sum(-1).data.shape
        (64,64)
        # If we just want to plot the result of the operation
        s.sum(-1, True).plot()

        """
        # modify time spend per spectrum
        if "Acquisition_instrument.SEM" in self.metadata:
            mp = self.metadata.Acquisition_instrument.SEM
        else:
            mp = self.metadata.Acquisition_instrument.TEM
        if mp.has_item('Detector.EDS.live_time'):
            mp.Detector.EDS.live_time = mp.Detector.EDS.live_time * \
                self.axes_manager.shape[axis]
        return super(EDSSpectrum, self).sum(axis)

    def rebin(self, new_shape):
        """Rebins the data to the new shape

        Parameters
        ----------
        new_shape: tuple of ints
            The new shape must be a divisor of the original shape

        """
        new_shape_in_array = []
        for axis in self.axes_manager._axes:
            new_shape_in_array.append(
                new_shape[axis.index_in_axes_manager])
        factors = (np.array(self.data.shape) /
                   np.array(new_shape_in_array))
        s = super(EDSSpectrum, self).rebin(new_shape)
        # modify time per spectrum
        if "Acquisition_instrument.SEM.Detector.EDS.live_time" in s.metadata:
            for factor in factors:
                s.metadata.Acquisition_instrument.SEM.Detector.EDS.live_time *= factor
        if "Acquisition_instrument.TEM.Detector.EDS.live_time" in s.metadata:
            for factor in factors:
                s.metadata.Acquisition_instrument.TEM.Detector.EDS.live_time *= factor
        return s

    def set_elements(self, elements):
        """Erase all elements and set them.

        Parameters
        ----------
        elements : list of strings
            A list of chemical element symbols.

        See also
        --------
        add_elements, set_line, add_lines.

        Examples
        --------

        >>> s = signals.EDSSEMSpectrum(np.arange(1024))
        >>> s.set_elements(['Ni', 'O'],['Ka','Ka'])
        Adding Ni_Ka Line
        Adding O_Ka Line
        >>> s.mapped_paramters.Acquisition_instrument.SEM.beam_energy = 10
        >>> s.set_elements(['Ni', 'O'])
        Adding Ni_La Line
        Adding O_Ka Line

        """
        # Erase previous elements and X-ray lines
        if "Sample.elements" in self.metadata:
            del self.metadata.Sample.elements
        self.add_elements(elements)

    def add_elements(self, elements):
        """Add elements and the corresponding X-ray lines.

        The list of elements is stored in `metadata.Sample.elements`

        Parameters
        ----------
        elements : list of strings
            The symbol of the elements.


        See also
        --------
        set_elements, add_lines, set_lines.

        """
        if not isiterable(elements) or isinstance(elements, basestring):
            raise ValueError(
                "Input must be in the form of a list. For example, "
                "if `s` is the variable containing this EDS spectrum:\n "
                ">>> s.add_elements(('C',))\n"
                "See the docstring for more information.")
        if "Sample.elements" in self.metadata:
            elements_ = set(self.metadata.Sample.elements)
        else:
            elements_ = set()
        for element in elements:
            if element in elements_db:
                elements_.add(element)
            else:
                raise ValueError(
                    "%s is not a valid chemical element symbol." % element)

        if not hasattr(self.metadata, 'Sample'):
            self.metadata.add_node('Sample')

        self.metadata.Sample.elements = sorted(list(elements_))

    def set_lines(self,
                  lines,
                  only_one=True,
                  only_lines=("Ka", "La", "Ma")):
        """Erase all Xrays lines and set them.

        See add_lines for details.

        Parameters
        ----------
        lines : list of strings
            A list of valid element X-ray lines to add e.g. Fe_Kb.
            Additionally, if `metadata.Sample.elements` is
            defined, add the lines of those elements that where not
            given in this list.
        only_one: bool
            If False, add all the lines of each element in
            `metadata.Sample.elements` that has not line
            defined in lines. If True (default),
            only add the line at the highest energy
            above an overvoltage of 2 (< beam energy / 2).
        only_lines : {None, list of strings}
            If not None, only the given lines will be added.

        See also
        --------
        add_lines, add_elements, set_elements..

        """
        if "Sample.xray_lines" in self.metadata:
            del self.metadata.Sample.xray_lines
        self.add_lines(lines=lines,
                       only_one=only_one,
                       only_lines=only_lines)

    def add_lines(self,
                  lines=(),
                  only_one=True,
                  only_lines=("Ka", "La", "Ma")):
        """Add X-rays lines to the internal list.

        Although most functions do not require an internal list of
        X-ray lines because they can be calculated from the internal
        list of elements, ocassionally it might be useful to customize the
        X-ray lines to be use by all functions by default using this method.
        The list of X-ray lines is stored in
        `metadata.Sample.xray_lines`

        Parameters
        ----------
        lines : list of strings
            A list of valid element X-ray lines to add e.g. Fe_Kb.
            Additionally, if `metadata.Sample.elements` is
            defined, add the lines of those elements that where not
            given in this list. If the list is empty (default), and
            `metadata.Sample.elements` is
            defined, add the lines of all those elements.
        only_one: bool
            If False, add all the lines of each element in
            `metadata.Sample.elements` that has not line
            defined in lines. If True (default),
            only add the line at the highest energy
            above an overvoltage of 2 (< beam energy / 2).
        only_lines : {None, list of strings}
            If not None, only the given lines will be added.

        See also
        --------
        set_lines, add_elements, set_elements.

        """
        if "Sample.xray_lines" in self.metadata:
            xray_lines = set(self.metadata.Sample.xray_lines)
        else:
            xray_lines = set()
        # Define the elements which Xray lines has been customized
        # So that we don't attempt to add new lines automatically
        elements = set()
        for line in xray_lines:
            elements.add(line.split("_")[0])
        end_energy = self.axes_manager.signal_axes[0].high_value
        for line in lines:
            try:
                element, subshell = line.split("_")
            except ValueError:
                raise ValueError(
                    "Invalid line symbol. "
                    "Please provide a valid line symbol e.g. Fe_Ka")
            if element in elements_db:
                elements.add(element)
                if subshell in elements_db[element]['Atomic_properties']['Xray_lines']:
                    lines_len = len(xray_lines)
                    xray_lines.add(line)
                    if lines_len != len(xray_lines):
                        print("%s line added," % line)
                    else:
                        print("%s line already in." % line)
                    if (self._get_line_energy(element + '_' + subshell) > end_energy):
                        print("Warning: %s %s is above the data energy range."
                              % (element, subshell))
                else:
                    raise ValueError(
                        "%s is not a valid line of %s." % (line, element))
            else:
                raise ValueError(
                    "%s is not a valid symbol of an element." % element)
        if "Sample.elements" in self.metadata:
            extra_elements = (set(self.metadata.Sample.elements) -
                              elements)
            if extra_elements:
                new_lines = self._get_lines_from_elements(
                    extra_elements,
                    only_one=only_one,
                    only_lines=only_lines)
                if new_lines:
                    self.add_lines(list(new_lines) + list(lines))
        self.add_elements(elements)
        if not hasattr(self.metadata, 'Sample'):
            self.metadata.add_node('Sample')
        if "Sample.xray_lines" in self.metadata:
            xray_lines = xray_lines.union(
                self.metadata.Sample.xray_lines)
        self.metadata.Sample.xray_lines = sorted(list(xray_lines))

    def _get_lines_from_elements(self,
                                 elements,
                                 only_one=False,
                                 only_lines=("Ka", "La", "Ma")):
        """Returns the X-ray lines of the given elements in spectral range
        of the data.

        Parameters
        ----------
        elements : list of strings
            A list containing the symbol of the chemical elements.
        only_one : bool
            If False, add all the lines of each element in the data spectral
            range. If True only add the line at the highest energy
            above an overvoltage of 2 (< beam energy / 2).
        only_lines : {None, list of strings}
            If not None, only the given lines will be returned.


        Returns
        -------

        """

        beam_energy = self._get_beam_energy()

        end_energy = self.axes_manager.signal_axes[0].high_value
        if beam_energy < end_energy:
            end_energy = beam_energy
        lines = []
        for element in elements:
            # Possible line (existing and excited by electron)
            element_lines = []
            for subshell in elements_db[element]['Atomic_properties']['Xray_lines'].keys():
                if only_lines and subshell not in only_lines:
                    continue
                if (self._get_line_energy(element + '_' + subshell) < end_energy):
                    element_lines.append(element + "_" + subshell)
            if only_one and element_lines:
            # Choose the best line
                select_this = -1
                for i, line in enumerate(element_lines):
                    if (self._get_line_energy(line) < beam_energy / 2):
                        select_this = i
                        break
                element_lines = [element_lines[select_this], ]

            if not element_lines:
                print(("There is not X-ray line for element %s " % element) +
                      "in the data spectral range")
            else:
                lines.extend(element_lines)
        return lines

    def get_lines_intensity(self,
                            xray_lines=None,
                            plot_result=False,
                            integration_window_factor=2.,
                            only_one=True,
                            only_lines=("Ka", "La", "Ma"),
                            **kwargs):
        """Return the intensity map of selected Xray lines.

        The intensities, the number of X-ray counts, are computed by
        suming the spectrum over the
        different X-ray lines. The sum window width
        is calculated from the energy resolution of the detector
        defined as defined in
        `self.metadata.Acquisition_instrument.SEM.Detector.EDS.energy_resolution_MnKa` or
        `self.metadata.Acquisition_instrument.SEM.Detector.EDS.energy_resolution_MnKa`.


        Parameters
        ----------

        xray_lines: {None, "best", list of string}
            If None,
            if `mapped.parameters.Sample.elements.xray_lines` contains a
            list of lines use those.
            If `mapped.parameters.Sample.elements.xray_lines` is undefined
            or empty but `mapped.parameters.Sample.elements` is defined,
            use the same syntax as `add_line` to select a subset of lines
            for the operation.
            Alternatively, provide an iterable containing
            a list of valid X-ray lines symbols.
        plot_result : bool
            If True, plot the calculated line intensities. If the current
            object is a single spectrum it prints the result instead.
        integration_window_factor: Float
            The integration window is centered at the center of the X-ray
            line and its width is defined by this factor (2 by default)
            times the calculated FWHM of the line.
        only_one : bool
            If False, use all the lines of each element in the data spectral
            range. If True use only the line at the highest energy
            above an overvoltage of 2 (< beam energy / 2).
        only_lines : {None, list of strings}
            If not None, use only the given lines.
        kwargs
            The extra keyword arguments for plotting. See
            `utils.plot.plot_signals`

        Returns
        -------
        intensities : list
            A list containing the intensities as Signal subclasses.

        Examples
        --------

        >>> specImg.get_lines_intensity(["C_Ka", "Ta_Ma"])

        See also
        --------

        set_elements, add_elements.

        """

        if xray_lines is None:
            if 'Sample.xray_lines' in self.metadata:
                xray_lines = self.metadata.Sample.xray_lines
            elif 'Sample.elements' in self.metadata:
                xray_lines = self._get_lines_from_elements(
                    self.metadata.Sample.elements,
                    only_one=only_one,
                    only_lines=only_lines)
            else:
                raise ValueError(
                    "Not X-ray line, set them with `add_elements`")

        intensities = []
        # test 1D Spectrum (0D problem)
            #signal_to_index = self.axes_manager.navigation_dimension - 2
        for Xray_line in xray_lines:
            line_energy, line_FWHM = self._get_line_energy(Xray_line,
                                                           FWHM_MnKa='auto')
            det = integration_window_factor * line_FWHM / 2.
            img = self[..., line_energy - det:line_energy + det
                       ].integrate1D(-1)
            img.metadata.General.title = (
                'Intensity of %s at %.2f %s from %s' %
                (Xray_line,
                 line_energy,
                 self.axes_manager.signal_axes[0].units,
                 self.metadata.General.title))
            if img.axes_manager.navigation_dimension >= 2:
                img = img.as_image([0, 1])
            elif img.axes_manager.navigation_dimension == 1:
                img.axes_manager.set_signal_dimension(1)
            if plot_result and img.axes_manager.signal_dimension == 0:
                print("%s at %s %s : Intensity = %.2f"
                      % (Xray_line,
                         line_energy,
                         self.axes_manager.signal_axes[0].units,
                         img.data))
            intensities.append(img)
        if plot_result and img.axes_manager.signal_dimension != 0:
            utils.plot.plot_signals(intensities, **kwargs)
        return intensities

    def get_take_off_angle(self):
        """Calculate the take-off-angle (TOA).

        TOA is the angle with which the X-rays leave the surface towards
        the detector. Parameters are read in 'SEM.tilt_stage',
        'Acquisition_instrument.SEM.Detector.EDS.azimuth_angle' and 'SEM.Detector.EDS.elevation_angle'
         in 'metadata'.

        Returns
        -------
        take_off_angle: float (Degree)

        See also
        --------
        utils.eds.take_off_angle

        Notes
        -----
        Defined by M. Schaffer et al., Ultramicroscopy 107(8), pp 587-597 (2007)
        """
        if self.metadata.Signal.signal_type == 'EDS_SEM':
            mp = self.metadata.Acquisition_instrument.SEM
        elif self.metadata.Signal.signal_type == 'EDS_TEM':
            mp = self.metadata.Acquisition_instrument.TEM

        tilt_stage = mp.tilt_stage
        azimuth_angle = mp.Detector.EDS.azimuth_angle
        elevation_angle = mp.Detector.EDS.elevation_angle

        TOA = utils.eds.take_off_angle(tilt_stage, azimuth_angle,
                                       elevation_angle)

        return TOA
