# -*- coding: utf-8 -*-
# Copyright 2007-2011 The Hyperspy developers
#
# This file is part of  Hyperspy.
#
#  Hyperspy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  Hyperspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  Hyperspy.  If not, see <http://www.gnu.org/licenses/>.
from __future__ import division

import numpy as np
from matplotlib import pyplot as plt

from hyperspy import utils
from hyperspy._signals.spectrum import Spectrum
from hyperspy.misc.eds.elements import elements as elements_db
from hyperspy.misc.eds import utils as utils_eds
from hyperspy.misc.utils import isiterable


class EDSSpectrum(Spectrum):
    _signal_type = "EDS"

    def __init__(self, *args, **kwards):
        Spectrum.__init__(self, *args, **kwards)
        if self.mapped_parameters.signal_type == 'EDS':
            print('The microscope type is not set. Use '
                  'set_signal_type(\'EDS_TEM\') or set_signal_type(\'EDS_SEM\')')

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
        if hasattr(self.mapped_parameters, 'SEM'):
            mp = self.mapped_parameters.SEM
        else:
            mp = self.mapped_parameters.TEM
        if hasattr(mp, 'EDS') and hasattr(mp.EDS, 'live_time'):
            mp.EDS.live_time = mp.EDS.live_time * self.axes_manager.shape[axis]
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
        if "SEM.EDS.live_time" in s.mapped_parameters:
            for factor in factors:
                s.mapped_parameters.SEM.EDS.live_time *= factor
        if "TEM.EDS.live_time" in s.mapped_parameters:
            for factor in factors:
                s.mapped_parameters.TEM.EDS.live_time *= factor
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
        >>> s.mapped_paramters.SEM.beam_energy = 10
        >>> s.set_elements(['Ni', 'O'])
        Adding Ni_La Line
        Adding O_Ka Line

        """
        # Erase previous elements and X-ray lines
        if "Sample.elements" in self.mapped_parameters:
            del self.mapped_parameters.Sample.elements
        self.add_elements(elements)

    def add_elements(self, elements):
        """Add elements and the corresponding X-ray lines.

        The list of elements is stored in `mapped_parameters.Sample.elements`

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
                "if `s` is the variable containing this EELS spectrum:\n "
                ">>> s.add_elements(('C',))\n"
                "See the docstring for more information.")
        if "Sample.elements" in self.mapped_parameters:
            elements_ = set(self.mapped_parameters.Sample.elements)
        else:
            elements_ = set()
        for element in elements:
            if element in elements_db:
                elements_.add(element)
            else:
                raise ValueError(
                    "%s is not a valid chemical element symbol." % element)

        if not hasattr(self.mapped_parameters, 'Sample'):
            self.mapped_parameters.add_node('Sample')

        self.mapped_parameters.Sample.elements = sorted(list(elements_))

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
            Additionally, if `mapped_parameters.Sample.elements` is
            defined, add the lines of those elements that where not
            given in this list.
        only_one: bool
            If False, add all the lines of each element in
            `mapped_parameters.Sample.elements` that has not line
            defined in lines. If True (default),
            only add the line at the highest energy
            above an overvoltage of 2 (< beam energy / 2).
        only_lines : {None, list of strings}
            If not None, only the given lines will be added.

        See also
        --------
        add_lines, add_elements, set_elements..

        """
        if "Sample.Xray_lines" in self.mapped_parameters:
            del self.mapped_parameters.Sample.Xray_lines
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
        `mapped_parameters.Sample.Xray_lines`

        Parameters
        ----------
        lines : list of strings
            A list of valid element X-ray lines to add e.g. Fe_Kb.
            Additionally, if `mapped_parameters.Sample.elements` is
            defined, add the lines of those elements that where not
            given in this list. If the list is empty (default), and
            `mapped_parameters.Sample.elements` is
            defined, add the lines of all those elements.
        only_one: bool
            If False, add all the lines of each element in
            `mapped_parameters.Sample.elements` that has not line
            defined in lines. If True (default),
            only add the line at the highest energy
            above an overvoltage of 2 (< beam energy / 2).
        only_lines : {None, list of strings}
            If not None, only the given lines will be added.

        See also
        --------
        set_lines, add_elements, set_elements.

        """
        if "Sample.Xray_lines" in self.mapped_parameters:
            Xray_lines = set(self.mapped_parameters.Sample.Xray_lines)
        else:
            Xray_lines = set()
        # Define the elements which Xray lines has been customized
        # So that we don't attempt to add new lines automatically
        elements = set()
        for line in Xray_lines:
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
                if subshell in elements_db[element]['Xray_energy']:
                    lines_len = len(Xray_lines)
                    Xray_lines.add(line)
                    if lines_len != len(Xray_lines):
                        print("%s line added," % line)
                    else:
                        print("%s line already in." % line)
                    if (elements_db[element]['Xray_energy'][subshell] >
                            end_energy):
                        print("Warning: %s %s is above the data energy range."
                              % (element, subshell))
                else:
                    raise ValueError(
                        "%s is not a valid line of %s." % (line, element))
            else:
                raise ValueError(
                    "%s is not a valid symbol of an element." % element)
        if "Sample.elements" in self.mapped_parameters:
            extra_elements = (set(self.mapped_parameters.Sample.elements) -
                              elements)
            if extra_elements:
                new_lines = self._get_lines_from_elements(
                    extra_elements,
                    only_one=only_one,
                    only_lines=only_lines)
                if new_lines:
                    self.add_lines(new_lines)
        self.add_elements(elements)
        if not hasattr(self.mapped_parameters, 'Sample'):
            self.mapped_parameters.add_node('Sample')
        if "Sample.Xray_lines" in self.mapped_parameters:
            Xray_lines = Xray_lines.union(
                self.mapped_parameters.Sample.Xray_lines)
        self.mapped_parameters.Sample.Xray_lines = sorted(list(Xray_lines))

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
        if hasattr(self.mapped_parameters, 'SEM') and \
                hasattr(self.mapped_parameters.SEM, 'beam_energy'):
            beam_energy = self.mapped_parameters.SEM.beam_energy
        elif hasattr(self.mapped_parameters, 'TEM') and \
                hasattr(self.mapped_parameters.TEM, 'beam_energy'):
            beam_energy = self.mapped_parameters.TEM.beam_energy
        else:
            raise AttributeError(
                "To use this method the beam energy `TEM.beam_energy` "
                "or `SEM.beam_energy` must be defined in "
                "`mapped_parameters`.")

        end_energy = self.axes_manager.signal_axes[0].high_value
        if beam_energy < end_energy:
            end_energy = beam_energy
        lines = []
        for element in elements:
            # Possible line (existing and excited by electron)
            element_lines = []
            for subshell in elements_db[element]['Xray_energy'].keys():
                if only_lines and subshell not in only_lines:
                    continue
                if (elements_db[element]['Xray_energy'][subshell] <
                        end_energy):

                    element_lines.append(element + "_" + subshell)
            if only_one and element_lines:
            # Choose the best line
                select_this = -1
                for i, line in enumerate(element_lines):
                    if (elements_db[element]['Xray_energy']
                            [line.split("_")[1]] < beam_energy / 2):
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
                            Xray_lines=None,
                            plot_result=False,
                            integration_window_factor=2.,
                            only_one=True,
                            only_lines=("Ka", "La", "Ma"),
                            **kwargs):
        """Return the intensity map of selected Xray lines.

        The intensity maps are computed by integrating the spectrum over the
        different X-ray lines. The integration window width
        is calculated from the energy resolution of the detector
        defined as defined in
        `self.mapped_parameters.SEM.EDS.energy_resolution_MnKa` or
        `self.mapped_parameters.SEM.EDS.energy_resolution_MnKa`.


        Parameters
        ----------

        Xray_lines: {None, "best", list of string}
            If None,
            if `mapped.parameters.Sample.elements.Xray_lines` contains a
            list of lines use those.
            If `mapped.parameters.Sample.elements.Xray_lines` is undefined
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

        if Xray_lines is None:
            if 'Sample.Xray_lines' in self.mapped_parameters:
                Xray_lines = self.mapped_parameters.Sample.Xray_lines
            elif 'Sample.elements' in self.mapped_parameters:
                Xray_lines = self._get_lines_from_elements(
                    self.mapped_parameters.Sample.elements,
                    only_one=only_one,
                    only_lines=only_lines)
            else:
                raise ValueError(
                    "Not X-ray line, set them with `add_elements`")

        if self.mapped_parameters.signal_type == 'EDS_SEM':
            FWHM_MnKa = self.mapped_parameters.SEM.EDS.energy_resolution_MnKa
        elif self.mapped_parameters.signal_type == 'EDS_TEM':
            FWHM_MnKa = self.mapped_parameters.TEM.EDS.energy_resolution_MnKa
        else:
            raise NotImplementedError(
                "This method only works for EDS_TEM or EDS_SEM signals. "
                "You can use `set_signal_type(\"EDS_TEM\")` or"
                "`set_signal_type(\"EDS_SEM\")` to convert to one of these"
                "signal types.")
        intensities = []
        # test 1D Spectrum (0D problem)
            #signal_to_index = self.axes_manager.navigation_dimension - 2
        for Xray_line in Xray_lines:
            element, line = utils_eds._get_element_and_line(Xray_line)
            line_energy = elements_db[element]['Xray_energy'][line]
            line_FWHM = utils_eds.get_FWHM_at_Energy(FWHM_MnKa, line_energy)
            det = integration_window_factor * line_FWHM / 2.
            img = self[..., line_energy - det:line_energy + det
                       ].integrate_simpson(-1)
            img.mapped_parameters.title = (
                'Intensity of %s at %.2f %s from %s' %
                (Xray_line,
                 line_energy,
                 self.axes_manager.signal_axes[0].units,
                 self.mapped_parameters.title))
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
        'SEM.EDS.azimuth_angle' and 'SEM.EDS.elevation_angle'
         in 'mapped_parameters'.

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
        if self.mapped_parameters.signal_type == 'EDS_SEM':
            mp = self.mapped_parameters.SEM
        elif self.mapped_parameters.signal_type == 'EDS_TEM':
            mp = self.mapped_parameters.TEM

        tilt_stage = mp.tilt_stage
        azimuth_angle = mp.EDS.azimuth_angle
        elevation_angle = mp.EDS.elevation_angle

        TOA = utils.eds.take_off_angle(tilt_stage, azimuth_angle,
                                       elevation_angle)

        return TOA

    def plot_Xray_lines(self,
                        Xray_lines=None,
                        only_one=False,
                        only_lines=("a", "b"),
                        **kwargs):
        """
        Annotate a spec.plot() with the name of the selected X-ray
        lines

        Parameters
        ----------
        Xray_lines: None or list of string
            If None,
            if `mapped.parameters.Sample.elements.Xray_lines` contains a
            list of lines use those.
            If `mapped.parameters.Sample.elements.Xray_lines` is undefined
            or empty but `mapped.parameters.Sample.elements` is defined,
            use the same syntax as `add_line` to select a subset of lines
            for the operation.
            Alternatively, provide an iterable containing
            a list of valid X-ray lines symbols.
        only_one : bool
            If False, use all the lines of each element in the data spectral
            range. If True use only the line at the highest energy
            above an overvoltage of 2 (< beam energy / 2).
        only_lines : None or list of strings
            If not None, use only the given lines (eg. ('a','Kb')).
            If None, use all lines.

        See also
        --------
        set_elements, add_elements

        """

        if only_lines is not None:
            only_lines = list(only_lines)
            for only_line in only_lines:
                if only_line == 'a':
                    only_lines.extend(['Ka', 'La', 'Ma'])
                elif only_line == 'b':
                    only_lines.extend(['Kb', 'Lb1', 'Mb'])

        if Xray_lines is None:
            if 'Sample.Xray_lines' in self.mapped_parameters:
                Xray_lines = self.mapped_parameters.Sample.Xray_lines
            elif 'Sample.elements' in self.mapped_parameters:
                Xray_lines = self._get_lines_from_elements(
                    self.mapped_parameters.Sample.elements,
                    only_one=only_one,
                    only_lines=only_lines)
            else:
                raise ValueError(
                    "Not X-ray line, set them with `add_elements`")

        line_energy = []
        intensity = []
        for Xray_line in Xray_lines:
            element, line = utils_eds._get_element_and_line(Xray_line)
            line_energy.append(elements_db[element]['Xray_energy'][line])
            relative_factor = elements_db['lines']['ratio_line'][line]
            a_eng = elements_db[element]['Xray_energy'][line[0] + 'a']
            # to improve
            # intensity.append(self[...,a_eng])
            intensity.append(self[..., a_eng].data.flatten().mean()
                             * relative_factor)

        self.plot()
        line = self._plot.signal_plot.ax_lines[0]
        #line.marker_data = [intens.data for intens in intensity]
        #line.marker_data = intensity
        for i in range(len(line_energy)):
            # f=intensity[i][self._plot.axes_manager.indices].data[0]
            #line.marker = line.ax.vlines(line_energy[i], 0, f * 0.8, color='black')
            # line.marker.set_animated(True)
            #line.marker_style = 'vline_EDS'
            #line.marker_no = i
            # line.marker = line.ax.text(line_energy[i], f * 1.1,
                # Xray_lines[i], rotation=90)
            #line.marker_style = 'text_EDS'
            #line.marker_no = i
            # line.marker.set_animated(True)
            line.marker = line.ax.vlines(
                line_energy[i],
                0,
                intensity[i] * 0.8,
                color='black')
            line.marker = line.ax.text(line_energy[i], intensity[i] * 1.1,
                                       Xray_lines[i], rotation=90)
        line.update()
        
        
        ### To go in spectrum update line. 
        #g = self.marker_data
        #if g is not None:
            #if self.marker_style is 'vline_EDS':
                ##self.marker.set_ydata(g[self.axes_manager.indices])
                #segments=self.marker.get_segments()
                #segments[0][-1,-1]=g[self.marker_no][self.axes_manager.indices[::-1]] * 0.8
                #segments[0][0,-1]=0
                ##print self.marker_no
                #self.marker.set_segments(segments)
                ##print self.marker.get_segments()
            #if self.marker_style is 'text_EDS':
                #position=list(self.marker.get_position())
                #position[-1]=g[self.marker_no][self.axes_manager.indices[::-1]] * 1.1
                #self.marker.set_position(position)
                ##print 'a'
        #self.marker_data = None
