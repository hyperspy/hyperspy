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
import itertools
import logging

import numpy as np
import warnings
import matplotlib
from matplotlib import pyplot as plt
from distutils.version import LooseVersion

from hyperspy import utils
from hyperspy.signal import BaseSignal
from hyperspy._signals.signal1d import Signal1D, LazySignal1D
from hyperspy.misc.elements import elements as elements_db
from hyperspy.misc.eds import utils as utils_eds
from hyperspy.misc.utils import isiterable
from hyperspy.utils.plot import markers


_logger = logging.getLogger(__name__)


class EDS_mixin:

    _signal_type = "EDS"

    def __init__(self, *args, **kwards):
        super().__init__(*args, **kwards)
        if self.metadata.Signal.signal_type == 'EDS':
            warnings.warn('The microscope type is not set. Use '
                          'set_signal_type(\'EDS_TEM\')  '
                          'or set_signal_type(\'EDS_SEM\')')
        self.metadata.Signal.binned = True
        self._xray_markers = {}

    def _get_line_energy(self, Xray_line, FWHM_MnKa=None):
        """
        Get the line energy and the energy resolution of a Xray line.

        The return values are in the same units than the signal axis

        Parameters
        ----------
        Xray_line : strings
            Valid element X-ray lines e.g. Fe_Kb
        FWHM_MnKa: {None, float, 'auto'}
            The energy resolution of the detector in eV
            if 'auto', used the one in
            'self.metadata.Acquisition_instrument.SEM.Detector.EDS.energy_resolution_MnKa'

        Returns
        -------
        float: the line energy, if FWHM_MnKa is None
        (float,float): the line energy and the energy resolution, if FWHM_MnKa
        is not None
        """

        units_name = self.axes_manager.signal_axes[0].units

        if FWHM_MnKa == 'auto':
            if self.metadata.Signal.signal_type == "EDS_SEM":
                FWHM_MnKa = self.metadata.Acquisition_instrument.SEM.\
                    Detector.EDS.energy_resolution_MnKa
            elif self.metadata.Signal.signal_type == "EDS_TEM":
                FWHM_MnKa = self.metadata.Acquisition_instrument.TEM.\
                    Detector.EDS.energy_resolution_MnKa
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
                line_FWHM = utils_eds.get_FWHM_at_Energy(
                    FWHM_MnKa, line_energy / 1000) * 1000
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
                % units_name)
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
                "The beam energy is not defined in `metadata`. "
                "Use `set_microscope_parameters` to set it.")

        units_name = self.axes_manager.signal_axes[0].units

        if units_name == 'eV':
            beam_energy *= 1000
        return beam_energy

    def _get_xray_lines_in_spectral_range(self, xray_lines):
        """
        Return the lines in the energy range

        Parameters
        ----------
        xray_lines: List of string
            The xray_lines

        Return
        ------
        The list of xray_lines in the energy range
        """
        ax = self.axes_manager.signal_axes[0]
        low_value = ax.low_value
        high_value = ax.high_value
        try:
            if self._get_beam_energy() < high_value:
                high_value = self._get_beam_energy()
        except AttributeError:
            # in case the beam energy is not defined in the metadata
            pass
        xray_lines_in_range = []
        xray_lines_not_in_range = []
        for xray_line in xray_lines:
            line_energy = self._get_line_energy(xray_line)
            if low_value < line_energy < high_value:
                xray_lines_in_range.append(xray_line)
            else:
                xray_lines_not_in_range.append(xray_line)
        return xray_lines_in_range, xray_lines_not_in_range

    def sum(self, axis=None, out=None):
        if axis is None:
            axis = self.axes_manager.navigation_axes
        # modify time spend per spectrum
        s = super().sum(axis=axis, out=out)
        s = out or s
        mp = None
        if s.metadata.get_item("Acquisition_instrument.SEM"):
            mp = s.metadata.Acquisition_instrument.SEM
            mp_old = self.metadata.Acquisition_instrument.SEM
        elif s.metadata.get_item("Acquisition_instrument.TEM"):
            mp = s.metadata.Acquisition_instrument.TEM
            mp_old = self.metadata.Acquisition_instrument.TEM
        if mp is not None and mp.has_item('Detector.EDS.live_time'):
            mp.Detector.EDS.live_time = mp_old.Detector.EDS.live_time * \
                self.data.size / s.data.size
        if out is None:
            return s
    sum.__doc__ = Signal1D.sum.__doc__

    def rebin(self, new_shape=None, scale=None, crop=True, out=None):
        factors = self._validate_rebin_args_and_get_factors(
            new_shape=new_shape,
            scale=scale,)
        m = super().rebin(new_shape=new_shape, scale=scale, crop=crop, out=out)
        m = out or m
        time_factor = np.prod([factors[axis.index_in_array]
                               for axis in m.axes_manager.navigation_axes])
        aimd = m.metadata.Acquisition_instrument
        if "Acquisition_instrument.SEM.Detector.EDS.real_time" in m.metadata:
            aimd.SEM.Detector.EDS.real_time *= time_factor
        elif "Acquisition_instrument.TEM.Detector.EDS.real_time" in m.metadata:
            aimd.TEM.Detector.EDS.real_time *= time_factor
        else:
            _logger.info(
                "real_time could not be found in the metadata and has not been updated.")
        if "Acquisition_instrument.SEM.Detector.EDS.live_time" in m.metadata:
            aimd.SEM.Detector.EDS.live_time *= time_factor
        elif "Acquisition_instrument.TEM.Detector.EDS.live_time" in m.metadata:
            aimd.TEM.Detector.EDS.live_time *= time_factor
        else:
            _logger.info(
                "Live_time could not be found in the metadata and has not been updated.")

        if out is None:
            return m
        else:
            out.events.data_changed.trigger(obj=out)
        return m

    rebin.__doc__ = BaseSignal.rebin.__doc__

    def set_elements(self, elements):
        """Erase all elements and set them.

        Parameters
        ----------
        elements : list of strings
            A list of chemical element symbols.

        See also
        --------
        add_elements, set_lines, add_lines

        Examples
        --------
        >>> s = hs.datasets.example_signals.EDS_SEM_Spectrum()
        >>> print(s.metadata.Sample.elements)
        >>> s.set_elements(['Al'])
        >>> print(s.metadata.Sample.elements)
        ['Al' 'C' 'Cu' 'Mn' 'Zr']
        ['Al']

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

        Examples
        --------
        >>> s = hs.datasets.example_signals.EDS_SEM_Spectrum()
        >>> print(s.metadata.Sample.elements)
        >>> s.add_elements(['Ar'])
        >>> print(s.metadata.Sample.elements)
        ['Al' 'C' 'Cu' 'Mn' 'Zr']
        ['Al', 'Ar', 'C', 'Cu', 'Mn', 'Zr']

        See also
        --------
        set_elements, add_lines, set_lines

        """
        if not isiterable(elements) or isinstance(elements, str):
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
        self.metadata.set_item('Sample.elements', sorted(list(elements_)))

    def _get_xray_lines(self, xray_lines=None, only_one=None,
                        only_lines=('a',)):
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
                    "Not X-ray line, set them with `add_elements`.")
        return xray_lines

    def set_lines(self,
                  lines,
                  only_one=True,
                  only_lines=('a',)):
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

        Examples
        --------
        >>> s = hs.datasets.example_signals.EDS_SEM_Spectrum()
        >>> s.add_lines()
        >>> print(s.metadata.Sample.xray_lines)
        >>> s.set_lines(['Cu_Ka'])
        >>> print(s.metadata.Sample.xray_lines)
        ['Al_Ka', 'C_Ka', 'Cu_La', 'Mn_La', 'Zr_La']
        ['Al_Ka', 'C_Ka', 'Cu_Ka', 'Mn_La', 'Zr_La']

        See also
        --------
        add_lines, add_elements, set_elements

        """
        only_lines = utils_eds._parse_only_lines(only_lines)
        if "Sample.xray_lines" in self.metadata:
            del self.metadata.Sample.xray_lines
        self.add_lines(lines=lines,
                       only_one=only_one,
                       only_lines=only_lines)

    def add_lines(self,
                  lines=(),
                  only_one=True,
                  only_lines=("a",)):
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

        Examples
        --------
        >>> s = hs.datasets.example_signals.EDS_SEM_Spectrum()
        >>> s.add_lines()
        >>> print(s.metadata.Sample.xray_lines)
        ['Al_Ka', 'C_Ka', 'Cu_La', 'Mn_La', 'Zr_La']

        >>> s = hs.datasets.example_signals.EDS_SEM_Spectrum()
        >>> s.set_microscope_parameters(beam_energy=30)
        >>> s.add_lines()
        >>> print(s.metadata.Sample.xray_lines)
        ['Al_Ka', 'C_Ka', 'Cu_Ka', 'Mn_Ka', 'Zr_La']

        >>> s = hs.datasets.example_signals.EDS_SEM_Spectrum()
        >>> s.add_lines()
        >>> print(s.metadata.Sample.xray_lines)
        >>> s.add_lines(['Cu_Ka'])
        >>> print(s.metadata.Sample.xray_lines)
        ['Al_Ka', 'C_Ka', 'Cu_La', 'Mn_La', 'Zr_La']
        ['Al_Ka', 'C_Ka', 'Cu_Ka', 'Cu_La', 'Mn_La', 'Zr_La']

        See also
        --------
        set_lines, add_elements, set_elements

        """
        only_lines = utils_eds._parse_only_lines(only_lines)
        if "Sample.xray_lines" in self.metadata:
            xray_lines = set(self.metadata.Sample.xray_lines)
        else:
            xray_lines = set()
        # Define the elements which Xray lines has been customized
        # So that we don't attempt to add new lines automatically
        elements = set()
        for line in xray_lines:
            elements.add(line.split("_")[0])
        for line in lines:
            try:
                element, subshell = line.split("_")
            except ValueError:
                raise ValueError(
                    "Invalid line symbol. "
                    "Please provide a valid line symbol e.g. Fe_Ka")
            if element in elements_db:
                elements.add(element)
                if subshell in elements_db[element]['Atomic_properties'
                                                    ]['Xray_lines']:
                    lines_len = len(xray_lines)
                    xray_lines.add(line)
                    if lines_len != len(xray_lines):
                        _logger.info("%s line added," % line)
                    else:
                        _logger.info("%s line already in." % line)
                else:
                    raise ValueError(
                        "%s is not a valid line of %s." % (line, element))
            else:
                raise ValueError(
                    "%s is not a valid symbol of an element." % element)
        xray_not_here = self._get_xray_lines_in_spectral_range(xray_lines)[1]
        for xray in xray_not_here:
            warnings.warn("%s is not in the data energy range." % xray)
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
                                 only_lines=("a",)):
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
        list of X-ray lines alphabetically sorted

        """

        only_lines = utils_eds._parse_only_lines(only_lines)
        try:
            beam_energy = self._get_beam_energy()
        except BaseException:
            # Fall back to the high_value of the energy axis
            beam_energy = self.axes_manager.signal_axes[0].high_value
        lines = []
        elements = [el if isinstance(el, str) else el.decode()
                    for el in elements]
        for element in elements:
            # Possible line (existing and excited by electron)
            element_lines = []
            for subshell in list(elements_db[element]['Atomic_properties'
                                                      ]['Xray_lines'].keys()):
                if only_lines and subshell not in only_lines:
                    continue
                element_lines.append(element + "_" + subshell)
            element_lines = self._get_xray_lines_in_spectral_range(
                element_lines)[0]
            if only_one and element_lines:
                # Choose the best line
                select_this = -1
                element_lines.sort()
                for i, line in enumerate(element_lines):
                    if (self._get_line_energy(line) < beam_energy / 2):
                        select_this = i
                        break
                element_lines = [element_lines[select_this], ]

            if not element_lines:
                _logger.info(
                    ("There is no X-ray line for element %s " % element) +
                    "in the data spectral range")
            else:
                lines.extend(element_lines)
        lines.sort()
        return lines

    def get_lines_intensity(self,
                            xray_lines=None,
                            integration_windows=2.,
                            background_windows=None,
                            plot_result=False,
                            only_one=True,
                            only_lines=("a",),
                            **kwargs):
        """Return the intensity map of selected Xray lines.

        The intensities, the number of X-ray counts, are computed by
        suming the spectrum over the
        different X-ray lines. The sum window width
        is calculated from the energy resolution of the detector
        as defined in 'energy_resolution_MnKa' of the metadata.
        Backgrounds average in provided windows can be subtracted from the
        intensities.

        Parameters
        ----------
        xray_lines: {None, "best", list of string}
            If None,
            if `metadata.Sample.elements.xray_lines` contains a
            list of lines use those.
            If `metadata.Sample.elements.xray_lines` is undefined
            or empty but `metadata.Sample.elements` is defined,
            use the same syntax as `add_line` to select a subset of lines
            for the operation.
            Alternatively, provide an iterable containing
            a list of valid X-ray lines symbols.
        integration_windows: Float or array
            If float, the width of the integration windows is the
            'integration_windows_width' times the calculated FWHM of the line.
            Else provide an array for which each row corresponds to a X-ray
            line. Each row contains the left and right value of the window.
        background_windows: None or 2D array of float
            If None, no background subtraction. Else, the backgrounds average
            in the windows are subtracted from the return intensities.
            'background_windows' provides the position of the windows in
            energy. Each line corresponds to a X-ray line. In a line, the two
            first values correspond to the limits of the left window and the
            two last values correspond to the limits of the right window.
        plot_result : bool
            If True, plot the calculated line intensities. If the current
            object is a single spectrum it prints the result instead.
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
            A list containing the intensities as BaseSignal subclasses.

        Examples
        --------
        >>> s = hs.datasets.example_signals.EDS_SEM_Spectrum()
        >>> s.get_lines_intensity(['Mn_Ka'], plot_result=True)
        Mn_La at 0.63316 keV : Intensity = 96700.00

        >>> s = hs.datasets.example_signals.EDS_SEM_Spectrum()
        >>> s.plot(['Mn_Ka'], integration_windows=2.1)
        >>> s.get_lines_intensity(['Mn_Ka'],
        >>>                       integration_windows=2.1, plot_result=True)
        Mn_Ka at 5.8987 keV : Intensity = 53597.00

        >>> s = hs.datasets.example_signals.EDS_SEM_Spectrum()
        >>> s.set_elements(['Mn'])
        >>> s.set_lines(['Mn_Ka'])
        >>> bw = s.estimate_background_windows()
        >>> s.plot(background_windows=bw)
        >>> s.get_lines_intensity(background_windows=bw, plot_result=True)
        Mn_Ka at 5.8987 keV : Intensity = 46716.00

        See also
        --------
        set_elements, add_elements, estimate_background_windows,
        plot

        """

        only_lines = utils_eds._parse_only_lines(only_lines)
        xray_lines = self._get_xray_lines(xray_lines, only_one=only_one,
                                          only_lines=only_lines)
        xray_lines, xray_not_here = self._get_xray_lines_in_spectral_range(
            xray_lines)
        for xray in xray_not_here:
            warnings.warn("%s is not in the data energy range." % xray +
                          "You can remove it with" +
                          "s.metadata.Sample.xray_lines.remove('%s')"
                          % xray)
        if hasattr(integration_windows, '__iter__') is False:
            integration_windows = self.estimate_integration_windows(
                windows_width=integration_windows, xray_lines=xray_lines)
        intensities = []
        ax = self.axes_manager.signal_axes[0]
        # test Signal1D (0D problem)
        # signal_to_index = self.axes_manager.navigation_dimension - 2
        for i, (Xray_line, window) in enumerate(
                zip(xray_lines, integration_windows)):
            line_energy, line_FWHM = self._get_line_energy(Xray_line,
                                                           FWHM_MnKa='auto')
            element, line = utils_eds._get_element_and_line(Xray_line)
            img = self.isig[window[0]:window[1]].integrate1D(-1)
            if np.issubdtype(img.data.dtype, np.integer):
                # The operations below require a float dtype with the default
                # numpy casting rule ('same_kind')
                img.change_dtype("float")
            if background_windows is not None:
                bw = background_windows[i]
                # TODO: test to prevent slicing bug. To be reomved when fixed
                indexes = [float(ax.value2index(de))
                           for de in list(bw) + window]
                if indexes[0] == indexes[1]:
                    bck1 = self.isig[bw[0]]
                else:
                    bck1 = self.isig[bw[0]:bw[1]].integrate1D(-1)
                if indexes[2] == indexes[3]:
                    bck2 = self.isig[bw[2]]
                else:
                    bck2 = self.isig[bw[2]:bw[3]].integrate1D(-1)
                corr_factor = (indexes[5] - indexes[4]) / (
                    (indexes[1] - indexes[0]) + (indexes[3] - indexes[2]))
                img = img - (bck1 + bck2) * corr_factor
            img.metadata.General.title = (
                'X-ray line intensity of %s: %s at %.2f %s' %
                (self.metadata.General.title,
                 Xray_line,
                 line_energy,
                 self.axes_manager.signal_axes[0].units,
                 ))
            img.axes_manager.set_signal_dimension(0)
            if plot_result and img.axes_manager.navigation_size == 1:
                print("%s at %s %s : Intensity = %.2f"
                      % (Xray_line,
                         line_energy,
                         ax.units,
                         img.data))
            img.metadata.set_item("Sample.elements", ([element]))
            img.metadata.set_item("Sample.xray_lines", ([Xray_line]))
            intensities.append(img)
        if plot_result and img.axes_manager.navigation_size != 1:
            utils.plot.plot_signals(intensities, **kwargs)
        return intensities

    def get_take_off_angle(self):
        """Calculate the take-off-angle (TOA).

        TOA is the angle with which the X-rays leave the surface towards
        the detector. Parameters are read in 'SEM.Stage.tilt_alpha',
        'Acquisition_instrument.SEM.Detector.EDS.azimuth_angle' and
        'SEM.Detector.EDS.elevation_angle' in 'metadata'.

        Returns
        -------
        take_off_angle: float
            in Degree

        Examples
        --------
        >>> s = hs.datasets.example_signals.EDS_SEM_Spectrum()
        >>> s.get_take_off_angle()
        37.0
        >>> s.set_microscope_parameters(tilt_stage=20.)
        >>> s.get_take_off_angle()
        57.0

        See also
        --------
        hs.eds.take_off_angle

        Notes
        -----
        Defined by M. Schaffer et al., Ultramicroscopy 107(8), pp 587-597
        (2007)
        """
        if self.metadata.Signal.signal_type == "EDS_SEM":
            mp = self.metadata.Acquisition_instrument.SEM
        elif self.metadata.Signal.signal_type == "EDS_TEM":
            mp = self.metadata.Acquisition_instrument.TEM

        tilt_stage = mp.Stage.tilt_alpha
        azimuth_angle = mp.Detector.EDS.azimuth_angle
        elevation_angle = mp.Detector.EDS.elevation_angle

        TOA = utils.eds.take_off_angle(tilt_stage, azimuth_angle,
                                       elevation_angle)

        return TOA

    def estimate_integration_windows(self,
                                     windows_width=2.,
                                     xray_lines=None):
        """
        Estimate a window of integration for each X-ray line.

        Parameters
        ----------
        windows_width: float
            The width of the integration windows is the 'windows_width' times
            the calculated FWHM of the line.
        xray_lines: None or list of string
            If None, use 'metadata.Sample.elements.xray_lines'. Else,
            provide an iterable containing a list of valid X-ray lines
            symbols.

        Return
        ------
        integration_windows: 2D array of float
            The positions of the windows in energy. Each row corresponds to a
            X-ray line. Each row contains the left and right value of the
            window.

        Examples
        --------
        >>> s = hs.datasets.example_signals.EDS_TEM_Spectrum()
        >>> s.add_lines()
        >>> iw = s.estimate_integration_windows()
        >>> s.plot(integration_windows=iw)
        >>> s.get_lines_intensity(integration_windows=iw, plot_result=True)
        Fe_Ka at 6.4039 keV : Intensity = 3710.00
        Pt_La at 9.4421 keV : Intensity = 15872.00

        See also
        --------
        plot, get_lines_intensity
        """
        xray_lines = self._get_xray_lines(xray_lines)
        integration_windows = []
        for Xray_line in xray_lines:
            line_energy, line_FWHM = self._get_line_energy(Xray_line,
                                                           FWHM_MnKa='auto')
            element, line = utils_eds._get_element_and_line(Xray_line)
            det = windows_width * line_FWHM / 2.
            integration_windows.append([line_energy - det, line_energy + det])
        return integration_windows

    def estimate_background_windows(self,
                                    line_width=[2, 2],
                                    windows_width=1,
                                    xray_lines=None):
        """
        Estimate two windows around each X-ray line containing only the
        background.

        Parameters
        ----------
        line_width: list of two floats
            The position of the two windows around the X-ray line is given by
            the `line_width` (left and right) times the calculated FWHM of the
            line.
        windows_width: float
            The width of the windows is is the `windows_width` times the
            calculated FWHM of the line.
        xray_lines: None or list of string
            If None, use `metadata.Sample.elements.xray_lines`. Else,
            provide an iterable containing a list of valid X-ray lines
            symbols.

        Return
        ------
        windows_position: 2D array of float
            The position of the windows in energy. Each line corresponds to a
            X-ray line. In a line, the two first values correspond to the
            limits of the left window and the two last values correspond to
            the limits of the right window.

        Examples
        --------
        >>> s = hs.datasets.example_signals.EDS_TEM_Spectrum()
        >>> s.add_lines()
        >>> bw = s.estimate_background_windows(line_width=[5.0, 2.0])
        >>> s.plot(background_windows=bw)
        >>> s.get_lines_intensity(background_windows=bw, plot_result=True)
        Fe_Ka at 6.4039 keV : Intensity = 2754.00
        Pt_La at 9.4421 keV : Intensity = 15090.00

        See also
        --------
        plot, get_lines_intensity
        """
        xray_lines = self._get_xray_lines(xray_lines)
        windows_position = []
        for xray_line in xray_lines:
            line_energy, line_FWHM = self._get_line_energy(xray_line,
                                                           FWHM_MnKa='auto')
            tmp = [
                line_energy - line_FWHM * line_width[0] -
                line_FWHM * windows_width,
                line_energy - line_FWHM * line_width[0],
                line_energy + line_FWHM * line_width[1],
                line_energy + line_FWHM * line_width[1] +
                line_FWHM * windows_width
            ]
            windows_position.append(tmp)
        windows_position = np.array(windows_position)
        # merge ovelapping windows
        index = windows_position.argsort(axis=0)[:, 0]
        for i in range(len(index) - 1):
            ia, ib = index[i], index[i + 1]
            if windows_position[ia, 2] > windows_position[ib, 0]:
                interv = np.append(windows_position[ia, :2],
                                   windows_position[ib, 2:])
                windows_position[ia] = interv
                windows_position[ib] = interv
        return windows_position

    def plot(self,
             xray_lines=False,
             only_lines=("a", "b"),
             only_one=False,
             background_windows=None,
             integration_windows=None,
             **kwargs):
        """
        Plot the EDS spectrum. The following markers can be added

        - The position of the X-ray lines and their names.
        - The background windows associated with each X-ray lines. A black line
        links the left and right window with the average value in each window.

        Parameters
        ----------
        xray_lines: {False, True, 'from_elements', list of string}
            If not False, indicate the position and the name of the X-ray
            lines.
            If True, if `metadata.Sample.elements.xray_lines` contains a
            list of lines use those. If `metadata.Sample.elements.xray_lines`
            is undefined or empty or if xray_lines equals 'from_elements' and
            `metadata.Sample.elements` is defined, use the same syntax as
            `add_line` to select a subset of lines for the operation.
            Alternatively, provide an iterable containing a list of valid X-ray
            lines symbols.
        only_lines : None or list of strings
            If not None, use only the given lines (eg. ('a','Kb')).
            If None, use all lines.
        only_one : bool
            If False, use all the lines of each element in the data spectral
            range. If True use only the line at the highest energy
            above an overvoltage of 2 (< beam energy / 2).
        background_windows: None or 2D array of float
            If not None, add markers at the position of the windows in energy.
            Each line corresponds to a X-ray lines. In a line, the two first
            value corresponds to the limit of the left window and the two
            last values corresponds to the limit of the right window.
        integration_windows: None or 'auto' or float or 2D array of float
            If not None, add markers at the position of the integration
            windows.
            If 'auto' (or float), the width of the integration windows is 2.0
            (or float) times the calculated FWHM of the line. see
            'estimate_integration_windows'.
            Else provide an array for which each row corresponds to a X-ray
            line. Each row contains the left and right value of the window.
        kwargs
            The extra keyword arguments for plot()

        Examples
        --------
        >>> s = hs.datasets.example_signals.EDS_SEM_Spectrum()
        >>> s.plot()

        >>> s = hs.datasets.example_signals.EDS_SEM_Spectrum()
        >>> s.plot(True)

        >>> s = hs.datasets.example_signals.EDS_TEM_Spectrum()
        >>> s.add_lines()
        >>> bw = s.estimate_background_windows()
        >>> s.plot(background_windows=bw)

        >>> s = hs.datasets.example_signals.EDS_SEM_Spectrum()
        >>> s.plot(['Mn_Ka'], integration_windows='auto')

        >>> s = hs.datasets.example_signals.EDS_TEM_Spectrum()
        >>> s.add_lines()
        >>> bw = s.estimate_background_windows()
        >>> s.plot(background_windows=bw, integration_windows=2.1)

        See also
        --------
        set_elements, add_elements, estimate_integration_windows,
        get_lines_intensity, estimate_background_windows
        """
        super().plot(**kwargs)
        self._plot_xray_lines(xray_lines, only_lines, only_one,
                              background_windows, integration_windows)

    def _plot_xray_lines(self, xray_lines=False, only_lines=("a", "b"),
                         only_one=False, background_windows=None,
                         integration_windows=None):
        if xray_lines is not False or\
                background_windows is not None or\
                integration_windows is not None:
            if xray_lines is False:
                xray_lines = True
            only_lines = utils_eds._parse_only_lines(only_lines)
            if xray_lines is True or xray_lines == 'from_elements':
                if 'Sample.xray_lines' in self.metadata \
                        and xray_lines != 'from_elements':
                    xray_lines = self.metadata.Sample.xray_lines
                elif 'Sample.elements' in self.metadata:
                    xray_lines = self._get_lines_from_elements(
                        self.metadata.Sample.elements,
                        only_one=only_one,
                        only_lines=only_lines)
                else:
                    _logger.warning(
                        "No elements defined, set them with `add_elements`")
                    # No X-rays lines, nothing to do then
                    return
            xray_lines, xray_not_here = self._get_xray_lines_in_spectral_range(
                xray_lines)
            for xray in xray_not_here:
                _logger.warning("%s is not in the data energy range." % xray)
            xray_lines = np.unique(xray_lines)
            self.add_xray_lines_markers(xray_lines)
            if background_windows is not None:
                self._add_background_windows_markers(background_windows)
            if integration_windows is not None:
                if integration_windows == 'auto':
                    integration_windows = 2.0
                if hasattr(integration_windows, '__iter__') is False:
                    integration_windows = self.estimate_integration_windows(
                        windows_width=integration_windows,
                        xray_lines=xray_lines)
                self._add_vertical_lines_groups(integration_windows,
                                                linestyle='--')

    def _add_vertical_lines_groups(self, position, **kwargs):
        """
        Add vertical markers for each group that shares the color.

        Parameters
        ----------
        position: 2D array of float
            The position on the signal axis. Each row corresponds to a
            group.
        kwargs
            keywords argument for markers.vertical_line
        """
        per_xray = len(position[0])
        if LooseVersion(matplotlib.__version__) >= "1.5.3":
            colors = itertools.cycle(np.sort(
                plt.rcParams['axes.prop_cycle'].by_key()["color"] * per_xray))
        else:
            colors = itertools.cycle(np.sort(
                plt.rcParams['axes.color_cycle'] * per_xray))
        for x, color in zip(np.ravel(position), colors):
            line = markers.vertical_line(x=x, color=color, **kwargs)
            self.add_marker(line, render_figure=False)
        self._render_figure(plot=['signal_plot'])

    def add_xray_lines_markers(self, xray_lines):
        """
        Add marker on a spec.plot() with the name of the selected X-ray
        lines

        Parameters
        ----------
        xray_lines: list of string
            A valid list of X-ray lines
        """

        line_energy = []
        intensity = []
        for xray_line in xray_lines:
            element, line = utils_eds._get_element_and_line(xray_line)
            line_energy.append(self._get_line_energy(xray_line))
            relative_factor = elements_db[element][
                'Atomic_properties']['Xray_lines'][line]['weight']
            a_eng = self._get_line_energy(element + '_' + line[0] + 'a')
            intensity.append(self.isig[a_eng].data * relative_factor)
        for i in range(len(line_energy)):
            line = markers.vertical_line_segment(
                x=line_energy[i], y1=None, y2=intensity[i] * 0.8)
            self.add_marker(line, render_figure=False)
            string = (r'$\mathrm{%s}_{\mathrm{%s}}$' %
                      utils_eds._get_element_and_line(xray_lines[i]))
            text = markers.text(
                x=line_energy[i], y=intensity[i] * 1.1, text=string,
                rotation=90)
            self.add_marker(text, render_figure=False)
            self._xray_markers[xray_lines[i]] = [line, text]
            line.events.closed.connect(self._xray_marker_closed)
            text.events.closed.connect(self._xray_marker_closed)
        self._render_figure(plot=['signal_plot'])

    def _xray_marker_closed(self, obj):
        marker = obj
        for xray_line, line_markers in reversed(list(
                self._xray_markers.items())):
            if marker in line_markers:
                line_markers.remove(marker)
            if not line_markers:
                self._xray_markers.pop(xray_line)

    def remove_xray_lines_markers(self, xray_lines):
        """
        Remove marker previosuly added on a spec.plot() with the name of the
        selected X-ray lines

        Parameters
        ----------
        xray_lines: list of string
            A valid list of X-ray lines to remove
        """
        for xray_line in xray_lines:
            if xray_line in self._xray_markers:
                line_markers = self._xray_markers[xray_line]
                while line_markers:
                    m = line_markers.pop()
                    m.close(render_figure=False)
        self._render_figure(plot=['signal_plot'])

    def _add_background_windows_markers(self,
                                        windows_position):
        """
        Plot the background windows associated with each X-ray lines.

        For X-ray lines, a black line links the left and right window with the
        average value in each window.

        Parameters
        ----------
        windows_position: 2D array of float
            The position of the windows in energy. Each line corresponds to a
            X-ray lines. In a line, the two first value corresponds to the
            limit of the left window and the two last values corresponds to the
            limit of the right window.

        See also
        --------
        estimate_background_windows, get_lines_intensity
        """
        self._add_vertical_lines_groups(windows_position)
        ax = self.axes_manager.signal_axes[0]
        for bw in windows_position:
            # TODO: test to prevent slicing bug. To be reomved when fixed
            if ax.value2index(bw[0]) == ax.value2index(bw[1]):
                y1 = self.isig[bw[0]].data
            else:
                y1 = self.isig[bw[0]:bw[1]].mean(-1).data
            if ax.value2index(bw[2]) == ax.value2index(bw[3]):
                y2 = self.isig[bw[2]].data
            else:
                y2 = self.isig[bw[2]:bw[3]].mean(-1).data
            line = markers.line_segment(
                x1=(bw[0] + bw[1]) / 2., x2=(bw[2] + bw[3]) / 2.,
                y1=y1, y2=y2, color='black')
            self.add_marker(line, render_figure=False)
        self._render_figure(plot=['signal_plot'])


class EDSSpectrum(EDS_mixin, Signal1D):
    pass


class LazyEDSSpectrum(EDSSpectrum, LazySignal1D):
    pass
