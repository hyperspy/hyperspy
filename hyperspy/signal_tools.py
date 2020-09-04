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

import logging
import functools
import copy

import numpy as np
import scipy as sp
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.text as mpl_text
import traits.api as t

from hyperspy import drawing
from hyperspy.docstrings.signal import HISTOGRAM_MAX_BIN_ARGS
from hyperspy.exceptions import SignalDimensionError
from hyperspy.axes import AxesManager, DataAxis
from hyperspy.drawing.widgets import VerticalLineWidget
from hyperspy import components1d
from hyperspy.component import Component
from hyperspy.ui_registry import add_gui_method
from hyperspy.misc.test_utils import ignore_warning
from hyperspy.misc.label_position import SpectrumLabelPosition
from hyperspy.misc.eels.tools import get_edges_near_energy, get_info_from_edges
from hyperspy.drawing.figure import BlittedFigure
from hyperspy.misc.array_tools import numba_histogram


_logger = logging.getLogger(__name__)


class SpanSelectorInSignal1D(t.HasTraits):
    ss_left_value = t.Float(np.nan)
    ss_right_value = t.Float(np.nan)
    is_ok = t.Bool(False)

    def __init__(self, signal):
        if signal.axes_manager.signal_dimension != 1:
            raise SignalDimensionError(
                signal.axes_manager.signal_dimension, 1)

        self.signal = signal
        self.axis = self.signal.axes_manager.signal_axes[0]
        self.span_selector = None
        if signal._plot is None or not signal._plot.is_active:
            signal.plot()
        self.span_selector_switch(on=True)

    def on_disabling_span_selector(self):
        pass

    def span_selector_switch(self, on):
        if not self.signal._plot.is_active:
            return

        if on is True:
            self.span_selector = \
                drawing.widgets.ModifiableSpanSelector(
                    self.signal._plot.signal_plot.ax,
                    onselect=self.update_span_selector_traits,
                    onmove_callback=self.update_span_selector_traits,)

        elif self.span_selector is not None:
            self.on_disabling_span_selector()
            self.span_selector.turn_off()
            self.span_selector = None

    def update_span_selector_traits(self, *args, **kwargs):
        if not self.signal._plot.is_active:
            return
        x0 = self.span_selector.rect.get_x()
        if x0 < self.axis.low_value:
            x0 = self.axis.low_value
        self.ss_left_value = x0
        x1 = self.ss_left_value + self.span_selector.rect.get_width()
        if x1 > self.axis.high_value:
            x1 = self.axis.high_value
        self.ss_right_value = x1

    def reset_span_selector(self):
        self.span_selector_switch(False)
        self.ss_left_value = np.nan
        self.ss_right_value = np.nan
        self.span_selector_switch(True)

    @property
    def is_span_selector_valid(self):
        return (not np.isnan(self.ss_left_value) and
                not np.isnan(self.ss_right_value) and
                self.ss_left_value <= self.ss_right_value)

class LineInSignal1D(t.HasTraits):

    """Adds a vertical draggable line to a spectrum that reports its
    position to the position attribute of the class.

    Attributes
    ----------
    position : float
        The position of the vertical line in the one dimensional signal. Moving
        the line changes the position but the reverse is not true.
    on : bool
        Turns on and off the line
    color : wx.Colour
        The color of the line. It automatically redraws the line.

    """
    position = t.Float()
    is_ok = t.Bool(False)
    on = t.Bool(False)
    # The following is disabled because as of traits 4.6 the Color trait
    # imports traitsui (!)
    # try:
    #     color = t.Color("black")
    # except ModuleNotFoundError:  # traitsui is not installed
    #     pass
    color_str = t.Str("black")

    def __init__(self, signal):
        if signal.axes_manager.signal_dimension != 1:
            raise SignalDimensionError(
                signal.axes_manager.signal_dimension, 1)

        self.signal = signal
        self.signal.plot()
        axis_dict = signal.axes_manager.signal_axes[0].get_axis_dictionary()
        am = AxesManager([axis_dict, ])
        am._axes[0].navigate = True
        # Set the position of the line in the middle of the spectral
        # range by default
        am._axes[0].index = int(round(am._axes[0].size / 2))
        self.axes_manager = am
        self.axes_manager.events.indices_changed.connect(
            self.update_position, [])
        self.on_trait_change(self.switch_on_off, 'on')

    def draw(self):
        self.signal._plot.signal_plot.figure.canvas.draw_idle()

    def switch_on_off(self, obj, trait_name, old, new):
        if not self.signal._plot.is_active:
            return

        if new is True and old is False:
            self._line = VerticalLineWidget(self.axes_manager)
            self._line.set_mpl_ax(self.signal._plot.signal_plot.ax)
            self._line.patch.set_linewidth(2)
            self._color_changed("black", "black")
            # There is not need to call draw because setting the
            # color calls it.

        elif new is False and old is True:
            self._line.close()
            self._line = None
            self.draw()

    def update_position(self, *args, **kwargs):
        if not self.signal._plot.is_active:
            return
        self.position = self.axes_manager.coordinates[0]

    def _color_changed(self, old, new):
        if self.on is False:
            return

        self._line.patch.set_color((self.color.Red() / 255.,
                                    self.color.Green() / 255.,
                                    self.color.Blue() / 255.,))
        self.draw()


@add_gui_method(toolkey="hyperspy.Signal1D.calibrate")
class Signal1DCalibration(SpanSelectorInSignal1D):
    left_value = t.Float(t.Undefined, label='New left value')
    right_value = t.Float(t.Undefined, label='New right value')
    offset = t.Float()
    scale = t.Float()
    units = t.Unicode()

    def __init__(self, signal):
        super(Signal1DCalibration, self).__init__(signal)
        if signal.axes_manager.signal_dimension != 1:
            raise SignalDimensionError(
                signal.axes_manager.signal_dimension, 1)
        self.units = self.axis.units
        self.scale = self.axis.scale
        self.offset = self.axis.offset
        self.last_calibration_stored = True

    def _left_value_changed(self, old, new):
        if self.span_selector is not None and \
                self.span_selector.range is None:
            return
        else:
            self._update_calibration()

    def _right_value_changed(self, old, new):
        if self.span_selector.range is None:
            return
        else:
            self._update_calibration()

    def _update_calibration(self, *args, **kwargs):
        # If the span selector or the new range values are not defined do
        # nothing
        if np.isnan(self.ss_left_value) or np.isnan(self.ss_right_value) or\
                t.Undefined in (self.left_value, self.right_value):
            return
        lc = self.axis.value2index(self.ss_left_value)
        rc = self.axis.value2index(self.ss_right_value)
        self.offset, self.scale = self.axis.calibrate(
            (self.left_value, self.right_value), (lc, rc),
            modify_calibration=False)

    def apply(self):
        if np.isnan(self.ss_left_value) or np.isnan(self.ss_right_value):
            _logger.warning("Select a range by clicking on the signal figure "
                            "and dragging before pressing Apply.")
            return
        elif self.left_value is t.Undefined or self.right_value is t.Undefined:
            _logger.warning("Select the new left and right values before "
                            "pressing apply.")
            return
        axis = self.axis
        axis.scale = self.scale
        axis.offset = self.offset
        axis.units = self.units
        self.span_selector_switch(on=False)
        self.signal._replot()
        self.span_selector_switch(on=True)
        self.last_calibration_stored = True

@add_gui_method(toolkey="hyperspy.EELSSpectrum.print_edges_table")
class EdgesRange(SpanSelectorInSignal1D):
    units = t.Unicode()
    edges_list = t.Tuple()
    only_major = t.Bool()
    order = t.Unicode('closest')
    complementary = t.Bool(True)

    def __init__(self, signal, active=None):
        if signal.axes_manager.signal_dimension != 1:
            raise SignalDimensionError(
                signal.axes_manager.signal_dimension, 1)

        if active is None:
            super(EdgesRange, self).__init__(signal)
            self.active_edges = []
        else:
            # if active is provided, it is non-interactive mode
            # so fix the active_edges and don't initialise the span selector
            self.signal = signal
            self.axis = self.signal.axes_manager.signal_axes[0]
            self.active_edges = list(active)

        self.active_complementary_edges = []
        self.units = self.axis.units
        self.slp = SpectrumLabelPosition(self.signal)
        self.btns = []

        self._get_edges_info_within_energy_axis()

        self.signal.axes_manager.events.indices_changed.connect(self._on_figure_changed,
                                                                [])
        self.signal._plot.signal_plot.events.closed.connect(
        lambda: self.signal.axes_manager.events.indices_changed.disconnect(
        self._on_figure_changed), [])

    def _get_edges_info_within_energy_axis(self):
        mid_energy = (self.axis.low_value + self.axis.high_value) / 2
        rng = self.axis.high_value - self.axis.low_value
        self.edge_all = np.asarray(get_edges_near_energy(mid_energy, rng,
                                                         order=self.order))
        info = get_info_from_edges(self.edge_all)

        energy_all = []
        relevance_all = []
        description_all = []
        for d in info:
            onset = d['onset_energy (eV)']
            relevance = d['relevance']
            threshold = d['threshold']
            edge_ = d['edge']
            description = threshold + '. '*(threshold !='' and edge_ !='') + edge_

            energy_all.append(onset)
            relevance_all.append(relevance)
            description_all.append(description)

        self.energy_all = np.asarray(energy_all)
        self.relevance_all = np.asarray(relevance_all)
        self.description_all = np.asarray(description_all)

    def _on_figure_changed(self):
        self.slp._set_active_figure_properties()
        self._plot_labels()
        self.signal._plot.signal_plot.update()

    def update_table(self):
        figure_changed = self.slp._check_signal_figure_changed()
        if figure_changed:
            self._on_figure_changed()

        if self.span_selector is not None:
            energy_mask = (self.ss_left_value <= self.energy_all) & \
                (self.energy_all <= self.ss_right_value)
            if self.only_major:
                relevance_mask = self.relevance_all == 'Major'
            else:
                relevance_mask = np.ones(len(self.edge_all), bool)

            mask = energy_mask & relevance_mask
            self.edges_list = tuple(self.edge_all[mask])
            energy = tuple(self.energy_all[mask])
            relevance = tuple(self.relevance_all[mask])
            description = tuple(self.description_all[mask])
        else:
            self.edges_list = ()
            energy, relevance, description = (), (), ()

        self._keep_valid_edges()

        return self.edges_list, energy, relevance, description

    def _keep_valid_edges(self):
        edge_all = list(self.signal._edge_markers.keys())
        for edge in edge_all:
            if (edge not in self.edges_list):
                if edge in self.active_edges:
                    self.active_edges.remove(edge)
                elif edge in self.active_complementary_edges:
                    self.active_complementary_edges.remove(edge)
                self.signal.remove_EELS_edges_markers([edge])
            elif (edge not in self.active_edges):
                self.active_edges.append(edge)

        self.on_complementary()
        self._plot_labels()

    def update_active_edge(self, change):
        state = change['new']
        edge = change['owner'].description

        if state:
            self.active_edges.append(edge)
        else:
            if edge in self.active_edges:
                self.active_edges.remove(edge)
            if edge in self.active_complementary_edges:
                self.active_complementary_edges.remove(edge)
            self.signal.remove_EELS_edges_markers([edge])

        figure_changed = self.slp._check_signal_figure_changed()
        if figure_changed:
            self._on_figure_changed()
        self.on_complementary()
        self._plot_labels()

    def on_complementary(self):

        if self.complementary:
            self.active_complementary_edges = \
                self.signal.get_complementary_edges(self.active_edges,
                                                    self.only_major)
        else:
            self.active_complementary_edges = []

    def check_btn_state(self):

        edges = [btn.description for btn in self.btns]
        for btn in self.btns:
            edge = btn.description
            if btn.value is False:
                if edge in self.active_edges:
                    self.active_edges.remove(edge)
                    self.signal.remove_EELS_edges_markers([edge])
                if edge in self.active_complementary_edges:
                    btn.value = True

            if btn.value is True and self.complementary:
                comp = self.signal.get_complementary_edges(self.active_edges,
                                                           self.only_major)
                for cedge in comp:
                    if cedge in edges:
                        pos = edges.index(cedge)
                        self.btns[pos].value = True

    def _plot_labels(self, active=None, complementary=None):
        # plot selected and/or complementary edges
        if active is None:
            active = self.active_edges
        if complementary is None:
            complementary = self.active_complementary_edges

        edges_on_signal = set(self.signal._edge_markers.keys())
        edges_to_show = set(set(active).union(complementary))
        edge_keep = edges_on_signal.intersection(edges_to_show)
        edge_remove =  edges_on_signal.difference(edge_keep)
        edge_add = edges_to_show.difference(edge_keep)

        self._clear_markers(edge_remove)

        # all edges to be shown on the signal
        edge_dict = self.signal._get_edges(edges_to_show, ('Major', 'Minor'))
        vm_new, tm_new = self.slp.get_markers(edge_dict)
        for k, edge in enumerate(edge_dict.keys()):
            v = vm_new[k]
            t = tm_new[k]

            if edge in edge_keep:
                # update position of vertical line segment
                self.signal._edge_markers[edge][0].data = v.data
                self.signal._edge_markers[edge][0].update()

                # update position of text box
                self.signal._edge_markers[edge][1].data = t.data
                self.signal._edge_markers[edge][1].update()
            elif edge in edge_add:
                # first argument as dictionary for consistency
                self.signal.plot_edges_label({edge: edge_dict[edge]},
                                             vertical_line_marker=[v],
                                             text_marker=[t])

    def _clear_markers(self, edges=None):
        if edges is None:
            edges = list(self.signal._edge_markers.keys())

        self.signal.remove_EELS_edges_markers(list(edges))

        for edge in edges:
            if edge in self.active_edges:
                self.active_edges.remove(edge)
            if edge in self.active_complementary_edges:
                self.active_complementary_edges.remove(edge)

class Signal1DRangeSelector(SpanSelectorInSignal1D):
    on_close = t.List()


class Smoothing(t.HasTraits):
    # The following is disabled because as of traits 4.6 the Color trait
    # imports traitsui (!)
    # try:
    #     line_color = t.Color("blue")
    # except ModuleNotFoundError:
    #     # traitsui is required to define this trait so it is not defined when
    #     # traitsui is not installed.
    #     pass
    line_color_ipy = t.Str("blue")
    differential_order = t.Int(0)

    @property
    def line_color_rgb(self):
        if hasattr(self, "line_color"):
            try:
                # PyQt and WX
                return np.array(self.line_color.Get()) / 255.
            except AttributeError:
                try:
                    # PySide
                    return np.array(self.line_color.getRgb()) / 255.
                except BaseException:
                    return matplotlib.colors.to_rgb(self.line_color_ipy)
        else:
            return matplotlib.colors.to_rgb(self.line_color_ipy)

    def __init__(self, signal):
        self.ax = None
        self.data_line = None
        self.smooth_line = None
        self.signal = signal
        self.single_spectrum = self.signal.get_current_signal().deepcopy()
        self.axis = self.signal.axes_manager.signal_axes[0].axis
        self.plot()

    def plot(self):
        if self.signal._plot is None or not self.signal._plot.is_active:
            self.signal.plot()
        hse = self.signal._plot
        l1 = hse.signal_plot.ax_lines[0]
        self.original_color = l1.line.get_color()
        l1.set_line_properties(color=self.original_color,
                               type='scatter')
        l2 = drawing.signal1d.Signal1DLine()
        l2.data_function = self.model2plot

        l2.set_line_properties(
            color=self.line_color_rgb,
            type='line')
        # Add the line to the figure
        hse.signal_plot.add_line(l2)
        l2.plot()
        self.data_line = l1
        self.smooth_line = l2
        self.smooth_diff_line = None

    def update_lines(self):
        self.smooth_line.update()
        if self.smooth_diff_line is not None:
            self.smooth_diff_line.update()

    def turn_diff_line_on(self, diff_order):

        self.signal._plot.signal_plot.create_right_axis()
        self.smooth_diff_line = drawing.signal1d.Signal1DLine()
        self.smooth_diff_line.axes_manager = self.signal.axes_manager
        self.smooth_diff_line.data_function = self.diff_model2plot
        self.smooth_diff_line.set_line_properties(
            color=self.line_color_rgb,
            type='line')
        self.signal._plot.signal_plot.add_line(self.smooth_diff_line,
                                               ax='right')

    def _line_color_ipy_changed(self):
        if hasattr(self, "line_color"):
            self.line_color = str(self.line_color_ipy)
        else:
            self._line_color_changed(None, None)

    def turn_diff_line_off(self):
        if self.smooth_diff_line is None:
            return
        self.smooth_diff_line.close()
        self.smooth_diff_line = None

    def _differential_order_changed(self, old, new):
        if new == 0:
            self.turn_diff_line_off()
            return
        if old == 0:
            self.turn_diff_line_on(new)
            self.smooth_diff_line.plot()
        else:
            self.smooth_diff_line.update(force_replot=False)

    def _line_color_changed(self, old, new):
        self.smooth_line.line_properties = {
            'color': self.line_color_rgb}
        if self.smooth_diff_line is not None:
            self.smooth_diff_line.line_properties = {
                'color': self.line_color_rgb}
        try:
            # it seems that changing the properties can be done before the
            # first rendering event, which can cause issue with blitting
            self.update_lines()
        except AttributeError:
            pass

    def diff_model2plot(self, axes_manager=None):
        smoothed = np.diff(self.model2plot(axes_manager),
                           self.differential_order)
        return smoothed

    def close(self):
        if self.signal._plot.is_active:
            if self.differential_order != 0:
                self.turn_diff_line_off()
            self.smooth_line.close()
            self.data_line.set_line_properties(
                color=self.original_color,
                type='line')


@add_gui_method(toolkey="hyperspy.Signal1D.smooth_savitzky_golay")
class SmoothingSavitzkyGolay(Smoothing):

    polynomial_order = t.Int(
        3,
        desc="The order of the polynomial used to fit the samples."
             "`polyorder` must be less than `window_length`.")

    window_length = t.Int(
        5,
        desc="`window_length` must be a positive odd integer.")

    increase_window_length = t.Button(orientation="horizontal", label="+")
    decrease_window_length = t.Button(orientation="horizontal", label="-")

    def _increase_window_length_fired(self):
        if self.window_length % 2:
            nwl = self.window_length + 2
        else:
            nwl = self.window_length + 1
        if nwl < self.signal.axes_manager[2j].size:
            self.window_length = nwl

    def _decrease_window_length_fired(self):
        if self.window_length % 2:
            nwl = self.window_length - 2
        else:
            nwl = self.window_length - 1
        if nwl > self.polynomial_order:
            self.window_length = nwl
        else:
            _logger.warning(
                "The window length must be greater than the polynomial order")

    def _polynomial_order_changed(self, old, new):
        if self.window_length <= new:
            self.window_length = new + 2 if new % 2 else new + 1
            _logger.warning(
                "Polynomial order must be < window length. "
                "Window length set to %i.", self.window_length)
        self.update_lines()

    def _window_length_changed(self, old, new):
        self.update_lines()

    def _differential_order_changed(self, old, new):
        if new > self.polynomial_order:
            self.polynomial_order += 1
            _logger.warning(
                "Differential order must be <= polynomial order. "
                "Polynomial order set to %i.", self.polynomial_order)
        super(
            SmoothingSavitzkyGolay,
            self)._differential_order_changed(
            old,
            new)

    def diff_model2plot(self, axes_manager=None):
        self.single_spectrum.data = self.signal().copy()
        self.single_spectrum.smooth_savitzky_golay(
            polynomial_order=self.polynomial_order,
            window_length=self.window_length,
            differential_order=self.differential_order)
        return self.single_spectrum.data

    def model2plot(self, axes_manager=None):
        self.single_spectrum.data = self.signal().copy()
        self.single_spectrum.smooth_savitzky_golay(
            polynomial_order=self.polynomial_order,
            window_length=self.window_length,
            differential_order=0)
        return self.single_spectrum.data

    def apply(self):
        self.signal.smooth_savitzky_golay(
            polynomial_order=self.polynomial_order,
            window_length=self.window_length,
            differential_order=self.differential_order)
        self.signal._replot()


@add_gui_method(toolkey="hyperspy.Signal1D.smooth_lowess")
class SmoothingLowess(Smoothing):
    smoothing_parameter = t.Range(low=0.001,
                                  high=0.99,
                                  value=0.1,
                                  )
    number_of_iterations = t.Range(low=1,
                                   value=1)

    def __init__(self, *args, **kwargs):
        super(SmoothingLowess, self).__init__(*args, **kwargs)

    def _smoothing_parameter_changed(self, old, new):
        if new == 0:
            self.smoothing_parameter = old
        else:
            self.update_lines()

    def _number_of_iterations_changed(self, old, new):
        self.update_lines()

    def model2plot(self, axes_manager=None):
        self.single_spectrum.data = self.signal().copy()
        self.single_spectrum.smooth_lowess(
            smoothing_parameter=self.smoothing_parameter,
            number_of_iterations=self.number_of_iterations,
            show_progressbar=False)

        return self.single_spectrum.data

    def apply(self):
        self.signal.smooth_lowess(
            smoothing_parameter=self.smoothing_parameter,
            number_of_iterations=self.number_of_iterations)
        self.signal._replot()


@add_gui_method(toolkey="hyperspy.Signal1D.smooth_total_variation")
class SmoothingTV(Smoothing):
    smoothing_parameter = t.Float(200)

    def _smoothing_parameter_changed(self, old, new):
        self.update_lines()

    def model2plot(self, axes_manager=None):
        self.single_spectrum.data = self.signal().copy()
        self.single_spectrum.smooth_tv(
            smoothing_parameter=self.smoothing_parameter,
            show_progressbar=False)

        return self.single_spectrum.data

    def apply(self):
        self.signal.smooth_tv(
            smoothing_parameter=self.smoothing_parameter)
        self.signal._replot()


@add_gui_method(toolkey="hyperspy.Signal1D.smooth_butterworth")
class ButterworthFilter(Smoothing):
    cutoff_frequency_ratio = t.Range(0.01, 1., 0.01)
    type = t.Enum('low', 'high')
    order = t.Int(2)

    def _cutoff_frequency_ratio_changed(self, old, new):
        self.update_lines()

    def _type_changed(self, old, new):
        self.update_lines()

    def _order_changed(self, old, new):
        self.update_lines()

    def model2plot(self, axes_manager=None):
        b, a = sp.signal.butter(self.order, self.cutoff_frequency_ratio,
                                self.type)
        smoothed = sp.signal.filtfilt(b, a, self.signal())
        return smoothed

    def apply(self):
        b, a = sp.signal.butter(self.order, self.cutoff_frequency_ratio,
                                self.type)
        f = functools.partial(sp.signal.filtfilt, b, a)
        self.signal.map(f)


class Load(t.HasTraits):
    filename = t.File
    lazy = t.Bool(False)


@add_gui_method(toolkey="hyperspy.Signal1D.contrast_editor")
class ImageContrastEditor(t.HasTraits):
    mpl_help = "See the matplotlib SymLogNorm for more information."
    ss_left_value = t.Float()
    ss_right_value = t.Float()
    bins = t.Int(100, desc="Number of bins used for the histogram.")
    gamma = t.Range(0.1, 3.0, 1.0)
    vmin_percentile = t.Range(0.0, 100.0, 0)
    vmax_percentile = t.Range(0.0, 100.0, 100)

    norm = t.Enum(
        'Linear',
        'Power',
        'Log',
        'Symlog',
        default='Linear')
    linthresh = t.Range(0.0, 1.0, 0.01, exclude_low=True, exclude_high=False,
                        desc="Range of value closed to zero, which are "
                        f"linearly extrapolated. {mpl_help}")
    linscale = t.Range(0.0, 10.0, 0.1, exclude_low=False, exclude_high=False,
                       desc="Number of decades to use for each half of "
                       f"the linear range. {mpl_help}")
    auto = t.Bool(True,
                  desc="Adjust automatically the display when changing "
                  "navigator indices. Unselect to keep the same display.")

    def __init__(self, image):
        super(ImageContrastEditor, self).__init__()
        self.image = image

        self.hspy_fig = BlittedFigure()
        self.hspy_fig.create_figure()
        self.create_axis()

        # self._vmin and self._vmax are used to compute the histogram
        # by default, the image display used these, except when there is a span
        # selector on the histogram
        self._vmin, self._vmax = self.image._vmin, self.image._vmax
        self.gamma = self.image.gamma
        self.linthresh = self.image.linthresh
        self.linscale = self.image.linscale
        if self.image._vmin_percentile is not None:
            self.vmin_percentile = float(
                self.image._vmin_percentile.split('th')[0])
        if self.image._vmax_percentile is not None:
            self.vmax_percentile = float(
                self.image._vmax_percentile.split('th')[0])

        # Copy the original value to be used when resetting the display
        self.vmin_original = self._vmin
        self.vmax_original = self._vmax
        self.gamma_original = self.gamma
        self.linthresh_original = self.linthresh
        self.linscale_original = self.linscale
        self.vmin_percentile_original = self.vmin_percentile
        self.vmax_percentile_original = self.vmax_percentile

        if self.image.norm == 'auto':
            self.norm = 'Linear'
        else:
            self.norm = self.image.norm.capitalize()
        self.norm_original = copy.deepcopy(self.norm)

        self.span_selector = None
        self.span_selector_switch(on=True)

        self.plot_histogram()

        if self.image.axes_manager is not None:
            self.image.axes_manager.events.indices_changed.connect(
                self._reset, [])
            self.hspy_fig.events.closed.connect(
                lambda: self.image.axes_manager.events.indices_changed.disconnect(
                    self._reset), [])

            # Disconnect update image to avoid image flickering and reconnect
            # it when necessary in the close method.
            self.image.disconnect()

    def create_axis(self):
        self.ax = self.hspy_fig.figure.add_subplot(111)
        animated = self.hspy_fig.figure.canvas.supports_blit
        self.ax.yaxis.set_animated(animated)
        self.ax.xaxis.set_animated(animated)
        self.hspy_fig.ax = self.ax

    def _gamma_changed(self, old, new):
        if self._vmin == self._vmax:
            return
        self.image.gamma = new
        if hasattr(self, "hist"):
            vmin, vmax = self._get_current_range()
            self.image.update(
                data_changed=False, auto_contrast=False, vmin=vmin, vmax=vmax)
            self.update_line()

    def _vmin_percentile_changed(self, old, new):
        if isinstance(new, str):
            new = float(new.split('th')[0])
        self.image.vmin = f"{new}th"
        # Before the tool is fully initialised
        if hasattr(self, "hist"):
            self._reset(indices_changed=False)
            self._reset_span_selector()

    def _vmax_percentile_changed(self, old, new):
        if isinstance(new, str):
            new = float(new.split('th')[0])
        self.image.vmax = f"{new}th"
        # Before the tool is fully initialised
        if hasattr(self, "hist"):
            self._reset(indices_changed=False)
            self._reset_span_selector()

    def _auto_changed(self, old, new):
        # Do something only if auto is ticked
        if new and hasattr(self, "hist"):
            self._reset(indices_changed=False)
            self._reset_span_selector()

    def _norm_changed(self, old, new):
        if hasattr(self, "hist"):
            self.image.norm = new.lower()
            self._reset(indices_changed=False)

    def _linthresh_changed(self, old, new):
        self.image.linthresh = new
        if hasattr(self, "hist"):
            self._reset(indices_changed=False)

    def _linscale_changed(self, old, new):
        self.image.linscale = new
        if hasattr(self, "hist"):
            self._reset(indices_changed=False)

    def span_selector_switch(self, on):
        if on is True:
            self.span_selector = \
                drawing.widgets.ModifiableSpanSelector(
                    self.ax,
                    onselect=self.update_span_selector,
                    onmove_callback=self.update_span_selector,
                    rectprops={"alpha":0.25, "color":'r'})
            self.span_selector.bounds_check = True

        elif self.span_selector is not None:
            self.span_selector.turn_off()
            self.span_selector = None

    def update_span_selector_traits(self, *args, **kwargs):
        self.ss_left_value = self.span_selector.rect.get_x()
        self.ss_right_value = self.ss_left_value + \
            self.span_selector.rect.get_width()

        self.update_line()

    def update_span_selector(self, *args, **kwargs):
        self.update_span_selector_traits()
        # switch off auto when using span selector
        if self.auto:
            self.auto = False
        vmin, vmax = self._get_current_range()
        self.image.update(data_changed=False, auto_contrast=False,
                          vmin=vmin, vmax=vmax)

    def _get_data(self):
        return self.image._current_data

    def _get_histogram(self, data):
        return numba_histogram(data, bins=self.bins,
                               ranges=(self._vmin, self._vmax))

    def _set_xaxis(self):
        self.xaxis = np.linspace(self._vmin, self._vmax, self.bins)
        # Set this attribute to restrict the span selector to the xaxis
        self.span_selector.step_ax = DataAxis(size=len(self.xaxis),
                                              offset=self.xaxis[1],
                                              scale=self.xaxis[1]-self.xaxis[0])

    def plot_histogram(self, max_num_bins=250):
        """Plot a histogram of the data.

        Parameters
        ----------
        %s

        Returns
        -------
        None

        """
        if self._vmin == self._vmax:
            return
        data = self._get_data()
        # masked data outside vmin/vmax
        data = np.ma.masked_outside(data, self._vmin, self._vmax).compressed()

        # Sturges rule
        sturges_bin_width = data.ptp() / (np.log2(data.size) + 1.0)

        iqr = np.subtract(*np.percentile(data, [75, 25]))
        fd_bin_width = 2.0 * iqr * data.size ** (-1.0 / 3.0)

        if fd_bin_width > 0:
            bin_width = min(fd_bin_width, sturges_bin_width)
        else:
            # limited variance: fd_bin_width may be zero
            bin_width = sturges_bin_width

        self.bins = min(int(np.ceil(data.ptp() / bin_width)), max_num_bins)

        self.hist_data = self._get_histogram(data)
        self._set_xaxis()
        self.hist = self.ax.fill_between(self.xaxis, self.hist_data,
                                         step="mid")
        self.ax.set_xlim(self._vmin, self._vmax)
        self.ax.set_ylim(0, self.hist_data.max())
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.line = self.ax.plot(*self._get_line(),
                                       color='#ff7f0e')[0]
        self.line.set_animated(self.ax.figure.canvas.supports_blit)
        plt.tight_layout(pad=0)

    plot_histogram.__doc__ %= HISTOGRAM_MAX_BIN_ARGS

    def update_histogram(self):
        if self._vmin == self._vmax:
            return
        color = self.hist.get_facecolor()
        self.hist.remove()
        self.hist_data = self._get_histogram(self._get_data())
        self.hist = self.ax.fill_between(self.xaxis, self.hist_data,
                                         step="mid", color=color)

        self.ax.set_xlim(self._vmin, self._vmax)
        if self.hist_data.max() != 0:
            self.ax.set_ylim(0, self.hist_data.max())
        self.update_line()
        self.ax.figure.canvas.draw_idle()

    def _get_line(self):
        cmin, cmax = self._get_current_range()
        xaxis = np.linspace(cmin, cmax, self.bins)
        max_hist = self.hist_data.max()
        if self.image.norm == "linear":
            values = ((xaxis-cmin)/(cmax-cmin)) * max_hist
        elif self.image.norm == "symlog":
            v = self._sym_log_transform(xaxis)
            values = (v-v[0]) / (v[-1]-v[0]) * max_hist
        elif self.image.norm == "log":
            v = np.log(xaxis)
            values = (v-v[0]) / (v[-1]-v[0]) * max_hist
        else:
            # if "auto" or "power" use the self.gamma value
            values = ((xaxis-cmin)/(cmax-cmin)) ** self.gamma * max_hist

        return xaxis, values

    def _sym_log_transform(self, arr):
        # adapted from matploltib.colors.SymLogNorm
        arr = arr.copy()
        _linscale_adj = (self.linscale / (1.0 - np.e ** -1))
        with np.errstate(invalid="ignore"):
            masked = np.abs(arr) > self.linthresh
        sign = np.sign(arr[masked])
        log = (_linscale_adj + np.log(np.abs(arr[masked]) / self.linthresh))
        log *= sign * self.linthresh
        arr[masked] = log
        arr[~masked] *= _linscale_adj

        return arr

    def update_line(self):
        if self._vmin == self._vmax:
            return
        self.line.set_data(*self._get_line())
        if self.ax.figure.canvas.supports_blit:
            self.hspy_fig._update_animated()
        else:
            self.ax.figure.canvas.draw_idle()

    def apply(self):
        if self.ss_left_value == self.ss_right_value:
            # No span selector, so we use the default vim and vmax values
            self._reset(auto=True, indices_changed=False)
        else:
            # When we apply the selected range and update the xaxis
            self._vmin, self._vmax = self._get_current_range()
            # Remove the span selector and set the new one ready to use
            self.span_selector_switch(False)
            self.span_selector_switch(True)
            self._reset(auto=False, indices_changed=False)

    def reset(self):
        # Reset the display as original
        self._reset_original_settings()
        self._reset_span_selector()

    def _reset_span_selector(self):
        if self.span_selector and self.span_selector.rect.get_x() > 0:
            # Remove the span selector and set the new one ready to use
            self.span_selector_switch(False)
            self.span_selector_switch(True)
            self._reset(indices_changed=False)

    def _reset_original_settings(self):
        if self.vmin_percentile_original is not None:
            self.vmin_percentile = self.vmin_percentile_original
        if self.vmax_percentile_original is not None:
            self.vmax_percentile = self.vmax_percentile_original
        self._vmin = self.vmin_original
        self._vmax = self.vmax_original
        self.norm = self.norm_original.capitalize()
        self.gamma = self.gamma_original
        self.linthresh = self.linthresh_original
        self.linscale = self.linscale_original

    def _get_current_range(self):
        if self.span_selector._get_span_width() != 0:
            # if we have a span selector, use it to set the display
            return self.ss_left_value, self.ss_right_value
        else:
            return self._vmin, self._vmax

    def close(self):
        # And reconnect the image if we close the ImageContrastEditor
        if self.image is not None:
            if self.auto:
                self.image.vmin = f"{self.vmin_percentile}th"
                self.image.vmax = f"{self.vmax_percentile}th"
            else:
                self.image.vmin = self._vmin
                self.image.vmax = self._vmax
            self.image.connect()
        self.hspy_fig.close()

    def _reset(self, auto=None, indices_changed=True):
        # indices_changed is used for the connection to the indices_changed
        # event of the axes_manager, which will require to update the displayed
        # image
        self.image.norm = self.norm.lower()
        if auto is None:
            auto = self.auto

        if auto:
            self.image.update(data_changed=indices_changed, auto_contrast=auto)
            self._vmin, self._vmax = self.image._vmin, self.image._vmax
            self._set_xaxis()
        else:
            vmin, vmax = self._get_current_range()
            self.image.update(data_changed=indices_changed, auto_contrast=auto,
                              vmin=vmin, vmax=vmax)

        self.update_histogram()
        self.update_span_selector_traits()

    def _show_help_fired(self):
        from pyface.message_dialog import information
        _help = _IMAGE_CONTRAST_EDITOR_HELP.replace("PERCENTILE",
                                                    _PERCENTILE_TRAITSUI)
        _ = information(None, _help, title="Help"),


_IMAGE_CONTRAST_EDITOR_HELP = \
"""
<h2>Image contrast editor</h2>
<p>This tool provides controls to adjust the contrast of the image.</p>

<h3>Basic parameters</h3>

<p><b>Auto</b>: If selected, adjust automatically the contrast when changing
nagivation axis by taking into account others parameters.</p>

PERCENTILE

<p><b>Bins</b>: Number of bins used in the histogram calculation</p>

<p><b>Norm</b>: Normalisation used to display the image.</p>

<p><b>Gamma</b>: Paramater of the power law transform, also known as gamma
correction. <i>Only available with the 'power' norm</i>.</p>


<h3>Advanced parameters</h3>

<p><b>Linear threshold</b>: Since the values close to zero tend toward infinity,
there is a need to have a range around zero that is linear.
This allows the user to specify the size of this range around zero.
<i>Only with the 'log' norm and when values <= 0 are displayed</i>.</p>

<p><b>Linear scale</b>: Since the values close to zero tend toward infinity,
there is a need to have a range around zero that is linear.
This allows the user to specify the size of this range around zero.
<i>Only with the 'log' norm and when values <= 0 are displayed</i>.</p>

<h3>Buttons</h3>

<p><b>Apply</b>: Calculate the histogram using the selected range defined by
the range selector.</p>

<p><b>Reset</b>: Reset the settings to their initial values.</p>

<p><b>OK</b>: Close this tool.</p>

"""

_PERCENTILE_TRAITSUI = \
"""<p><b>vmin percentile</b>: The percentile value defining the number of
pixels left out of the lower bounds.</p>

<p><b>vmax percentile</b>: The percentile value defining the number of
pixels left out of the upper bounds.</p>"""

_PERCENTILE_IPYWIDGETS = \
"""<p><b>vmin/vmax percentile</b>: The percentile values defining the number of
pixels left out of the lower and upper bounds.</p>"""

IMAGE_CONTRAST_EDITOR_HELP_IPYWIDGETS = _IMAGE_CONTRAST_EDITOR_HELP.replace(
    "PERCENTILE", _PERCENTILE_IPYWIDGETS)


@add_gui_method(toolkey="hyperspy.Signal1D.integrate_in_range")
class IntegrateArea(SpanSelectorInSignal1D):
    integrate = t.Button()

    def __init__(self, signal, signal_range=None):
        if signal.axes_manager.signal_dimension != 1:
            raise SignalDimensionError(
                signal.axes.signal_dimension, 1)

        self.signal = signal
        self.axis = self.signal.axes_manager.signal_axes[0]
        self.span_selector = None
        if (not hasattr(self.signal, '_plot') or self.signal._plot is None or
                not self.signal._plot.is_active):
            self.signal.plot()
        self.span_selector_switch(on=True)

    def apply(self):
        integrated_spectrum = self.signal._integrate_in_range_commandline(
            signal_range=(
                self.ss_left_value,
                self.ss_right_value)
        )
        # Replaces the original signal inplace with the new integrated spectrum
        plot = False
        if self.signal._plot and integrated_spectrum.axes_manager.shape != ():
            self.signal._plot.close()
            plot = True
        self.signal.__init__(**integrated_spectrum._to_dictionary())
        self.signal._assign_subclass()
        self.signal.axes_manager.set_signal_dimension(0)

        if plot is True:
            self.signal.plot()


@add_gui_method(toolkey="hyperspy.Signal1D.remove_background")
class BackgroundRemoval(SpanSelectorInSignal1D):
    background_type = t.Enum(
        'Doniach',
        'Exponential',
        'Gaussian',
        'Lorentzian',
        'Offset',
        'Polynomial',
        'Power law',
        'Skew normal',
        'Split Voigt',
        'Voigt',
        default='Power law')
    polynomial_order = t.Range(1, 10)
    fast = t.Bool(True,
                  desc=("Perform a fast (analytic, but possibly less accurate)"
                        " estimation of the background. Otherwise use "
                        "non-linear least squares."))
    zero_fill = t.Bool(
        False,
        desc=("Set all spectral channels lower than the lower \n"
              "bound of the fitting range to zero (this is the \n"
              "default behavior of Gatan's DigitalMicrograph). \n"
              "Otherwise leave the pre-fitting region as-is \n"
              "(useful for inspecting quality of background fit)."))
    background_estimator = t.Instance(Component)
    bg_line_range = t.Enum('from_left_range',
                           'full',
                           'ss_range',
                           default='full')
    red_chisq = t.Float(np.nan)

    def __init__(self, signal, background_type='Power law', polynomial_order=2,
                 fast=True, plot_remainder=True, zero_fill=False,
                 show_progressbar=None, model=None):
        super(BackgroundRemoval, self).__init__(signal)
        # setting the polynomial order will change the backgroud_type to
        # polynomial, so we set it before setting the background type
        self.bg_line = None
        self.rm_line = None
        self.background_estimator = None
        self.fast = fast
        self.plot_remainder = plot_remainder
        if model is None:
            from hyperspy.models.model1d import Model1D
            model = Model1D(signal)
        self.model = model
        self.polynomial_order = polynomial_order
        if background_type in ['Power Law', 'PowerLaw']:
            background_type = 'Power law'
        if background_type in ['Skew Normal', 'SkewNormal']:
            background_type = 'Skew normal'
        if background_type in ['Split voigt', 'SplitVoigt']:
            background_type = 'Split Voigt'
        self.background_type = background_type
        self.zero_fill = zero_fill
        self.show_progressbar = show_progressbar
        self.set_background_estimator()

        self.signal.axes_manager.events.indices_changed.connect(self._fit, [])

    def on_disabling_span_selector(self):
        # Disconnect event
        self.disconnect()
        if self.bg_line is not None:
            self.bg_line.close()
            self.bg_line = None
        if self.rm_line is not None:
            self.rm_line.close()
            self.rm_line = None
            self.signal._plot.signal_plot.close_right_axis()

    def set_background_estimator(self):
        if self.model is not None:
            for component in self.model:
                self.model.remove(component)
        self.background_estimator, self.bg_line_range = _get_background_estimator(
            self.background_type, self.polynomial_order)
        if self.model is not None and len(self.model) == 0:
            self.model.append(self.background_estimator)
        if not self.fast and self.is_span_selector_valid:
            self.background_estimator.estimate_parameters(
                self.signal, self.ss_left_value,
                self.ss_right_value,
                only_current=True)

    def _polynomial_order_changed(self, old, new):
        self.set_background_estimator()
        self.span_selector_changed()

    def _background_type_changed(self, old, new):
        self.set_background_estimator()
        self.span_selector_changed()

    def _fast_changed(self, old, new):
        if self.span_selector is None or not self.is_span_selector_valid:
            return
        self._fit()
        self._update_line()

    def _ss_left_value_changed(self, old, new):
        if not (np.isnan(self.ss_right_value) or np.isnan(self.ss_left_value)):
            self.span_selector_changed()

    def _ss_right_value_changed(self, old, new):
        if not (np.isnan(self.ss_right_value) or np.isnan(self.ss_left_value)):
            self.span_selector_changed()

    def create_background_line(self):
        self.bg_line = drawing.signal1d.Signal1DLine()
        self.bg_line.data_function = self.bg_to_plot
        self.bg_line.set_line_properties(
            color='blue',
            type='line',
            scaley=False)
        self.signal._plot.signal_plot.add_line(self.bg_line)
        self.bg_line.autoscale = ''
        self.bg_line.plot()

    def create_remainder_line(self):
        self.rm_line = drawing.signal1d.Signal1DLine()
        self.rm_line.data_function = self.rm_to_plot
        self.rm_line.set_line_properties(
            color='green',
            type='line',
            scaley=False)
        self.signal._plot.signal_plot.create_right_axis(color='green')
        self.signal._plot.signal_plot.add_line(self.rm_line, ax='right')
        self.rm_line.plot()

    def bg_to_plot(self, axes_manager=None, fill_with=np.nan):
        if self.bg_line_range == 'from_left_range':
            bg_array = np.zeros(self.axis.axis.shape)
            bg_array[:] = fill_with
            from_index = self.axis.value2index(self.ss_left_value)
            bg_array[from_index:] = self.background_estimator.function(
                self.axis.axis[from_index:])
            to_return = bg_array
        elif self.bg_line_range == 'full':
            to_return = self.background_estimator.function(self.axis.axis)
        elif self.bg_line_range == 'ss_range':
            bg_array = np.zeros(self.axis.axis.shape)
            bg_array[:] = fill_with
            from_index = self.axis.value2index(self.ss_left_value)
            to_index = self.axis.value2index(self.ss_right_value)
            bg_array[from_index:] = self.background_estimator.function(
                self.axis.axis[from_index:to_index])
            to_return = bg_array

        if self.signal.metadata.Signal.binned is True:
            to_return *= self.axis.scale
        return to_return

    def rm_to_plot(self, axes_manager=None, fill_with=np.nan):
        return self.signal() - self.bg_line.line.get_ydata()

    def span_selector_changed(self):
        if not self.is_span_selector_valid:
            return
        try:
            self._fit()
            self._update_line()
        except:
            pass

    def _fit(self):
        if not self.is_span_selector_valid:
            return
        # Set signal range here to set correctly the channel_switches for
        # the chisq calculation when using fast
        self.model.set_signal_range(self.ss_left_value, self.ss_right_value)
        if self.fast:
            self.background_estimator.estimate_parameters(
                self.signal, self.ss_left_value,
                self.ss_right_value,
                only_current=True)
            # Calculate chisq
            self.model._calculate_chisq()
        else:
            self.model.fit()
        self.red_chisq = self.model.red_chisq.data[
            self.model.axes_manager.indices]

    def _update_line(self):
        if self.bg_line is None:
            self.create_background_line()
        else:
            self.bg_line.update(render_figure=False, update_ylimits=False)
        if self.plot_remainder:
            if self.rm_line is None:
                self.create_remainder_line()
            else:
                self.rm_line.update(render_figure=True,
                                    update_ylimits=True)

    def apply(self):
        if not self.is_span_selector_valid:
            return
        return_model = (self.model is not None)
        result = self.signal._remove_background_cli(
            signal_range=(self.ss_left_value, self.ss_right_value),
            background_estimator=self.background_estimator,
            fast=self.fast,
            zero_fill=self.zero_fill,
            show_progressbar=self.show_progressbar,
            model=self.model,
            return_model=return_model)
        new_spectra = result[0] if return_model else result
        self.signal.data = new_spectra.data
        self.signal.events.data_changed.trigger(self)

    def disconnect(self):
        axes_manager = self.signal.axes_manager
        if self._fit in axes_manager.events.indices_changed.connected:
            axes_manager.events.indices_changed.disconnect(self._fit)


def _get_background_estimator(background_type, polynomial_order=1):
    """
    Assign 1D component to specified background type.

    Parameters
    ----------
    background_type : str
        The name of the component to model the background.
    polynomial_order : int, optional
        The polynomial order used in the polynomial component

    Raises
    ------
    ValueError
        When the background type is not a valid string.

    Returns
    -------
    background_estimator : Component1D
        The component mdeling the background.
    bg_line_range : 'full' or 'from_left_range'
        The range to draw the component (used in the BackgroundRemoval tool)

    """
    background_type = background_type.lower().replace(' ', '')
    if background_type == 'doniach':
        background_estimator = components1d.Doniach()
        bg_line_range = 'full'
    elif background_type == 'gaussian':
        background_estimator = components1d.Gaussian()
        bg_line_range = 'full'
    elif background_type == 'lorentzian':
        background_estimator = components1d.Lorentzian()
        bg_line_range = 'full'
    elif background_type == 'offset':
        background_estimator = components1d.Offset()
        bg_line_range = 'full'
    elif background_type == 'polynomial':
        with ignore_warning(message="The API of the `Polynomial` component"):
            background_estimator = components1d.Polynomial(
                 order=polynomial_order, legacy=False)
        bg_line_range = 'full'
    elif background_type == 'powerlaw':
        background_estimator = components1d.PowerLaw()
        bg_line_range = 'from_left_range'
    elif background_type == 'exponential':
        background_estimator = components1d.Exponential()
        bg_line_range = 'from_left_range'
    elif background_type == 'skewnormal':
        background_estimator = components1d.SkewNormal()
        bg_line_range = 'full'
    elif background_type == 'splitvoigt':
        background_estimator = components1d.SplitVoigt()
        bg_line_range = 'full'
    elif background_type == 'voigt':
        with ignore_warning(message="The API of the `Voigt` component"):
            background_estimator = components1d.Voigt(legacy=False)
        bg_line_range = 'full'
    else:
        raise ValueError(f"Background type '{background_type}' not recognized.")

    return background_estimator, bg_line_range


SPIKES_REMOVAL_INSTRUCTIONS = (
    "To remove spikes from the data:\n\n"

    "   1. Click \"Show derivative histogram\" to "
    "determine at what magnitude the spikes are present.\n"
    "   2. Enter a suitable threshold (lower than the "
    "lowest magnitude outlier in the histogram) in the "
    "\"Threshold\" box, which will be the magnitude "
    "from which to search. \n"
    "   3. Click \"Find next\" to find the first spike.\n"
    "   4. If desired, the width and position of the "
    "boundaries used to replace the spike can be "
    "adjusted by clicking and dragging on the displayed "
    "plot.\n "
    "   5. View the spike (and the replacement data that "
    "will be added) and click \"Remove spike\" in order "
    "to alter the data as shown. The tool will "
    "automatically find the next spike to replace.\n"
    "   6. Repeat this process for each spike throughout "
    "the dataset, until the end of the dataset is "
    "reached.\n"
    "   7. Click \"OK\" when finished to close the spikes "
    "removal tool.\n\n"

    "Note: Various settings can be configured in "
    "the \"Advanced settings\" section. Hover the "
    "mouse over each parameter for a description of what "
    "it does."

    "\n")


@add_gui_method(toolkey="hyperspy.SimpleMessage")
class SimpleMessage(t.HasTraits):
    text = t.Str

    def __init__(self, text=""):
        self.text = text


@add_gui_method(toolkey="hyperspy.Signal1D.spikes_removal_tool")
class SpikesRemoval(SpanSelectorInSignal1D):
    interpolator_kind = t.Enum(
        'Linear',
        'Spline',
        default='Linear',
        desc="the type of interpolation to use when\n"
             "replacing the signal where a spike has been replaced")
    threshold = t.Float(400, desc="the derivative magnitude threshold above\n"
                        "which to find spikes")
    click_to_show_instructions = t.Button()
    show_derivative_histogram = t.Button()
    spline_order = t.Range(1, 10, 3,
                           desc="the order of the spline used to\n"
                           "connect the reconstructed data")
    interpolator = None
    default_spike_width = t.Int(5,
                                desc="the width over which to do the interpolation\n"
                                "when removing a spike (this can be "
                                "adjusted for each\nspike by clicking "
                                     "and dragging on the display during\n"
                                     "spike replacement)")
    index = t.Int(0)
    add_noise = t.Bool(True,
                       desc="whether to add noise to the interpolated\nportion"
                       "of the spectrum. The noise properties defined\n"
                       "in the Signal metadata are used if present,"
                            "otherwise\nshot noise is used as a default")

    def __init__(self, signal, navigation_mask=None, signal_mask=None):
        super(SpikesRemoval, self).__init__(signal)
        self.interpolated_line = None
        self.coordinates = [coordinate for coordinate in
                            signal.axes_manager._am_indices_generator()
                            if (navigation_mask is None or not
                                navigation_mask[coordinate[::-1]])]
        self.signal = signal
        self.line = signal._plot.signal_plot.ax_lines[0]
        self.ax = signal._plot.signal_plot.ax
        signal._plot.auto_update_plot = False
        if len(self.coordinates) > 1:
            signal.axes_manager.indices = self.coordinates[0]
        self.index = 0
        self.argmax = None
        self.derivmax = None
        self.kind = "linear"
        self._temp_mask = np.zeros(self.signal().shape, dtype='bool')
        self.signal_mask = signal_mask
        self.navigation_mask = navigation_mask
        md = self.signal.metadata
        from hyperspy.signal import BaseSignal

        if "Signal.Noise_properties" in md:
            if "Signal.Noise_properties.variance" in md:
                self.noise_variance = md.Signal.Noise_properties.variance
                if isinstance(md.Signal.Noise_properties.variance, BaseSignal):
                    self.noise_type = "heteroscedastic"
                else:
                    self.noise_type = "white"
            else:
                self.noise_type = "shot noise"
        else:
            self.noise_type = "shot noise"

    def _threshold_changed(self, old, new):
        self.index = 0
        self.update_plot()

    def _click_to_show_instructions_fired(self):
        from pyface.message_dialog import information
        _ = information(None, SPIKES_REMOVAL_INSTRUCTIONS,
                        title="Instructions"),

    def _show_derivative_histogram_fired(self):
        self.signal._spikes_diagnosis(signal_mask=self.signal_mask,
                                      navigation_mask=self.navigation_mask)

    def detect_spike(self):
        derivative = np.diff(self.signal())
        if self.signal_mask is not None:
            derivative[self.signal_mask[:-1]] = 0
        if self.argmax is not None:
            left, right = self.get_interpolation_range()
            self._temp_mask[left:right] = True
            derivative[self._temp_mask[:-1]] = 0
        if abs(derivative.max()) >= self.threshold:
            self.argmax = derivative.argmax()
            self.derivmax = abs(derivative.max())
            return True
        else:
            return False

    def _reset_line(self):
        if self.interpolated_line is not None:
            self.interpolated_line.close()
            self.interpolated_line = None
            self.reset_span_selector()

    def find(self, back=False):
        self._reset_line()
        ncoordinates = len(self.coordinates)
        spike = self.detect_spike()
        with self.signal.axes_manager.events.indices_changed.suppress():
            while not spike and (
                    (self.index < ncoordinates - 1 and back is False) or
                    (self.index > 0 and back is True)):
                if back is False:
                    self.index += 1
                else:
                    self.index -= 1
                spike = self.detect_spike()

        if spike is False:
            m = SimpleMessage()
            m.text = 'End of dataset reached'
            try:
                m.gui()
            except (NotImplementedError, ImportError):
                # This is only available for traitsui, ipywidgets has a
                # progress bar instead.
                pass
            except ValueError as error:
                _logger.warning(error)
            self.index = 0
            self._reset_line()
            return
        else:
            minimum = max(0, self.argmax - 50)
            maximum = min(len(self.signal()) - 1, self.argmax + 50)
            thresh_label = DerivativeTextParameters(
                text=r"$\mathsf{\delta}_\mathsf{max}=$",
                color="black")
            self.ax.legend([thresh_label], [repr(int(self.derivmax))],
                           handler_map={DerivativeTextParameters:
                                        DerivativeTextHandler()},
                           loc='best')
            self.ax.set_xlim(
                self.signal.axes_manager.signal_axes[0].index2value(
                    minimum),
                self.signal.axes_manager.signal_axes[0].index2value(
                    maximum))
            self.update_plot()
            self.create_interpolation_line()

    def update_plot(self):
        if self.interpolated_line is not None:
            self.interpolated_line.close()
            self.interpolated_line = None
        self.reset_span_selector()
        self.update_spectrum_line()
        if len(self.coordinates) > 1:
            self.signal._plot.pointer._on_navigate(self.signal.axes_manager)

    def update_spectrum_line(self):
        self.line.auto_update = True
        self.line.update()
        self.line.auto_update = False

    def _index_changed(self, old, new):
        self.signal.axes_manager.indices = self.coordinates[new]
        self.argmax = None
        self._temp_mask[:] = False

    def on_disabling_span_selector(self):
        if self.interpolated_line is not None:
            self.interpolated_line.close()
            self.interpolated_line = None

    def _spline_order_changed(self, old, new):
        self.kind = self.spline_order
        self.span_selector_changed()

    def _add_noise_changed(self, old, new):
        self.span_selector_changed()

    def _interpolator_kind_changed(self, old, new):
        if new == 'linear':
            self.kind = new
        else:
            self.kind = self.spline_order
        self.span_selector_changed()

    def _ss_left_value_changed(self, old, new):
        if not (np.isnan(self.ss_right_value) or np.isnan(self.ss_left_value)):
            self.span_selector_changed()

    def _ss_right_value_changed(self, old, new):
        if not (np.isnan(self.ss_right_value) or np.isnan(self.ss_left_value)):
            self.span_selector_changed()

    def create_interpolation_line(self):
        self.interpolated_line = drawing.signal1d.Signal1DLine()
        self.interpolated_line.data_function = self.get_interpolated_spectrum
        self.interpolated_line.set_line_properties(
            color='blue',
            type='line')
        self.signal._plot.signal_plot.add_line(self.interpolated_line)
        self.interpolated_line.auto_update = False
        self.interpolated_line.autoscale = ''
        self.interpolated_line.plot()

    def get_interpolation_range(self):
        axis = self.signal.axes_manager.signal_axes[0]
        if np.isnan(self.ss_left_value) or np.isnan(self.ss_right_value):
            left = self.argmax - self.default_spike_width
            right = self.argmax + self.default_spike_width
        else:
            left = axis.value2index(self.ss_left_value)
            right = axis.value2index(self.ss_right_value)

        # Clip to the axis dimensions
        nchannels = self.signal.axes_manager.signal_shape[0]
        left = left if left >= 0 else 0
        right = right if right < nchannels else nchannels - 1

        return left, right

    def get_interpolated_spectrum(self, axes_manager=None):
        data = self.signal().copy()
        axis = self.signal.axes_manager.signal_axes[0]
        left, right = self.get_interpolation_range()
        if self.kind == 'linear':
            pad = 1
        else:
            pad = self.spline_order
        ileft = left - pad
        iright = right + pad
        ileft = np.clip(ileft, 0, len(data))
        iright = np.clip(iright, 0, len(data))
        left = int(np.clip(left, 0, len(data)))
        right = int(np.clip(right, 0, len(data)))
        if ileft == 0:
            # Extrapolate to the left
            if right == iright:
                right -= 1
            data[:right] = data[right:iright].mean()

        elif iright == len(data):
            # Extrapolate to the right
            if left == ileft:
                left += 1
            data[left:] = data[ileft:left].mean()

        else:
            # Interpolate
            x = np.hstack((axis.axis[ileft:left], axis.axis[right:iright]))
            y = np.hstack((data[ileft:left], data[right:iright]))
            intp = sp.interpolate.interp1d(x, y, kind=self.kind)
            data[left:right] = intp(axis.axis[left:right])

        # Add noise
        if self.add_noise is True:
            if self.noise_type == "white":
                data[left:right] += np.random.normal(
                    scale=np.sqrt(self.noise_variance),
                    size=right - left)
            elif self.noise_type == "heteroscedastic":
                noise_variance = self.noise_variance(
                    axes_manager=self.signal.axes_manager)[left:right]
                noise = [np.random.normal(scale=np.sqrt(item))
                         for item in noise_variance]
                data[left:right] += noise
            else:
                data[left:right] = np.random.poisson(
                    np.clip(data[left:right], 0, np.inf))

        return data

    def span_selector_changed(self):
        if self.interpolated_line is None:
            return
        else:
            self.interpolated_line.update()

    def apply(self):
        if not self.interpolated_line:  # No spike selected
            return
        self.signal()[:] = self.get_interpolated_spectrum()
        self.signal.events.data_changed.trigger(obj=self.signal)
        self.update_spectrum_line()
        self.interpolated_line.close()
        self.interpolated_line = None
        self.reset_span_selector()
        self.find()


@add_gui_method(toolkey="hyperspy.Signal2D.find_peaks")
class PeaksFinder2D(t.HasTraits):
    method = t.Enum(
        'Local max',
        'Max',
        'Minmax',
        'Zaefferer',
        'Stat',
        'Laplacian of Gaussian',
        'Difference of Gaussian',
        'Template matching',
        default='Local Max')
    # For "Local max" method
    local_max_distance = t.Range(1, 20, value=3)
    local_max_threshold = t.Range(0, 20., value=10)
    # For "Max" method
    max_alpha = t.Range(0, 6., value=3)
    max_distance = t.Range(1, 20, value=10)
    # For "Minmax" method
    minmax_distance = t.Range(0, 6., value=3)
    minmax_threshold = t.Range(0, 20., value=10)
    # For "Zaefferer" method
    zaefferer_grad_threshold = t.Range(0, 0.2, value=0.1)
    zaefferer_window_size = t.Range(2, 80, value=40)
    zaefferer_distance_cutoff = t.Range(1, 100., value=50)
    # For "Stat" method
    stat_alpha = t.Range(0, 2., value=1)
    stat_window_radius = t.Range(5, 20, value=10)
    stat_convergence_ratio = t.Range(0, 0.1, value=0.05)
    # For "Laplacian of Gaussian" method
    log_min_sigma = t.Range(0, 2., value=1)
    log_max_sigma = t.Range(0, 100., value=50)
    log_num_sigma = t.Range(0, 20., value=10)
    log_threshold = t.Range(0, 0.4, value=0.2)
    log_overlap = t.Range(0, 1., value=0.5)
    log_log_scale = t.Bool(False)
    # For "Difference of Gaussian" method
    dog_min_sigma = t.Range(0, 2., value=1)
    dog_max_sigma = t.Range(0, 100., value=50)
    dog_sigma_ratio = t.Range(0, 3.2, value=1.6)
    dog_threshold = t.Range(0, 0.4, value=0.2)
    dog_overlap = t.Range(0, 1., value=0.5)
    # For "Cross correlation" method
    xc_template = None
    xc_distance = t.Range(0, 100., value=5.)
    xc_threshold = t.Range(0, 10., value=0.5)

    random_navigation_position = t.Button()
    compute_over_navigation_axes = t.Button()

    show_navigation_sliders = t.Bool(False)

    def __init__(self, signal, method, peaks=None, **kwargs):
        self._attribute_argument_mapping_local_max = {
            'local_max_distance': 'min_distance',
            'local_max_threshold': 'threshold_abs',
            }
        self._attribute_argument_mapping_max = {
            'max_alpha': 'alpha',
            'max_distance': 'distance',
            }
        self._attribute_argument_mapping_local_minmax = {
            'minmax_distance': 'distance',
            'minmax_threshold': 'threshold',
            }
        self._attribute_argument_mapping_local_zaefferer = {
            'zaefferer_grad_threshold': 'grad_threshold',
            'zaefferer_window_size': 'window_size',
            'zaefferer_distance_cutoff': 'distance_cutoff',
            }
        self._attribute_argument_mapping_local_stat = {
            'stat_alpha': 'alpha',
            'stat_window_radius': 'window_radius',
            'stat_convergence_ratio': 'convergence_ratio',
            }
        self._attribute_argument_mapping_local_log = {
            'log_min_sigma': 'min_sigma',
            'log_max_sigma': 'max_sigma',
            'log_num_sigma': 'num_sigma',
            'log_threshold': 'threshold',
            'log_overlap': 'overlap',
            'log_log_scale': 'log_scale',
            }
        self._attribute_argument_mapping_local_dog = {
            'dog_min_sigma': 'min_sigma',
            'dog_max_sigma': 'max_sigma',
            'dog_sigma_ratio': 'sigma_ratio',
            'dog_threshold': 'threshold',
            'dog_overlap': 'overlap',
            }
        self._attribute_argument_mapping_local_xc = {
            'xc_template': 'template',
            'xc_distance': 'distance',
            'xc_threshold': 'threshold',
            }

        self._attribute_argument_mapping_dict = {
            'local_max': self._attribute_argument_mapping_local_max,
            'max': self._attribute_argument_mapping_max,
            'minmax': self._attribute_argument_mapping_local_minmax,
            'zaefferer': self._attribute_argument_mapping_local_zaefferer,
            'stat': self._attribute_argument_mapping_local_stat,
            'laplacian_of_gaussian': self._attribute_argument_mapping_local_log,
            'difference_of_gaussian': self._attribute_argument_mapping_local_dog,
            'template_matching': self._attribute_argument_mapping_local_xc,
            }

        if signal.axes_manager.signal_dimension != 2:
            raise SignalDimensionError(
                signal.axes.signal_dimension, 2)

        self._set_parameters_observer()
        self.on_trait_change(self.set_random_navigation_position,
                             'random_navigation_position')

        self.signal = signal
        self.peaks = peaks
        if self.signal._plot is None or not self.signal._plot.is_active:
            self.signal.plot()
        if self.signal.axes_manager.navigation_size > 0:
            self.show_navigation_sliders = True
            self.signal.axes_manager.events.indices_changed.connect(
                self._update_peak_finding, [])
            self.signal._plot.signal_plot.events.closed.connect(self.disconnect, [])
        # Set initial parameters:
        # As a convenience, if the template argument is provided, we keep it
        # even if the method is different, to be able to use it later.
        if 'template' in kwargs.keys():
            self.xc_template = kwargs['template']
        if method is not None:
            self.method = method.capitalize().replace('_', ' ')
        self._parse_paramaters_initial_values(**kwargs)
        self._update_peak_finding()

    def _parse_paramaters_initial_values(self, **kwargs):
        # Get the attribute to argument mapping for the current method
        arg_mapping = self._attribute_argument_mapping_dict[
            self._normalise_method_name(self.method)]
        for attr, arg in arg_mapping.items():
            if arg in kwargs.keys():
                setattr(self, attr, kwargs[arg])

    def _update_peak_finding(self, method=None):
        if method is None:
            method = self.method.lower().replace(' ', '_')
        self._find_peaks_current_index(method=method)
        self._plot_markers()

    def _method_changed(self, old, new):
        if new == 'Template matching' and self.xc_template is None:
            raise RuntimeError('The "template" argument is required.')
        self._update_peak_finding(method=new)

    def _parameter_changed(self, old, new):
        self._update_peak_finding()

    def _set_parameters_observer(self):
        for parameters_mapping in self._attribute_argument_mapping_dict.values():
            for parameter in list(parameters_mapping.keys()):
                self.on_trait_change(self._parameter_changed, parameter)

    def _get_parameters(self, method):
        # Get the attribute to argument mapping for the given method
        arg_mapping = self._attribute_argument_mapping_dict[method]
        # return argument and values as kwargs
        return {arg: getattr(self, attr) for attr, arg in arg_mapping.items()}

    def _normalise_method_name(self, method):
        return method.lower().replace(' ', '_')

    def _find_peaks_current_index(self, method):
        method = self._normalise_method_name(method)
        self.peaks.data = self.signal.find_peaks(method, current_index=True,
                                                 interactive=False,
                                                 **self._get_parameters(method))

    def _plot_markers(self):
        if self.signal._plot is not None and self.signal._plot.is_active:
            self.signal._plot.signal_plot.remove_markers(render_figure=True)
        peaks_markers = self._peaks_to_marker()
        self.signal.add_marker(peaks_markers, render_figure=True)

    def _peaks_to_marker(self, markersize=20, add_numbers=True,
                         color='red'):
        # make marker_list for current index
        from hyperspy.drawing._markers.point import Point

        x_axis = self.signal.axes_manager.signal_axes[0]
        y_axis = self.signal.axes_manager.signal_axes[1]

        marker_list = [Point(x=x_axis.index2value(x),
                             y=y_axis.index2value(y),
                             color=color,
                             size=markersize)
            for x, y in zip(self.peaks.data[:, 1], self.peaks.data[:, 0])]

        return marker_list

    def compute_navigation(self):
        method = self._normalise_method_name(self.method)
        with self.signal.axes_manager.events.indices_changed.suppress():
            self.peaks.data = self.signal.find_peaks(
                method, interactive=False, current_index=False,
                **self._get_parameters(method))

    def close(self):
        # remove markers
        if self.signal._plot is not None and self.signal._plot.is_active:
            self.signal._plot.signal_plot.remove_markers(render_figure=True)
        self.disconnect()

    def disconnect(self):
        # disconnect event
        am = self.signal.axes_manager
        if self._update_peak_finding in am.events.indices_changed.connected:
            am.events.indices_changed.disconnect(self._update_peak_finding)

    def set_random_navigation_position(self):
        index = np.random.randint(0, self.signal.axes_manager._max_index)
        self.signal.axes_manager.indices = np.unravel_index(index,
            tuple(self.signal.axes_manager._navigation_shape_in_array))[::-1]


# For creating a text handler in legend (to label derivative magnitude)
class DerivativeTextParameters(object):

    def __init__(self, text, color):
        self.my_text = text
        self.my_color = color


class DerivativeTextHandler(object):

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        patch = mpl_text.Text(
            text=orig_handle.my_text,
            color=orig_handle.my_color)
        handlebox.add_artist(patch)
        return patch
