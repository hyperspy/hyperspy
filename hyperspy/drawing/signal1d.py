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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

from hyperspy.drawing.figure import BlittedFigure
from hyperspy.drawing import utils
from hyperspy.events import Event, Events


class Signal1DFigure(BlittedFigure):

    """
    """

    def __init__(self, title=""):
        super(Signal1DFigure, self).__init__()
        self.figure = None
        self.ax = None
        self.right_ax = None
        self.ax_lines = list()
        self.right_ax_lines = list()
        self.ax_markers = list()
        self.axes_manager = None
        self.right_axes_manager = None

        # Labels
        self.xlabel = ''
        self.ylabel = ''
        self.title = title
        self.create_figure()
        self.create_axis()

        # Color cycles
        self._color_cycles = {
            'line': utils.ColorCycle(),
            'step': utils.ColorCycle(),
            'scatter': utils.ColorCycle(), }

    def create_figure(self):
        self.figure = utils.create_figure(
            window_title="Figure " + self.title if self.title
            else None)
        utils.on_figure_window_close(self.figure, self._on_close)
        self.figure.canvas.mpl_connect('draw_event', self._on_draw)

    def create_axis(self):
        self.ax = self.figure.add_subplot(111)
        self.ax.yaxis.set_animated(True)
        self.ax.xaxis.set_animated(True)
        self.ax.hspy_fig = self

    def create_right_axis(self):
        if self.ax is None:
            self.create_axis()
        if self.right_ax is None:
            self.right_ax = self.ax.twinx()
            self.right_ax.hspy_fig = self
            self.right_ax.yaxis.set_animated(True)

    def add_line(self, line, ax='left'):
        if ax == 'left':
            line.ax = self.ax
            if line.axes_manager is None:
                line.axes_manager = self.axes_manager
            self.ax_lines.append(line)
            line.sf_lines = self.ax_lines
        elif ax == 'right':
            line.ax = self.right_ax
            self.right_ax_lines.append(line)
            line.sf_lines = self.right_ax_lines
            if line.axes_manager is None:
                line.axes_manager = self.right_axes_manager
        line.axis = self.axis
        # Automatically asign the color if not defined
        if line.color is None:
            line.color = self._color_cycles[line.type]()
        # Or remove it from the color cycle if part of the cycle
        # in this round
        else:
            rgba_color = mpl.colors.colorConverter.to_rgba(line.color)
            if rgba_color in self._color_cycles[line.type].color_cycle:
                self._color_cycles[line.type].color_cycle.remove(
                    rgba_color)

    def plot(self):
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_title(self.title)
        x_axis_upper_lims = []
        x_axis_lower_lims = []
        for line in self.ax_lines:
            line.plot()
            x_axis_lower_lims.append(line.axis.axis[0])
            x_axis_upper_lims.append(line.axis.axis[-1])
        for marker in self.ax_markers:
            marker.plot()
        plt.xlim(np.min(x_axis_lower_lims), np.max(x_axis_upper_lims))
        # To be discussed
        self.axes_manager.events.indices_changed.connect(self.update, [])
        self.events.closed.connect(
            lambda: self.axes_manager.events.indices_changed.disconnect(
                self.update), [])
        if hasattr(self.figure, 'tight_layout'):
            try:
                self.figure.tight_layout()
            except:
                # tight_layout is a bit brittle, we do this just in case it
                # complains
                pass

    def _on_close(self):
        if self.figure is None:
            return  # Already closed
        for line in self.ax_lines + self.right_ax_lines:
            line.close()
        super(Signal1DFigure, self)._on_close()

    def update(self):
        for marker in self.ax_markers:
            marker.update()
        for line in self.ax_lines + \
                self.right_ax_lines:
            line.update()
        # To be discussed
        # self.ax.hspy_fig._draw_animated()


class Signal1DLine(object):

    """Line that can be added to Signal1DFigure.

    Attributes
    ----------
    type : {'scatter', 'step', 'line'}
        Select the line drawing style.
    line_properties : dictionary
        Accepts a dictionary of valid (i.e. recognized by mpl.plot)
        containing valid line properties. In addition it understands
        the keyword `type` that can take the following values:
        {'scatter', 'step', 'line'}

    Methods
    -------
    set_line_properties
        Enables setting the line_properties attribute using keyword
        arguments.

    Raises
    ------
    ValueError
        If an invalid keyword value is passed to line_properties.

    """

    def __init__(self):
        self.events = Events()
        self.events.closed = Event("""
            Event that triggers when the line is closed.

            Arguments:
                obj:  Signal1DLine instance
                    The instance that triggered the event.
            """, arguments=["obj"])
        self.sf_lines = None
        self.ax = None
        # Data attributes
        self.data_function = None
        self.axis = None
        self.axes_manager = None
        self.auto_update = True
        self.get_complex = False

        # Properties
        self.line = None
        self.autoscale = False
        self.plot_indices = False
        self.text = None
        self.text_position = (-0.1, 1.05,)
        self._line_properties = {}
        self.type = "line"

    @property
    def line_properties(self):
        return self._line_properties

    @line_properties.setter
    def line_properties(self, kwargs):
        if 'type' in kwargs:
            self.type = kwargs['type']
            del kwargs['type']

        if 'color' in kwargs:
            color = kwargs['color']
            del kwargs['color']
            self.color = color

        for key, item in kwargs.items():
            if item is None and key in self._line_properties:
                del self._line_properties[key]
            else:
                self._line_properties[key] = item
        if self.line is not None:
            plt.setp(self.line, **self.line_properties)
            self.ax.figure.canvas.draw()

    def set_line_properties(self, **kwargs):
        self.line_properties = kwargs

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, value):
        lp = {}
        if value == 'scatter':
            lp['marker'] = 'o'
            lp['linestyle'] = 'None'
            lp['markersize'] = 1

        elif value == 'line':
            lp['linestyle'] = '-'
            lp['marker'] = "None"
            lp['drawstyle'] = "default"
        elif value == 'step':
            lp['drawstyle'] = 'steps-mid'
            lp['marker'] = "None"
        else:
            raise ValueError(
                "`type` must be one of "
                "{\'scatter\', \'line\', \'step\'}"
                "but %s was given" % value)
        self._type = value
        self.line_properties = lp
        if self.color is not None:
            self.color = self.color

    @property
    def color(self):
        if 'color' in self.line_properties:
            return self.line_properties['color']
        elif 'markeredgecolor' in self.line_properties:
            return self.line_properties['markeredgecolor']
        else:
            return None

    @color.setter
    def color(self, color):
        if self._type == 'scatter':
            self.set_line_properties(markeredgecolor=color)
            if 'color' in self._line_properties:
                del self._line_properties['color']
        else:
            if color is None and 'color' in self._line_properties:
                del self._line_properties['color']
            else:
                self._line_properties['color'] = color
            self.set_line_properties(markeredgecolor=None)

        if self.line is not None:
            plt.setp(self.line, **self.line_properties)
            self.ax.figure.canvas.draw()

    def plot(self, data=1):
        f = self.data_function
        if self.get_complex is False:
            data = f(axes_manager=self.axes_manager).real
        else:
            data = f(axes_manager=self.axes_manager).imag
        if self.line is not None:
            self.line.remove()
        self.line, = self.ax.plot(self.axis.axis, data,
                                  **self.line_properties)
        self.line.set_animated(True)
        self.axes_manager.events.indices_changed.connect(self.update, [])
        self.events.closed.connect(
            lambda: self.axes_manager.events.indices_changed.disconnect(
                self.update), [])
        if not self.axes_manager or self.axes_manager.navigation_size == 0:
            self.plot_indices = False
        if self.plot_indices is True:
            if self.text is not None:
                self.text.remove()
            self.text = self.ax.text(*self.text_position,
                                     s=str(self.axes_manager.indices),
                                     transform=self.ax.transAxes,
                                     fontsize=12,
                                     color=self.line.get_color(),
                                     animated=True)
        self.ax.figure.canvas.draw()

    def update(self, force_replot=False):
        """Update the current spectrum figure"""
        if self.auto_update is False:
            return
        if force_replot is True:
            self.close()
            self.plot()
        if self.get_complex is False:
            ydata = self.data_function(axes_manager=self.axes_manager).real
        else:
            ydata = self.data_function(axes_manager=self.axes_manager).imag

        old_xaxis = self.line.get_xdata()
        if len(old_xaxis) != self.axis.size or \
                np.any(np.not_equal(old_xaxis, self.axis.axis)):
            self.ax.set_xlim(self.axis.axis[0], self.axis.axis[-1])
            self.line.set_data(self.axis.axis, ydata)
        else:
            self.line.set_ydata(ydata)

        if self.autoscale is True:
            self.ax.relim()
            y1, y2 = np.searchsorted(self.axis.axis,
                                     self.ax.get_xbound())
            y2 += 2
            y1, y2 = np.clip((y1, y2), 0, len(ydata - 1))
            clipped_ydata = ydata[y1:y2]
            y_max, y_min = (np.nanmax(clipped_ydata),
                            np.nanmin(clipped_ydata))
            if self.get_complex:
                yreal = self.data_function(axes_manager=self.axes_manager).real
                clipped_yreal = yreal[y1:y2]
                y_min = min(y_min, clipped_yreal.min())
                y_max = max(y_max, clipped_yreal.max())
            self.ax.set_ylim(y_min, y_max)
        if self.plot_indices is True:
            self.text.set_text(self.axes_manager.indices)
        try:
            self.ax.hspy_fig._draw_animated()
        except:
            # There may be errors if the figure does no longer exist.
            pass

    def close(self):
        if self.line in self.ax.lines:
            self.ax.lines.remove(self.line)
        if self.text and self.text in self.ax.texts:
            self.ax.texts.remove(self.text)
        if self.sf_lines and self in self.sf_lines:
            self.sf_lines.remove(self)
        self.events.closed.trigger(obj=self)
        for f in self.events.closed.connected:
            self.events.closed.disconnect(f)
        try:
            self.ax.figure.canvas.draw()
        except:
            pass


def _plot_component(factors, idx, ax=None, cal_axis=None,
                    comp_label='PC'):
    if ax is None:
        ax = plt.gca()
    if cal_axis is not None:
        x = cal_axis.axis
        plt.xlabel(cal_axis.units)
    else:
        x = np.arange(factors.shape[0])
        plt.xlabel('Channel index')
    ax.plot(x, factors[:, idx], label='%s %i' % (comp_label, idx))
    return ax


def _plot_loading(loadings, idx, axes_manager, ax=None,
                  comp_label='PC', no_nans=True, calibrate=True,
                  cmap=plt.cm.gray):
    if ax is None:
        ax = plt.gca()
    if no_nans:
        loadings = np.nan_to_num(loadings)
    if axes_manager.navigation_dimension == 2:
        extent = None
        # get calibration from a passed axes_manager
        shape = axes_manager._navigation_shape_in_array
        if calibrate:
            extent = (axes_manager._axes[0].low_value,
                      axes_manager._axes[0].high_value,
                      axes_manager._axes[1].high_value,
                      axes_manager._axes[1].low_value)
        im = ax.imshow(loadings[idx].reshape(shape), cmap=cmap, extent=extent,
                       interpolation='nearest')
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
    elif axes_manager.navigation_dimension == 1:
        if calibrate:
            x = axes_manager._axes[0].axis
        else:
            x = np.arange(axes_manager._axes[0].size)
        ax.step(x, loadings[idx])
    else:
        raise ValueError('View not supported')
