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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import logging
import inspect
from functools import partial

from hyperspy.drawing.figure import BlittedFigure
from hyperspy.drawing import utils
from hyperspy.events import Event, Events
from hyperspy.exceptions import VisibleDeprecationWarning
from hyperspy.misc.test_utils import ignore_warning


_logger = logging.getLogger(__name__)


class Signal1DFigure(BlittedFigure):

    """
    Signal1DFigure has also an 'axis' property that has to be properly defined at initialization
    """

    def __init__(self, title=""):
        super(Signal1DFigure, self).__init__()
        self.figure = None
        self.ax = None
        self.right_ax = None
        self.ax_lines = list()
        self.right_ax_lines = list()
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
        #spines
        self.spine_spacing=1

    def create_axis(self):
        self.ax = self.figure.add_subplot(111,picker=True)
        animated = self.figure.canvas.supports_blit
        self.ax.yaxis.set_animated(animated)
        self.ax.xaxis.set_animated(animated)
        self.ax.hspy_fig = self

    def create_right_axis(self, color='black'):
        if self.ax is None:
            self.create_axis()
        if self.right_ax is None:
            self.right_ax = self.ax.twinx()
            self.right_ax.hspy_fig = self
            self.right_ax.yaxis.set_animated(self.figure.canvas.supports_blit)
            self.right_ax.tick_params(axis='y', labelcolor=color)
        plt.tight_layout()

    def close_right_axis(self):
        if self.right_ax is not None:
            for lines in self.right_ax_lines:
                lines.close()
            self.right_ax.axes.get_yaxis().set_visible(False)
            self.right_ax = None
    def make_patch_spines_invisible(self,ax):
        """
        from https://matplotlib.org/3.1.1/gallery/ticks_and_spines/multiple_yaxis_with_spines.html
        :param ax:
        :return:
        """
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    def add_line(self, line, ax='left', connect_navigation=False):
        """
        Add Signal1DLine to figure

        Parameters
        ----------
        line : Signal1DLine object
            Line to be added to the figure.
        ax : {'left', 'right'}, optional
            Position the y axis, either 'left'. The default is 'left'.
        connect_navigation : bool, optional
            Connect the update of the line to the `indices_changed` event of
            the axes_manager. This only necessary when adding a line to the
            left since the `indices_changed` event is already connected to
            the `update` method of `Signal1DFigure`. The default is False.

        Returns
        -------
        None.

        """
        if ax == 'left':
            if not self.ax_lines:
                line.ax=self.ax
                line.exponent_position=0
                line.ax.yaxis.set_animated(self.figure.canvas.supports_blit)
            else:
                line.ax = self.ax.twinx()# twinx is difficult with picker use from mpl_toolkits.axes_grid1 import host_subplot instead
                #see https://github.com/matplotlib/matplotlib/issues/5581
                #see snippet at the end of the file

       #largely inspired from https://matplotlib.org/3.1.1/gallery/ticks_and_spines/multiple_yaxis_with_spines.html

                line.ax.spines["right"].set_position(("axes", self.spine_spacing))

                line.exponent_position=self.spine_spacing
                line.ax.yaxis.offsetText.set_position((self.spine_spacing,1))
                self.spine_spacing+=0.15
                #position the exponent at the right position

                self.make_patch_spines_invisible(line.ax)

                line.ax.spines["right"].set_visible(True)

            line.ax.hspy_fig = self
            #sometimes you need to change the figure ax to the last line; otherwise, you might add a ROI to the wrong ax, then it would be responsive form the UI
            #obviously not now...
            line.ax.yaxis.set_animated(self.figure.canvas.supports_blit)
            line.ax.ticklabel_format(axis='y', style='sci',scilimits=(0,0),useOffset=True)
            line.ax.yaxis.label.set_color(line.color)
            line.ax.tick_params(axis='y', colors=line.color)

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
        if connect_navigation:
            f = partial(line._auto_update_line, update_ylimits=True)
            line.axes_manager.events.indices_changed.connect(f, [])
            line.events.closed.connect(
                lambda: line.axes_manager.events.indices_changed.disconnect(f),
                [])
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
        self.ax.figure.canvas.draw()
        if hasattr(self.figure, 'tight_layout'):
            try:
                self.figure.tight_layout()
            except BaseException:
                # tight_layout is a bit brittle, we do this just in case it
                # complains
                pass

    def plot(self, data_function_kwargs={}, **kwargs):
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_title(self.title)
        x_axis_upper_lims = []
        x_axis_lower_lims = []
        for line in self.ax_lines:
            line.plot(data_function_kwargs=data_function_kwargs, **kwargs)
            x_axis_lower_lims.append(line.axis.axis[0])
            x_axis_upper_lims.append(line.axis.axis[-1])
        for marker in self.ax_markers:
            marker.plot(render_figure=False)
        plt.xlim(np.min(x_axis_lower_lims), np.max(x_axis_upper_lims))
        self.axes_manager.events.indices_changed.connect(self.update, [])
        self.events.closed.connect(
            lambda: self.axes_manager.events.indices_changed.disconnect(
                self.update), [])

        self.ax.figure.canvas.draw_idle()
        if hasattr(self.figure, 'tight_layout'):
            try:
                self.figure.tight_layout()
            except BaseException:
                # tight_layout is a bit brittle, we do this just in case it
                # complains
                pass
        self.figure.canvas.draw()

    def _on_close(self):
        _logger.debug('Closing Signal1DFigure.')
        if self.figure is None:
            return  # Already closed
        for line in self.ax_lines + self.right_ax_lines:
            line.close()
        super(Signal1DFigure, self)._on_close()
        _logger.debug('Signal1DFigure Closed.')

    def update(self):
        """
        Update lines, markers and render at the end.
        This method is connected to the `indices_changed` event of the
        `axes_manager`.
        """

        def update_lines(ax, ax_lines):
            y_min, y_max = np.nan, np.nan
            for line in ax_lines:
                # save on figure rendering and do it at the end
                # don't update the y limits
                line._auto_update_line(render_figure=False,
                                       update_ylimits=False)
                y_min = np.nanmin([y_min, line._y_min])
                y_max = np.nanmax([y_max, line._y_max])
            ax.set_ylim(y_min, y_max)

        for marker in self.ax_markers:
            marker.update()
        # Left and right axis needs to be updated separetely to set the
        # correct y limits of each axes
        update_lines(self.ax, self.ax_lines)
        if self.right_ax is not None:
            update_lines(self.right_ax, self.right_ax_lines)
        for line in self.ax_lines + self.right_ax_lines:
            # save on figure rendering and do it at the end
            line._auto_update_line(render_figure=False)
        if self.ax.figure.canvas.supports_blit:
            self.ax.hspy_fig._update_animated()
        else:
            self.ax.figure.canvas.draw_idle()


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
    auto_update: bool
        If False, executing ``_auto_update_line`` does not update the
        line plot.

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
        # args to pass to `__call__`
        self.data_function_kwargs = {}
        self.axis = None
        self.axes_manager = None
        self._plot_imag = False
        self.norm = 'linear'

        # Properties
        self.auto_update = True
        self.autoscale = 'v'
        self._y_min = np.nan
        self._y_max = np.nan
        self.line = None
        self.autoscale = False
        self.plot_indices = False
        self.text = None
        self.text_position = (-0.1, 1.05,)
        self._line_properties = {}
        self.type = "line"
        self.exponent_position=None# a list of the x positions (ie, fitting the spine position) for the exponent
        self.time=0 # for debugging purpose
        self.axbackground=None


    @property
    def get_complex(self):
        warnings.warn("The `get_complex` attribute is deprecated and will be"
                      "removed in 2.0, please use `_plot_imag` instead.",
                      VisibleDeprecationWarning)
        return self._plot_imag

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
            self.ax.figure.canvas.draw_idle()

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
            self.ax.figure.canvas.draw_idle()

    def plot(self, data=1, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        data = self._get_data()
        if self.line is not None:
            self.line.remove()

        norm = self.norm
        if norm == 'log':
            plot = self.ax.semilogy
        elif (isinstance(norm, mpl.colors.Normalize) or
              (inspect.isclass(norm) and issubclass(norm, mpl.colors.Normalize))
              ):
            raise ValueError("Matplotlib Normalize instance or subclass can "
                             "be used for Signal2D only.")
        elif norm not in ["auto", "linear"]:
            raise ValueError("`norm` paramater should be 'auto', 'linear' or "
                             "'log' for Signal1D.")
        else:
            plot = self.ax.plot
        self.line, = plot(self.axis.axis, data, **self.line_properties)
        self.line.set_animated(self.ax.figure.canvas.supports_blit)
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
                                     animated=self.ax.figure.canvas.supports_blit)
        self._y_min, self._y_max = self.ax.get_ylim()
        self.ax.figure.canvas.draw_idle()

    def _get_data(self, real_part=False):
        if self._plot_imag and not real_part:
            ydata = self.data_function(axes_manager=self.axes_manager,
                                       **self.data_function_kwargs).imag
        else:
            ydata = self.data_function(axes_manager=self.axes_manager,
                                       **self.data_function_kwargs).real
        return ydata

    def _auto_update_line(self, update_ylimits=False, **kwargs):
        """Updates the line plot only if `auto_update` is `True`.

        This is useful to connect to events that automatically update the line.

        """
        if self.auto_update:
            if 'render_figure' not in kwargs.keys():
                # if markers are plotted, we don't render the figure now but
                # once the markers have been updated
                kwargs['render_figure'] = (
                    len(self.ax.hspy_fig.ax_markers) == 0)
            self.update(self, update_ylimits=update_ylimits, **kwargs)

    def update(self, force_replot=False, render_figure=True,
               update_ylimits=False):
        """Update the current spectrum figure

        Parameters:
        -----------
        force_replot : bool
            If True, close and open the figure. Default is False.
        render_figure : bool
            If True, render the figure. Useful to avoid firing matplotlib
            drawing events too often. Default is True.
        update_ylimits : bool
            If True, update the y-limits. This is useful to avoid the figure
            flickering when different lines update the y-limits consecutively,
            in which case, this is done in `Signal1DFigure.update`.
            Default is False.

        """
        if force_replot is True:
            self.close()
            self.plot(data_function_kwargs=self.data_function_kwargs,
                      norm=self.norm)

        ydata = self._get_data()
        old_xaxis = self.line.get_xdata()
        if len(old_xaxis) != self.axis.size or \
                np.any(np.not_equal(old_xaxis, self.axis.axis)):
            self.ax.set_xlim(self.axis.axis[0], self.axis.axis[-1])
            self.line.set_data(self.axis.axis, ydata)
        else:
            self.line.set_ydata(ydata)

        if 'x' in self.autoscale:
            self.ax.set_xlim(self.axis.axis[0], self.axis.axis[-1])

        if 'v' in self.autoscale:
            self.ax.relim()
            y1, y2 = np.searchsorted(self.axis.axis,
                                     self.ax.get_xbound())
            y2 += 2
            y1, y2 = np.clip((y1, y2), 0, len(ydata - 1))
            clipped_ydata = ydata[y1:y2]
            with ignore_warning(category=RuntimeWarning):
                # In case of "All-NaN slices"
                y_max, y_min = (np.nanmax(clipped_ydata),
                                np.nanmin(clipped_ydata))

            if self._plot_imag:
                # Add real plot
                yreal = self._get_data(real_part=True)
                clipped_yreal = yreal[y1:y2]
                with ignore_warning(category=RuntimeWarning):
                    # In case of "All-NaN slices"
                    y_min = min(y_min, np.nanmin(clipped_yreal))
                    y_max = max(y_max, np.nanmin(clipped_yreal))
            if y_min == y_max:
                # To avoid matplotlib UserWarning when calling `set_ylim`
                y_min, y_max = y_min - 0.1, y_max + 0.1
            if not np.isfinite(y_min):
                y_min = None  # data are -inf or all NaN
            if not np.isfinite(y_max):
                y_max = None  # data are inf or all NaN
            if y_min is not None:
                self._y_min = y_min
            if y_max is not None:
                self._y_max = y_max
            if update_ylimits:
                # Most of the time, we don't want to call `set_ylim` now to
                # avoid flickering of the figure. However, we use the values
                # `self._y_min` and `self._y_max` in `Signal1DFigure.update`
                self.ax.set_ylim(self._y_min, self._y_max)

        if self.plot_indices is True:
            self.text.set_text(self.axes_manager.indices)

        if render_figure:
            if self.ax.figure.canvas.supports_blit:
                #my tests showed using update_animated was longer than using draw_idle
                #self.ax.hspy_fig._update_animated()

                #In case the solution with blit does not work... use only draw_idle!
                #self.ax.figure.canvas.draw_idle()

                #the following seems to work (not sure why, but the following order of actions have to be kept this way)
                #note that the axbackground should be initialized when the corresponding ROI widget is selected
                #otherwise we have the wrong background displayed until the roi is released
                #this probably requires modifying roi.BaseInteractiveROI to add a "on_selected" event

                #if we add the next line, it is slower still workable, and the shadow disseapears...
                #self.ax.figure.canvas.draw_idle()


                self.axbackground = self.ax.figure.canvas.copy_from_bbox(self.ax.bbox)
                self.ax.draw_artist(self.line)
                self.ax.figure.canvas.blit(self.ax.bbox)

                self.ax.figure.canvas.restore_region(self.axbackground)
                #don't forget the flush_events here...
                self.ax.figure.canvas.flush_events()
                self.ax.figure.canvas.draw_idle()


                #the following is a copy of a working version
                #self.axbackground = self.ax.figure.canvas.copy_from_bbox(self.ax.bbox)
                #self.ax.draw_artist(self.line)
                #self.ax.figure.canvas.blit(self.ax.bbox)
                #self.ax.figure.canvas.restore_region(self.axbackground)
                #self.ax.figure.canvas.flush_events()
                #self.ax.figure.canvas.draw_idle()


            else:
                self.ax.figure.canvas.draw_idle()
        #this last line is to make sure that the exponents are on their respetive spines
        self.ax.yaxis.get_offset_text().set_x(self.exponent_position)

    def close(self):
        _logger.debug('Closing `Signal1DLine`.')
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
            self.ax.figure.canvas.draw_idle()
        except BaseException:
            pass
        _logger.debug('`Signal1DLine` closed.')


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

# #snippet for an alternative to twinx that would support picking with multiple axes
# from matplotlib import pyplot as plt
# from mpl_toolkits.axes_grid1 import host_subplot
# from numpy import arange, sin,cos, pi
#
# host_a = host_subplot(111)
# para_a = host_a.twin()
# para2=host_a.twin()
#
#
# t = arange(0.0, 3.0, 0.01)
# s = sin(2*pi*t)
# u=cos(2*pi*t)
# v=s*u
# line, = host_a.plot(t, s, picker=5)
# host_a.yaxis.label.set_color('red')
#
# otherline,=para_a.plot(t, u, picker=5)
# thirdline,=para2.plot(t, v, picker=5)
# host_a.spines['left'].set_picker(5)
#
# def onpick(event):
#     print("picked")
# plt.gcf().canvas.mpl_connect('pick_event', onpick)
#
