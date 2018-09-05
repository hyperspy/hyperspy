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
# along with HyperSpy. If not, see <http://www.gnu.org/licenses/>.

import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from traits.api import Undefined
import logging
import inspect
import copy

from hyperspy.drawing import widgets
from hyperspy.drawing import utils
from hyperspy.signal_tools import ImageContrastEditor
from hyperspy.misc import math_tools
from hyperspy.misc import rgb_tools
from hyperspy.drawing.figure import BlittedFigure
from hyperspy.ui_registry import DISPLAY_DT, TOOLKIT_DT
from hyperspy.docstrings.plot import PLOT2D_DOCSTRING


_logger = logging.getLogger(__name__)


class ImagePlot(BlittedFigure):

    """Class to plot an image with the necessary machinery to update
    the image when the coordinates of an AxesManager change.

    Attributes
    ----------
    data_fuction : function or method
        A function that returns a 2D array when called without any
        arguments.
    %s
    pixel_units : {None, string}
        The pixel units for the scale bar.
    plot_indices : bool
    title : str
        The title is printed at the top of the image.

    """ % PLOT2D_DOCSTRING

    def __init__(self):
        super(ImagePlot, self).__init__()
        self.data_function = None
        # Args to pass to `__call__`
        self.data_function_kwargs = {}
        self.pixel_units = None
        self.colorbar = True
        self._colorbar = None
        self.quantity_label = ''
        self.figure = None
        self.ax = None
        self.title = ''
        self._vmin_user = None
        self._vmax_user = None
        self._vmin_auto = None
        self._vmax_auto = None
        self._ylabel = ''
        self._xlabel = ''
        self.plot_indices = True
        self._text = None
        self._text_position = (0, 1.05,)
        self.axes_manager = None
        self.axes_off = False
        self._aspect = 1
        self._extent = None
        self.xaxis = None
        self.yaxis = None
        self.min_aspect = 0.1
        self.saturated_pixels = 0.2
        self.ax_markers = list()
        self.scalebar_color = "white"
        self._user_scalebar = None
        self._auto_scalebar = False
        self._user_axes_ticks = None
        self._auto_axes_ticks = True
        self.no_nans = False
        self.centre_colormap = "auto"
        self.norm = "auto"

    @property
    def vmax(self):
        if self._vmax_user is not None:
            return self._vmax_user
        else:
            return self._vmax_auto

    @vmax.setter
    def vmax(self, vmax):
        self._vmax_user = vmax

    @property
    def vmin(self):
        if self._vmin_user is not None:
            return self._vmin_user
        else:
            return self._vmin_auto

    @vmin.setter
    def vmin(self, vmin):
        self._vmin_user = vmin

    @property
    def axes_ticks(self):
        if self._user_axes_ticks is None:
            if self.scalebar is False:
                return True
            else:
                return self._auto_axes_ticks
        else:
            return self._user_axes_ticks

    @axes_ticks.setter
    def axes_ticks(self, value):
        self._user_axes_ticks = value

    @property
    def scalebar(self):
        if self._user_scalebar is None:
            return self._auto_scalebar
        else:
            return self._user_scalebar

    @scalebar.setter
    def scalebar(self, value):
        if value is False:
            self._user_scalebar = value
        else:
            self._user_scalebar = None

    def configure(self):
        xaxis = self.xaxis
        yaxis = self.yaxis

        if (xaxis.units == yaxis.units) and (xaxis.scale == yaxis.scale):
            self._auto_scalebar = True
            self._auto_axes_ticks = False
            self.pixel_units = xaxis.units
        else:
            self._auto_scalebar = False
            self._auto_axes_ticks = True

        # Signal2D labels
        self._xlabel = '{}'.format(xaxis)
        if xaxis.units is not Undefined:
            self._xlabel += ' ({})'.format(xaxis.units)

        self._ylabel = '{}'.format(yaxis)
        if yaxis.units is not Undefined:
            self._ylabel += ' ({})'.format(yaxis.units)

        # Calibrate the axes of the navigator image
        self._extent = (xaxis.axis[0] - xaxis.scale / 2.,
                        xaxis.axis[-1] + xaxis.scale / 2.,
                        yaxis.axis[-1] + yaxis.scale / 2.,
                        yaxis.axis[0] - yaxis.scale / 2.)
        self._calculate_aspect()

    def _calculate_aspect(self):
        xaxis = self.xaxis
        yaxis = self.yaxis
        factor = 1
        # Apply aspect ratio constraint
        if self.min_aspect:
            min_asp = self.min_aspect
            if yaxis.size / xaxis.size < min_asp:
                factor = min_asp * xaxis.size / yaxis.size
                self._auto_scalebar = False
                self._auto_axes_ticks = True
            elif yaxis.size / xaxis.size > min_asp ** -1:
                factor = min_asp ** -1 * xaxis.size / yaxis.size
                self._auto_scalebar = False
                self._auto_axes_ticks = True
        self._aspect = np.abs(factor * xaxis.scale / yaxis.scale)
        # print(self._aspect)

    def optimize_contrast(self, data):
        if (self._vmin_user is not None and self._vmax_user is not None):
            return
        self._vmin_auto, self._vmax_auto = utils.contrast_stretching(
            data, self.saturated_pixels)

    def create_figure(self, max_size=None, min_size=2, **kwargs):
        """Create matplotlib figure

        The figure size is automatically computed by default, taking into
        account the x and y dimensions of the image. Alternatively the figure
        size can be defined by passing the ``figsize`` keyword argument.

        Parameters
        ----------
        max_size, min_size: number
            The maximum and minimum size of the axes in inches. These have
            no effect when passing the ``figsize`` keyword to manually set
            the figure size.
        **kwargs
            All keyword arguments are passed to ``plt.figure``.

        """
        if "figsize" not in kwargs:
            if self.scalebar is True:
                wfactor = 1.0 + plt.rcParams['font.size'] / 100
            else:
                wfactor = 1

            height = abs(self._extent[3] - self._extent[2]) * self._aspect
            width = abs(self._extent[1] - self._extent[0])
            figsize = np.array((width * wfactor, height)) * \
                max(plt.rcParams['figure.figsize']) / \
                max(width * wfactor, height)
            kwargs["figsize"] = figsize.clip(min_size, max_size)
        if "disable_xyscale_keys" not in kwargs:
            kwargs["disable_xyscale_keys"] = True
        super().create_figure(**kwargs)

    def create_axis(self):
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title(self.title)
        self.ax.set_xlabel(self._xlabel)
        self.ax.set_ylabel(self._ylabel)
        if self.axes_ticks is False:
            self.ax.set_xticks([])
            self.ax.set_yticks([])
        self.ax.hspy_fig = self
        if self.axes_off:
            self.ax.axis('off')

    def plot(self, data_function_kwargs={}, **kwargs):
        self.data_function_kwargs = data_function_kwargs
        self.configure()
        if self.figure is None:
            self.create_figure()
            self.create_axis()
        data = self.data_function(axes_manager=self.axes_manager,
                                  **self.data_function_kwargs)
        if rgb_tools.is_rgbx(data):
            self.colorbar = False
            data = rgb_tools.rgbx2regular_array(data, plot_friendly=True)
        self.optimize_contrast(data)
        if (not self.axes_manager or
                self.axes_manager.navigation_size == 0):
            self.plot_indices = False
        if self.plot_indices is True:
            if self._text is not None:
                self._text.remove()
            self._text = self.ax.text(
                *self._text_position,
                s=str(self.axes_manager.indices),
                transform=self.ax.transAxes,
                fontsize=12,
                color='red',
                animated=self.figure.canvas.supports_blit)
        for marker in self.ax_markers:
            marker.plot()
        self.update(**kwargs)
        if self.scalebar is True:
            if self.pixel_units is not None:
                self.ax.scalebar = widgets.ScaleBar(
                    ax=self.ax,
                    units=self.pixel_units,
                    animated=self.figure.canvas.supports_blit,
                    color=self.scalebar_color,
                )

        if self.colorbar:
            self._add_colorbar()

        if hasattr(self.figure, 'tight_layout'):
            try:
                if self.axes_ticks == 'off' and not self.colorbar:
                    plt.subplots_adjust(0, 0, 1, 1)
                else:
                    self.figure.tight_layout()
            except BaseException:
                # tight_layout is a bit brittle, we do this just in case it
                # complains
                pass

        self.connect()
        self.figure.canvas.draw()

    def _add_colorbar(self):
        self._colorbar = plt.colorbar(self.ax.images[0], ax=self.ax)
        self.set_quantity_label()
        self._colorbar.set_label(
            self.quantity_label, rotation=-90, va='bottom')
        self._colorbar.ax.yaxis.set_animated(
            self.figure.canvas.supports_blit)

    def update(self, **kwargs):
        ims = self.ax.images
        # update extent:
        self._extent = (self.xaxis.axis[0] - self.xaxis.scale / 2.,
                        self.xaxis.axis[-1] + self.xaxis.scale / 2.,
                        self.yaxis.axis[-1] + self.yaxis.scale / 2.,
                        self.yaxis.axis[0] - self.yaxis.scale / 2.)

        # Turn on centre_colormap if a diverging colormap is used.
        if self.centre_colormap == "auto":
            if "cmap" in kwargs:
                cmap = kwargs["cmap"]
            elif ims:
                cmap = ims[0].get_cmap().name
            else:
                cmap = plt.cm.get_cmap().name
            if cmap in utils.MPL_DIVERGING_COLORMAPS:
                self.centre_colormap = True
            else:
                self.centre_colormap = False
        redraw_colorbar = False
        data = rgb_tools.rgbx2regular_array(
            self.data_function(axes_manager=self.axes_manager,
                               **self.data_function_kwargs),
            plot_friendly=True)
        numrows, numcols = data.shape[:2]
        for marker in self.ax_markers:
            marker.update()
        if len(data.shape) == 2:
            def format_coord(x, y):
                try:
                    col = self.xaxis.value2index(x)
                except ValueError:  # out of axes limits
                    col = -1
                try:
                    row = self.yaxis.value2index(y)
                except ValueError:
                    row = -1
                if col >= 0 and row >= 0:
                    z = data[row, col]
                    return 'x=%1.4g, y=%1.4g, intensity=%1.4g' % (x, y, z)
                else:
                    return 'x=%1.4g, y=%1.4g' % (x, y)
            self.ax.format_coord = format_coord
            old_vmax, old_vmin = self.vmax, self.vmin
            self.optimize_contrast(data)
            # If there is an image, any of the contrast bounds have changed and
            # the new contrast bounds are not the same redraw the colorbar.
            if (ims and (old_vmax != self.vmax or old_vmin != self.vmin) and
                    self.vmax != self.vmin):
                redraw_colorbar = True
                ims[0].autoscale()
        redraw_colorbar = redraw_colorbar and self.colorbar
        if self.plot_indices is True:
            self._text.set_text(self.axes_manager.indices)
        if self.no_nans:
            data = np.nan_to_num(data)
        if self.centre_colormap:
            vmin, vmax = utils.centre_colormap_values(self.vmin, self.vmax)
        else:
            vmin, vmax = self.vmin, self.vmax

        norm = copy.copy(self.norm)
        if norm == 'log':
            norm = LogNorm(vmin=self.vmin, vmax=self.vmax)
        elif inspect.isclass(norm) and issubclass(norm, Normalize):
            norm = norm(vmin=self.vmin, vmax=self.vmax)
        elif norm not in ['auto', 'linear']:
            raise ValueError("`norm` paramater should be 'auto', 'linear', "
                             "'log' or a matplotlib Normalize instance or "
                             "subclass.")
        else:
            # set back to matplotlib default
            norm = None

        if ims:  # the images has already been drawn previously
            ims[0].set_data(data)
            self.ax.set_xlim(self._extent[:2])
            self.ax.set_ylim(self._extent[2:])
            ims[0].set_extent(self._extent)
            self._calculate_aspect()
            self.ax.set_aspect(self._aspect)
            ims[0].set_norm(norm)
            ims[0].norm.vmax, ims[0].norm.vmin = vmax, vmin
            if redraw_colorbar:
                # ims[0].autoscale()
                self._colorbar.draw_all()
                self._colorbar.solids.set_animated(
                    self.figure.canvas.supports_blit
                )
            else:
                ims[0].changed()
            if self.figure.canvas.supports_blit:
                self._update_animated()
            else:
                self.figure.canvas.draw_idle()
        else:  # no signal have been drawn yet
            new_args = {'interpolation': 'nearest',
                        'vmin': vmin,
                        'vmax': vmax,
                        'extent': self._extent,
                        'aspect': self._aspect,
                        'animated': self.figure.canvas.supports_blit,
                        'norm': norm}
            new_args.update(kwargs)
            self.ax.imshow(data,
                           **new_args)
            self.figure.canvas.draw_idle()

        if self.axes_ticks == 'off':
            self.ax.set_axis_off()

    def _update(self):
        # This "wrapper" because on_trait_change fiddles with the
        # method arguments and auto contrast does not work then
        self.update()

    def gui_adjust_contrast(self, display=True, toolkit=None):
        ceditor = ImageContrastEditor(self)
        return ceditor.gui(display=display, toolkit=toolkit)
    gui_adjust_contrast.__doc__ = \
        """
        Display widgets to adjust image contrast if available.
        Parameters
        ----------
        %s
        %s

        """ % (DISPLAY_DT, TOOLKIT_DT)

    def connect(self):
        self.figure.canvas.mpl_connect('key_press_event',
                                       self.on_key_press)
        if self.axes_manager:
            self.axes_manager.events.indices_changed.connect(self.update, [])
            self.events.closed.connect(
                lambda: self.axes_manager.events.indices_changed.disconnect(
                    self.update), [])

    def on_key_press(self, event):
        if event.key == 'h':
            self.gui_adjust_contrast()
        if event.key == 'l':
            self.toggle_norm()

    def toggle_norm(self):
        self.norm = 'linear' if self.norm == 'log' else 'log'
        self.update()
        if self.colorbar:
            self._colorbar.remove()
            self._add_colorbar()
            self.figure.canvas.draw_idle()

    def set_quantity_label(self):
        if 'power_spectrum' in self.data_function_kwargs.keys():
            if self.norm == 'log':
                if 'FFT' in self.quantity_label:
                    self.quantity_label = self.quantity_label.replace(
                        'Power spectral density', 'FFT')
                else:
                    of = ' of ' if self.quantity_label else ''
                    self.quantity_label = 'Power spectral density' + of + \
                        self.quantity_label
            else:
                self.quantity_label = self.quantity_label.replace(
                    'Power spectral density of ', '')
                self.quantity_label = self.quantity_label.replace(
                    'Power spectral density', '')

    def set_contrast(self, vmin, vmax):
        self.vmin, self.vmax = vmin, vmax
        self.update()

    def optimize_colorbar(self,
                          number_of_ticks=5,
                          tolerance=5,
                          step_prec_max=1):
        vmin, vmax = self.vmin, self.vmax
        _range = vmax - vmin
        step = _range / (number_of_ticks - 1)
        step_oom = math_tools.order_of_magnitude(step)

        def optimize_for_oom(oom):
            self.colorbar_step = math.floor(step / 10 ** oom) * 10 ** oom
            self.colorbar_vmin = math.floor(vmin / 10 ** oom) * 10 ** oom
            self.colorbar_vmax = self.colorbar_vmin + \
                self.colorbar_step * (number_of_ticks - 1)
            self.colorbar_locs = (
                np.arange(0, number_of_ticks) *
                self.colorbar_step +
                self.colorbar_vmin)

        def check_tolerance():
            if abs(self.colorbar_vmax - vmax) / vmax > (
                tolerance / 100.) or abs(self.colorbar_vmin - vmin
                                         ) > (tolerance / 100.):
                return True
            else:
                return False

        optimize_for_oom(step_oom)
        i = 1
        while check_tolerance() and i <= step_prec_max:
            optimize_for_oom(step_oom - i)
            i += 1
