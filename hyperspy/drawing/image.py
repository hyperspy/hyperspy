# -*- coding: utf-8 -*-
# Copyright 2007-2020 The HyperSpy developers
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
from distutils.version import LooseVersion

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm, SymLogNorm, PowerNorm
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
from hyperspy.misc.test_utils import ignore_warning
from hyperspy.defaults_parser import preferences


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

    def __init__(self, title=""):
        super(ImagePlot, self).__init__()
        self.data_function = None
        self.data_function_kwargs = {}

        # Attribute matching the arguments of
        # `hyperspy._signal.signal2d.signal2D.plot`
        self.autoscale = "v"
        self.saturated_pixels = None
        self.norm = "auto"
        self.vmin = None
        self.vmax = None
        self.gamma = 1.0
        self.linthresh = 0.01
        self.linscale = 0.1
        self.scalebar = True
        self.scalebar_color = "white"
        self.axes_ticks = None
        self.axes_off = False
        self.axes_manager = None
        self.no_nans = False
        self.colorbar = True
        self.centre_colormap = "auto"
        self.min_aspect = 0.1

        # Other attributes
        self.pixel_units = None
        self._colorbar = None
        self.quantity_label = ''
        self.figure = None
        self.ax = None
        self.title = title

        # user provided numeric values
        self._vmin_numeric = None
        self._vmax_numeric = None
        self._vmax_auto = None
        # user provided percentile values
        self._vmin_percentile = None
        self._vmax_percentile = None
        # default values used when the numeric and percentile are None
        self._vmin_default = f"{preferences.Plot.saturated_pixels / 2}th"
        self._vmax_default = f"{100 - preferences.Plot.saturated_pixels / 2}th"
        # use to store internally the numeric value of contrast
        self._vmin = None
        self._vmax = None

        self._ylabel = ''
        self._xlabel = ''
        self.plot_indices = True
        self._text = None
        self._text_position = (0, 1.05,)
        self._aspect = 1
        self._extent = None
        self.xaxis = None
        self.yaxis = None
        self.ax_markers = list()
        self._user_scalebar = None
        self._auto_scalebar = False
        self._user_axes_ticks = None
        self._auto_axes_ticks = True
        self._is_rgb = False

    @property
    def vmax(self):
        if self._vmax_numeric is not None:
            return self._vmax_numeric
        elif self._vmax_percentile is not None:
            return self._vmax_percentile
        else:
            return self._vmax_default

    @vmax.setter
    def vmax(self, vmax):
        if isinstance(vmax, str):
            self._vmax_percentile = vmax
            self._vmax_numeric = None
        elif isinstance(vmax, (int, float)):
            self._vmax_numeric = vmax
            self._vmax_percentile = None
        elif vmax is None:
            self._vmax_percentile = self._vmax_numeric = None
        else:
            raise TypeError("`vmax` must be a number or a string.")

    @property
    def vmin(self):
        if self._vmin_numeric is not None:
            return self._vmin_numeric
        elif self._vmin_percentile is not None:
            return self._vmin_percentile
        else:
            return self._vmin_default

    @vmin.setter
    def vmin(self, vmin):
        if isinstance(vmin, str):
            self._vmin_percentile = vmin
            self._vmin_numeric = None
        elif isinstance(vmin, (int, float)):
            self._vmin_numeric = vmin
            self._vmin_percentile = None
        elif vmin is None:
            self._vmin_percentile = self._vmin_numeric = None
        else:
            raise TypeError("`vmax` must be a number or a string.")

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
        self._extent = [xaxis.axis[0] - xaxis.scale / 2.,
                        xaxis.axis[-1] + xaxis.scale / 2.,
                        yaxis.axis[-1] + yaxis.scale / 2.,
                        yaxis.axis[0] - yaxis.scale / 2.]
        self._calculate_aspect()
        if self.saturated_pixels is not None:
            from hyperspy.exceptions import VisibleDeprecationWarning
            VisibleDeprecationWarning("`saturated_pixels` is deprecated and will be "
                            "removed in 2.0. Please use `vmin` and `vmax` "
                            "instead.")
            self._vmin_percentile = self.saturated_pixels / 2
            self._vmax_percentile = self.saturated_pixels / 2


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

    def _calculate_vmin_max(self, data, auto_contrast=False,
                            vmin=None, vmax=None):
        # Calculate vmin and vmax using `utils.contrast_stretching` when:
        # - auto_contrast is True
        # - self.vmin or self.vmax is of tpye str
        if (auto_contrast and (isinstance(self.vmin, str) or
            isinstance(self.vmax, str))):
            with ignore_warning(category=RuntimeWarning):
                # In case of "All-NaN slices"
                vmin, vmax = utils.contrast_stretching(
                    data, self.vmin, self.vmax)
        else:
            vmin, vmax = self._vmin_numeric, self._vmax_numeric
        # provided vmin, vmax override the calculated value
        if isinstance(vmin, (int, float)):
            vmin = vmin
        if isinstance(vmax, (int, float)):
            vmax = vmax
        if vmin == np.nan:
            vmin = None
        if vmax == np.nan:
            vmax = None

        return vmin, vmax

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
            All keyword arguments are passed to
            :py:func:`matplotlib.pyplot.figure`.

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

        if (not self.axes_manager or self.axes_manager.navigation_size == 0):
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
        for attribute in ['vmin', 'vmax']:
            if attribute in kwargs.keys():
                setattr(self, attribute, kwargs.pop(attribute))
        self.update(data_changed=True, auto_contrast=True, **kwargs)
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
        # Bug extend='min' or extend='both' and power law norm
        # Use it when it is fixed in matplotlib
        self._colorbar = plt.colorbar(self.ax.images[0], ax=self.ax)
        self.set_quantity_label()
        self._colorbar.set_label(
            self.quantity_label, rotation=-90, va='bottom')
        self._colorbar.ax.yaxis.set_animated(
            self.figure.canvas.supports_blit)

    def _update_data(self):
        # self._current_data caches the displayed data.
        self._current_data =  self.data_function(
                axes_manager=self.axes_manager,
                **self.data_function_kwargs)

    def update(self, data_changed=True, auto_contrast=None, vmin=None,
               vmax=None, **kwargs):
        """
        Parameters
        ----------
        data_changed : bool, optional
            Fetch and update the data to display. It can be used to avoid
            unnecessarily reading of the data from disk with working with lazy
            signal. The default is True.
        auto_contrast : bool or None, optional
            Force automatic resetting of the intensity limits. If None, the
            intensity values will change when 'v' is in autoscale.
            Default is None.
        vmin, vmax : float or str
            `vmin` and `vmax` are used to normalise the displayed data.
        **kwargs : dict
            The kwargs are passed to :py:func:`matplotlib.pyplot.imshow`.

        Raises
        ------
        ValueError
            When the selected ``norm`` is not valid or the data are not
            compatible with the selected ``norm``.
        """
        if auto_contrast is None:
            auto_contrast = 'v' in self.autoscale
        if data_changed:
            # When working with lazy signals the following may reread the data
            # from disk unnecessarily, for example when updating the image just
            # to recompute the histogram to adjust the contrast. In those cases
            # use `data_changed=True`.
            _logger.debug("Updating image slowly because `data_changed=True`")
            self._update_data()
        data = self._current_data
        if rgb_tools.is_rgbx(data):
            self.colorbar = False
            data = rgb_tools.rgbx2regular_array(data, plot_friendly=True)
            data = self._current_data = data
            self._is_rgb = True
        ims = self.ax.images

        # Turn on centre_colormap if a diverging colormap is used.
        if not self._is_rgb and self.centre_colormap == "auto":
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


        for marker in self.ax_markers:
            marker.update()

        if not self._is_rgb:
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
                    if np.isfinite(z):
                        return f'x={x:1.4g}, y={y:1.4g}, intensity={z:1.4g}'
                return f'x={x:1.4g}, y={y:1.4g}'
            self.ax.format_coord = format_coord

            old_vmin, old_vmax = self._vmin, self._vmax

            if auto_contrast:
                vmin, vmax = self._calculate_vmin_max(data, auto_contrast,
                                                      vmin, vmax)
            else:
                # use the value store internally when not explicitely defined
                if vmin is None:
                    vmin = old_vmin
                if vmax is None:
                    vmax = old_vmax

            # If there is an image, any of the contrast bounds have changed and
            # the new contrast bounds are not the same redraw the colorbar.
            if (ims and (old_vmin != vmin or old_vmax != vmax) and
                    vmin != vmax):
                redraw_colorbar = True
                ims[0].autoscale()
            if self.centre_colormap:
                vmin, vmax = utils.centre_colormap_values(vmin, vmax)

            if self.norm == 'auto' and self.gamma != 1.0:
                self.norm = 'power'
            norm = copy.copy(self.norm)
            if norm == 'power':
                # with auto norm, we use the power norm when gamma differs from its
                # default value.
                norm = PowerNorm(self.gamma, vmin=vmin, vmax=vmax)
            elif norm == 'log':
                if np.nanmax(data) <= 0:
                    raise ValueError('All displayed data are <= 0 and can not '
                                    'be plotted using `norm="log"`. '
                                    'Use `norm="symlog"` to plot on a log scale.')
                if np.nanmin(data) <= 0:
                    vmin = np.nanmin(np.where(data > 0, data, np.inf))

                norm = LogNorm(vmin=vmin, vmax=vmax)
            elif norm == 'symlog':
                sym_log_kwargs = {'linthresh':self.linthresh,
                                  'linscale':self.linscale,
                                  'vmin':vmin, 'vmax':vmax}
                if LooseVersion(matplotlib.__version__) >= LooseVersion("3.2"):
                    sym_log_kwargs['base'] = 10
                norm = SymLogNorm(**sym_log_kwargs)
            elif inspect.isclass(norm) and issubclass(norm, Normalize):
                norm = norm(vmin=vmin, vmax=vmax)
            elif norm not in ['auto', 'linear']:
                raise ValueError("`norm` paramater should be 'auto', 'linear', "
                                "'log', 'symlog' or a matplotlib Normalize  "
                                "instance or subclass.")
            else:
                # set back to matplotlib default
                norm = None

            self._vmin, self._vmax = vmin, vmax

        redraw_colorbar = redraw_colorbar and self.colorbar

        if self.plot_indices is True:
            self._text.set_text(self.axes_manager.indices)
        if self.no_nans:
            data = np.nan_to_num(data)

        if ims:  # the images has already been drawn previously
            ims[0].set_data(data)
            # update extent:
            if 'x' in self.autoscale:
                self._extent[0] = self.xaxis.axis[0] - self.xaxis.scale / 2
                self._extent[1] = self.xaxis.axis[-1] + self.xaxis.scale / 2
                self.ax.set_xlim(self._extent[:2])
            if 'y' in self.autoscale:
                self._extent[2] = self.yaxis.axis[-1] + self.yaxis.scale / 2
                self._extent[3] = self.yaxis.axis[0] - self.yaxis.scale / 2
                self.ax.set_ylim(self._extent[2:])
            if 'x' in self.autoscale or 'y' in self.autoscale:
                ims[0].set_extent(self._extent)
            self._calculate_aspect()
            self.ax.set_aspect(self._aspect)
            if not self._is_rgb:
                ims[0].set_norm(norm)
                ims[0].norm.vmax, ims[0].norm.vmin = vmax, vmin
            if redraw_colorbar:
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
                        'extent': self._extent,
                        'aspect': self._aspect,
                        'animated': self.figure.canvas.supports_blit,
                        }
            if not self._is_rgb:
                if norm is None:
                    new_args.update({'vmin': vmin, 'vmax':vmax})
                else:
                    new_args['norm'] = norm
            new_args.update(kwargs)
            self.ax.imshow(data, **new_args)
            self.figure.canvas.draw_idle()

        if self.axes_ticks == 'off':
            self.ax.set_axis_off()

    def _update(self):
        # This "wrapper" because on_trait_change fiddles with the
        # method arguments and auto contrast does not work then
        self.update(data_changed=False)

    def gui_adjust_contrast(self, display=True, toolkit=None):
        if self._is_rgb:
            raise NotImplementedError(
                "Constrast adjustment of RGB images is not implemented")
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
            if self.update not in self.axes_manager.events.indices_changed.connected:
                self.axes_manager.events.indices_changed.connect(self.update, [])
            if self.disconnect not in self.events.closed.connected:
                self.events.closed.connect(self.disconnect, [])

    def disconnect(self):
        if self.axes_manager:
            if self.update in self.axes_manager.events.indices_changed.connected:
                self.axes_manager.events.indices_changed.disconnect(self.update)

    def on_key_press(self, event):
        if event.key == 'h':
            self.gui_adjust_contrast()
        if event.key == 'l':
            self.toggle_norm()

    def toggle_norm(self):
        self.norm = 'linear' if self.norm == 'log' else 'log'
        self.update(data_changed=False)
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
        self.update(data_changed=False, auto_contrast=True)

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
