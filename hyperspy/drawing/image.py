# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

import copy
import inspect
import logging
import math

import numpy as np
from rsciio.utils import rgb
from traits.api import Undefined

from hyperspy.docstrings.plot import PLOT2D_DOCSTRING
from hyperspy.drawing import utils
from hyperspy.drawing.backends import get_backend
from hyperspy.drawing.backends._protocol import BackendCapabilityError
from hyperspy.drawing.figure import AbstractImageFigure
from hyperspy.drawing.norm import HyperNorm, LogNorm, PowerNorm, SymLogNorm
from hyperspy.misc import math_tools
from hyperspy.misc.test_utils import ignore_warning
from hyperspy.signal_tools import ImageContrastEditor
from hyperspy.ui_registry import DISPLAY_DT, TOOLKIT_DT

_logger = logging.getLogger(__name__)


class ImagePlot(AbstractImageFigure):
    """Class to plot an image with the necessary machinery to update
    the image when the coordinates of an AxesManager change.

    Attributes
    ----------
    data_function : function or method
        A function that returns a 2D array when called without any
        arguments.
    %s
    pixel_units : {None, string}
        The pixel units for the scale bar.
    plot_indices : bool
    title : str
        The title is printed at the top of the image.

    """ % PLOT2D_DOCSTRING

    def __init__(self, title="", **kwargs):
        super().__init__()
        self.data_function = None
        self.data_function_kwargs = {}

        # Attribute matching the arguments of
        # `hyperspy._signal.signal2d.signal2D.plot`
        self.autoscale = "v"
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
        self.quantity_label = ""
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
        # use to store internally the numeric value of contrast
        self._vmin = None
        self._vmax = None

        self._ylabel = ""
        self._xlabel = ""
        self.plot_indices = True
        self._text = None
        self._text_position = (
            0,
            1.05,
        )
        self._aspect = 1
        self._extent = None
        self.xaxis = None
        self.yaxis = None
        self.ax_markers = list()
        self._user_scalebar = None
        self._auto_scalebar = False
        self._user_axes_ticks = None
        self._auto_axes_ticks = True
        # Store mpl_connect cid so we can disconnect before reconnecting;
        # unlike Event.connect which deduplicates via connected list,
        # canvas.mpl_connect always creates a new handler each call.
        self._key_press_cid = None
        self._is_rgb = False

    @property
    def vmax(self):
        if self._vmax_numeric is not None:
            return self._vmax_numeric
        elif self._vmax_percentile is not None:
            return self._vmax_percentile
        else:
            return "100th"

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
            raise TypeError("`vmax` must be a number, a string or `None`.")

    @property
    def vmin(self):
        if self._vmin_numeric is not None:
            return self._vmin_numeric
        elif self._vmin_percentile is not None:
            return self._vmin_percentile
        else:
            return "0th"

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
            raise TypeError("`vmin` must be a number, a string or `None`.")

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

        if (
            xaxis.is_uniform
            and yaxis.is_uniform
            and (xaxis.units == yaxis.units)
            and (abs(xaxis.scale) == abs(yaxis.scale))
        ):
            self._auto_scalebar = True
            self._auto_axes_ticks = False
            self.pixel_units = xaxis.units
        else:
            self._auto_scalebar = False
            self._auto_axes_ticks = True

        # Signal2D labels
        self._xlabel = "{}".format(xaxis)
        if xaxis.units is not Undefined:
            self._xlabel += " ({})".format(xaxis.units)

        self._ylabel = "{}".format(yaxis)
        if yaxis.units is not Undefined:
            self._ylabel += " ({})".format(yaxis.units)

        # Calibrate the axes of the navigator image
        if xaxis.is_uniform and yaxis.is_uniform:
            xaxis_half_px = xaxis.scale / 2.0
            yaxis_half_px = yaxis.scale / 2.0
        else:
            xaxis_half_px = 0
            yaxis_half_px = 0
        # Calibrate the axes of the navigator image
        self._extent = [
            xaxis.axis[0] - xaxis_half_px,
            xaxis.axis[-1] + xaxis_half_px,
            yaxis.axis[-1] + yaxis_half_px,
            yaxis.axis[0] - yaxis_half_px,
        ]
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
            elif yaxis.size / xaxis.size > min_asp**-1:
                factor = min_asp**-1 * xaxis.size / yaxis.size
                self._auto_scalebar = False
                self._auto_axes_ticks = True
        if xaxis.is_uniform and yaxis.is_uniform:
            self._aspect = abs(factor * xaxis.scale / yaxis.scale)
        else:
            self._aspect = 1.0

    def _calculate_vmin_max(self, data, auto_contrast=False, vmin=None, vmax=None):
        vminprovided = vmin
        vmaxprovided = vmax
        # Calculate vmin and vmax using `utils.contrast_stretching` when:
        # - auto_contrast is True
        # - self.vmin or self.vmax is of tpye str
        if auto_contrast and (isinstance(self.vmin, str) or isinstance(self.vmax, str)):
            with ignore_warning(category=RuntimeWarning):
                # In case of "All-NaN slices"
                vmin, vmax = utils.contrast_stretching(data, self.vmin, self.vmax)
        else:
            vmin, vmax = self._vmin_numeric, self._vmax_numeric
        # provided vmin, vmax override the calculated value
        if isinstance(vminprovided, (int, float)):
            vmin = vminprovided
        if isinstance(vmaxprovided, (int, float)):
            vmax = vmaxprovided
        if vmin == np.nan:
            vmin = None
        if vmax == np.nan:
            vmax = None

        return vmin, vmax

    def create_figure(self, max_size=None, min_size=2, **kwargs):
        """Create figure

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
            All keyword arguments are passed to the backend's figure creation.

        """
        if "figsize" not in kwargs:
            if self.scalebar is True:
                wfactor = 1.1
            else:
                wfactor = 1

            height = abs(self._extent[3] - self._extent[2]) * self._aspect
            width = abs(self._extent[1] - self._extent[0])
            default_size = 6.0
            figsize = (
                np.array((width * wfactor, height))
                * default_size
                / max(width * wfactor, height)
            )
            kwargs["figsize"] = figsize.clip(min_size, max_size)
        kwargs.pop("disable_xyscale_keys", None)
        super().create_figure(**kwargs)

    def create_axis(self):
        backend = get_backend()
        self.ax = backend.create_axes(self.figure)
        backend.set_title(self.ax, self.title)
        backend.set_xlabel(self.ax, self._xlabel)
        backend.set_ylabel(self.ax, self._ylabel)
        if self.axes_ticks is False:
            backend.set_xticklabels(self.ax, [])
            backend.set_yticklabels(self.ax, [])
        self.ax.hspy_fig = self
        if self.axes_off:
            backend.set_axis_off(self.ax)

    def plot(self, data_function_kwargs={}, **kwargs):
        self.data_function_kwargs = data_function_kwargs
        self.configure()
        if self.figure is None:
            fig = kwargs.pop("fig", None)
            _on_figure_window_close = kwargs.pop("_on_figure_window_close", None)
            self.create_figure(fig=fig, _on_figure_window_close=_on_figure_window_close)
            self.create_axis()

        if not self.axes_manager or self.axes_manager.navigation_size == 0:
            self.plot_indices = False
        if self.plot_indices is True:
            backend = get_backend()
            if self._text is not None:
                backend.remove_text(self.ax, self._text)
            self._text = backend.add_text(
                self.ax,
                *self._text_position,
                s=str(self.axes_manager.indices),
                transform="axes",
                fontsize=12,
                color="red",
            )
        for marker in self.ax_markers:
            marker.plot()
        for attribute in ["vmin", "vmax"]:
            if attribute in kwargs.keys():
                setattr(self, attribute, kwargs.pop(attribute))
        self.update(data_changed=True, auto_contrast=True, **kwargs)
        if self.scalebar is True:
            if self.pixel_units is not None:
                try:
                    self._scalebar_handle = get_backend().create_scalebar(
                        ax=self.ax,
                        units=self.pixel_units,
                        animated=get_backend().supports_blit(self.figure),
                        color=self.scalebar_color,
                    )
                    self.ax.scalebar = self._scalebar_handle
                except BackendCapabilityError:
                    self._scalebar_handle = None

        if self.colorbar:
            self._add_colorbar()

        if not (self.axes_ticks == "off" and not self.colorbar):
            get_backend().tight_layout(self.figure)

        self.connect()
        self.render_figure()

    def _add_colorbar(self):
        # Bug extend='min' or extend='both' and power law norm
        # Use it when it is fixed in matplotlib
        backend = get_backend()
        im_handle = backend.get_image_handle(self.ax)
        self._colorbar = backend.add_colorbar(self.figure, im_handle, self.ax)
        self.set_quantity_label()
        backend.colorbar_set_label(self._colorbar, self.quantity_label)

    def _update_data(self):
        # self._current_data caches the displayed data.
        data = self.data_function(
            axes_manager=self.axes_manager, **self.data_function_kwargs
        )
        # the colorbar of matplotlib ~< 3.2 doesn't support bool array
        if data.dtype == bool:
            data = data.astype(int)
        self._current_data = data

    def update(
        self, data_changed=True, auto_contrast=None, vmin=None, vmax=None, **kwargs
    ):
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
            The kwargs are passed to :func:`matplotlib.pyplot.imshow`.

        Raises
        ------
        ValueError
            When the selected ``norm`` is not valid or the data are not
            compatible with the selected ``norm``.
        """
        if auto_contrast is None:
            auto_contrast = "v" in self.autoscale
        if data_changed:
            # When working with lazy signals the following may reread the data
            # from disk unnecessarily, for example when updating the image just
            # to recompute the histogram to adjust the contrast. In those cases
            # use `data_changed=True`.
            _logger.debug("Updating image slowly because `data_changed=True`")
            self._update_data()
        data = self._current_data
        if rgb.is_rgbx(data):
            self.colorbar = False
            data = rgb.rgbx2regular_array(data, plot_friendly=True)
            data = self._current_data = data
            self._is_rgb = True

        backend = get_backend()
        handle = backend.get_image_handle(self.ax)

        # Turn on centre_colormap if a diverging colormap is used.
        if not self._is_rgb and self.centre_colormap == "auto":
            if "cmap" in kwargs:
                cmap = kwargs["cmap"]
            elif handle is not None:
                try:
                    cmap = backend.get_image_cmap_name(handle)
                except BackendCapabilityError:
                    from hyperspy.defaults_parser import preferences

                    cmap = preferences.Plot.cmap_signal
            else:
                from hyperspy.defaults_parser import preferences

                cmap = preferences.Plot.cmap_signal
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
                        return f"x={x:1.4g}, y={y:1.4g}, intensity={z:1.4g}"
                return f"x={x:1.4g}, y={y:1.4g}"

            if hasattr(self.ax, "format_coord"):
                self.ax.format_coord = format_coord

            old_vmin, old_vmax = self._vmin, self._vmax

            if auto_contrast:
                vmin, vmax = self._calculate_vmin_max(data, auto_contrast, vmin, vmax)
            else:
                # use the value store internally when not explicitly defined
                if vmin is None:
                    vmin = old_vmin
                if vmax is None:
                    vmax = old_vmax

            # If there is an image, any of the contrast bounds have changed and
            # the new contrast bounds are not the same redraw the colorbar.
            if (
                handle is not None
                and (old_vmin != vmin or old_vmax != vmax)
                and vmin != vmax
            ):
                redraw_colorbar = True
            if self.centre_colormap:
                vmin, vmax = utils.centre_colormap_values(vmin, vmax)

            if self.norm == "auto" and self.gamma != 1.0:
                self.norm = "power"
            norm = copy.copy(self.norm)
            if norm == "power":
                # with auto norm, we use the power norm when gamma differs from its
                # default value.
                norm = PowerNorm(self.gamma, vmin=vmin, vmax=vmax)
            elif norm == "log":
                if np.nanmax(data) <= 0:
                    raise ValueError(
                        "All displayed data are <= 0 and can not "
                        'be plotted using `norm="log"`. '
                        'Use `norm="symlog"` to plot on a log scale.'
                    )
                if np.nanmin(data) <= 0:
                    vmin = np.nanmin(np.where(data > 0, data, np.inf))

                norm = LogNorm(vmin=vmin, vmax=vmax)
            elif norm == "symlog":
                sym_log_kwargs = {
                    "linthresh": self.linthresh,
                    "linscale": self.linscale,
                    "vmin": vmin,
                    "vmax": vmax,
                    "base": 10,
                }
                norm = SymLogNorm(**sym_log_kwargs)
            elif inspect.isclass(norm) and issubclass(norm, HyperNorm):
                norm = norm(vmin=vmin, vmax=vmax)
            elif isinstance(norm, HyperNorm):
                pass  # already a concrete HyperNorm instance; pass through
            elif norm not in ["auto", "linear"]:
                raise ValueError(
                    "`norm` parameter should be 'auto', 'linear', "
                    "'log', 'symlog', a HyperNorm subclass, or a HyperNorm instance."
                )
            else:
                norm = None

            self._vmin, self._vmax = vmin, vmax

        redraw_colorbar = redraw_colorbar and self.colorbar

        if self.plot_indices is True:
            backend.update_text(self._text, str(self.axes_manager.indices))
        if self.no_nans:
            data = np.nan_to_num(data)

        if handle is not None:  # the images have already been drawn previously
            backend.image_set_data(handle, data)
            # update extent:
            if "x" in self.autoscale:
                self._extent[0] = self.xaxis.axis[0] - self.xaxis.scale / 2
                self._extent[1] = self.xaxis.axis[-1] + self.xaxis.scale / 2
                backend.set_xlim(self.ax, self._extent[0], self._extent[1])
            if "y" in self.autoscale:
                self._extent[2] = self.yaxis.axis[-1] + self.yaxis.scale / 2
                self._extent[3] = self.yaxis.axis[0] - self.yaxis.scale / 2
                backend.set_ylim(self.ax, self._extent[2], self._extent[3])
            if "x" in self.autoscale or "y" in self.autoscale:
                backend.image_set_extent(handle, self._extent)
            self._calculate_aspect()
            backend.set_aspect(self.ax, self._aspect)
            if not self._is_rgb:
                backend.image_set_norm(handle, norm)
                backend.image_set_clim(handle, vmin, vmax)
            if redraw_colorbar:
                backend.colorbar_redraw(self._colorbar, self.figure)
            self.render_figure()
        else:  # no signal have been drawn yet
            new_args = {}
            if not self._is_rgb:
                if norm is None:
                    new_args.update({"vmin": vmin, "vmax": vmax})
                else:
                    new_args["norm"] = norm
            new_args.update(kwargs)
            if self.xaxis.is_uniform and self.yaxis.is_uniform:
                # pcolormesh doesn't have extent and aspect as arguments
                # aspect is set earlier via self.ax.set_aspect() anyways
                new_args.update({"extent": self._extent, "aspect": self._aspect})
                backend.plot_image(self.ax, data, **new_args)
            else:
                backend.plot_mesh(
                    self.ax, self.xaxis.axis, self.yaxis.axis, data, **new_args
                )
        if self.axes_ticks == "off":
            backend.set_axis_off(self.ax)

    def _update(self):
        # This "wrapper" because on_trait_change fiddles with the
        # method arguments and auto contrast does not work then
        self.update(data_changed=False)

    def gui_adjust_contrast(self, display=True, toolkit=None):
        if self._is_rgb:
            raise NotImplementedError(
                "Contrast adjustment of RGB images is not implemented"
            )
        ceditor = ImageContrastEditor(self)
        return ceditor.gui(display=display, toolkit=toolkit)

    gui_adjust_contrast.__doc__ = """
        Display widgets to adjust image contrast if available.

        Parameters
        ----------
        %s
        %s

        """ % (DISPLAY_DT, TOOLKIT_DT)

    def connect(self):
        # in case the figure is not displayed
        if self.figure is not None:
            self._key_cid = get_backend().connect_key_press(
                self.figure, self.on_key_press
            )
        if self.axes_manager:
            if self.update not in self.axes_manager.events.indices_changed.connected:
                self.axes_manager.events.indices_changed.connect(self.update, [])
            if self.disconnect not in self.events.closed.connected:
                self.events.closed.connect(self.disconnect, [])

    def disconnect(self):
        if hasattr(self, "_key_cid"):
            get_backend().disconnect_event(self.figure, self._key_cid)
        if self.axes_manager:
            if self.update in self.axes_manager.events.indices_changed.connected:
                self.axes_manager.events.indices_changed.disconnect(self.update)

    def on_key_press(self, event):
        if event.key == "h":
            self.gui_adjust_contrast()
        if event.key == "l":
            self.toggle_norm()

    def toggle_norm(self):
        self.norm = "linear" if self.norm == "log" else "log"
        self.update(data_changed=False)
        if self.colorbar:
            backend = get_backend()
            backend.colorbar_remove(self._colorbar)
            self._add_colorbar()
            # Use render_figure instead of canvas.draw_idle() to go through the
            # blit render pipeline when supported (see BlittedFigure.render_figure).
            self.render_figure()

    def set_quantity_label(self):
        if "power_spectrum" in self.data_function_kwargs.keys():
            if self.norm == "log":
                if "FFT" in self.quantity_label:
                    self.quantity_label = self.quantity_label.replace(
                        "Power spectral density", "FFT"
                    )
                else:
                    of = " of " if self.quantity_label else ""
                    self.quantity_label = (
                        "Power spectral density" + of + self.quantity_label
                    )
            else:
                self.quantity_label = self.quantity_label.replace(
                    "Power spectral density of ", ""
                )
                self.quantity_label = self.quantity_label.replace(
                    "Power spectral density", ""
                )

    def set_contrast(self, vmin, vmax):
        self.vmin, self.vmax = vmin, vmax
        self.update(data_changed=False, auto_contrast=True)

    def optimize_colorbar(self, number_of_ticks=5, tolerance=5, step_prec_max=1):
        vmin, vmax = self.vmin, self.vmax
        _range = vmax - vmin
        step = _range / (number_of_ticks - 1)
        step_oom = math_tools.order_of_magnitude(step)

        def optimize_for_oom(oom):
            self.colorbar_step = math.floor(step / 10**oom) * 10**oom
            self.colorbar_vmin = math.floor(vmin / 10**oom) * 10**oom
            self.colorbar_vmax = self.colorbar_vmin + self.colorbar_step * (
                number_of_ticks - 1
            )
            self.colorbar_locs = (
                np.arange(0, number_of_ticks) * self.colorbar_step + self.colorbar_vmin
            )

        def check_tolerance():
            if abs(self.colorbar_vmax - vmax) / vmax > (tolerance / 100.0) or abs(
                self.colorbar_vmin - vmin
            ) > (tolerance / 100.0):
                return True
            else:
                return False

        optimize_for_oom(step_oom)
        i = 1
        while check_tolerance() and i <= step_prec_max:
            optimize_for_oom(step_oom - i)
            i += 1
