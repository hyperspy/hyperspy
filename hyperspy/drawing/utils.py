# -*- coding: utf-8 -*-
# Copyright 2007-2024 The HyperSpy developers
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
import itertools
import logging
import textwrap
import warnings
from functools import partial

import dask.array as da
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import traits.api as t
from matplotlib.backend_bases import key_press_handler
from mpl_toolkits.axes_grid1 import make_axes_locatable
from packaging.version import Version
from rsciio.utils import rgb_tools

import hyperspy
import hyperspy.api as hs
from hyperspy.defaults_parser import preferences
from hyperspy.misc.utils import to_numpy

_logger = logging.getLogger(__name__)


def contrast_stretching(data, vmin=None, vmax=None):
    """Estimate bounds of the data to display.

    Parameters
    ----------
    data: numpy array
    vmin, vmax: scalar, str, None
        If str, formatted as 'xth', use this value to calculate the percentage
        of pixels that are left out of the lower and upper bounds.
        For example, for a vmin of '1th', 1% of the lowest will be ignored to
        estimate the minimum value. Similarly, for a vmax value of '1th', 1%
        of the highest value will be ignored in the estimation of the maximum
        value. See :func:`numpy.percentile` for more explanation.
        If None, use the percentiles value set in the preferences.
        If float of integer, keep this value as bounds.

    Returns
    -------
    vmin, vmax: scalar
        The low and high bounds.

    Raises
    ------
    ValueError
        if the value of `vmin` `vmax` is out of the valid range for percentile
        calculation (in case of string values).

    """
    if np.issubdtype(data.dtype, bool):
        # in case of boolean, simply return 0, 1
        return 0, 1

    def _parse_value(value, value_name):
        if value is None:
            if value_name == "vmin":
                value = "0th"
            elif value_name == "vmax":
                value = "100th"
        if isinstance(value, str):
            value = float(value.split("th")[0])
        if not 0 <= value <= 100:
            raise ValueError(f"{value_name} must be in the range[0, 100].")
        return value

    if np.ma.is_masked(data):
        # If there is a mask, compressed the data to remove the masked data
        data = np.ma.masked_less_equal(data, 0).compressed()

    # If vmin, vmax are float or int, we keep the value, if not we calculate
    # the precentile value
    if not isinstance(vmin, (float, int)):
        vmin = np.nanpercentile(data, _parse_value(vmin, "vmin"))
    if not isinstance(vmax, (float, int)):
        vmax = np.nanpercentile(data, _parse_value(vmax, "vmax"))

    return vmin, vmax


MPL_DIVERGING_COLORMAPS = [
    "BrBG",
    "bwr",
    "coolwarm",
    "PiYG",
    "PRGn",
    "PuOr",
    "RdBu",
    "RdGy",
    "RdYIBu",
    "RdYIGn",
    "seismic",
    "Spectral",
]
# Add reversed colormaps
MPL_DIVERGING_COLORMAPS += [cmap + "_r" for cmap in MPL_DIVERGING_COLORMAPS]


def centre_colormap_values(vmin, vmax):
    """Calculate vmin and vmax to set the colormap midpoint to zero.

    Parameters
    ----------
    vmin, vmax : float
        The range of data to display.

    Returns
    -------
    float
        The values to obtain a centre colormap.

    """

    absmax = max(abs(vmin), abs(vmax))
    return -absmax, absmax


def create_figure(
    window_title=None,
    _on_figure_window_close=None,
    disable_xyscale_keys=False,
    **kwargs,
):
    """Create a matplotlib figure.

    This function adds the possibility to execute another function
    when the figure is closed and to easily set the window title. Any
    keyword argument is passed to the plt.figure function.

    Parameters
    ----------
    window_title : None, string, optional
        Default None.
    _on_figure_window_close : None, function, optional
        Default None.
    disable_xyscale_keys : bool, optional
        Disable the `k`, `l` and `L` shortcuts which toggle the x or y axis
        between linear and log scale. Default False.

    Returns
    -------
    fig : plt.figure

    """
    fig = plt.figure(**kwargs)
    if window_title is not None:
        # remove non-alphanumeric characters to prevent file saving problems
        # This is a workaround for:
        #   https://github.com/matplotlib/matplotlib/issues/9056
        reserved_characters = r'<>"/\|?*'
        for c in reserved_characters:
            window_title = window_title.replace(c, "")
        window_title = window_title.replace("\n", " ")
        window_title = window_title.replace(":", " -")
        fig.canvas.manager.set_window_title(window_title)
    if disable_xyscale_keys and hasattr(fig.canvas, "toolbar"):
        # hack the `key_press_handler` to disable the `k`, `l`, `L` shortcuts
        manager = fig.canvas.manager
        fig.canvas.mpl_disconnect(manager.key_press_handler_id)
        manager.key_press_handler_id = manager.canvas.mpl_connect(
            "key_press_event",
            lambda event: key_press_handler_custom(event, manager.canvas),
        )
    if _on_figure_window_close is not None:
        on_figure_window_close(fig, _on_figure_window_close)
    return fig


def key_press_handler_custom(event, canvas):
    if event.key not in ["k", "l", "L"]:
        key_press_handler(event, canvas, canvas.manager.toolbar)


def on_figure_window_close(figure, function):
    """Connects a close figure signal to a given function.

    Parameters
    ----------

    figure : matplotlib.figure.Figure
        The figure to close
    function : callable

    """

    def function_wrapper(evt):
        function()

    figure.canvas.mpl_connect("close_event", function_wrapper)


def plot_RGB_map(im_list, normalization="single", dont_plot=False):
    """Plot 2 or 3 maps in RGB.

    Parameters
    ----------
    im_list : list of Signal2D instances
    normalization : 'single', 'global', optional
        Default 'single'.
    dont_plot : bool, optional
        Default False.

    Returns
    -------
    array: RGB matrix

    """
    #    from widgets import cursors
    height, width = im_list[0].data.shape[:2]
    rgb = np.zeros((height, width, 3))
    rgb[:, :, 0] = im_list[0].data.squeeze()
    rgb[:, :, 1] = im_list[1].data.squeeze()
    if len(im_list) == 3:
        rgb[:, :, 2] = im_list[2].data.squeeze()
    if normalization == "single":
        for i in range(len(im_list)):
            rgb[:, :, i] /= rgb[:, :, i].max()
    elif normalization == "global":
        rgb /= rgb.max()
    rgb = rgb.clip(0, rgb.max())
    if not dont_plot:
        figure = plt.figure()
        ax = figure.add_subplot(111)
        ax.frameon = False
        ax.set_axis_off()
        ax.imshow(rgb, interpolation="nearest")
        #        cursors.set_mpl_ax(ax)
        figure.canvas.draw_idle()
    else:
        return rgb


def subplot_parameters(fig):
    """Returns a list of the subplot parameters of a mpl figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure

    Returns
    -------
    tuple : (left, bottom, right, top, wspace, hspace)

    """
    wspace = fig.subplotpars.wspace
    hspace = fig.subplotpars.hspace
    left = fig.subplotpars.left
    right = fig.subplotpars.right
    top = fig.subplotpars.top
    bottom = fig.subplotpars.bottom
    return left, bottom, right, top, wspace, hspace


class ColorCycle:
    _color_cycle = [
        mcolors.to_rgba(color) for color in ("b", "g", "r", "c", "m", "y", "k")
    ]

    def __init__(self):
        self.color_cycle = copy.copy(self._color_cycle)

    def __call__(self):
        if not self.color_cycle:
            self.color_cycle = copy.copy(self._color_cycle)
        return self.color_cycle.pop(0)


def plot_signals(
    signal_list, sync=True, navigator="auto", navigator_list=None, **kwargs
):
    """Plot several signals at the same time.

    Parameters
    ----------
    signal_list : list of :class:`~.api.signals.BaseSignal`
        If sync is set to True, the signals must have the
        same navigation shape, but not necessarily the same signal shape.
    sync : bool, optional
        If True (default), the signals will share navigation. All the signals
        must have the same navigation shape for this to work, but not
        necessarily the same signal shape.
    navigator : None, :class:`~.api.signals.BaseSignal` or str
    {``'auto'`` | ``'spectrum'`` | ``'slider'`` }, default ``"auto"``
        See signal.plot docstring for full description.
    navigator_list : None, list of :class:`~.api.signals.BaseSignal` or list of str, default None
        Set different navigator options for the signals. Must use valid
        navigator arguments: 'auto', None, 'spectrum', 'slider', or a
        HyperSpy Signal. The list must have the same size as signal_list.
        If None, the argument specified in navigator will be used.
    **kwargs : dict
        Any extra keyword arguments are passed to each signal ``plot`` method.

    Examples
    --------

    >>> s1 = hs.signals.Signal1D(np.arange(100).reshape((10, 10)))
    >>> s2 = hs.signals.Signal1D(np.arange(100).reshape((10, 10)) * -1)
    >>> hs.plot.plot_signals([s1, s2])

    Specifying the navigator:

    >>> hs.plot.plot_signals([s1, s2], navigator="slider") # doctest: +SKIP

    Specifying the navigator for each signal:

    >>> s3 = hs.signals.Signal1D(np.ones((10, 10)))
    >>> s_nav = hs.signals.Signal1D(np.ones((10)))
    >>> hs.plot.plot_signals(
    ...    [s1, s2, s3], navigator_list=["slider", None, s_nav]
    ...    ) # doctest: +SKIP

    """

    from hyperspy.signal import BaseSignal

    if navigator_list:
        if not (len(signal_list) == len(navigator_list)):
            raise ValueError(
                "signal_list and navigator_list must" " have the same size"
            )

    if sync:
        axes_manager_list = []
        for signal in signal_list:
            axes_manager_list.append(signal.axes_manager)

        if not navigator_list:
            navigator_list = []
        if navigator is None:
            navigator_list.extend([None] * len(signal_list))
        elif isinstance(navigator, BaseSignal):
            navigator_list.append(navigator)
            navigator_list.extend([None] * (len(signal_list) - 1))
        elif navigator == "slider":
            navigator_list.append("slider")
            navigator_list.extend([None] * (len(signal_list) - 1))
        elif navigator == "spectrum":
            navigator_list.extend(["spectrum"] * len(signal_list))
        elif navigator == "auto":
            navigator_list.extend(["auto"] * len(signal_list))
        else:
            raise ValueError(
                'navigator must be one of "spectrum","auto",'
                ' "slider", None, a Signal instance'
            )

        # Check to see if the spectra have the same navigational shapes
        temp_shape_first = axes_manager_list[0].navigation_shape
        for i, axes_manager in enumerate(axes_manager_list):
            temp_shape = axes_manager.navigation_shape
            if not (temp_shape_first == temp_shape):
                raise ValueError("The spectra do not have the same navigation shape")
            axes_manager_list[i] = axes_manager.deepcopy()
            if i > 0:
                for axis0, axisn in zip(
                    axes_manager_list[0].navigation_axes,
                    axes_manager_list[i].navigation_axes,
                ):
                    axes_manager_list[i]._axes[axisn.index_in_array] = axis0
            del axes_manager

        for signal, navigator, axes_manager in zip(
            signal_list, navigator_list, axes_manager_list
        ):
            signal.plot(axes_manager=axes_manager, navigator=navigator, **kwargs)

    # If sync is False
    else:
        if not navigator_list:
            navigator_list = []
            navigator_list.extend([navigator] * len(signal_list))
        for signal, navigator in zip(signal_list, navigator_list):
            signal.plot(navigator=navigator, **kwargs)


def _make_heatmap_subplot(spectra, **plot_kwargs):
    from hyperspy._signals.signal2d import Signal2D

    im = Signal2D(spectra.data, axes=spectra.axes_manager._get_axes_dicts())
    im.metadata.General.title = spectra.metadata.General.title
    im.plot(**plot_kwargs)
    return im._plot.signal_plot.ax


def set_xaxis_lims(mpl_ax, hs_axis):
    """
    Set the matplotlib axis limits to match that of a HyperSpy axis.

    Parameters
    ----------
    mpl_ax : :class:`matplotlib.axis.Axis`
        The ``matplotlib`` axis to change.
    hs_axis : :class:`~hyperspy.axes.DataAxis`
        The data axis that contains the values which control the scaling.
    """
    x_axis_lower_lim = hs_axis.axis[0]
    x_axis_upper_lim = hs_axis.axis[-1]
    mpl_ax.set_xlim(x_axis_lower_lim, x_axis_upper_lim)


def _make_overlap_plot(spectra, ax, color, linestyle, **kwargs):
    for spectrum_index, (spectrum, color, linestyle) in enumerate(
        zip(spectra, color, linestyle)
    ):
        x_axis = spectrum.axes_manager.signal_axes[0]
        spectrum = _transpose_if_required(spectrum, 1)
        ax.plot(
            x_axis.axis, _parse_array(spectrum), color=color, ls=linestyle, **kwargs
        )
        set_xaxis_lims(ax, x_axis)
    _set_spectrum_xlabel(spectra, ax)
    ax.set_ylabel("Intensity")
    ax.autoscale(tight=True)


def _make_cascade_subplot(spectra, ax, color, linestyle, padding=1, **kwargs):
    max_value = 0
    for spectrum in spectra:
        spectrum_yrange = np.nanmax(spectrum.data) - np.nanmin(spectrum.data)
        if spectrum_yrange > max_value:
            max_value = spectrum_yrange
    for i, (spectrum, color, linestyle) in enumerate(zip(spectra, color, linestyle)):
        x_axis = spectrum.axes_manager.signal_axes[0]
        data = _parse_array(_transpose_if_required(spectrum, 1))
        data_to_plot = (data - data.min()) / float(max_value) + i * padding
        ax.plot(x_axis.axis, data_to_plot, color=color, ls=linestyle, **kwargs)
        set_xaxis_lims(ax, x_axis)
    _set_spectrum_xlabel(spectra, ax)
    ax.set_yticks([])
    ax.autoscale(tight=True)


def _plot_spectrum(spectrum, ax, color="blue", linestyle="-", **kwargs):
    x_axis = spectrum.axes_manager.signal_axes[0]
    ax.plot(x_axis.axis, _parse_array(spectrum), color=color, ls=linestyle, **kwargs)
    set_xaxis_lims(ax, x_axis)


def _set_spectrum_xlabel(spectrum, ax):
    s = spectrum[-1] if isinstance(spectrum, (list, tuple)) else spectrum
    x_axis = s.axes_manager.signal_axes[0]
    ax.set_xlabel("%s (%s)" % (x_axis.name, x_axis.units))


def _transpose_if_required(signal, expected_dimension):
    # EDS profiles or maps have signal dimension = 0 and navigation dimension
    # 1 or 2. For convenience, transpose the signal if possible
    if (
        signal.axes_manager.signal_dimension == 0
        and signal.axes_manager.navigation_dimension == expected_dimension
    ):
        return signal.T
    else:
        return signal


def _parse_array(signal):
    """Convenience function to parse array from a signal."""
    data = signal.data
    if isinstance(data, da.Array):
        data = data.compute()
    return to_numpy(data)


def plot_images(
    images,
    cmap=None,
    no_nans=False,
    per_row=3,
    label="auto",
    labelwrap=30,
    suptitle=None,
    suptitle_fontsize=18,
    colorbar="default",
    centre_colormap="auto",
    scalebar=None,
    scalebar_color="white",
    axes_decor="all",
    padding=None,
    tight_layout=False,
    aspect="auto",
    min_asp=0.1,
    namefrac_thresh=0.4,
    fig=None,
    vmin=None,
    vmax=None,
    overlay=False,
    colors="auto",
    alphas=1.0,
    legend_picking=True,
    legend_loc="upper right",
    pixel_size_factor=None,
    **kwargs,
):
    """Plot multiple images either as sub-images or overlayed in one figure.

    Parameters
    ----------
    images : list of :class:`~.api.signals.Signal2D` or :class:`~.api.signals.BaseSignal`
        `images` should be a list of Signals to plot. For
        :class:`~.api.signals.BaseSignal` with navigation dimensions 2 and
        signal dimension 0, the signal will be transposed to form a `Signal2D`.
        Multi-dimensional images will have each plane plotted as a separate
        image. If any of the signal shapes is not suitable, a ValueError will be
        raised.
    cmap : None, (list of) matplotlib.colors.Colormap or str, default None
        The colormap used for the images, by default uses the setting
        ``color map signal`` from the plot preferences. A list of colormaps can
        also be provided, and the images will cycle through them. Optionally,
        the value ``'mpl_colors'`` will cause the cmap to loop through the
        default ``matplotlib`` colors (to match with the default output of the
        :func:`~.api.plot.plot_spectra` method).
        Note: if using more than one colormap, using the ``'single'``
        option for ``colorbar`` is disallowed.
    no_nans : bool, optional
        If True, set nans to zero for plotting.
    per_row : int, optional
        The number of plots in each row.
    label : None, str, list of str, optional
        Control the title labeling of the plotted images.
        If None, no titles will be shown.
        If 'auto' (default), function will try to determine suitable titles
        using Signal2D titles, falling back to the 'titles' option if no good
        short titles are detected.
        Works best if all images to be plotted have the same beginning
        to their titles.
        If 'titles', the title from each image's `metadata.General.title`
        will be used.
        If any other single str, images will be labeled in sequence using
        that str as a prefix.
        If a list of str, the list elements will be used to determine the
        labels (repeated, if necessary).
    labelwrap : int, optional
        Integer specifying the number of characters that will be used on
        one line.
        If the function returns an unexpected blank figure, lower this
        value to reduce overlap of the labels between figures.
    suptitle : str, optional
        Title to use at the top of the figure. If called with label='auto',
        this parameter will override the automatically determined title.
    suptitle_fontsize : int, optional
        Font size to use for super title at top of figure.
    colorbar : 'default', 'multi', 'single', None, optional
        Controls the type of colorbars that are plotted, incompatible with
        ``overlay=True``.
        If 'default', same as 'multi' when ``overlay=False``, otherwise same
        as ``None``.
        If 'multi', individual colorbars are plotted for each (non-RGB) image.
        If 'single', all (non-RGB) images are plotted on the same scale,
        and one colorbar is shown for all.
        If None, no colorbar is plotted.
    centre_colormap : 'auto', True, False, optional
        If True, the centre of the color scheme is set to zero. This is
        particularly useful when using diverging color schemes. If 'auto'
        (default), diverging color schemes are automatically centred.
    scalebar : None, 'all', list of int, optional
        If None (or False), no scalebars will be added to the images.
        If 'all', scalebars will be added to all images.
        If list of ints, scalebars will be added to each image specified.
    scalebar_color : str, optional
        A valid MPL color string; will be used as the scalebar color.
    axes_decor : 'all', 'ticks', 'off', None, optional
        Controls how the axes are displayed on each image; default is 'all'.
        If 'all', both ticks and axis labels will be shown.
        If 'ticks', no axis labels will be shown, but ticks/labels will.
        If 'off', all decorations and frame will be disabled.
        If None, no axis decorations will be shown, but ticks/frame will.
    padding : None, dict, optional
        This parameter controls the spacing between images.
        If None, default options will be used.
        Otherwise, supply a dictionary with the spacing options as
        keywords and desired values as values.
        Values should be supplied as used in
        :func:`matplotlib.pyplot.subplots_adjust`,
        and can be 'left', 'bottom', 'right', 'top', 'wspace' (width) and
        'hspace' (height).
    tight_layout : bool, optional
        If true, hyperspy will attempt to improve image placement in
        figure using matplotlib's tight_layout.
        If false, repositioning images inside the figure will be left as
        an exercise for the user.
    aspect : str, float, int, optional
        If 'auto', aspect ratio is auto determined, subject to min_asp.
        If 'square', image will be forced onto square display.
        If 'equal', aspect ratio of 1 will be enforced.
        If float (or int/long), given value will be used.
    min_asp : float, optional
        Minimum aspect ratio to be used when plotting images.
    namefrac_thresh : float, optional
        Threshold to use for auto-labeling. This parameter controls how
        much of the titles must be the same for the auto-shortening of
        labels to activate. Can vary from 0 to 1. Smaller values
        encourage shortening of titles by auto-labeling, while larger
        values will require more overlap in titles before activing the
        auto-label code.
    fig : matplotlib.figure.Figure, default None
        If set, the images will be plotted to an existing matplotlib figure.
    vmin, vmax: scalar, str, None
        If str, formatted as 'xth', use this value to calculate the percentage
        of pixels that are left out of the lower and upper bounds.
        For example, for a vmin of '1th', 1% of the lowest will be ignored to
        estimate the minimum value. Similarly, for a vmax value of '1th', 1%
        of the highest value will be ignored in the estimation of the maximum
        value. It must be in the range [0, 100].
        See :func:`numpy.percentile` for more explanation.
        If None, use the percentiles value set in the preferences.
        If float or integer, keep this value as bounds.
        Note: vmin is ignored when overlaying images.
    overlay : bool, optional
        If True, overlays the images with different colors rather than plotting
        each image as a subplot.
    colors : 'auto', list of str, optional
        If list, it must contains colors acceptable to matplotlib [1]_.
        If ``'auto'``, colors will be taken from matplotlib.colors.BASE_COLORS.
    alphas : float or list of float, optional
        Float value or a list of floats corresponding to the alpha value of
        each color.
    legend_picking: bool, optional
        If True (default), an image can be toggled on and off by clicking on
        the legended line. For ``overlay=True`` only.
    legend_loc : str, int, optional
        This parameter controls where the legend is placed on the figure
        see the :func:`matplotlib.pyplot.legend` docstring for valid values
    pixel_size_factor : None, int or float, optional
        If ``None`` (default), the size of the figure is taken from the
        matplotlib ``rcParams``. Otherwise sets the size of the figure when
        plotting an overlay image. The higher the number the larger the figure
        and therefore a greater number of pixels are used. This value will be
        ignored if a Figure is provided.
    **kwargs, optional
        Additional keyword arguments passed to :func:`matplotlib.pyplot.imshow`.

    Returns
    -------
    axes_list : list
        A list of subplot axes that hold the images.

    See Also
    --------
    plot_spectra : Plotting of multiple spectra
    plot_signals : Plotting of multiple signals
    plot_histograms : Compare signal histograms

    References
    ----------
    .. [1] Matplotlib colors API: https://matplotlib.org/stable/api/colors_api.html.

    Notes
    -----
    `interpolation` is a useful parameter to provide as a keyword
    argument to control how the space between pixels is interpolated. A
    value of ``'nearest'`` will cause no interpolation between pixels.

    `tight_layout` is known to be quite brittle, so an option is provided
    to disable it. Turn this option off if output is not as expected,
    or try adjusting `label`, `labelwrap`, or `per_row`.

    """

    def __check_single_colorbar(cbar):
        if cbar == "single":
            raise ValueError(
                "Cannot use a single colorbar with multiple "
                "colormaps. Please check for compatible "
                "arguments."
            )

    from hyperspy.drawing.widgets import ScaleBar
    from hyperspy.signal import BaseSignal

    # Check that we have a hyperspy signal
    im = [images] if not isinstance(images, (list, tuple)) else images
    for image in im:
        if not isinstance(image, BaseSignal):
            raise ValueError(
                "`images` must be a list of image signals or a "
                "multi-dimensional signal. "
                f"{repr(type(images))} was given."
            )

    # For list of EDS maps, transpose the BaseSignal
    if isinstance(images, (list, tuple)):
        images = [_transpose_if_required(image, 2) for image in images]

    # If input is >= 1D signal (e.g. for multi-dimensional plotting),
    # copy it and put it in a list so labeling works out as (x,y) when plotting
    if isinstance(images, BaseSignal) and images.axes_manager.navigation_dimension > 0:
        images = [images._deepcopy_with_new_data(images.data)]

    n = 0
    for i, sig in enumerate(images):
        if sig.axes_manager.signal_dimension != 2:
            raise ValueError(
                "This method only plots signals that are images. "
                "The signal dimension must be equal to 2. "
                "The signal at position " + repr(i) + " was " + repr(sig) + "."
            )
        # increment n by the navigation size, or by 1 if the navigation size is
        # <= 0
        n += (
            sig.axes_manager.navigation_size
            if sig.axes_manager.navigation_size > 0
            else 1
        )

    # Check compatibility of colorbar and overlay arguments
    if overlay and colorbar != "default":
        _logger.info(
            f"`colorbar='{colorbar}'` is incompatible with "
            "`overlay=True`. Colorbar is disable."
        )
        colorbar = None
    # Setting the default value
    elif colorbar == "default":
        colorbar = "multi"

    # If no cmap given, get default colormap from pyplot:
    if cmap is None:
        cmap = [preferences.Plot.cmap_signal]
    elif cmap == "mpl_colors":
        color = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
        cmap = _make_cmaps(color)
        __check_single_colorbar(colorbar)
    # cmap is list, tuple, or something else iterable (but not string):
    elif hasattr(cmap, "__iter__") and not isinstance(cmap, str):
        try:
            cmap = [c.name for c in cmap]  # convert colormap to string
        except AttributeError:
            cmap = [c for c in cmap]  # c should be string if not colormap
        __check_single_colorbar(colorbar)
    elif isinstance(cmap, mpl.colors.Colormap):
        cmap = [cmap.name]  # convert single colormap to list with string
    elif isinstance(cmap, str):
        cmap = [cmap]  # cmap is single string, so make it a list
    else:
        # Didn't understand cmap input, so raise error
        raise ValueError(
            "The provided cmap value was not understood. Please " "check input values."
        )

    # If any of the cmaps given are diverging, and auto-centering, set the
    # appropriate flag:
    if centre_colormap == "auto":
        centre_colormaps = []
        for c in cmap:
            if c in MPL_DIVERGING_COLORMAPS:
                centre_colormaps.append(True)
            else:
                centre_colormaps.append(False)
    # if it was True, just convert to list
    elif centre_colormap:
        centre_colormaps = [True]
    # likewise for false
    elif not centre_colormap:
        centre_colormaps = [False]

    # finally, convert lists to cycle generators for adaptive length:
    centre_colormaps = itertools.cycle(centre_colormaps)
    cmap = itertools.cycle(cmap)

    # Sort out the labeling:
    div_num = 0
    all_match = False
    shared_titles = False
    user_labels = False

    if label is None:
        pass
    elif label == "auto":
        # Use some heuristics to try to get base string of similar titles

        label_list = [x.metadata.General.title for x in images]

        # Find the shortest common string between the image titles
        # and pull that out as the base title for the sequence of images
        # array in which to store arrays
        res = np.zeros((len(label_list), len(label_list[0]) + 1))
        res[:, 0] = 1

        # j iterates the strings
        for j in range(len(label_list)):
            # i iterates length of substring test
            for i in range(1, len(label_list[0]) + 1):
                # stores whether or not characters in title match
                res[j, i] = label_list[0][:i] in label_list[j]

        # sum up the results (1 is True, 0 is False) and create
        # a substring based on the minimum value (this will be
        # the "smallest common string" between all the titles
        if res.all():
            basename = label_list[0]
            div_num = len(label_list[0])
            all_match = True
        else:
            div_num = int(min(np.sum(res, 1)))
            basename = label_list[0][: div_num - 1]
            all_match = False

        # trim off any '(' or ' ' characters at end of basename
        if div_num > 1:
            basename = basename.strip()
            if len(basename) > 1 and basename[len(basename) - 1] == "(":
                basename = basename[:-1]

        # namefrac is ratio of length of basename to the image name
        # if it is high (e.g. over 0.5), we can assume that all images
        # share the same base
        if len(label_list[0]) > 0:
            namefrac = float(len(basename)) / len(label_list[0])
        else:
            # If label_list[0] is empty, it means there was probably no
            # title set originally, so nothing to share
            namefrac = 0

        if namefrac > namefrac_thresh:
            # there was a significant overlap of label beginnings
            shared_titles = True
            # only use new suptitle if one isn't specified already
            if suptitle is None:
                suptitle = basename

        else:
            # there was not much overlap, so default back to 'titles' mode
            shared_titles = False
            label = "titles"
            div_num = 0

    elif label == "titles":
        # Set label_list to each image's pre-defined title
        label_list = [x.metadata.General.title for x in images]

    elif isinstance(label, str):
        # Set label_list to an indexed list, based off of label
        label_list = [f"{label} {num}" for num in range(n)]

    elif isinstance(label, list) and all(isinstance(x, str) for x in label):
        label_list = label
        user_labels = True
        # If list of labels is longer than the number of images, just use the
        # first n elements
        if len(label_list) > n:
            del label_list[n:]
        if len(label_list) < n:
            label_list *= (n // len(label_list)) + 1
            del label_list[n:]

    else:
        raise ValueError("Did not understand input of labels.")

    # Check if we need to add a scalebar for some of the images
    if isinstance(scalebar, (list, tuple)) and all(
        isinstance(x, int) for x in scalebar
    ):
        scalelist = True
    else:
        scalelist = False

    if scalebar not in [None, False, "all"] and scalelist is False:
        raise ValueError(
            "Did not understand scalebar input. Must be None, "
            "'all', or list of ints."
        )

    # Determine appropriate number of images per row
    if overlay:
        # only a single image
        per_row = rows = 1
    else:
        rows = int(np.ceil(n / float(per_row)))
        if n < per_row:
            per_row = n

    # Set overall figure size and define figure (if not pre-existing)
    if fig is None:
        w, h = plt.rcParams["figure.figsize"]
        dpi = plt.rcParams["figure.dpi"]
        if overlay and axes_decor == "off":
            shape = images[0].axes_manager.signal_shape
            if pixel_size_factor is None:
                # Cap the maximum dimension of figure to
                # plt.rcParams['figure.figsize']
                aspect_ratio = shape[0] / shape[1]
                if aspect_ratio >= w / h:
                    if label is not None and w / aspect_ratio < 1.0:
                        # Needs enough space for the labels
                        w = 1.0 * aspect_ratio
                    figsize = (w, w / aspect_ratio)
                else:
                    if scalebar is not None and h * aspect_ratio < 2.0:
                        # Needs enough width for the scalebar
                        h = 2.0 / aspect_ratio
                    figsize = (h * aspect_ratio, h)
            else:
                figsize = [pixel_size_factor * v / dpi for v in shape]
        else:
            k = max(w, h) / max(per_row, rows)
            figsize = [k * i for i in (per_row, rows)]
        f = plt.figure(figsize=figsize, dpi=dpi)
    else:
        f = fig

    # Initialize list to hold subplot axes
    axes_list = []

    # Initialize list of rgb tags
    isrgb = [False] * len(images)

    # Check to see if there are any rgb images in list
    # and tag them using the isrgb list
    for i, img in enumerate(images):
        if rgb_tools.is_rgbx(img.data):
            isrgb[i] = True

    # Determine how many non-rgb images there are
    non_rgb = list(itertools.compress(images, [not j for j in isrgb]))
    if len(non_rgb) == 0 and colorbar is not None:
        colorbar = None
        warnings.warn("Sorry, colorbar is not implemented for RGB images.")

    def check_list_length(arg, arg_name):
        if isinstance(arg, (list, tuple)):
            if len(arg) != n:
                _logger.warning(
                    f"The provided {arg_name} values are ignored "
                    "because the length of the list does not "
                    "match the number of images"
                )
                arg = [None] * n
        return arg

    # Find global min and max values of all the non-rgb images for use with
    # 'single' scalebar, otherwise define this value later.
    if colorbar == "single":
        # check that vmin and vmax are not list
        if any([isinstance(v, (tuple, list)) for v in [vmin, vmax]]):
            _logger.warning(
                "The provided vmin or vmax value are ignored "
                "because it needs to be a scalar or a str "
                "to be compatible with a single colorbar. "
                "The default values are used instead."
            )
            vmin, vmax = None, None
        vmin_max = np.array(
            [contrast_stretching(_parse_array(i), vmin, vmax) for i in non_rgb]
        )
        _vmin, _vmax = vmin_max[:, 0].min(), vmin_max[:, 1].max()
        if next(centre_colormaps):
            _vmin, _vmax = centre_colormap_values(_vmin, _vmax)

    else:
        vmin = check_list_length(vmin, "vmin")
        vmax = check_list_length(vmax, "vmax")

    idx = 0
    ax_im_list = [0] * len(isrgb)

    # Replot: create a list to store references to the images
    replot_ims = []

    def transparent_single_color_cmap(color):
        """Return a single color matplotlib cmap with the transparency increasing
        linearly from 0 to 1."""
        return mcolors.LinearSegmentedColormap.from_list(
            "", [mcolors.to_rgba(color, 0), mcolors.to_rgba(color, 1)]
        )

    # Below is for overlayed images
    if overlay:
        # Check if images all have same scale and therefore can be overlayed.
        for im in images:
            if im.axes_manager[0].scale != images[0].axes_manager[0].scale:
                raise ValueError(
                    "Images are not the same scale and so should" "not be overlayed."
                )

        if vmin is not None:
            _logger.warning("`vmin` is ignored when overlaying images.")

        import matplotlib.patches as mpatches

        factor = plt.rcParams["font.size"] / 100
        if not suptitle and axes_decor == "off":
            ax = f.add_axes([0, 0, 1, 1])
        else:
            ax = f.add_subplot()
        patches = []

        # If no colors are selected use BASE_COLORS
        if colors == "auto":
            colors = []
            for i in range(len(images)):
                colors.append(list(mcolors.BASE_COLORS)[i])

        # If no alphas are selected use 1.0
        if isinstance(alphas, float):
            alphas_list = []
            for i in range(len(images)):
                alphas_list.append(alphas)
            alphas = alphas_list

        ax.imshow(np.zeros_like(images[0].data), cmap="gray")

        # Loop through each image
        for i, im in enumerate(images):
            # Set vmin and vmax
            centre = next(centre_colormaps)  # get next value for centreing
            data = _parse_array(im)

            _vmin = data.min()
            _vmax = vmax[idx] if isinstance(vmax, (tuple, list)) else vmax
            _vmin, _vmax = contrast_stretching(data, _vmin, _vmax)
            if centre:
                _logger.warning("Centering is ignored when overlaying images.")

            ax.imshow(
                data,
                vmin=_vmin,
                vmax=_vmax,
                cmap=transparent_single_color_cmap(colors[i]),
                alpha=alphas[i],
                **kwargs,
            )

            if label is not None:
                if shared_titles:
                    legend_label = label_list[i][div_num - 1 :]
                else:
                    legend_label = label_list[i]

                patches.append(mpatches.Patch(color=colors[i], label=legend_label))

        if label is not None:
            plt.legend(handles=patches, loc=legend_loc)
            if legend_picking:
                animate_legend(fig=f, ax=ax, plot_type="images")

        set_axes_decor(ax, axes_decor)

        if scalebar == "all":
            axes = im.axes_manager.signal_axes
            ax.scalebar = ScaleBar(
                ax=ax,
                units=im.axes_manager[0].units,
                color=scalebar_color,
            )
        axes_list.append(ax)

    # Below is for non-overlayed images
    else:
        # Loop through each image, adding subplot for each one
        for i, ims in enumerate(images):
            # Get handles for the signal axes and axes_manager
            axes_manager = ims.axes_manager
            if axes_manager.navigation_dimension > 0:
                ims = ims._deepcopy_with_new_data(ims.data)
                # Use flyback iterpath to get "natural",
                # i.e. order the user would except
                ims.axes_manager.iterpath = "flyback"
            for j, im in enumerate(ims):
                ax = f.add_subplot(rows, per_row, idx + 1)
                axes_list.append(ax)
                centre = next(centre_colormaps)  # get next value for centring
                data = _parse_array(im)

                # Enable RGB plotting
                if rgb_tools.is_rgbx(data):
                    data = rgb_tools.rgbx2regular_array(data, plot_friendly=True)
                    _vmin, _vmax = None, None
                elif colorbar != "single":
                    _vmin, _vmax = _parse_vmin_vmax(data, vmin, vmax, idx, centre)

                # Remove NaNs (if requested)
                if no_nans:
                    data = np.nan_to_num(data)

                # Get handles for the signal axes and axes_manager
                axes_manager = im.axes_manager
                axes = axes_manager.signal_axes

                # Set dimensions of images
                xaxis = axes[0]
                yaxis = axes[1]

                # Keep extent consistent with `Signal.plot`
                if xaxis.is_uniform and yaxis.is_uniform:
                    xaxis_half_px = xaxis.scale / 2.0
                    yaxis_half_px = yaxis.scale / 2.0
                else:
                    xaxis_half_px = 0
                    yaxis_half_px = 0
                extent = [
                    xaxis.axis[0] - xaxis_half_px,
                    xaxis.axis[-1] + xaxis_half_px,
                    yaxis.axis[-1] + yaxis_half_px,
                    yaxis.axis[0] - yaxis_half_px,
                ]

                if not isinstance(aspect, (int, float)) and aspect not in [
                    "auto",
                    "square",
                    "equal",
                ]:
                    _logger.warning(
                        "Did not understand aspect ratio input. "
                        "Using 'auto' as default."
                    )
                    aspect = "auto"

                if not xaxis.is_uniform or not yaxis.is_uniform:
                    asp = None
                elif aspect == "auto":
                    if float(yaxis.size) / xaxis.size < min_asp:
                        factor = min_asp * float(xaxis.size) / yaxis.size
                    elif float(yaxis.size) / xaxis.size > min_asp**-1:
                        factor = min_asp**-1 * float(xaxis.size) / yaxis.size
                    else:
                        factor = 1
                    asp = abs(factor * float(xaxis.scale) / yaxis.scale)
                elif aspect == "square":
                    asp = abs(extent[1] - extent[0]) / abs(extent[3] - extent[2])
                elif aspect == "equal":
                    asp = 1
                elif isinstance(aspect, (int, float)):
                    asp = aspect
                if "interpolation" not in kwargs.keys():
                    kwargs["interpolation"] = "nearest"

                # Plot image data, using _vmin and _vmax to set bounds,
                # or allowing them to be set automatically if using individual
                # colorbars
                kwargs.update({"cmap": next(cmap), "extent": extent, "aspect": asp})
                axes_im = ax.imshow(data, vmin=_vmin, vmax=_vmax, **kwargs)
                ax_im_list[i] = axes_im

                # If an axis trait is undefined, shut off :
                if (
                    xaxis.units == t.Undefined
                    or yaxis.units == t.Undefined
                    or xaxis.name == t.Undefined
                    or yaxis.name == t.Undefined
                ):
                    if axes_decor == "all":
                        _logger.warning(
                            "Axes labels were requested, but one "
                            "or both of the "
                            "axes units and/or name are undefined. "
                            "Axes decorations have been set to "
                            "'ticks' instead."
                        )
                        axes_decor = "ticks"
                # If all traits are defined, set labels as appropriate:
                else:
                    ax.set_xlabel(axes[0].name + " axis (" + axes[0].units + ")")
                    ax.set_ylabel(axes[1].name + " axis (" + axes[1].units + ")")

                if label:
                    if all_match:
                        title = ""
                    elif shared_titles:
                        title = label_list[i][div_num - 1 :]
                    else:
                        if len(ims) == n:
                            # This is true if we are plotting just 1
                            # multi-dimensional Signal2D
                            title = label_list[idx]
                        elif user_labels:
                            title = label_list[idx]
                        else:
                            title = label_list[i]

                    if ims.axes_manager.navigation_size > 1 and not user_labels:
                        title += " %s" % str(ims.axes_manager.indices)

                    ax.set_title(textwrap.fill(title, labelwrap))

                # Set axes decorations based on user input
                set_axes_decor(ax, axes_decor)

                # If using independent colorbars, add them
                if colorbar == "multi" and not isrgb[i]:
                    div = make_axes_locatable(ax)
                    cax = div.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(axes_im, cax=cax)

                # Add scalebars as necessary
                if (scalelist and idx in scalebar) or scalebar == "all":
                    ax.scalebar = ScaleBar(
                        ax=ax,
                        units=axes[0].units,
                        color=scalebar_color,
                    )
                # Replot: store references to the images
                replot_ims.append(im)

                idx += 1

    # If using a single colorbar, add it, and do tight_layout, ensuring that
    # a colorbar is only added based off of non-rgb Images:
    if colorbar == "single":
        foundim = None
        for i in range(len(isrgb)):
            if (not isrgb[i]) and foundim is None:
                foundim = i

        if foundim is not None:
            f.subplots_adjust(right=0.8)
            cbar_ax = f.add_axes([0.9, 0.1, 0.03, 0.8])
            f.colorbar(ax_im_list[foundim], cax=cbar_ax)
            if tight_layout:
                # tight_layout, leaving room for the colorbar
                plt.tight_layout(rect=[0, 0, 0.9, 1])
        elif tight_layout:
            plt.tight_layout()

    elif tight_layout:
        plt.tight_layout()

    # Set top bounds for shared titles and add suptitle
    if suptitle:
        f.subplots_adjust(top=0.85)
        f.suptitle(suptitle, fontsize=suptitle_fontsize)

    # Adjust subplot spacing according to user's specification
    if padding is not None:
        plt.subplots_adjust(**padding)

    # Replot: connect function
    def on_dblclick(event):
        # On the event of a double click, replot the selected subplot
        if not event.inaxes or not event.dblclick:
            return

        idx_ = axes_list.index(event.inaxes)
        ax_ = axes_list[idx_]
        im_ = replot_ims[idx_]

        # Use some of the info in the subplot
        cm = ax_.images[0].get_cmap()
        clim = ax_.images[0].get_clim()

        sbar = False
        if (scalelist and idx_ in scalebar) or scalebar == "all":
            sbar = True

        im_.plot(
            colorbar=bool(colorbar),
            vmin=clim[0],
            vmax=clim[1],
            no_nans=no_nans,
            aspect=asp,
            scalebar=sbar,
            scalebar_color=scalebar_color,
            cmap=cm,
        )

    f.canvas.mpl_connect("button_press_event", on_dblclick)

    def update_image(image, ax, image_index):
        data = image.data
        im = ax.images[0]
        im.set_data(data)
        _vmin, _vmax = _parse_vmin_vmax(data, vmin, vmax, image_index, centre)
        im.set_clim(vmin=_vmin, vmax=_vmax)
        ax.get_figure().canvas.draw()

    for i, (image, ax) in enumerate(zip(images, axes_list)):
        f = partial(update_image, image, ax, i)
        image.events.data_changed.connect(f, [])
        # disconnect event when closing figure
        disconnect = partial(image.events.data_changed.disconnect, f)
        on_figure_window_close(ax.get_figure(), disconnect)

    return axes_list


def _parse_vmin_vmax(data, vmin, vmax, index, centre):
    _vmin = vmin[index] if isinstance(vmin, (tuple, list)) else vmin
    _vmax = vmax[index] if isinstance(vmax, (tuple, list)) else vmax
    _vmin, _vmax = contrast_stretching(data, _vmin, _vmax)
    if centre:
        _vmin, _vmax = centre_colormap_values(_vmin, _vmax)

    return _vmin, _vmax


def set_axes_decor(ax, axes_decor):
    if axes_decor == "off":
        ax.axis("off")
    elif axes_decor == "ticks":
        ax.set_xlabel("")
        ax.set_ylabel("")
    elif axes_decor == "all":
        pass
    elif axes_decor is None:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels([])
        ax.set_yticklabels([])


def make_cmap(colors, name="my_colormap", position=None, bit=False, register=True):
    """
    Create a matplotlib colormap with customized colors, optionally registering
    it with matplotlib for simplified use.

    Adapted from Chris Slocum's code at:
    https://github.com/CSlocumWX/custom_colormap/blob/master/custom_colormaps/custom_colormaps.py
    and used under the terms of that code's BSD-3 license

    Parameters
    ----------
    colors : iterable
        list of either tuples containing rgb values, or html strings
        Colors should be arranged so that the first color is the lowest
        value for the colorbar and the last is the highest.
    name : str, optional
        Name of colormap to use when registering with matplotlib. Default
        'my_colormap'.
    position : None, iterable, optional
        List containing the values (from [0,1]) that dictate the position
        of each color within the colormap. If None (default), the colors
        will be equally-spaced within the colorbar.
    bit : bool, optional
        True if RGB colors are given in 8-bit [0 to 255] or False if given
        in arithmetic basis [0 to 1] (default).
    register : bool, optional
        Switch to control whether or not to register the custom colormap
        with matplotlib in order to enable use by just the name string.
        Default True.
    """
    bit_rgb = np.linspace(0, 1, 256)

    if position is None:
        position = np.linspace(0, 1, len(colors))
    else:
        if len(position) != len(colors):
            raise ValueError("Position length must be the same as colors")
        elif position[0] != 0 or position[-1] != 1:
            raise ValueError("Position must start with 0 and end with 1")

    cdict = {"red": [], "green": [], "blue": []}

    for pos, color in zip(position, colors):
        if isinstance(color, str):
            color = mcolors.to_rgb(color)
        elif bit:
            color = (bit_rgb[color[0]], bit_rgb[color[1]], bit_rgb[color[2]])

        cdict["red"].append((pos, color[0], color[0]))
        cdict["green"].append((pos, color[1], color[1]))
        cdict["blue"].append((pos, color[2], color[2]))

    cmap = mcolors.LinearSegmentedColormap(name, cdict, 256)

    if register:
        try:
            # Introduced in matplotlib 3.5
            mpl.colormaps.register(cmap, name=name)
        except AttributeError:
            # Deprecated in matplotlib 3.5
            mpl.cm.register_cmap(name, cmap)
    return cmap


def plot_spectra(
    spectra,
    style="overlap",
    color=None,
    linestyle=None,
    drawstyle="default",
    padding=1.0,
    legend=None,
    legend_picking=True,
    legend_loc="upper right",
    fig=None,
    ax=None,
    auto_update=None,
    **kwargs,
):
    """Plot several spectra in the same figure.

    Parameters
    ----------
    spectra : list of :class:`~.api.signals.Signal1D` or :class:`~.api.signals.BaseSignal`
        Ordered spectra list of signal to plot. If `style` is "cascade" or
        "mosaic", the spectra can have different size and axes.
        For  :class:`~.api.signals.BaseSignal` with navigation dimensions 1
        and signal dimension 0, the signal will be transposed to form a
        :class:`~.api.signals.Signal1D`.
    style : {``'overlap'`` | ``'cascade'`` | ``'mosaic'`` | ``'heatmap'``}, default 'overlap'
        The style of the plot: 'overlap' (default), 'cascade', 'mosaic', or
        'heatmap'.
    color : None or (list of) matplotlib color, default None
        Sets the color of the lines of the plots (no action on 'heatmap').
        For a list, if its length is less than the number of spectra to plot,
        the colors will be cycled. If `None` (default), use default matplotlib
        color cycle.
    linestyle : None or (list of) matplotlib line style, default None
        Sets the line style of the plots (no action on 'heatmap').
        The main line style are ``'-'``, ``'--'``, ``'-.'``, ``':'``.
        For a list, if its length is less than the number of
        spectra to plot, linestyle will be cycled.
        If `None`, use continuous lines (same as ``'-'``).
    drawstyle : {``'default'`` | ``'steps'`` | ``'steps-pre'`` | ``'steps-mid'`` | ``'steps-post'``},
        default 'default'
        The drawstyle determines how the points are connected, no action with
        ``style='heatmap'``. See
        :meth:`matplotlib.lines.Line2D.set_drawstyle` for more information.
        The ``'default'`` value is defined by matplotlib.
    padding : float, default 1.0
        Option for "cascade". 1 guarantees that there is no overlapping.
        However, in many cases, a value between 0 and 1 can produce a tighter
        plot without overlapping. Negative values have the same effect but
        reverse the order of the spectra without reversing the order of the
        colors.
    legend : None, list of str, ``'auto'``, default None
       If list of string, legend for 'cascade' or title for 'mosaic' is
       displayed. If 'auto', the title of each spectra (metadata.General.title)
       is used. Default None.
    legend_picking : bool, default True
        If True (default), a spectrum can be toggled on and off by clicking on
        the legended line.
    legend_loc : str or int, optional
        This parameter controls where the legend is placed on the figure;
        see the pyplot.legend docstring for valid values. Default ``'upper right'``.
    fig : None, matplotlib.figure.Figure, default None
        If None (default), a default figure will be created. Specifying `fig` will
        not work for the 'heatmap' style.
    ax : None, matplotlib.axes.Axes, default None
        If None (default), a default ax will be created. Will not work for ``'mosaic'``
        or ``'heatmap'`` style.
    auto_update : bool or None, default None
        If True, the plot will update when the data are changed. Only supported
        with style='overlap' and a list of signal with navigation dimension 0.
        If None (default), update the plot only for style='overlap'.
    **kwargs : dict
        Depending on the style used, the keyword arguments are passed to different functions

        - ``"overlap"`` or ``"cascade"``: arguments passed to :func:`matplotlib.pyplot.figure`
        - ``"mosiac"``: arguments passed to :func:`matplotlib.pyplot.subplots`
        - ``"heatmap"``: arguments  passed to :meth:`~.api.signals.Signal2D.plot`.

    Examples
    --------
    >>> s = hs.signals.Signal1D(np.arange(100).reshape((10, 10)))
    >>> hs.plot.plot_spectra(s, style='cascade', color='red', padding=0.5)
    <Axes: xlabel='<undefined> (<undefined>)'>

    To save the plot as a png-file

    >>> ax = hs.plot.plot_spectra(s)
    >>> ax.figure.savefig("test.png") # doctest: +SKIP

    Returns
    -------
    :class:`matplotlib.axes.Axes` or list of :class:`matplotlib.axes.Axes`
        An array is returned when `style` is 'mosaic'.

    """
    from hyperspy.signal import BaseSignal

    def _reverse_legend(ax_, legend_loc_):
        """
        Reverse the ordering of a matplotlib legend (to be more consistent
        with the default ordering of plots in the 'cascade' and 'overlap'
        styles.

        Parameters
        ----------
        ax_: matplotlib axes

        legend_loc_: str, int
            This parameter controls where the legend is placed on the
            figure; see the pyplot.legend docstring for valid values.
        """
        line = ax_.get_legend()
        labels = [lb.get_text() for lb in list(line.get_texts())]
        # "legendHandles" is deprecated in matplotlib 3.7.0 in favour of
        # "legend_handles".
        if Version(mpl.__version__) >= Version("3.7"):
            handles = line.legend_handles
        else:
            handles = line.legendHandles
        ax_.legend(reversed(handles), reversed(labels), loc=legend_loc_)

    # Before v1.3 default would read the value from prefereces.
    if style == "default":
        style = "overlap"

    if color is not None:
        if isinstance(color, str):
            color = itertools.cycle([color])
        elif hasattr(color, "__iter__"):
            color = itertools.cycle(color)
        else:
            raise ValueError(
                "Color must be None, a valid matplotlib color "
                "string, or a list of valid matplotlib colors."
            )
    else:
        color = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    if linestyle is not None:
        if isinstance(linestyle, str):
            linestyle = itertools.cycle([linestyle])
        elif hasattr(linestyle, "__iter__"):
            linestyle = itertools.cycle(linestyle)
    else:
        linestyle = ["-"] * len(spectra)

    if legend is not None:
        if isinstance(legend, str):
            if legend == "auto":
                legend = [spec.metadata.General.title for spec in spectra]
            else:
                raise ValueError("legend must be None, 'auto' or a list of " "strings.")

    if style == "overlap":
        if fig is None:
            fig = plt.figure(**kwargs)
        if ax is None:
            ax = fig.add_subplot(111)
        _make_overlap_plot(spectra, ax, color, linestyle, drawstyle=drawstyle)
        if legend is not None:
            ax.legend(legend, loc=legend_loc)
            _reverse_legend(ax, legend_loc)
            if legend_picking is True:
                animate_legend(fig=fig, ax=ax, plot_type="spectra")
    elif style == "cascade":
        if fig is None:
            fig = plt.figure(**kwargs)
        if ax is None:
            ax = fig.add_subplot(111)
        _make_cascade_subplot(
            spectra, ax, color, linestyle, padding=padding, drawstyle=drawstyle
        )
        if legend is not None:
            ax.legend(legend, loc=legend_loc)
            _reverse_legend(ax, legend_loc)
            if legend_picking is True:
                animate_legend(fig=fig, ax=ax)

    elif style == "mosaic":
        default_fsize = plt.rcParams["figure.figsize"]
        figsize = (default_fsize[0], default_fsize[1] * len(spectra))
        fig, subplots = plt.subplots(len(spectra), 1, figsize=figsize, **kwargs)
        if legend is None:
            legend = [legend] * len(spectra)
        for spectrum, ax, color, linestyle, legend in zip(
            spectra, subplots, color, linestyle, legend
        ):
            spectrum = _transpose_if_required(spectrum, 1)
            _plot_spectrum(
                spectrum, ax, color=color, linestyle=linestyle, drawstyle=drawstyle
            )
            ax.set_ylabel("Intensity")
            if legend is not None:
                ax.set_title(legend)
            if not isinstance(spectra, BaseSignal):
                _set_spectrum_xlabel(spectrum, ax)
        if isinstance(spectra, BaseSignal):
            _set_spectrum_xlabel(spectrum, ax)
        fig.tight_layout()

    elif style == "heatmap":
        if not isinstance(spectra, BaseSignal):
            import hyperspy.utils

            spectra = [_transpose_if_required(spectrum, 1) for spectrum in spectra]
            spectra = hyperspy.utils.stack(spectra)
        with spectra.unfolded():
            ax = _make_heatmap_subplot(spectra, **kwargs)
            ax.set_ylabel("Spectra")
    ax = ax if style != "mosaic" else subplots

    def update_line(spectrum, line):
        x_axis = spectrum.axes_manager[-1].axis
        line.set_data(x_axis, spectrum.data)
        fig = line.get_figure()
        ax = fig.get_axes()[0]
        # `relim` needs to be called before `autoscale_view`
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()

    if auto_update is None and style == "overlap":
        auto_update = True

    if auto_update:
        if style != "overlap":
            raise ValueError(
                "auto_update=True is only supported with " "style='overlap'."
            )

        for spectrum, line in zip(spectra, ax.get_lines()):
            f = partial(update_line, spectrum, line)
            spectrum.events.data_changed.connect(f, [])
            # disconnect event when closing figure
            disconnect = partial(spectrum.events.data_changed.disconnect, f)
            on_figure_window_close(fig, disconnect)

    return ax


def animate_legend(fig=None, ax=None, plot_type="spectra"):
    """Animate the legend of a figure.

    A spectrum or image can be toggled on and off by clicking on the line in
    the legend.

    Parameters
    ----------

    fig: None, matplotlib.figure, optional
        If None (default), pick the current figure using "plt.gcf".
    ax:  {None, matplotlib.axes}, optional
        If None (Default), pick the current axes using "plt.gca".

    Notes
    -----
    Code inspired from legend_picking.py in the matplotlib gallery.

    """
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()

    leg = ax.get_legend()

    if plot_type == "spectra":
        lines = ax.lines[::-1]
        leglines = leg.get_lines()
    elif plot_type == "images":
        lines = ax.images[1:]
        leglines = leg.get_patches()

    lined = dict()

    for legline, origline in zip(leglines, lines):
        if plot_type == "spectra":
            legline.set_pickradius(preferences.Plot.pick_tolerance)
        legline.set_picker(True)
        lined[legline] = origline

    def onpick(event):
        # on the pick event, find the orig line corresponding to the
        # legend proxy line, and toggle the visibility
        legline = event.artist
        if legline.axes == ax:
            origline = lined[legline]
            vis = not origline.get_visible()
            origline.set_visible(vis)
            # Change the alpha on the line in the legend so we can see what lines
            # have been toggled
            if vis:
                legline.set_alpha(1.0)
            else:
                legline.set_alpha(0.2)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("pick_event", onpick)


def plot_histograms(
    signal_list,
    bins="fd",
    range_bins=None,
    color=None,
    linestyle=None,
    legend="auto",
    fig=None,
    **kwargs,
):
    """Plot the histogram of every signal in the list in one figure.

    This function creates a histogram for each signal and plots the list with
    the :func:`~.api.plot.plot_spectra` function.

    Parameters
    ----------
    signal_list : iterable
        Ordered list of spectra to plot. If ``style`` is ``"cascade"`` or
        ``"mosaic"``, the spectra can have different size and axes.
    bins : int, list or str, optional
        If bins is a string, then it must be one of:

         - ``'knuth'`` : use Knuth's rule to determine bins,
         - ``'scott'`` : use Scott's rule to determine bins,
         - ``'fd'`` : use the Freedman-diaconis rule to determine bins,
         - ``'blocks'`` : use bayesian blocks for dynamic bin widths.
    range_bins : None or tuple, optional
        The minimum and maximum range for the histogram. If not specified,
        it will be (``x.min()``, ``x.max()``).
    color : None, (list of) matplotlib color, optional
        Sets the color of the lines of the plots. For a list, if its length is
        less than the number of spectra to plot, the colors will be cycled.
        If `None`, use default matplotlib color cycle.
    linestyle : None, (list of) matplotlib line style, optional
        The main line styles are ``'-'``, ``'--'``, ``'-.'``, ``':'``.
        For a list, if its length is less than the number of
        spectra to plot, linestyle will be cycled.
        If `None`, use continuous lines (same as ``'-'``).
    legend : None, list of str, ``'auto'``, default ``'auto'``
       Display a legend. If 'auto', the title of each spectra
       (metadata.General.title) is used.
    legend_picking : bool, default True
        If True, a spectrum can be toggled on and off by clicking on
        the line in the legend.
    fig : None, matplotlib.figure.Figure, default None
        If None (default), a default figure will be created.
    **kwargs
        other keyword arguments (weight and density) are described in
        :func:`numpy.histogram`.

    Examples
    --------
    Histograms of two random chi-square distributions.

    >>> img = hs.signals.Signal2D(np.random.chisquare(1, [10, 10, 100]))
    >>> img2 = hs.signals.Signal2D(np.random.chisquare(2, [10, 10, 100]))
    >>> hs.plot.plot_histograms([img, img2], legend=['hist1', 'hist2'])
    <Axes: xlabel='value (<undefined>)', ylabel='Intensity'>

    Returns
    -------
    matplotlib.axes.Axes or list of matplotlib.axes.Axes
        An array is returned when ``style='mosaic'``.

    """
    hists = []
    for obj in signal_list:
        hists.append(obj.get_histogram(bins=bins, range_bins=range_bins, **kwargs))
    return plot_spectra(
        hists,
        style="overlap",
        color=color,
        linestyle=linestyle,
        drawstyle="steps-mid",
        legend=legend,
        fig=fig,
    )


def picker_kwargs(value, kwargs=None):
    if kwargs is None:
        kwargs = {}
    # picker is deprecated in favor of pickradius
    if Version(mpl.__version__) >= Version("3.3.0"):
        kwargs.update({"pickradius": value, "picker": True})
    else:
        kwargs["picker"] = value

    return kwargs


def _create_span_roi_group(sig_ax, N):
    """
    Creates a set of `N` of :py:class:`~.roi.SpanROI`\\ s that sit along axis at sensible positions.

    Arguments
    ---------
    sig_ax: DataAxis
        The axis over which the ROI will be placed
    N: int
        The number of ROIs
    """
    axis = sig_ax.axis
    ax_range = axis[-1] - axis[0]
    span_width = ax_range / (2 * N)

    spans = []

    for i in range(N):
        # create a span that has a unique range
        span = hs.roi.SpanROI(i * span_width + axis[0], (i + 1) * span_width + axis[0])

        spans.append(span)

    return spans


def _create_rect_roi_group(sig_wax, sig_hax, N):
    """
    Creates a set of `N` :py:class:`~roi.RectangularROI`\\ s that sit along
    `waxis` and `haxis` at sensible positions.

    Arguments
    ---------
    sig_wax: DataAxis
        The width axis over which the ROI will be placed
    sig_hax: DataAxis
        The height axis over which the ROI will be placed
    N: int
        The number of ROIs
    """
    waxis = sig_wax.axis
    haxis = sig_hax.axis

    w_range = waxis[-1] - waxis[0]
    h_range = haxis[-1] - haxis[0]

    span_w_width = w_range / (2 * N)
    span_h_width = h_range / (2 * N)

    rects = []

    for i in range(N):
        # create a span that has a unique range
        rect = hs.roi.RectangularROI(
            left=i * span_w_width + waxis[0],
            top=i * span_h_width + haxis[0],
            right=(i + 1) * span_w_width + waxis[0],
            bottom=(i + 1) * span_h_width + haxis[0],
        )

        rects.append(rect)

    return rects


def _make_cmaps(colors):
    cmap_name = []
    for n_color, color in enumerate(colors):
        color = mcolors.to_hex(color)
        name = f"single_color_{color}"
        if name not in plt.colormaps():
            make_cmap(colors=["#000000", color], name=name)
        cmap_name.append(name)
    return cmap_name


def _add_colored_frame(ax, color, animated=True):
    colored_frame = patches.Rectangle(
        (0, 0),
        1,
        1,
        linewidth=10,
        edgecolor=color,
        facecolor="none",
        transform=ax.transAxes,
        animated=animated and ax.get_figure().canvas.supports_blit,
    )
    ax.add_patch(colored_frame)


def _roi_sum(signal, roi, axes, out=None):
    sliced_signal = roi(signal=signal, axes=axes)
    if out is not None:
        # use np.sum if the data doesn't contain nan
        # ~2x (or more for larger array) faster than nansum
        f = np.nansum if np.isnan(sliced_signal.data).any() else np.sum
        out.data[:] = f(sliced_signal, axis=axes)
        out.events.data_changed.trigger(obj=out)
    else:
        # we don't case if this is not optimised for speed since this is
        # expected to be called only when setting up the out signal
        s = sliced_signal.nansum(axis=axes)
        # Reset signal to default Signal1D or Signal2D
        s.set_signal_type("")
        s.metadata.General.title = "Integrated intensity"
        return s


def plot_roi_map(
    signal,
    rois=1,
    color=None,
    cmap=None,
    single_figure=False,
    single_figure_kwargs=None,
    **kwargs,
):
    """
    Plot one or multiple ROI maps of a ``signal``.

    Uses regions of interest (ROIs) to select ranges along the signal axis.

    For each ROI, a plot is generated of the summed data values within
    this signal ROI at each point in the ``signal``'s navigation space.

    The ROIs can be moved interactively and the corresponding map plots will
    update automatically.

    Parameters
    ----------
    signal : :class:`~.api.signals.BaseSignal`
        The signal to inspect.
    rois : int, subclass of :class:`hyperspy.roi.BaseROI` or list of :class:`hyperspy.roi.BaseROI`
        ROIs to slice the signal in signal space. If ``int``, define the number of
        ROIs to use.
    color : list of str or None
        Color of the ROIs. Any string supported by matplotlib to define a color
        can be used. The length of the list must be equal to the number ROIs.
        If None (default), the default matplotlib colors are used.
    cmap : str of list of str or None
        Only for signals with navigation dimension of 2. Define the colormap of the map(s).
        If string, any string supported by matplotlib to define a colormap can be used
        and a colored frame matching the ROI color will be added to the map.
        If list of str, it must be the same length as the ROIs.
        If None (default), the colors from the ``color`` argument are used and no colored
        frame is added.
    single_figure : bool
        Whether to plot on a single figure or several figures.
        If True, :func:`~.api.plot.plot_images` or :func:`~.plot.plot_spectra`
        will be used, depending on the navigation dimension of the signal.
    single_figure_kwargs : dict
        Only when ``single_figure=True``. Keywords arguments are passed to
        :func:`~.api.plot.plot_images` or :func:`~.plot.plot_spectra`
        depending on the navigation dimension of the signal.
        If None, default ``kwargs`` are used with the following changes
        ``scalebar=[0]``, ``axes_decor="off"`` and ``suptitle=""``.
    **kwargs : dict
        The keyword argument are passed to :meth:`~.api.signals.Signal2D.plot`.

    Returns
    -------
    rois : list of :class:`hyperspy.roi.BaseROI`
        The ROI objects that slice ``signal``.
    roi_sums : :class:`~.api.signals.BaseSignal`
        The sums of the signals defined by the ROIs.

    Notes
    -----
    Performance consideration:

    - :class:`~.api.roi.RectangularROI` is ~2x faster than :class:`~.api.roi.CircleROI`.
    - If the data sliced by the ROI contains :obj:`numpy.nan`, :func:`numpy.nansum`
      will be used instead of :func:`numpy.sum` at the cost of a speed penalty (more than
      2 times slower).
    - Plotting ROI maps on a single figure is slower than on separate figures.

    Examples
    --------
    **3D hyperspectral data**

    For 3D hyperspectral data, the ROIs used will be instances of
    :class:`~.api.roi.SpanROI`. Therefore, these ROIs can be used to select
    particular spectral ranges, e.g. a particular peak.

    The map generated for a given ROI is therefore the sum of this spectral
    region at each point in the hyperspectral map. Therefore, regions of the
    sample where this peak is bright will be bright in this map.

    **4D STEM**

    For 4D STEM data, by default, the ROIs used will be instances of
    :class:`~.api.roi.RectangularROI`. Other hyperspy ROIs, such as
    :class:`~.api.roi.CircleROI` can be used. These ROIs can be used
    to select particular regions in reciprocal space, e.g. a particular
    diffraction spot.

    The map generated for a given ROI is the intensity of this
    region at each point in the scan. Therefore, regions of the
    scan where a particular spot is intense will appear bright.
    """
    if signal._plot is None or not signal._plot.is_active:
        signal.plot()

    sig_dims = len(signal.axes_manager.signal_axes)
    nav_dims = len(signal.axes_manager.navigation_axes)

    if sig_dims not in [1, 2]:
        raise ValueError("The signal must have signal dimension of 1 or 2.")

    if nav_dims == 0:
        raise ValueError("Navigation dimension must be larger than 0.")

    if isinstance(rois, int):
        if sig_dims == 1:
            rois = _create_span_roi_group(signal.axes_manager.signal_axes[0], rois)
        elif sig_dims == 2:
            rois = _create_rect_roi_group(*signal.axes_manager.signal_axes, rois)
    if isinstance(rois, hyperspy.roi.BaseROI):
        rois = [rois]

    if color is None:
        color = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
    elif isinstance(color, (list, tuple)):
        if len(rois) != len(color):
            raise ValueError(
                f"The number of rois ({len(rois)}) must match "
                f"the number of colors ({len(color)})."
            )
    else:  # pragma: no cover
        raise ValueError("Provided value of color is not supported.")

    add_colored_frame = False
    if cmap is None:
        cmap = _make_cmaps(color)
    elif isinstance(cmap, str):
        add_colored_frame = True
        cmap = [cmap] * len(rois)
    elif isinstance(color, (list, tuple)):
        if len(rois) != len(cmap):
            raise ValueError(
                f"The number of rois ({len(rois)}) must match "
                f"the number of cmap ({len(cmap)})."
            )
    else:  # pragma: no cover
        raise ValueError("Provided value of cmap is not supported.")

    roi_sums = []
    axes = tuple(axis.index_in_axes_manager for axis in signal.axes_manager.signal_axes)

    for i, (roi, color_, cmap_) in enumerate(zip(rois, color, cmap)):
        # add it to the sum over all positions
        roi.add_widget(signal, axes=axes, color=color_)

        # create the signal that is the sum slice
        roi_sum = _roi_sum(signal, roi=roi, axes=axes)
        # Transpose shape to swap these points per nav into signal points.
        # Cap to 2, since hyperspy can't handle signal dimension higher than 2.
        # Take the two first navigation dimensions by default.
        roi_sum = roi_sum.transpose(
            signal_axes=min(roi_sum.axes_manager.navigation_dimension, 2)
        )

        # connect the span signal changing range to the value of span_sum
        hs.interactive(
            _roi_sum,
            event=roi.events.changed,
            signal=signal,
            roi=roi,
            axes=axes,
            out=roi_sum,
            recompute_out_event=None,
        )

        roi_sums.append(roi_sum)

        if not single_figure:
            roi_sum.plot(cmap=cmap_, **kwargs)
            # Remove widget from signal plot when closing maps figure
            roi_sum._plot.signal_plot.events.closed.connect(roi.remove_widget, [])

            if add_colored_frame:
                _add_colored_frame(roi_sum._plot.signal_plot.ax, color_)

    if single_figure:
        if nav_dims == 1:
            axs = plot_spectra(roi_sums, color=color, **kwargs)
        else:
            # default plot kwargs
            single_figure_kwargs_ = dict(scalebar=[0], axes_decor="off", suptitle="")
            # overwrite defaults with user defined kwargs
            single_figure_kwargs_.update(kwargs)
            axs = plot_images(roi_sums, cmap=cmap, **single_figure_kwargs_)
            if add_colored_frame:
                # hs.plot.plot_images doesn't use blitting
                for ax, color_ in zip(axs, color):
                    _add_colored_frame(ax, color_, animated=False)

        def remove_widgets():  # pragma: no cover
            for roi in rois:
                roi.remove_widget()

        if not isinstance(axs, list):
            axs = [axs]
        on_figure_window_close(axs[0].get_figure(), remove_widgets)

    # return all ya bits for future messing around.
    return rois, roi_sums
