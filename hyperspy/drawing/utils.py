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

import copy
import itertools
import textwrap
from traits import trait_base
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import hyperspy as hs
from distutils.version import LooseVersion

from hyperspy.defaults_parser import preferences


def contrast_stretching(data, saturated_pixels):
    """Calculate bounds that leaves out a given percentage of the data.

    Parameters
    ----------
    data: numpy array
    saturated_pixels: scalar
        The percentage of pixels that are left out of the bounds.  For example,
        the low and high bounds of a value of 1 are the 0.5% and 99.5%
        percentiles. It must be in the [0, 100] range.

    Returns
    -------
    vmin, vmax: scalar
        The low and high bounds

    Raises
    ------
    ValueError if the value of `saturated_pixels` is out of the valid range.

    """
    # Sanity check
    if not 0 <= saturated_pixels <= 100:
        raise ValueError(
            "saturated_pixels must be a scalar in the range[0, 100]")
    nans = np.isnan(data)
    if nans.any():
        data = data[~nans]
    vmin = np.percentile(data, saturated_pixels / 2.)
    vmax = np.percentile(data, 100 - saturated_pixels / 2.)
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
    "Spectral", ]
# Add reversed colormaps
MPL_DIVERGING_COLORMAPS += [cmap + "_r" for cmap in MPL_DIVERGING_COLORMAPS]


def centre_colormap_values(vmin, vmax):
    """Calculate vmin and vmax to set the colormap midpoint to zero.

    Parameters
    ----------
    vmin, vmax : scalar
        The range of data to display.

    Returns
    -------
    cvmin, cvmax : scalar
        The values to obtain a centre colormap.

    """

    absmax = max(abs(vmin), abs(vmax))
    return -absmax, absmax


def create_figure(window_title=None,
                  _on_figure_window_close=None,
                  **kwargs):
    """Create a matplotlib figure.

    This function adds the possibility to execute another function
    when the figure is closed and to easily set the window title. Any
    keyword argument is passed to the plt.figure function

    Parameters
    ----------
    window_title : string
    _on_figure_window_close : function

    Returns
    -------

    fig : plt.figure

    """
    fig = plt.figure(**kwargs)
    if window_title is not None:
        fig.canvas.set_window_title(window_title)
    if _on_figure_window_close is not None:
        on_figure_window_close(fig, _on_figure_window_close)
    return fig


def on_figure_window_close(figure, function):
    """Connects a close figure signal to a given function.

    Parameters
    ----------

    figure : mpl figure instance
    function : function

    """
    backend = plt.get_backend()
    if backend not in ("GTKAgg", "WXAgg", "TkAgg", "Qt4Agg"):
        return

    window = figure.canvas.manager.window
    if not hasattr(figure, '_on_window_close'):
        figure._on_window_close = list()
    if function not in figure._on_window_close:
        figure._on_window_close.append(function)

    if backend == 'GTKAgg':
        def function_wrapper(*args):
            function()
        window.connect('destroy', function_wrapper)

    elif backend == 'WXAgg':
        # In linux the following code produces a segmentation fault
        # so it is enabled only for Windows
        import wx

        def function_wrapper(event):
            # When using WX window.connect does not supports multiple functions
            for f in figure._on_window_close:
                f()
            plt.close(figure)
        window.Bind(wx.EVT_CLOSE, function_wrapper)

    elif backend == 'TkAgg':
        def function_wrapper(*args):
            # When using TK window.connect does not supports multiple functions
            for f in figure._on_window_close:
                f()
        figure.canvas.manager.window.bind("<Destroy>", function_wrapper)

    elif backend == 'Qt4Agg':
        # PyQt
        # In PyQt window.connect supports multiple functions
        from IPython.external.qt_for_kernel import QtCore
        window.connect(window, QtCore.SIGNAL('closing()'), function)
    else:
        raise AttributeError("The %s backend is not supported. " % backend)


def plot_RGB_map(im_list, normalization='single', dont_plot=False):
    """Plot 2 or 3 maps in RGB.

    Parameters
    ----------
    im_list : list of Signal2D instances
    normalization : {'single', 'global'}
    dont_plot : bool

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
    if normalization == 'single':
        for i in range(rgb.shape[2]):
            rgb[:, :, i] /= rgb[:, :, i].max()
    elif normalization == 'global':
        rgb /= rgb.max()
    rgb = rgb.clip(0, rgb.max())
    if not dont_plot:
        figure = plt.figure()
        ax = figure.add_subplot(111)
        ax.frameon = False
        ax.set_axis_off()
        ax.imshow(rgb, interpolation='nearest')
#        cursors.set_mpl_ax(ax)
        figure.canvas.draw()
    else:
        return rgb


def subplot_parameters(fig):
    """Returns a list of the subplot parameters of a mpl figure.

    Parameters
    ----------
    fig : mpl figure

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
    _color_cycle = [mpl.colors.colorConverter.to_rgba(color) for color
                    in ('b', 'g', 'r', 'c', 'm', 'y', 'k')]

    def __init__(self):
        self.color_cycle = copy.copy(self._color_cycle)

    def __call__(self):
        if not self.color_cycle:
            self.color_cycle = copy.copy(self._color_cycle)
        return self.color_cycle.pop(0)


def plot_signals(signal_list, sync=True, navigator="auto",
                 navigator_list=None, **kwargs):
    """Plot several signals at the same time.

    Parameters
    ----------
    signal_list : list of BaseSignal instances
        If sync is set to True, the signals must have the
        same navigation shape, but not necessarily the same signal shape.
    sync : True or False, default "True"
        If True: the signals will share navigation, all the signals
        must have the same navigation shape for this to work, but not
        necessarily the same signal shape.
    navigator : {"auto", None, "spectrum", "slider", BaseSignal}, default "auto"
        See signal.plot docstring for full description
    navigator_list : {List of navigator arguments, None}, default None
        Set different navigator options for the signals. Must use valid
        navigator arguments: "auto", None, "spectrum", "slider", or a
        hyperspy Signal. The list must have the same size as signal_list.
        If None, the argument specified in navigator will be used.
    **kwargs
        Any extra keyword arguments are passed to each signal `plot` method.

    Example
    -------

    >>> s_cl = hs.load("coreloss.dm3")
    >>> s_ll = hs.load("lowloss.dm3")
    >>> hs.plot.plot_signals([s_cl, s_ll])

    Specifying the navigator:

    >>> s_cl = hs.load("coreloss.dm3")
    >>> s_ll = hs.load("lowloss.dm3")
    >>> hs.plot.plot_signals([s_cl, s_ll], navigator="slider")

    Specifying the navigator for each signal:

    >>> s_cl = hs.load("coreloss.dm3")
    >>> s_ll = hs.load("lowloss.dm3")
    >>> s_edx = hs.load("edx.dm3")
    >>> s_adf = hs.load("adf.dm3")
    >>> hs.plot.plot_signals(
            [s_cl, s_ll, s_edx], navigator_list=["slider",None,s_adf])

    """

    import hyperspy.signal

    if navigator_list:
        if not (len(signal_list) == len(navigator_list)):
            raise ValueError(
                "signal_list and navigator_list must"
                " have the same size")

    if sync:
        axes_manager_list = []
        for signal in signal_list:
            axes_manager_list.append(signal.axes_manager)

        if not navigator_list:
            navigator_list = []
        if navigator is None:
            navigator_list.extend([None] * len(signal_list))
        elif navigator is "slider":
            navigator_list.append("slider")
            navigator_list.extend([None] * (len(signal_list) - 1))
        elif isinstance(navigator, hyperspy.signal.BaseSignal):
            navigator_list.append(navigator)
            navigator_list.extend([None] * (len(signal_list) - 1))
        elif navigator is "spectrum":
            navigator_list.extend(["spectrum"] * len(signal_list))
        elif navigator is "auto":
            navigator_list.extend(["auto"] * len(signal_list))
        else:
            raise ValueError(
                "navigator must be one of \"spectrum\",\"auto\","
                " \"slider\", None, a Signal instance")

        # Check to see if the spectra have the same navigational shapes
        temp_shape_first = axes_manager_list[0].navigation_shape
        for i, axes_manager in enumerate(axes_manager_list):
            temp_shape = axes_manager.navigation_shape
            if not (temp_shape_first == temp_shape):
                raise ValueError(
                    "The spectra does not have the same navigation shape")
            axes_manager_list[i] = axes_manager.deepcopy()
            if i > 0:
                for axis0, axisn in zip(axes_manager_list[0].navigation_axes,
                                        axes_manager_list[i].navigation_axes):
                    axes_manager_list[i]._axes[axisn.index_in_array] = axis0
            del axes_manager

        for signal, navigator, axes_manager in zip(signal_list,
                                                   navigator_list,
                                                   axes_manager_list):
            signal.plot(axes_manager=axes_manager,
                        navigator=navigator,
                        **kwargs)

    # If sync is False
    else:
        if not navigator_list:
            navigator_list = []
            navigator_list.extend([navigator] * len(signal_list))
        for signal, navigator in zip(signal_list, navigator_list):
            signal.plot(navigator=navigator,
                        **kwargs)


def _make_heatmap_subplot(spectra):
    from hyperspy._signals.signal2d import Signal2D
    im = Signal2D(spectra.data, axes=spectra.axes_manager._get_axes_dicts())
    im.metadata.General.title = spectra.metadata.General.title
    im.plot()
    return im._plot.signal_plot.ax


def _make_overlap_plot(spectra, ax, color="blue", line_style='-'):
    if isinstance(color, str):
        color = [color] * len(spectra)
    if isinstance(line_style, str):
        line_style = [line_style] * len(spectra)
    for spectrum_index, (spectrum, color, line_style) in enumerate(
            zip(spectra, color, line_style)):
        x_axis = spectrum.axes_manager.signal_axes[0]
        ax.plot(x_axis.axis, spectrum.data, color=color, ls=line_style)
    _set_spectrum_xlabel(spectra if isinstance(spectra, hs.signals.BaseSignal)
                         else spectra[-1], ax)
    ax.set_ylabel('Intensity')
    ax.autoscale(tight=True)


def _make_cascade_subplot(
        spectra, ax, color="blue", line_style='-', padding=1):
    max_value = 0
    for spectrum in spectra:
        spectrum_yrange = (np.nanmax(spectrum.data) -
                           np.nanmin(spectrum.data))
        if spectrum_yrange > max_value:
            max_value = spectrum_yrange
    if isinstance(color, str):
        color = [color] * len(spectra)
    if isinstance(line_style, str):
        line_style = [line_style] * len(spectra)
    for spectrum_index, (spectrum, color, line_style) in enumerate(
            zip(spectra, color, line_style)):
        x_axis = spectrum.axes_manager.signal_axes[0]
        data_to_plot = ((spectrum.data - spectrum.data.min()) /
                        float(max_value) + spectrum_index * padding)
        ax.plot(x_axis.axis, data_to_plot, color=color, ls=line_style)
    _set_spectrum_xlabel(spectra if isinstance(spectra, hs.signals.BaseSignal)
                         else spectra[-1], ax)
    ax.set_yticks([])
    ax.autoscale(tight=True)


def _plot_spectrum(spectrum, ax, color="blue", line_style='-'):
    x_axis = spectrum.axes_manager.signal_axes[0]
    ax.plot(x_axis.axis, spectrum.data, color=color, ls=line_style)


def _set_spectrum_xlabel(spectrum, ax):
    x_axis = spectrum.axes_manager.signal_axes[0]
    ax.set_xlabel("%s (%s)" % (x_axis.name, x_axis.units))


def plot_images(images,
                cmap=None,
                no_nans=False,
                per_row=3,
                label='auto',
                labelwrap=30,
                suptitle=None,
                suptitle_fontsize=18,
                colorbar='multi',
                centre_colormap="auto",
                saturated_pixels=0,
                scalebar=None,
                scalebar_color='white',
                axes_decor='all',
                padding=None,
                tight_layout=False,
                aspect='auto',
                min_asp=0.1,
                namefrac_thresh=0.4,
                fig=None,
                *args,
                **kwargs):
    """Plot multiple images as sub-images in one figure.

        Parameters
        ----------
        images : list
            `images` should be a list of Signals (Images) to plot
            If any signal is not an image, a ValueError will be raised
            multi-dimensional images will have each plane plotted as a separate
            image
        cmap : matplotlib colormap, optional
            The colormap used for the images, by default read from pyplot
        no_nans : bool, optional
            If True, set nans to zero for plotting.
        per_row : int, optional
            The number of plots in each row
        label : None, str, or list of str, optional
            Control the title labeling of the plotted images.
            If None, no titles will be shown.
            If 'auto' (default), function will try to determine suitable titles
            using Signal2D titles, falling back to the 'titles' option if no good
            short titles are detected.
            Works best if all images to be plotted have the same beginning
            to their titles.
            If 'titles', the title from each image's metadata.General.title
            will be used.
            If any other single str, images will be labeled in sequence using
            that str as a prefix.
            If a list of str, the list elements will be used to determine the
            labels (repeated, if necessary).
        labelwrap : int, optional
            integer specifying the number of characters that will be used on
            one line
            If the function returns an unexpected blank figure, lower this
            value to reduce overlap of the labels between each figure
        suptitle : str, optional
            Title to use at the top of the figure. If called with label='auto',
            this parameter will override the automatically determined title.
        suptitle_fontsize : int, optional
            Font size to use for super title at top of figure
        colorbar : {'multi', None, 'single'}
            Controls the type of colorbars that are plotted.
            If None, no colorbar is plotted.
            If 'multi' (default), individual colorbars are plotted for each
            (non-RGB) image
            If 'single', all (non-RGB) images are plotted on the same scale,
            and one colorbar is shown for all
        centre_colormap : {"auto", True, False}
            If True the centre of the color scheme is set to zero. This is
            specially useful when using diverging color schemes. If "auto"
            (default), diverging color schemes are automatically centred.
        saturated_pixels: scalar
            The percentage of pixels that are left out of the bounds.  For
            example, the low and high bounds of a value of 1 are the 0.5% and
            99.5% percentiles. It must be in the [0, 100] range.
        scalebar : {None, 'all', list of ints}, optional
            If None (or False), no scalebars will be added to the images.
            If 'all', scalebars will be added to all images.
            If list of ints, scalebars will be added to each image specified.
        scalebar_color : str, optional
            A valid MPL color string; will be used as the scalebar color
        axes_decor : {'all', 'ticks', 'off', None}, optional
            Controls how the axes are displayed on each image; default is 'all'
            If 'all', both ticks and axis labels will be shown
            If 'ticks', no axis labels will be shown, but ticks/labels will
            If 'off', all decorations and frame will be disabled
            If None, no axis decorations will be shown, but ticks/frame will
        padding : None or dict, optional
            This parameter controls the spacing between images.
            If None, default options will be used
            Otherwise, supply a dictionary with the spacing options as
            keywords and desired values as values
            Values should be supplied as used in pyplot.subplots_adjust(),
            and can be:
                'left', 'bottom', 'right', 'top', 'wspace' (width),
                and 'hspace' (height)
        tight_layout : bool, optional
            If true, hyperspy will attempt to improve image placement in
            figure using matplotlib's tight_layout
            If false, repositioning images inside the figure will be left as
            an exercise for the user.
        aspect : str or numeric, optional
            If 'auto', aspect ratio is auto determined, subject to min_asp.
            If 'square', image will be forced onto square display.
            If 'equal', aspect ratio of 1 will be enforced.
            If float (or int/long), given value will be used.
        min_asp : float, optional
            Minimum aspect ratio to be used when plotting images
        namefrac_thresh : float, optional
            Threshold to use for auto-labeling. This parameter controls how
            much of the titles must be the same for the auto-shortening of
            labels to activate. Can vary from 0 to 1. Smaller values
            encourage shortening of titles by auto-labeling, while larger
            values will require more overlap in titles before activing the
            auto-label code.
        fig : mpl figure, optional
            If set, the images will be plotted to an existing MPL figure
        *args, **kwargs, optional
            Additional arguments passed to matplotlib.imshow()

        Returns
        -------
        axes_list : list
            a list of subplot axes that hold the images

        See Also
        --------
        plot_spectra : Plotting of multiple spectra
        plot_signals : Plotting of multiple signals
        plot_histograms : Compare signal histograms

        Notes
        -----
        `interpolation` is a useful parameter to provide as a keyword
        argument to control how the space between pixels is interpolated. A
        value of ``'nearest'`` will cause no interpolation between pixels.

        `tight_layout` is known to be quite brittle, so an option is provided
        to disable it. Turn this option off if output is not as expected,
        or try adjusting `label`, `labelwrap`, or `per_row`

    """
    from hyperspy.drawing.widgets import ScaleBar
    from hyperspy.misc import rgb_tools
    from hyperspy.signal import BaseSignal

    if isinstance(images, BaseSignal) and len(images) is 1:
        images.plot()
        ax = plt.gca()
        return ax
    elif not isinstance(images, (list, tuple, BaseSignal)):
        raise ValueError("images must be a list of image signals or "
                         "multi-dimensional signal."
                         " " + repr(type(images)) + " was given.")

    # Get default colormap from pyplot:
    if cmap is None:
        cmap = plt.get_cmap().name
    elif isinstance(cmap, mpl.colors.Colormap):
        cmap = cmap.name
    if centre_colormap == "auto":
        if cmap in MPL_DIVERGING_COLORMAPS:
            centre_colormap = True
        else:
            centre_colormap = False

    if "vmin" in kwargs:
        user_vmin = kwargs["vmin"]
        del kwargs["vmin"]
    else:
        user_vmin = None
    if "vmax" in kwargs:
        user_vmax = kwargs["vmax"]
        del kwargs["vmax"]
    else:
        user_vmax = None
    # If input is >= 1D signal (e.g. for multi-dimensional plotting),
    # copy it and put it in a list so labeling works out as (x,y) when plotting
    if isinstance(images,
                  BaseSignal) and images.axes_manager.navigation_dimension > 0:
        images = [images._deepcopy_with_new_data(images.data)]

    n = 0
    for i, sig in enumerate(images):
        if sig.axes_manager.signal_dimension != 2:
            raise ValueError("This method only plots signals that are images. "
                             "The signal dimension must be equal to 2. "
                             "The signal at position " + repr(i) +
                             " was " + repr(sig) + ".")
        # increment n by the navigation size, or by 1 if the navigation size is
        # <= 0
        n += (sig.axes_manager.navigation_size
              if sig.axes_manager.navigation_size > 0
              else 1)

    # Sort out the labeling:
    div_num = 0
    all_match = False
    shared_titles = False
    user_labels = False

    if label is None:
        pass
    elif label is 'auto':
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
            basename = label_list[0][:div_num - 1]
            all_match = False

        # trim off any '(' or ' ' characters at end of basename
        if div_num > 1:
            while True:
                if basename[len(basename) - 1] == '(':
                    basename = basename[:-1]
                elif basename[len(basename) - 1] == ' ':
                    basename = basename[:-1]
                else:
                    break

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
            label = 'titles'
            div_num = 0

    elif label is 'titles':
        # Set label_list to each image's pre-defined title
        label_list = [x.metadata.General.title for x in images]

    elif isinstance(label, str):
        # Set label_list to an indexed list, based off of label
        label_list = [label + " " + repr(num) for num in range(n)]

    elif isinstance(label, list) and all(
            isinstance(x, str) for x in label):
        label_list = label
        user_labels = True
        # If list of labels is longer than the number of images, just use the
        # first n elements
        if len(label_list) > n:
            del label_list[n:]
        if len(label_list) < n:
            label_list *= (n / len(label_list)) + 1
            del label_list[n:]

    else:
        raise ValueError("Did not understand input of labels.")

    # Determine appropriate number of images per row
    rows = int(np.ceil(n / float(per_row)))
    if n < per_row:
        per_row = n

    # Set overall figure size and define figure (if not pre-existing)
    if fig is None:
        k = max(plt.rcParams['figure.figsize']) / max(per_row, rows)
        f = plt.figure(figsize=(tuple(k * i for i in (per_row, rows))))
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

    # Determine how many non-rgb Images there are
    non_rgb = list(itertools.compress(images, [not j for j in isrgb]))
    if len(non_rgb) is 0 and colorbar is not None:
        colorbar = None
        warnings.warn("Sorry, colorbar is not implemented for RGB images.")

    # Find global min and max values of all the non-rgb images for use with
    # 'single' scalebar
    if colorbar is 'single':
        g_vmin, g_vmax = contrast_stretching(np.concatenate(
            [i.data.flatten() for i in non_rgb]), saturated_pixels)
        g_vmin = user_vmin if user_vmin is not None else g_vmin
        g_vmax = user_vmax if user_vmax is not None else g_vmax
        if centre_colormap:
            g_vmin, g_vmax = centre_colormap_values(g_vmin, g_vmax)

    # Check if we need to add a scalebar for some of the images
    if isinstance(scalebar, list) and all(isinstance(x, int)
                                          for x in scalebar):
        scalelist = True
    else:
        scalelist = False

    idx = 0
    ax_im_list = [0] * len(isrgb)
    # Loop through each image, adding subplot for each one
    for i, ims in enumerate(images):
        # Get handles for the signal axes and axes_manager
        axes_manager = ims.axes_manager
        if axes_manager.navigation_dimension > 0:
            ims = ims._deepcopy_with_new_data(ims.data)
        for j, im in enumerate(ims):
            idx += 1
            ax = f.add_subplot(rows, per_row, idx)
            axes_list.append(ax)
            data = im.data

            # Enable RGB plotting
            if rgb_tools.is_rgbx(data):
                data = rgb_tools.rgbx2regular_array(data, plot_friendly=True)
                l_vmin, l_vmax = None, None
            else:
                data = im.data
                # Find min and max for contrast
                l_vmin, l_vmax = contrast_stretching(data, saturated_pixels)
                l_vmin = user_vmin if user_vmin is not None else l_vmin
                l_vmax = user_vmax if user_vmax is not None else l_vmax
                if centre_colormap:
                    l_vmin, l_vmax = centre_colormap_values(l_vmin, l_vmax)

            # Remove NaNs (if requested)
            if no_nans:
                data = np.nan_to_num(data)

            # Get handles for the signal axes and axes_manager
            axes_manager = im.axes_manager
            axes = axes_manager.signal_axes

            # Set dimensions of images
            xaxis = axes[0]
            yaxis = axes[1]

            extent = (
                xaxis.low_value,
                xaxis.high_value,
                yaxis.high_value,
                yaxis.low_value,
            )

            if not isinstance(aspect, (int, float)) and aspect not in [
                    'auto', 'square', 'equal']:
                print('Did not understand aspect ratio input. '
                      'Using \'auto\' as default.')
                aspect = 'auto'

            if aspect is 'auto':
                if float(yaxis.size) / xaxis.size < min_asp:
                    factor = min_asp * float(xaxis.size) / yaxis.size
                elif float(yaxis.size) / xaxis.size > min_asp ** -1:
                    factor = min_asp ** -1 * float(xaxis.size) / yaxis.size
                else:
                    factor = 1
                asp = np.abs(factor * float(xaxis.scale) / yaxis.scale)
            elif aspect is 'square':
                asp = abs(extent[1] - extent[0]) / abs(extent[3] - extent[2])
            elif aspect is 'equal':
                asp = 1
            elif isinstance(aspect, (int, float)):
                asp = aspect
            if 'interpolation' not in kwargs.keys():
                kwargs['interpolation'] = 'nearest'

            # Plot image data, using vmin and vmax to set bounds,
            # or allowing them to be set automatically if using individual
            # colorbars
            if colorbar is 'single' and not isrgb[i]:
                axes_im = ax.imshow(data,
                                    cmap=cmap, extent=extent,
                                    vmin=g_vmin, vmax=g_vmax,
                                    aspect=asp,
                                    *args, **kwargs)
                ax_im_list[i] = axes_im
            else:
                axes_im = ax.imshow(data,
                                    cmap=cmap,
                                    extent=extent,
                                    vmin=l_vmin,
                                    vmax=l_vmax,
                                    aspect=asp,
                                    *args, **kwargs)
                ax_im_list[i] = axes_im

            # If an axis trait is undefined, shut off :
            if isinstance(xaxis.units, trait_base._Undefined) or  \
                    isinstance(yaxis.units, trait_base._Undefined) or \
                    isinstance(xaxis.name, trait_base._Undefined) or \
                    isinstance(yaxis.name, trait_base._Undefined):
                if axes_decor is 'all':
                    warnings.warn(
                        'Axes labels were requested, but one '
                        'or both of the '
                        'axes units and/or name are undefined. '
                        'Axes decorations have been set to '
                        '\'ticks\' instead.')
                    axes_decor = 'ticks'
            # If all traits are defined, set labels as appropriate:
            else:
                ax.set_xlabel(axes[0].name + " axis (" + axes[0].units + ")")
                ax.set_ylabel(axes[1].name + " axis (" + axes[1].units + ")")

            if label:
                if all_match:
                    title = ''
                elif shared_titles:
                    title = label_list[i][div_num - 1:]
                else:
                    if len(ims) == n:
                        # This is true if we are plotting just 1
                        # multi-dimensional Signal2D
                        title = label_list[idx - 1]
                    elif user_labels:
                        title = label_list[idx - 1]
                    else:
                        title = label_list[i]

                if ims.axes_manager.navigation_size > 1 and not user_labels:
                    title += " %s" % str(ims.axes_manager.indices)

                ax.set_title(textwrap.fill(title, labelwrap))

            # Set axes decorations based on user input
            if axes_decor is 'off':
                ax.axis('off')
            elif axes_decor is 'ticks':
                ax.set_xlabel('')
                ax.set_ylabel('')
            elif axes_decor is 'all':
                pass
            elif axes_decor is None:
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_xticklabels([])
                ax.set_yticklabels([])

            # If using independent colorbars, add them
            if colorbar is 'multi' and not isrgb[i]:
                div = make_axes_locatable(ax)
                cax = div.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(axes_im, cax=cax)

            # Add scalebars as necessary
            if (scalelist and idx - 1 in scalebar) or scalebar is 'all':
                ax.scalebar = ScaleBar(
                    ax=ax,
                    units=axes[0].units,
                    color=scalebar_color,
                )

    # If using a single colorbar, add it, and do tight_layout, ensuring that
    # a colorbar is only added based off of non-rgb Images:
    if colorbar is 'single':
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

    # If we want to plot scalebars, loop through the list of axes and add them
    if scalebar is None or scalebar is False:
        # Do nothing if no scalebars are called for
        pass
    elif scalebar is 'all':
        # scalebars were taken care of in the plotting loop
        pass
    elif scalelist:
        # scalebars were taken care of in the plotting loop
        pass
    else:
        raise ValueError("Did not understand scalebar input. Must be None, "
                         "\'all\', or list of ints.")

    # Adjust subplot spacing according to user's specification
    if padding is not None:
        plt.subplots_adjust(**padding)

    return axes_list


def plot_spectra(
        spectra,
        style='default',
        color=None,
        line_style=None,
        padding=1.,
        legend=None,
        legend_picking=True,
        legend_loc='upper right',
        fig=None,
        ax=None,
        **kwargs):
    """Plot several spectra in the same figure.

    Extra keyword arguments are passed to `matplotlib.figure`.

    Parameters
    ----------
    spectra : iterable object
        Ordered spectra list to plot. If `style` is "cascade" or "mosaic"
        the spectra can have different size and axes.
    style : {'default', 'overlap','cascade', 'mosaic', 'heatmap'}
        The style of the plot. The default is "overlap" and can be
        customized in `preferences`.
    color : matplotlib color or a list of them or `None`
        Sets the color of the lines of the plots (no action on 'heatmap').
        If a list, if its length is less than the number of spectra to plot,
        the colors will be cycled. If `None`, use default matplotlib color
        cycle.
    line_style: matplotlib line style or a list of them or `None`
        Sets the line style of the plots (no action on 'heatmap').
        The main line style are '-','--','steps','-.',':'.
        If a list, if its length is less than the number of
        spectra to plot, line_style will be cycled. If
        If `None`, use continuous lines, eg: ('-','--','steps','-.',':')
    padding : float, optional, default 0.1
        Option for "cascade". 1 guarantees that there is not overlapping.
        However, in many cases a value between 0 and 1 can produce a tighter
        plot without overlapping. Negative values have the same effect but
        reverse the order of the spectra without reversing the order of the
        colors.
    legend: None or list of str or 'auto'
       If list of string, legend for "cascade" or title for "mosaic" is
       displayed. If 'auto', the title of each spectra (metadata.General.title)
       is used.
    legend_picking: bool
        If true, a spectrum can be toggle on and off by clicking on
        the legended line.
    legend_loc : str or int
        This parameter controls where the legend is placed on the figure;
        see the pyplot.legend docstring for valid values
    fig : matplotlib figure or None
        If None, a default figure will be created. Specifying fig will
        not work for the 'heatmap' style.
    ax : matplotlib ax (subplot) or None
        If None, a default ax will be created. Will not work for 'mosaic'
        or 'heatmap' style.
    **kwargs
        remaining keyword arguments are passed to matplotlib.figure() or
        matplotlib.subplots(). Has no effect on 'heatmap' style.

    Example
    -------
    >>> s = hs.load("some_spectra")
    >>> hs.plot.plot_spectra(s, style='cascade', color='red', padding=0.5)

    To save the plot as a png-file

    >>> hs.plot.plot_spectra(s).figure.savefig("test.png")

    Returns
    -------
    ax: matplotlib axes or list of matplotlib axes
        An array is returned when `style` is "mosaic".

    """
    import hyperspy.signal

    if style == "default":
        style = preferences.Plot.default_style_to_compare_spectra

    if color is not None:
        if isinstance(color, str):
            color = itertools.cycle([color])
        elif hasattr(color, "__iter__"):
            color = itertools.cycle(color)
        else:
            raise ValueError("Color must be None, a valid matplotlib color "
                             "string or a list of valid matplotlib colors.")
    else:
        if LooseVersion(mpl.__version__) >= "1.5.3":
            color = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()["color"])
        else:
            color = itertools.cycle(plt.rcParams['axes.color_cycle'])

    if line_style is not None:
        if isinstance(line_style, str):
            line_style = itertools.cycle([line_style])
        elif hasattr(line_style, "__iter__"):
            line_style = itertools.cycle(line_style)
        else:
            raise ValueError("line_style must be None, a valid matplotlib"
                             " line_style string or a list of valid matplotlib"
                             " line_style.")
    else:
        line_style = ['-'] * len(spectra)

    if legend is not None:
        if isinstance(legend, str):
            if legend == 'auto':
                legend = [spec.metadata.General.title for spec in spectra]
            else:
                raise ValueError("legend must be None, 'auto' or a list of"
                                 " string")

        elif hasattr(legend, "__iter__"):
            legend = itertools.cycle(legend)

    if style == 'overlap':
        if fig is None:
            fig = plt.figure(**kwargs)
        if ax is None:
            ax = fig.add_subplot(111)
        _make_overlap_plot(spectra,
                           ax,
                           color=color,
                           line_style=line_style,)
        if legend is not None:
            plt.legend(legend, loc=legend_loc)
            if legend_picking is True:
                animate_legend(figure=fig)
    elif style == 'cascade':
        if fig is None:
            fig = plt.figure(**kwargs)
        if ax is None:
            ax = fig.add_subplot(111)
        _make_cascade_subplot(spectra,
                              ax,
                              color=color,
                              line_style=line_style,
                              padding=padding)
        if legend is not None:
            plt.legend(legend, loc=legend_loc)
    elif style == 'mosaic':
        default_fsize = plt.rcParams["figure.figsize"]
        figsize = (default_fsize[0], default_fsize[1] * len(spectra))
        fig, subplots = plt.subplots(
            len(spectra), 1, figsize=figsize, **kwargs)
        if legend is None:
            legend = [legend] * len(spectra)
        for spectrum, ax, color, line_style, legend in zip(
                spectra, subplots, color, line_style, legend):
            _plot_spectrum(spectrum, ax, color=color, line_style=line_style)
            ax.set_ylabel('Intensity')
            if legend is not None:
                ax.set_title(legend)
            if not isinstance(spectra, hyperspy.signal.BaseSignal):
                _set_spectrum_xlabel(spectrum, ax)
        if isinstance(spectra, hyperspy.signal.BaseSignal):
            _set_spectrum_xlabel(spectrum, ax)
        fig.tight_layout()

    elif style == 'heatmap':
        if not isinstance(spectra, hyperspy.signal.BaseSignal):
            import hyperspy.utils
            spectra = hyperspy.utils.stack(spectra)
        with spectra.unfolded():
            ax = _make_heatmap_subplot(spectra)
            ax.set_ylabel('Spectra')
    ax = ax if style != "mosaic" else subplots

    return ax


def animate_legend(figure='last'):
    """Animate the legend of a figure.

    A spectrum can be toggle on and off by clicking on the legended line.

    Parameters
    ----------

    figure: 'last' | matplotlib.figure
        If 'last' pick the last figure

    Note
    ----

    Code inspired from legend_picking.py in the matplotlib gallery

    """
    if figure == 'last':
        figure = plt.gcf()
        ax = plt.gca()
    else:
        ax = figure.axes[0]
    lines = ax.lines
    lined = dict()
    leg = ax.get_legend()
    for legline, origline in zip(leg.get_lines(), lines):
        legline.set_picker(5)  # 5 pts tolerance
        lined[legline] = origline

    def onpick(event):
        # on the pick event, find the orig line corresponding to the
        # legend proxy line, and toggle the visibility
        legline = event.artist
        origline = lined[legline]
        vis = not origline.get_visible()
        origline.set_visible(vis)
        # Change the alpha on the line in the legend so we can see what lines
        # have been toggled
        if vis:
            legline.set_alpha(1.0)
        else:
            legline.set_alpha(0.2)
        figure.canvas.draw()

    figure.canvas.mpl_connect('pick_event', onpick)

    plt.show()


def plot_histograms(signal_list,
                    bins='freedman',
                    range_bins=None,
                    color=None,
                    line_style=None,
                    legend='auto',
                    fig=None,
                    **kwargs):
    """Plot the histogram of every signal in the list in the same figure.

    This function creates a histogram for each signal and plot the list with
    the `utils.plot.plot_spectra` function.

    Parameters
    ----------
    signal_list : iterable
        Ordered spectra list to plot. If `style` is "cascade" or "mosaic"
        the spectra can have different size and axes.
    bins : int or list or str, optional
        If bins is a string, then it must be one of:
        'knuth' : use Knuth's rule to determine bins
        'scotts' : use Scott's rule to determine bins
        'freedman' : use the Freedman-diaconis rule to determine bins
        'blocks' : use bayesian blocks for dynamic bin widths
    range_bins : tuple or None, optional.
        the minimum and maximum range for the histogram. If not specified,
        it will be (x.min(), x.max())
    color : valid matplotlib color or a list of them or `None`, optional.
        Sets the color of the lines of the plots. If a list, if its length is
        less than the number of spectra to plot, the colors will be cycled. If
        If `None`, use default matplotlib color cycle.
    line_style: valid matplotlib line style or a list of them or `None`,
    optional.
        The main line style are '-','--','steps','-.',':'.
        If a list, if its length is less than the number of
        spectra to plot, line_style will be cycled. If
        If `None`, use continuous lines, eg: ('-','--','steps','-.',':')
    legend: None or list of str or 'auto', optional.
       Display a legend. If 'auto', the title of each spectra
       (metadata.General.title) is used.
    legend_picking: bool, optional.
        If true, a spectrum can be toggle on and off by clicking on
        the legended line.
    fig : matplotlib figure or None, optional.
        If None, a default figure will be created.
    **kwargs
        other keyword arguments (weight and density) are described in
        np.histogram().
    Example
    -------
    Histograms of two random chi-square distributions
    >>> img = hs.signals.Signal2D(np.random.chisquare(1,[10,10,100]))
    >>> img2 = hs.signals.Signal2D(np.random.chisquare(2,[10,10,100]))
    >>> hs.plot.plot_histograms([img,img2],legend=['hist1','hist2'])

    Returns
    -------
    ax: matplotlib axes or list of matplotlib axes
        An array is returned when `style` is "mosaic".

    """
    hists = []
    for obj in signal_list:
        hists.append(obj.get_histogram(bins=bins,
                                       range_bins=range_bins, **kwargs))
    if line_style is None:
        line_style = 'steps'
    return plot_spectra(hists, style='overlap', color=color,
                        line_style=line_style, legend=legend, fig=fig)
