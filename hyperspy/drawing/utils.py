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

import copy
import itertools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from hyperspy.misc.utils import unfold_if_multidim


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
            # When using WX window.connect does not supports multiple funtions
            for f in figure._on_window_close:
                f()
            plt.close(figure)
        window.Bind(wx.EVT_CLOSE, function_wrapper)

    elif backend == 'TkAgg':
        def function_wrapper(*args):
            # When using TK window.connect does not supports multiple funtions
            for f in figure._on_window_close:
                f()
        figure.canvas.manager.window.bind("<Destroy>", function_wrapper)

    elif backend == 'Qt4Agg':
        # PyQt
        # In PyQt window.connect supports multiple funtions
        from IPython.external.qt_for_kernel import QtCore
        window.connect(window, QtCore.SIGNAL('closing()'), function)
    else:
        raise AttributeError("The %s backend is not supported. " % backend)


def plot_RGB_map(im_list, normalization='single', dont_plot=False):
    """Plot 2 or 3 maps in RGB.

    Parameters
    ----------
    im_list : list of Image instances
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
        for i in xrange(rgb.shape[2]):
            rgb[:, :, i] /= rgb[:,:, i].max()
    elif normalization == 'global':
        rgb /= rgb.max()
    rgb = rgb.clip(0, rgb.max())
    if not dont_plot:
        figure = plt.figure()
        ax = figure.add_subplot(111)
        ax.frameon = False
        ax.set_axis_off()
        ax.imshow(rgb, interpolation='nearest')
#        cursors.add_axes(ax)
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
    return (left, bottom, right, top, wspace, hspace)


class ColorCycle():
    _color_cycle = [mpl.colors.colorConverter.to_rgba(color) for color
                    in ('b', 'g', 'r', 'c', 'm', 'y', 'k')]

    def __init__(self):
        self.color_cycle = copy.copy(self._color_cycle)

    def __call__(self):
        if not self.color_cycle:
            self.color_cycle = copy.copy(self._color_cycle)
        return self.color_cycle.pop(0)


def _make_heatmap_subplot(spectra):
    from hyperspy._signals.image import Image
    im = Image(spectra.data, axes=spectra.axes_manager._get_axes_dicts())
    im.mapped_parameters.title = spectra.mapped_parameters.title
    im.plot()
    return im._plot.signal_plot.ax

def _make_cascade_subplot(spectra, ax, color="blue", padding=1):
    max_value = 0
    for spectrum in spectra:
        spectrum_yrange = (np.nanmax(spectrum.data) -
                           np.nanmin(spectrum.data))
        if spectrum_yrange > max_value:
            max_value = spectrum_yrange
    for spectrum_index, (spectrum, color) in enumerate(zip(spectra, color)):
        x_axis = spectrum.axes_manager.signal_axes[0]
        data_to_plot = ((spectrum.data - spectrum.data.min()) /
                            float(max_value) + spectrum_index * padding)
        ax.plot(x_axis.axis, data_to_plot, color=color)
    _set_spectrum_xlabel(spectrum, ax)
    ax.set_yticks([])
    ax.autoscale(tight=True)

def _plot_spectrum(spectrum, ax, color="blue"):
    x_axis = spectrum.axes_manager.signal_axes[0]
    ax.plot(x_axis.axis, spectrum.data, color=color)

def _set_spectrum_xlabel(spectrum, ax):
    x_axis = spectrum.axes_manager.signal_axes[0]
    ax.set_xlabel("%s (%s)" % (x_axis.name, x_axis.units))

def plot_spectra(
        spectra,
        style='cascade',
        color=None,
        padding=0.1,
        fig=None,):
    """Plot several spectral in the same figure.

    Extra keyword arguments are passed to `matplotlib.figure`.

    Parameters
    ----------
    spectra : iterable
        Ordered spectra list to plot. If `style` is "cascade" or "mosaic"
        the spectra can have diffent size and axes.
    style : {'cascade', 'mosaic', 'heatmap'}
        The style of the plot.
    color : valid matplotlib color or a list of them or `None`
        Sets the color of the lines of the plots when `style` is "cascade"
        or "mosaic". If a list, if its length is
        less than the number of spectra to plot, the colors will be cycled. If
        If `None`, use default matplotlib color cycle.
    padding : float, optional, default 0.1
        1 guarantees that there is not overlapping. However,
        in many cases a value between 0 and 1 can produce a tigther plot
        without overlapping. Negative values have the same effect but
        reverse the order of the spectra without reversing the order of the
        colors.
    fig : {matplotlib figure, None}
        If None, a default figure will be created.

    Example
    -------
    >>> s = load("some_spectra")
    >>> utils.plot.plot_spectra(s, style='cascade', color='red', padding=0.5)

    To save the plot as a png-file

    >>> utils.plot.plot_spectra(s).figure.savefig("test.png")


    Returns
    -------
    ax: {matplotlib axes | array of matplotlib axes}
        An array is returned when `style` is "mosaic".

    """
    import hyperspy.signal

    if color is not None:
        if hasattr(color, "__iter__"):
            color  = itertools.cycle(color)
        elif isinstance(color, basestring):
            color  = itertools.cycle([color])
        else:
            raise ValueError("Color must be None, a valid matplotlib color "
                            "string or a list of valid matplotlib colors.")
    else:
        color  = itertools.cycle(plt.rcParams['axes.color_cycle'])


    if style == 'cascade':
        if fig is None:
            fig = plt.figure()
        ax = fig.add_subplot(111)
        _make_cascade_subplot(spectra,
                              ax,
                              color=color,
                              padding=padding)

    elif style == 'mosaic':
        default_fsize = plt.rcParams["figure.figsize"]
        figsize = (default_fsize[0], default_fsize[1] * len(spectra))
        fig, subplots = plt.subplots(len(spectra), 1, figsize=figsize)
        for ax, spectrum, color in zip(subplots, spectra, color):
            _plot_spectrum(spectrum, ax, color=color)
            ax.set_yticks([])
            if not isinstance(spectra, hyperspy.signal.Signal):
                _set_spectrum_xlabel(spectrum, ax)
        if isinstance(spectra, hyperspy.signal.Signal):
            _set_spectrum_xlabel(spectrum, ax)
        fig.tight_layout()

    elif style == 'heatmap':
        if not isinstance(spectra, hyperspy.signal.Signal):
            import hyperspy.utils
            spectra = hyperspy.utils.stack(spectra)
        refold = unfold_if_multidim(spectra)
        ax = _make_heatmap_subplot(spectra)
        if refold is True:
            spectra.fold()
    ax = ax if style != "mosaic" else subplots

    return ax
