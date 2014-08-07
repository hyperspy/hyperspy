# -*- coding: utf-8 -*-
# Copyright 2007-2011 The HyperSpy developers
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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from hyperspy.misc.utils import unfold_if_multidim
from hyperspy.defaults_parser import preferences


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


def plot_signals(signal_list, sync=True, navigator="auto",
                 navigator_list=None):
    """Plot several signals at the same time.

    Parameters
    ----------
    signal_list : list of Signal instances
        If sync is set to True, the signals must have the
        same navigation shape, but not necessarily the same signal shape.
    sync : True or False, default "True"
        If True: the signals will share navigation, all the signals
        must have the same navigation shape for this to work, but not
        necessarily the same signal shape.
    navigator : {"auto", None, "spectrum", "slider", Signal}, default "auto"
        See signal.plot docstring for full description
    navigator_list : {List of navigator arguments, None}, default None
        Set different navigator options for the signals. Must use valid
        navigator arguments: "auto", None, "spectrum", "slider", or a
        hyperspy Signal. The list must have the same size as signal_list.
        If None, the argument specified in navigator will be used.

    Example
    -------

    >>> s_cl = load("coreloss.dm3")
    >>> s_ll = load("lowloss.dm3")
    >>> utils.plot.plot_signals([s_cl, s_ll])

    Specifying the navigator:

    >>> s_cl = load("coreloss.dm3")
    >>> s_ll = load("lowloss.dm3")
    >>> utils.plot_signals([s_cl, s_ll], navigator="slider")

    Specifying the navigator for each signal:

    >>> s_cl = load("coreloss.dm3")
    >>> s_ll = load("lowloss.dm3")
    >>> s_edx = load("edx.dm3")
    >>> s_adf = load("adf.dm3")
    >>> utils.plot.plot_signals(
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
        elif isinstance(navigator, hyperspy.signal.Signal):
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
            signal.plot(axes_manager=axes_manager, navigator=navigator)

    # If sync is False
    else:
        if not navigator_list:
            navigator_list = []
            navigator_list.extend([navigator] * len(signal_list))
        for signal, navigator in zip(signal_list, navigator_list):
            signal.plot(navigator=navigator)


def _make_heatmap_subplot(spectra):
    from hyperspy._signals.image import Image
    im = Image(spectra.data, axes=spectra.axes_manager._get_axes_dicts())
    im.metadata.General.title = spectra.metadata.General.title
    im.plot()
    return im._plot.signal_plot.ax


def _make_overlap_plot(spectra, ax, color="blue", line_style='-'):
    for spectrum_index, (spectrum, color, line_style) in enumerate(
            zip(spectra, color, line_style)):
        x_axis = spectrum.axes_manager.signal_axes[0]
        ax.plot(x_axis.axis, spectrum.data, color=color, ls=line_style)
    _set_spectrum_xlabel(spectrum, ax)
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
    for spectrum_index, (spectrum, color, line_style) in enumerate(
            zip(spectra, color, line_style)):
        x_axis = spectrum.axes_manager.signal_axes[0]
        data_to_plot = ((spectrum.data - spectrum.data.min()) /
                        float(max_value) + spectrum_index * padding)
        ax.plot(x_axis.axis, data_to_plot, color=color, ls=line_style)
    _set_spectrum_xlabel(spectrum, ax)
    ax.set_yticks([])
    ax.autoscale(tight=True)


def _plot_spectrum(spectrum, ax, color="blue", line_style='-'):
    x_axis = spectrum.axes_manager.signal_axes[0]
    ax.plot(x_axis.axis, spectrum.data, color=color, ls=line_style)


def _set_spectrum_xlabel(spectrum, ax):
    x_axis = spectrum.axes_manager.signal_axes[0]
    ax.set_xlabel("%s (%s)" % (x_axis.name, x_axis.units))


def plot_spectra(
        spectra,
        style='default',
        color=None,
        line_style=None,
        padding=1.,
        legend=None,
        legend_picking=True,
        fig=None,
        ax=None,):
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
        the colors will be cycled. If `None`, use default matplotlib color cycle.
    line_style: matplotlib line style or a list of them or `None`
        Sets the line style of the plots (no action on 'heatmap').
        The main line style are '-','--','steps','-.',':'.
        If a list, if its length is less than the number of
        spectra to plot, line_style will be cycled. If
        If `None`, use continuous lines, eg: ('-','--','steps','-.',':')
    padding : float, optional, default 0.1
        Option for "cascade". 1 guarantees that there is not overlapping. However,
        in many cases a value between 0 and 1 can produce a tighter plot
        without overlapping. Negative values have the same effect but
        reverse the order of the spectra without reversing the order of the
        colors.
    legend: None or list of str or 'auto'
       If list of string, legend for "cascade" or title for "mosaic" is
       displayed. If 'auto', the title of each spectra (metadata.General.title)
       is used.
    legend_picking: bool
        If true, a spectrum can be toggle on and off by clicking on
        the legended line.
    fig : matplotlib figure or None
        If None, a default figure will be created. Specifying fig will
        not work for the 'heatmap' style.
    ax : matplotlib ax (subplot) or None
        If None, a default ax will be created. Will not work for 'mosaic'
        or 'heatmap' style.

    Example
    -------
    >>> s = load("some_spectra")
    >>> utils.plot.plot_spectra(s, style='cascade', color='red', padding=0.5)

    To save the plot as a png-file

    >>> utils.plot.plot_spectra(s).figure.savefig("test.png")

    Returns
    -------
    ax: matplotlib axes or list of matplotlib axes
        An array is returned when `style` is "mosaic".

    """
    import hyperspy.signal

    if style == "default":
        style = preferences.Plot.default_style_to_compare_spectra

    if color is not None:
        if hasattr(color, "__iter__"):
            color = itertools.cycle(color)
        elif isinstance(color, basestring):
            color = itertools.cycle([color])
        else:
            raise ValueError("Color must be None, a valid matplotlib color "
                             "string or a list of valid matplotlib colors.")
    else:
        color = itertools.cycle(plt.rcParams['axes.color_cycle'])

    if line_style is not None:
        if hasattr(line_style, "__iter__"):
            line_style = itertools.cycle(line_style)
        elif isinstance(line_style, basestring):
            line_style = itertools.cycle([line_style])
        else:
            raise ValueError("line_style must be None, a valid matplotlib"
                             " line_style string or a list of valid matplotlib"
                             " line_style.")
    else:
        line_style = ['-'] * len(spectra)

    if legend is not None:
        if hasattr(legend, "__iter__"):
            legend = itertools.cycle(legend)
        elif legend == 'auto':
            legend = [spec.metadata.General.title for spec in spectra]
        else:
            raise ValueError("legend must be None, 'auto' or a list of string")

    if style == 'overlap':
        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot(111)
        _make_overlap_plot(spectra,
                           ax,
                           color=color,
                           line_style=line_style,)
        if legend is not None:
            plt.legend(legend)
            if legend_picking is True:
                animate_legend(figure=fig)
    elif style == 'cascade':
        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot(111)
        _make_cascade_subplot(spectra,
                              ax,
                              color=color,
                              line_style=line_style,
                              padding=padding)
        if legend is not None:
            plt.legend(legend)
    elif style == 'mosaic':
        default_fsize = plt.rcParams["figure.figsize"]
        figsize = (default_fsize[0], default_fsize[1] * len(spectra))
        fig, subplots = plt.subplots(len(spectra), 1, figsize=figsize)
        if legend is None:
            legend = [legend] * len(spectra)
        for spectrum, ax, color, line_style, legend in zip(spectra,
                                                           subplots, color, line_style, legend):
            _plot_spectrum(spectrum, ax, color=color, line_style=line_style)
            ax.set_ylabel('Intensity')
            if legend is not None:
                ax.set_title(legend)
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
        ax.set_ylabel('Spectra')
        if refold is True:
            spectra.fold()
    ax = ax if style != "mosaic" else subplots

    return ax


def animate_legend(figure='last'):
    """Animate the legend of a figure.

    A spectrum can be toggle on and off by clicking on the legended line.

    Parameters
    ----------

    figure: 'last' | matplolib.figure
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
        the spectra can have diffent size and axes.
    bins : int or list or str, optional
        If bins is a string, then it must be one of:
        'knuth' : use Knuth's rule to determine bins
        'scotts' : use Scott's rule to determine bins
        'freedman' : use the Freedman-diaconis rule to determine bins
        'blocks' : use bayesian blocks for dynamic bin widths
    range_bins : tuple or None, optional.
        the minimum and maximum range for the histogram. If not specified,
        it will be (x.min(), x.max())
    color : valid matplotlib color or a list of them or `None`, otional.
        Sets the color of the lines of the plots. If a list, if its length is
        less than the number of spectra to plot, the colors will be cycled. If
        If `None`, use default matplotlib color cycle.
    line_style: valid matplotlib line style or a list of them or `None`, otional.
        The main line style are '-','--','steps','-.',':'.
        If a list, if its length is less than the number of
        spectra to plot, line_style will be cycled. If
        If `None`, use continuous lines, eg: ('-','--','steps','-.',':')
    legend: None or list of str or 'auto', otional.
       Display a legend. If 'auto', the title of each spectra
       (metadata.General.title) is used.
    legend_picking: bool, otional.
        If true, a spectrum can be toggle on and off by clicking on
        the legended line.
    fig : matplotlib figure or None, otional.
        If None, a default figure will be created.
    **kwargs
        other keyword arguments (weight and density) are described in
        np.histogram().
    Example
    -------
    Histograms of two random chi-square distributions
    >>> img = signals.Image(np.random.chisquare(1,[10,10,100]))
    >>> img2 = signals.Image(np.random.chisquare(2,[10,10,100]))
    >>> utils.plot.plot_histograms([img,img2],legend=['hist1','hist2'])

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
