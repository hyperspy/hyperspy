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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def does_figure_object_exists(fig_obj):
    """Test if a figure really exist
    """
    if fig_obj is None:
        return False
    else:
        # Test if the figure really exists. If not call the reset function 
        # and start again. This is necessary because with some backends 
        # Hyperspy fails to connect the close event to the function.
        try:
            fig_obj.show()
            return True
        except:
            fig_obj = None
            return False
                
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
    """Connects a close figure signal to a given function
    
    Parameters
    ----------
    
    figure : mpl figure instance
    function : function
    """
    window = figure.canvas.manager.window
    backend = plt.get_backend()
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
            for f in figure._on_window_close:
                f()
            plt.close(figure)
        window.Bind(wx.EVT_CLOSE, function_wrapper)
        
    elif backend == 'TkAgg':
        def function_wrapper(*args):
                function()
        figure.canvas.manager.window.bind("<Destroy>", function_wrapper)

    elif backend == 'Qt4Agg':
        from PyQt4.QtCore import SIGNAL
        window = figure.canvas.manager.window
        window.connect(window, SIGNAL('destroyed()'), function)


def plot_RGB_map(im_list, normalization = 'single', dont_plot = False):
    """Plots 2 or 3 maps in RGB
    
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
    height,width = im_list[0].data.shape[:2]
    rgb = np.zeros((height, width,3))
    rgb[:,:,0] = im_list[0].data.squeeze()
    rgb[:,:,1] = im_list[1].data.squeeze()
    if len(im_list) == 3:
        rgb[:,:,2] = im_list[2].data.squeeze()
    if normalization == 'single':
        for i in xrange(rgb.shape[2]):
            rgb[:,:,i] /= rgb[:,:,i].max()
    elif normalization == 'global':
        rgb /= rgb.max()
    rgb = rgb.clip(0,rgb.max())
    if not dont_plot:
        figure = plt.figure()
        ax = figure.add_subplot(111)
        ax.frameon = False
        ax.set_axis_off()
        ax.imshow(rgb, interpolation = 'nearest')
#        cursors.add_axes(ax)
        figure.canvas.draw()
    else:
        return rgb
        
def subplot_parameters(fig):
    """Returns a list of the subplot paramters of a mpl figure
    
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

def _make_heatmap_subplot(spectra, ax):
    x_axis = spectra.axes_manager.signal_axes[0]
    data = spectra.data
    if spectra.axes_manager.navigation_size == 0:
        ax.imshow(
            [data],
            cmap=plt.cm.jet,
            aspect='auto')
    else:
        y_axis = spectra.axes_manager.navigation_axes[0]
        ax.imshow(
            data,
            extent=[
                x_axis.low_value,
                x_axis.high_value,
                y_axis.low_value,
                y_axis.high_value],
            aspect='auto',
            interpolation="none")
        ax.set_ylabel(y_axis.units)
    ax.set_xlabel(x_axis.units)
    return(ax)

def _make_cascade_subplot(spectra, ax, color='red', reverse_yaxis=False):
    navigation_length = spectra.axes_manager.navigation_size
    if isinstance(color, str):
        if navigation_length == 0:
            color_array = [color] 
        else:
            color_array = [color]*navigation_length
    else:
        color_array = color
    if spectra.axes_manager.navigation_size == 0:
        x_axis = spectra.axes_manager.signal_axes[0]
        data = spectra.data
        ax.plot(x_axis.axis, data, color=color_array[0])
    else:
        max_value = 0 
        for spectrum in spectra:
            spectrum_max_value = spectrum.data.max()
            if spectrum_max_value > max_value:
                max_value = spectrum_max_value
        y_axis = spectra.axes_manager.navigation_axes[0]
        if reverse_yaxis:
            spectra_data = spectra.data[::-1]
        else:
            spectra_data = spectra.data
        for spectrum_index, spectrum_data in enumerate(spectra_data):
            x_axis = spectrum.axes_manager.signal_axes[0]
            data_to_plot = (spectrum_data/float(max_value) + 
                            y_axis.axis[spectrum_index])
            ax.plot(x_axis.axis, data_to_plot,
                    color=color_array[spectrum_index])
        ax.set_ylabel(y_axis.units)

    ax.set_xlim(x_axis.low_value, x_axis.high_value)
    ax.set_xlabel(x_axis.units)
    return(ax)

def _make_mosaic_subplot(spectrum, ax, color='red'):
    x_axis = spectrum.axes_manager.signal_axes[0]
    data = spectrum.data
    ax.plot(x_axis.axis, data, color=color)
    ax.set_xlim(x_axis.low_value, x_axis.high_value)
    ax.set_xlabel(x_axis.units)
    return(ax)
    
def plot_spectra(
    spectra, 
    style='cascade', 
    color='red',
    reverse_yaxis=False,
    filename=None):
    """Parameters
    -----------------
    spectra : iterable
        Ordered spectra list to plot. If style is cascade it is 
        possible to supply several lists of spectra of the same 
        lenght to plot multiple spectra in the same axes.
    style : {'cascade', 'mosaic', 'multiple_files', 'heatmap'}
        The style of the plot. multiple_files will plot every spectra
        in its own file.
    color : string or list of strings, optional
        Sets the color of the plots. If string sets all plots to color.
        If list of strings: the list must be the same length as the
        navigation length of the spectra to be plotted. Default is red
    reverse_yaxis : bool, optional
        Reverse the plotting direction of the navigational axis for 
        cascade style plotting.
    filename : None or string
        If None, raise a window with the plot and return the figure.

    Returns
    -----------
    fig: Matplotlib figure
    
    """
    navigation_length = spectra.axes_manager.navigation_size
    if isinstance(color, str):
        if navigation_length == 0:
            color_array = [color] 
        else:
            color_array = [color]*navigation_length
    else:
        color_array = color

    if style == 'cascade':
        fig = plt.figure()
        ax = fig.add_subplot(111)
        _make_cascade_subplot(spectra, ax,
                              color=color_array,
                              reverse_yaxis=reverse_yaxis)

        if filename is None:
            return(fig)
        else:
            fig.savefig(filename)

    elif style == 'mosaic':
        #Need to find a way to automatically scale the figure size
        fig, subplots = plt.subplots(1, len(spectra))
        for ax, spectrum, color in zip(subplots, spectra, color_array):
            _make_mosaic_subplot(spectrum, ax, color=color)
        if filename is None:
            return(fig)
        else:
            fig.savefig(filename)
    
    elif style == 'multiple_files':
        for spectrum_index, spectrum in enumerate(spectra):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            _make_mosaic_subplot(spectrum, ax,
                                 color=color_array[spectrum_index])
            #Currently only works with png images
            #Should use some more clever method
            filename = filename.replace('.png', '')
            fig.savefig(
                    filename +
                    '_' +
                    str(spectrum_index) +
                    '.png')

    elif style == 'heatmap':
        fig = plt.figure()
        ax = fig.add_subplot(111)
        _make_heatmap_subplot(spectra, ax) 
        if filename is None:
            return(fig)
        else:
            fig.savefig(filename)