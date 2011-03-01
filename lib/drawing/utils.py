# This file is part of EELSLab.
#
# EELSLab is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# EELSLab is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with EELSLab; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  
# USA

import os

import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

def does_figure_object_exists(fig_obj):
    '''Test if a figure really exist
    '''
    if fig_obj is None:
        return False
    else:
        # Test if the figure really exists. If not call the reset function 
        # and start again. This is necessary because with some backends 
        # EELSLab fails to connect the close event to the function.
        try:
            fig_obj.show()
            return True
        except:
            fig_obj = None
            return False
                
def create_figure(window_title = None, _on_window_close = None):
    fig = plt.figure()
    if window_title is not None:
        fig.canvas.set_window_title(window_title)
    if _on_window_close is not None:
        on_window_close(fig, _on_window_close)
    return fig
                
def on_window_close(figure, function):
    '''Connects a close figure signal to a given function
    
    Parameters
    ----------
    
    figure : mpl figure instance
    function : function
    '''
    window = figure.canvas.manager.window
    backend = plt.get_backend()
    if backend == 'GTKAgg':
        def function_wrapper(*args):
                # This wrapper is needed for the destroying process to carry on
                # after the function call
                function()
        window.connect('destroy', function_wrapper)
    elif backend == 'WXAgg':
        # In linux the following code produces a segmentation fault
        # so it is enabled only for Windows
        if os.name in ['nt','dos']:
            import wx
            def function_wrapper(event):
                # This wrapper is needed for the destroying process to carry on
                # after the function call
                function()
                event.Skip()
            window.Bind(wx.EVT_WINDOW_DESTROY, function_wrapper)
        
#    elif matplotlib.get_backend() == 'TkAgg':
        # Tkinter does not return the window when sending the closing
        # signal. Furthermore, for some reason that I don't understand,
        # it blocks ipython. For this reason it is now disable until I find a
        # way around the problem.
#        figure.canvas.manager.window.bind("<Destroy>",function)
#    elif matplotlib.get_backend() == 'Qt4Agg':
#        from PyQt4.QtCore import SIGNAL
#        window = figure.canvas.manager.window
#        window.connect(window, SIGNAL('destroyed()'), function)
        # I don't understand the qt syntax for connecting. Therefore, it is 
        # disable until I have the time to study it.

def plot_RGB_map(im_list, normalization = 'single', dont_plot = False):
    '''Plots 2 or 3 maps in RGB
    
    Parameters
    ----------
    im_list : list of Image instances
    normalization : {'single', 'global'}
    dont_plot : bool
    
    Returns
    -------
    array: RGB matrix
    '''
    from widgets import cursors
    width, height = im_list[0].data_cube.shape[:2]
    rgb = np.zeros((height, width,3))
    rgb[:,:,0] = im_list[0].data_cube.T.squeeze()
    rgb[:,:,1] = im_list[1].data_cube.T.squeeze()
    if len(im_list) == 3:
        rgb[:,:,2] = im_list[2].data_cube.T.squeeze()
    if normalization == 'single':
        for i in range(rgb.shape[2]):
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
        cursors.add_axes(ax)
        figure.canvas.draw()
    else:
        return rgb
        
def subplot_parameters(fig):
    '''Returns a list of the subplot paramters of a mpl figure
    
    Parameters
    ----------
    fig : mpl figure
    
    Returns
    -------
    tuple : (left, bottom, right, top, wspace, hspace)
    '''
    wspace = fig.subplotpars.wspace
    hspace = fig.subplotpars.hspace
    left = fig.subplotpars.left
    right = fig.subplotpars.right
    top = fig.subplotpars.top
    bottom = fig.subplotpars.bottom
    return (left, bottom, right, top, wspace, hspace)
