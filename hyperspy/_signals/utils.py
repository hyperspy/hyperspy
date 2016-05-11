# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from skimage import draw

# import matplotlib.cm as cm
import io
import os
from nbformat import read, write


class RoiRect(object):
    """ Class for getting a mouse drawn rectangle
    Based on the example from:
    http://matplotlib.org/users/event_handling.html#draggable-rectangle-exercise
    Note that:

    * It makes only one roi

    """

    def __init__(self):
        self.ax = plt.gca()
        self.rect = Rectangle((0, 0), 1, 1, fc='none', ec='r')
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.ax.add_patch(self.rect)
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        print('press')
        self.x0 = event.xdata
        self.y0 = event.ydata
        self.rect.set_linestyle('dashed')
        self.set = False

    def on_release(self, event):
        print('release')
        self.x1 = event.xdata
        self.y1 = event.ydata
        self.rect.set_width(self.x1 - self.x0)
        self.rect.set_height(self.y1 - self.y0)
        self.rect.set_xy((self.x0, self.y0))
        self.rect.set_linestyle('solid')
        self.ax.figure.canvas.draw()
        self.set = True
        self.ax.figure.canvas.mpl_disconnect(self.on_press)
        self.ax.figure.canvas.mpl_disconnect(self.on_release)
        self.ax.figure.canvas.mpl_disconnect(self.on_motion)

    def on_motion(self, event):
        # on motion will move the rect if the mouse
        if self.x0 is None:
            return
        if self.set:
            return
        # if event.inaxes != self.rect.axes: return
        self.x1 = event.xdata
        self.y1 = event.ydata
        self.rect.set_width(self.x1 - self.x0)
        self.rect.set_height(self.y1 - self.y0)
        self.rect.set_xy((self.x0, self.y0))
        self.ax.figure.canvas.draw()


class RoiPoint(object):
    """ Class for getting a mouse drawn rectangle
    Based on the example from:
    http://matplotlib.org/users/event_handling.html#draggable-rectangle-exercise
    Note that:

    * It makes only one roi

    """

    def __init__(self):
        self.ax = plt.gca()
        #        self.rect = Rectangle((0,0), 1, 1,fc='none', ec='r')
        self.x0 = None
        self.y0 = None
        self.visible = True
        self.set = False
        self.plt_style = 'r+'
        #        self.x1 = None
        #        self.y1 = None
        #        self.ax.add_patch(self.rect)
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)

    #        self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        if not self.set:
            print('press')
            self.x0 = event.xdata
            self.y0 = event.ydata

            self.draw()

    def on_release(self, event):
        if not self.set:
            print('release')
            #        self.x1 = event.xdata
            #        self.y1 = event.ydata
            #        self.rect.set_width(self.x1 - self.x0)
            #        self.rect.set_height(self.y1 - self.y0)
            #        self.rect.set_xy((self.x0, self.y0))
            #        self.rect.set_linestyle('solid')
            self.ax.figure.canvas.draw()
            self.set = True
            self.ax.figure.canvas.mpl_disconnect('button_press_event')
            self.ax.figure.canvas.mpl_disconnect('button_release_event')

    def draw(self):
        if not self.visible:
            return
        self.ax.plot(self.x0, self.y0, self.plt_style)


# def on_motion(self, event):
#        # on motion will move the rect if the mouse
#        if self.x0 is None: return
#        if self.set: return
#        # if event.inaxes != self.rect.axes: return
#        self.x1 = event.xdata
#        self.y1 = event.ydata
#        self.rect.set_width(self.x1 - self.x0)
#        self.rect.set_height(self.y1 - self.y0)
#        self.rect.set_xy((self.x0, self.y0))
#        self.ax.figure.canvas.draw()
# class roi_rect_new(object):
#    ''' Class for getting a mouse drawn rectangle
#    '''
#    def __init__(self):
#        self.ax = plt.gca()
#        self.rect = Rectangle((0,0), 1, 1, facecolor='None', edgecolor='green')
#        self.x0 = None
#        self.y0 = None
#        self.x1 = None
#        self.y1 = None
#        self.ax.add_patch(self.rect)
#        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
#        self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
#        self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
#    def on_press(self, event):
#        print 'press'
#        self.x0 = event.xdata
#        self.y0 = event.ydata
#        self.x1 = event.xdata
#        self.y1 = event.ydata
#        self.rect.set_width(self.x1 - self.x0)
#        self.rect.set_height(self.y1 - self.y0)
#        self.rect.set_xy((self.x0, self.y0))
#        self.rect.set_linestyle('dashed')
#        self.ax.figure.canvas.draw()
#    def on_motion(self,event):
#        if self.on_press is True:
#            return
#        self.x1 = event.xdata
#        self.y1 = event.ydata
#        self.rect.set_width(self.x1 - self.x0)
#        self.rect.set_height(self.y1 - self.y0)
#        self.rect.set_xy((self.x0, self.y0))
#        self.rect.set_linestyle('dashed')
#        self.ax.figure.canvas.draw()
#    def on_release(self, event):
#        print 'release'
#        self.x1 = event.xdata
#        self.y1 = event.ydata
#        self.rect.set_width(self.x1 - self.x0)
#        self.rect.set_height(self.y1 - self.y0)
#        self.rect.set_xy((self.x0, self.y0))
#        self.rect.set_linestyle('solid')
#        self.ax.figure.canvas.draw()
#        print self.x0,self.x1,self.y0,self.y1
#        return [self.x0,self.x1,self.y0,self.y1]

def poly_to_mask(vertex_row_coords, vertex_col_coords, shape):
    """
    Creates a poligon mask
    """
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


def wrap_to_pi(angle):
    """
    Wrap a given angle in radians to the range -pi to pi.

    @param angle : The angle to be wrapped
    @param type angle : float
    @return : Wrapped angle
    @rtype : float
    """
    return np.mod(angle + np.pi, 2.0 * np.pi) - np.pi


def wrap(angle):
    """
    Wrap a given angle in radians to the range 0 to 2pi.

    @param angle : The angle to be wrapped
    @param type angle : float
    @return : Wrapped angle
    @rtype : float
    """
    return angle % (2 * np.pi)


def hspy_to_2Dnp(hyperspy_signal):
    """
    Transform the 2D and 3D hyperspy.data in a 2D numpy array,
    with the 2nd dimension being the signal dimension.
    """
    TwoDnp = np.empty(
        [hyperspy_signal.axes_manager.navigation_size, hyperspy_signal.axes_manager.signal_size],
        dtype=hyperspy_signal.data.dtype)

    if np.size(hyperspy_signal.data.shape) == 2:
        TwoDnp = hyperspy_signal.data

    if np.size(hyperspy_signal.data.shape) == 3:
        a0, b0, c0 = hyperspy_signal.data.shape
        TwoDnp = hyperspy_signal.data.reshape(a0 * b0, c0)

    return TwoDnp


def remove_outputs(fname):
    """
    remove the outputs from a notebook "fname" and create a new notebook
    """
    with io.open(fname, 'r') as f:
        nb = read(f, 'json')
    for ws in nb.worksheets:
        for cell in ws.cells:
            if cell.cell_type == 'code':
                cell.outputs = []

    base, ext = os.path.splitext(fname)
    new_ipynb = "%s_removed%s" % (base, ext)
    with io.open(new_ipynb, 'w', encoding='utf8') as f:
        write(nb, f, 'json')
    print('wrote {}'.format(new_ipynb))

    return "Done"
