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

import numpy as np
import matplotlib.pyplot as plt


class Marker(object):

    """Marker that can be added to the signal figure

    Attributes
    ----------

    type : {'line','axvline','axhline','text','pointer'}
        Select the type of markers
    orientation : {None,'v','h'}
        Orientation for lines. 'v' is vertical, 'h' is horizontal.
    marker_properties : dictionary
        Accepts a dictionary of valid (i.e. recognized by mpl.plot)
        containing valid line properties. In addition it understands
        the keyword `type` that can take the following values:
        {'line','text'}

    Methods
    -------
    set_marker_properties
        Enables setting the line_properties attribute using keyword
        arguments.

    set_data
        Set the data in a structured array. Each field of data should have
        the same dimensions than the nagivation axes. Some fields need to be defined
        depending on the type.
        For 'line': 'x1','y1','x2','y2' (All of them if orientation
            is None)
        For 'axvline': 'x1'
        For 'axhline': 'y1'
        For 'text': 'x1','y1','text'
        For 'pointer': 'x1','y1','size'. 'size' is optional

    Example
    -------

    >>> s = signals.Spectrum(random.random([10,100]))
    >>> m = utils.plot.marker()
    >>> m.type = 'axvline'
    >>> m.set_marker_properties(color='green')
    >>> m.set_data(x1=range(10))
    >>> s.plot()
    >>> s._plot.signal_plot.add_marker(m)
    >>> m.plot()

    >>> im = signals.Image(random.random([10,50,50]))
    >>> m = utils.plot.marker()
    >>> m.type = 'text'
    >>> m.set_marker_properties(fontsize = 30,color='red')
    >>> m.set_data(x1=range(10),y1=range(10)[::-1],text='hello')
    >>> im.plot()
    >>> im._plot.signal_plot.add_marker(m)
    >>> m.plot()


    """

    def __init__(self):
        # Data attributes
        self.data = None
        self.axes_manager = None
        self.auto_update = True

        # Properties
        self.marker = None
        self.orientation = None
        self._marker_properties = {}
        self.type = "line"

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, value):
        lp = {}
        if value == 'text':
            lp['color'] = 'black'
            lp['linewidth'] = None
        elif value == 'line':
            lp['linewidth'] = 1
            lp['color'] = 'black'
        elif value == 'axvline':
            lp['linewidth'] = 1
            lp['color'] = 'black'
        elif value == 'axhline':
            lp['linewidth'] = 1
            lp['color'] = 'black'
        elif value == 'pointer':
            lp['color'] = 'black'
            lp['linewidth'] = None
        else:
            raise ValueError(
                "`type` must be one of "
                "{\'line\',\'axvline\',\'axhline\',\'text\',\'pointer\'}"
                "but %s was given" % value)
        self._type = value
        self.marker_properties = lp

    @property
    def marker_properties(self):
        return self._marker_properties

    @marker_properties.setter
    def marker_properties(self, kwargs):
        if 'type' in kwargs:
            self.type = kwargs['type']
            del kwargs['type']

        for key, item in kwargs.iteritems():
            if item is None and key in self._marker_properties:
                del self._marker_properties[key]
            else:
                self._marker_properties[key] = item
        if self.marker is not None:
            plt.setp(self.marker, **self.marker_properties)
            try:
                # self.ax.figure.canvas.draw()
                self.ax.hspy_fig._draw_animated()
            except:
                pass

    def set_marker_properties(self, **kwargs):
        self.marker_properties = kwargs

    def plot(self):
        data = self.data
        if self.type == 'text':
            indices = self.axes_manager.indices[::-1]
            self.marker = self.ax.text(self.get_data_position('x1'),
                                       self.get_data_position(
                                           'y1'), self.get_data_position('text'),
                                       **self.marker_properties)
        elif self.type == 'line':
            self.marker = self.ax.vlines(0, 0, 1,
                                         **self.marker_properties)
            self.set_line_segment()
        elif self.type == 'axvline':
            self.marker = self.ax.axvline(self.get_data_position('x1'),
                                          **self.marker_properties)
        elif self.type == 'axhline':
            self.marker = self.ax.axhline(self.get_data_position('y1'),
                                          **self.marker_properties)
        elif self.type == 'pointer':
            self.marker = self.ax.scatter(self.get_data_position('x1'),
                                          self.get_data_position(
                                              'y1'),
                                          **self.marker_properties)
            if self.get_data_position('size') is None:
                self.set_data(size=20)
                data = self.data
            self.marker._sizes = [self.get_data_position('size')]

        self.marker.set_animated(True)
        # To be discussed, done in Spectrum figure once.
        # self.axes_manager.connect(self.update)
        try:
            self.ax.hspy_fig._draw_animated()
            # self.ax.figure.canvas.draw()
        except:
            pass

    def set_data(self, x1=None, y1=None,
                 x2=None, y2=None, text=None, size=None):
        self.data = np.array((np.array(x1), np.array(y1),
                              np.array(x2), np.array(y2),
                              np.array(text), np.array(size)),
                             dtype=[('x1', object), ('y1', object),
                                    ('x2', object), ('y2', object),
                                    ('text', object), ('size', object)])

    def get_data_position(self, ind):
        data = self.data
        indices = self.axes_manager.indices[::-1]
        if data[ind].item()[()] is None:
            return None
        elif hasattr(data[ind].item()[()], "__iter__"):
            return data[ind].item()[indices]
        else:
            return data[ind].item()[()]

    def set_line_segment(self):
        segments = self.marker.get_segments()
        if self.orientation is None:
            segments[0][0, 0] = self.get_data_position('x1')
            segments[0][0, 1] = self.get_data_position('y1')
            segments[0][1, 0] = self.get_data_position('x2')
            segments[0][1, 1] = self.get_data_position('y2')
        elif 'v' in self.orientation:
            segments[0][0, 0] = self.get_data_position('x1')
            segments[0][1, 0] = segments[0][0, 0]
            if self.get_data_position('y1') is None:
                segments[0][0, 1] = plt.getp(self.marker.axes, 'ylim')[0]
            else:
                segments[0][0, 1] = self.get_data_position('y1')
            if self.get_data_position('y2') is None:
                segments[0][1, 1] = plt.getp(self.marker.axes, 'ylim')[1]
            else:
                segments[0][1, 1] = self.get_data_position('y2')
        elif 'h' in self.orientation:
            segments[0][0, 1] = self.get_data_position('y1')
            segments[0][1, 1] = segments[0][0, 1]
            if self.get_data_position('x1') is None:
                segments[0][0, 0] = plt.getp(self.marker.axes, 'xlim')[0]
            else:
                segments[0][0, 0] = self.get_data_position('x1')
            if self.get_data_position('x2') is None:
                segments[0][1, 0] = plt.getp(self.marker.axes, 'xlim')[1]
            else:
                segments[0][1, 0] = self.get_data_position('x2')
        self.marker.set_segments(segments)

    def close(self):
        try:
            self.marker.remove()
            # m.ax.figure.canvas.draw()
            self.ax.hspy_fig._draw_animated()
        except:
            pass

    def update(self):
        """Update the current spectrum figure"""
        data = self.data

        if self.auto_update is False:
            return
        if self.type == 'text':
            self.marker.set_position([self.get_data_position('x1'),
                                      self.get_data_position('y1')])
            self.marker.set_text(self.get_data_position('text'))
        elif self.type == 'line':
            self.set_line_segment()
        elif self.type == 'axvline':
            self.marker.set_xdata(self.get_data_position('x1'))
        elif self.type == 'axhline':
            self.marker.set_ydata(self.get_data_position('y1'))
        elif self.type == 'pointer':
            self.marker.set_offsets([self.get_data_position('x1'),
                                     self.get_data_position('y1')])
            self.marker._sizes = [self.get_data_position('size')]
        # To be discussed, done in SpectrumLine once.
        # try:
            # self.ax.figure.canvas.draw()
            # self.ax.hspy_fig._draw_animated()
        # except:
            # pass
