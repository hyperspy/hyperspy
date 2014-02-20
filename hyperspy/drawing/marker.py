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

    """Marker that can be added to a figure

    Attributes
    ----------

    type : {'line','text','pointer'}
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
        Set the data in a structured array.
        For type='line', 'x1','y1','x2','y2' can be defined
        For type='text', 'x1','y1','text' can be defined
        For type='pointer', 'x1','y1','size' can be defined



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
        elif value == 'pointer':
            lp['color'] = 'black'
            lp['linewidth'] = None
            #lp['pickradius'] = 5.0
        else:
            raise ValueError(
                "`type` must be one of "
                "{\'line\',\'text\',\'pointer\'}"
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
        elif self.type == 'pointer':
            self.marker = self.ax.scatter(self.get_data_position('x1'),
                                       self.get_data_position(
                                           'y1'),
                                         **self.marker_properties)
            if self.get_data_position('size') is None:
                self.set_data(size=20)
                data = self.data
            self.marker._sizes=[self.get_data_position('size')]

        self.marker.set_animated(True)
        self.axes_manager.connect(self.update)
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
                                    ('text', object),('size', object)])

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
        elif self.type == 'pointer':
            self.marker.set_offsets([self.get_data_position('x1'),
                                      self.get_data_position('y1')])
            self.marker._sizes = [self.get_data_position('size')]        
        try:
            #self.ax.figure.canvas.draw()
            self.ax.hspy_fig._draw_animated()
        except:
            pass
