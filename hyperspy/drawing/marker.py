# -*- coding: utf-8 -*-
# Copyright 2007-2015 The HyperSpy developers
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

import numpy as np
import matplotlib.pyplot as plt


class MarkerBase(object):

    """Marker that can be added to the signal figure

    Attributes
    ----------
    marker_properties : dictionary
        Accepts a dictionary of valid (i.e. recognized by mpl.plot)
        containing valid line properties. In addition it understands
        the keyword `type` that can take the following values:
        {'line', 'text'}
    """

    def __init__(self):
        # Data attributes
        self.data = None
        self.axes_manager = None
        self.ax = None
        self.auto_update = True

        # Properties
        self.marker = None
        self._marker_properties = {}

    @property
    def marker_properties(self):
        return self._marker_properties

    @marker_properties.setter
    def marker_properties(self, kwargs):

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
        """
        Set the line_properties attribute using keyword
        arguments.
        """
        self.marker_properties = kwargs

    def set_data(self, x1=None, y1=None,
                 x2=None, y2=None, text=None, size=None):
        """
        Set data to the structured array. Each field of data should have
        the same dimensions than the nagivation axes. The other fields are
        overwritten.
        """
        self.data = np.array((np.array(x1), np.array(y1),
                              np.array(x2), np.array(y2),
                              np.array(text), np.array(size)),
                             dtype=[('x1', object), ('y1', object),
                                    ('x2', object), ('y2', object),
                                    ('text', object), ('size', object)])
        self._is_marker_static()

    def add_data(self, **kwargs):
        """
        Add data to the structured array. Each field of data should have
        the same dimensions than the nagivation axes. The other fields are
        not changed.
        """
        if self.data is None:
            self.set_data(**kwargs)
        else:
            for key in kwargs.keys():
                self.data[key][()] = np.array(kwargs[key])
        self._is_marker_static()

    def _is_marker_static(self):
        if np.alltrue([hasattr(self.data[key].item()[()], "__iter__") is False
                       for key in self.data.dtype.names]):
            self.auto_update = False
        else:
            self.auto_update = True

    def get_data_position(self, ind):
        data = self.data
        if data[ind].item()[()] is None:
            return None
        elif hasattr(data[ind].item()[()], "__iter__") and \
                self.auto_update:
            indices = self.axes_manager.indices[::-1]
            return data[ind].item()[indices]
        else:
            return data[ind].item()[()]

    def close(self):
        try:
            self.marker.remove()
            # m.ax.figure.canvas.draw()
            self.ax.hspy_fig._draw_animated()
        except:
            pass
