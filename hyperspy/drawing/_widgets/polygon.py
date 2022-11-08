# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
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


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector

from hyperspy.drawing.widgets import MPLWidgetBase
from hyperspy.roi import PolygonROI


# class PolygonWidget(Widget2DBase, ResizersMixin):
class PolygonWidget(MPLWidgetBase):

    """
    """

    def __init__(self, axes_manager, mpl_ax = None, **kwargs):
        super(PolygonWidget, self).__init__(axes_manager, **kwargs)

        self._vertices = []

    def set_mpl_ax(self, ax):
        if ax is self.ax or ax is None:
            return  # Do nothing
        # Disconnect from previous axes if set
        if self.ax is not None and self.is_on:
            self.disconnect()
        self.ax = ax
        if self.is_on is True:
            self.widget = PolygonSelector(ax, onselect=self._onselect, useblit=True)

    def _onselect(self, verts):
        self._vertices = verts
        print(verts)
        bounding_box = (min(verts[0]), max(verts[0]), min(verts[1]), max(verts[1]))
        self.position = ( ( bounding_box[0] + bounding_box[1] ) / 2,
            ( bounding_box[2] + bounding_box[3] ) / 2 )

    def get_centre(self):
        """Returns the xy coordinates of the patch centre. In this implementation, the
        centre of the patch is the centre of the polygon's bounding box.
        """
        return self.position

    def get_roi(self):
        return PolygonROI(self._vertices)

    def get_mask(self, scale = None):
        if scale is None:
            scale = self.axes[0].scale, self.axes[1].scale
            print(scale)
        return PolygonROI(self._vertices).boolean_mask(x_scale=scale[0], y_scale=scale[1], xy_max=(self.axes[0].scale*self.axes[0].size, 
        self.axes[1].scale*self.axes[1].size))
