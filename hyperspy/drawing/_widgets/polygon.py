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
from matplotlib.path import Path

from hyperspy.drawing.widgets import MPLWidgetBase
from hyperspy.roi import PolygonROI


# class PolygonWidget(Widget2DBase, ResizersMixin):
class PolygonWidget(MPLWidgetBase):

    """
    """

    def __init__(self, axes_manager, mpl_ax = None,  continuous = False, **kwargs):
        super(PolygonWidget, self).__init__(axes_manager, **kwargs)

        self._vertices_list = []

        self._continuous = continuous

        self._finished = True

        self._shift_pressed = False
        self._otherfocus = False
        self._lines = []
        self._linesplot = []

    def set_mpl_ax(self, ax):
        if ax is self.ax or ax is None:
            return  # Do nothing
        # Disconnect from previous axes if set
        if self.ax is not None and self.is_on:
            self.disconnect()
        self.ax = ax

        
        if self._continuous:
            self.ax.figure.canvas.mpl_connect("button_press_event", self._pressevent)
            self.ax.figure.canvas.mpl_connect("button_release_event", self._releaseevent)
            self.ax.figure.canvas.mpl_connect("button_release_event", self._releaseevent)
        if self.is_on is True:
            self.widgets = [PolygonSelector(ax, onselect=lambda v: self._onselect(v,0), useblit=True)]
            self._finished = False


    def _onselect(self, verts, currid):
        # self._vertices_list.append(verts.copy())
        # print(verts)
        bounding_box = (min(verts[0]), max(verts[0]), min(verts[1]), max(verts[1]))
        self.position = ( ( bounding_box[0] + bounding_box[1] ) / 2,
            ( bounding_box[2] + bounding_box[3] ) / 2 )

        prevlen = len(self.widgets)
        if self._continuous and currid == prevlen - 1:
            self._finished = True

    def _pressevent(self, event):

        # if  hasattr(event, "key") and event.key and "shift" in event.key:
        #     self._shift_pressed = True


        if not self._continuous:
            return
        
        if not event.inaxes.axes or event.inaxes.axes!=self.ax.axes: 
            return


        # x,y = event.xdata, event.ydata
        # if self._shift_pressed:
        #     tocheck = self.widgets
        #     if not self._finished:
        #         tocheck = self.widgets[:-1]

            
        #     for widg in tocheck:
        #         if Path(widg.verts).contains_point((x,y)):
        #             self.widgets[-1].active = False
        #             widg.active = True
        #             self._otherfocus = True
        #             return
        #     return

            

        # for widg in self.widgets:
        #     grab_range_sq = widg.vertex_select_radius**2
        #     if any((vx-x)**2 + (vy-y)**2 < grab_range_sq for vx,vy in widg.verts):
        #         return

        if self._is_on and self._finished and self.widgets[0]._selection_completed:

            self._lines.append(self.widgets[0].verts)
            
            event = self.widgets[0]._clean_event(event)
            self.widgets[0]._xs, self.widgets[0]._ys = [event.xdata], [event.ydata]
            self.widgets[0]._selection_completed = False
            self.widgets[0].set_visible(True)

            
            self._linesplot.append(self.ax.plot(*zip(*(self._lines[-1] + [self._lines[-1][0]])), animated = True))

            self._finished = False
            self.ax.figure.canvas.draw_idle() # This function has to be here, 



            # prevlen = len(self.widgets)
            # self.widgets[-1].active = False
            # self.widgets.append(PolygonSelector(self.ax, onselect=lambda v, ind=prevlen: self._onselect(v,ind), useblit=True))
            # self._finished = False
            

    def _releaseevent(self, event):
        pass
        # if hasattr(event, "key") and event.key  and "shift" in event.key:
        #     self._shift_pressed = False

        # if self._otherfocus:
        #     for w in self.widgets:
        #         w.active = False
        #     self.widgets[-1].active = True
        #     self._otherfocus = False

        
        

    def get_centre(self):
        """Returns the xy coordinates of the patch centre. In this implementation, the
        centre of the patch is the centre of the polygon's bounding box.
        """
        return self.position

    def get_roi(self):
        if self.widgets[-1]._selection_completed:
            return PolygonROI(self.widgets[-1].verts)
        elif self._lines:
            return PolygonROI(self.widgets[-1].verts)
        return PolygonROI()

    def get_mask(self, scale = None):
        if scale is None:
            scale = self.axes[0].scale, self.axes[1].scale
            print(scale)

        mask_base_vertices = []
        if self.widgets[-1]._selection_completed:
            mask_base_vertices = self.widgets[0].verts
        
        mask = PolygonROI(mask_base_vertices).boolean_mask(x_scale=scale[0], y_scale=scale[1], xy_max=(self.axes[0].scale*self.axes[0].size, 
            self.axes[1].scale*self.axes[1].size))

        for verts in self._lines:
            mask = np.logical_or(mask, PolygonROI(verts).boolean_mask(x_scale=scale[0], y_scale=scale[1], xy_max=(self.axes[0].scale*self.axes[0].size, 
                self.axes[1].scale*self.axes[1].size)))
        return mask
