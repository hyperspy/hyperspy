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

from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path

from hyperspy.drawing.widgets import MPLWidgetBase
from hyperspy.roi import PolygonROI


# class PolygonWidget(Widget2DBase, ResizersMixin):
class PolygonWidget(MPLWidgetBase):

    """
    """

    def __init__(self, axes_manager, mpl_ax = None,  multiple = False, **kwargs):
        super().__init__(axes_manager, **kwargs)

        self._widget = None

        self._vertices_list = []
        self._vertices_plots = []

        self._multiple = multiple
        self._finished = True

    def set_mpl_ax(self, ax):
        if ax is self.ax or ax is None:
            return  # Do nothing
        # Disconnect from previous axes if set
        if self.ax is not None and self.is_on:
            self.disconnect()
        self.ax = ax

        
        if self._multiple:
            self.ax.figure.canvas.mpl_connect("button_press_event", self._pressevent)
            self.ax.figure.canvas.mpl_connect("button_release_event", self._releaseevent)
            self.ax.figure.canvas.mpl_connect("button_release_event", self._releaseevent)

        if self.is_on is True:
            handle_props = dict(color="blue")
            props = dict(color="blue")

            self._widget = PolygonSelector(ax, onselect=self._onselect, useblit=True,
                handle_props=handle_props, props=props)
            self._finished = False


    def _onselect(self, verts):
        bounding_box = (min(verts[0]), max(verts[0]), min(verts[1]), max(verts[1]))
        self.position = ( ( bounding_box[0] + bounding_box[1] ) / 2,
            ( bounding_box[2] + bounding_box[3] ) / 2 )

        self._finished = True

    def _pressevent(self, event):

        if hasattr(event, "key") and event.key and "shift" in event.key:
            return
        
        if not event.inaxes.axes or event.inaxes.axes!=self.ax.axes: 
            return

        x,y = event.xdata, event.ydata
        if self._finished or len(self._widget.verts) == 0:
            # If clicked within another polygon, set that polygon to active
            for i, vertices in enumerate(self._vertices_list):
                if Path(vertices).contains_point((x,y)):
                    if self._finished:
                        self._vertices_list.append(self._widget.verts)
                        closed_polygon = zip(*(self._widget.verts + [self._widget.verts[0]]))
                        self._vertices_plots.append(self.ax.plot(*closed_polygon, animated = True))
                    
                    closed_polygon = (list(c) for c in zip(*(vertices + [vertices[0]])))
                    self._widget._xs, self._widget._ys = closed_polygon
                    self._widget.set_visible(True)
                    self._widget._selection_completed = True

                    del self._vertices_list[i]
                    self._vertices_plots[i][0].remove()
                    del self._vertices_plots[i]

                    self.ax.figure.canvas.draw_idle()
                    self._widget._draw_polygon()
                    return

            
        # Do not make new polygon if within grab range of widget vertices
        grab_range_sq = self._widget.vertex_select_radius**2
        if any((vx-x)**2 + (vy-y)**2 < grab_range_sq for vx,vy in self._widget.verts):
            return

        if self._is_on and self._finished:

            self._vertices_list.append(self._widget.verts)
            
            event = self._widget._clean_event(event)
            self._widget._xs, self._widget._ys = [event.xdata], [event.ydata]
            self._widget._selection_completed = False
            self._widget.set_visible(True)

            
            self._vertices_plots.append(self.ax.plot(*zip(*(self._vertices_list[-1] + [self._vertices_list[-1][0]])), animated = True))

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
        if self._finished:
            return PolygonROI(self._vertices_list + [self._widget.verts])
        else:
            return PolygonROI(self._vertices_list)

    def get_mask(self, scale = None):
        if scale is None:
            scale = self.axes[0].scale, self.axes[1].scale
            print(scale)
        
        return self.get_roi().boolean_mask(scalex=scale[0], scaley=scale[1], xy_max=(self.axes[0].scale*self.axes[0].size, 
                self.axes[1].scale*self.axes[1].size))
