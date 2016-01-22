# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
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
import matplotlib.transforms as transforms

from hyperspy.drawing.widgets import DraggableWidgetBase


class LabelWidget(DraggableWidgetBase):

    """A draggable text widget. Adds the attributes 'string', 'text_color' and
    'bbox'. These are all arguments for matplotlib's Text artist. The default
    y-coordinate of the label is set to 0.9.
    """

    def __init__(self, axes_manager):
        super(LabelWidget, self).__init__(axes_manager)
        self.string = ''
        self._snap_position = False
        if not self.axes:
            self._pos = np.array((0, 0.9))
        self.text_color = 'black'
        self.bbox = None

    def _set_axes(self, axes):
        super(LabelWidget, self)._set_axes(axes)
        if len(self.axes) == 1:
            self._pos = np.array((self.axes[0].offset, 0.9))
        else:
            self._pos = np.array((self.axes[0].offset,
                                  self.axes[1].offset))

    def _validate_pos(self, pos):
        if len(self.axes) == 1:
            pos = np.maximum(pos, self.axes[0].low_value)
            pos = np.minimum(pos, self.axes[0].high_value)
        elif len(self.axes) > 1:
            pos = np.maximum(pos, [a.low_value for a in self.axes[0:2]])
            pos = np.minimum(pos, [a.high_value for a in self.axes[0:2]])
        else:
            raise ValueError()
        return pos

    def _update_patch_position(self):
        if self.is_on() and self.patch:
            self.patch[0].set_x(self.position[0])
            self.patch[0].set_y(self.position[1])
            self.draw_patch()

    def _set_patch(self):
        ax = self.ax
        trans = transforms.blended_transform_factory(
            ax.transData, ax.transAxes)
        self.patch = [ax.text(
            self.position[0],
            self.position[1],
            self.string,
            color=self.text_color,
            picker=5,
            transform=trans,
            horizontalalignment='left',
            bbox=self.bbox,
            animated=self.blit)]

    def _onmousemove(self, event):
        """on mouse motion draw the cursor if picked"""
        if self.picked is True and event.inaxes:
            self.position = (event.xdata, event.ydata)
