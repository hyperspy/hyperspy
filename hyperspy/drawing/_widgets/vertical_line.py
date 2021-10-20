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

from hyperspy.drawing.widgets import Widget1DBase


class VerticalLineWidget(Widget1DBase):

    """A draggable, vertical line widget.
    """

    def _update_patch_position(self):
        if self.is_on() and self.patch:
            self.patch[0].set_xdata(self._pos[0])
            self.draw_patch()

    def _set_patch(self):
        ax = self.ax
        self.patch = [ax.axvline(self._pos[0],
                                 color=self.color,
                                 alpha=self.alpha,
                                 picker=5)]

    def _onmousemove(self, event):
        """on mouse motion draw the cursor if picked"""
        if self.picked is True and event.inaxes:
            self.position = (event.xdata,)
