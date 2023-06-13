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

from hyperspy.drawing.widget import Widget1DBase
from hyperspy.drawing.utils import picker_kwargs
from hyperspy.defaults_parser import preferences


class HorizontalLineWidget(Widget1DBase):

    """A draggable, horizontal line widget.
    """

    def _update_patch_position(self):
        if self.is_on and self.patch:
            self.patch[0].set_ydata([self._pos[0]])
            self.draw_patch()

    def _add_patch_to(self, ax):
        """Create and add the matplotlib patches to 'ax'"""
        self.blit = hasattr(ax, 'hspy_fig') and ax.figure.canvas.supports_blit
        self._set_patch()
        for p in self.patch:
            p.set_animated(self.blit)

    def _set_patch(self):
        ax = self.ax
        kwargs = picker_kwargs(preferences.Plot.pick_tolerance)
        self._patch = [
            ax.axhline(self._pos[0], color=self.color, alpha=self.alpha, **kwargs)
            ]

    def _onmousemove(self, event):
        """on mouse motion draw the cursor if picked"""
        if self.picked is True and event.inaxes:
            self.position = (event.ydata,)
