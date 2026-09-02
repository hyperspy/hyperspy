# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
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

from hyperspy.drawing.backends import get_backend
from hyperspy.drawing.widget import Widget1DBase


class VerticalLineWidget(Widget1DBase):
    """A draggable, vertical line widget."""

    def _update_patch_position(self):
        if self.is_on and self.patch:
            get_backend().update_line_pointer(self.patch[0], "x", float(self._pos[0]))
            self.draw_patch()

    def _add_patch_to(self, ax):
        backend = get_backend()
        self.blit = backend.supports_blit_from_ax(ax)
        handle = backend.create_line_pointer(
            ax, "x", float(self._pos[0]), color=self.color
        )
        self._patch = [handle]
        backend.set_pointer_style(handle, animated=self.blit)
        backend.connect_widget_drag(handle, self._on_widget_drag)

    def _on_widget_drag(self, x, *args):
        """Native-widget drag callback."""
        self.position = (x,)

    def _onjumpclick(self, event):
        if event.key == "shift" and event.inaxes and self.is_pointer:
            self.position = (event.xdata,)

    def _onmousemove(self, event):
        """on mouse motion draw the cursor if picked"""
        if self.picked is True and event.inaxes:
            self.position = (event.xdata,)
