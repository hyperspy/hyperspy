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

from hyperspy.defaults_parser import preferences
from hyperspy.drawing.utils import picker_kwargs
from hyperspy.drawing.widget import Widget1DBase


class VerticalLineWidget(Widget1DBase):
    """A draggable, vertical line widget."""

    def _update_patch_position(self):
        if self.is_on and self.patch:
            self.patch[0].set_xdata([self._pos[0]])
            self.draw_patch()
        native_vline = getattr(self, "_native_vline", None)
        if native_vline is not None:
            # Push updated (snapped) position back to the anyplotlib widget.
            native_vline.set(x=float(self._pos[0]))

    def _add_patch_to(self, ax):
        """Create and add the matplotlib patches to 'ax'"""
        from hyperspy.drawing.backends import get_backend

        self.blit = hasattr(ax, "hspy_fig") and get_backend().supports_blit(
            getattr(ax, "figure", None)
        )
        if not hasattr(ax, "axvline"):
            # Non-matplotlib backend: attach a native draggable vline widget.
            plot = getattr(ax, "_plot", None)
            if (
                self.is_pointer
                and plot is not None
                and hasattr(plot, "add_vline_widget")
            ):
                native_w = plot.add_vline_widget(
                    x=float(self._pos[0]), color=self.color
                )
                self._native_vline = native_w
                _self = self

                def _on_drag(event):
                    _self.position = (native_w.x,)

                native_w.add_event_handler(_on_drag, "pointer_move")
            return
        self._set_patch()
        for p in self.patch:
            p.set_animated(self.blit)

    def _set_patch(self):
        ax = self.ax
        kwargs = picker_kwargs(preferences.Plot.pick_tolerance)
        self._patch = [
            ax.axvline(self._pos[0], color=self.color, alpha=self.alpha, **kwargs)
        ]

    def _onjumpclick(self, event):
        if event.key == "shift" and event.inaxes and self.is_pointer:
            self.position = (event.xdata,)

    def _onmousemove(self, event):
        """on mouse motion draw the cursor if picked"""
        if self.picked is True and event.inaxes:
            self.position = (event.xdata,)
