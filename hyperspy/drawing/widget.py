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

from __future__ import division

import matplotlib.pyplot as plt
import numpy as np

from utils import on_figure_window_close
from hyperspy.events import Events, Event


class WidgetBase(object):

    """Base class for interactive widgets/patches. A widget creates and
    maintains one or more matplotlib patches, and manages the interaction code
    so that the user can maniuplate it on the fly.

    This base class implements functionality witch is common to all such
    widgets, mainly the code that manages the patch, axes management, and
    sets up common events ('changed' and 'closed').

    Any inherting subclasses must implement the following methods:
        _set_patch(self)
        _on_navigate(obj, name, old, new)  # Only for widgets that can navigate

    It should also make sure to fill the 'axes' attribute as early as
    possible (but after the base class init), so that it is available when
    needed.
    """

    def __init__(self, axes_manager=None, **kwargs):
        self.axes_manager = axes_manager
        self.axes = list()
        self.ax = None
        self.picked = False
        self._size = 1.
        self.color = 'red'
        self.__is_on = True
        self.background = None
        self.patch = []
        self.cids = list()
        self.blit = True
        self.events = Events()
        self.events.changed = Event(doc="""
            Event that triggers when the widget has a significant change.

            The event triggers after the internal state of the widget has been
            updated.

            Arguments:
            ----------
                widget:
                    The widget that changed
            """, arguments=['widget'])
        self.events.closed = Event(doc="""
            Event that triggers when the widget closed.

            The event triggers after the widget has already been closed.

            Arguments:
            ----------
                widget:
                    The widget that closed
            """, arguments=['widget'])
        self._navigating = False
        super(WidgetBase, self).__init__(**kwargs)

    def is_on(self):
        """Determines if the widget is set to draw if valid (turned on).
        """
        return self.__is_on

    def set_on(self, value):
        """Change the on state of the widget. If turning off, all patches will
        be removed from the matplotlib axes and the widget will disconnect from
        all events. If turning on, the patch(es) will be added to the
        matplotlib axes, and the widget will connect to its default events.
        """
        if value is not self.is_on() and self.ax is not None:
            if value is True:
                self._add_patch_to(self.ax)
                self.connect(self.ax)
            elif value is False:
                for container in [
                        self.ax.patches,
                        self.ax.lines,
                        self.ax.artists,
                        self.ax.texts]:
                    for p in self.patch:
                        if p in container:
                            container.remove(p)
                self.disconnect(self.ax)
            try:
                self.draw_patch()
            except:  # figure does not exist
                pass
            if value is False:
                self.ax = None
        if hasattr(super(WidgetBase, self), 'set_on'):
            super(WidgetBase, self).set_on(value)
        self.__is_on = value

    def _set_patch(self):
        """Create the matplotlib patch(es), and store it in self.patch
        """
        if hasattr(super(WidgetBase, self), '_set_patch'):
            super(WidgetBase, self)._set_patch()
        # Must be provided by the subclass

    def _add_patch_to(self, ax):
        """Create and add the matplotlib patches to 'ax'
        """
        self._set_patch()
        for p in self.patch:
            ax.add_artist(p)
            p.set_animated(hasattr(ax, 'hspy_fig'))
        if hasattr(super(WidgetBase, self), '_add_patch_to'):
            super(WidgetBase, self)._add_patch_to(ax)

    def set_mpl_ax(self, ax):
        """Set the matplotlib Axes that the widget will draw to. If the widget
        on state is True, it will also add the patch to the Axes, and connect
        to its default events.
        """
        if ax is self.ax:
            return  # Do nothing
        # Disconnect from previous axes if set
        if self.ax is not None and self.is_on():
            self.disconnect(self.ax)
        self.ax = ax
        canvas = ax.figure.canvas
        if self.is_on() is True:
            self._add_patch_to(ax)
            self.connect(ax)
            if self._navigating:
                self.connect_navigate()
            canvas.draw()

    def connect(self, ax):
        """Connect to the matplotlib Axes' events.
        """
        on_figure_window_close(ax.figure, self.close)

    def connect_navigate(self):
        """Connect to the axes_manager such that changes in the widget or in
        the axes_manager are reflected in the other.
        """
        if self._navigating:
            self.disconnect_navigate()
        self.axes_manager.events.indices_changed.connect(self._on_navigate)
        self._on_navigate(self.axes_manager)    # Update our position
        self._navigating = True

    def disconnect_navigate(self):
        """Disconnect a previous naivgation connection.
        """
        self.axes_manager.events.indices_changed.disconnect(self._on_navigate)
        self._navigating = False

    def _on_navigate(self, axes_manager):
        """Callback for axes_manager's change notification.
        """
        pass    # Implement in subclass!

    def disconnect(self, ax):
        """Disconnect from all events (both matplotlib and navigation).
        """
        for cid in self.cids:
            try:
                ax.figure.canvas.mpl_disconnect(cid)
            except:
                pass
        if self._navigating:
            self.disconnect_navigate()

    def close(self, window=None):
        """Set the on state to off (removes patch and disconnects), and trigger
        events.closed.
        """
        self.set_on(False)
        self.events.closed.trigger(self)

    def draw_patch(self, *args):
        """Update the patch drawing.
        """
        if hasattr(self.ax, 'hspy_fig'):
            self.ax.hspy_fig._draw_animated()
        else:
            self.ax.figure.canvas.draw_idle()

    def _v2i(self, axis, v):
        """Wrapped version of DataAxis.value2index, which bounds the index
        inbetween axis.low_index and axis.high_index+1, and does not raise a
        ValueError.
        """
        try:
            return axis.value2index(v)
        except ValueError:
            if v > axis.high_value:
                return axis.high_index + 1
            elif v < axis.low_value:
                return axis.low_index
            else:
                raise

    def _i2v(self, axis, i):
        """Wrapped version of DataAxis.index2value, which bounds the value
        inbetween axis.low_value and axis.high_value+axis.scale, and does not
        raise a ValueError.
        """
        try:
            return axis.index2value(i)
        except ValueError:
            if i > axis.high_index:
                return axis.high_value + axis.scale
            elif i < axis.low_index:
                return axis.low_value
            else:
                raise

