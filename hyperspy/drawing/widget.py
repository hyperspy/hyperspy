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
from matplotlib.backend_bases import MouseEvent
import numpy as np

from hyperspy.drawing.utils import on_figure_window_close
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

    def __init__(self, axes_manager=None, color='red', alpha=1.0, **kwargs):
        self.axes_manager = axes_manager
        self._axes = list()
        self.ax = None
        self.picked = False
        self.selected = False
        self._selected_artist = None
        self._size = 1.
        self._pos = np.array([0.])
        self.__is_on = True
        self.background = None
        self.patch = []
        self.color = color
        self.alpha = alpha
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
            """, arguments=['obj'])
        self.events.closed = Event(doc="""
            Event that triggers when the widget closed.

            The event triggers after the widget has already been closed.

            Arguments:
            ----------
                widget:
                    The widget that closed
            """, arguments=['obj'])
        self._navigating = False
        super(WidgetBase, self).__init__(**kwargs)

    def _get_axes(self):
        return self._axes

    def _set_axes(self, axes):
        if axes is None:
            self._axes = list()
        else:
            self._axes = axes

    axes = property(lambda s: s._get_axes(),
                    lambda s, v: s._set_axes(v))

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
        did_something = False
        if value is not self.is_on() and self.ax is not None:
            did_something = True
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
                self.disconnect()
        if hasattr(super(WidgetBase, self), 'set_on'):
            super(WidgetBase, self).set_on(value)
        if did_something:
            self.draw_patch()
            if value is False:
                self.ax = None
        self.__is_on = value

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, color):
        self._color = color
        for p in self.patch:
            p.set_color(self._color)

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = alpha
        for p in self.patch:
            p.set_alpha(self._alpha)

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
        self.blit = hasattr(ax, 'hspy_fig') and ax.figure.canvas.supports_blit
        for p in self.patch:
            ax.add_artist(p)
            p.set_animated(self.blit)
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
            self.disconnect()
        self.ax = ax
        canvas = ax.figure.canvas
        if self.is_on() is True:
            self._add_patch_to(ax)
            self.connect(ax)
            canvas.draw_idle()
            self.select()

    def select(self):
        """
        Cause this widget to be the selected widget in its MPL axes. This
        assumes that the widget has its patch added to the MPL axes.
        """
        if not self.patch or not self.is_on() or not self.ax:
            return

        canvas = self.ax.figure.canvas
        # Simulate a pick event
        x, y = self.patch[0].get_transform().transform_point((0, 0))
        mouseevent = MouseEvent('pick_event', canvas, x, y)
        # when the widget is added programatically, mouseevent can be "empty"
        if mouseevent.button:
            canvas.pick_event(mouseevent, self.patch[0])
        self.picked = False

    def connect(self, ax):
        """Connect to the matplotlib Axes' events.
        """
        on_figure_window_close(ax.figure, self.close)
        if self._navigating:
            self.connect_navigate()

    def connect_navigate(self):
        """Connect to the axes_manager such that changes in the widget or in
        the axes_manager are reflected in the other.
        """
        if self._navigating:
            self.disconnect_navigate()
        self.axes_manager.events.indices_changed.connect(
            self._on_navigate, {'obj': 'axes_manager'})
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

    def disconnect(self):
        """Disconnect from all events (both matplotlib and navigation).
        """
        for cid in self.cids:
            try:
                self.ax.figure.canvas.mpl_disconnect(cid)
            except BaseException:
                pass
        if self._navigating:
            self.disconnect_navigate()

    def close(self, window=None):
        """Set the on state to off (removes patch and disconnects), and trigger
        events.closed.
        """
        self.set_on(False)
        self.events.closed.trigger(obj=self)

    def draw_patch(self, *args):
        """Update the patch drawing.
        """
        try:
            if self.blit and hasattr(self.ax, 'hspy_fig'):
                self.ax.hspy_fig._update_animated()
            elif self.ax.figure is not None:
                self.ax.figure.canvas.draw_idle()
        except AttributeError:
            pass  # When figure is None, typically when closing

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

    def __str__(self):
        return "{} with id {}".format(self.__class__.__name__, id(self))


class DraggableWidgetBase(WidgetBase):

    """Adds the `position` and `indices` properties, and adds a framework for
    letting the user drag the patch around. Also adds the `moved` event.

    The default behavior is that `position` snaps to the values corresponding
    to the values of the axes grid (i.e. no subpixel values). This behavior
    can be controlled by the property `snap_position`.

    Any inheritors must override these methods:
        _onmousemove(self, event)
        _update_patch_position(self)
        _set_patch(self)
    """

    def __init__(self, axes_manager, **kwargs):
        super(DraggableWidgetBase, self).__init__(axes_manager, **kwargs)
        self.events.moved = Event(doc="""
            Event that triggers when the widget was moved.

            The event triggers after the internal state of the widget has been
            updated. This event does not differentiate on how the position of
            the widget was changed, so it is the responsibility of the user
            to suppress events as neccessary to avoid closed loops etc.

            Arguments:
            ----------
                obj:
                    The widget that was moved.
            """, arguments=['obj'])
        self._snap_position = True

        # Set default axes
        if self.axes_manager is not None:
            if self.axes_manager.navigation_dimension > 0:
                self.axes = self.axes_manager.navigation_axes[0:1]
            else:
                self.axes = self.axes_manager.signal_axes[0:1]
        else:
            self._pos = np.array([0.])

    def _set_axes(self, axes):
        super(DraggableWidgetBase, self)._set_axes(axes)
        if self.axes:
            self._pos = np.array([ax.low_value for ax in self.axes])

    def _get_indices(self):
        """Returns a tuple with the position (indices).
        """
        idx = []
        for i in range(len(self.axes)):
            idx.append(self.axes[i].value2index(self._pos[i]))
        return tuple(idx)

    def _set_indices(self, value):
        """Sets the position of the widget (by indices). The dimensions should
        correspond to that of the 'axes' attribute. Calls _pos_changed if the
        value has changed, which is then responsible for triggering any
        relevant events.
        """
        if np.ndim(value) == 0 and len(self.axes) == 1:
            self.position = [self.axes[0].index2value(value)]
        elif len(self.axes) != len(value):
            raise ValueError()
        else:
            p = []
            for i in range(len(self.axes)):
                p.append(self.axes[i].index2value(value[i]))
            self.position = p

    indices = property(lambda s: s._get_indices(),
                       lambda s, v: s._set_indices(v))

    def _pos_changed(self):
        """Call when the position of the widget has changed. It triggers the
        relevant events, and updates the patch position.
        """
        if self._navigating:
            with self.axes_manager.events.indices_changed.suppress_callback(
                    self._on_navigate):
                for i in range(len(self.axes)):
                    self.axes[i].value = self._pos[i]
        self.events.moved.trigger(self)
        self.events.changed.trigger(self)
        self._update_patch_position()

    def _validate_pos(self, pos):
        """Validates the passed position. Depending on the position and the
        implementation, this can either fire a ValueError, or return a modified
        position that has valid values. Or simply return the unmodified
        position if everything is ok.

        This default implementation bounds the position within the axes limits.
        """
        if len(pos) != len(self.axes):
            raise ValueError()
        pos = np.maximum(pos, [ax.low_value for ax in self.axes])
        pos = np.minimum(pos, [ax.high_value for ax in self.axes])
        if self.snap_position:
            pos = self._do_snap_position(pos)
        return pos

    def _get_position(self):
        """Providies the position of the widget (by values) in a tuple.
        """
        return tuple(
            self._pos.tolist())  # Don't pass reference, and make it clear

    def _set_position(self, position):
        """Sets the position of the widget (by values). The dimensions should
        correspond to that of the 'axes' attribute. Calls _pos_changed if the
        value has changed, which is then responsible for triggering any
        relevant events.
        """
        position = self._validate_pos(position)
        if np.any(self._pos != position):
            self._pos = np.array(position)
            self._pos_changed()

    position = property(lambda s: s._get_position(),
                        lambda s, v: s._set_position(v))

    def _do_snap_position(self, value=None):
        """Snaps position to axes grid. Returns snapped value. If value is
        passed as an argument, the internal state is left untouched, if not
        the position attribute is updated to the snapped value.
        """
        value = np.array(value) if value is not None else self._pos
        for i, ax in enumerate(self.axes):
            value[i] = ax.index2value(ax.value2index(value[i]))
        return value

    def _set_snap_position(self, value):
        self._snap_position = value
        if value:
            snap_value = self._do_snap_position(self._pos)
            if np.any(self._pos != snap_value):
                self._pos = snap_value
                self._pos_changed()

    snap_position = property(lambda s: s._snap_position,
                             lambda s, v: s._set_snap_position(v))

    def connect(self, ax):
        super(DraggableWidgetBase, self).connect(ax)
        canvas = ax.figure.canvas
        self.cids.append(
            canvas.mpl_connect('motion_notify_event', self._onmousemove))
        self.cids.append(canvas.mpl_connect('pick_event', self.onpick))
        self.cids.append(canvas.mpl_connect(
            'button_release_event', self.button_release))

    def _on_navigate(self, axes_manager):
        if axes_manager is self.axes_manager:
            p = self._pos.tolist()
            for i, a in enumerate(self.axes):
                p[i] = a.value
            self.position = p    # Use property to trigger events

    def onpick(self, event):
        # Callback for MPL pick event
        self.picked = (event.artist in self.patch)
        self._selected_artist = event.artist
        if hasattr(super(DraggableWidgetBase, self), 'onpick'):
            super(DraggableWidgetBase, self).onpick(event)
        self.selected = self.picked

    def _onmousemove(self, event):
        """Callback for mouse movement. For dragging, the implementor would
        normally check that the widget is picked, and that the event.inaxes
        Axes equals self.ax.
        """
        # This method must be provided by the subclass
        pass

    def _update_patch_position(self):
        """Updates the position of the patch on the plot.
        """
        # This method must be provided by the subclass
        pass

    def _update_patch_geometry(self):
        """Updates all geometry of the patch on the plot.
        """
        self._update_patch_position()

    def button_release(self, event):
        """whenever a mouse button is released"""
        if event.button != 1:
            return
        if self.picked is True:
            self.picked = False


class Widget1DBase(DraggableWidgetBase):

    """A base class for 1D widgets.

    It sets the right dimensions for size and
    position, adds the 'border_thickness' attribute and initalizes the 'axes'
    attribute to the first two navigation axes if possible, if not, the two
    first signal_axes are used. Other than that it mainly supplies common
    utility functions for inheritors, and implements required functions for
    ResizableDraggableWidgetBase.

    The implementation for ResizableDraggableWidgetBase methods all assume that
    a Rectangle patch will be used, centered on position. If not, the
    inheriting class will have to override those as applicable.
    """

    def _set_position(self, position):
        try:
            len(position)
        except TypeError:
            position = (position,)
        super(Widget1DBase, self)._set_position(position)

    def _validate_pos(self, pos):
        pos = np.maximum(pos, self.axes[0].low_value)
        pos = np.minimum(pos, self.axes[0].high_value)
        return super(Widget1DBase, self)._validate_pos(pos)


class ResizableDraggableWidgetBase(DraggableWidgetBase):

    """Adds the `size` property and get_size_in_axes method, and adds a
    framework for letting the user resize the patch, including resizing by
    key strokes ('+', '-'). Also adds the 'resized' event.

    Utility functions for resizing are implemented by `increase_size` and
    `decrease_size`, which will in-/decrement the size by 1. Other utility
    functions include `get_centre` and `get_centre_indices` which returns the
    center position, and the internal _apply_changes which helps make sure that
    only one 'changed' event is fired for a combined move and resize.

    Any inheritors must override these methods:
        _update_patch_position(self)
        _update_patch_size(self)
        _update_patch_geometry(self)
        _set_patch(self)
    """

    def __init__(self, axes_manager, **kwargs):
        super(ResizableDraggableWidgetBase, self).__init__(
            axes_manager, **kwargs)
        if not self.axes:
            self._size = np.array([1])
        self.size_step = 1      # = one step in index space
        self._snap_size = True
        self.events.resized = Event(doc="""
            Event that triggers when the widget was resized.

            The event triggers after the internal state of the widget has been
            updated. This event does not differentiate on how the size of
            the widget was changed, so it is the responsibility of the user
            to suppress events as neccessary to avoid closed loops etc.

            Arguments:
            ----------
                obj:
                    The widget that was resized.
            """, arguments=['obj'])
        self.no_events_while_dragging = False
        self._drag_store = None

    def _set_axes(self, axes):
        super(ResizableDraggableWidgetBase, self)._set_axes(axes)
        if self.axes:
            self._size = np.array([ax.scale for ax in self.axes])

    def _get_size(self):
        """Getter for 'size' property. Returns the size as a tuple (to prevent
        unintended in-place changes).
        """
        return tuple(self._size.tolist())

    def _set_size(self, value):
        """Setter for the 'size' property.

        Calls _size_changed to handle size change, if the value has changed.

        """
        value = np.minimum(value, [ax.size * ax.scale for ax in self.axes])
        value = np.maximum(value,
                           self.size_step * [ax.scale for ax in self.axes])
        if self.snap_size:
            value = self._do_snap_size(value)
        if np.any(self._size != value):
            self._size = value
            self._size_changed()

    size = property(lambda s: s._get_size(), lambda s, v: s._set_size(v))

    def _do_snap_size(self, value=None):
        value = np.array(value) if value is not None else self._size
        for i, ax in enumerate(self.axes):
            value[i] = round(value[i] / ax.scale) * ax.scale
        return value

    def _set_snap_size(self, value):
        self._snap_size = value
        if value:
            snap_value = self._do_snap_size(self._size)
            if np.any(self._size != snap_value):
                self._size = snap_value
                self._size_changed()

    snap_size = property(lambda s: s._snap_size,
                         lambda s, v: s._set_snap_size(v))

    def _set_snap_all(self, value):
        # Snap position first, as snapped size can depend on position.
        self.snap_position = value
        self.snap_size = value

    snap_all = property(lambda s: s.snap_size and s.snap_position,
                        lambda s, v: s._set_snap_all(v))

    def increase_size(self):
        """Increment all sizes by 1. Applied via 'size' property.
        """
        self.size = np.array(self.size) + \
            self.size_step * np.array([a.scale for a in self.axes])

    def decrease_size(self):
        """Decrement all sizes by 1. Applied via 'size' property.
        """
        self.size = np.array(self.size) - \
            self.size_step * np.array([a.scale for a in self.axes])

    def _size_changed(self):
        """Triggers resize and changed events, and updates the patch.
        """
        self.events.resized.trigger(self)
        self.events.changed.trigger(self)
        self._update_patch_size()

    def get_size_in_indices(self):
        """Gets the size property converted to the index space (via 'axes'
        attribute).
        """
        s = list()
        for i in range(len(self.axes)):
            s.append(int(round(self._size[i] / self.axes[i].scale)))
        return np.array(s)

    def set_size_in_indices(self, value):
        """Sets the size property converted to the index space (via 'axes'
        attribute).
        """
        s = list()
        for i in range(len(self.axes)):
            s.append(int(round(value[i] * self.axes[i].scale)))
        self.size = s   # Use property to get full processing

    def get_centre(self):
        """Get's the center indices. The default implementation is simply the
        position + half the size in axes space, which should work for any
        symmetric widget, but more advanced widgets will need to decide whether
        to return the center of gravity or the geometrical center of the
        bounds.
        """
        return self._pos + self._size() / 2.0

    def get_centre_index(self):
        """Get's the center position (in index space). The default
        implementation is simply the indices + half the size, which should
        work for any symmetric widget, but more advanced widgets will need to
        decide whether to return the center of gravity or the geometrical
        center of the bounds.
        """
        return self.indices + self.get_size_in_indices() / 2.0

    def _update_patch_size(self):
        """Updates the size of the patch on the plot.
        """
        # This method must be provided by the subclass
        pass

    def _update_patch_geometry(self):
        """Updates all geometry of the patch on the plot.
        """
        # This method must be provided by the subclass
        pass

    def on_key_press(self, event):
        if event.key == "+":
            self.increase_size()
        if event.key == "-":
            self.decrease_size()

    def connect(self, ax):
        super(ResizableDraggableWidgetBase, self).connect(ax)
        canvas = ax.figure.canvas
        self.cids.append(canvas.mpl_connect('key_press_event',
                                            self.on_key_press))

    def onpick(self, event):
        if hasattr(super(ResizableDraggableWidgetBase, self), 'onpick'):
            super(ResizableDraggableWidgetBase, self).onpick(event)
        if self.picked:
            self._drag_store = (self.position, self.size)

    def _apply_changes(self, old_size, old_position):
        """Evalutes whether the widget has been moved/resized, and triggers
        the correct events and updates the patch geometry. This function has
        the advantage that the geometry is updated only once, preventing
        flickering, and the 'changed' event only fires once.
        """
        moved = self.position != old_position
        resized = self.size != old_size
        if moved:
            if self._navigating:
                e = self.axes_manager.events.indices_changed
                with e.suppress_callback(self._on_navigate):
                    for i in range(len(self.axes)):
                        self.axes[i].index = self.indices[i]
        if moved or resized:
            # Update patch first
            if moved and resized:
                self._update_patch_geometry()
            elif moved:
                self._update_patch_position()
            else:
                self._update_patch_size()
            # Then fire events
            if not self.no_events_while_dragging or not self.picked:
                if moved:
                    self.events.moved.trigger(self)
                if resized:
                    self.events.resized.trigger(self)
                self.events.changed.trigger(self)

    def button_release(self, event):
        """whenever a mouse button is released"""
        picked = self.picked
        super(ResizableDraggableWidgetBase, self).button_release(event)
        if event.button != 1:
            return
        if picked and self.picked is False:
            if self.no_events_while_dragging and self._drag_store:
                self._apply_changes(*self._drag_store)


class Widget2DBase(ResizableDraggableWidgetBase):

    """A base class for 2D widgets. It sets the right dimensions for size and
    position, adds the 'border_thickness' attribute and initalizes the 'axes'
    attribute to the first two navigation axes if possible, if not, the two
    first signal_axes are used. Other than that it mainly supplies common
    utility functions for inheritors, and implements required functions for
    ResizableDraggableWidgetBase.

    The implementation for ResizableDraggableWidgetBase methods all assume that
    a Rectangle patch will be used, centered on position. If not, the
    inheriting class will have to override those as applicable.
    """

    def __init__(self, axes_manager, **kwargs):
        super(Widget2DBase, self).__init__(axes_manager, **kwargs)
        self.border_thickness = 2

        # Set default axes
        if self.axes_manager is not None:
            if self.axes_manager.navigation_dimension > 1:
                self.axes = self.axes_manager.navigation_axes[0:2]
            elif self.axes_manager.signal_dimension > 1:
                self.axes = self.axes_manager.signal_axes[0:2]
            elif len(self.axes_manager.shape) > 1:
                self.axes = (self.axes_manager.signal_axes +
                             self.axes_manager.navigation_axes)
            else:
                raise ValueError("2D widget needs at least two axes!")
        else:
            self._pos = np.array([0, 0])
            self._size = np.array([1, 1])

    def _get_patch_xy(self):
        """Returns the xy position of the widget. In this default
        implementation, the widget is centered on the position.
        """
        return self._pos - self._size / 2.

    def _get_patch_bounds(self):
        """Returns the bounds of the patch in the form of a tuple in the order
        left, top, width, height. In matplotlib, 'bottom' is used instead of
        'top' as the naming assumes an upwards pointing y-axis, meaning the
        lowest value corresponds to bottom. However, our widgets will normally
        only go on images (which has an inverted y-axis in MPL by default), so
        we define the lowest value to be termed 'top'.
        """
        xy = self._get_patch_xy()
        xs, ys = self.size
        return (xy[0], xy[1], xs, ys)        # x,y,w,h

    def _update_patch_position(self):
        if self.is_on() and self.patch:
            self.patch[0].set_xy(self._get_patch_xy())
            self.draw_patch()

    def _update_patch_size(self):
        self._update_patch_geometry()

    def _update_patch_geometry(self):
        if self.is_on() and self.patch:
            self.patch[0].set_bounds(*self._get_patch_bounds())
            self.draw_patch()


class ResizersMixin(object):
    """
    Widget mix-in for adding resizing manipulation handles.

    The default handles are green boxes displayed on the outside corners of the
    boundaries. By default, the handles are only displayed when the widget is
    selected (`picked` in matplotlib terminology).

    Attributes:
    -----------
        resizers : {bool}
            Property that determines whether the resizer handles should be used
        resize_color : {matplotlib color}
            The color of the resize handles.
        resize_pixel_size : {tuple | None}
            Size of the resize handles in screen pixels. If None, it is set
            equal to the size of one 'data-pixel' (image pixel size).
        resizer_picked : {False | int}
            Inidcates which, if any, resizer was selected the last time the
            widget was picked. `False` if another patch was picked, or the
            index of the resizer handle that was picked.

    """

    def __init__(self, resizers=True, **kwargs):
        super(ResizersMixin, self).__init__(**kwargs)
        self.resizer_picked = False
        self.pick_offset = (0, 0)
        self.resize_color = 'lime'
        self.resize_pixel_size = (5, 5)  # Set to None to make one data pixel
        self._resizers = resizers
        self._resizer_handles = []
        self._resizers_on = False
        # The `_resizers_on` attribute reflects whether handles are actually on
        # as compared to `_resizers` which is whether the user wants them on.
        # The difference is e.g. for turning on and off handles when the
        # widget is selected/deselected.

    @property
    def resizers(self):
        return self._resizers

    @resizers.setter
    def resizers(self, value):
        if self._resizers != value:
            self._resizers = value
            self._set_resizers(value, self.ax)

    def _update_resizers(self):
        """Update resizer handles' patch geometry.
        """
        pos = self._get_resizer_pos()
        rsize = self._get_resizer_size()
        for i, r in enumerate(self._resizer_handles):
            r.set_xy(pos[i])
            r.set_width(rsize[0])
            r.set_height(rsize[1])

    def _set_resizers(self, value, ax):
        """Turns the resizers on/off, in much the same way that _set_patch
        works.
        """
        if ax is not None:
            if value:
                for r in self._resizer_handles:
                    ax.add_artist(r)
                    r.set_animated(self.blit)
            else:
                for container in [
                        ax.patches,
                        ax.lines,
                        ax.artists,
                        ax.texts]:
                    for r in self._resizer_handles:
                        if r in container:
                            container.remove(r)
            self._resizers_on = value
            self.draw_patch()

    def _get_resizer_size(self):
        """Gets the size of the resizer handles in axes coordinates. If
        'resize_pixel_size' is None, a size of one pixel will be used.
        """
        invtrans = self.ax.transData.inverted()
        if self.resize_pixel_size is None:
            rsize = [ax.scale for ax in self.axes]
        else:
            rsize = np.abs(invtrans.transform(self.resize_pixel_size) -
                           invtrans.transform((0, 0)))
        return rsize

    def _get_resizer_offset(self):
        """Utility for getting the distance from the boundary box to the
        center of the resize handles.
        """
        invtrans = self.ax.transData.inverted()
        border = self.border_thickness
        # Transform the border thickness into data values
        dl = np.abs(invtrans.transform((border, border)) -
                    invtrans.transform((0, 0))) / 2
        rsize = self._get_resizer_size()
        return rsize / 2 + dl

    def _get_resizer_pos(self):
        """Get the positions of the resizer handles.
        """
        invtrans = self.ax.transData.inverted()
        border = self.border_thickness
        # Transform the border thickness into data values
        dl = np.abs(invtrans.transform((border, border)) -
                    invtrans.transform((0, 0))) / 2
        rsize = self._get_resizer_size()
        xs, ys = self._size

        positions = []
        rp = np.array(self._get_patch_xy())
        p = rp - rsize + dl                         # Top left
        positions.append(p)
        p = rp + (xs - dl[0], -rsize[1] + dl[1])    # Top right
        positions.append(p)
        p = rp + (-rsize[0] + dl[0], ys - dl[1])    # Bottom left
        positions.append(p)
        p = rp + (xs - dl[0], ys - dl[1])           # Bottom right
        positions.append(p)
        return positions

    def _set_patch(self):
        """Creates the resizer handles, irregardless of whether they will be
        used or not.
        """
        if hasattr(super(ResizersMixin, self), '_set_patch'):
            super(ResizersMixin, self)._set_patch()

        if self._resizer_handles:
            self._set_resizers(False, self.ax)
        self._resizer_handles = []
        rsize = self._get_resizer_size()
        pos = self._get_resizer_pos()
        for i in range(len(pos)):
            r = plt.Rectangle(pos[i], rsize[0], rsize[1], animated=self.blit,
                              fill=True, lw=0, fc=self.resize_color,
                              picker=True,)
            self._resizer_handles.append(r)

    def set_on(self, value):
        """Turns on/off resizers whet widget is turned on/off.
        """
        if self.resizers and value != self._resizers_on:
            self._set_resizers(value, self.ax)
        if hasattr(super(ResizersMixin, self), 'set_on'):
            super(ResizersMixin, self).set_on(value)

    def onpick(self, event):
        """Picking of main patch is same as for widget base, but this also
        handles picking of the resize handles. If a resize handles is picked,
        `picked` is set to `True`, and `resizer_picked` is set to an integer
        indicating which handle was picked (0-3 for top left, top right, bottom
        left, bottom right). It is set to `False` if another widget was picked.

        If the main patch is picked, the offset from the picked pixel to the
        `position` is stored in `pick_offset`. This can be used in e.g.
        `_onmousemove` to ease dragging code (prevent widget center/corner
        snapping to mouse).
        """
        if event.artist in self._resizer_handles:
            corner = self._resizer_handles.index(event.artist)
            self.resizer_picked = corner
            self.picked = True
        elif self.picked:
            if self.resizers and not self._resizers_on:
                self._set_resizers(True, self.ax)
            x = event.mouseevent.xdata
            y = event.mouseevent.ydata
            self.pick_offset = (x - self._pos[0], y - self._pos[1])
            self.resizer_picked = False
        else:
            self._set_resizers(False, self.ax)
        if hasattr(super(ResizersMixin, self), 'onpick'):
            super(ResizersMixin, self).onpick(event)

    def _add_patch_to(self, ax):
        """Same as widget base, but also adds resizers if 'resizers' property
        is True.
        """
        if self.resizers:
            self._set_resizers(True, ax)
        if hasattr(super(ResizersMixin, self), '_add_patch_to'):
            super(ResizersMixin, self)._add_patch_to(ax)
