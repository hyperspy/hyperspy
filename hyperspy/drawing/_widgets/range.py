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
from matplotlib.widgets import SpanSelector
import inspect
import logging

from hyperspy.drawing.widgets import ResizableDraggableWidgetBase
from hyperspy.events import Events, Event


_logger = logging.getLogger(__name__)
# Track if we have already warned when the widget is out of range
already_warn_out_of_range = False


def in_interval(number, interval):
    if interval[0] <= number <= interval[1]:
        return True
    else:
        return False


class RangeWidget(ResizableDraggableWidgetBase):

    """RangeWidget is a span-patch based widget, which can be
    dragged and resized by mouse/keys. Basically a wrapper for
    ModifiablepanSelector so that it conforms to the common widget interface.

    For optimized changes of geometry, the class implements two methods
    'set_bounds' and 'set_ibounds', to set the geometry of the rectangle by
    value and index space coordinates, respectivly.

    Implements the internal method _validate_geometry to make sure the patch
    will always stay within bounds.
    """

    def __init__(self, axes_manager, ax=None, alpha=0.5, **kwargs):
        # Parse all kwargs for the matplotlib SpanSelector
        self._SpanSelector_kwargs = {}
        for key in inspect.signature(SpanSelector).parameters.keys():
            if key in kwargs:
                self._SpanSelector_kwargs[key] = kwargs.pop(key)
        super(RangeWidget, self).__init__(axes_manager, alpha=alpha, **kwargs)
        self.span = None

    def set_on(self, value):
        if value is not self.is_on() and self.ax is not None:
            if value is True:
                self._add_patch_to(self.ax)
                self.connect(self.ax)
            elif value is False:
                self.disconnect()
            try:
                self.ax.figure.canvas.draw_idle()
            except BaseException:  # figure does not exist
                pass
            if value is False:
                self.ax = None
        self.__is_on = value

    def _add_patch_to(self, ax):
        self.span = ModifiableSpanSelector(ax, **self._SpanSelector_kwargs)
        self.span.set_initial(self._get_range())
        self.span.bounds_check = True
        self.span.snap_position = self.snap_position
        self.span.snap_size = self.snap_size
        self.span.can_switch = True
        self.span.events.changed.connect(self._span_changed, {'obj': 'widget'})
        self.span.step_ax = self.axes[0]
        self.span.tolerance = 5
        self.patch = [self.span.rect]
        self.patch[0].set_color(self.color)
        self.patch[0].set_alpha(self.alpha)

    def _span_changed(self, widget):
        r = self._get_range()
        pr = widget.range
        if r != pr:
            dx = self.axes[0].scale
            x = pr[0] + 0.5 * dx
            w = pr[1] + 0.5 * dx - x
            old_position, old_size = self.position, self.size
            self._pos = np.array([x])
            self._size = np.array([w])
            self._validate_geometry()
            if self._pos != np.array([x]) or self._size != np.array([w]):
                self._update_patch_size()
            self._apply_changes(old_size=old_size, old_position=old_position)

    def _get_range(self):
        p = self._pos[0]
        w = self._size[0]
        offset = self.axes[0].scale
        p -= 0.5 * offset
        return (p, p + w)

    def _parse_bounds_args(self, args, kwargs):
        if len(args) == 1:
            return args[0]
        elif len(args) == 4:
            return args
        elif len(kwargs) == 1 and 'bounds' in kwargs:
            return kwargs.values()[0]
        else:
            x = kwargs.pop('x', kwargs.pop('left', self._pos[0]))
            if 'right' in kwargs:
                w = kwargs.pop('right') - x
            else:
                w = kwargs.pop('w', kwargs.pop('width', self._size[0]))
            return x, w

    def set_ibounds(self, *args, **kwargs):
        """
        Set bounds by indices. Bounds can either be specified in order left,
        bottom, width, height; or by keywords:
         * 'bounds': tuple (left, width)
         OR
         * 'x'/'left'
         * 'w'/'width', alternatively 'right'
        If specifying with keywords, any unspecified dimensions will be kept
        constant (note: width will be kept, not right).
        """

        ix, iw = self._parse_bounds_args(args, kwargs)
        x = self.axes[0].index2value(ix)
        w = self._i2v(self.axes[0], ix + iw) - x

        old_position, old_size = self.position, self.size
        self._pos = np.array([x])
        self._size = np.array([w])
        self._apply_changes(old_size=old_size, old_position=old_position)

    def set_bounds(self, *args, **kwargs):
        """
        Set bounds by values. Bounds can either be specified in order left,
        bottom, width, height; or by keywords:
         * 'bounds': tuple (left, width)
         OR
         * 'x'/'left'
         * 'w'/'width', alternatively 'right' (x+w)
        If specifying with keywords, any unspecified dimensions will be kept
        constant (note: width will be kept, not right).
        """
        global already_warn_out_of_range

        def warn(obj, parameter, value):
            global already_warn_out_of_range
            if not already_warn_out_of_range:
                _logger.info('{}: {} is out of range. It is therefore set '
                             'to the value of {}'.format(obj, parameter, value))
                already_warn_out_of_range = True

        x, w = self._parse_bounds_args(args, kwargs)
        l0, h0 = self.axes[0].low_value, self.axes[0].high_value
        scale = self.axes[0].scale

        in_range = 0
        if x < l0:
            x = l0
            warn(self, '`x` or `left`', x)
        elif h0 <= x:
            x = h0 - scale
            warn(self, '`x` or `left`', x)
        else:
            in_range += 1
        if w < scale:
            w = scale
            warn(self, '`width` or `right`', w)
        elif not (l0 + scale <= x + w <= h0 + scale):
            if self.size != np.array([w]):  # resize
                w = h0 + scale - self.position[0]
                warn(self, '`width` or `right`', w)
            if self.position != np.array([x]):  # moved
                x = h0 + scale - self.size[0]
                warn(self, '`x` or `left`', x)
        else:
            in_range += 1

        # if we are in range again, reset `already_warn_out_of_range` to False
        if in_range == 2 and already_warn_out_of_range:
            _logger.info('{} back in range.'.format(self.__class__.__name__))
            already_warn_out_of_range = False

        old_position, old_size = self.position, self.size
        self._pos = np.array([x])
        self._size = np.array([w])
        self._apply_changes(old_size=old_size, old_position=old_position)

    def _update_patch_position(self):
        self._update_patch_geometry()

    def _update_patch_size(self):
        self._update_patch_geometry()

    def _update_patch_geometry(self):
        if self.is_on() and self.span is not None:
            self.span.range = self._get_range()

    def disconnect(self):
        super(RangeWidget, self).disconnect()
        if self.span:
            self.span.turn_off()
            self.span = None

    def _set_snap_position(self, value):
        super(RangeWidget, self)._set_snap_position(value)
        self.span.snap_position = value
        self._update_patch_geometry()

    def _set_snap_size(self, value):
        super(RangeWidget, self)._set_snap_size(value)
        self.span.snap_size = value
        self._update_patch_size()

    def _validate_geometry(self, x1=None):
        """Make sure the entire patch always stays within bounds. First the
        position (either from position property or from x1 argument), is
        limited within the bounds. Then, if the right edge are out of
        bounds, the position is changed so that they will be at the limit.

        The modified geometry is stored, but no change checks are performed.
        Call _apply_changes after this in order to process any changes (the
        size might change if it is set larger than the bounds size).
        """
        xaxis = self.axes[0]

        # Make sure widget size is not larger than axes
        self._size[0] = min(self._size[0], xaxis.size * xaxis.scale)

        # Make sure x1 is within bounds
        if x1 is None:
            x1 = self._pos[0]  # Get it if not supplied
        if x1 < xaxis.low_value:
            x1 = xaxis.low_value
        elif x1 > xaxis.high_value:
            x1 = xaxis.high_value

        # Make sure x2 is with upper bound.
        # If not, keep dims, and change x1!
        x2 = x1 + self._size[0]
        if x2 > xaxis.high_value + xaxis.scale:
            x2 = xaxis.high_value + xaxis.scale
            x1 = x2 - self._size[0]

        self._pos = np.array([x1])
        # Apply snaps if appropriate
        if self.snap_position:
            self._do_snap_position()
        if self.snap_size:
            self._do_snap_size()


class ModifiableSpanSelector(SpanSelector):

    def __init__(self, ax, **kwargs):
        onselect = kwargs.pop('onselect', self.dummy)
        direction = kwargs.pop('direction', 'horizontal')
        useblit = kwargs.pop('useblit', ax.figure.canvas.supports_blit)
        SpanSelector.__init__(self, ax, onselect, direction=direction,
                              useblit=useblit, span_stays=False, **kwargs)
        # The tolerance in points to pick the rectangle sizes
        self.tolerance = 1
        self.on_move_cid = None
        self._range = None
        self.step_ax = None
        self.bounds_check = False
        self.buttonDown = False
        self.snap_size = False
        self.snap_position = False
        self.events = Events()
        self.events.changed = Event(doc="""
            Event that triggers when the widget was changed.

            Arguments:
            ----------
                obj:
                    The widget that changed
            """, arguments=['obj'])
        self.events.moved = Event(doc="""
            Event that triggers when the widget was moved.

            Arguments:
            ----------
                obj:
                    The widget that changed
            """, arguments=['obj'])
        self.events.resized = Event(doc="""
            Event that triggers when the widget was resized.

            Arguments:
            ----------
                obj:
                    The widget that changed
            """, arguments=['obj'])
        self.can_switch = False

    def dummy(self, *args, **kwargs):
        pass

    def _get_range(self):
        self.update_range()
        return self._range

    def _set_range(self, value):
        self.update_range()
        if self._range != value:
            resized = (
                self._range[1] -
                self._range[0]) != (
                value[1] -
                value[0])
            moved = self._range[0] != value[0]
            self._range = value
            if moved:
                self._set_span_x(value[0])
                self.events.moved.trigger(self)
            if resized:
                self._set_span_width(value[1] - value[0])
                self.events.resized.trigger(self)
            if moved or resized:
                self.draw_patch()
                self.events.changed.trigger(self)

    range = property(_get_range, _set_range)

    def _set_span_x(self, value):
        if self.direction == 'horizontal':
            self.rect.set_x(value)
        else:
            self.rect.set_y(value)

    def _set_span_width(self, value):
        if self.direction == 'horizontal':
            self.rect.set_width(value)
        else:
            self.rect.set_height(value)

    def _get_span_x(self):
        if self.direction == 'horizontal':
            return self.rect.get_x()
        else:
            return self.rect.get_y()

    def _get_span_width(self):
        if self.direction == 'horizontal':
            return self.rect.get_width()
        else:
            return self.rect.get_height()

    def _get_mouse_position(self, event):
        if self.direction == 'horizontal':
            return event.xdata
        else:
            return event.ydata

    def set_initial(self, initial_range=None):
        """
        Remove selection events, set the spanner, and go to modify mode.
        """
        if initial_range is not None:
            self.range = initial_range

        self.disconnect_events()
        # And connect to the new ones
        self.connect_event('button_press_event', self.mm_on_press)
        self.connect_event('button_release_event', self.mm_on_release)
        self.connect_event('draw_event', self.update_background)

        self.rect.set_visible(True)
        self.rect.contains = self.contains

    def update(self, *args):
        # Override the SpanSelector `update` method to blit properly all
        # artirts before we go to "modify mode" in `set_initial`.
        self.draw_patch()

    def draw_patch(self, *args):
        """Update the patch drawing.
        """
        try:
            if self.useblit and hasattr(self.ax, 'hspy_fig'):
                self.ax.hspy_fig._update_animated()
            elif self.ax.figure is not None:
                self.ax.figure.canvas.draw_idle()
        except AttributeError:
            pass  # When figure is None, typically when closing

    def contains(self, mouseevent):
        x, y = self.rect.get_transform().inverted().transform_point(
            (mouseevent.x, mouseevent.y))
        v = x if self.direction == 'vertical' else y
        # Assert y is correct first
        if not (0.0 <= v <= 1.0):
            return False, {}
        x_pt = self._get_point_size_in_data_units()
        hit = self._range[0] - x_pt, self._range[1] + x_pt
        if hit[0] < self._get_mouse_position < hit[1]:
            return True, {}
        return False, {}

    def release(self, event):
        """When the button is realeased, the span stays in the screen and the
        iteractivity machinery passes to modify mode"""
        if self.pressv is None or (self.ignore(event) and not self.buttonDown):
            return
        self.buttonDown = False
        self.update_range()
        self.set_initial()

    def _get_point_size_in_data_units(self):
        # Calculate the point size in data units
        invtrans = self.ax.transData.inverted()
        (x, y) = (1, 0) if self.direction == 'horizontal' else (0, 1)
        x_pt = self.tolerance * abs((invtrans.transform((x, y)) -
                                     invtrans.transform((0, 0)))[y])
        return x_pt

    def mm_on_press(self, event):
        if self.ignore(event) and not self.buttonDown:
            return
        self.buttonDown = True

        x_pt = self._get_point_size_in_data_units()

        # Determine the size of the regions for moving and stretching
        self.update_range()
        left_region = self._range[0] - x_pt, self._range[0] + x_pt
        right_region = self._range[1] - x_pt, self._range[1] + x_pt
        middle_region = self._range[0] + x_pt, self._range[1] - x_pt

        if in_interval(self._get_mouse_position(event), left_region) is True:
            self.on_move_cid = \
                self.canvas.mpl_connect('motion_notify_event',
                                        self.move_left)
        elif in_interval(self._get_mouse_position(event), right_region):
            self.on_move_cid = \
                self.canvas.mpl_connect('motion_notify_event',
                                        self.move_right)
        elif in_interval(self._get_mouse_position(event), middle_region):
            self.pressv = self._get_mouse_position(event)
            self.on_move_cid = \
                self.canvas.mpl_connect('motion_notify_event',
                                        self.move_rect)
        else:
            return

    def update_range(self):
        self._range = (self._get_span_x(),
                       self._get_span_x() + self._get_span_width())

    def switch_left_right(self, x, left_to_right):
        if left_to_right:
            if self.step_ax is not None:
                if x > self.step_ax.high_value + self.step_ax.scale:
                    return
            w = self._range[1] - self._range[0]
            r0 = self._range[1]
            self._set_span_x(r0)
            r1 = r0 + w
            self.canvas.mpl_disconnect(self.on_move_cid)
            self.on_move_cid = \
                self.canvas.mpl_connect('motion_notify_event',
                                        self.move_right)
        else:
            if self.step_ax is not None:
                if x < self.step_ax.low_value - self.step_ax.scale:
                    return
            w = self._range[1] - self._range[0]
            r1 = self._range[0]
            r0 = r1 - w
            self.canvas.mpl_disconnect(self.on_move_cid)
            self.on_move_cid = \
                self.canvas.mpl_connect('motion_notify_event',
                                        self.move_left)
        self._range = (r0, r1)

    def move_left(self, event):
        if self.buttonDown is False or self.ignore(event):
            return
        x = self._get_mouse_position(event)
        if self.step_ax is not None:
            if (self.bounds_check and
                    x < self.step_ax.low_value - self.step_ax.scale):
                return
            if self.snap_position:
                snap_offset = self.step_ax.offset - 0.5 * self.step_ax.scale
            elif self.snap_size:
                snap_offset = self._range[1]
            if self.snap_position or self.snap_size:
                rem = (x - snap_offset) % self.step_ax.scale
                if rem / self.step_ax.scale < 0.5:
                    rem = -rem
                else:
                    rem = self.step_ax.scale - rem
                x += rem
        # Do not move the left edge beyond the right one.
        if x >= self._range[1]:
            if self.can_switch and x > self._range[1]:
                self.switch_left_right(x, True)
                self.move_right(event)
            return
        width_increment = self._range[0] - x
        if self._get_span_width() + width_increment <= 0:
            return
        self._set_span_x(x)
        self._set_span_width(self._get_span_width() + width_increment)
        self.update_range()
        self.events.moved.trigger(self)
        self.events.resized.trigger(self)
        self.events.changed.trigger(self)
        if self.onmove_callback is not None:
            self.onmove_callback(*self._range)
        self.draw_patch()

    def move_right(self, event):
        if self.buttonDown is False or self.ignore(event):
            return
        x = self._get_mouse_position(event)
        if self.step_ax is not None:
            if (self.bounds_check and
                    x > self.step_ax.high_value + self.step_ax.scale):
                return
            if self.snap_size:
                snap_offset = self._range[0]
                rem = (x - snap_offset) % self.step_ax.scale
                if rem / self.step_ax.scale < 0.5:
                    rem = -rem
                else:
                    rem = self.step_ax.scale - rem
                x += rem
        # Do not move the right edge beyond the left one.
        if x <= self._range[0]:
            if self.can_switch and x < self._range[0]:
                self.switch_left_right(x, False)
                self.move_left(event)
            return
        width_increment = x - self._range[1]
        if self._get_span_width() + width_increment <= 0:
            return
        self._set_span_width(self._get_span_width() + width_increment)
        self.update_range()
        self.events.resized.trigger(self)
        self.events.changed.trigger(self)
        if self.onmove_callback is not None:
            self.onmove_callback(*self._range)
        self.draw_patch()

    def move_rect(self, event):
        if self.buttonDown is False or self.ignore(event):
            return
        x_increment = self._get_mouse_position(event) - self.pressv
        if self.step_ax is not None:
            if self.snap_position:
                rem = x_increment % self.step_ax.scale
                if rem / self.step_ax.scale < 0.5:
                    rem = -rem
                else:
                    rem = self.step_ax.scale - rem
                x_increment += rem
        self._set_span_x(self._get_span_x() + x_increment)
        self.update_range()
        self.pressv += x_increment
        self.events.moved.trigger(self)
        self.events.changed.trigger(self)
        if self.onmove_callback is not None:
            self.onmove_callback(*self._range)
        self.draw_patch()

    def mm_on_release(self, event):
        if self.buttonDown is False or self.ignore(event):
            return
        self.buttonDown = False
        self.canvas.mpl_disconnect(self.on_move_cid)
        self.on_move_cid = None

    def turn_off(self):
        self.disconnect_events()
        if self.on_move_cid is not None:
            self.canvas.mpl_disconnect(self.on_move_cid)
        self.ax.patches.remove(self.rect)
        self.ax.figure.canvas.draw_idle()
