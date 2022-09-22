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

import inspect
import logging
from packaging.version import Version

import matplotlib
import numpy as np

from hyperspy.drawing.widget import ResizableDraggableWidgetBase
from hyperspy.defaults_parser import preferences

if Version(matplotlib.__version__) >= Version('3.6.0'):
    from matplotlib.widgets import SpanSelector
else:
    from hyperspy.external.matplotlib.widgets import SpanSelector

_logger = logging.getLogger(__name__)


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

    def __init__(self, axes_manager, ax=None, color='r', alpha=0.25, **kwargs):
        # Parse all kwargs for the matplotlib SpanSelector
        self._SpanSelector_kwargs = {}
        for key in inspect.signature(SpanSelector).parameters.keys():
            if key in kwargs:
                self._SpanSelector_kwargs[key] = kwargs.pop(key)

        self._SpanSelector_kwargs.update(
            dict(onselect=lambda *args, **kwargs: None,
                 interactive=True,
                 onmove_callback=self._span_changed,
                 drag_from_anywhere=True,
                 ignore_event_outside=True,
                 grab_range=preferences.Plot.pick_tolerance)
            )
        self._SpanSelector_kwargs.setdefault('direction', 'horizontal')
        super(RangeWidget, self).__init__(axes_manager, color=color, alpha=alpha,
                                          **kwargs)
        self.span = None

    def set_on(self, value, render_figure=True):
        if value is not self.is_on and self.ax is not None:
            if value is True:
                self._add_patch_to(self.ax)
                self.connect(self.ax)
            elif value is False:
                self.disconnect()
                self.span = None
                self.ax = None
            if render_figure:
                self.draw_patch()

        self._is_on = value

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, color):
        self._color = color
        if getattr(self, 'span', None) is not None:
            self.span.set_props(color=color)
            self.span.set_handle_props(color=color)

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = alpha
        if getattr(self, 'span', None) is not None:
            self.span.set_props(alpha=alpha)
            self.span.set_handle_props(alpha=min(1.0, alpha*2))

    @property
    def patch(self):
        return self.span.artists

    def _do_snap_position(self, *args, **kwargs):
        # set span extents to snap position
        self._set_span_extents(*self.span.extents)
        return self.span.extents[0]

    def _set_snap_position(self, value):
        self._snap_position = value
        if self.span is None:
            return
        axis = self.axes[0]
        if value and axis.is_uniform:
            o, values = axis.scale / 2, axis.axis
            self.span.snap_values = np.append(values - o, [values[-1] + o])
            self._do_snap_position()
        else:
            self.span.snap_values = None

    def _add_patch_to(self, ax):
        self.ax = ax
        self._SpanSelector_kwargs.update(
            props={"alpha":self.alpha, "color":self.color},
            handle_props={"alpha":min(1.0, self.alpha*2), "color":self.color},
            useblit=ax.figure.canvas.supports_blit,
            )
        self.span = SpanSelector(ax, **self._SpanSelector_kwargs)
        self._set_span_extents(*self._get_range())
        self._patch = list(self.span.artists)

    def disconnect(self):
        self.span.disconnect_events()
        super().disconnect()

    def _set_span_extents(self, left, right):
        self.span.extents = (left, right)
        # update internal state range widget
        self._span_changed()

    def _span_changed(self, *args, **kwargs):
        extents = self.span.extents
        self._pos = np.array([extents[0]])
        self._size = np.array([extents[1] - extents[0]])
        self.events.changed.trigger(self)

    def _get_range(self):
        p = self._pos[0]
        w = self._size[0]
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
        self.set_bounds(left=x, width=w)

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
        x, w = self._parse_bounds_args(args, kwargs)
        if self.span is not None:
            axis = self.axes[0]
            if axis.is_uniform and w <= axis.scale:
                w = axis.scale
            x0, x1 = np.clip([x, x+w], axis.axis[0], axis.axis[-1])
            self._set_span_extents(x0, x1)

    def _update_patch_position(self):
        self._update_patch_geometry()

    def _update_patch_size(self):
        self._update_patch_geometry()

    def _update_patch_geometry(self):
        if self.is_on and self.span is not None:
            self._set_span_extents(*self._get_range())
