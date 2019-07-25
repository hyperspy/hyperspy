# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy.  If not, see <http://www.gnu.org/licenses/>.

"""Region of interests (ROIs).

ROIs operate on `BaseSignal` instances and include widgets for interactive
operation.

The following 1D ROIs are available:

    Point1DROI
        Single element ROI of a 1D signal.

    SpanROI
        Interval ROI of a 1D signal.

The following 2D ROIs are available:

    Point2DROI
        Single element ROI of a 2D signal.

    RectangularROI
        Rectagular ROI of a 2D signal.

    CircleROI
        (Hollow) circular ROI of a 2D signal

    Line2DROI
        Line profile of a 2D signal with customisable width.

"""

from functools import partial

import traits.api as t
import numpy as np

from hyperspy.events import Events, Event
from hyperspy.interactive import interactive
from hyperspy.axes import DataAxis
from hyperspy.drawing import widgets
from hyperspy.ui_registry import add_gui_method


class BaseROI(t.HasTraits):

    """Base class for all ROIs.

    Provides some basic functionality that is likely to be shared between all
    ROIs, and serve as a common type that can be checked for.
    """

    def __init__(self):
        """Sets up events.changed event, and inits HasTraits.
        """
        super(BaseROI, self).__init__()
        self.events = Events()
        self.events.changed = Event("""
            Event that triggers when the ROI has changed.

            What constitues a change varies from ROI to ROI, but in general it
            should correspond to the region selected by the ROI has changed.

            Arguments:
            ----------
                roi :
                    The ROI that was changed.
            """, arguments=['roi'])
        self.signal_map = dict()

    _ndim = 0
    ndim = property(lambda s: s._ndim)

    def is_valid(self):
        """
        Determine if the ROI is in a valid state.

        This is typically determined by all the coordinates being defined,
        and that the values makes sense relative to each other.
        """
        raise NotImplementedError()

    def update(self):
        """Function responsible for updating anything that depends on the ROI.
        It should be called by implementors whenever the ROI changes.
        The base implementation simply triggers the changed event.
        """
        if self.is_valid():
            self.events.changed.trigger(self)

    def _get_ranges(self):
        """
        Utility to get the value ranges that the ROI would select.

        If the ROI is point base or is rectangluar in nature, these can be used
        slice a signal. Extracted from `_make_slices` to ease implementation
        in inherited ROIs.
        """
        raise NotImplementedError()

    def _make_slices(self, axes_collection, axes, ranges=None):
        """
        Utility function to make a slice structure that will slice all the axes
        in 'axes_collection'. The axes in the `axes` argument will be sliced by
        the ROI, all other axes with 'slice(None)'. Alternatively, if 'ranges'
        is passed, `axes[i]` will be sliced with 'ranges[i]'.
        """
        if ranges is None:
            # Use ROI to slice
            ranges = self._get_ranges()
        slices = []
        for ax in axes_collection:
            if ax in axes:
                i = axes.index(ax)
                try:
                    ilow = ax.value2index(ranges[i][0])
                except ValueError:
                    if ranges[i][0] < ax.low_value:
                        ilow = ax.low_index
                    else:
                        raise
                if len(ranges[i]) == 1:
                    slices.append(ilow)
                else:
                    try:
                        ihigh = ax.value2index(ranges[i][1])
                    except ValueError:
                        if ranges[i][1] > ax.high_value:
                            ihigh = ax.high_index + 1
                        else:
                            raise
                    slices.append(slice(ilow, ihigh))
            else:
                slices.append(slice(None))
        return tuple(slices)

    def __call__(self, signal, out=None, axes=None):
        """Slice the signal according to the ROI, and return it.

        Arguments
        ---------
        signal : Signal
            The signal to slice with the ROI.
        out : Signal, default = None
            If the 'out' argument is supplied, the sliced output will be put
            into this instead of returning a Signal. See Signal.__getitem__()
            for more details on 'out'.
        axes : specification of axes to use, default = None
            The axes argument specifies which axes the ROI will be applied on.
            The items in the collection can be either of the following:
                * a tuple of:
                    - DataAxis. These will not be checked with
                      signal.axes_manager.
                    - anything that will index signal.axes_manager
                * For any other value, it will check whether the navigation
                  space can fit the right number of axis, and use that if it
                  fits. If not, it will try the signal space.
        """
        if axes is None and signal in self.signal_map:
            axes = self.signal_map[signal][1]
        else:
            axes = self._parse_axes(axes, signal.axes_manager)

        natax = signal.axes_manager._get_axes_in_natural_order()
        slices = self._make_slices(natax, axes)
        nav_axes = [ax.navigate for ax in axes]
        nav_dim = signal.axes_manager.navigation_dimension
        if True in nav_axes:
            if False in nav_axes:

                slicer = signal.inav[slices[:nav_dim]].isig.__getitem__
                slices = slices[nav_dim:]
            else:
                slicer = signal.inav.__getitem__
                slices = slices[0:nav_dim]
        else:
            slicer = signal.isig.__getitem__
            slices = slices[nav_dim:]

        roi = slicer(slices, out=out)
        return roi

    def _parse_axes(self, axes, axes_manager):
        """Utility function to parse the 'axes' argument to a list of
        DataAxis.

        Arguments
        ---------
        axes : specification of axes to use, default = None
            The axes argument specifies which axes the ROI will be applied on.
            The DataAxis in the collection can be either of the following:
                * a tuple of:
                    - DataAxis. These will not be checked with
                      signal.axes_manager.
                    - anything that will index signal.axes_manager
                * For any other value, it will check whether the navigation
                  space can fit the right number of axis, and use that if it
                  fits. If not, it will try the signal space.
        axes_manager : AxesManager
            The AxesManager to use for parsing axes, if axes is not already a
            tuple of DataAxis.

        Returns
        -------
        [<DataAxis>]
        """
        nd = self.ndim
        if isinstance(axes, (tuple, list)):
            axes_out = axes_manager[axes[:nd]]
        else:
            if axes_manager.navigation_dimension >= nd:
                axes_out = axes_manager.navigation_axes[:nd]
            elif axes_manager.signal_dimension >= nd:
                axes_out = axes_manager.signal_axes[:nd]
            elif nd == 2 and axes_manager.navigation_dimension == 1 and \
                    axes_manager.signal_dimension == 1:
                # We probably have a navigator plot including both nav and sig
                # axes.
                axes_out = [axes_manager.signal_axes[0],
                            axes_manager.navigation_axes[0]]
            else:
                raise ValueError("Could not find valid axes configuration.")

        return axes_out


def _get_mpl_ax(plot, axes):
    """
    Returns MPL Axes that contains the `axes`.

    The space of the first DataAxis in axes will be used to determine which
    plot's matplotlib Axes to return.

    Arguments:
    ----------
        plot : MPL_HyperExplorer
            The explorer that contains the navigation and signal plots
        axes : collection of DataAxis
            The axes to infer from.
    """
    if axes[0].navigate:
        ax = plot.navigator_plot.ax
    else:
        if len(axes) == 2 and axes[1].navigate:
            ax = plot.navigator_plot.ax
        else:
            ax = plot.signal_plot.ax
    return ax


class BaseInteractiveROI(BaseROI):

    """Base class for interactive ROIs, i.e. ROIs with widget interaction.
    The base class defines a lot of the common code for interacting with
    widgets, but inhertors need to implement the following functions:

    _get_widget_type()
    _apply_roi2widget(widget)
    _set_from_widget(widget)
    """

    def __init__(self):
        super(BaseInteractiveROI, self).__init__()
        self.widgets = set()
        self._applying_widget_change = False

    def update(self):
        """Function responsible for updating anything that depends on the ROI.
        It should be called by implementors whenever the ROI changes.
        This implementation  updates the widgets associated with it, and
        triggers the changed event.
        """
        if self.is_valid():
            if not self._applying_widget_change:
                self._update_widgets()
            self.events.changed.trigger(self)

    def _update_widgets(self, exclude=None):
        """Internal function for updating the associated widgets to the
        geometry contained in the ROI.

        Arguments
        ---------
        exclude : set()
            A set of widgets to exclude from the update. Useful e.g. if a
            widget has triggered a change in the ROI: Then all widgets,
            excluding the one that was the source for the change, should be
            updated.
        """
        if exclude is None:
            exclude = set()
        if not isinstance(exclude, set):
            exclude = set(exclude)
        for w in self.widgets - exclude:
            with w.events.changed.suppress_callback(self._on_widget_change):
                self._apply_roi2widget(w)

    def _get_widget_type(self, axes, signal):
        """Get the type of a widget that can represent the ROI on the given
        axes and signal.
        """
        raise NotImplementedError()

    def _apply_roi2widget(self, widget):
        """This function is responsible for applying the ROI geometry to the
        widget. When this function is called, the widget's events are already
        suppressed, so this should not be necessary for _apply_roi2widget to
        handle.
        """
        raise NotImplementedError()

    def _set_from_widget(self, widget):
        """Sets the internal representation of the ROI from the passed widget,
        without doing anything to events.
        """
        raise NotImplementedError()

    def interactive(self, signal, navigation_signal="same", out=None,
                    color="green", **kwargs):
        """Creates an interactively sliced Signal (sliced by this ROI) via
        hyperspy.interactive.

        Arguments:
        ----------
        signal : Signal
            The source signal to slice
        navigation_signal : Signal, None or "same" (default)
            If not None, it will automatically create a widget on
            navigation_signal. Passing "same" is identical to passing the same
            signal to 'signal' and 'navigation_signal', but is less ambigous,
            and allows "same" to be the default value.
        out : Signal
            If not None, it will use 'out' as the output instead of returning
            a new Signal.
        color : Matplotlib color specifier (default: 'green')
            The color for the widget. Any format that matplotlib uses should be
            ok. This will not change the color fo any widget passed with the
            'widget' argument.
        **kwargs
            All kwargs are passed to the roi __call__ method which is called
            interactivel on any roi attribute change.

        """
        if hasattr(signal, '_plot_kwargs'):
            kwargs.update({'_plot_kwargs': signal._plot_kwargs})
            # in case of complex signal, it is possible to shift the signal
            # during plotting, if so this is currently not supported and we
            # raise a NotImplementedError
            if signal._plot.signal_data_function_kwargs.get(
                    'fft_shift', False):
                raise NotImplementedError('ROIs are not supported when data '
                                          'are shifted during plotting.')
        if isinstance(navigation_signal, str) and navigation_signal == "same":
            navigation_signal = signal
        if navigation_signal is not None:
            if navigation_signal not in self.signal_map:
                self.add_widget(navigation_signal, color=color,
                                axes=kwargs.get("axes", None))
        if (self.update not in
                signal.axes_manager.events.any_axis_changed.connected):
            signal.axes_manager.events.any_axis_changed.connect(
                self.update,
                [])
        if out is None:
            return interactive(self.__call__,
                               event=self.events.changed,
                               signal=signal,
                               **kwargs)
        else:
            return interactive(self.__call__,
                               event=self.events.changed,
                               signal=signal, out=out, **kwargs)

    def _on_widget_change(self, widget):
        """Callback for widgets' 'changed' event. Updates the internal state
        from the widget, and triggers events (excluding connections to the
        source widget).
        """
        with self.events.suppress():
            self._bounds_check = False
            self._applying_widget_change = True
            try:
                self._set_from_widget(widget)
            finally:
                self._bounds_check = True
                self._applying_widget_change = False
        self._update_widgets(exclude=(widget,))
        self.events.changed.trigger(self)

    def add_widget(self, signal, axes=None, widget=None,
                   color='green', **kwargs):
        """Add a widget to visually represent the ROI, and connect it so any
        changes in either are reflected in the other. Note that only one
        widget can be added per signal/axes combination.

        Arguments:
        ----------
        signal : Signal
            The signal to witch the widget is added. This is used to determine
            with plot to add the widget to, and it supplies the axes_manager
            for the widget.
        axes : specification of axes to use, default = None
            The axes argument specifies which axes the ROI will be applied on.
            The DataAxis in the collection can be either of the following:
                * a tuple of:
                    - DataAxis. These will not be checked with
                      signal.axes_manager.
                    - anything that will index signal.axes_manager
                * For any other value, it will check whether the navigation
                  space can fit the right number of axis, and use that if it
                  fits. If not, it will try the signal space.
        widget : Widget or None (default)
            If specified, this is the widget that will be added. If None, the
            default widget will be used, as given by _get_widget_type().
        color : Matplotlib color specifier (default: 'green')
            The color for the widget. Any format that matplotlib uses should be
            ok. This will not change the color fo any widget passed with the
            'widget' argument.
        kwargs:
            All keyword argument are passed to the widget constructor.
        """
        axes = self._parse_axes(axes, signal.axes_manager,)
        if widget is None:
            widget = self._get_widget_type(
                axes, signal)(
                signal.axes_manager, **kwargs)
            widget.color = color

        # Remove existing ROI, if it exsists and axes match
        if signal in self.signal_map and \
                self.signal_map[signal][1] == axes:
            self.remove_widget(signal)

        # Set DataAxes
        widget.axes = axes
        if widget.ax is None:
            if signal._plot is None:
                raise Exception(
                    "%s does not have an active plot. Plot the signal before "
                    "calling this method using its `plot` method." %
                    repr(signal))

            ax = _get_mpl_ax(signal._plot, axes)
            widget.set_mpl_ax(ax)
        with widget.events.changed.suppress_callback(self._on_widget_change):
            self._apply_roi2widget(widget)

        # Connect widget changes to on_widget_change
        widget.events.changed.connect(self._on_widget_change,
                                      {'obj': 'widget'})
        # When widget closes, remove from internal list
        widget.events.closed.connect(self._remove_widget, {'obj': 'widget'})
        self.widgets.add(widget)
        self.signal_map[signal] = (widget, axes)
        return widget

    def _remove_widget(self, widget):
        widget.events.closed.disconnect(self._remove_widget)
        widget.events.changed.disconnect(self._on_widget_change)
        widget.close()
        for signal, w in self.signal_map.items():
            if w[0] == widget:
                self.signal_map.pop(signal)
                break

    def remove_widget(self, signal):
        if signal in self.signal_map:
            w = self.signal_map.pop(signal)[0]
            self._remove_widget(w)


class BasePointROI(BaseInteractiveROI):

    """Base ROI class for point ROIs, i.e. ROIs with a unit size in each of its
    dimensions.
    """

    def __call__(self, signal, out=None, axes=None):
        if axes is None and signal in self.signal_map:
            axes = self.signal_map[signal][1]
        else:
            axes = self._parse_axes(axes, signal.axes_manager)
        s = super(BasePointROI, self).__call__(signal=signal, out=out,
                                               axes=axes)
        return s


def guess_vertical_or_horizontal(axes, signal):
    # Figure out whether to use horizontal or veritcal line:
    if axes[0].navigate:
        plotdim = len(signal._plot.navigator_data_function().shape)
        axdim = signal.axes_manager.navigation_dimension
        idx = signal.axes_manager.navigation_axes.index(axes[0])
    else:
        plotdim = len(signal._plot.signal_data_function().shape)
        axdim = signal.axes_manager.signal_dimension
        idx = signal.axes_manager.signal_axes.index(axes[0])

    if plotdim == 2:  # Plot is an image
        # axdim == 1 and plotdim == 2 indicates "spectrum stack"
        if idx == 0 and axdim != 1:    # Axis is horizontal
            return "vertical"
        else:  # Axis is vertical
            return "horizontal"
    elif plotdim == 1:  # It is a spectrum
        return "vertical"
    else:
        raise ValueError(
            "Could not find valid widget type for the given `axes` value")


@add_gui_method(toolkey="Point1DROI")
class Point1DROI(BasePointROI):

    """Selects a single point in a 1D space. The coordinate of the point in the
    1D space is stored in the 'value' trait.
    """
    value = t.CFloat(t.Undefined)
    _ndim = 1

    def __init__(self, value):
        super(Point1DROI, self).__init__()
        self.value = value

    def is_valid(self):
        return self.value != t.Undefined

    def _value_changed(self, old, new):
        self.update()

    def _get_ranges(self):
        ranges = ((self.value,),)
        return ranges

    def _set_from_widget(self, widget):
        self.value = widget.position[0]

    def _apply_roi2widget(self, widget):
        widget.position = (self.value,)

    def _get_widget_type(self, axes, signal):
        direction = guess_vertical_or_horizontal(axes=axes, signal=signal)
        if direction == "vertical":
            return widgets.VerticalLineWidget
        elif direction == "horizontal":
            return widgets.HorizontalLineWidget
        else:
            raise ValueError("direction must be either horizontal or vertical")

    def __repr__(self):
        return "%s(value=%g)" % (
            self.__class__.__name__,
            self.value)


@add_gui_method(toolkey="Point2DROI")
class Point2DROI(BasePointROI):

    """Selects a single point in a 2D space. The coordinates of the point in
    the 2D space are stored in the traits 'x' and 'y'.
    """
    x, y = (t.CFloat(t.Undefined),) * 2
    _ndim = 2

    def __init__(self, x, y):
        super(Point2DROI, self).__init__()
        self.x, self.y = x, y

    def is_valid(self):
        return t.Undefined not in (self.x, self.y)

    def _x_changed(self, old, new):
        self.update()

    def _y_changed(self, old, new):
        self.update()

    def _get_ranges(self):
        ranges = ((self.x,), (self.y,),)
        return ranges

    def _set_from_widget(self, widget):
        self.x, self.y = widget.position

    def _apply_roi2widget(self, widget):
        widget.position = (self.x, self.y)

    def _get_widget_type(self, axes, signal):
        return widgets.SquareWidget

    def __repr__(self):
        return "%s(x=%g, y=%g)" % (
            self.__class__.__name__,
            self.x, self.y)


@add_gui_method(toolkey="SpanROI")
class SpanROI(BaseInteractiveROI):

    """Selects a range in a 1D space. The coordinates of the range in
    the 1D space are stored in the traits 'left' and 'right'.
    """
    left, right = (t.CFloat(t.Undefined),) * 2
    _ndim = 1

    def __init__(self, left, right):
        super(SpanROI, self).__init__()
        self._bounds_check = True   # Use reponsibly!
        self.left, self.right = left, right

    def is_valid(self):
        return (t.Undefined not in (self.left, self.right) and
                self.right >= self.left)

    def _right_changed(self, old, new):
        if self._bounds_check and \
                self.left is not t.Undefined and new <= self.left:
            self.right = old
        else:
            self.update()

    def _left_changed(self, old, new):
        if self._bounds_check and \
                self.right is not t.Undefined and new >= self.right:
            self.left = old
        else:
            self.update()

    def _get_ranges(self):
        ranges = ((self.left, self.right),)
        return ranges

    def _set_from_widget(self, widget):
        value = (widget.position[0], widget.position[0] + widget.size[0])
        self.left, self.right = value

    def _apply_roi2widget(self, widget):
        widget.set_bounds(left=self.left, right=self.right)

    def _get_widget_type(self, axes, signal):
        direction = guess_vertical_or_horizontal(axes=axes, signal=signal)
        if direction == "vertical":
            return partial(widgets.RangeWidget, direction="horizontal")
        elif direction == "horizontal":
            return partial(widgets.RangeWidget, direction="vertical")
        else:
            raise ValueError("direction must be either horizontal or vertical")

    def __repr__(self):
        return "%s(left=%g, right=%g)" % (
            self.__class__.__name__,
            self.left,
            self.right)


@add_gui_method(toolkey="RectangularROI")
class RectangularROI(BaseInteractiveROI):

    """Selects a range in a 2D space. The coordinates of the range in
    the 2D space are stored in the traits 'left', 'right', 'top' and 'bottom'.
    Convenience properties 'x', 'y', 'width' and 'height' are also available,
    but cannot be used for initialization.
    """
    top, bottom, left, right = (t.CFloat(t.Undefined),) * 4
    _ndim = 2

    def __init__(self, left, top, right, bottom):
        super(RectangularROI, self).__init__()
        self._bounds_check = True   # Use reponsibly!
        self.top, self.bottom, self.left, self.right = top, bottom, left, right

    def is_valid(self):
        return (t.Undefined not in (self.top, self.bottom,
                                    self.left, self.right) and
                self.right >= self.left and self.bottom >= self.top)

    def _top_changed(self, old, new):
        if self._bounds_check and \
                self.bottom is not t.Undefined and new >= self.bottom:
            self.top = old
        else:
            self.update()

    @property
    def width(self):
        """Returns / sets the width of the ROI"""
        return self.right - self.left

    @width.setter
    def width(self, value):
        if value == self.width:
            return
        self.right -= self.width - value

    @property
    def height(self):
        """Returns / sets the height of the ROI"""
        return self.bottom - self.top

    @height.setter
    def height(self, value):
        if value == self.height:
            return
        self.bottom -= self.height - value

    @property
    def x(self):
        """Returns / sets the x coordinate of the ROI without changing its
        width"""
        return self.left

    @x.setter
    def x(self, value):
        if value != self.x:
            diff = value - self.x
            try:
                self._applying_widget_change = True
                self._bounds_check = False
                with self.events.changed.suppress():
                    self.right += diff
                    self.left += diff
            finally:
                self._applying_widget_change = False
                self._bounds_check = True
                self.update()

    @property
    def y(self):
        """Returns / sets the y coordinate of the ROI without changing its
        height"""
        return self.top

    @y.setter
    def y(self, value):
        if value != self.y:
            diff = value - self.y
            try:
                self._applying_widget_change = True
                self._bounds_check = False
                with self.events.changed.suppress():
                    self.top += diff
                    self.bottom += diff
            finally:
                self._applying_widget_change = False
                self._bounds_check = True
                self.update()

    def _bottom_changed(self, old, new):
        if self._bounds_check and \
                self.top is not t.Undefined and new <= self.top:
            self.bottom = old
        else:
            self.update()

    def _right_changed(self, old, new):
        if self._bounds_check and \
                self.left is not t.Undefined and new <= self.left:
            self.right = old
        else:
            self.update()

    def _left_changed(self, old, new):
        if self._bounds_check and \
                self.right is not t.Undefined and new >= self.right:
            self.left = old
        else:
            self.update()

    def _get_ranges(self):
        ranges = ((self.left, self.right), (self.top, self.bottom),)
        return ranges

    def _set_from_widget(self, widget):
        p = np.array(widget.position)
        s = np.array(widget.size)
        (self.left, self.top), (self.right, self.bottom) = (p, p + s)

    def _apply_roi2widget(self, widget):
        widget.set_bounds(left=self.left, bottom=self.bottom,
                          right=self.right, top=self.top)

    def _get_widget_type(self, axes, signal):
        return widgets.RectangleWidget

    def __repr__(self):
        return "%s(left=%g, top=%g, right=%g, bottom=%g)" % (
            self.__class__.__name__,
            self.left,
            self.top,
            self.right,
            self.bottom)


@add_gui_method(toolkey="CircleROI")
class CircleROI(BaseInteractiveROI):

    cx, cy, r, r_inner = (t.CFloat(t.Undefined),) * 4
    _ndim = 2

    def __init__(self, cx, cy, r, r_inner=None):
        super(CircleROI, self).__init__()
        self._bounds_check = True   # Use reponsibly!
        self.cx, self.cy, self.r = cx, cy, r
        if r_inner:
            self.r_inner = r_inner

    def is_valid(self):
        return (t.Undefined not in (self.cx, self.cy, self.r,) and
                (self.r_inner is t.Undefined or
                 t.Undefined not in (self.r, self.r_inner) and
                 self.r >= self.r_inner))

    def _cx_changed(self, old, new):
        self.update()

    def _cy_changed(self, old, new):
        self.update()

    def _r_changed(self, old, new):
        if self._bounds_check and \
                self.r_inner is not t.Undefined and new < self.r_inner:
            self.r = old
        else:
            self.update()

    def _r_inner_changed(self, old, new):
        if self._bounds_check and \
                self.r is not t.Undefined and new >= self.r:
            self.r_inner = old
        else:
            self.update()

    def _set_from_widget(self, widget):
        """Sets the internal representation of the ROI from the passed widget,
        without doing anything to events.
        """
        self.cx, self.cy = widget.position
        self.r, self.r_inner = widget.size

    def _apply_roi2widget(self, widget):
        widget.position = (self.cx, self.cy)
        inner = self.r_inner if self.r_inner != t.Undefined else 0.0
        widget.size = (self.r, inner)

    def _get_widget_type(self, axes, signal):
        return widgets.CircleWidget

    def __call__(self, signal, out=None, axes=None):
        """Slice the signal according to the ROI, and return it.

        Arguments
        ---------
        signal : Signal
            The signal to slice with the ROI.
        out : Signal, default = None
            If the 'out' argument is supplied, the sliced output will be put
            into this instead of returning a Signal. See Signal.__getitem__()
            for more details on 'out'.
        axes : specification of axes to use, default = None
            The axes argument specifies which axes the ROI will be applied on.
            The items in the collection can be either of the following:
                * a tuple of:
                    - DataAxis. These will not be checked with
                      signal.axes_manager.
                    - anything that will index signal.axes_manager
                * For any other value, it will check whether the navigation
                  space can fit the right number of axis, and use that if it
                  fits. If not, it will try the signal space.
        """

        if axes is None and signal in self.signal_map:
            axes = self.signal_map[signal][1]
        else:
            axes = self._parse_axes(axes, signal.axes_manager)

        natax = signal.axes_manager._get_axes_in_natural_order()
        # Slice original data with a circumscribed rectangle
        cx = self.cx + 0.5001 * axes[0].scale
        cy = self.cy + 0.5001 * axes[1].scale
        ranges = [[cx - self.r, cx + self.r],
                  [cy - self.r, cy + self.r]]
        slices = self._make_slices(natax, axes, ranges)
        ir = [slices[natax.index(axes[0])],
              slices[natax.index(axes[1])]]
        vx = axes[0].axis[ir[0]] - cx
        vy = axes[1].axis[ir[1]] - cy
        gx, gy = np.meshgrid(vx, vy)
        gr = gx**2 + gy**2
        mask = gr > self.r**2
        if self.r_inner != t.Undefined:
            mask |= gr < self.r_inner**2
        tiles = []
        shape = []
        chunks = []
        for i in range(len(slices)):
            if signal._lazy:
                chunks.append(signal.data.chunks[i][0])
            if i == natax.index(axes[0]):
                thisshape = mask.shape[0]
                tiles.append(thisshape)
                shape.append(thisshape)
            elif i == natax.index(axes[1]):
                thisshape = mask.shape[1]
                tiles.append(thisshape)
                shape.append(thisshape)
            else:
                tiles.append(signal.axes_manager._axes[i].size)
                shape.append(1)
        mask = mask.reshape(shape)

        nav_axes = [ax.navigate for ax in axes]
        nav_dim = signal.axes_manager.navigation_dimension
        if True in nav_axes:
            if False in nav_axes:

                slicer = signal.inav[slices[:nav_dim]].isig.__getitem__
                slices = slices[nav_dim:]
            else:
                slicer = signal.inav.__getitem__
                slices = slices[0:nav_dim]
        else:
            slicer = signal.isig.__getitem__
            slices = slices[nav_dim:]

        roi = slicer(slices, out=out)
        roi = out or roi
        if roi._lazy:
            import dask.array as da
            mask = da.from_array(mask, chunks=chunks)
            mask = da.broadcast_to(mask, tiles)
            # By default promotes dtype to float if required
            roi.data = da.where(mask, np.nan, roi.data)
        else:
            mask = np.broadcast_to(mask, tiles)
            roi.data = np.ma.masked_array(roi.data, mask, hard_mask=True)
        if out is None:
            return roi
        else:
            out.events.data_changed.trigger(out)

    def __repr__(self):
        if self.r_inner == t.Undefined:
            return "%s(cx=%g, cy=%g, r=%g)" % (
                self.__class__.__name__,
                self.cx,
                self.cy,
                self.r)
        else:
            return "%s(cx=%g, cy=%g, r=%g, r_inner=%g)" % (
                self.__class__.__name__,
                self.cx,
                self.cy,
                self.r,
                self.r_inner)


@add_gui_method(toolkey="Line2DROI")
class Line2DROI(BaseInteractiveROI):

    x1, y1, x2, y2, linewidth = (t.CFloat(t.Undefined),) * 5
    _ndim = 2

    def __init__(self, x1, y1, x2, y2, linewidth=0):
        super(Line2DROI, self).__init__()
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.linewidth = linewidth

    def is_valid(self):
        return t.Undefined not in (self.x1, self.y1, self.x2, self.y2)

    def _x1_changed(self, old, new):
        self.update()

    def _x2_changed(self, old, new):
        self.update()

    def _y1_changed(self, old, new):
        self.update()

    def _y2_changed(self, old, new):
        self.update()

    def _linewidth_changed(self, old, new):
        self.update()

    def _set_from_widget(self, widget):
        """Sets the internal representation of the ROI from the passed widget,
        without doing anything to events.
        """
        c = widget.position
        s = widget.size[0]
        (self.x1, self.y1), (self.x2, self.y2) = c
        self.linewidth = s

    def _apply_roi2widget(self, widget):
        widget.position = (self.x1, self.y1), (self.x2, self.y2)
        widget.size = np.array([self.linewidth])

    def _get_widget_type(self, axes, signal):
        return widgets.Line2DWidget

    @staticmethod
    def _line_profile_coordinates(src, dst, linewidth=1):
        """Return the coordinates of the profile of an image along a scan line.

        Parameters
        ----------
        src : 2-tuple of numeric scalar (float or int)
            The start point of the scan line.
        dst : 2-tuple of numeric scalar (float or int)
            The end point of the scan line.
        linewidth : int, optional
            Width of the scan, perpendicular to the line
        Returns
        -------
        coords : array, shape (2, N, C), float
            The coordinates of the profile along the scan line. The length of
            the profile is the ceil of the computed length of the scan line.
        Notes
        -----
        This is a utility method meant to be used internally by skimage
        functions. The destination point is included in the profile, in
        contrast to standard numpy indexing.

        """
        src_row, src_col = src = np.asarray(src, dtype=float)
        dst_row, dst_col = dst = np.asarray(dst, dtype=float)
        d_row, d_col = dst - src
        theta = np.arctan2(d_row, d_col)

        length = np.ceil(np.hypot(d_row, d_col) + 1).astype(int)
        # we add one above because we include the last point in the profile
        # (in contrast to standard numpy indexing)
        line_col = np.linspace(src_col, dst_col, length)
        line_row = np.linspace(src_row, dst_row, length)
        data = np.zeros((2, length, linewidth))
        data[0, :, :] = np.tile(line_col, [linewidth, 1]).T
        data[1, :, :] = np.tile(line_row, [linewidth, 1]).T

        if linewidth != 1:
            # we subtract 1 from linewidth to change from pixel-counting
            # (make this line 3 pixels wide) to point distances (the
            # distance between pixel centers)
            col_width = (linewidth - 1) * np.sin(-theta) / 2
            row_width = (linewidth - 1) * np.cos(theta) / 2
            row_off = np.linspace(-row_width, row_width, linewidth)
            col_off = np.linspace(-col_width, col_width, linewidth)
            data[0, :, :] += np.tile(col_off, [length, 1])
            data[1, :, :] += np.tile(row_off, [length, 1])
        return data

    @property
    def length(self):
        p0 = np.array((self.x1, self.y1), dtype=np.float)
        p1 = np.array((self.x2, self.y2), dtype=np.float)
        d_row, d_col = p1 - p0
        return np.hypot(d_row, d_col)

    def angle(self, axis='horizontal', units='degrees'):
        """"Angle between ROI line and selected axis

        Parameters
        ----------
        axis : str, {'horizontal', 'vertical'}, optional
            Select axis against which the angle of the ROI line is measured.
            'x' is alias to 'horizontal' and 'y' is 'vertical'
            (Default: 'horizontal')
        units : str, {'degrees', 'radians'}
            The angle units of the output
            (Default: 'degrees')

        Returns
        -------
        angle : float

        Examples
        --------
        >>> import hyperspy.api as hs
        >>> hs.roi.Line2DROI(0., 0., 1., 2., 1)
        >>> r.angle()
        63.43494882292201
        """

        x = self.x2 - self.x1
        y = self.y2 - self.y1

        if units == 'degrees':
            conversation = 180. / np.pi
        elif units == 'radians':
            conversation = 1.
        else:
            raise ValueError("Units are not recognized. Use  either 'degrees' or 'radians'.")

        if axis == 'horizontal':
            return np.arctan2(y, x) * conversation
        elif axis == 'vertical':
            return np.arctan2(x, y) * conversation
        else:
            raise ValueError("Axis is not recognized. "
                             "Use  either 'horizontal' or 'vertical'.")

    @staticmethod
    def profile_line(img, src, dst, axes, linewidth=1,
                     order=1, mode='constant', cval=0.0):
        """Return the intensity profile of an image measured along a scan line.

        Parameters
        ----------
        img : numeric array, shape (M, N[, C])
            The image, either grayscale (2D array) or multichannel
            (3D array, where the final axis contains the channel
            information).
        src : 2-tuple of numeric scalar (float or int)
            The start point of the scan line.
        dst : 2-tuple of numeric scalar (float or int)
            The end point of the scan line.
        linewidth : int, optional
            Width of the scan, perpendicular to the line
        order : int in {0, 1, 2, 3, 4, 5}, optional
            The order of the spline interpolation to compute image values at
            non-integer coordinates. 0 means nearest-neighbor interpolation.
        mode : string, one of {'constant', 'nearest', 'reflect', 'wrap'},
                optional
            How to compute any values falling outside of the image.
        cval : float, optional
            If `mode` is 'constant', what constant value to use outside the
            image.
        Returns
        -------
        return_value : array
            The intensity profile along the scan line. The length of the
            profile is the ceil of the computed length of the scan line.
        Examples
        --------
        >>> x = np.array([[1, 1, 1, 2, 2, 2]])
        >>> img = np.vstack([np.zeros_like(x), x, x, x, np.zeros_like(x)])
        >>> img
        array([[0, 0, 0, 0, 0, 0],
               [1, 1, 1, 2, 2, 2],
               [1, 1, 1, 2, 2, 2],
               [1, 1, 1, 2, 2, 2],
               [0, 0, 0, 0, 0, 0]])
        >>> profile_line(img, (2, 1), (2, 4))
        array([ 1.,  1.,  2.,  2.])
        Notes
        -----
        The destination point is included in the profile, in contrast to
        standard numpy indexing.

        """
        import scipy.ndimage as nd
        # Convert points coordinates from axes units to pixels
        p0 = ((src[0] - axes[0].offset) / axes[0].scale,
              (src[1] - axes[1].offset) / axes[1].scale)
        p1 = ((dst[0] - axes[0].offset) / axes[0].scale,
              (dst[1] - axes[1].offset) / axes[1].scale)
        if linewidth < 0:
            raise ValueError("linewidth must be positive number")
        linewidth_px = linewidth / np.min([ax.scale for ax in axes])
        linewidth_px = int(round(linewidth_px))
        # Minimum size 1 pixel
        linewidth_px = linewidth_px if linewidth_px >= 1 else 1
        perp_lines = Line2DROI._line_profile_coordinates(p0, p1,
                                                         linewidth=linewidth_px)
        if img.ndim > 2:
            idx = [ax.index_in_array for ax in axes]
            if idx[0] < idx[1]:
                img = np.rollaxis(img, idx[0], 0)
                img = np.rollaxis(img, idx[1], 1)
            else:
                img = np.rollaxis(img, idx[1], 0)
                img = np.rollaxis(img, idx[0], 0)
            orig_shape = img.shape
            img = np.reshape(img, orig_shape[0:2] +
                             (np.product(orig_shape[2:]),))
            pixels = [nd.map_coordinates(img[..., i].T, perp_lines,
                                         order=order, mode=mode, cval=cval)
                      for i in range(img.shape[2])]
            i0 = min(axes[0].index_in_array, axes[1].index_in_array)
            pixels = np.transpose(np.asarray(pixels), (1, 2, 0))
            intensities = pixels.mean(axis=1)
            intensities = np.rollaxis(
                np.reshape(intensities,
                           intensities.shape[0:1] + orig_shape[2:]),
                0, i0 + 1)
        else:
            pixels = nd.map_coordinates(img, perp_lines,
                                        order=order, mode=mode, cval=cval)
            intensities = pixels.mean(axis=1)

        return intensities

    def __call__(self, signal, out=None, axes=None, order=0):
        """Slice the signal according to the ROI, and return it.

        Arguments
        ---------
        signal : Signal
            The signal to slice with the ROI.
        out : Signal, default = None
            If the 'out' argument is supplied, the sliced output will be put
            into this instead of returning a Signal. See Signal.__getitem__()
            for more details on 'out'.
        axes : specification of axes to use, default = None
            The axes argument specifies which axes the ROI will be applied on.
            The items in the collection can be either of the following:
                * a tuple of:
                    - DataAxis. These will not be checked with
                      signal.axes_manager.
                    - anything that will index signal.axes_manager
                * For any other value, it will check whether the navigation
                  space can fit the right number of axis, and use that if it
                  fits. If not, it will try the signal space.
        order : The spline interpolation order to use when extracting the line
            profile. 0 means nearest-neighbor interpolation, and is both the
            default and the fastest.
        """
        if axes is None and signal in self.signal_map:
            axes = self.signal_map[signal][1]
        else:
            axes = self._parse_axes(axes, signal.axes_manager)
        profile = Line2DROI.profile_line(signal.data,
                                         (self.x1, self.y1),
                                         (self.x2, self.y2),
                                         axes=axes,
                                         linewidth=self.linewidth,
                                         order=order)
        length = np.linalg.norm(np.diff(
            np.array(((self.x1, self.y1), (self.x2, self.y2))), axis=0),
            axis=1)[0]
        if out is None:
            axm = signal.axes_manager.deepcopy()
            i0 = min(axes[0].index_in_array, axes[1].index_in_array)
            axm.remove([ax.index_in_array + 3j for ax in axes])
            axis = DataAxis(profile.shape[i0],
                            scale=length / profile.shape[i0],
                            units=axes[0].units,
                            navigate=axes[0].navigate)
            axis.axes_manager = axm
            axm._axes.insert(i0, axis)
            from hyperspy.signals import BaseSignal
            roi = BaseSignal(profile, axes=axm._get_axes_dicts(),
                             metadata=signal.metadata.deepcopy(
            ).as_dictionary(),
                original_metadata=signal.original_metadata.
                deepcopy().as_dictionary())
            return roi
        else:
            out.data = profile
            i0 = min(axes[0].index_in_array, axes[1].index_in_array)
            ax = out.axes_manager._axes[i0]
            size = len(profile)
            scale = length / len(profile)
            axchange = size != ax.size or scale != ax.scale
            if axchange:
                ax.size = len(profile)
                ax.scale = length / len(profile)
            out.events.data_changed.trigger(out)

    def __repr__(self):
        return "%s(x1=%g, y1=%g, x2=%g, y2=%g, linewidth=%g)" % (
            self.__class__.__name__,
            self.x1,
            self.y1,
            self.x2,
            self.y2,
            self.linewidth)
