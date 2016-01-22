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

import traits.api as t
import numpy as np

from hyperspy.events import Events, Event
from hyperspy.axes import DataAxis
from hyperspy.drawing import widgets


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

    def _make_slices(self, axes_collecion, axes, ranges=None):
        """
        Utility function to make a slice structure that will slice all the axes
        in 'axes_collecion'. The axes in the `axes` argument will be sliced by
        the ROI, all other axes with 'slice(None)'. Alternatively, if 'ranges'
        is passed, `axes[i]` will be sliced with 'ranges[i]'.
        """
        if ranges is None:
            # Use ROI to slice
            ranges = self._get_ranges()
        slices = []
        for ax in axes_collecion:
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
                        ihigh = 1 + ax.value2index(
                            ranges[i][1], rounding=lambda x: round(x - 1))
                    except ValueError:
                        if ranges[i][0] < ax.high_value:
                            ihigh = ax.high_index + 1
                        else:
                            raise
                    slices.append(slice(ilow, ihigh))
            else:
                slices.append(slice(None))
        return tuple(slices)

    def __call__(self, signal, axes=None):
        """Slice the signal according to the ROI, and return it.

        Arguments
        ---------
        signal : Signal
            The signal to slice with the ROI.
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
        if axes[0].navigate:
            if len(axes) == 2 and not axes[1].navigate:
                # Special case, since we can no longer slice axes in different
                # spaces together.
                return signal.inav[slices[0]].isig[slices[1]]
            slicer = signal.inav.__getitem__
            slices = slices[0:signal.axes_manager.navigation_dimension]
        else:
            slicer = signal.isig.__getitem__
            slices = slices[signal.axes_manager.navigation_dimension:]
        roi = slicer(slices)
        return roi

    def _parse_axes(self, axes, axes_manager):
        """Utility function to parse the 'axes' argument to a tuple of
        DataAxis, and find the matplotlib Axes that contains it.

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
        (tuple(<DataAxis>), matplotlib Axes)
        """
        nd = self.ndim
        axes_out = []
        if isinstance(axes, (tuple, list)):
            for i in xrange(nd):
                if isinstance(axes[i], DataAxis):
                    axes_out.append(axes[i])
                else:
                    axes_out.append(axes_manager[axes[i]])
        else:
            if axes_manager.navigation_dimension >= nd:
                axes_out = axes_manager.navigation_axes[:nd]
            elif axes_manager.signal_dimension >= nd:
                axes_out = axes_manager.signal_axes[:nd]
            elif nd == 2 and axes_manager.navigation_dimensions == 1 and \
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

    def update(self):
        """Function responsible for updating anything that depends on the ROI.
        It should be called by implementors whenever the ROI changes.
        This implementation  updates the widgets associated with it, and
        triggers the changed event.
        """
        if self.is_valid():
            self._update_widgets()
            self.events.changed.trigger(self)

    def _update_widgets(self, exclude=set()):
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

    def _on_widget_change(self, widget):
        """Callback for widgets' 'changed' event. Updates the internal state
        from the widget, and triggers events (excluding connections to the
        source widget).
        """
        with self.events.suppress():
            self._bounds_check = False
            try:
                self._set_from_widget(widget)
            finally:
                self._bounds_check = True
        self._update_widgets(exclude=(widget,))
        self.events.changed.trigger(self)

    def add_widget(self, signal, axes=None, widget=None, color='green'):
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
                * "navigation" or "signal", in which the first axes of that
                  space's axes will be used.
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
        """
        axes = self._parse_axes(axes, signal.axes_manager,)
        if widget is None:
            widget = self._get_widget_type(axes, signal)(signal.axes_manager)
            widget.color = color

        # Remove existing ROI, if it exsists and axes match
        if signal in self.signal_map and \
                self.signal_map[signal][1] == axes:
            self.remove_widget(signal)

        if axes is not None:
            # Set DataAxes
            widget.axes = axes
        with widget.events.changed.suppress_callback(self._on_widget_change):
            self._apply_roi2widget(widget)
        if widget.ax is None:
            ax = _get_mpl_ax(signal._plot, axes)
            widget.set_mpl_ax(ax)

        # Connect widget changes to on_widget_change
        widget.events.changed.connect(self._on_widget_change, 1)
        # When widget closes, remove from internal list
        widget.events.closed.connect(self._remove_widget, 1)
        self.widgets.add(widget)
        self.signal_map[signal] = (widget, axes)
        return widget

    def _remove_widget(self, widget):
        widget.events.closed.disconnect(self._remove_widget)
        widget.events.changed.disconnect(self._on_widget_change)
        widget.close()
        for signal, w in self.signal_map.iteritems():
            if w == widget:
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
    pass    # Only used for identification purposes currently


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
                return widgets.VerticalLine
            else:  # Axis is vertical
                return widgets.HorizontalLine
        elif plotdim == 1:  # It is a spectrum
            return widgets.VerticalLine
        else:
            raise ValueError("Could not find valid widget type")

    def __repr__(self):
        return "%s(value=%f)" % (
            self.__class__.__name__,
            self.value)


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
        return widgets.DraggableSquare

    def __repr__(self):
        return "%s(x=%f, y=%f)" % (
            self.__class__.__name__,
            self.x, self.y)


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
        return widgets.Range

    def __repr__(self):
        return "%s(left=%f, right=%f)" % (
            self.__class__.__name__,
            self.left,
            self.right)


class RectangularROI(BaseInteractiveROI):

    """Selects a range in a 2D space. The coordinates of the range in
    the 2D space are stored in the traits 'left', 'right', 'top' and 'bottom'.
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
        return widgets.Rectangle

    def __repr__(self):
        return "%s(left=%f, top=%f, right=%f, bottom=%f)" % (
            self.__class__.__name__,
            self.left,
            self.top,
            self.right,
            self.bottom)
