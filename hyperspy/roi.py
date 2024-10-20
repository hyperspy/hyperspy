# -*- coding: utf-8 -*-
# Copyright 2007-2024 The HyperSpy developers
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
# along with HyperSpy.  If not, see <https://www.gnu.org/licenses/#GPL>.

"""
Region of interests (ROIs) operate on :py:class:`~.api.signals.BaseSignal`
instances and include widgets for interactive operation.

The following 1D ROIs are available:

.. list-table:: 1D ROIs
   :widths: 25 75

   * - :class:`~.api.roi.Point1DROI`
     - Single element ROI of a 1D signal
   * - :class:`~.api.roi.SpanROI`
     - Interval ROI of a 1D signal

The following 2D ROIs are available:

.. list-table:: 2D ROIs
   :widths: 25 75

   * - :class:`~.api.roi.Point2DROI`
     - Single element ROI of a 2D signal
   * - :class:`~.api.roi.RectangularROI`
     - Rectagular ROI of a 2D signal
   * - :class:`~.api.roi.CircleROI`
     - (Hollow) circular ROI of a 2D signal
   * - :class:`~.api.roi.Line2DROI`
     - Line profile of a 2D signal with customisable width
   * - :class:`~api.roi.,PolygonROI`
     - Polygonal ROI with a customisable shape.

"""

from functools import partial

import numpy as np
import traits.api as t

import hyperspy.api as hs
from hyperspy.axes import UniformDataAxis
from hyperspy.drawing import widgets
from hyperspy.events import Event, Events
from hyperspy.interactive import interactive
from hyperspy.misc.utils import is_cupy_array
from hyperspy.ui_registry import add_gui_method

not_set_error_msg = (
    "Some ROI parameters have not yet been set. " "Set them before slicing a signal."
)


PARSE_AXES_DOCSTRING = """axes : None, str, int or :class:`hyperspy.axes.DataAxis`, default None
            The axes argument specifies which axes the ROI will be applied on.
            The axes in the collection can be either of the following:

            * Anything that can index the provided ``axes_manager``.
            * a tuple or list of:

              - :class:`hyperspy.axes.DataAxis`
              - anything that can index the provided ``axes_manager``

            * ``None``, it will check whether the widget can be added to the
              navigator, i.e. if dimensionality matches, and use it if
              possible, otherwise it will try the signal space. If none of the
              two attempts work, an error message will be raised.
"""


class BaseROI(t.HasTraits):
    """Base class for all ROIs.

    Provides some basic functionalities that are likely to be shared between all
    ROIs, and serve as a common type that can be checked for.

    Attributes
    ----------
    signal_map : dict
        Mapping of ``signal``:(``widget``, ``axes``) to keep track to the signals
        (and corresponding widget/signal axes) on which the ROI has been added.
        This dictionary is populated in :meth:`BaseInteractiveROI.add_widget`
    parameters : dict
        Mapping of parameters name and values for all parameters of the ROI.
    """

    def __init__(self):
        """Sets up events.changed event, and inits HasTraits."""
        super(BaseROI, self).__init__()
        self.events = Events()
        self.events.changed = Event(
            """
            Event that triggers when the ROI has changed.

            What constitues a change varies from ROI to ROI, but in general it
            should correspond to the region selected by the ROI being changed.

            Parameters
            ----------
            roi :
                The ROI that was changed.
            """,
            arguments=["roi"],
        )
        self.signal_map = dict()

    def __getitem__(self, *args, **kwargs):
        return tuple(self.parameters.values()).__getitem__(*args, **kwargs)

    def __repr__(self):
        para = []
        for name, value in self.parameters.items():
            if value is t.Undefined:
                para.append(f"{name}={value}")
            else:
                # otherwise format value with the General specifer
                para.append(f"{name}={value:G}")
        return f"{self.__class__.__name__}({', '.join(para)})"

    _ndim = 0
    ndim = property(lambda s: s._ndim)

    @property
    def parameters(self):
        raise NotImplementedError()

    def is_valid(self):
        """
        Determine if the ROI is in a valid state.

        This is typically determined by all the coordinates being defined,
        and that the values makes sense relative to each other.
        """
        return t.Undefined not in tuple(self)

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

        If the ROI is point base or is rectangular in nature, these can be used
        to slice a signal. Extracted from
        :meth:`~hyperspy.roi.BaseROI._make_slices` to ease implementation
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

        Parameters
        ----------
        signal : Signal
            The signal to slice with the ROI.
        out : Signal, default = None
            If the 'out' argument is supplied, the sliced output will be put
            into this instead of returning a Signal. See Signal.__getitem__()
            for more details on 'out'.
        %s
        """
        if not self.is_valid():
            raise ValueError(not_set_error_msg)
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

    __call__.__doc__ %= PARSE_AXES_DOCSTRING

    def _parse_axes(self, axes, axes_manager):
        """Utility function to parse the 'axes' argument to a list of
        :class:`~hyperspy.axes.DataAxis`.

        Parameters
        ----------
        %s
        axes_manager : :class:`~hyperspy.axes.AxesManager`
            The AxesManager to use for parsing axes

        Returns
        -------
        tuple of :class:`~hyperspy.axes.DataAxis`
        """
        nd = self.ndim
        if axes is None:
            if axes_manager.navigation_dimension >= nd:
                axes_out = axes_manager.navigation_axes[:nd]
            elif axes_manager.signal_dimension >= nd:
                axes_out = axes_manager.signal_axes[:nd]
            elif (
                nd == 2
                and axes_manager.navigation_dimension == 1
                and axes_manager.signal_dimension == 1
            ):
                # We probably have a navigator plot including both nav and sig
                # axes.
                axes_out = (
                    axes_manager.signal_axes[0],
                    axes_manager.navigation_axes[0],
                )
            else:
                raise ValueError("Could not find valid axes configuration.")
        else:
            if isinstance(axes, (tuple, list)) and len(axes) > nd:
                raise ValueError(
                    "The length of the provided `axes` is larger "
                    "than the dimensionality of the ROI."
                )
            axes_out = axes_manager[axes]

        if not isinstance(axes_out, tuple):
            axes_out = (axes_out,)

        return axes_out

    _parse_axes.__doc__ %= PARSE_AXES_DOCSTRING


def _get_mpl_ax(plot, axes):
    """
    Returns matplotlib Axes that contains the hyperspy axis.

    The space of the first DataAxis in axes will be used to determine which
    plot's :class:`matplotlib.axes.Axes` to return.

    Parameters
    ----------
    plot : MPL_HyperExplorer
        The explorer that contains the navigation and signal plots.
    axes : collection of DataAxis
        The axes to infer from.
    """
    if not plot.is_active:
        raise RuntimeError(
            "The signal needs to be plotted before using this " "function."
        )

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
    widgets, but inheritors need to implement the following functions:

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

        Parameters
        ----------
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

    def _set_default_values(self, signal, axes=None):
        """When the ROI is called interactively with Undefined parameters,
        use these values instead.
        """
        raise NotImplementedError()

    def _set_from_widget(self, widget):
        """Sets the internal representation of the ROI from the passed widget,
        without doing anything to events.
        """
        raise NotImplementedError()

    def interactive(
        self,
        signal,
        navigation_signal="same",
        out=None,
        color="green",
        snap=True,
        **kwargs,
    ):
        """Creates an interactively sliced Signal (sliced by this ROI) via
        :func:`~hyperspy.api.interactive`.

        Parameters
        ----------
        signal : hyperspy.api.signals.BaseSignal (or subclass)
            The source signal to slice.
        navigation_signal : hyperspy.api.signals.BaseSignal (or subclass), None or "same" (default)
            The signal the ROI will be added to, for navigation purposes
            only. Only the source signal will be sliced.
            If not None, it will automatically create a widget on
            navigation_signal. Passing ``"same"`` is identical to passing the
            same signal to ``"signal"`` and ``"navigation_signal"``, but is less
            ambigous, and allows "same" to be the default value.
        out : hyperspy.api.signals.BaseSignal (or subclass)
            If not None, it will use 'out' as the output instead of
            returning a new Signal.
        color : matplotlib color, default: ``'green'``
            The color for the widget. Any format that matplotlib uses should be
            ok. This will not change the color for any widget passed with the
            'widget' argument.
        snap : bool, default True
            If True, the ROI will be snapped to the axes values.
        **kwargs
            All kwargs are passed to the roi ``__call__`` method which is
            called interactively on any roi parameter change.

        Returns
        -------
        :class:`~hyperspy.api.signals.BaseSignal` (or subclass)
            Signal updated with the current ROI selection
            when the ROI is changed.

        """
        if hasattr(signal, "_plot_kwargs"):
            kwargs.update({"_plot_kwargs": signal._plot_kwargs})
            # in case of complex signal, it is possible to shift the signal
            # during plotting, if so this is currently not supported and we
            # raise a NotImplementedError
            if signal._plot.signal_data_function_kwargs.get("fft_shift", False):
                raise NotImplementedError(
                    "ROIs are not supported when data " "are shifted during plotting."
                )

        if isinstance(navigation_signal, str) and navigation_signal == "same":
            navigation_signal = signal
        if navigation_signal is not None:
            if navigation_signal not in self.signal_map:
                self.add_widget(
                    navigation_signal,
                    color=color,
                    snap=snap,
                    axes=kwargs.get("axes", None),
                )
        if self.update not in signal.axes_manager.events.any_axis_changed.connected:
            signal.axes_manager.events.any_axis_changed.connect(self.update, [])
        if out is None:
            return interactive(
                self.__call__, event=self.events.changed, signal=signal, **kwargs
            )
        else:
            return interactive(
                self.__call__,
                event=self.events.changed,
                signal=signal,
                out=out,
                **kwargs,
            )

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

    def add_widget(
        self, signal, axes=None, widget=None, color="green", snap=None, **kwargs
    ):
        """Add a widget to visually represent the ROI, and connect it so any
        changes in either are reflected in the other. Note that only one
        widget can be added per signal/axes combination.

        Parameters
        ----------
        signal : hyperspy.api.signals.BaseSignal (or subclass)
            The signal to which the widget is added. This is used to determine
            which plot to add the widget to, and it supplies the axes_manager
            for the widget.
        %s
        widget : hyperspy widget or None, default None
            If specified, this is the widget that will be added. If None, the
            default widget will be used.
        color : matplotlib color, default ``'green'``
            The color for the widget. Any format that matplotlib uses should be
            ok. This will not change the color for any widget passed with the
            ``'widget'`` argument.
        snap : bool or None, default None
            If True, the ROI will be snapped to the axes values, non-uniform
            axes are not supported. If None, it will be disabled (set to
            ``False``) for signals containing non-uniform axes.
        **kwargs : dict
            All keyword arguments are passed to the widget constructor.

        Returns
        -------
        hyperspy widget
            The widget of the ROI.
        """

        axes = self._parse_axes(
            axes,
            signal.axes_manager,
        )

        # Undefined if roi initialised without specifying parameters
        if t.Undefined in tuple(self):
            self._set_default_values(signal, axes=axes)

        if signal._plot is None or signal._plot.signal_plot is None:
            raise RuntimeError(
                f"{repr(signal)} does not have an active plot. Plot the "
                "signal before calling this method."
            )

        if widget is None:
            widget = self._get_widget_type(axes, signal)(signal.axes_manager, **kwargs)
            widget.color = color

        # Remove existing ROI, if it exists and axes match
        if signal in self.signal_map and self.signal_map[signal][1] == axes:
            self.remove_widget(signal)

        if widget.ax is None:
            ax = _get_mpl_ax(signal._plot, axes)
            widget.set_mpl_ax(ax)

        # Set DataAxes
        widget.axes = axes
        with widget.events.changed.suppress_callback(self._on_widget_change):
            self._apply_roi2widget(widget)

            if snap is None:
                if any(not axis.is_uniform for axis in axes):
                    # Disable snapping for non-uniform axes
                    snap = False
                else:
                    snap = True

            # We need to snap after the widget value have been set
            if hasattr(widget, "snap_all"):
                widget.snap_all = snap
            else:
                widget.snap_position = snap

        # Connect widget changes to on_widget_change
        widget.events.changed.connect(self._on_widget_change, {"obj": "widget"})
        # When widget closes, remove from internal list
        widget.events.closed.connect(self._remove_widget, {"obj": "widget"})
        self.widgets.add(widget)
        self.signal_map[signal] = (widget, axes)
        return widget

    add_widget.__doc__ %= PARSE_AXES_DOCSTRING

    def _remove_widget(self, widget, render_figure=True):
        widget.events.closed.disconnect(self._remove_widget)
        widget.events.changed.disconnect(self._on_widget_change)
        widget.close(render_figure=render_figure)
        for signal, w in self.signal_map.items():
            if w[0] == widget:
                self.signal_map.pop(signal)
                break
            # disconnect events which has been added when
            if self.update in signal.axes_manager.events.any_axis_changed.connected:
                signal.axes_manager.events.any_axis_changed.disconnect(self.update)

    def remove_widget(self, signal=None, render_figure=True):
        """
        Removing a widget from a signal consists of two tasks:

        1. Disconnect the interactive operations associated with this ROI
           and the specified signal ``signal``.
        2. Removing the widget from the plot.

        Parameters
        ----------
        signal : hyperspy.api.signals.BaseSignal (or subclass)
            The signal from which the interactive operations will be
            disconnected. If None, remove from all signals.
        render_figure : bool, default True
            If False, the figure will not be rendered after removing the widget
            in order to save redraw events.

        """
        if signal is None:
            signal = list(self.signal_map.keys())
        elif isinstance(signal, hs.signals.BaseSignal):
            signal = [signal]

        for s in signal:
            if s in self.signal_map:
                w = self.signal_map.pop(s)[0]
                self._remove_widget(w, render_figure)


class BasePointROI(BaseInteractiveROI):
    """Base ROI class for point ROIs, i.e. ROIs with a unit size in each of its
    dimensions.
    """

    def __call__(self, signal, out=None, axes=None):
        if axes is None and signal in self.signal_map:
            axes = self.signal_map[signal][1]
        else:
            axes = self._parse_axes(axes, signal.axes_manager)
        s = super(BasePointROI, self).__call__(signal=signal, out=out, axes=axes)
        return s


def guess_vertical_or_horizontal(axes, signal):
    # Figure out whether to use horizontal or vertical line:
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
        if idx == 0 and axdim != 1:  # Axis is horizontal
            return "vertical"
        else:  # Axis is vertical
            return "horizontal"
    elif plotdim == 1:  # It is a spectrum
        return "vertical"
    else:
        raise ValueError("Could not find valid widget type for the given `axes` value")


@add_gui_method(toolkey="hyperspy.Point1DROI")
class Point1DROI(BasePointROI):
    """Selects a single point in a 1D space. The coordinate of the point in the
    1D space is stored in the 'value' trait.

    ``Point1DROI`` can be used in place of a tuple containing the value of ``value``.


    Examples
    --------

    >>> roi = hs.roi.Point1DROI(0.5)
    >>> value, = roi
    >>> print(value)
    0.5

    """

    value = t.CFloat(t.Undefined)
    _ndim = 1

    def __init__(self, value=None):
        super().__init__()
        value = value if value is not None else t.Undefined
        self.value = value

    def _set_default_values(self, signal, axes=None):
        if axes is None:
            axes = self._parse_axes(None, signal.axes_manager)
        # If roi parameters are undefined, use center of axes
        self.value = axes[0]._parse_value("rel0.5")

    @property
    def parameters(self):
        return {"value": self.value}

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


@add_gui_method(toolkey="hyperspy.Point2DROI")
class Point2DROI(BasePointROI):
    """Selects a single point in a 2D space. The coordinates of the point in
    the 2D space are stored in the traits ``'x'`` and ``'y'``.

    ``Point2DROI`` can be used in place of a tuple containing the coordinates
    of the point (x, y).


    Examples
    --------

    >>> roi = hs.roi.Point2DROI(3, 5)
    >>> x, y = roi
    >>> print(x, y)
    3.0 5.0

    """

    x, y = (t.CFloat(t.Undefined),) * 2
    _ndim = 2

    def __init__(self, x=None, y=None):
        super().__init__()
        x, y = (para if para is not None else t.Undefined for para in (x, y))

        self.x, self.y = x, y

    def _set_default_values(self, signal, axes=None):
        if axes is None:
            axes = self._parse_axes(None, signal.axes_manager)
        # If roi parameters are undefined, use center of axes
        self.x = axes[0]._parse_value("rel0.5")
        self.y = axes[1]._parse_value("rel0.5")

    @property
    def parameters(self):
        return {"x": self.x, "y": self.y}

    def _x_changed(self, old, new):
        self.update()

    def _y_changed(self, old, new):
        self.update()

    def _get_ranges(self):
        ranges = (
            (self.x,),
            (self.y,),
        )
        return ranges

    def _set_from_widget(self, widget):
        self.x, self.y = widget.position

    def _apply_roi2widget(self, widget):
        widget.position = (self.x, self.y)

    def _get_widget_type(self, axes, signal):
        return widgets.SquareWidget


@add_gui_method(toolkey="hyperspy.SpanROI")
class SpanROI(BaseInteractiveROI):
    """Selects a range in a 1D space. The coordinates of the range in
    the 1D space are stored in the traits ``'left'`` and ``'right'``.

    ``SpanROI`` can be used in place of a tuple containing the left and right values.

    Examples
    --------

    >>> roi = hs.roi.SpanROI(-3, 5)
    >>> left, right = roi
    >>> print(left, right)
    -3.0 5.0

    """

    left, right = (t.CFloat(t.Undefined),) * 2
    _ndim = 1

    def __init__(self, left=None, right=None):
        super().__init__()
        self._bounds_check = True  # Use responsibly!
        if left is not None and right is not None and left >= right:
            raise ValueError(f"`left` ({left}) must be smaller than `right` ({right}).")
        left, right = (
            para if para is not None else t.Undefined for para in (left, right)
        )
        self.left, self.right = left, right

    def _set_default_values(self, signal, axes=None):
        if axes is None:
            axes = self._parse_axes(None, signal.axes_manager)
        # If roi parameters are undefined, use center of axes
        self.left, self.right = _get_central_half_limits_of_axis(axes[0])

    @property
    def parameters(self):
        return {"left": self.left, "right": self.right}

    def is_valid(self):
        return t.Undefined not in tuple(self) and self.right >= self.left

    def _right_changed(self, old, new):
        if self._bounds_check and self.left is not t.Undefined and new <= self.left:
            self.right = old
        else:
            self.update()

    def _left_changed(self, old, new):
        if self._bounds_check and self.right is not t.Undefined and new >= self.right:
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
        if widget.span is not None:
            widget._set_span_extents(self.left, self.right)

    def _get_widget_type(self, axes, signal):
        direction = guess_vertical_or_horizontal(axes=axes, signal=signal)
        if direction == "vertical":
            return partial(widgets.RangeWidget, direction="horizontal")
        elif direction == "horizontal":
            return partial(widgets.RangeWidget, direction="vertical")
        else:
            raise ValueError("direction must be either horizontal or vertical")


@add_gui_method(toolkey="hyperspy.RectangularROI")
class RectangularROI(BaseInteractiveROI):
    """Selects a range in a 2D space. The coordinates of the range in
    the 2D space are stored in the traits ``'left'``, ``'right'``, ``'top'`` and ``'bottom'``.
    Convenience properties ``'x'``, ``'y'``, ``'width'`` and ``'height'`` are also available,
    but cannot be used for initialization.

    ``RectangularROI`` can be used in place of a tuple containing (left, right, top, bottom).

    Examples
    --------

    >>> roi = hs.roi.RectangularROI(left=0, right=10, top=20, bottom=20.5)
    >>> left, right, top, bottom = roi
    >>> print(left, right, top, bottom)
    0.0 10.0 20.0 20.5
    """

    top, bottom, left, right = (t.CFloat(t.Undefined),) * 4
    _ndim = 2

    def __init__(self, left=None, top=None, right=None, bottom=None):
        super(RectangularROI, self).__init__()
        left, top, right, bottom = (
            para if para is not None else t.Undefined
            for para in (left, top, right, bottom)
        )
        self._bounds_check = True  # Use reponsibly!
        self.left, self.top, self.right, self.bottom = left, top, right, bottom

    def __getitem__(self, *args, **kwargs):
        # Note: RectangularROI is currently indexed in a different way
        # than it is initialised. This should be fixed properly in a PR.
        _tuple = (self.left, self.right, self.top, self.bottom)
        return _tuple.__getitem__(*args, **kwargs)

    def _set_default_values(self, signal, axes=None):
        # Need to turn of bounds checking or undefined values trigger error
        old_bounds_check = self._bounds_check
        self._bounds_check = False
        if axes is None:
            axes = self._parse_axes(None, signal.axes_manager)

        # If roi parameters are undefined, use center of axes
        self.left, self.right = _get_central_half_limits_of_axis(axes[0])
        self.top, self.bottom = _get_central_half_limits_of_axis(axes[1])
        self._bounds_check = old_bounds_check

    @property
    def parameters(self):
        return {
            "left": self.left,
            "top": self.top,
            "right": self.right,
            "bottom": self.bottom,
        }

    def is_valid(self):
        return (
            t.Undefined not in tuple(self)
            and self.right >= self.left
            and self.bottom >= self.top
        )

    def _top_changed(self, old, new):
        if self._bounds_check and self.bottom is not t.Undefined and new >= self.bottom:
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
        if self._bounds_check and self.top is not t.Undefined and new <= self.top:
            self.bottom = old
        else:
            self.update()

    def _right_changed(self, old, new):
        if self._bounds_check and self.left is not t.Undefined and new <= self.left:
            self.right = old
        else:
            self.update()

    def _left_changed(self, old, new):
        if self._bounds_check and self.right is not t.Undefined and new >= self.right:
            self.left = old
        else:
            self.update()

    def _get_ranges(self):
        ranges = (
            (self.left, self.right),
            (self.top, self.bottom),
        )
        return ranges

    def _set_from_widget(self, widget):
        p = np.array(widget.position)
        s = np.array(widget.size)
        (self.left, self.top), (self.right, self.bottom) = (p, p + s)

    def _apply_roi2widget(self, widget):
        widget.set_bounds(
            left=self.left, bottom=self.bottom, right=self.right, top=self.top
        )

    def _get_widget_type(self, axes, signal):
        return widgets.RectangleWidget


@add_gui_method(toolkey="hyperspy.CircleROI")
class CircleROI(BaseInteractiveROI):
    """Selects a circular or annular region in a 2D space. The coordinates of
    the center of the circle are stored in the 'cx' and 'cy' attributes. The
    radius in the `r` attribute. If an internal radius is defined using the
    `r_inner` attribute, then an annular region is selected instead.
    `CircleROI` can be used in place of a tuple containing `(cx, cy, r)`, `(cx,
    cy, r, r_inner)` when `r_inner` is not `None`.
    """

    cx, cy, r, r_inner = (t.CFloat(t.Undefined),) * 3 + (t.CFloat(0.0),)
    _ndim = 2

    def __init__(self, cx=None, cy=None, r=None, r_inner=0):
        super(CircleROI, self).__init__()
        cx, cy, r = (para if para is not None else t.Undefined for para in (cx, cy, r))

        self._bounds_check = True  # Use reponsibly!
        self.cx, self.cy, self.r, self.r_inner = cx, cy, r, r_inner

    def _set_default_values(self, signal, axes=None):
        if axes is None:
            axes = self._parse_axes(None, signal.axes_manager)
        ax0, ax1 = axes

        # If roi parameters are undefined, use center of axes
        self.cx = ax0._parse_value("rel0.5")
        self.cy = ax1._parse_value("rel0.5")

        rx = (ax0.high_value - ax0.low_value) / 2
        ry = (ax1.high_value - ax1.low_value) / 2
        self.r = min(rx, ry)

    @property
    def parameters(self):
        return {"cx": self.cx, "cy": self.cy, "r": self.r, "r_inner": self.r_inner}

    def is_valid(self):
        return t.Undefined not in tuple(self) and self.r >= self.r_inner

    def _cx_changed(self, old, new):
        self.update()

    def _cy_changed(self, old, new):
        self.update()

    def _r_changed(self, old, new):
        if self._bounds_check and new < self.r_inner:
            self.r = old
        else:
            self.update()

    def _r_inner_changed(self, old, new):
        if self._bounds_check and self.r is not t.Undefined and new >= self.r:
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
        widget.size = (self.r, self.r_inner)

    def _get_widget_type(self, axes, signal):
        return widgets.CircleWidget

    def __call__(self, signal, out=None, axes=None):
        if not self.is_valid():
            raise ValueError(not_set_error_msg)

        if axes is None and signal in self.signal_map:
            axes = self.signal_map[signal][1]
        else:
            axes = self._parse_axes(axes, signal.axes_manager)

        for axis in axes:
            if not axis.is_uniform:
                raise NotImplementedError(
                    "This ROI cannot operate on a non-uniform axis."
                )
        natax = signal.axes_manager._get_axes_in_natural_order()
        # Slice original data with a circumscribed rectangle
        cx = self.cx + 0.5001 * axes[0].scale
        cy = self.cy + 0.5001 * axes[1].scale
        ranges = [[cx - self.r, cx + self.r], [cy - self.r, cy + self.r]]
        slices = self._make_slices(natax, axes, ranges)
        ir = [slices[natax.index(axes[0])], slices[natax.index(axes[1])]]

        vx = axes[0].axis[ir[0]] - cx
        vy = axes[1].axis[ir[1]] - cy

        # convert to cupy array when necessary
        if is_cupy_array(signal.data):
            import cupy as cp

            vx, vy = cp.array(vx), cp.array(vy)

        gx, gy = np.meshgrid(vx, vy)
        gr = gx**2 + gy**2
        mask = gr > self.r**2
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
        mask = np.broadcast_to(mask, tiles)
        # roi.data = np.ma.masked_array(roi.data, mask, hard_mask=True)
        roi.data = np.where(mask, np.nan, roi.data)
        if out is None:
            return roi
        else:
            out.events.data_changed.trigger(out)


@add_gui_method(toolkey="hyperspy.Line2DROI")
class Line2DROI(BaseInteractiveROI):
    """Selects a line of a given width in 2D space. The coordinates of the end points of the line are stored in the `x1`, `y1`, `x2`, `y2` parameters.
    The length is available in the `length` parameter and the method `angle` computes the angle of the line with the axes.

    `Line2DROI` can be used in place of a tuple containing the coordinates of the two end-points of the line and the linewdith `(x1, y1, x2, y2, linewidth)`.
    """

    x1, y1, x2, y2, linewidth = (t.CFloat(t.Undefined),) * 4 + (t.CFloat(0.0),)
    _ndim = 2

    def __init__(self, x1=None, y1=None, x2=None, y2=None, linewidth=0):
        super().__init__()
        x1, y1, x2, y2 = (
            para if para is not None else t.Undefined for para in (x1, y1, x2, y2)
        )

        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.linewidth = linewidth

    def _set_default_values(self, signal, axes=None):
        if axes is None:
            axes = self._parse_axes(None, signal.axes_manager)
        # If roi parameters are undefined, use center of axes
        self.x1, self.x2 = _get_central_half_limits_of_axis(axes[0])
        self.y1, self.y2 = _get_central_half_limits_of_axis(axes[1])

    @property
    def parameters(self):
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "linewidth": self.linewidth,
        }

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
        p0 = np.array((self.x1, self.y1), dtype=float)
        p1 = np.array((self.x2, self.y2), dtype=float)
        d_row, d_col = p1 - p0
        return np.hypot(d_row, d_col)

    def angle(self, axis="horizontal", units="degrees"):
        """ "Angle between ROI line and selected axis

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
        >>> r = hs.roi.Line2DROI(0., 0., 1., 2.)
        >>> print(r.angle())
        63.43494882292201
        """

        x = self.x2 - self.x1
        y = self.y2 - self.y1

        if units == "degrees":
            conversation = 180.0 / np.pi
        elif units == "radians":
            conversation = 1.0
        else:
            raise ValueError(
                "Units are not recognized. Use  either 'degrees' or 'radians'."
            )

        if axis == "horizontal":
            return np.arctan2(y, x) * conversation
        elif axis == "vertical":
            return np.arctan2(x, y) * conversation
        else:
            raise ValueError(
                "Axis is not recognized. " "Use  either 'horizontal' or 'vertical'."
            )

    @staticmethod
    def profile_line(
        img, src, dst, axes, linewidth=1, order=1, mode="constant", cval=0.0
    ):
        """Return the intensity profile of an image measured along a scan line.

        Parameters
        ----------
        img : numpy.ndarray
            The image, either grayscale (2D array) or multichannel
            (3D array, where the final axis contains the channel
            information).
        src : tuple of float, tuple of int
            The start point of the scan line. Length of tuple is 2.
        dst : tuple of float, tuple of int
            The end point of the scan line. Length of tuple is 2.
        linewidth : int, optional
            Width of the scan, perpendicular to the line
        order : {0, 1, 2, 3, 4, 5}, optional
            The order of the spline interpolation to compute image values at
            non-integer coordinates. 0 means nearest-neighbor interpolation.
        mode : {'constant', 'nearest', 'reflect', 'wrap'}, optional
            How to compute any values falling outside of the image.
        cval : float, optional
            If ``mode='constant'``, what constant value to use outside the
            image.

        Returns
        -------
        numpy.ndarray
            The intensity profile along the scan line. The length of the
            profile is the ceil of the computed length of the scan line.

        Notes
        -----
        The destination point is included in the profile, in contrast to
        standard numpy indexing. Requires uniform navigation axes.

        """
        for axis in axes:
            if not axis.is_uniform:
                raise NotImplementedError(
                    "Line profiles on data with non-uniform axes is not implemented."
                )

        import scipy.ndimage as nd

        # Convert points coordinates from axes units to pixels
        p0 = (
            (src[0] - axes[0].offset) / axes[0].scale,
            (src[1] - axes[1].offset) / axes[1].scale,
        )
        p1 = (
            (dst[0] - axes[0].offset) / axes[0].scale,
            (dst[1] - axes[1].offset) / axes[1].scale,
        )
        if linewidth < 0:
            raise ValueError("linewidth must be positive number")
        linewidth_px = linewidth / np.min([ax.scale for ax in axes])
        linewidth_px = int(round(linewidth_px))
        # Minimum size 1 pixel
        linewidth_px = linewidth_px if linewidth_px >= 1 else 1
        perp_lines = Line2DROI._line_profile_coordinates(p0, p1, linewidth=linewidth_px)
        if img.ndim > 2:
            idx = [ax.index_in_array for ax in axes]
            if idx[0] < idx[1]:
                img = np.rollaxis(img, idx[0], 0)
                img = np.rollaxis(img, idx[1], 1)
            else:
                img = np.rollaxis(img, idx[1], 0)
                img = np.rollaxis(img, idx[0], 0)
            orig_shape = img.shape
            img = np.reshape(img, orig_shape[0:2] + (np.prod(orig_shape[2:]),))
            pixels = [
                nd.map_coordinates(
                    img[..., i].T, perp_lines, order=order, mode=mode, cval=cval
                )
                for i in range(img.shape[2])
            ]
            i0 = min(axes[0].index_in_array, axes[1].index_in_array)
            pixels = np.transpose(np.asarray(pixels), (1, 2, 0))
            intensities = pixels.mean(axis=1)
            intensities = np.rollaxis(
                np.reshape(intensities, intensities.shape[0:1] + orig_shape[2:]),
                0,
                i0 + 1,
            )
        else:
            pixels = nd.map_coordinates(
                img, perp_lines, order=order, mode=mode, cval=cval
            )
            intensities = pixels.mean(axis=1)

        return intensities

    def __call__(self, signal, out=None, axes=None, order=0):
        """Slice the signal according to the ROI, and return it.

        Parameters
        ----------
        signal : Signal
            The signal to slice with the ROI.
        out : Signal, default = None
            If the 'out' argument is supplied, the sliced output will be put
            into this instead of returning a Signal. See Signal.__getitem__()
            for more details on 'out'.
        %s
        order : The spline interpolation order to use when extracting the line
            profile. 0 means nearest-neighbor interpolation, and is both the
            default and the fastest.
        """
        if not self.is_valid():
            raise ValueError(not_set_error_msg)
        if axes is None and signal in self.signal_map:
            axes = self.signal_map[signal][1]
        else:
            axes = self._parse_axes(axes, signal.axes_manager)
        profile = Line2DROI.profile_line(
            signal.data,
            (self.x1, self.y1),
            (self.x2, self.y2),
            axes=axes,
            linewidth=self.linewidth,
            order=order,
        )
        length = np.linalg.norm(
            np.diff(np.array(((self.x1, self.y1), (self.x2, self.y2))), axis=0), axis=1
        )[0]
        if out is None:
            axm = signal.axes_manager.deepcopy()
            i0 = min(axes[0].index_in_array, axes[1].index_in_array)
            axm.remove([ax.index_in_array + 3j for ax in axes])
            axis = UniformDataAxis(
                size=profile.shape[i0],
                scale=length / profile.shape[i0],
                units=axes[0].units,
                navigate=axes[0].navigate,
            )
            axis.axes_manager = axm
            axm._axes.insert(i0, axis)
            from hyperspy.signals import BaseSignal

            roi = BaseSignal(
                profile,
                axes=axm._get_axes_dicts(),
                metadata=signal.metadata.deepcopy().as_dictionary(),
                original_metadata=signal.original_metadata.deepcopy().as_dictionary(),
            )
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


class PolygonROI(BaseInteractiveROI):
    """Selects a polygonal region in a 2D space. The coordinates of the
    polygon vertices are given in the `vertices` attribute, which is a
    list where each entry is a tuple (x, y) of a vertex's coordinates.
    An edge runs from the final to the first point in the list.
    If a polygon self overlaps, the overlapping areas may be considered
    outside of the polygon and masked away.
    """

    _vertices = []
    _ndim = 2

    def __init__(self, vertices=None):
        """
        Parameters
        ----------
        vertices : list of tuples
            List containing (x, y) values of the vertices of a polygon."""
        super().__init__()

        if vertices:
            self.vertices = vertices

    @property
    def parameters(self):
        return {"vertices": self.vertices}

    def __getitem__(self, *args, **kwargs):
        _tuple = tuple(self._vertices)
        return _tuple.__getitem__(*args, **kwargs)

    def __repr__(self):
        para = []
        for name, value in self.parameters.items():
            parastr = f"{name}=["
            parastr += ", ".join(f"({x:G}, {y:G})" for x, y in value)
            parastr += "]"
            para.append(parastr)
        return f"{self.__class__.__name__}({', '.join(para)})"

    def is_valid(self):
        """
        The polygon is defined as valid if either zero or more than two
        vertices are fully defined.
        """
        valid = False
        try:
            valid = (
                len(self._vertices) == 0
                or len(self._vertices) > 2
                and all(
                    (None not in vertex and len(vertex) == 2)
                    for vertex in self._vertices
                )
            )
        finally:
            return valid

    @property
    def vertices(self):
        """Returns a list where each entry contains a `(x, y)` tuple
        of the vertices of the polygon. The polygon is not closed.
        Returns an empty list if no polygon is set."""

        return self._vertices.copy()

    @vertices.setter
    def vertices(self, vertices):
        """Sets the vertices of the polygon to the `vertices` argument,
        where each vertex is to be given as a tuple `(x, y)` where `x`
        and `y` are its coordinates. The list is set to loop around,
        such that an edge runs from the final to the first vertex in
        the list.

        Parameters
        ----------
        vertices : list of tuples
            List of (x, y) values of the vertices of the polygon.
        """
        old_vertices = self._vertices
        self._vertices = vertices
        if not self.is_valid():
            self._vertices = old_vertices
            raise ValueError(
                "`vertices` is not an empty list or a list of fully defined two-dimensional "
                + f"points with at least three entries:\n{vertices}"
            )
        if self.widgets:
            self._update_widgets()

    def _apply_roi(
        self, signal, inverted=False, out=None, axes=None, additional_polygons=None
    ):
        if not self.is_valid():
            raise ValueError(not_set_error_msg)

        if axes is None and signal in self.signal_map:
            axes = self.signal_map[signal][1]
        else:
            axes = self._parse_axes(axes, signal.axes_manager)

        for axis in axes:
            if not axis.is_uniform:
                raise NotImplementedError(
                    "This ROI cannot operate on a non-uniform axis."
                )
        natax = signal.axes_manager._get_axes_in_natural_order()
        if not inverted and (
            self._vertices  # Make sure at least one polygon is not empty
            or (
                additional_polygons is not None
                and any(polygon for polygon in additional_polygons)
            )
        ):
            # Slice original data with a circumscribed rectangle

            polygons = [self._vertices]
            # In case of combining multiple PolygonROI, all vertices must be considered
            if additional_polygons is not None:
                polygons += [polygon for polygon in additional_polygons]

            left = min(x for polygon in polygons for x, y in polygon)
            right = max(x for polygon in polygons for x, y in polygon) + axes[1].scale
            top = min(y for polygon in polygons for x, y in polygon)
            bottom = max(y for polygon in polygons for x, y in polygon) + axes[0].scale
        else:
            # Do not slice if selection is to be inverted or is empty
            left, right = axes[0].low_value, axes[0].high_value + axes[0].scale
            top, bottom = axes[1].low_value, axes[1].high_value + axes[1].scale

        ranges = [[left, right], [top, bottom]]
        slices = self._make_slices(natax, axes, ranges)
        ir = [slices[natax.index(axes[0])], slices[natax.index(axes[1])]]

        mask = self._boolean_mask(
            axes=axes, xy_max=(right, bottom), additional_polygons=additional_polygons
        )

        mask = mask[ir[1], ir[0]]
        if not inverted:
            mask = np.logical_not(mask)  # Masked out areas should be True

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
            # slices = slices[::-1] # Slicing in signal axes is reversed

        roi = slicer(slices, out=out)
        roi = out or roi
        if roi._lazy:
            import dask.array as da

            mask = da.from_array(mask, chunks=chunks)
        mask = np.broadcast_to(mask, tiles)
        # roi.data = np.ma.masked_array(roi.data, mask, hard_mask=True)
        roi.data = np.where(mask, np.nan, roi.data)
        if out is None:
            return roi
        else:
            out.events.data_changed.trigger(out)

    def __call__(self, signal, inverted=False, out=None, axes=None):
        return self._apply_roi(signal, inverted=inverted, out=out, axes=axes)

    def _combine(
        self, signal, inverted=False, out=None, axes=None, additional_polygons=None
    ):
        return self._apply_roi(
            signal,
            inverted=inverted,
            out=out,
            axes=axes,
            additional_polygons=additional_polygons,
        )

    def _rasterized_mask(
        self, polygon_vertices, xy_max=None, xy_min=None, x_scale=None, y_scale=None
    ):
        """Utility function to rasterize a polygon into a boolean numpy
            array. The interior of the polygon is `True`.

        Parameters
        ----------
        xy_max : tuple, optional
            The maximum x and y values that the rasterized mask covers, given in
            the same coordinate space as the polygon vertices. The defaults are
            the max values in the `vertices` member attribute.
        xy_min : tuple, optional
            The minimum x and y values that the mask covers, given in the same coordinate
            space as the polygon vertices.
        x_scale : float, optional
            This gives the scale of the second axis of the signal. In other words,
            how many units in coordinate space that corresponds to one step between
            columns.
        y_scale : float
            This gives the scale of the second axis of the signal. In other words,
            how many units in coordinate space that corresponds to one step between
            rows.

        Returns
        -------
        return_value : array
            boolean numpy array of the rasterized polygon. The entire or parts of
            the polygon ROI can be within this shape.
        """

        if not xy_max:
            x_max = max(x for x, y in polygon_vertices)
            y_max = max(y for x, y in polygon_vertices)
            xy_max = (x_max, y_max)

        xy_min = xy_min if xy_min is not None else (0, 0)
        x_scale = x_scale if x_scale is not None else 1.0
        y_scale = y_scale if y_scale is not None else 1.0

        min_index_x = round(xy_min[0] / x_scale)
        min_index_y = round(xy_min[1] / y_scale)
        max_index_x = round(xy_max[0] / x_scale)
        max_index_y = round(xy_max[1] / y_scale)

        mask = np.zeros(
            (max_index_y - min_index_y + 1, max_index_x - min_index_x + 1), dtype=bool
        )

        for row in range(mask.shape[0]):
            row_y = (min_index_y + row) * y_scale

            intersections = []
            for vertexind in range(len(polygon_vertices)):
                x1, y1 = polygon_vertices[vertexind]
                x2, y2 = polygon_vertices[(vertexind + 1) % len(polygon_vertices)]

                # Only find intersection if line segment passes through row
                if (y1 > row_y) != (y2 > row_y):
                    intersection = x1 + (x2 - x1) * (row_y - y1) / (y2 - y1)
                    intersections.append(intersection)
                elif y1 == row_y:
                    if y1 == y2:
                        # Ensures edges parallel with row_y are included
                        x_start = min(x1, x2)
                        x_end = max(x1, x2)
                        raster_start = round(x_start / x_scale) - min_index_x
                        raster_end = round(x_end / x_scale) - min_index_x
                        mask[row, raster_start : raster_end + 1] = True
                    else:
                        # Ensures vertices landing exactly on row_y are included
                        mask[row, round(x1 / x_scale) - min_index_x] = True

            intersections.sort()

            for i in range(1, len(intersections), 2):
                raster_start = round(intersections[i - 1] / x_scale) - min_index_x
                raster_end = round(intersections[i] / x_scale) - min_index_x
                mask[row, raster_start : raster_end + 1] = True

        return mask

    def _boolean_mask(
        self,
        axes_manager=None,
        axes=None,
        xy_max=None,
        xy_min=None,
        x_scale=None,
        y_scale=None,
        additional_polygons=None,
    ):
        """Function to rasterize the polygon into a boolean numpy array. The
            interior of the polygon is by default `True`.

        Parameters
        ----------
        axes_manager : :py:class:`~hyperspy.axes.AxesManager`, optional
            If supplied, the rasterization parameters not explicitly given will be
            extracted from this. The axes can be given with the `axes` attribute.
        axes : list, optional
            List of the axes in `axes_manager` that are to be used as a basis for
            the rasterization if other parameters aren't supplied.
        xy_max : tuple, optional
            The maximum x and y values that the rasterized mask covers, given in
            the same coordinate space as the polygon vertices. The defaults are
            the max values in the `vertices` member attribute.
        xy_min : tuple, optional
            The minimum x and y values that the mask covers, given in the same coordinate
            space as the polygon vertices.
        x_scale : float, optional
            This gives the scale of the second axis of the signal. In other words,
            how many units in coordinate space that corresponds to one step between
            columns.
        y_scale : float, optional
            This gives the scale of the second axis of the signal. In other words,
            how many units in coordinate space that corresponds to one step between
            rows.
        additional_polygons : list, optional
            List containing further polygons to be added to the mask, resulting
            in a raster of several polygons. Each inner list gives the vertices
            of one individual polygon. These do not need to be closed.

        Returns
        -------
        return_value : array
            boolean numpy array of the rasterized polygon. The entire or parts of
            the polygon ROI can be within this shape.
        """

        # Get missing default values from `axes_manager` or `axes`
        if axes_manager or axes:
            axes = axes if not axes_manager else self._parse_axes(axes, axes_manager)

            xy_max = (
                xy_max
                if xy_max is not None
                else (axes[0].high_value, axes[1].high_value)
            )
            xy_min = (
                xy_min if xy_min is not None else (axes[0].low_value, axes[1].low_value)
            )

            x_scale = x_scale if x_scale is not None else axes[0].scale
            y_scale = y_scale if y_scale is not None else axes[1].scale

        # Empty ROI
        if not self._vertices:
            mask = None
            if xy_max:
                xy_min = xy_min if xy_min is not None else (0, 0)

                min_index_x = round(xy_min[0] / x_scale)
                min_index_y = round(xy_min[1] / y_scale)
                max_index_x = round(xy_max[0] / x_scale)
                max_index_y = round(xy_max[1] / y_scale)

                mask = np.zeros(
                    (max_index_y - min_index_y + 1, max_index_x - min_index_x + 1),
                    dtype=bool,
                )

        else:
            mask = self._rasterized_mask(
                polygon_vertices=self._vertices,
                xy_max=xy_max,
                xy_min=xy_min,
                x_scale=x_scale,
                y_scale=y_scale,
            )

        if additional_polygons is not None:
            for polygon in additional_polygons:
                if self._vertices is polygon:
                    continue

                other_mask = self._rasterized_mask(
                    polygon_vertices=polygon,
                    xy_max=xy_max,
                    xy_min=xy_min,
                    x_scale=x_scale,
                    y_scale=y_scale,
                )
                if mask is None:
                    mask = other_mask
                else:
                    # Expand mask to encompass both if needed
                    if (
                        other_mask.shape[0] > mask.shape[0]
                        or other_mask.shape[1] > mask.shape[1]
                    ):
                        expanded_mask = np.zeros(
                            np.max((mask.shape, other_mask.shape), axis=0), dtype=bool
                        )
                        expanded_mask[: mask.shape[0], : mask.shape[1]] = mask
                        mask = expanded_mask

                    mask[: other_mask.shape[0], : other_mask.shape[1]] |= other_mask

        return mask

    def _get_widget_type(self, axes, signal):
        return widgets.PolygonWidget

    def _apply_roi2widget(self, widget):
        """This function is responsible for applying the ROI geometry to the
        widget. When this function is called, the widget's events are already
        suppressed, so this should not be necessary for _apply_roi2widget to
        handle.
        """
        widget.set_vertices(self._vertices)

    def _set_default_values(self, signal, axes=None):
        """When the ROI is called interactively with Undefined parameters,
        use these values instead.
        """
        self.vertices = []

    def _set_from_widget(self, widget):
        """Sets the internal representation of the ROI from the passed widget,
        without doing anything to events.
        """
        self.vertices = widget.get_vertices()


def _get_central_half_limits_of_axis(ax):
    "Return indices of the central half of a DataAxis"
    return ax._parse_value("rel0.25"), ax._parse_value("rel0.75")


def combine_rois(signal, rois, inverted=False, out=None, axes=None):
    """Slice the signal according by combining a list of ROIs, by default
        returning a sliced copy.
        Currently only implemented for a list of `PolygonROI`s.

    Parameters
    ----------
    signal : Signal
        The signal to slice with the ROI.
    rois : list of ROIs
        List containing the ROIs to be sliced, making it possible
        to combine several ROI shapes.
    inverted : boolean, default = False
        If `True`, everything outside of the ROIs supplied will be
        retained, with the insides of the ROIs becoming NaN
    out : Signal, default = None
        If the `out` argument is supplied, the sliced output will be put
        into this instead of returning a Signal. See Signal.__getitem__()
        for more details on `out`.
    axes : list, optional
        List of the axes in the signal that the ROIs are applied to.
    %s
    """

    for roi in rois:
        if not isinstance(roi, PolygonROI):
            raise NotImplementedError(
                "`combine_rois` is currently only implemented for `PolygonROI`."
            )

    polygonrois = rois
    other_polygons = [
        polygon._vertices for polygon in polygonrois[1:] if polygon.is_valid()
    ]

    sliced_signal = polygonrois[0]._combine(
        signal,
        inverted=inverted,
        out=out,
        axes=axes,
        additional_polygons=other_polygons,
    )

    return sliced_signal


def mask_from_rois(
    rois,
    axes_manager=None,
    axes=None,
    xy_max=None,
    xy_min=None,
    x_scale=None,
    y_scale=None,
):
    """Function to rasterize a list of ROIs into a boolean numpy array. The
        interior of the ROIs are by default `True`.
        Currently only implemented for a list of `PolygonROI`s.

    Parameters
    ----------
    rois : list of ROIs
        List containing the ROIs to be added to the mask, making it possible
        to combine several ROI shapes.
    axes_manager : :py:class:`~hyperspy.axes.AxesManager`, optional
        If supplied, the rasterization parameters not explicitly given will be
        extracted from this. The axes can be given with the `axes` attribute.
    axes : list, optional
        List of the axes in `axes_manager` that are to be used as a basis for
        the rasterization if other parameters aren't supplied.
    xy_max : tuple, optional
        The maximum x and y values that the rasterized mask covers, given in
        the same coordinate space as the polygon vertices. The defaults are
        the max values in the `vertices` of the supplied ROIs.
    xy_min : tuple, optional
        The minimum x and y values that the mask covers, given in the same coordinate
        space as the polygon vertices. The default is from `(0, 0)`.
    x_scale : float, optional
        This gives the scale of the second axis of the signal. In other words,
        how many units in coordinate space that corresponds to one step between
        columns. Default is 1.
    y_scale : float, optional
        This gives the scale of the second axis of the signal. In other words,
        how many units in coordinate space that corresponds to one step between
        rows. Default is 1.

    Returns
    -------
    return_value : array
        boolean numpy array of the rasterized ROIs. Depending on the limits set for
        the rasterization, parts of the ROIs may be outside of this.
    """

    for roi in rois:
        if not isinstance(roi, PolygonROI):
            raise NotImplementedError(
                "`mask_from_rois` is currently only implemented for `PolygonROI`."
            )

    polygonrois = rois
    other_polygons = [
        polygon._vertices for polygon in polygonrois[1:] if polygon.is_valid()
    ]

    mask = polygonrois[0]._boolean_mask(
        axes_manager=axes_manager,
        axes=axes,
        xy_max=xy_max,
        xy_min=xy_min,
        x_scale=x_scale,
        y_scale=y_scale,
        additional_polygons=other_polygons,
    )

    return mask
