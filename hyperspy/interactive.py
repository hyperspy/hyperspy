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

import inspect


def _connect_events(event, to_connect):
    try:
        for ev in event:
            # Iterable of events, connect all of them
            ev.connect(to_connect, [])
    except TypeError:
        # It was not an iterable, connect the single event
        event.connect(to_connect, [])


class Interactive:

    """Chainable operations on Signals that update on events.

    """

    def __init__(self, f, event="auto",
                 recompute_out_event="auto",
                 *args, **kwargs):
        """Update operation result when a given event is triggered.

        Parameters
        ----------
        f: function or method
            A function that returns an object and that optionally can place the
            result in an object given through the `out` keyword.
        event: {Event, "auto", None, iterable of events}
            Update the result of the operation when the event is triggered.
            If "auto" and `f` is a method of a Signal class instance its
            `data_changed` event is selected if the function takes an `out`
            argument. If None, `update` is not connected to any event. The
            default is "auto". It is also possible to pass an iterable of
            events, in which case all the events are connected.
        recompute_out_event: {Event, "auto", None, iterable of events}
            Optional argument. If supplied, this event causes a full
            recomputation of a new object. Both the data and axes of the new
            object are then copied over to the existing `out` object. Only
            useful for `Signal` or other objects that have an attribute
            `axes_manager`. If "auto" and `f` is a method of a Signal class
            instance its `AxesManager` `any_axis_chaged` event is selected.
            Otherwise the `Signal`
            `data_changed` event is selected. If None, `recompute_out` is not
            connected to any event. The default is "auto". It is also possible
            to pass an iterable of events, in which case all the events are
            connected.


        *args, **kwargs
            Arguments and keyword arguments to be passed to `f`.

        """
        from hyperspy.signal import BaseSignal
        self.f = f
        self.args = args
        self.kwargs = kwargs
        _plot_kwargs = self.kwargs.pop('_plot_kwargs', None)
        if 'out' in self.kwargs:
            self.f(*self.args, **self.kwargs)
            self.out = self.kwargs.pop('out')
        else:
            self.out = self.f(*self.args, **self.kwargs)
        # Reuse the `_plot_kwargs` for the roi if available
        if _plot_kwargs and 'signal' in self.kwargs:
            self.out._plot_kwargs = self.kwargs['signal']._plot_kwargs
        try:
            fargs = list(inspect.signature(self.f).parameters.keys())
        except TypeError:
            # This is probably a Cython function that is not supported by
            # inspect.
            fargs = []
        has_out = "out" in fargs
        # If it is a BaseSignal method
        if hasattr(f, "__self__") and isinstance(f.__self__, BaseSignal):
            if event == "auto":
                event = self.f.__self__.events.data_changed
            if recompute_out_event == "auto":
                recompute_out_event = \
                    self.f.__self__.axes_manager.events.any_axis_changed
        else:
            event = None if event == "auto" else event
            recompute_out_event = (None if recompute_out_event == "auto"
                                   else recompute_out_event)
        if recompute_out_event:
            _connect_events(recompute_out_event, self.recompute_out)
        if event:
            if has_out:
                _connect_events(event, self.update)
            else:
                #  We "simulate" out by triggering `recompute_out` instead.
                _connect_events(event, self.recompute_out)

    def recompute_out(self):
        out = self.f(*self.args, **self.kwargs)
        if out.data.shape == self.out.data.shape:
            # Keep the same array if possible.
            self.out.data[:] = out.data[:]
        else:
            self.out.data = out.data
        # The following may trigger an `any_axis_changed` event and, therefore,
        # it must precede the `data_changed` trigger below.
        self.out.axes_manager.update_axes_attributes_from(
            out.axes_manager._axes, attributes=["scale", "offset", "size"])
        self.out.events.data_changed.trigger(self.out)

    def update(self):
        self.f(*self.args, out=self.out, **self.kwargs)


def interactive(f, event="auto", recompute_out_event="auto", *args, **kwargs):
    cls = Interactive(f, event, recompute_out_event, *args, **kwargs)
    return cls.out


interactive.__doc__ = Interactive.__init__.__doc__
