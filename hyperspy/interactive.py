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

import inspect


def _connect_events(event, to_connect):
    try:
        for ev in event:
            # Iterable of events, connect all of them
            ev.connect(to_connect)
    except TypeError:
        # It was not an iterable, connect the single event
        event.connect(to_connect)


def _disconnect_events(event, to_disconnect):
    """Disconnect a callable from one or more events.

    Mirrors :func:`_connect_events` — handles both single events and
    iterables of events.
    """
    try:
        for ev in event:
            ev.disconnect(to_disconnect)
    except TypeError:
        event.disconnect(to_disconnect)


class Interactive:
    r"""
    Chainable operations on Signals that update on events. The operation
    result will be updated when a given event is triggered.

    Parameters
    ----------
    f : callable
        A function that returns an object and that optionally can place the
        result in an object given through the ``out`` keyword.
    event : (list of) :class:`~hyperspy.events.Event`, str ("auto") or None
        Update the result of the operation when the event is triggered.
        If ``"auto"`` and ``f`` is a method of a Signal class instance its
        ``data_changed`` event is selected if the function takes an ``out``
        argument. If None, ``update`` is not connected to any event. The
        default is ``"auto"``. It is also possible to pass an iterable of
        events, in which case all the events are connected.
    recompute_out_event : (list of) :class:`~hyperspy.events.Event`, str ("auto") or None
        Optional argument. If supplied, this event causes a full
        recomputation of a new object. Both the data and axes of the new
        object are then copied over to the existing `out` object. Only
        useful for signals or other objects that have an attribute
        ``axes_manager``. If ``"auto"`` and ``f`` is a method of a Signal class
        instance its ``AxesManager`` ``any_axis_changed`` event is selected.
        Otherwise, the signal ``data_changed`` event is selected.
        If None, ``recompute_out`` is not connected to any event.
        The default is ``"auto"``. It is also possible to pass an iterable of
        events, in which case all the events are connected.
    *args :
        Arguments to be passed to ``f``.
    **kwargs : dict
        Keyword arguments to be passed to ``f``.

    """

    def __init__(self, f, event="auto", recompute_out_event="auto", *args, **kwargs):
        from hyperspy.signal import BaseSignal

        self.f = f
        self.args = args
        self.kwargs = kwargs
        _plot_kwargs = self.kwargs.pop("_plot_kwargs", None)
        if "out" in self.kwargs:
            self.f(*self.args, **self.kwargs)
            self.out = self.kwargs.pop("out")
        else:
            self.out = self.f(*self.args, **self.kwargs)
        # Reuse the `_plot_kwargs` for the roi if available
        if _plot_kwargs and "signal" in self.kwargs:
            self.out._plot_kwargs = self.kwargs["signal"]._plot_kwargs
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
                recompute_out_event = (
                    self.f.__self__.axes_manager.events.any_axis_changed
                )
        else:
            event = None if event == "auto" else event
            recompute_out_event = (
                None if recompute_out_event == "auto" else recompute_out_event
            )
        self._event = event
        self._recompute_out_event = recompute_out_event
        self._has_out = has_out
        if recompute_out_event is not None:
            _connect_events(recompute_out_event, self.recompute_out)
        if event is not None:
            if has_out:
                _connect_events(event, self.update)
            else:
                #  We "simulate" out by triggering `recompute_out` instead.
                _connect_events(event, self.recompute_out)

    def recompute_out(self, *args, **kwargs):
        out = self.f(*self.args, **self.kwargs)
        if out is None:
            return
        if out.data.shape == self.out.data.shape:
            # Keep the same array if possible.
            self.out.data[:] = out.data[:]
        else:
            self.out.data = out.data
        self.out.axes_manager.update_axes_attributes_from(out.axes_manager._axes)
        self.out.events.data_changed.emit(self.out)

    def update(self, *args, **kwargs):
        self.f(*self.args, out=self.out, **self.kwargs)

    def close(self):
        """Disconnect all event handlers and release internal references.

        After calling this method, the operation will no longer respond to
        any previously connected events. The ``out`` attribute remains
        accessible for reading the last computed result.

        Examples
        --------
        >>> from hyperspy.interactive import Interactive
        >>> s = hs.signals.Signal1D(np.arange(10.))
        >>> op = Interactive(s.sum, event=None, recompute_out_event=None, axis=0)
        >>> op.close()
        >>> op.out.data
        array([45.])
        """
        # Disconnect event handlers to prevent handler leaks — the
        # Interactive object would otherwise remain reachable through
        # the event system's internal callback registries and could
        # never be garbage-collected.
        if self._recompute_out_event:
            _disconnect_events(self._recompute_out_event, self.recompute_out)
        if self._event:
            if self._has_out:
                _disconnect_events(self._event, self.update)
            else:
                _disconnect_events(self._event, self.recompute_out)
        self._event = None
        self._recompute_out_event = None


def interactive(f, event="auto", recompute_out_event="auto", *args, **kwargs):
    """%s

    Returns
    -------
    :class:`~hyperspy.signal.BaseSignal` or subclass
        Signal updated with the operation result when a given event is
        triggered.

    """
    cls = Interactive(f, event, recompute_out_event, *args, **kwargs)
    return cls.out


interactive.__doc__ %= Interactive.__doc__
