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
import re
import threading
import time
import warnings
from collections.abc import Iterable
from contextlib import contextmanager
from inspect import Parameter, Signature

from psygnal import Signal, SignalGroup, SignalInstance

from hyperspy.exceptions import VisibleDeprecationWarning

# Regex for validating Python identifier names (used in argument validation)
_RE_ARG_NAME = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")

# Empty signature shared as default for SignalInstance-compatible __init__
_EMPTY_SIGNATURE = Signature()


def EventSignal(*types, description="", arguments=None, **kwargs):
    """Return a :class:`psygnal.Signal` descriptor that creates :class:`Event` instances.

    Use this as a class attribute on a :class:`~psygnal.SignalGroup`
    subclass.  When accessed on an instance it returns an :class:`Event`
    that supports both the native psygnal API and the legacy HyperSpy
    event API.

    Parameters
    ----------
    *types : type | :class:`~inspect.Signature`
        Accepted types for the signal signature (passed to psygnal).
    description : str
        Optional description for the signal.
    arguments : iterable, optional
        Deprecated.  Declared trigger argument names (and optional defaults).
        E.g. ``("x", ("y", 0.0))``.
    **kwargs
        Extra keyword arguments forwarded to :class:`psygnal.Signal`.

    Returns
    -------
    psygnal.Signal
        A signal descriptor configured to instantiate :class:`Event`.
    """
    # Backward-compat: if a Signature was not provided but legacy arguments
    # were, build a keyword-only Signature from the argument declaration so
    # psygnal's connection-time validation understands the expected kwargs.
    if not types and arguments:
        params = []
        for arg in arguments:
            if isinstance(arg, (tuple, list)):
                name, default = arg
                params.append(Parameter(name, Parameter.KEYWORD_ONLY, default=default))
            else:
                params.append(Parameter(arg, Parameter.KEYWORD_ONLY))
        types = (Signature(params),)

    class _Event(Event):
        def __init__(self, signature=None, **init_kwargs):
            if signature is None:
                signature = _EMPTY_SIGNATURE
            init_kwargs.setdefault("description", description)
            init_kwargs["arguments"] = arguments
            super().__init__(signature, **init_kwargs)

    kwargs.setdefault("check_nargs_on_connect", False)
    return Signal(
        *types,
        description=description,
        signal_instance_class=_Event,
        **kwargs,
    )


class Event(SignalInstance):
    """Event class.

    Subclasses :class:`psygnal.SignalInstance` so that all native psygnal
    methods (``emit``, ``connect``, ``disconnect``, ``blocked``, ``block``,
    ``unblock``) are available directly.  The legacy HyperSpy API (``trigger``,
    ``connect(kwargs=...)``, ``suppress``, ``suppress_callback``,
    ``.connected``, ``arguments=``) is preserved as a deprecated shim on top
    of the inherited psygnal machinery.

    .. deprecated:: 2.5
        :class:`~.events.Event` is deprecated and will be removed in HyperSpy 3.0.
        Use :external:class:`psygnal.Signal` instead.
        See the :ref:`events_migration` section for guidance on migrating code
        to the psygnal-backed event system.

    Parameters
    ----------
    signature : :class:`~inspect.Signature`
        psygnal signal signature (default: empty).
    doc : str, optional
        Deprecated alias for *description*.
    arguments : iterable, optional
        Deprecated.  Declared trigger argument names (and optional defaults).
    instance : Any, optional
        Object to which this signal is bound.
    name : str, optional
        Optional name for the signal.
    description : str, optional
        Short description of the signal.
    check_nargs_on_connect : bool
        Whether psygnal should check argument counts on connect
        (default: ``False`` — kwargs-based dispatch makes this unnecessary).
    check_types_on_connect : bool
        Whether psygnal checks types on connect (default: ``False``).
    reemission : str
        psygnal re-emission policy (default: ``"immediate"``).

    """

    def __init__(
        self,
        signature=_EMPTY_SIGNATURE,
        *,
        doc=None,
        arguments=None,
        instance=None,
        name=None,
        description="",
        check_nargs_on_connect=False,
        check_types_on_connect=False,
        reemission="immediate",
    ):
        # Backward compat: doc → description
        if doc is not None:
            description = doc

        # Validate argument names and default ordering at construction time
        if arguments:
            self._validate_arguments(arguments)
        self._arguments = tuple(arguments) if arguments else None

        # Build the psygnal SignalInstance
        super().__init__(
            signature,
            instance=instance,
            name=name,
            description=description,
            check_nargs_on_connect=check_nargs_on_connect,
            check_types_on_connect=check_types_on_connect,
            reemission=reemission,
        )

        # Backward compat: make instance __doc__ reflect the user description
        # so that ``__str__`` / ``__repr__`` format matches the old class.
        self.__doc__ = description

        # Legacy suppression flag — separate from psygnal's _is_blocked.
        # Used by Event.suppress() and SignalGroup container suppression.
        self._suppress = False

        # Tracking for the deprecated connect/disconnect API
        self._connected_originals = set()  # all original callables
        # original → (wrapper, spec) where spec is the kwargs= value
        self._wrapper_map = {}
        # Dereferenced slot callback → 'all' | 'some' | 'map' (for dispatch ordering)
        self._slot_mode = {}

        # Set of callables currently suppressed by suppress_callback
        self._suppressed_callbacks = set()

        # Throttle state — see throttle() context manager
        self._throttle_interval = None  # seconds; None = disabled
        self._throttle_next_allowed = 0.0  # monotonic timestamp

        # Debounce state — see debounce() context manager
        self._debounce_interval = None  # seconds; None = disabled
        self._debounce_timer = None  # threading.Timer or None
        self._debounce_pending_kwargs = None

        # Max listeners guard — see connect()
        self._max_listeners = None  # None = unlimited

    # -- backward-compat property ------------------------------------------

    @property
    def arguments(self):
        """Declared trigger argument names (deprecated)."""
        return self._arguments

    # -- argument validation (construction time) ---------------------------

    @staticmethod
    def _validate_arguments(arguments):
        """Validate argument names and default ordering (raises on failure)."""
        defaults = []
        for arg in arguments:
            if isinstance(arg, (tuple, list)):
                defaults.append(arg[1])
                arg = arg[0]
            elif defaults:
                raise SyntaxError("non-default argument follows default argument")
            m = _RE_ARG_NAME.match(arg)
            if m is None or m.end() != len(arg):
                raise ValueError("Argument name invalid: %s" % arg)

    # -- emit (native psygnal override) ------------------------------------

    def emit(self, *args, **kwargs):
        """Emit the signal, calling every connected slot with ``**kwargs``.

        Bypasses psygnal's ``_run_emit_loop`` (which expects positional
        args) and instead iterates ``_slots`` directly.  Exceptions
        raised by callbacks are NOT caught — they propagate immediately,
        aborting remaining slots.

        Respects :func:`psygnal.SignalInstance.block` /
        :func:`psygnal.SignalInstance.unblock` (``_is_blocked``), the
        legacy ``_suppress`` flag, and optionally :meth:`throttle` /
        :meth:`debounce` rate-limiters.

        Positional arguments are mapped to declared argument names (via
        ``_arguments``, if set) — matching the :meth:`trigger` behaviour
        for backward compatibility with code like ``emit(signal)``.
        """
        # Map positional args to declared argument names (same as trigger())
        if args:
            if self._arguments:
                for name, val in zip(self._arguments, args, strict=True):
                    kwargs.setdefault(name, val)
            else:
                raise TypeError(
                    f"{type(self).__name__}.emit() received unexpected "
                    f"positional argument(s): {args!r}. Use keyword arguments."
                )

        if self._is_blocked or self._suppress:
            return

        # Throttle guard: skip if still within the cooldown interval
        if self._throttle_interval is not None:
            now = time.monotonic()
            if now < self._throttle_next_allowed:
                return
            self._throttle_next_allowed = now + self._throttle_interval

        # Debounce guard: defer emission, resetting timer on each call
        if self._debounce_interval is not None:
            self._debounce_pending_kwargs = kwargs
            if self._debounce_timer is not None:
                self._debounce_timer.cancel()
            self._debounce_timer = threading.Timer(
                self._debounce_interval, self._debounce_fire
            )
            self._debounce_timer.start()
            return

        self._emit_dispatch(kwargs)

    def _emit_dispatch(self, kwargs):
        """Core dispatch: validate, order, and invoke callbacks."""
        if self._arguments:
            kwargs = self._validate_emit_kwargs(kwargs)

        # Snapshot slots so connect/disconnect during dispatch are safe.
        # Dispatch in legacy order: "all" → "some" → "map".  This
        # preserves exception-abort semantics — a callback in the "all"
        # group can raise TypeError (aborting dispatch) before a "map"
        # callback would hit KeyError on a missing kwarg.
        all_callbacks = []
        some_callbacks = []
        map_callbacks = []

        for slot in list(self._slots):
            callback = slot.dereference()
            if callback is None:
                continue
            mode = self._slot_mode.get(callback, "all")
            if mode == "all":
                all_callbacks.append(callback)
            elif mode == "some":
                some_callbacks.append(callback)
            else:  # "map"
                map_callbacks.append(callback)

        for callback in all_callbacks + some_callbacks + map_callbacks:
            original = self._find_original(callback)
            if original in self._suppressed_callbacks:
                continue
            callback(**kwargs)

    def _debounce_fire(self):
        """Called by the debounce timer — fires the pending emission."""
        kwargs = self._debounce_pending_kwargs
        self._debounce_pending_kwargs = None
        self._debounce_timer = None
        if kwargs is not None:
            self._emit_dispatch(kwargs)

    @contextmanager
    def throttle(self, interval):
        """Context manager that rate-limits emissions to at most one per *interval* seconds.

        While active, repeated ``emit()`` calls within the interval are
        silently dropped — only the first emission in each window passes through.

        Parameters
        ----------
        interval : float
            Minimum time in seconds between allowed emissions.

        Examples
        --------
        >>> with event.throttle(0.5):
        ...     for _ in range(100):
        ...         event.emit(x=1)  # only fires ~once every 0.5s
        """
        prev_interval = self._throttle_interval
        prev_next = self._throttle_next_allowed
        self._throttle_interval = interval
        self._throttle_next_allowed = 0.0
        try:
            yield
        finally:
            self._throttle_interval = prev_interval
            self._throttle_next_allowed = prev_next

    @contextmanager
    def debounce(self, interval):
        """Context manager that defers emissions until *interval* seconds of silence.

        Each ``emit()`` resets the internal timer.  The signal only fires
        after *interval* seconds have passed since the last ``emit()`` call.

        Parameters
        ----------
        interval : float
            Quiet period in seconds before the deferred emission fires.

        Examples
        --------
        >>> with event.debounce(0.3):
        ...     event.emit(x=1)   # timer starts
        ...     event.emit(x=2)   # timer resets
        ...     # 0.3s later → callbacks receive {x: 2}
        """
        prev_interval = self._debounce_interval
        self._debounce_interval = interval
        try:
            yield
        finally:
            if self._debounce_timer is not None:
                self._debounce_timer.cancel()
            self._debounce_interval = prev_interval
            self._debounce_timer = None
            self._debounce_pending_kwargs = None

    def _validate_emit_kwargs(self, kwargs):
        """Validate emit kwargs against the declared ``_arguments``.

        Builds an :class:`inspect.Signature`, binds, applies defaults,
        and returns the complete kwargs dict.  Raises ``TypeError``
        (matching the old ``_trigger_maker`` behaviour) when arguments
        are unexpected or required args are missing.
        """
        params = []
        for arg in self._arguments:
            if isinstance(arg, (tuple, list)):
                name, default = arg[0], arg[1]
                params.append(Parameter(name, Parameter.KEYWORD_ONLY, default=default))
            else:
                params.append(Parameter(arg, Parameter.KEYWORD_ONLY))
        sig = Signature(params)
        ba = sig.bind(**kwargs)
        ba.apply_defaults()
        return ba.arguments

    def _find_original(self, callback):
        """Look through ``_wrapper_map`` to find the original callable
        for *callback* (the dereferenced slot).  Returns *callback*
        itself if no wrapper entry matches.
        """
        for orig, (wrapper, _spec) in self._wrapper_map.items():
            stored_deref = (
                wrapper.dereference() if hasattr(wrapper, "dereference") else wrapper
            )
            if wrapper is callback or stored_deref is callback:
                return orig
        return callback

    # -- connect (deprecated kwargs= shim + native) ------------------------

    def connect(self, function, kwargs="all", **psygnal_opts):
        """Connect a function to the event.

        .. deprecated:: 2.5
            The ``kwargs=`` parameter is deprecated and will be removed in HyperSpy 3.0.
            See :external:func:`psygnal.SignalInstance.connect` for the new API and use adapter
            functions when filtering or renaming kwargs is needed.

        Parameters
        ----------
        function : callable
            The function to call when the event triggers.
        kwargs : str, dict, list, or tuple
            If ``"all"`` (default), every trigger keyword argument is
            forwarded to *function*.  A dictionary renames trigger kwargs
            to function parameter names.  A list or tuple forwards only
            the named subset.  ``"auto"`` inspects the function signature
            to determine which parameters to forward.
        **psygnal_opts
            Additional keyword arguments forwarded to
            :func:`psygnal.SignalInstance.connect`.
        """
        if not callable(function):
            raise TypeError("Only callables can be registered")
        if function in self._connected_originals:
            raise ValueError("Function %s already connected to %s." % (function, self))

        if kwargs != "all":
            warnings.warn(
                "Event.connect(kwargs=...) is deprecated and will be removed in HyperSpy 3.0. "
                "See the `psygnal.Signal.connect` for the new API and use adapter functions instead.",
                VisibleDeprecationWarning,
                stacklevel=2,
            )

        # Max-listeners guard — warn when exceeding the configured limit
        if self._max_listeners is not None and len(self._slots) >= self._max_listeners:
            warnings.warn(
                f"Event {self!r} has {len(self._slots)} connected slots "
                f"(max_listeners={self._max_listeners}).",
                VisibleDeprecationWarning,
                stacklevel=2,
            )

        # Resolve "auto" mode
        if kwargs == "auto":
            kwargs = self._auto_kwargs(function)

        if kwargs == "all":
            super().connect(function, **psygnal_opts)
            self._connected_originals.add(function)
            self._slot_mode[function] = "all"

        elif isinstance(kwargs, dict):
            wrapper = self._make_dict_wrapper(function, kwargs)
            super().connect(wrapper, **psygnal_opts)
            self._wrapper_map[function] = (wrapper, kwargs)
            self._connected_originals.add(function)
            self._slot_mode[wrapper] = "map"

        elif isinstance(kwargs, (list, tuple)):
            spec = tuple(kwargs)
            wrapper = self._make_list_wrapper(function, spec)
            super().connect(wrapper, **psygnal_opts)
            self._wrapper_map[function] = (wrapper, spec)
            self._connected_originals.add(function)
            self._slot_mode[wrapper] = "some"

        else:
            raise ValueError("Invalid value passed to kwargs.")

    @staticmethod
    def _auto_kwargs(function):
        """Inspect *function* signature and return ``"all"`` or a list of
        normal parameter names."""
        spec = inspect.signature(function)
        has_var_positional = False
        has_var_keyword = False
        normal_params = []
        for name, par in spec.parameters.items():
            if par.kind == Parameter.VAR_POSITIONAL:
                has_var_positional = True
            elif par.kind == Parameter.VAR_KEYWORD:
                has_var_keyword = True
            else:
                normal_params.append(name)
        if has_var_positional and not has_var_keyword:
            raise NotImplementedError(
                "Connecting to variable argument "
                "functions is not supported in auto "
                "connection mode."
            )
        elif has_var_keyword:
            return "all"
        else:
            return normal_params

    @staticmethod
    def _make_dict_wrapper(function, kwarg_map):
        """Return a callable that renames trigger kwargs for *function*."""
        return lambda **kw: function(
            **{target: kw[source] for source, target in kwarg_map.items()}
        )

    @staticmethod
    def _make_list_wrapper(function, kwarg_list):
        """Return a callable that forwards only the named subset of kwargs."""
        return lambda **kw: function(**{k: kw.get(k) for k in kwarg_list})

    # -- disconnect (deprecated) -------------------------------------------

    def disconnect(self, function):
        """Disconnect *function* from the event."""
        # Look up wrapper if this function was connected with a kwargs map
        if function in self._wrapper_map:
            wrapper, _spec = self._wrapper_map.pop(function)
        else:
            wrapper = function

        super().disconnect(wrapper, missing_ok=False)
        self._connected_originals.discard(function)
        self._slot_mode.pop(wrapper, None)

    # -- _try_discard (weak-reference cleanup override) --------------------

    def _try_discard(self, callback, missing_ok=True):
        """Called by psygnal when a weakly-referenced slot is GC'd.

        Cleans up the corresponding entries in ``_wrapper_map`` and
        ``_connected_originals`` so they stay in sync with ``_slots``.
        """
        # callback is the slot wrapper (StrongFunction / WeakFunction)
        derefed = (
            callback.dereference() if hasattr(callback, "dereference") else callback
        )

        if derefed is not None:
            # Walk _wrapper_map to see if this slot belongs to a wrapper entry
            for original, (wrapper, _spec) in list(self._wrapper_map.items()):
                # Compare slot's dereferenced callback against the stored wrapper
                stored_deref = (
                    wrapper.dereference()
                    if hasattr(wrapper, "dereference")
                    else wrapper
                )
                if (
                    wrapper is callback
                    or stored_deref is callback
                    or stored_deref is derefed
                ):
                    del self._wrapper_map[original]
                    self._connected_originals.discard(original)
                    self._slot_mode.pop(wrapper, None)
                    self._slot_mode.pop(stored_deref, None)
                    break
            else:
                # Not a wrapper — directly connected
                self._connected_originals.discard(derefed)
                self._slot_mode.pop(derefed, None)

        super()._try_discard(callback, missing_ok=missing_ok)

    # -- deprecated trigger -------------------------------------------------

    def trigger(self, *args, **kwargs):
        """Trigger the event (legacy API — use :meth:`emit` instead).

        Accepts positional arguments (mapped to declared argument names
        in ``_arguments`` if available) and keyword arguments, then
        delegates to :meth:`emit`.
        """
        warnings.warn(
            "Event.trigger() is deprecated and will be removed in HyperSpy 3.0. "
            "Use emit() instead.",
            VisibleDeprecationWarning,
            stacklevel=2,
        )
        # Map positional args to declared argument names
        if args:
            if self._arguments:
                for name, val in zip(self._arguments, args):
                    kwargs.setdefault(name, val)
            else:
                # No declared arguments — pass positionally as kwargs names
                for i, val in enumerate(args):
                    kwargs[f"_arg{i}"] = val
        self.emit(**kwargs)

    # -- deprecated .connected property ------------------------------------

    @property
    def connected(self):
        """Set of connected functions.

        .. deprecated:: 2.5
            This will be removed in HyperSpy 3.0. The connected funtions will
            need to be tracked by the user code when needed.
        """
        warnings.warn(
            "Event.connected is deprecated and will be removed in HyperSpy 3.0. "
            "Use the native psygnal API to inspect connections.",
            VisibleDeprecationWarning,
            stacklevel=2,
        )
        return set(self._connected_originals)

    # -- suppress / suppress_callback (deprecated context managers) --------

    @contextmanager
    def suppress(self):
        """Context manager to temporarily suppress event emission.

        .. deprecated:: 2.5
            This will be removed in HyperSpy 3.0.
            Use :func:`psygnal.SignalInstance.blocked` (or
            :func:`psygnal.SignalInstance.block` /
            :func:`psygnal.SignalInstance.unblock`) instead.

        Examples
        --------
        >>> with obj.events.myevent.suppress():  # doctest: +SKIP
        ...     obj.val_a = a
        ...     obj.val_b = b
        >>> obj.events.myevent.trigger()  # doctest: +SKIP

        See Also
        --------
        suppress_callback
        """
        warnings.warn(
            "Event.suppress() is deprecated and will be removed in HyperSpy 3.0. "
            "Use blocked() or block()/unblock() instead.",
            VisibleDeprecationWarning,
            stacklevel=2,
        )
        old = self._suppress
        self._suppress = True
        try:
            yield
        finally:
            self._suppress = old

    @contextmanager
    def suppress_callback(self, function):
        """Context manager to temporarily suppress a single callback.

        .. deprecated:: 2.5
            This will be removed in HyperSpy 3.0.
            Use the native psygnal ``disconnect`` / ``reconnect`` pattern
            instead.

        Examples
        --------
        >>> with obj.events.myevent.suppress_callback(f):  # doctest: +SKIP
        ...     obj.val_a = a
        ...     obj.val_b = b
        >>> obj.events.myevent.trigger()  # doctest: +SKIP

        See Also
        --------
        suppress
        """
        warnings.warn(
            "Event.suppress_callback() is deprecated and will be removed in HyperSpy 3.0. "
            "Use the native psygnal ``disconnect`` / ``reconnect`` pattern instead.",
            VisibleDeprecationWarning,
            stacklevel=2,
        )
        was_suppressed = function in self._suppressed_callbacks
        if not was_suppressed:
            self._suppressed_callbacks.add(function)
        try:
            yield
        finally:
            if not was_suppressed:
                self._suppressed_callbacks.discard(function)

    # -- copy / repr / str --------------------------------------------------

    def __deepcopy__(self, memo):
        dc = type(self)()
        memo[id(self)] = dc
        return dc

    def __repr__(self):
        return "<hyperspy.events.Event: " + repr(self._connected_originals) + ">"

    def __str__(self):
        if self.__doc__:
            edoc = inspect.getdoc(self) or ""
            doclines = edoc.splitlines()
            e_short = doclines[0] if len(doclines) > 0 else edoc
            text = (
                "<hyperspy.events.Event: "
                + e_short
                + ": "
                + str(self._connected_originals)
                + ">"
            )
        else:
            text = self.__repr__()
        return text


class EventSuppressor(object):
    """
    Object to enforce a variety of suppression types simultaneously

    Targets to be suppressed can be added by the function `add()`, or given
    in the constructor. Valid targets are:

    * `Event`: The entire Event will be suppressed
    * :external:class:`psygnal.SignalGroup`: All events in the group will be suppressed
    * (Event, callback): The callback will be suppressed in Event
    * (:external:class:`psygnal.SignalGroup`, callback): The callback will be suppressed in each event in
      the SignalGroup where it is connected.
    * Any iterable collection of the above target types

    .. deprecated:: 2.5
        :class:`~.events.EventSuppressor` is deprecated and will be removed in HyperSpy 3.0.
        Use the ``psygnal.SignalGroup.blocked()`` context manager instead.
        See the :ref:`events_migration` section for guidance on migrating code
        to the psygnal-backed event system.

    Examples
    --------
    >>> es = EventSuppressor((event1, callback1), (event1, callback2)) # doctest: +SKIP
    >>> es.add(event2, callback2) # doctest: +SKIP
    >>> es.add(event3) # doctest: +SKIP
    >>> es.add(events_container1) # doctest: +SKIP
    >>> es.add(events_container2, callback1) # doctest: +SKIP
    >>> es.add(event4, (events_container3, callback2)) # doctest: +SKIP

    >>> with es.suppress(): # doctest: +SKIP
    ...     do_something()
    """
    def __init__(self, *to_suppress):
        warnings.warn(
            "hyperspy.events.EventSuppressor is deprecated and will be removed in HyperSpy 3.0. "
            "Use psygnal.SignalGroup.blocked() context manager instead.",
            VisibleDeprecationWarning,
            stacklevel=2,
        )
        self._cms = []
        if len(to_suppress) > 0:
            self.add(*to_suppress)

    def _add_single(self, target):
        # Identify and initializes the CM, but doesn't enter it
        if self._is_tuple_target(target):
            if isinstance(target[0], Event):
                cm = target[0].suppress_callback(target[1])
                self._cms.append(cm)
            else:
                # target[0] is SignalGroup — iterate its signal instances
                for sig in target[0]._psygnal_instances.values():
                    self._cms.append(sig.suppress_callback(target[1]))
        else:
            if isinstance(target, SignalGroup):
                cm = target.blocked()
            else:
                cm = target.suppress()
            self._cms.append(cm)

    def _is_tuple_target(self, candidate):
        v = (
            isinstance(candidate, Iterable)
            and not isinstance(candidate, SignalGroup)
            and len(candidate) == 2
            and isinstance(candidate[0], (Event, SignalGroup))
            and callable(candidate[1])
        )
        return v

    def _is_target(self, candidate):
        v = isinstance(candidate, (Event, SignalGroup)) or self._is_tuple_target(
            candidate
        )
        return v

    def add(self, *to_suppress):
        """
        Add one or more targets to be suppressed

        Valid targets are:
         - `Event`: The entire Event will be suppressed
         - ``SignalGroup``: All events in the group will be suppressed
         - (Event, callback): The callback will be suppressed in Event
         - (``SignalGroup``, callback): The callback will be suppressed in each event
           in the SignalGroup where it is connected.
         - Any iterable collection of the above target types
        """
        # Remove useless layers of iterables:
        while (
            isinstance(to_suppress, Iterable)
            and not isinstance(to_suppress, SignalGroup)
            and len(to_suppress) == 1
        ):
            to_suppress = to_suppress[0]
        # If single target passed, add directly:
        if self._is_target(to_suppress):
            self._add_single(to_suppress)
        elif isinstance(to_suppress, Iterable):
            if len(to_suppress) == 0:
                raise ValueError("No viable suppression targets added!")
            for t in to_suppress:
                if self._is_target(t):
                    self._add_single(t)
        else:
            raise ValueError("No viable suppression targets added!")

    @contextmanager
    def suppress(self):
        """
        Use this function with a 'with' statement to temporarily suppress
        all events added. When the 'with' lock completes, the old suppression
        values will be restored.

        See Also
        --------
        Event.suppress
        Event.suppress_callback
        """
        # We don't suppress any exceptions, so we can use simple CM management:
        cms = []
        try:
            for cm in self._cms:
                cm.__enter__()
                cms.append(cm)  # Only add entered CMs to list
            yield
        finally:
            # Completed succefully or exception occured, unwind all
            for cm in reversed(cms):
                # We don't use exception info, so simply pass blanks
                cm.__exit__(None, None, None)
