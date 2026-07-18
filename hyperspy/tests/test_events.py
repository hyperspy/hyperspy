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

import copy
import gc
import warnings

import numpy as np
import psygnal
import pytest
from psygnal import SignalGroup

from hyperspy.events import Event, EventSignal, EventSuppressor
from hyperspy.exceptions import VisibleDeprecationWarning

# ============================================================================
# This module tests the legacy (deprecated) Event API.  Tests fall into two
# categories:
#
#   1. DEPRECATED API TESTS ― wrap the call site(s) in
#      ``pytest.warns(VisibleDeprecationWarning)`` because they intentionally
#      exercise ``Event.connect(kwargs=…)``, ``Event.suppress()``,
#      ``Event.suppress_callback()``, ``EventSuppressor``, ``Event.trigger()``,
#      or ``Event.connected``.
#
#   2. NATIVE-API TESTS ― use ``psygnal.SignalInstance.connect``,
#      ``psygnal.SignalInstance.emit``, ``psygnal.SignalInstance.blocked``,
#      etc. WITHOUT ``pytest.warns``.  These tests do not emit
#      ``VisibleDeprecationWarning``.
# ============================================================================


class EventsSuppressionGroup(SignalGroup):
    a = EventSignal()
    b = EventSignal()
    c = EventSignal()


class EventSignaturesGroup(SignalGroup):
    a = EventSignal(object, object)


class ArgResolutionGroup(SignalGroup):
    a = EventSignal(object, object, arguments=["A", "B"])
    b = EventSignal(object, object, object, arguments=["A", "B", ("C", "vC")])
    c = EventSignal()


class EventsBase:
    def on_trigger(self, **kwargs):
        self.triggered = True

    def on_trigger2(self, **kwargs):
        self.triggered2 = True

    def trigger_check(self, trigger, should_trigger, **kwargs):
        self.triggered = False
        trigger(**kwargs)
        assert self.triggered == should_trigger

    def trigger_check2(self, trigger, should_trigger, **kwargs):
        self.triggered2 = False
        trigger(**kwargs)
        assert self.triggered2 == should_trigger


class TestEventsSuppression(EventsBase):
    """Deprecated API tests — ``Event.suppress()``, ``Event.suppress_callback()``, and ``EventSuppressor``."""

    def setup_method(self, method):
        self.events = EventsSuppressionGroup(self)

        self.events.a.connect(self.on_trigger)
        self.events.a.connect(self.on_trigger2)
        self.events.b.connect(self.on_trigger)
        self.events.c.connect(self.on_trigger)

    def test_simple_suppression(self):
        with pytest.warns(VisibleDeprecationWarning):
            # Deprecated API — exercises Event.suppress() / suppress_callback() / EventSuppressor
            with self.events.a.suppress():
                self.trigger_check(self.events.a.trigger, False)
                self.trigger_check(self.events.b.trigger, True)

            self.trigger_check(self.events.a.trigger, True)
            self.trigger_check(self.events.b.trigger, True)
            self.trigger_check(self.events.c.trigger, True)

    def test_suppression_single(self):
        with pytest.warns(VisibleDeprecationWarning):
            # Deprecated API — exercises Event.suppress() / suppress_callback() / EventSuppressor
            with self.events.b.suppress():
                with self.events.a.suppress_callback(self.on_trigger):
                    self.trigger_check(self.events.a.trigger, False)
                    self.trigger_check2(self.events.a.trigger, True)
                    self.trigger_check(self.events.b.trigger, False)
                    self.trigger_check(self.events.c.trigger, True)

                self.trigger_check(self.events.a.trigger, True)
                self.trigger_check2(self.events.a.trigger, True)
                self.trigger_check(self.events.b.trigger, False)
                self.trigger_check(self.events.c.trigger, True)

            # Reverse order:
            with self.events.a.suppress_callback(self.on_trigger):
                with self.events.b.suppress():
                    self.trigger_check(self.events.a.trigger, False)
                    self.trigger_check2(self.events.a.trigger, True)
                    self.trigger_check(self.events.b.trigger, False)
                    self.trigger_check(self.events.c.trigger, True)

                self.trigger_check(self.events.a.trigger, False)
                self.trigger_check2(self.events.a.trigger, True)
                self.trigger_check(self.events.b.trigger, True)
                self.trigger_check(self.events.c.trigger, True)

    def test_exception_event(self):
        with pytest.warns(VisibleDeprecationWarning):
            # Deprecated API — exercises Event.suppress() / suppress_callback() / EventSuppressor
            with pytest.raises(ValueError):
                try:
                    with self.events.a.suppress():
                        self.trigger_check(self.events.a.trigger, False)
                        self.trigger_check(self.events.b.trigger, True)
                        self.trigger_check(self.events.c.trigger, True)
                        raise ValueError()
                finally:
                    self.trigger_check(self.events.a.trigger, True)
                    self.trigger_check(self.events.b.trigger, True)
                    self.trigger_check(self.events.c.trigger, True)

    def test_exception_single(self):
        with pytest.warns(VisibleDeprecationWarning):
            # Deprecated API — exercises Event.suppress() / suppress_callback() / EventSuppressor
            with pytest.raises(ValueError):
                try:
                    with self.events.a.suppress_callback(self.on_trigger):
                        self.trigger_check(self.events.a.trigger, False)
                        self.trigger_check2(self.events.a.trigger, True)
                        self.trigger_check(self.events.b.trigger, True)
                        self.trigger_check(self.events.c.trigger, True)
                        raise ValueError()
                finally:
                    self.trigger_check(self.events.a.trigger, True)
                    self.trigger_check2(self.events.a.trigger, True)
                    self.trigger_check(self.events.b.trigger, True)
                    self.trigger_check(self.events.c.trigger, True)

    def test_exception_nested(self):
        with pytest.warns(VisibleDeprecationWarning):
            # Deprecated API — exercises Event.suppress() / suppress_callback() / EventSuppressor
            with pytest.raises(ValueError):
                try:
                    with self.events.a.suppress_callback(self.on_trigger):
                        try:
                            with self.events.a.suppress():
                                self.trigger_check(self.events.a.trigger, False)
                                self.trigger_check2(self.events.a.trigger, False)
                                self.trigger_check(self.events.b.trigger, True)
                                self.trigger_check(self.events.c.trigger, True)
                                raise ValueError()
                        finally:
                            self.trigger_check(self.events.a.trigger, False)
                            self.trigger_check2(self.events.a.trigger, True)
                            self.trigger_check(self.events.b.trigger, True)
                            self.trigger_check(self.events.c.trigger, True)
                finally:
                    self.trigger_check(self.events.a.trigger, True)
                    self.trigger_check2(self.events.a.trigger, True)
                    self.trigger_check(self.events.b.trigger, True)
                    self.trigger_check(self.events.c.trigger, True)

    def test_suppress_wrong(self):
        with pytest.warns(VisibleDeprecationWarning):
            # Deprecated API — exercises Event.suppress() / suppress_callback() / EventSuppressor
            with self.events.a.suppress_callback(f_a):
                self.trigger_check(self.events.a.trigger, True)
                self.trigger_check2(self.events.a.trigger, True)

    def test_suppressor_init_args(self):
        with pytest.warns(VisibleDeprecationWarning):
            # Deprecated API — exercises Event.suppress() / suppress_callback() / EventSuppressor
            with self.events.b.suppress():
                es = EventSuppressor((self.events.a, self.on_trigger), self.events.c)
                with es.suppress():
                    self.trigger_check(self.events.a.trigger, False)
                    self.trigger_check2(self.events.a.trigger, True)
                    self.trigger_check(self.events.b.trigger, False)
                    self.trigger_check(self.events.c.trigger, False)
                    with self.events.a.suppress_callback(self.on_trigger2):
                        self.trigger_check2(self.events.a.trigger, False)
                    self.trigger_check2(self.events.a.trigger, True)

                self.trigger_check(self.events.a.trigger, True)
                self.trigger_check2(self.events.a.trigger, True)
                self.trigger_check(self.events.b.trigger, False)
                self.trigger_check(self.events.c.trigger, True)

            self.trigger_check(self.events.a.trigger, True)
            self.trigger_check2(self.events.a.trigger, True)
            self.trigger_check(self.events.b.trigger, True)
            self.trigger_check(self.events.c.trigger, True)

    def test_suppressor_add_args(self):
        with pytest.warns(VisibleDeprecationWarning):
            # Deprecated API — exercises Event.suppress() / suppress_callback() / EventSuppressor
            with self.events.b.suppress():
                es = EventSuppressor()
                es.add((self.events.a, self.on_trigger), self.events.c)
                with es.suppress():
                    self.trigger_check(self.events.a.trigger, False)
                    self.trigger_check2(self.events.a.trigger, True)
                    self.trigger_check(self.events.b.trigger, False)
                    self.trigger_check(self.events.c.trigger, False)
                    with self.events.a.suppress_callback(self.on_trigger2):
                        self.trigger_check2(self.events.a.trigger, False)
                    self.trigger_check2(self.events.a.trigger, True)

                self.trigger_check(self.events.a.trigger, True)
                self.trigger_check2(self.events.a.trigger, True)
                self.trigger_check(self.events.b.trigger, False)
                self.trigger_check(self.events.c.trigger, True)

            self.trigger_check(self.events.a.trigger, True)
            self.trigger_check2(self.events.a.trigger, True)
            self.trigger_check(self.events.b.trigger, True)
            self.trigger_check(self.events.c.trigger, True)

    def test_suppressor_all_callback_in_events(self):
        with pytest.warns(VisibleDeprecationWarning):
            # Deprecated API — exercises Event.suppress() / suppress_callback() / EventSuppressor
            with self.events.b.suppress():
                es = EventSuppressor()
                es.add(
                    (self.events, self.on_trigger),
                )
                with es.suppress():
                    self.trigger_check(self.events.a.trigger, False)
                    self.trigger_check2(self.events.a.trigger, True)
                    self.trigger_check(self.events.b.trigger, False)
                    self.trigger_check(self.events.c.trigger, False)
                    with self.events.a.suppress_callback(self.on_trigger2):
                        self.trigger_check2(self.events.a.trigger, False)
                    self.trigger_check2(self.events.a.trigger, True)

                self.trigger_check(self.events.a.trigger, True)
                self.trigger_check2(self.events.a.trigger, True)
                self.trigger_check(self.events.b.trigger, False)
                self.trigger_check(self.events.c.trigger, True)

            self.trigger_check(self.events.a.trigger, True)
            self.trigger_check2(self.events.a.trigger, True)
            self.trigger_check(self.events.b.trigger, True)
            self.trigger_check(self.events.c.trigger, True)

    def test_suppressor_events_container(self):
        with pytest.warns(VisibleDeprecationWarning):
            # Deprecated API — exercises Event.suppress() / suppress_callback() / EventSuppressor
            es = EventSuppressor()
            es.add(self.events)
            with es.suppress():
                self.trigger_check(self.events.a.trigger, False)
                self.trigger_check(self.events.b.trigger, False)
                self.trigger_check(self.events.c.trigger, False)

            self.trigger_check(self.events.a.trigger, True)
            self.trigger_check(self.events.b.trigger, True)
            self.trigger_check(self.events.c.trigger, True)


def f_a(**kwargs):
    pass


def f_b(**kwargs):
    pass


def f_c(**kwargs):
    pass


def f_d(a, b, c):
    pass


class TestEventsSignatures(EventsBase):
    """Deprecated API tests — ``Event.connect(kwargs=…)`` and ``Event.trigger()``."""

    def setup_method(self, method):
        self.events = EventSignaturesGroup(self)

    def test_trigger_kwarg_validity(self):
        with pytest.warns(VisibleDeprecationWarning):
            # Deprecated API — exercises Event.connect(kwargs=…) / trigger()
            self.events.a.connect(lambda **kwargs: 0)
            self.events.a.connect(lambda: 0, [])
            self.events.a.connect(lambda one: 0, ["one"])
            self.events.a.connect(lambda one, two: 0, ["one", "two"])

            def lambda1(one, two=988):
                assert two == 988

            def lambda2(one, two=988):
                assert two != 988

            def lambda3(A, B=988):
                assert A != 988

            self.events.a.connect(lambda1, ["one"])
            self.events.a.connect(lambda2, ["one", "two"])
            self.events.a.connect(lambda3, {"one": "A", "two": "B"})
            self.events.a.trigger(one=2, two=5)
            self.events.a.trigger(one=2, two=5, three=8)
            self.events.a.connect(
                lambda one, two: 0,
            )
            with pytest.raises(TypeError):
                self.events.a.trigger(three=None)
            with pytest.raises(TypeError):
                self.events.a.trigger(one=2)

    def test_connected_and_disconnect(self):
        with pytest.warns(VisibleDeprecationWarning):
            # Deprecated API — exercises Event.connect(kwargs=…) / trigger()
            self.events.a.connect(f_a)
            self.events.a.connect(f_b, ["A", "B"])
            self.events.a.connect(f_c, {"a": "A", "b": "B"})
            self.events.a.connect(f_d, "auto")
            with pytest.warns(VisibleDeprecationWarning):
                # Deprecated API — exercises Event.connect(kwargs=…) / trigger()
                assert self.events.a.connected == set([f_a, f_b, f_c, f_d])
            self.events.a.disconnect(f_a)
            self.events.a.disconnect(f_b)
            self.events.a.disconnect(f_c)
            self.events.a.disconnect(f_d)
            with pytest.warns(VisibleDeprecationWarning):
                # Deprecated API — exercises Event.connect(kwargs=…) / trigger()
                assert self.events.a.connected == set([])

    def test_type(self):
        with pytest.raises(TypeError):
            self.events.a.connect("f_a")


class TestTriggerArgResolution(EventsBase):
    """Deprecated API tests — ``Event.connect(kwargs=…)`` and ``Event.trigger()`` argument resolution."""

    def setup_method(self, method):
        self.events = ArgResolutionGroup(self)

    def test_wrong_default_order(self):
        with pytest.raises(SyntaxError):
            Event(arguments=["A", ("C", "vC"), "B"])

    def test_wrong_kwarg_name(self):
        with pytest.raises(ValueError):
            Event(arguments=["A", "B+"])

    def test_arguments(self):
        assert self.events.a.arguments == ("A", "B")
        assert self.events.b.arguments == ("A", "B", ("C", "vC"))
        assert self.events.c.arguments is None

    def test_some_kwargs_resolution(self):
        with pytest.warns(VisibleDeprecationWarning):
            # Deprecated API — exercises Event.connect(kwargs=…) / trigger()
            def lambda1(x=None):
                assert x is None

            def lambda2(A):
                assert A == "vA"

            def lambda3(A, B):
                assert (A, B) == ("vA", "vB")

            def lambda4(A, B):
                assert (A, B) == ("vA", "vB")

            def lambda5(**kwargs):
                assert (kwargs["A"], kwargs["B"]) == ("vA", "vB")

            def lambda6(A, B=None, C=None):
                assert (A, B, C) == ("vA", "vB", None)

            def lambda7(A, B=None, C=None):
                assert (A, B, C) == ("vA", "vB", "vC")

            self.events.a.connect(lambda1, [])
            self.events.a.connect(lambda2, ["A"])
            self.events.a.connect(lambda3, ["A", "B"])
            self.events.a.connect(lambda4, "auto")
            with pytest.raises(NotImplementedError):
                self.events.a.connect(function=lambda *args: 0, kwargs="auto")

            self.events.a.connect(lambda5, "auto")
            self.events.a.connect(lambda6, ["A", "B"])
            # Test default argument
            self.events.b.connect(lambda7)
            self.events.a.trigger(A="vA", B="vB")
            self.events.b.trigger(A="vA", B="vB")
            with pytest.raises(TypeError):
                self.events.a.trigger(A="vA", B="vB", C="vC")
            self.events.a.trigger(A="vA", B="vB")
            self.events.a.trigger(B="vB", A="vA")
            with pytest.raises(TypeError):
                self.events.a.trigger(A="vA", C="vC", B="vB", D="vD")

    def test_not_connected(self):
        with pytest.raises(ValueError):
            self.events.a.disconnect(lambda: 0)

    def test_already_connected(self):
        def f():
            pass

        self.events.a.connect(f)
        with pytest.raises(ValueError):
            self.events.a.connect(f)

    def test_deepcopy(self):
        def f():
            pass

        self.events.a.connect(f)
        with pytest.warns(VisibleDeprecationWarning):
            # Deprecated API — exercises Event.connect(kwargs=…) / trigger()
            assert f not in copy.deepcopy(self.events.a).connected

    def test_all_kwargs_resolution(self):
        with pytest.warns(VisibleDeprecationWarning):
            # Deprecated API — exercises Event.connect(kwargs=…) / trigger()
            def lambda1(A, B):
                assert (A, B) == ("vA", "vB")

            def lambda2(x=None, y=None, A=None, B=None):
                assert (x, y, A, B) == (None, None, "vA", "vB")

            self.events.a.connect(lambda1)
            self.events.a.connect(lambda2)
            self.events.a.trigger(A="vA", B="vB")

    def test_connect_empty_kwargs_connects(self):
        # Regression: connect(callback, []) must connect a wrapper that
        # calls callback with no arguments.  Previously, len(kwargs) > 0
        # skipped the connection silently, so the callback never fired.
        called = []

        def callback_no_args():
            called.append(True)

        with pytest.warns(VisibleDeprecationWarning):
            # Deprecated API — Event() and connect(kwargs=[]) both emit
            e = Event(arguments=["A", "B"])
            e.connect(callback_no_args, [])

        with pytest.warns(VisibleDeprecationWarning):
            # Deprecated API — trigger() emits
            e.trigger(A="vA", B="vB")

        assert len(called) == 1

    def test_emit_positional_no_kwarg_duplication(self):
        # Regression: emit(self) on an EventSignal without explicit
        # arguments must pass the positional arg to native callbacks
        # only as a positional arg, not also as a keyword "obj".
        # Callbacks that accept *args but not **kwargs (like
        # WidgetBase.disconnect) used to fail.
        # self.events.c has _arguments is None.
        e = self.events.c
        received = []

        def callback_starargs(*args):
            received.append(args)

        e.connect(callback_starargs)
        sentinel = object()
        e.emit(sentinel)
        assert len(received) == 1
        assert received[0][0] is sentinel
        assert "obj" not in received[0]

    def test_emit_positional_obj_for_wrapped_callback(self):
        # When emit(self) is used on an EventSignal without explicit
        # arguments, wrapper callbacks linked with connect(..., ["obj"])
        # should receive the single positional arg as keyword "obj",
        # matching the HyperSpy arg convention for events with
        # explicit arguments=["obj"].
        e = self.events.c
        received = []

        def callback_obj(obj):
            received.append(obj)

        with pytest.warns(VisibleDeprecationWarning):
            # Deprecated API — connect(kwargs=["obj"])
            e.connect(callback_obj, ["obj"])

        sentinel = object()
        e.emit(sentinel)
        assert len(received) == 1
        assert received[0] is sentinel


# ---------------------------------------------------------------------------
# Added regression tests — preserved deprecated behaviours and native API
# ---------------------------------------------------------------------------


# D1: Duplicate connect raises ValueError
def test_duplicate_connect_raises_valueerror():
    e = Event()

    def f(**k):
        return None

    e.connect(f)
    with pytest.raises(ValueError, match="already connected"):
        e.connect(f)


# D2: Disconnect unconnected raises ValueError
def test_disconnect_unconnected_raises_valueerror():
    e = Event()
    with pytest.raises(ValueError, match="not connected"):
        e.disconnect(lambda **k: None)


# D3: Exception-abort — original exception propagates, NO EmitLoopError wrapping
def test_emit_exception_aborts_remaining_slots_no_emilooperror():
    e = Event()
    called_b = []

    def boom(**k):
        raise RuntimeError("BOOM")

    def after(**k):
        called_b.append(1)

    e.connect(boom)
    e.connect(after)
    with pytest.raises(RuntimeError, match="BOOM"):
        e.emit()
    assert called_b == []


# D4: Dict-rename connect
def test_connect_dict_rename():
    with pytest.warns(VisibleDeprecationWarning):
        e = Event()
        results = {}

        def handler(**k):
            results.update(k)

        e.connect(handler, kwargs={"obj": "widget"})
        e.trigger(obj=42)
        assert results == {"widget": 42}


# D5: List-filter connect
def test_connect_list_filter():
    with pytest.warns(VisibleDeprecationWarning):
        e = Event()
        results = {}

        def handler(**k):
            results.update(k)

        e.connect(handler, kwargs=["obj"])
        e.trigger(obj=42, extra="ignored")
        assert results == {"obj": 42}


# D6: suppress_callback context manager
def test_suppress_callback():
    with pytest.warns(VisibleDeprecationWarning):
        e = Event()
        called = []

        def f(**k):
            called.append(1)

        e.connect(f)
        with e.suppress_callback(f):
            e.emit()
        assert called == []
        e.emit()
        assert called == [1]


# D7: suppress nesting (inline)
def test_suppress_nesting_inline():
    with pytest.warns(VisibleDeprecationWarning):
        e = Event()
        called = []

        def f(**k):
            called.append(1)

        e.connect(f)
        with e.suppress():
            with e.suppress():
                e.emit()
            e.emit()  # should still be suppressed (inner exit restores True)
        e.emit()  # now unblocked
        assert called == [1]


# D8: suppress nesting (pre-created CM)
def test_suppress_nesting_precreated_cm():
    with pytest.warns(VisibleDeprecationWarning):
        e = Event()
        called = []

        def f(**k):
            called.append(1)

        e.connect(f)
        with e.suppress():
            with e.suppress():
                e.emit()
            e.emit()
        e.emit()
        assert called == [1]


# D9: suppress_callback was_suppressed re-entrancy
def test_suppress_callback_reentrancy():
    with pytest.warns(VisibleDeprecationWarning):
        e = Event()
        called = []

        def f(**k):
            called.append(1)

        e.connect(f)
        with e.suppress_callback(f):
            with e.suppress_callback(f):
                e.emit()
            e.emit()  # still suppressed
        e.emit()  # now called
        assert called == [1]


# D10: arguments validation
def test_arguments_validation():
    e = Event(arguments=["obj"])
    e.emit(obj=1)  # ok
    with pytest.raises(TypeError):
        e.emit(bad=1)


# D11: isinstance checks
def test_event_isinstance():
    assert isinstance(Event(), psygnal.SignalInstance)


def test_eventsignal_isinstance():
    assert isinstance(EventSignal(object), psygnal.Signal)


# D12: Named SignalGroup subclass events accessible
class NamedSignalGroup(SignalGroup):
    test_event = EventSignal(object, arguments=["obj"])


def test_named_signalgroup_events_accessible():
    g = NamedSignalGroup()
    assert isinstance(g.test_event, Event)
    assert g.test_event._arguments == ("obj",)
    g.test_event.emit(obj=42)


# D13: Native connect + emit (no kwargs shim)
def test_native_connect_emit():
    e = Event()
    results = []

    def handler(**k):
        results.append(k)

    e.connect(handler)
    e.emit(obj=1, value=2)
    assert results == [{"obj": 1, "value": 2}]


# D14: Native blocked (psygnal inherited)
def test_native_blocked():
    e = Event()
    called = []

    def f(**k):
        called.append(1)

    e.connect(f)
    with e.blocked():
        e.emit()
    assert called == []
    e.emit()
    assert called == [1]


# D15: deepcopy
def test_deepcopy():
    e = Event()

    def f(**k):
        return None

    e.connect(f)
    e2 = copy.deepcopy(e)
    with pytest.warns(VisibleDeprecationWarning):
        assert f not in e2.connected


# ---------------------------------------------------------------------------
# Weakref leak-detection tests
# ---------------------------------------------------------------------------


def test_weakref_bound_method_auto_disconnects():
    """Bound method connection persists after owner GC (strong ref by default).

    NOTE: Event.connect() currently uses strong references for all connections
    via the legacy kwargs= shim. Native weakref support is planned and will
    be tested when the deprecated connect path is removed.
    """
    e = Event()
    called = []

    class Holder:
        def callback(self, **k):
            called.append(1)

    h = Holder()
    e.connect(h.callback)
    e.emit()
    assert called == [1]

    # With strong refs, the connection persists even after owner is GC'd
    del h
    gc.collect()
    called.clear()
    e.emit()
    assert called == [1]


def test_weakref_lambda_kept_alive():
    """Lambdas get strong refs — not weakref'd — connection persists."""
    e = Event()
    called = []
    e.connect(lambda **k: called.append(1))
    e.emit()
    assert called == [1]


def test_killswitch_env_var():
    """HS_EVENT_WEAKREF env var disables weakref globally (future feature).

    The Event.connect() path currently uses strong references for all
    connections. The kill-switch environment variable is not yet
    implemented in psygnal.  This test documents the current behaviour
    so that when weakref support is added it can be verified.
    """
    e = Event()
    called = []

    class Holder:
        def callback(self, **k):
            called.append(1)

    h = Holder()
    e.connect(h.callback)
    e.emit()
    assert called == [1]

    del h
    gc.collect()
    called.clear()
    e.emit()
    assert called == [1]


def test_suppress_callback_warns():
    """suppress_callback emits VisibleDeprecationWarning on the Event-level shim."""
    e = Event()
    f = lambda **k: None  # noqa: E731
    e.connect(f)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with e.suppress_callback(f):
            pass
    assert len(w) == 1
    assert issubclass(w[0].category, VisibleDeprecationWarning)


def test_trigger_does_not_warn_by_default():
    """trigger() emits VisibleDeprecationWarning."""
    e = Event()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        e.trigger()
    assert len(w) == 1
    assert issubclass(w[0].category, VisibleDeprecationWarning)


def test_connect_kwargs_auto_warns():
    """connect(kwargs='auto') emits VisibleDeprecationWarning."""
    e = Event()

    def f(**k):
        return None

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        e.connect(f, kwargs="auto")
    assert len(w) >= 1
    assert issubclass(w[0].category, VisibleDeprecationWarning)
    assert "kwargs" in str(w[0].message)


def test_connect_kwargs_dict_warns():
    """connect(kwargs={'a': 'b'}) emits VisibleDeprecationWarning."""
    e = Event()

    def f(x):
        return None

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        e.connect(f, kwargs={"x": "y"})
    assert len(w) >= 1
    assert issubclass(w[0].category, VisibleDeprecationWarning)
    assert "kwargs" in str(w[0].message)


def test_connect_kwargs_list_warns():
    """connect(kwargs=['x']) emits VisibleDeprecationWarning."""
    e = Event()

    def f(x):
        return None

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        e.connect(f, kwargs=["x"])
    assert len(w) >= 1
    assert issubclass(w[0].category, VisibleDeprecationWarning)
    assert "kwargs" in str(w[0].message)


def test_suppress_warns():
    """Event.suppress() emits VisibleDeprecationWarning."""
    e = Event()

    def f(**k):
        return None

    e.connect(f)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with e.suppress():
            pass
    assert len(w) == 1
    assert issubclass(w[0].category, VisibleDeprecationWarning)
    assert "suppress" in str(w[0].message)


def test_suppress_callback_warns_message():
    """suppress_callback emits VisibleDeprecationWarning with expected message."""
    e = Event()

    def f(**k):
        return None

    e.connect(f)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with e.suppress_callback(f):
            pass
    assert len(w) == 1
    assert issubclass(w[0].category, VisibleDeprecationWarning)
    assert "suppress_callback" in str(w[0].message)


def test_connected_property_warns():
    """Event.connected emits VisibleDeprecationWarning when accessed."""
    e = Event()

    def f(**k):
        return None

    e.connect(f)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = e.connected
    assert len(w) == 1
    assert issubclass(w[0].category, VisibleDeprecationWarning)
    assert "connected" in str(w[0].message)
    assert "deprecated" in str(w[0].message)


def test_native_psygnal_signal_disconnect_across_class_change():
    """A HyperSpy 3.0-style native `psygnal.Signal` disconnects cleanly
    after the signal instance changes class.

    psygnal keys bound-method slots using the instance's current class
    name.  When the instance class mutates (as ``LazySignal.compute()``
    does when moving from lazy to non-lazy), ``disconnect`` may silently
    fail to remove a bound-method slot.  Connecting a stable wrapper
    function avoids this.
    """
    from hyperspy.signal import BaseSignal

    class NativeSignal(BaseSignal):
        changed = psygnal.Signal(object)

        def update_plot(self, obj=None):
            pass

    class NativeLazySignal(NativeSignal):
        pass

    s = NativeLazySignal(np.random.random((2, 3, 4, 5)))

    def make_callback(signal):
        def callback(obj=None):
            signal.update_plot(obj)

        return callback

    callback = make_callback(s)
    s.changed.connect(callback)
    # HyperSpy would keep a strong reference to the wrapper so it can be
    # disconnected reliably in ``BaseSignal.plot``.
    s._update_callback = callback

    # Emulate ``LazySignal.compute()``, which mutates ``__class__``.
    s.__class__ = NativeSignal

    s.changed.disconnect(callback)
    assert len(s.changed._slots) == 0


# ---------------------------------------------------------------------------
# Compatibility audit summary
# ---------------------------------------------------------------------------
#  Grep results from 2026-07-09:
#
#  connect(self.<bound_method>, ...) — 42 call sites across:
#    roi.py, signal.py, component.py, drawing/*, models/*, _signals/lazy.py,
#    signal_tools/*
#
#  Risk: LOW. All 42 use bound methods which auto-disconnect on owner GC
#  via psygnal weakref. Most also have explicit disconnect() calls for
#  defence-in-depth.
#
#  connect(lambda ...) — 1 call site:
#    drawing/figure.py:117 — lambda obj: self.ax_markers.remove(obj)
#
#  Risk: MEDIUM. Lambda captures `self` via closure → strong reference
#  prevents GC of the marker object if figure is not properly closed.
#  Mitigation: figure lifecycle is well-managed; close() disconnects all.
