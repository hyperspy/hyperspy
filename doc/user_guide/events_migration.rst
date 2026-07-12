.. _events_migration:

Events Migration Guide
======================

HyperSpy 3.0 introduces a significant update to the event system, moving to a psygnal-backed implementation. This guide outlines the changes and provides instructions for migrating your code.

1. What changes in HyperSpy 3.0
-------------------------------

The deprecated Event methods and classes are removed in HyperSpy 3.0. The event system now uses the psygnal-native surface.

* **Removed methods**: ``trigger``, ``connect`` with ``kwargs=``, ``suppress``, ``suppress_callback``, ``.connected``, and ``arguments=``.
* **psygnal native API**: Use ``emit(**kwargs)``, ``connect(callback)``, ``disconnect(callback)``, ``blocked()``, ``block()``, and ``unblock()``.
* **EventSuppressor**: This class is removed. Use ``psygnal.SignalGroup.blocked()`` instead.
* **Event Declaration**: Events must be declared as :class:`psygnal.Signal` class attributes on named :class:`psygnal.SignalGroup` subclasses.

2. Migration guide with code examples
-------------------------------------

The following examples show how to update your code from the old Event API to the new psygnal-native API.

Triggering events
~~~~~~~~~~~~~~~~~

Before:

.. code-block:: python

    events.my_event.trigger(obj=signal)

After:

.. code-block:: python

    events.my_event.emit(obj=signal)

Connecting with keyword arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before:

.. code-block:: python

    events.my_event.connect(my_handler, kwargs=["obj"])
    events.my_event.connect(f, kwargs={"obj": "widget"})

After:

.. code-block:: python

    events.my_event.connect(lambda obj: my_handler(obj=obj))
    events.my_event.connect(lambda obj: f(widget=obj))

Suppressing events
~~~~~~~~~~~~~~~~~~

Before:

.. code-block:: python

    with events.suppress():
        # do something
        pass

After:

.. code-block:: python

    with events.blocked():
        # do something
        pass

Event declaration
~~~~~~~~~~~~~~~~~

Before (dynamic events):

.. code-block:: python

    from hyperspy.events import Events, Event
    events = Events()
    events.my_event = Event()

After (named SignalGroup subclass):

.. code-block:: python

    from psygnal import SignalGroup, Signal
    from hyperspy.axes import AxesManager

    class MyEvents(SignalGroup):
        my_event = Signal(AxesManager)

    events = MyEvents()

Suppressing a specific callback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``suppress_callback`` method is removed. Use a temporary disconnect and reconnect pattern instead.

Before:

.. code-block:: python

    events.my_event.suppress_callback(f)

After:

.. code-block:: python

    events.my_event.disconnect(f)
    # perform actions
    events.my_event.connect(f)

Replacing ``.connected`` introspection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``.connected`` attribute is removed in HyperSpy 3.0. It was commonly used to inspect which callables were subscribed to an event, but psygnal expects callers to **own and manage their own connections** instead of querying the event for its subscribers.

Checking membership
^^^^^^^^^^^^^^^^^^^

Before:

.. code-block:: python

    if my_callback not in events.my_event.connected:
        events.my_event.connect(my_callback)

After — track the connection state on the owner:

.. code-block:: python

    self._my_event_connected = False

    def ensure_connected(self):
        if not self._my_event_connected:
            events.my_event.connect(my_callback)
            self._my_event_connected = True

Or use the handle returned by ``connect`` and clean it up explicitly:

.. code-block:: python

    self._disconnect_my_event = events.my_event.connect(my_callback)

    def cleanup(self):
        self._disconnect_my_event()

Disconnecting all listeners
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before:

.. code-block:: python

    for f in list(events.my_event.connected):
        events.my_event.disconnect(f)

After — keep a list of disconnect handles:

.. code-block:: python

    class MyObject:
        def __init__(self):
            self._event_handles = []

        def add_listener(self, callback):
            self._event_handles.append(events.my_event.connect(callback))

        def close(self):
            for disconnect in self._event_handles:
                disconnect()
            self._event_handles.clear()

Checking the number of listeners
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before:

.. code-block:: python

    assert len(events.my_event.connected) == 0

After:

.. code-block:: python

    # In most cases you do not need to count listeners. Instead,
    # track the callbacks you registered yourself.
    assert len(self._event_handles) == 0

Copying and diffing connection sets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before:

.. code-block:: python

    old = events.my_event.connected.copy()
    # ... add temporary listeners ...
    for f in events.my_event.connected - old:
        events.my_event.disconnect(f)

After — own the list of callbacks you added:

.. code-block:: python

    temp_handles = []
    for callback in temporary_callbacks:
        temp_handles.append(events.my_event.connect(callback))

    # ... later ...
    for disconnect in temp_handles:
        disconnect()
    temp_handles.clear()

Group-level connections
~~~~~~~~~~~~~~~~~~~~~~~

You can now connect to all events in a group using ``SignalGroup.all``.

.. code-block:: python

    def handler(info):
        print(f"Event {info.signal_name} emitted with {info.args}")

    events.all.connect(handler)

Bound methods and mutable classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In HyperSpy, a signal instance can change class during its lifetime
(e.g. ``LazySignal.compute()`` converts a lazy signal to a regular one,
or operations such as ``_assign_subclass`` change the concrete signal type).
Because ``psygnal`` keys bound-method slots using the class name of the bound
instance, ``disconnect`` may fail to find the slot after a class change.

If a callback must be disconnected after a class mutation, connect a
wrapper function instead of a bound method, keep a strong reference to that
wrapper, and disconnect by the same wrapper object:

.. code-block:: python

    from hyperspy.signal import BaseSignal
    from psygnal import Signal

    class MySignal(BaseSignal):
        changed = Signal(object)

        def update_plot(self, obj=None):
            pass

    s = MySignal(data)  # data is a suitable numpy array

    def make_callback(signal):
        def callback(obj=None):
            signal.update_plot(obj)
        return callback

    callback = make_callback(s)
    s.changed.connect(callback)
    s._update_callback = callback  # strong reference

    # ... later, e.g. on figure close or after compute ...
    s.changed.disconnect(callback)

Do **not** rely on ``s.changed.disconnect(s.update_plot)`` for callbacks that
must survive class mutation.

3. New features available in 2.5+
---------------------------------

HyperSpy 2.5+ introduces several new features to the event system:

* **Group-level connections**: Connect to all events in a ``SignalGroup`` via ``SignalGroup.all``.
* **Group-level suppression**: Use ``SignalGroup.blocked()`` to suppress all events in a group at once.
* **Throttling and debouncing**: Use ``Event.throttle(ms)`` and ``Event.debounce(ms)`` to limit the frequency of event emissions.
* **Leak detection**: Set ``max_listeners`` to receive warnings when an event has an unusually high number of subscribers, which can help detect memory leaks.
* **EventedObjectProxy**: Detection of mutation in numpy arrays (experimental/future).

4. Timeline
-----------

* **HyperSpy 2.5**: Psygnal-backed event system introduced with a deprecated Event API shim. The ``Events`` class is removed. Events are declared as ``EventSignal`` on ``SignalGroup`` subclasses. Internal code is migrated to the native API.
* **HyperSpy 3.0**: Deprecated Event methods are removed. Only the psygnal-native API remains.
