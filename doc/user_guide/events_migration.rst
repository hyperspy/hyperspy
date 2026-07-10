.. _events_migration:

Events Migration Guide
======================

HyperSpy 3.0 introduces a significant update to the event system, moving to a psygnal-backed implementation. This guide outlines the changes and provides instructions for migrating your code.

1. What changes in HyperSpy 3.0
-------------------------------

The deprecated Event methods and classes are removed in HyperSpy 3.0. The event system now uses the psygnal-native surface.

* **Removed methods**: ``trigger``, ``connect`` with ``kwargs=``, ``suppress``, ``suppress_callback``, ``.connected``, and ``arguments=``.
* **Native API**: Use ``emit(**kwargs)``, ``connect(callback)``, ``disconnect(callback)``, ``blocked()``, ``block()``, and ``unblock()``.
* **EventSuppressor**: This class is removed. Use ``SignalGroup.blocked()`` instead.
* **Event Declaration**: Events must be declared as ``EventSignal`` class attributes on named ``SignalGroup`` subclasses.

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

    from hyperspy.events import SignalGroup, EventSignal

    class MyEvents(SignalGroup):
        my_event = EventSignal(argnames=("obj",))

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

Group-level connections
~~~~~~~~~~~~~~~~~~~~~~~

You can now connect to all events in a group using ``SignalGroup.all``.

.. code-block:: python

    def handler(info):
        print(f"Event {info.signal_name} emitted with {info.args}")

    events.all.connect(handler)

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
