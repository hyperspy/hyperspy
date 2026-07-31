
.. _events-label:

Events
******

.. seealso::

    For migrating code to the psygnal-backed event system introduced in
    HyperSpy 3.0, see the :ref:`events_migration`.

Events are a mechanism to send notifications. HyperSpy events are
decentralised, meaning that there is not a central events dispatcher.
Instead, each object that can emit events has an ``events``
attribute that is an instance of :external:class:`psygnal.SignalGroup` and that contains
instances of :external:class:`psygnal.SignalInstance` (HyperSpy's
:class:`~.events.Event` class is a deprecated subclass of it). When
triggered the first keyword argument, ``obj``, contains the object that
the event belongs to. Different events may be triggered by other keyword
arguments too.

Connecting to events
--------------------

The following example shows how to connect to the ``index_changed`` event of
:class:`~.axes.DataAxis` that is triggered with ``obj`` and ``index`` keywords:

.. code-block:: python

    >>> s = hs.signals.Signal1D(np.random.random((10,100)))
    >>> nav_axis = s.axes_manager.navigation_axes[0]
    >>> nav_axis.name = "x"
    >>> def on_index_changed(obj, index):
    ...    print("on_index_changed_called")
    ...    print("Axis name: ", obj.name)
    ...    print("Index: ", index)

    >>> nav_axis.events.index_changed.connect(on_index_changed)
    >>> s.axes_manager.indices = (3,)
    on_index_changed_called
    Axis name: x
    Index: 3
    >>> s.axes_manager.indices = (9,)
    on_index_changed_called
    Axis name: x
    Index: 9

By default every keyword argument passed to
:external:func:`~psygnal.SignalInstance.emit` is forwarded to the connected
callback.  To connect a function whose signature does not match the event's
keyword arguments, wrap it in an adapter.  For example, to forward only the
``index`` keyword argument to ``on_index_changed2``:

.. code-block:: python

    >>> def on_index_changed2(index):
    ...    print("on_index_changed2_called")
    ...    print("Index: ", index)

    >>> nav_axis.events.index_changed.connect(
    ...     lambda obj, index: on_index_changed2(index)
    ... )
    >>> s.axes_manager.indices = (0,)
    on_index_changed_called
    Axis name: x
    Index: 0
    on_index_changed2_called
    Index: 0

To pass no arguments at all to the callback, discard them in the adapter:

.. code-block:: python

    >>> def on_index_changed3():
    ...    print("on_index_changed3_called")

    >>> nav_axis.events.index_changed.connect(
    ...     lambda **kwargs: on_index_changed3()
    ... )
    >>> s.axes_manager.indices = (1,)
    on_index_changed_called
    Axis name: x
    Index: 1
    on_index_changed2_called
    Index: 1
    on_index_changed3_called

Or allow the callback to accept keyword arguments

.. code-block:: python

    >>> def on_index_changed4(**kwargs):
    ...    print("on_index_changed4_called")

    >>> nav_axis.events.index_changed.connect(on_index_changed3)
    >>> s.axes_manager.indices = (1,)
    on_index_changed_called
    Axis name: x
    Index: 1
    on_index_changed2_called
    Index: 1
    on_index_changed3_called
    on_index_changed4_called

Keyword arguments can also be renamed when forwarding them to the callback:

.. code-block:: python

    >>> def on_index_changed5(arg):
    ...    print("on_index_changed4_called")
    ...    print("Index: ", arg)

    >>> nav_axis.events.index_changed.connect(
    ...     lambda obj, index: on_index_changed4(arg=index)
    ... )
    >>> s.axes_manager.indices = (4,)
    on_index_changed_called
    Axis name: x
    Index: 4
    on_index_changed2_called
    Index: 4
    on_index_changed3_called
    on_index_changed4_called
    on_index_changed5_called
    Index: 4

Suppressing events
------------------

The following example shows how to block all callbacks of a given event and
all callbacks of all events of an object using the
:external:func:`~psygnal.SignalInstance.blocked` context manager.

.. code-block:: python

    >>> with nav_axis.events.index_changed.blocked():
    ...    s.axes_manager.indices = (6,)

    >>> with nav_axis.events.blocked():
    ...    s.axes_manager.indices = (5,)

To temporarily disable a single callback, disconnect it and reconnect it
afterwards.  When using adapter functions (as above), keep a reference to the
adapter so you can pass the same object to
:external:func:`~psygnal.SignalInstance.disconnect`:

.. code-block:: python

    >>> _adapter = lambda obj, index: on_index_changed2(index)
    >>> nav_axis.events.index_changed.connect(_adapter)
    >>> nav_axis.events.index_changed.disconnect(_adapter)
    >>> s.axes_manager.indices = (7,)
    on_index_changed_called
    Axis name: x
    Index: 7
    on_index_changed3_called
    on_index_changed4_called
    Index: 7
    >>> nav_axis.events.index_changed.connect(_adapter)

Triggering events
-----------------

Although usually there is no need to trigger events manually, there are
cases where it is required. When triggering events manually it is important
to pass the right keywords as specified in the event docstring. In the
following example we change the :attr:`~.api.signals.BaseSignal.data` attribute of a
:class:`~.api.signals.BaseSignal` manually and we then emit the ``data_changed``
event using :external:func:`~psygnal.SignalInstance.emit`.

.. code-block:: python

    >>> s = hs.signals.Signal1D(np.random.random((10,100)))
    >>> s.data[:] = 0
    >>> s.events.data_changed.emit(obj=s)
