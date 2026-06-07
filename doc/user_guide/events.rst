
.. _events-label:

Events
******

Events are a mechanism to send notifications. HyperSpy events are
decentralised, meaning that there is not a central events dispatcher.
Instead, each object that can emit events has an ``events``
attribute that is an instance of :class:`~.events.Events` and that contains
instances of  :class:`~.events.Event` as attributes. When triggered the
first keyword argument, `obj` contains the object that the events belongs to.
Different events may be triggered by other keyword arguments too.

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


It is possible to select the keyword arguments that are passed to the
connected. For example, in the following only the ``index`` keyword argument is
passed to ``on_index_changed2`` and none to ``on_index_changed3``:

.. code-block:: python

    >>> def on_index_changed2(index):
    ...    print("on_index_changed2_called")
    ...    print("Index: ", index)

    >>> nav_axis.events.index_changed.connect(on_index_changed2, ["index"])
    >>> s.axes_manager.indices = (0,)
    on_index_changed_called
    Axis name: x
    Index: 0
    on_index_changed2_called
    Index: 0
    >>> def on_index_changed3():
    ...    print("on_index_changed3_called")

    >>> nav_axis.events.index_changed.connect(on_index_changed3, [])
    >>> s.axes_manager.indices = (1,)
    on_index_changed_called
    Axis name: x
    Index: 1
    on_index_changed2_called
    Index: 1
    on_index_changed3_called


It is also possible to map trigger keyword arguments to connected function
keyword arguments as follows:

.. code-block:: python

    >>> def on_index_changed4(arg):
    ...    print("on_index_changed4_called")
    ...    print("Index: ", arg)

    >>> nav_axis.events.index_changed.connect(on_index_changed4, {"index" : "arg"})
    >>> s.axes_manager.indices = (4,)
    on_index_changed_called
    Axis name: x
    Index: 4
    on_index_changed2_called
    Index: 4
    on_index_changed3_called
    on_index_changed4_called
    Index: 4

Axis change events
------------------

The ``axis_changed`` event is a unified event that triggers whenever any axis 
property changes. This includes changes to name, units, scale, offset, axis array, 
navigation state, binning state, and any other axis properties. This event is 
available for all axis types:

.. code-block:: python

    >>> s = hs.signals.Signal1D(np.random.random((10,100)))
    >>> axis = s.axes_manager.signal_axes[0]
    >>> def on_axis_changed(obj):
    ...    print(f"Axis '{obj.name}' has changed!")
    ...    print(f"Current scale: {getattr(obj, 'scale', 'N/A')}")
    ...    print(f"Current units: {obj.units}")

    >>> axis.events.axis_changed.connect(on_axis_changed)
    >>> axis.name = "Energy"
    Axis 'Energy' has changed!
    Current scale: 1.0
    Current units: <undefined>
    >>> axis.units = "eV"
    Axis 'Energy' has changed!
    Current scale: 1.0
    Current units: eV

For :class:`~.axes.UniformDataAxis`, changes to scale and offset also trigger 
the event:

.. code-block:: python

    >>> uniform_axis = hs.axes.UniformDataAxis(size=100, scale=0.1, offset=0)
    >>> uniform_axis.events.axis_changed.connect(on_axis_changed)
    >>> uniform_axis.scale = 0.2
    Axis '<undefined>' has changed!
    Current scale: 0.2
    Current units: <undefined>
    >>> uniform_axis.offset = 10.0
    Axis '<undefined>' has changed!
    Current scale: 0.2
    Current units: <undefined>

For :class:`~.axes.FunctionalDataAxis`, changes to expression parameters also 
trigger the event:

.. code-block:: python

    >>> func_axis = hs.axes.FunctionalDataAxis(size=10, expression="x ** power", power=2)
    >>> func_axis.events.axis_changed.connect(on_axis_changed)
    >>> func_axis.power = 3  # This will trigger the axis_changed event
    Axis '<undefined>' has changed!
    Current scale: N/A
    Current units: <undefined>

The ``axis_changed`` event provides a unified way to monitor all types of axis 
changes without needing to connect to multiple specific events.

Suppressing events
------------------

The following example shows how to suppress single callbacks, all callbacks of
a given event and all callbacks of all events of an object.

.. code-block:: python

    >>> with nav_axis.events.index_changed.suppress_callback(on_index_changed2):
    ...    s.axes_manager.indices = (7,)
    on_index_changed_called
    Axis name: x
    Index: 7
    on_index_changed3_called
    on_index_changed4_called
    Index: 7
    >>> with nav_axis.events.index_changed.suppress():
    ...    s.axes_manager.indices = (6,)

    >>> with nav_axis.events.suppress():
    ...    s.axes_manager.indices = (5,)


Triggering events
-----------------

Although usually there is no need to trigger events manually, there are
cases where it is required. When triggering events manually it is important
to pass the right keywords as specified in the event docstring. In the
following example we change the :attr:`~.api.signals.BaseSignal.data` attribute of a
:class:`~.api.signals.BaseSignal` manually and we then trigger the ``data_changed``
event.

.. code-block:: python

    >>> s = hs.signals.Signal1D(np.random.random((10,100)))
    >>> s.data[:] = 0
    >>> s.events.data_changed.trigger(obj=s)
