.. _plotting-label:

Interactive Plotting
====================
Interactive plotting in hyperspy is driven through event handling and rendered
through a pluggable plotting backend (see :ref:`plotting-backends-label`).

Specifically, for some signal ``s``, when the ``index`` value for some :class:`~.axes.BaseDataAxis`
is changed, then the signal plot is updated to reflect the data at that index.  Each signal has a
``_get_current_data`` function, which will return the data at the current navigation index.

For lazy signals, the ``_get_current_data`` function works slightly differently as the current chunk is cached.  As a result,
the ``_get_current_data`` function first checks if the current chunk is cached and then either computes the chunk where the
navigation index resides or just pulls the value from the cached chunk.

.. _plotting-backends-label:

Plotting backends
-----------------

The generic drawing layer in ``hyperspy/drawing`` (figures, explorers,
widgets, markers) draws through the active backend, returned by
``hyperspy.drawing.backends.get_backend()``, rather than calling a plotting
library itself.  Matplotlib is the default; the backend is selected with
``hs.preferences.Plot.backend``.  The remaining direct uses of matplotlib
are the matplotlib-only composers in ``drawing/utils.py`` (such as
:func:`~.api.plot.plot_images`), raw-``Collection`` markers and the
span/polygon selectors described below; none of them is imported when
``hyperspy.api`` is.

Backends are discovered through the ``hyperspy.backends`` entry-point group.
A third-party package registers one in its ``pyproject.toml``:

.. code-block:: toml

    [project.entry-points."hyperspy.backends"]
    mybackend = "my_package.backend:MyBackend"

.. note::

   The backend protocol is experimental: it may change before it is
   declared stable.

The contract lives in ``hyperspy/drawing/backends``:

``_protocol.py``
    ``PlottingBackend``, the core protocol every backend implements, made of
    ``BlitMixin`` (optional blitting, no-op defaults), ``PointerMixin``
    (widget primitives; unsupported ones raise ``BackendCapabilityError``) and
    the core drawing methods.  ``ArtistBackend`` is an optional extension,
    described below.
``_events.py``
    The input events handed to every event handler.
``_stub.py``
    An annotated template to start a new backend from.

A backend must override ``get_explorer``: the protocol's default returns the
abstract ``HyperExplorer``.  Widget and marker primitives a backend cannot
provide raise ``BackendCapabilityError``; most callers catch it and degrade
(for example, a scale bar is skipped), while a user request that cannot be
honoured at all, such as a marker built from a raw matplotlib
``Collection``, raises it to the user.

Input events
^^^^^^^^^^^^

Handlers connected with ``connect_key_press``, ``connect_mouse_move``,
``connect_mouse_press``, ``connect_mouse_release`` (and ``connect_pick``, see
below) receive the dataclasses of ``_events.py`` (``KeyEvent``,
``PointerEvent`` and ``PickEvent``), never the backend's own event object.
The backend converts its native event before calling the handler; the
native event remains available as ``event.native``.  The conversion
normalises what differs between libraries:

* key names are canonical: lower-case named keys (``"left"``, ``"pageup"``,
  ``"f5"``) prefixed by the held modifiers in the order ``ctrl``, ``alt``,
  ``super``, ``shift`` (``"ctrl+left"``).  Printable characters keep their
  case and are not prefixed with ``shift`` (``"A"``, ``"+"``).
  ``canonical_key`` and ``split_key`` convert the common spellings, such as
  the browser's ``"ArrowLeft"`` or Qt's ``"meta"``.
* mouse buttons are numbered from 1 (left) to 3 (right).
* positions are given in data coordinates (``xdata``, ``ydata``; ``None``
  outside the axes, in which case ``inaxes`` is ``None`` too) and in display
  coordinates (``x``, ``y``).
* held modifiers are a frozenset drawn from ``{"ctrl", "alt", "shift",
  "super"}``.

Widget interaction
^^^^^^^^^^^^^^^^^^

Widgets (navigation pointers and ROI widgets, in ``drawing/widget.py`` and
``drawing/_widgets``) own their geometry and the events they emit; the
backend only draws them and reports user input.  There are two ways the
drags reach a widget, and a backend uses one or the other:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * -
     - Native widgets (core protocol)
     - Pickable artists (``ArtistBackend``)
   * - Used by
     - anyplotlib
     - matplotlib
   * - Drawing
     - ``create_*_pointer`` creates an interactive widget of the plotting
       library.
     - ``create_*_pointer`` creates a plain artist; resize handles are extra
       artists from ``create_rect_patch``.
   * - Dragging
     - The library moves the widget and calls the callback registered with
       ``connect_widget_drag`` with the new geometry in data coordinates.
       The widget's ``_on_widget_drag`` stores it.
     - A ``PickEvent`` (``connect_pick``) selects the widget, then each
       ``PointerEvent`` from ``connect_mouse_move`` moves it in
       ``_onmousemove`` until the button is released.
   * - Selection
     - None needed.
     - ``select()`` calls ``simulate_pick``.

Both paths end in the same place: the widget's position or size property is
set, which updates the axes manager or ROI and fires the widget's events.
Whatever the path, every backend implements ``connect_widget_drag`` (an
artist backend as a no-op) and the pointer connections: widgets always
connect ``connect_mouse_move``, ``connect_mouse_press`` (shift-click jumps a
navigation pointer) and ``connect_mouse_release``.  Only ``connect_pick``
belongs to the extension.
The line, rectangle, circle and line2d widgets implement both entry points
(``_on_widget_drag`` and the pick/move handlers), so they work on either
kind of backend; ``LabelWidget`` can only be dragged on an artist backend.
Keyboard handling (``+``/``-`` resizing, navigation keys) uses
``connect_key_press`` on both.

The generic layer calls ``ArtistBackend`` methods only when
``isinstance(backend, ArtistBackend)`` holds, through
``get_artist_backend()``, which returns ``None`` otherwise.  A backend must
therefore define all of the extension's methods or none of them.  Besides
the pick path, the extension provides the artist-only features; without it,
widgets have no separate resize handles, markers built from a raw matplotlib
``Collection`` raise ``BackendCapabilityError``, and ``tight_layout`` is
skipped.  A new backend should normally use native widgets and not
implement the extension.

The span and polygon selectors used by ``RangeWidget`` and ``PolygonWidget``
(``create_span_selector`` and ``create_polygon_selector``) still expose the
API of matplotlib's ``SpanSelector`` and ``PolygonSelector``; the anyplotlib
backend provides adapters with that API.

Testing a backend
^^^^^^^^^^^^^^^^^

``hyperspy/tests/drawing/backend_conformance.py`` holds behavioural tests
that every backend runs: the protocol shape, the drawing primitives, the
widget primitives (including a native drag or an artist pick), the input
event contract and end-to-end plotting of signals and ROIs.  A backend runs
them by subclassing ``BackendConformance`` with a ``backend`` fixture and a
``driver`` fixture, a ``BackendDriver`` subclass that emits the backend's
own native events exactly as the library would (``fire_key``,
``fire_pointer``, and ``drag_pointer`` for native widgets); see
``test_backend_conformance.py`` for the matplotlib and anyplotlib drivers.
``backend_name`` must be the name of an installed entry point: the
end-to-end tests select the backend by that name and fail if it does not
activate.  Features a backend declares unsupported with
``BackendCapabilityError`` are skipped.  Third-party backends can run the
same suite from their own test suite:

.. code-block:: python

    import pytest

    # Show assertion values from the shared suite.
    pytest.register_assert_rewrite("hyperspy.tests.drawing.backend_conformance")

    from hyperspy.tests.drawing.backend_conformance import (
        BackendConformance,
        BackendDriver,
    )

    class TestMyBackendConformance(BackendConformance):
        backend_name = "mybackend"  # the entry-point name

        @pytest.fixture
        def backend(self):
            return MyBackend()

        @pytest.fixture
        def driver(self):
            return MyDriver()  # a BackendDriver subclass
