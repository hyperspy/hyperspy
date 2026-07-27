"""anyplotlib plotting backend for hyperspy.

Targets anyplotlib >= 0.5.0.  Several things that used to need a workaround
here are native upstream as of that release and are used directly:

* ``Widget.set(_notify=False)`` instead of wrapping every Python-initiated
  mutation in ``pause_events()``.
* ``plot_box`` / ``data_to_display`` / ``display_to_data`` instead of
  re-deriving the renderer's padding constants and letterbox maths here.
* ``size_units="px"`` for markers whose sizes are display points.
* Native ``line`` / ``vline`` / ``hline`` widget kinds.
* ``snap_values`` and ``orientation="vertical"`` on the range widget.
* ``set_scalebar_style`` for the scale bar's colour.
"""

from __future__ import annotations

import numpy as np

from hyperspy.drawing.backends._protocol import BackendCapabilityError

_NOT_YET = "anyplotlib does not yet support '{}'. See docs/anyplotlib_improvements.md."


def _unwrap_cycling(value):
    """Return *value* unchanged, or unwrap a 1-element cycling sequence to a scalar.

    HyperSpy stores singleton style values as ``(v,)`` tuples so that MPL
    collections cycle through them.  anyplotlib expects either a bare scalar
    or an array whose length matches the number of markers; a 1-element list
    fails ``_broadcast`` when n > 1, so we flatten it here.
    """
    if hasattr(value, "__len__") and not isinstance(value, str) and len(value) == 1:
        v = value[0]
        if isinstance(v, str):
            return v
        try:
            return float(v)
        except (TypeError, ValueError):
            # e.g. an RGBA tuple wrapped for cycling — hand back the element.
            return v.tolist() if hasattr(v, "tolist") else v
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


# Per-marker kwargs that anyplotlib broadcasts against the number of markers
# (or coerces to a scalar).  HyperSpy sends any of them as a 1-element cycling
# sequence when the user gave a single value, which ``_broadcast`` rejects
# whenever there is more than one marker.
_CYCLING_KWARGS = frozenset(
    {
        "widths",
        "heights",
        "angles",
        "sizes",
        "radius",
        "U",
        "V",
        "linewidths",
        "edgecolors",
        "facecolors",
        "hover_edgecolors",
        "hover_facecolors",
    }
)


class _AplFigureProxy:
    """Proxy for a single axes panel within a shared anyplotlib Figure.

    When signal and navigator share one anyplotlib Figure (combined layout),
    each plot gets its own proxy. Both proxies share the same underlying
    anyplotlib Figure widget so that ``display()`` is only called once.
    """

    def __init__(self, real_fig, ax):
        self._real_fig = real_fig  # shared anyplotlib Figure widget
        self._hspy_ax = ax
        self._hspy_on_close = None


def _snap(value, snap_values):
    """Return the entry of *snap_values* closest to *value* (identity if None).

    Only used for initial extents; once the widget exists, anyplotlib's own
    ``snap_values`` handles snapping inside the drag.
    """
    if snap_values is None:
        return value
    arr = np.asarray(snap_values, dtype=float)
    if arr.size == 0:
        return value
    return float(arr[np.argmin(np.abs(arr - float(value)))])


# ---------------------------------------------------------------------------
# Widget coordinates
#
# Overlay widgets on a ``Plot2D`` are positioned in *image pixel indices* —
# ``add_rectangle_widget`` derives its defaults from ``image_width``, and the
# renderer maps index ``i`` to ``(i + 0.5) / image_width`` of the drawn image
# rect.  HyperSpy speaks calibrated units throughout, and the two coincide only
# when ``scale == 1`` and ``offset == 0``.  On a calibrated image (say 0.015
# nm/px) an ROI spanning 1.91–5.75 nm was therefore drawn at pixel 1.9 with a
# width of 3.8 px: a few-pixel smudge in the corner, impossible to grab.
#
# The panel's own ``x_axis`` / ``y_axis`` arrays carry one coordinate per pixel,
# so interpolating against them converts both ways and copes with non-uniform
# axes for free.  1-D panels have no ``image_width`` and are already in data
# units, so they are left alone.
# ---------------------------------------------------------------------------


def _pixel_centres(edge0, edge1, count):
    """Return the coordinate of each pixel centre across an extent.

    ``extent`` describes the outer *edges* of the image, so sampling it with
    ``linspace(edge0, edge1, count)`` — as this backend used to — spaces the
    values by ``(edge1 - edge0) / (count - 1)`` and puts the first one half a
    pixel too far out.  On a 5-pixel axis of unit scale that is a step of 1.25
    instead of 1, which skews both the tick labels and any widget positioned
    against the axis.
    """
    size = (float(edge1) - float(edge0)) / float(count)
    return np.linspace(float(edge0) + size / 2.0, float(edge1) - size / 2.0, int(count))


def _pixel_axes(plot):
    """Return ``(x, y)`` calibrations if *plot* positions widgets in pixels.

    Each entry is ``(first_centre, pixel_size)`` mapping a pixel index to a
    calibrated coordinate, or ``None`` for a panel already in data units
    (a 1-D plot, which has no ``image_width``).
    """
    state = getattr(plot, "_state", None)
    if not isinstance(state, dict) or "image_width" not in state:
        return None, None
    x_axis, y_axis = state.get("x_axis"), state.get("y_axis")
    if x_axis is None or y_axis is None:
        return None, None
    if len(x_axis) < 2 or len(y_axis) < 2:
        return None, None

    def _calibration(axis):
        first, last = float(axis[0]), float(axis[-1])
        size = (last - first) / (len(axis) - 1)
        return None if size == 0 else (first, size)

    return _calibration(x_axis), _calibration(y_axis)


def _to_pixels(axis, value):
    """Map a calibrated coordinate onto a fractional pixel index."""
    if axis is None or value is None:
        return value
    first, size = axis
    return (float(value) - first) / size


def _to_data(axis, pixel):
    """Inverse of :func:`_to_pixels`."""
    if axis is None or pixel is None:
        return pixel
    first, size = axis
    return first + float(pixel) * size


def _pixel_span(axis, length):
    """Convert a length in calibrated units to a length in pixels."""
    if axis is None or length is None:
        return length
    return abs(float(length) / axis[1])


def _data_span(axis, pixels):
    """Inverse of :func:`_pixel_span`."""
    if axis is None or pixels is None:
        return pixels
    return abs(float(pixels) * axis[1])


#: Marker keys holding ``[x, y]`` positions, and how deeply they are nested.
#: ``offsets`` is a list of points, ``segments`` a list of point pairs, and
#: ``vertices_list`` a list of polygons.
_MARKER_POSITION_KEYS = {"offsets": 1, "segments": 2, "vertices_list": 2}

#: Marker keys holding a length along x or along y.
_MARKER_LENGTH_KEYS = {
    "U": "x",
    "widths": "x",
    "radius": "x",
    "V": "y",
    "heights": "y",
}


def _nested_map(value, depth, fn):
    """Apply *fn* to every ``[x, y]`` pair *depth* levels down."""
    if depth == 0:
        pair = list(value)
        return [fn(pair[0], 0)] + ([fn(pair[1], 1)] if len(pair) > 1 else [])
    return [_nested_map(item, depth - 1, fn) for item in value]


def _markers_to_pixels(plot, marker_type, translated):
    """Rewrite marker geometry from calibrated units into image pixels.

    Markers drawn in the ``data`` space on a ``Plot2D`` go through
    ``_imgToCanvas2d`` exactly like the overlay widgets, so they are addressed
    in image pixel indices, and lengths (an arrow's U/V, a rectangle's width)
    are multiplied by the image's canvas-per-pixel scale.  HyperSpy supplies
    calibrated units, so on a calibrated image the markers bunch up near the
    origin — the arrows example put 1024 arrows spanning 0-6.28 into the first
    six pixels of a 100-pixel axis.
    """
    if translated.get("transform") != "data":
        return translated
    x_cal, y_cal = _pixel_axes(plot)
    if x_cal is None and y_cal is None:
        return translated

    if marker_type in ("vlines", "hlines"):
        # These carry one coordinate per entry, along their own axis.
        axis = x_cal if marker_type == "vlines" else y_cal
        offsets = translated.get("offsets")
        if offsets is not None:
            translated["offsets"] = [[_to_pixels(axis, v[0])] for v in offsets]
        return translated

    def _point(value, axis_index):
        return _to_pixels(x_cal if axis_index == 0 else y_cal, value)

    for key, depth in _MARKER_POSITION_KEYS.items():
        if key in translated and translated[key] is not None:
            translated[key] = _nested_map(translated[key], depth, _point)

    if translated.get("size_units") == "px":
        return translated

    for key, which in _MARKER_LENGTH_KEYS.items():
        if key not in translated or translated[key] is None:
            continue
        axis = x_cal if which == "x" else y_cal
        value = translated[key]
        if np.isscalar(value):
            translated[key] = _pixel_span(axis, value)
        else:
            translated[key] = [_pixel_span(axis, v) for v in np.asarray(value).ravel()]
    return translated


def _remember_plot(handle, plot):
    """Tag a widget with the panel it belongs to, and return it.

    ``connect_widget_drag`` and the ``update_*`` methods are handed only the
    widget, but they need the panel's axes to convert coordinates back to
    calibrated units.
    """
    try:
        handle._hspy_plot = plot
    except AttributeError:  # pragma: no cover - defensive
        pass
    return handle


class _AplSpanSelector:
    """matplotlib ``SpanSelector`` façade over anyplotlib's native range widget.

    :class:`~.drawing._widgets.range.RangeWidget` drives the MPL selector API
    directly — ``extents``, ``artists``, ``snap_values``, ``set_props``,
    ``connect_event``, ``_selection_completed``.  anyplotlib's range widget
    covers the same ground with different names, so the translation lives here
    rather than forcing every caller to branch.

    Snapping is handed to the widget (``snap_values``, 0.5.0) so it happens
    inside the JS drag; correcting positions in Python afterwards would move
    an edge the user is still holding.

    ``set_handle_props`` folds into ``set_props``: anyplotlib draws the grab
    handles in the widget colour and has no separate handle style.
    """

    def __init__(self, plot, x0, x1, color="red", orientation="horizontal"):
        self._plot = plot
        self._orientation = orientation
        self._widget = plot.add_range_widget(
            float(x0), float(x1), color=color, orientation=orientation
        )
        self._handlers = []
        # hyperspy sets this to suppress matplotlib's "drag to create first"
        # state; nothing to suppress here, but the attribute must exist.
        self._selection_completed = True

    # ── geometry ─────────────────────────────────────────────────────────

    @property
    def extents(self):
        return (float(self._widget.get("x0")), float(self._widget.get("x1")))

    @extents.setter
    def extents(self, value):
        left, right = value
        if right < left:
            left, right = right, left
        self._widget.set(_notify=False, x0=float(left), x1=float(right))

    @property
    def snap_values(self):
        return self._widget.get("snap_values")

    @snap_values.setter
    def snap_values(self, values):
        # Native as of 0.5.0: the drag itself lands only on allowed values.
        self._widget.set(
            _notify=False,
            snap_values=None if values is None else [float(v) for v in values],
        )

    @property
    def artists(self):
        return [self._widget]

    # ── styling ──────────────────────────────────────────────────────────

    def set_props(self, **props):
        if "color" in props:
            self._widget.set(_notify=False, color=props["color"])

    set_handle_props = set_props

    # ── events / lifecycle ───────────────────────────────────────────────

    def connect_event(self, event_name, fn):
        """Register *fn* for an MPL event name.

        Only ``motion_notify_event`` is meaningful here: hyperspy uses it to
        learn that the user moved the span.  It maps to the widget's own
        ``pointer_move``, which fires on drag.
        """
        if event_name != "motion_notify_event":
            return None

        def _handler(event):
            fn(event)

        self._widget.add_event_handler(_handler, "pointer_move")
        self._handlers.append(_handler)
        return _handler

    def disconnect_events(self):
        for handler in self._handlers:
            try:
                self._widget.remove_handler(handler, "pointer_move")
            except (KeyError, AttributeError, ValueError):
                pass
        self._handlers = []

    def clear(self):
        self.disconnect_events()
        try:
            self._widget.remove()
        except (KeyError, AttributeError):
            pass


class _AplPolygonSelector:
    """matplotlib ``PolygonSelector`` façade over anyplotlib's polygon widget.

    hyperspy's :class:`~.api.roi.PolygonROI` only needs ``verts`` (get/set),
    an ``onselect`` callback fired when the user edits the polygon, plus the
    ``set_props``/``disconnect_events`` lifecycle the other selectors share.

    The native widget is only created once there are >= 3 vertices: anyplotlib
    has no "draw me interactively from scratch" mode, so a polygon with fewer
    has nothing to show yet.
    """

    def __init__(self, plot, color="red", linewidth=2, onselect=None):
        self._plot = plot
        self._color = color
        self._linewidth = linewidth
        self._onselect = onselect
        self._widget = None
        self._verts = []

    @property
    def verts(self):
        if self._widget is not None:
            xa, ya = _pixel_axes(self._plot)
            return [
                (_to_data(xa, vx), _to_data(ya, vy))
                for vx, vy in self._widget.get("vertices", [])
            ]
        return list(self._verts)

    @verts.setter
    def verts(self, value):
        self._verts = [tuple(float(c) for c in v) for v in value]
        if len(self._verts) < 3:
            return
        xa, ya = _pixel_axes(self._plot)
        pixels = [[_to_pixels(xa, vx), _to_pixels(ya, vy)] for vx, vy in self._verts]
        if self._widget is None:
            self._widget = _remember_plot(
                self._plot.add_widget(
                    "polygon",
                    vertices=pixels,
                    color=self._color,
                    linewidth=float(self._linewidth),
                ),
                self._plot,
            )
            self._connect_onselect()
        else:
            self._widget.set(_notify=False, vertices=pixels)

    def _connect_onselect(self):
        """Report browser-side polygon edits back to the ROI.

        Without this the widget's new vertices live only in the JS state:
        ``PolygonROI`` never hears about them, so nothing downstream of the
        ROI recomputes when the polygon is dragged.
        """
        if self._widget is None or self._onselect is None:
            return

        def _handler(event):
            self._onselect(self.verts)

        self._widget.add_event_handler(_handler, "pointer_move")

    def set_props(self, **props):
        if "color" in props:
            self._color = props["color"]
            if self._widget is not None:
                self._widget.set(_notify=False, color=self._color)

    set_handle_props = set_props

    def connect_event(self, event_name, fn):
        if self._widget is None or event_name != "motion_notify_event":
            return None

        def _handler(event):
            fn(event)

        self._widget.add_event_handler(_handler, "pointer_move")
        return _handler

    def disconnect_events(self):
        pass  # handlers die with the widget

    def clear(self):
        if self._widget is not None:
            try:
                self._widget.remove()
            except (KeyError, AttributeError):
                pass
            self._widget = None
        self._verts = []


class _AplLine2DPatch:
    """``matplotlib.lines.Line2D`` façade over anyplotlib's line primitives.

    :class:`~.drawing._widgets.line2d.Line2DWidget` builds its patches before
    it has an axes to put them on (``_set_patch`` runs ahead of
    ``add_artist``), so this stays detached until :meth:`materialise` is
    called with the target axes.

    The main segment becomes a real two-endpoint ``line`` widget (0.5.0), so a
    ``Line2DROI`` is draggable.  The dotted width-indicator lines are
    decoration and become a ``'lines'`` marker group instead.

    Only the slice of the ``Line2D`` API the widget actually uses is
    provided: ``set_data``, ``remove``, and the style/animation no-ops.
    """

    def __init__(self, x, y, color="red", linewidth=1.0, alpha=1.0, interactive=True):
        self._xy = (np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        self._color = color
        self._linewidth = float(linewidth)
        self._alpha = float(alpha) if alpha is not None else 1.0
        self._interactive = interactive
        self._widget = None  # native line widget (main segment)
        self._group = None  # 'lines' marker group (width indicators)
        self._plot = None

    def materialise(self, plot):
        self._plot = plot
        x, y = self._xy
        if self._interactive and hasattr(plot, "add_line_widget") and len(x) >= 2:
            self._widget = plot.add_line_widget(
                x1=float(x[0]),
                y1=float(y[0]),
                x2=float(x[-1]),
                y2=float(y[-1]),
                color=self._color,
                linewidth=self._linewidth,
            )
        else:
            self._group = plot.markers.add(
                "lines",
                segments=AnyplotlibBackend._polyline_segments(x, y),
                edgecolors=self._color,
                linewidths=self._linewidth,
            )

    def set_data(self, x, y=None):
        if y is None:  # Line2D.set_data accepts a single (2, N) array
            x, y = np.asarray(x, dtype=float)
        self._xy = (np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        x, y = self._xy
        if self._widget is not None and len(x) >= 2:
            self._widget.set(
                _notify=False,
                x1=float(x[0]),
                y1=float(y[0]),
                x2=float(x[-1]),
                y2=float(y[-1]),
            )
        elif self._group is not None:
            self._group.set(segments=AnyplotlibBackend._polyline_segments(x, y))

    def set_style(self, *, color=None, alpha=None):
        if color is not None:
            self._color = color
        if alpha is not None:
            self._alpha = float(alpha)
        if color is None:
            return
        if self._widget is not None:
            self._widget.set(_notify=False, color=self._color)
        elif self._group is not None:
            self._group.set(edgecolors=self._color)

    def connect_drag(self, on_drag):
        """Report endpoint moves as (x1, y1, x2, y2) in data coords."""
        if self._widget is None:
            return
        w = self._widget

        def _cb(event):
            on_drag(w.x1, w.y1, w.x2, w.y2)

        w.add_event_handler(AnyplotlibBackend._wrap(_cb), "pointer_move")

    def remove(self):
        for obj in (self._widget, self._group):
            if obj is not None:
                try:
                    obj.remove()
                except Exception:
                    pass
        self._widget = None
        self._group = None


class AnyplotlibBackend:
    """Maps hyperspy drawing primitives to the anyplotlib API.

    Methods that raise ``BackendCapabilityError`` name a capability anyplotlib
    does not have; see ``docs/anyplotlib_improvements.md`` for the running
    list and which release resolved each one.
    """

    # ── Figure lifecycle ──────────────────────────────────────────────────

    def create_figure(self, title=None, on_close=None, **kwargs):
        import anyplotlib as apl

        # If a pre-created panel proxy is passed, adopt it (combined layout).
        fig_kwarg = kwargs.pop("fig", None)
        if isinstance(fig_kwarg, _AplFigureProxy):
            if on_close is not None:
                fig_kwarg._hspy_on_close = on_close
            return fig_kwarg

        figsize = kwargs.pop("figsize", (640, 480))
        figsize = tuple(float(v) for v in figsize)  # normalize (handles ndarray)
        if max(figsize) < 50:
            # matplotlib uses inches; convert to pixels at 96 dpi
            figsize = (int(figsize[0] * 96), int(figsize[1] * 96))
        else:
            figsize = (int(figsize[0]), int(figsize[1]))
        fig, ax = apl.subplots(1, 1, figsize=figsize)
        fig._hspy_ax = ax
        ax.figure = fig  # hyperspy widgets use ax.figure to reach the figure
        if on_close is not None:
            fig._hspy_on_close = on_close
        return fig

    def close_figure(self, fig):
        if fig is None:
            return
        if isinstance(fig, _AplFigureProxy):
            # Proxy: call the close callback but don't close the shared figure.
            on_close = fig._hspy_on_close
            fig._hspy_on_close = None
            if on_close is not None:
                for fn in on_close if isinstance(on_close, list) else [on_close]:
                    try:
                        fn()
                    except Exception:
                        pass
            return
        on_close = getattr(fig, "_hspy_on_close", None)
        if on_close is not None:
            fig._hspy_on_close = None
            for fn in on_close if isinstance(on_close, list) else [on_close]:
                try:
                    fn()
                except Exception:
                    pass
        try:
            fig.close()
        except Exception:
            pass

    def create_combined_figure_panels(self, figsize=None):
        """Create a 2-panel anyplotlib Figure; return (nav_proxy, signal_proxy).

        Both proxies wrap the same underlying Figure widget so ``draw_idle``
        shows the figure only once both panels have rendered (no half-drawn
        flicker).  Pass the proxies as ``fig=`` in ``navigator_kwds`` and the
        main ``kwargs`` respectively.
        """
        import anyplotlib as apl

        figsize = tuple(figsize) if figsize else (1280, 640)
        figsize = tuple(float(v) for v in figsize)
        if max(figsize) < 50:
            figsize = (int(figsize[0] * 96), int(figsize[1] * 96))
        else:
            figsize = (int(figsize[0]), int(figsize[1]))

        fig, axes = apl.subplots(1, 2, figsize=figsize)
        nav_ax, sig_ax = axes[0], axes[1]
        for ax in (nav_ax, sig_ax):
            ax.figure = fig
        # Display is deferred until both panels have drawn.
        fig._hspy_panels_remaining = 2
        return _AplFigureProxy(fig, nav_ax), _AplFigureProxy(fig, sig_ax)

    def ensure_displayed(self, fig):
        """Force-display fig, bypassing the panel countdown.

        Called from signal.py after plot() completes so that a figure is
        always shown even when the navigator was skipped (slider / None).
        """
        real = fig._real_fig if isinstance(fig, _AplFigureProxy) else fig
        if real is None or getattr(real, "_hspy_displayed", False):
            return
        real._hspy_panels_remaining = 0
        try:
            from IPython.display import display

            display(real)
            real._hspy_displayed = True
        except ImportError:
            pass

    def draw_idle(self, fig):
        real = fig._real_fig if isinstance(fig, _AplFigureProxy) else fig
        if real is None or getattr(real, "_hspy_displayed", False):
            return
        real._hspy_drawn = True
        remaining = getattr(real, "_hspy_panels_remaining", 1)
        if remaining > 1:
            # Another panel is still to draw — wait so the figure appears whole.
            real._hspy_panels_remaining = remaining - 1
            return
        real._hspy_panels_remaining = 0
        try:
            from IPython.display import display

            display(real)
            real._hspy_displayed = True
        except ImportError:
            pass

    # ── Blitting (anyplotlib repaints natively; all no-ops) ───────────────

    def supports_blit(self, fig):
        return False

    def copy_background(self, fig):
        return None

    def restore_background(self, fig, background):
        pass

    def blit(self, fig):
        pass

    def connect_draw_event(self, fig, fn):
        return None

    def disconnect_event(self, fig_or_ax, cid):
        plot = self._get_plot(fig_or_ax)
        if plot is None or cid is None:
            return
        if hasattr(plot, "remove_handler"):
            try:
                plot.remove_handler(cid)
            except (KeyError, ValueError):
                pass

    def draw_animated_artists(self, fig):
        pass

    # ── Axes ──────────────────────────────────────────────────────────────

    def create_axes(self, fig, animate_axis=False, **kwargs):
        return fig._hspy_ax

    def set_xlabel(self, ax, label):
        if ax._plot is not None:
            ax._plot.set_xlabel(label)
        else:
            ax._hspy_pending_xlabel = label

    def set_ylabel(self, ax, label):
        if ax._plot is not None:
            ax._plot.set_ylabel(label)
        else:
            ax._hspy_pending_ylabel = label

    def set_title(self, ax, title):
        if ax._plot is not None:
            ax._plot.set_title(title)
        else:
            ax._hspy_pending_title = title

    def set_xlim(self, ax, xmin, xmax):
        if ax._plot is not None:
            ax._plot.set_xlim(xmin, xmax)

    def set_ylim(self, ax, ymin, ymax):
        if ax._plot is not None:
            ax._plot.set_ylim(ymin, ymax)

    def get_xlim(self, ax):
        plot = ax._plot
        if plot is None:
            return (0.0, 1.0)
        for name in ("get_xlim", "get_xbound"):
            fn = getattr(plot, name, None)
            if fn is not None:
                return fn()
        return (0.0, 1.0)

    def get_ylim(self, ax):
        if ax._plot is not None and hasattr(ax._plot, "get_ylim"):
            return ax._plot.get_ylim()
        return (0.0, 1.0)

    def get_xbound(self, ax):
        if ax._plot is not None:
            return ax._plot.get_xbound()
        return (0.0, 1.0)

    def set_axis_off(self, ax):
        if ax._plot is not None:
            ax._plot.set_axis_off()

    def set_aspect(self, ax, ratio):
        if ax._plot is not None and hasattr(ax._plot, "set_aspect"):
            ax._plot.set_aspect(ratio)

    def add_right_axis(self, ax, color="black"):
        raise BackendCapabilityError(_NOT_YET.format("add_right_axis (twinx)"))

    def remove_right_axis(self, ax, right_ax):
        raise BackendCapabilityError(_NOT_YET.format("remove_right_axis"))

    # ── 1-D line plotting ─────────────────────────────────────────────────

    # MPL marker codes → the symbols anyplotlib's renderer knows.  Codes with
    # no close equivalent fall back to 'o' (see _norm_marker).
    _MARKER_MAP = {
        "o": "o",
        "s": "s",
        "^": "^",
        "v": "v",
        "D": "D",
        "d": "D",
        "+": "+",
        "x": "x",
        ".": "o",
        ",": "o",
        "*": "D",
        "None": "none",
        "none": "none",
        "": "none",
        " ": "none",
    }

    @classmethod
    def _norm_marker(cls, marker):
        if marker is None:
            return "none"
        return cls._MARKER_MAP.get(str(marker), "o")

    @classmethod
    def _norm_line_props(cls, props):
        """Translate hyperspy's matplotlib line vocabulary to anyplotlib kwargs.

        hyperspy describes lines the way matplotlib does (``linestyle="None"``
        for a markers-only scatter, ``drawstyle="steps-mid"`` for a step plot,
        ``markeredgecolor`` standing in for ``color`` on scatter lines).
        anyplotlib names some of these differently, so normalise here rather
        than at every call site.

        Only keys present in *props* are returned, so callers can distinguish
        "not specified" from "specified as the default".
        """
        out = {}

        if "color" in props and props["color"] is not None:
            out["color"] = props["color"]
        elif props.get("markeredgecolor") is not None:
            # Scatter lines carry their colour as markeredgecolor: hyperspy's
            # Signal1DLine.color setter moves it there and drops "color".
            out["color"] = props["markeredgecolor"]

        if "linewidth" in props and props["linewidth"] is not None:
            out["linewidth"] = float(props["linewidth"])
        if "alpha" in props and props["alpha"] is not None:
            out["alpha"] = float(props["alpha"])

        if "marker" in props:
            out["marker"] = cls._norm_marker(props["marker"])
        if props.get("markersize") is not None:
            out["markersize"] = float(props["markersize"])

        # drawstyle="steps-mid" is matplotlib's step plot; anyplotlib folds it
        # into linestyle as "step-mid" and it must win over the plain
        # linestyle that hyperspy sends alongside it.
        linestyle = props.get("linestyle")
        drawstyle = props.get("drawstyle")
        if linestyle is not None:
            if str(linestyle) in ("None", "none", "", " "):
                # Markers-only.  Native as of 0.5.0: the renderer skips the
                # connecting stroke instead of us faking it with 'solid'.
                out["linestyle"] = "none"
                out.setdefault("marker", cls._norm_marker(props.get("marker", "o")))
            else:
                out["linestyle"] = str(linestyle)
        if drawstyle is not None and str(drawstyle).startswith("steps"):
            out["linestyle"] = "step-mid"

        return out

    def plot_line(self, ax, x, y, **props):
        """Draw a line; return an opaque handle.

        Three cases, because ``Axes.plot`` *replaces* the panel's plot when
        called twice (it does not overlay):

        * no plot on ax yet → ``ax.plot`` (primary ``Plot1D``)
        * ax holds a ``Plot1D`` → ``Plot1D.add_line`` (overlay ``Line1D``)
        * ax holds a ``Plot2D`` → a ``'lines'`` marker group (e.g. scale bar)
        """
        style = self._norm_line_props(props)
        color = style.pop("color", "#4fc3f7")
        linewidth = style.pop("linewidth", 1.5)
        style.setdefault("linestyle", "solid")
        style.setdefault("alpha", 1.0)
        x_arr = np.asarray(x) if x is not None else None
        y_arr = np.asarray(y)

        existing = getattr(ax, "_plot", None)
        image_plot = getattr(ax, "_hspy_image_plot", None)

        if existing is None:
            plot = ax.plot(
                y_arr,
                axes=[x_arr] if x_arr is not None else None,
                color=color,
                linewidth=linewidth,
                **style,
            )
            self._apply_pending_labels(ax, plot)
            return plot

        if image_plot is not None or not hasattr(existing, "add_line"):
            # Image panel: overlay drawn as a native 'lines' marker group.
            target = image_plot or existing
            group = target.markers.add(
                "lines",
                segments=self._polyline_segments(x_arr, y_arr),
                edgecolors=color,
                linewidths=linewidth,
            )
            return group

        return existing.add_line(
            y_arr,
            x_axis=x_arr,
            color=color,
            linewidth=linewidth,
            **style,
        )

    @staticmethod
    def _polyline_segments(x, y):
        """Convert polyline arrays to the (N-1, 2, 2) segment list wire format."""
        pts = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
        return [pts[i : i + 2].tolist() for i in range(len(pts) - 1)]

    def _apply_pending_labels(self, ax, plot):
        """Apply any labels buffered before a plot was attached to ax."""
        for attr, method in (
            ("_hspy_pending_xlabel", "set_xlabel"),
            ("_hspy_pending_ylabel", "set_ylabel"),
            ("_hspy_pending_title", "set_title"),
        ):
            val = getattr(ax, attr, None)
            if val is not None:
                getattr(plot, method)(val)
                try:
                    delattr(ax, attr)
                except AttributeError:
                    pass

    def update_line(self, handle, x, y):
        if hasattr(handle, "set_data"):  # Plot1D and Line1D share this API
            handle.set_data(np.asarray(y), x_axis=np.asarray(x))
        elif hasattr(handle, "set"):  # 'lines' MarkerGroup on an image panel
            handle.set(segments=self._polyline_segments(x, y))

    def remove_line(self, ax, handle):
        if hasattr(handle, "set_data") and not hasattr(handle, "remove"):
            return  # primary Plot1D cannot be removed from its panel
        try:
            handle.remove()  # Line1D and MarkerGroup
        except Exception:
            pass

    def set_line_props(self, handle, **props):
        style = self._norm_line_props(props)
        if hasattr(handle, "set_color"):  # primary Plot1D
            if "color" in style:
                handle.set_color(style["color"])
            if "linewidth" in style:
                handle.set_linewidth(style["linewidth"])
            if "linestyle" in style:
                handle.set_linestyle(style["linestyle"])
            if "alpha" in style:
                handle.set_alpha(style["alpha"])
            if "marker" in style or "markersize" in style:
                handle.set_marker(
                    style.get("marker", handle._state.get("line_marker", "none")),
                    style.get("markersize"),
                )
        elif hasattr(handle, "_data"):  # 'lines' MarkerGroup
            updates = {}
            if "color" in style:
                updates["edgecolors"] = style["color"]
            if "linewidth" in style:
                updates["linewidths"] = style["linewidth"]
            if updates:
                handle.set(**updates)
        else:  # Line1D: native color/linewidth/linestyle/alpha properties
            for name in ("color", "linewidth", "linestyle", "alpha"):
                if name in style:
                    setattr(handle, name, style[name])

    def line_get_xdata(self, handle):
        return getattr(handle, "x", None)

    def line_get_color(self, handle):
        if hasattr(handle, "color"):  # primary Plot1D or Line1D property
            return handle.color
        if hasattr(handle, "_data"):  # 'lines' MarkerGroup
            return handle._data.get("edgecolors", "#4fc3f7")
        return "#4fc3f7"

    def line_get_linewidth(self, handle):
        if hasattr(handle, "_state"):  # primary Plot1D
            return float(handle._state.get("line_linewidth", 1.5))
        if hasattr(handle, "linewidth"):  # Line1D property
            return float(handle.linewidth)
        if hasattr(handle, "_data"):  # 'lines' MarkerGroup
            return float(handle._data.get("linewidths", 1.5))
        return 1.5

    # ── Text annotations ─────────────────────────────────────────────────

    # MPL-style named font sizes; anyplotlib add_text wants a numeric px size.
    _MPL_FONTSIZE = {
        "xx-small": 7,
        "x-small": 8,
        "small": 10,
        "medium": 12,
        "large": 14,
        "x-large": 17,
        "xx-large": 20,
    }

    def add_text(self, ax, x, y, s, transform="data", **kwargs):
        plot = self._primary_plot(ax)
        if plot is None:
            return None
        color = kwargs.get("color", "#ff0000")
        fontsize = kwargs.get("fontsize", kwargs.get("size", 12))
        if isinstance(fontsize, str):
            fontsize = self._MPL_FONTSIZE.get(fontsize, 12)
        return plot.add_text(
            float(x),
            float(y),
            str(s),
            color=color,
            fontsize=float(fontsize),
            transform=self._SPACE_MAP.get(transform, "data"),
        )

    def update_text(self, handle, s):
        if handle is not None:
            handle.set_text(str(s))

    def remove_text(self, ax, handle):
        if handle is None:
            return
        try:
            handle.remove()
        except Exception:
            pass

    def text_set_color(self, handle, color):
        if handle is not None:
            handle.set_color(color)

    def text_get_color(self, handle):
        if handle is None:
            return "white"
        group = getattr(handle, "_group", None)
        if group is not None:
            return group._data.get("color", "white")
        return "white"

    def artist_set_animated(self, handle, animated):
        pass  # anyplotlib repaints natively; no per-artist animated flag

    # ── 2-D image plotting ────────────────────────────────────────────────

    def plot_image(
        self,
        ax,
        data,
        extent=None,
        vmin=None,
        vmax=None,
        norm=None,
        cmap=None,
        **kwargs,
    ):
        aspect = kwargs.pop("aspect", None)
        arr = np.asarray(data)
        axes = None
        if extent is not None:
            x0, x1, y0, y1 = extent
            axes = [
                _pixel_centres(x0, x1, arr.shape[1]),
                _pixel_centres(y1, y0, arr.shape[0]),
            ]
        plot = ax.imshow(arr, axes=axes, cmap=cmap, vmin=vmin, vmax=vmax)
        if aspect not in (None, "auto", "equal", 1, 1.0):
            plot.set_aspect(float(aspect))
        self._apply_pending_labels(ax, plot)
        ax._hspy_image_plot = plot
        if norm is not None:
            self.image_set_norm(plot, norm)
        return plot

    def plot_mesh(self, ax, x, y, data, **kwargs):
        return ax.pcolormesh(
            np.asarray(data),
            x_edges=np.asarray(x),
            y_edges=np.asarray(y),
        )

    def image_set_data(self, handle, data):
        handle.set_data(np.asarray(data))

    def image_set_extent(self, handle, extent):
        x0, x1, y0, y1 = extent
        w = handle._state["image_width"]
        h = handle._state["image_height"]
        handle.set_extent(_pixel_centres(x0, x1, w), _pixel_centres(y1, y0, h))

    def image_set_clim(self, handle, vmin, vmax):
        handle.set_clim(vmin, vmax)

    def image_set_norm(self, handle, norm):
        from hyperspy.drawing.norm import (
            HyperNorm,
            LinearNorm,
            LogNorm,
            SymLogNorm,
        )

        mode = "linear"
        if isinstance(norm, LogNorm):
            mode = "log"
        elif isinstance(norm, SymLogNorm):
            mode = "symlog"
        elif isinstance(norm, (HyperNorm, LinearNorm)):
            mode = "linear"
        if hasattr(handle, "set_scale_mode"):
            handle.set_scale_mode(mode)
        vmin = getattr(norm, "vmin", None)
        vmax = getattr(norm, "vmax", None)
        if vmin is not None or vmax is not None:
            handle.set_clim(vmin, vmax)

    def get_image_handle(self, ax):
        handle = getattr(ax, "_hspy_image_plot", None)
        if handle is not None:
            return handle
        return getattr(ax, "_plot", None)

    def add_colorbar(self, fig, im_handle, ax, divider=None, size=None, pad=None):
        im_handle.set_colorbar_visible(True)
        if pad is not None and hasattr(im_handle, "set_colorbar_pad"):
            # Native as of 0.5.0 — previously accepted and silently dropped.
            im_handle.set_colorbar_pad(pad)
        return _AplColorbar(im_handle)

    def colorbar_set_label(self, cb, label):
        cb._im.set_colorbar_label(label)

    def colorbar_remove(self, cb):
        cb._im.set_colorbar_visible(False)

    def colorbar_redraw(self, cb, fig):
        pass

    # ── Events ───────────────────────────────────────────────────────────

    @staticmethod
    def _wrap(fn):
        """Return a plain function wrapping fn.

        anyplotlib's add_event_handler sets fn._event_types, which fails on
        bound methods (they have no __dict__). Wrapping guarantees a plain
        function object that allows arbitrary attribute assignment.
        """

        def _handler(*args, **kwargs):
            return fn(*args, **kwargs)

        return _handler

    def connect_key_press(self, fig_or_ax, fn):
        plot = self._get_plot(fig_or_ax)
        if plot is not None:
            return plot.add_event_handler(self._wrap(fn), "key_down")
        return None

    def connect_mouse_move(self, fig_or_ax, fn):
        plot = self._get_plot(fig_or_ax)
        if plot is not None:
            return plot.add_event_handler(self._wrap(fn), "pointer_move")
        return None

    def connect_mouse_press(self, fig_or_ax, fn):
        plot = self._get_plot(fig_or_ax)
        if plot is not None:
            return plot.add_event_handler(self._wrap(fn), "pointer_down")
        return None

    def connect_mouse_release(self, fig_or_ax, fn):
        plot = self._get_plot(fig_or_ax)
        if plot is not None:
            return plot.add_event_handler(self._wrap(fn), "pointer_up")
        return None

    def connect_pick(self, fig_or_ax, fn):
        return None

    def _get_plot(self, fig_or_ax):
        """Return the Plot1D/Plot2D attached to the given figure or axes."""
        plot = getattr(fig_or_ax, "_plot", None)
        if plot is not None:
            return plot
        ax = getattr(fig_or_ax, "_hspy_ax", None)
        if ax is not None:
            return getattr(ax, "_plot", None)
        return None

    def _primary_plot(self, ax):
        """Return the primary (image or line) plot for widget creation."""
        image_plot = getattr(ax, "_hspy_image_plot", None)
        if image_plot is not None:
            return image_plot
        return getattr(ax, "_plot", None)

    # ── Pointers / widgets ────────────────────────────────────────────────

    def create_line_pointer(self, ax, axis, pos, color="red"):
        plot = self._primary_plot(ax)
        if plot is None:
            raise RuntimeError("ax has no plot; call plot_line or plot_image first")
        # A Signal1D with one navigation axis gets a *2-D* navigator (the whole
        # dataset as an image) with a horizontal line marking the current row,
        # so this lands on a Plot2D and needs converting just like the region
        # widgets do.  A genuine 1-D panel reports no calibration and is left
        # in data units.
        x_cal, y_cal = _pixel_axes(plot)
        if axis == "x":
            value = _to_pixels(x_cal, pos)
            if hasattr(plot, "add_vline_widget"):
                handle = plot.add_vline_widget(x=float(value), color=color)
            else:
                handle = plot.add_widget(
                    "crosshair", cx=float(value), cy=0.0, color=color
                )
        else:
            value = _to_pixels(y_cal, pos)
            if hasattr(plot, "add_hline_widget"):
                handle = plot.add_hline_widget(y=float(value), color=color)
            else:
                handle = plot.add_widget(
                    "crosshair", cx=0.0, cy=float(value), color=color
                )
        return _remember_plot(handle, plot)

    def update_line_pointer(self, handle, axis, pos):
        wtype = handle.get("type") if hasattr(handle, "get") else None
        x_cal, y_cal = _pixel_axes(getattr(handle, "_hspy_plot", None))
        value = _to_pixels(x_cal if axis == "x" else y_cal, pos)
        # _notify=False: a Python-side update must not echo back through the
        # drag callback and feed into navigation (0.5.0; was pause_events).
        if wtype == "crosshair":
            if axis == "x":
                handle.set(_notify=False, cx=float(value))
            else:
                handle.set(_notify=False, cy=float(value))
        elif axis == "x":
            handle.set(_notify=False, x=float(value))
        else:
            handle.set(_notify=False, y=float(value))

    def connect_widget_drag(self, handle, on_drag):
        if isinstance(handle, _AplLine2DPatch):
            handle.connect_drag(on_drag)
            return
        wtype = handle.get("type") if hasattr(handle, "get") else None
        # The widget reports pixel indices; hyperspy expects calibrated units.
        xa, ya = _pixel_axes(getattr(handle, "_hspy_plot", None))

        if wtype == "vline":

            def _cb(event):
                on_drag(_to_data(xa, handle.x))

        elif wtype == "hline":

            def _cb(event):
                on_drag(_to_data(ya, handle.y))

        elif wtype == "crosshair":

            def _cb(event):
                on_drag(_to_data(xa, handle.cx), _to_data(ya, handle.cy))

        elif wtype == "rectangle":
            # Region widgets report their full geometry: corner plus size.
            # The hyperspy widget maps the corner back to its own position
            # convention (see RectangleWidget._on_widget_drag).

            def _cb(event):
                on_drag(
                    _to_data(xa, handle.x),
                    _to_data(ya, handle.y),
                    _data_span(xa, handle.w),
                    _data_span(ya, handle.h),
                )

        elif wtype in ("circle", "annular"):

            def _cb(event):
                on_drag(_to_data(xa, handle.cx), _to_data(ya, handle.cy))

        elif wtype == "polygon":

            def _cb(event):
                on_drag(
                    [
                        (_to_data(xa, vx), _to_data(ya, vy))
                        for vx, vy in handle.get("vertices", [])
                    ]
                )

        else:
            return
        handle.add_event_handler(self._wrap(_cb), "pointer_move")

    def create_rect_pointer(
        self, ax, x, y, w, h, color="red", linewidth=2, pointer=False
    ):
        plot = self._primary_plot(ax)
        if plot is None or not hasattr(plot, "add_widget"):
            raise BackendCapabilityError(_NOT_YET.format("create_rect_pointer"))
        xa, ya = _pixel_axes(plot)
        px, py = _to_pixels(xa, x), _to_pixels(ya, y)
        pw, ph = _pixel_span(xa, w), _pixel_span(ya, h)
        if pointer:
            # The navigator pointer marks the current navigation position.  A
            # crosshair reads better than a rectangle on an interactive JS
            # panel; 0.5.0 also lets a degenerate (single-axis) navigator use a
            # real vline/hline rather than a crosshair pinned to an edge.
            if h <= 0 and hasattr(plot, "add_vline_widget"):
                handle = plot.add_vline_widget(
                    x=float(px) + float(pw) / 2.0, color=color
                )
            elif w <= 0 and hasattr(plot, "add_hline_widget"):
                handle = plot.add_hline_widget(
                    y=float(py) + float(ph) / 2.0, color=color
                )
            else:
                handle = plot.add_widget(
                    "crosshair",
                    cx=float(px) + float(pw) / 2.0,
                    cy=float(py) + float(ph) / 2.0,
                    color=color,
                )
        else:
            # Region selector (e.g. RectangularROI): a native rectangle widget
            # with built-in JS move/resize handles.
            handle = plot.add_widget(
                "rectangle",
                x=float(px),
                y=float(py),
                w=float(pw),
                h=float(ph),
                color=color,
            )
        return _remember_plot(handle, plot)

    def update_rect_pointer(self, handle, x, y, w, h):
        wtype = handle.get("type") if hasattr(handle, "get") else None
        xa, ya = _pixel_axes(getattr(handle, "_hspy_plot", None))
        px, py = _to_pixels(xa, x), _to_pixels(ya, y)
        pw, ph = _pixel_span(xa, w), _pixel_span(ya, h)
        if wtype == "crosshair":
            handle.set(
                _notify=False,
                cx=float(px) + float(pw) / 2.0,
                cy=float(py) + float(ph) / 2.0,
            )
        elif wtype == "vline":
            handle.set(_notify=False, x=float(px) + float(pw) / 2.0)
        elif wtype == "hline":
            handle.set(_notify=False, y=float(py) + float(ph) / 2.0)
        else:
            handle.set(
                _notify=False,
                x=float(px),
                y=float(py),
                w=float(pw),
                h=float(ph),
            )

    def create_circle_pointer(
        self, ax, cx, cy, r_outer, r_inner=0.0, color="red", linewidth=2, alpha=1.0
    ):
        plot = self._primary_plot(ax)
        if plot is None or not hasattr(plot, "add_widget"):
            raise BackendCapabilityError(_NOT_YET.format("create_circle_pointer"))
        xa, ya = _pixel_axes(plot)
        pcx, pcy = _to_pixels(xa, cx), _to_pixels(ya, cy)
        # A radius is a length, and the widget is drawn as a true circle, so it
        # can only follow one axis' calibration.
        p_outer = _pixel_span(xa, r_outer)
        p_inner = _pixel_span(xa, r_inner)
        # One native widget covers both cases, so the returned list always has
        # a single element (matplotlib needs two patches for the annulus).
        if r_inner > 0:
            handle = plot.add_widget(
                "annular",
                cx=float(pcx),
                cy=float(pcy),
                r_outer=float(p_outer),
                r_inner=float(p_inner),
                color=color,
                linewidth=float(linewidth),
            )
        else:
            handle = plot.add_widget(
                "circle",
                cx=float(pcx),
                cy=float(pcy),
                r=float(p_outer),
                color=color,
                linewidth=float(linewidth),
            )
        return [_remember_plot(handle, plot)]

    def update_circle_pointer(self, ax, handles, cx, cy, r_outer, r_inner=0.0):
        handle = handles[0]
        is_annular = (
            handle.get("type") == "annular" if hasattr(handle, "get") else False
        )
        if is_annular != (r_inner > 0):
            # circle ↔ annulus: the native widget kinds are distinct, so swap.
            style = handle._data if hasattr(handle, "_data") else {}
            new = self.create_circle_pointer(
                ax,
                cx,
                cy,
                r_outer,
                r_inner,
                color=style.get("color", "red"),
                linewidth=style.get("linewidth", 2),
            )
            for old in handles:
                self.remove_pointer(ax, old)
            return new

        xa, ya = _pixel_axes(getattr(handle, "_hspy_plot", None))
        pcx, pcy = _to_pixels(xa, cx), _to_pixels(ya, cy)
        if is_annular:
            handle.set(
                _notify=False,
                cx=float(pcx),
                cy=float(pcy),
                r_outer=float(_pixel_span(xa, r_outer)),
                r_inner=float(_pixel_span(xa, r_inner)),
            )
        else:
            handle.set(
                _notify=False,
                cx=float(pcx),
                cy=float(pcy),
                r=float(_pixel_span(xa, r_outer)),
            )
        return handles

    def remove_pointer(self, ax, handle):
        if handle is None:
            return
        # 0.5.0: widgets remove themselves; no need to re-derive the plot.
        remove = getattr(handle, "remove", None)
        if callable(remove):
            try:
                remove()
                return
            except (KeyError, AttributeError):
                pass
        plot = self._primary_plot(ax)
        if plot is None:
            return
        try:
            plot.remove_widget(handle)
        except (KeyError, AttributeError):
            pass

    def set_pointer_style(self, handle, *, color=None, alpha=None, animated=None):
        # animated is a blit concept; anyplotlib repaints natively.
        if isinstance(handle, _AplLine2DPatch):
            handle.set_style(color=color, alpha=alpha)
            return
        updates = {}
        if color is not None:
            updates["color"] = color
        if alpha is not None:
            updates["alpha"] = float(alpha)
        if updates and hasattr(handle, "set"):
            handle.set(_notify=False, **updates)

    def add_artist(self, ax, artist):
        # Only patches this backend itself created can be attached; any other
        # (matplotlib) artist has no anyplotlib representation.
        if isinstance(artist, _AplLine2DPatch):
            plot = self._primary_plot(ax)
            if plot is not None:
                artist.materialise(plot)

    def create_rect_patch(self, pos, w, h, **kwargs):
        raise BackendCapabilityError(
            _NOT_YET.format("create_rect_patch (resizer handles)")
        )

    def get_data_transform_inverse(self, ax):
        raise BackendCapabilityError(_NOT_YET.format("get_data_transform_inverse"))

    def transform_point(self, transform, point):
        raise BackendCapabilityError(_NOT_YET.format("transform_point"))

    def simulate_pick(self, ax, patch):
        pass

    # ── Marker collections (MPL fallback path) ────────────────────────────

    def add_collection(self, ax, collection):
        raise BackendCapabilityError(_NOT_YET.format("add_collection (markers)"))

    def collection_update(self, handle, **kwargs):
        raise BackendCapabilityError(_NOT_YET.format("collection_update (markers)"))

    def collection_remove(self, ax, handle):
        raise BackendCapabilityError(_NOT_YET.format("collection_remove (markers)"))

    # ── Rendering hooks ───────────────────────────────────────────────────

    def render_figure_from_ax(self, ax):
        self.draw_idle(getattr(ax, "figure", None))

    def invalidate_blit_background(self, ax):
        pass  # anyplotlib repaints natively; no blit-background cache

    def supports_blit_from_ax(self, ax):
        return False

    # ── Selectors ─────────────────────────────────────────────────────────

    def create_span_selector(self, ax, **kwargs):
        plot = self._primary_plot(ax)
        if plot is None or not hasattr(plot, "add_range_widget"):
            raise BackendCapabilityError(_NOT_YET.format("create_span_selector"))
        direction = kwargs.get("direction", "horizontal")
        props = kwargs.get("props") or {}
        if direction == "vertical":
            # 0.5.0: a vertical range selects on the value axis.
            lo, hi = self.get_ylim(ax)
        else:
            lo, hi = self.get_xlim(ax)
        return _AplSpanSelector(
            plot, lo, hi, color=props.get("color", "red"), orientation=direction
        )

    def create_polygon_selector(self, ax, **kwargs):
        plot = self._primary_plot(ax)
        if plot is None or not hasattr(plot, "add_widget"):
            raise BackendCapabilityError(_NOT_YET.format("create_polygon_selector"))
        props = kwargs.get("props") or {}
        return _AplPolygonSelector(
            plot,
            color=props.get("color", "red"),
            linewidth=props.get("linewidth", 2),
            onselect=kwargs.get("onselect"),
        )

    # ── Coordinate spaces ─────────────────────────────────────────────────

    _COORD_SPACES = ("data", "axes", "display", "xaxis", "yaxis", "relative")

    # hyperspy/MPL coordinate-space tokens -> anyplotlib's narrower transform
    # vocabulary (markers.py._VALID_TRANSFORMS == {"data", "axes", "display"}).
    _SPACE_MAP = {
        "data": "data",
        "axes": "axes",
        "display": "display",
        "xaxis": "data",
        "yaxis": "data",
        "relative": "data",
    }

    def get_ax_transform(self, ax, kind):
        """Return the coordinate-space token for *kind*.

        anyplotlib transforms are plain space strings (the same tokens that
        markers accept), not matplotlib ``Transform`` objects.
        """
        if kind not in self._COORD_SPACES:
            raise BackendCapabilityError(_NOT_YET.format(f"get_ax_transform({kind!r})"))
        return kind

    def convert_coords(self, ax, points, from_space, to_space):
        """Convert points between coordinate spaces.

        Display-space maths is delegated to anyplotlib's own
        ``data_to_display`` / ``display_to_data`` (0.5.0) rather than
        re-deriving the renderer's padding constants and letterbox fit here —
        upstream's version is the one verified against the renderer.
        """
        for space in (from_space, to_space):
            if space not in self._COORD_SPACES:
                raise BackendCapabilityError(
                    _NOT_YET.format(f"convert_coords({space!r})")
                )
        plot = self._primary_plot(ax)
        if plot is None:
            raise RuntimeError("ax has no plot; call plot_line or plot_image first")

        pts = np.atleast_2d(np.asarray(points, dtype=float))
        if pts.shape[-1] != 2:
            raise ValueError(f"points must have shape (N, 2); got {pts.shape}")

        x0, x1 = self.get_xlim(ax)
        y0, y1 = self.get_ylim(ax)
        xspan = (x1 - x0) or 1.0
        yspan = (y1 - y0) or 1.0

        def _to_data(vals, space):
            if space in ("data", "xaxis", "yaxis", "relative"):
                return vals
            if space == "axes":
                return np.column_stack(
                    [x0 + vals[:, 0] * xspan, y0 + vals[:, 1] * yspan]
                )
            return np.atleast_2d(plot.display_to_data(vals))  # "display"

        def _from_data(vals, space):
            if space in ("data", "xaxis", "yaxis", "relative"):
                return vals
            if space == "axes":
                return np.column_stack(
                    [(vals[:, 0] - x0) / xspan, (vals[:, 1] - y0) / yspan]
                )
            return np.atleast_2d(plot.data_to_display(vals))  # "display"

        result = np.atleast_2d(_from_data(_to_data(pts, from_space), to_space))
        return result if np.ndim(points) == 2 else result[0]

    # ── Native marker collections ─────────────────────────────────────────

    def create_markers(self, ax, marker_type, **kwargs):
        """Add a native anyplotlib marker group to *ax*.

        Parameters
        ----------
        marker_type : str
            One of the ``MarkerType.*`` string constants.
        **kwargs : dict
            HyperSpy/MPL-style marker kwargs plus ``offset_space`` and
            ``transform_space`` (popped before translation).

        Returns
        -------
        anyplotlib.markers.MarkerGroup
            The live handle; pass to ``update_markers`` / ``remove_markers``.
        """
        plot = self._primary_plot(ax)
        if plot is None:
            raise RuntimeError("ax has no plot; call plot_line or plot_image first")

        offset_space = kwargs.pop("offset_space", "data")
        kwargs.pop("transform_space", None)  # handled via offset_space
        size_units = kwargs.get("units", "points")

        translated = self._translate_marker_kwargs(marker_type, offset_space, kwargs)

        if marker_type == "points" and hasattr(plot, "set_clim"):
            # 2-D panels have no 'points' marker type; mirror Plot2D.add_points,
            # which renders points as circles.  hyperspy Points sizes are
            # display points, so ask for px sizes (0.5.0) rather than
            # converting display->data once and letting zoom rescale them.
            marker_type = "circles"
            sizes = translated.pop("sizes", None)
            if sizes is not None:
                radius = np.atleast_1d(np.asarray(sizes, dtype=float)) / 2.0
                translated["radius"] = (
                    float(radius[0]) if radius.size == 1 else radius.tolist()
                )
                if size_units not in ("x", "y", "xy", "width", "height"):
                    translated["size_units"] = "px"

        _markers_to_pixels(plot, marker_type, translated)

        try:
            return _remember_plot(plot.markers.add(marker_type, **translated), plot)
        except ValueError as exc:
            raise BackendCapabilityError(
                f"anyplotlib does not support marker type '{marker_type}' "
                f"on this plot type: {exc}"
            ) from exc

    def update_markers(self, handle, **kwargs):
        """Update a ``MarkerGroup`` returned by ``create_markers``."""
        if not kwargs:
            return
        marker_type = handle._type
        # Derive the stored coordinate space so vlines/hlines un-segment correctly.
        offset_space = handle._data.get("transform", "data")
        if handle._data.get("size_units") == "px" and "sizes" in kwargs:
            sizes = np.atleast_1d(np.asarray(kwargs.pop("sizes"), dtype=float))
            radius = sizes / 2.0
            kwargs["radius"] = float(radius[0]) if radius.size == 1 else radius.tolist()
        translated = self._translate_marker_kwargs(marker_type, offset_space, kwargs)
        _markers_to_pixels(getattr(handle, "_hspy_plot", None), marker_type, translated)
        if translated:
            handle.set(**translated)

    def remove_markers(self, ax, handle):
        try:
            handle.remove()
        except Exception:
            pass

    @staticmethod
    def _translate_marker_kwargs(marker_type, offset_space, kwargs):
        """Translate HyperSpy/MPL-style marker kwargs to anyplotlib wire kwargs.

        Handles:
        - Coordinate space strings → anyplotlib ``transform``
        - ``circles.sizes`` → ``radius``
        - ``vlines/hlines`` full segments → 1-D offset lists
        - ``polygons.verts`` → ``vertices_list``
        - ``colors`` (MPL plural) → ``edgecolors``
        - ``linewidth`` (singular) → ``linewidths``
        - Singleton cycling sequences → scalars (see ``_CYCLING_KWARGS``)
        - Strips MPL-only kwargs (``units``, ``patches``, ``drawstyle``, …)
        """
        out = {}
        out["transform"] = AnyplotlibBackend._SPACE_MAP.get(offset_space, "data")

        # Work on a shallow copy so we can pop without mutating the caller's dict.
        work = dict(kwargs)

        # ── colour / linewidth renaming ─────────────────────────────────────
        if "colors" in work:
            val = work.pop("colors")
            if isinstance(val, (list, tuple)) and len(val) == 1:
                val = val[0]
            work["edgecolors"] = val

        if "linewidth" in work and "linewidths" not in work:
            work["linewidths"] = work.pop("linewidth")

        # ── type-specific positional key translations ───────────────────────
        if marker_type == "circles":
            # HyperSpy passes MPL-style ``sizes`` (display-unit area);
            # anyplotlib circles uses ``radius``.
            if "sizes" in work:
                out["radius"] = _unwrap_cycling(work.pop("sizes"))

        elif marker_type == "points":
            if "sizes" in work:
                work["sizes"] = _unwrap_cycling(work["sizes"])

        elif marker_type in ("vlines", "hlines"):
            # VerticalLines/HorizontalLines expand positions into full
            # [[x,0],[x,1]] / [[0,y],[1,y]] segments for the MPL path.
            # anyplotlib vlines/hlines want [[x], ...] / [[y], ...].
            if "segments" in work:
                segs = np.asarray(work.pop("segments"), dtype=float)
                if marker_type == "vlines":
                    out["offsets"] = [[float(v)] for v in segs[:, 0, 0]]
                else:
                    out["offsets"] = [[float(v)] for v in segs[:, 0, 1]]
            # Positions are always in data space for span-line types.
            out["transform"] = "data"

        elif marker_type == "polygons":
            # MPL PolyCollection uses ``verts``; anyplotlib uses ``vertices_list``.
            if "verts" in work:
                verts = work.pop("verts")
                out["vertices_list"] = [
                    np.asarray(v, dtype=float).tolist() for v in verts
                ]

        # ── copy remaining compatible kwargs ────────────────────────────────
        _STRIP = {"offset_transform", "units", "patches", "drawstyle"}
        for k, v in work.items():
            if k in _STRIP:
                continue
            if k in _CYCLING_KWARGS:
                # Singleton style/geometry values arrive as 1-element cycling
                # sequences; anyplotlib wants a scalar or one value per marker.
                out[k] = _unwrap_cycling(v)
            elif hasattr(v, "tolist"):
                # Eagerly convert numpy arrays so downstream JSON
                # serialisation works.
                out[k] = v.tolist()
            else:
                out[k] = v

        return out

    # ── Remaining protocol methods ────────────────────────────────────────

    def plot_step(self, ax, x, y, **props):
        # anyplotlib renders a step plot via the "step-mid" linestyle, which
        # _norm_line_props derives from drawstyle.
        props.setdefault("drawstyle", "steps-mid")
        return self.plot_line(ax, x, y, **props)

    def create_line2d_patch(self, x, y, **kwargs):
        # Detached until add_artist supplies the axes — see _AplLine2DPatch.
        # The main segment (Line2DWidget passes marker="s" for its draggable
        # line) becomes a native two-endpoint widget; the dotted width
        # indicators are decoration and become a 'lines' marker group.
        return _AplLine2DPatch(
            x,
            y,
            color=kwargs.get("c", kwargs.get("color", "red")),
            linewidth=kwargs.get("lw", kwargs.get("linewidth", 1.0)),
            alpha=kwargs.get("alpha", 1.0),
            interactive=kwargs.get("marker") == "s",
        )

    def create_circle_patch(self, xy, radius, **kwargs):
        raise BackendCapabilityError(_NOT_YET.format("create_circle_patch"))

    def set_autoscale(self, ax, enable):
        pass  # anyplotlib manages zoom internally

    def set_xticklabels(self, ax, labels):
        pass  # cosmetic; anyplotlib tick control not yet exposed

    def set_yticklabels(self, ax, labels):
        pass  # cosmetic; anyplotlib tick control not yet exposed

    def set_xticks(self, ax, ticks):
        pass  # cosmetic; anyplotlib tick control not yet exposed

    def set_yticks(self, ax, ticks):
        pass  # cosmetic; anyplotlib tick control not yet exposed

    def tight_layout(self, fig):
        pass  # anyplotlib uses constrained layout automatically

    def get_figure_from_ax(self, ax):
        fig = getattr(ax, "figure", None)
        if fig is not None:
            return fig
        raise BackendCapabilityError(_NOT_YET.format("get_figure_from_ax"))

    def connect_close_event(self, fig, fn):
        # anyplotlib close handling is done via on_close= at figure creation
        # time; there is no post-hoc connect mechanism yet.
        return None

    def get_explorer(self, signal_dim):
        if signal_dim == 0:
            from hyperspy.drawing.he import HyperExplorer

            return HyperExplorer
        elif signal_dim == 1:
            from hyperspy.drawing.backends.anyplotlib._explorers import (
                Apl_HyperSignal1D_Explorer,
            )

            return Apl_HyperSignal1D_Explorer
        elif signal_dim == 2:
            from hyperspy.drawing.backends.anyplotlib._explorers import (
                Apl_HyperImage_Explorer,
            )

            return Apl_HyperImage_Explorer
        raise ValueError(f"Plotting is not supported for signal_dim={signal_dim}.")

    def create_signal1d_figure(self, title="", on_close=None, **kwargs):
        from hyperspy.drawing.signal1d import Signal1DFigure

        return Signal1DFigure(title=title, _on_figure_window_close=on_close, **kwargs)

    def create_image_figure(self, title="", **kwargs):
        from hyperspy.drawing.image import ImagePlot

        return ImagePlot(title=title, **kwargs)

    def create_scalebar(self, ax, units, pixel_size=None, color="white", **kwargs):
        """Enable anyplotlib's native floating scale bar on ax's Plot2D.

        anyplotlib draws its own auto-sized, auto-positioned scale bar
        whenever a panel has calibrated axes and ``units != 'px'`` — no
        separate artist needed.  Re-``set_extent`` with the panel's own
        (already calibrated) axis arrays just to flip the units string turns
        it on.

        Falls back to the generic marker-based ``ScaleBar`` when there is no
        ``Plot2D`` yet, or when ``pixel_size`` is given explicitly (an
        uncalibrated axis with a manual pixel size, which the native bar has
        no equivalent for).

        When the axes simply have no usable units — an uncalibrated axis
        reports ``units`` as the ``traits.Undefined`` sentinel, which is not
        JSON-serialisable and must never reach ``Plot2D._state`` — no bar is
        drawn at all. anyplotlib labels such a panel in pixels and keeps its
        ticks, whereas the generic fallback would stamp ``10 <undefined>``
        across the image.
        """
        plot = self._primary_plot(ax)
        has_plot = plot is not None and hasattr(plot, "set_extent")
        usable_units = isinstance(units, str) and units not in ("", "px")

        if not (has_plot and pixel_size is None and usable_units):
            if has_plot and pixel_size is None:
                # No units to show, so nothing worth drawing.
                return _AplNoScalebar()
            from hyperspy.drawing._widgets.scalebar import ScaleBar

            return ScaleBar(ax=ax, units=units, pixel_size=pixel_size, color=color)

        x_axis = np.asarray(plot._state["x_axis"], dtype=float)
        y_axis = np.asarray(plot._state["y_axis"], dtype=float)
        plot.set_extent(x_axis, y_axis, units=str(units))
        # 0.5.0: the native bar takes a colour, so scalebar_color is honoured
        # on this path too — it used to work only on the fallback.
        if color is not None and hasattr(plot, "set_scalebar_style"):
            plot.set_scalebar_style(color=color)
        return _AplNativeScalebar(plot)

    def remove_scalebar(self, ax, handle):
        if not isinstance(handle, _AplNativeScalebar):
            remove = getattr(handle, "remove", None)
            if callable(remove):
                remove()
            return
        plot = handle.plot
        x_axis = np.asarray(plot._state["x_axis"], dtype=float)
        y_axis = np.asarray(plot._state["y_axis"], dtype=float)
        plot.set_extent(x_axis, y_axis, units="px")

    def get_image_cmap_name(self, handle):
        return getattr(handle, "colormap_name", "gray")


class _AplColorbar:
    """Handle returned by ``add_colorbar``.

    anyplotlib's colorbar is a flag on the image plot rather than a separate
    artist, so the handle just carries the plot it belongs to.
    """

    def __init__(self, im_handle):
        self._im = im_handle


class _AplNativeScalebar:
    """Sentinel for anyplotlib's built-in floating scale bar.

    Not an artist — the bar is drawn by the renderer whenever the panel has
    calibrated axes.  ``remove_scalebar`` flips the units back to ``'px'``.
    """

    def __init__(self, plot):
        self.plot = plot


class _AplNoScalebar:
    """Sentinel for "this panel deliberately has no scale bar".

    Used when the axes carry no usable units. anyplotlib labels such a panel
    in pixels and draws its ticks, which says everything a bar could; the
    generic marker-based fallback would instead stamp the image with
    ``10 <undefined>``, since an uncalibrated axis reports ``units`` as the
    ``traits.Undefined`` sentinel.
    """

    def remove(self):
        pass
