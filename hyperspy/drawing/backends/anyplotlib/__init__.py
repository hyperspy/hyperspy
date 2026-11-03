"""anyplotlib plotting backend for hyperspy.

Requires anyplotlib >= 0.11.0.  Widgets and markers are created with
``coords="data"`` so they are placed, updated and reported in hyperspy's
calibrated units; Python-initiated widget updates use
``Widget.set(notify=False)`` so they do not echo back through the drag
callbacks.
"""

from __future__ import annotations

import weakref

import numpy as np

from hyperspy.drawing.backends._events import (
    KeyEvent,
    PointerEvent,
    canonical_key,
    canonical_modifiers,
)
from hyperspy.drawing.backends._protocol import BackendCapabilityError, TextMarkup

_NOT_YET = "the anyplotlib backend does not support '{}'."


def _jsonable(value):
    """Return *value* (circle sizes) in a JSON-serialisable form.

    anyplotlib broadcasts the one-element sequences hyperspy uses for
    singleton style values (``(v,)``) itself; only numpy containers need
    converting.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        try:
            return np.asarray(value).tolist()
        except ValueError:  # ragged
            return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


class _AplCombinedLayout:
    """Shared state of the panels of one combined Figure."""

    def __init__(self, fig, n):
        self.fig = fig
        # The figure is displayed once every panel has drawn.
        self.panels_remaining = n


class _AplFigureProxy:
    """One panel of a shared anyplotlib Figure (combined navigator/signal layout).

    Every proxy holds the same underlying Figure widget so it is displayed
    only once.  Closing a panel runs its own callbacks but never closes the
    shared figure.
    """

    def __init__(self, layout, ax):
        self._layout = layout
        self._real_fig = layout.fig
        self.ax = ax
        self.on_close = []


# Figure -> its combined-panel layout, so ``draw_idle`` on the real figure
# (``ax.figure`` of a panel) honours the panel countdown.  The value is weak:
# the layout references the figure, which would otherwise keep the key alive.
_FIGURE_LAYOUTS = weakref.WeakKeyDictionary()


def _quiet(fn):
    """Wrap a zero-argument close callback as an anyplotlib handler."""

    def handler(event=None):
        try:
            fn()
        except Exception:
            pass

    return handler


# hyperspy widget linewidths are matplotlib points; anyplotlib strokes are CSS
# pixels.  Convert so the strokes match the matplotlib backend's weight.
_PX_PER_PT = 96.0 / 72.0


def _stroke_px(linewidth, default=2.0):
    """Convert a linewidth in points to CSS pixels for native widgets."""
    if linewidth is None:
        linewidth = default
    return float(linewidth) * _PX_PER_PT


class _AplSpanSelector:
    """matplotlib ``SpanSelector`` façade over anyplotlib's native range widget.

    :class:`~.drawing._widgets.range.RangeWidget` drives the matplotlib
    selector API (``extents``, ``artists``, ``snap_values``, ``set_props``,
    ``connect_event``, ``_selection_completed``); this maps it onto the
    anyplotlib widget.  Snapping is delegated to the widget so it happens
    during the JS drag.  anyplotlib has no separate handle style, so
    ``set_handle_props`` is ``set_props``.
    """

    def __init__(self, plot, x0, x1, color="red", orientation="horizontal"):
        self._plot = plot
        self._orientation = orientation
        self._widget = plot.add_range_widget(
            float(x0), float(x1), color=color, orientation=orientation
        )
        self._handlers = []
        # Read by RangeWidget; there is no "drag to create" state to suppress.
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
        self._widget.set(x0=float(left), x1=float(right), notify=False)

    @property
    def snap_values(self):
        return self._widget.get("snap_values")

    @snap_values.setter
    def snap_values(self, values):
        self._widget.set(
            snap_values=None if values is None else [float(v) for v in values],
            notify=False,
        )

    @property
    def artists(self):
        return [self._widget]

    # ── styling ──────────────────────────────────────────────────────────

    def set_props(self, **props):
        if "color" in props:
            self._widget.set(color=props["color"], notify=False)

    set_handle_props = set_props

    # ── events / lifecycle ───────────────────────────────────────────────

    def connect_event(self, event_name, fn):
        """Map ``motion_notify_event`` onto the widget's ``pointer_move``."""
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

    Provides what :class:`~.api.roi.PolygonROI` uses: ``verts``, an
    ``onselect`` callback fired on browser-side edits, and the
    ``set_props`` / ``disconnect_events`` lifecycle.  anyplotlib cannot draw a
    polygon interactively from scratch, so the native widget is only created
    once there are at least three vertices.
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
            return [tuple(v) for v in self._widget.get("vertices", [])]
        return list(self._verts)

    @verts.setter
    def verts(self, value):
        self._verts = [tuple(float(c) for c in v) for v in value]
        if len(self._verts) < 3:
            return
        vertices = [list(v) for v in self._verts]
        if self._widget is None:
            self._widget = self._plot.add_widget(
                "polygon",
                vertices=vertices,
                color=self._color,
                linewidth=_stroke_px(self._linewidth),
                coords="data",
            )
            self._connect_onselect()
        else:
            self._widget.set(vertices=vertices, notify=False)

    def _connect_onselect(self):
        """Report browser-side polygon edits back to the ROI."""
        if self._widget is None or self._onselect is None:
            return

        def _handler(event):
            self._onselect(self.verts)

        self._widget.add_event_handler(_handler, "pointer_move")

    def set_props(self, **props):
        if "color" in props:
            self._color = props["color"]
            if self._widget is not None:
                self._widget.set(color=self._color, notify=False)

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
    """Handle returned by ``create_line2d_pointer``.

    The main (interactive) segment of a
    :class:`~.drawing._widgets.line2d.Line2DWidget` becomes a draggable
    two-endpoint ``line`` widget; its dotted width indicators become a
    ``'lines'`` marker group.  :meth:`materialise` attaches it to a panel.
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
        self._on_drag = None

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
                linewidth=_stroke_px(self._linewidth),
                coords="data",
            )
            self._connect_drag()
        else:
            self._group = plot.markers.add(
                "lines",
                segments=AnyplotlibBackend._polyline_segments(x, y),
                edgecolors=self._color,
                linewidths=_stroke_px(self._linewidth),
                coords="data",
            )

    def set_data(self, x, y=None):
        if y is None:  # Line2D.set_data accepts a single (2, N) array
            x, y = np.asarray(x, dtype=float)
        self._xy = (np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        x, y = self._xy
        if self._widget is not None and len(x) >= 2:
            self._widget.set(
                x1=float(x[0]),
                y1=float(y[0]),
                x2=float(x[-1]),
                y2=float(y[-1]),
                notify=False,
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
            self._widget.set(color=self._color, notify=False)
        elif self._group is not None:
            self._group.set(edgecolors=self._color)

    def connect_drag(self, on_drag):
        """Report endpoint moves as (x1, y1, x2, y2) in data coords."""
        self._on_drag = on_drag
        self._connect_drag()

    def _connect_drag(self):
        if self._widget is None or self._on_drag is None:
            return
        w, on_drag = self._widget, self._on_drag

        def _cb(event):
            on_drag(w.x1, w.y1, w.x2, w.y2)

        w.add_event_handler(_cb, "pointer_move")

    def remove(self):
        for obj in (self._widget, self._group):
            if obj is not None:
                try:
                    obj.remove()
                except Exception:
                    pass
        self._widget = None
        self._group = None


def _figsize_px(figsize, default):
    """Normalise a figure size to integer pixels.

    Values below 50 are taken to be matplotlib inches and converted at 96 dpi.
    """
    if figsize is None:
        figsize = default
    figsize = tuple(float(v) for v in figsize)
    if max(figsize) < 50:
        figsize = (figsize[0] * 96, figsize[1] * 96)
    return (int(figsize[0]), int(figsize[1]))


def _texts_tex_lite() -> bool:
    """Whether the installed anyplotlib typesets and rotates texts markers.

    Rotation threading through ``add_texts`` marks the release that routes
    ``$...$`` markup through the TeX engine; older releases draw raw
    strings only.
    """
    try:
        import inspect

        import anyplotlib as _apl

        return "rotation" in inspect.signature(_apl.Plot2D.add_texts).parameters
    except (ImportError, AttributeError, ValueError):
        # A missing or older anyplotlib means the capability is absent:
        # fall back to the safe plain-text declaration.
        return False


class AnyplotlibBackend:
    """Maps hyperspy drawing primitives to the anyplotlib API.

    Methods that raise ``BackendCapabilityError`` name a capability anyplotlib
    does not have, so the caller can fall back (as ``Markers.plot`` does) or
    report it rather than draw the wrong thing.
    """

    # Mathtext labels typeset through anyplotlib's TeX engine and rotation
    # is honoured when the installed release supports it; older releases
    # draw raw strings only, so they keep the plain-text declaration and
    # the marker layer degrades mathtext gracefully instead.
    text_markup = TextMarkup.TEX_LITE if _texts_tex_lite() else TextMarkup.PLAIN
    supports_text_rotation = _texts_tex_lite()

    def __init__(self):
        import anyplotlib as apl

        # The data-coordinate and identity API this backend relies on arrived
        # in anyplotlib 0.11; fail clearly rather than with an AttributeError
        # deep inside a plot call.
        if not hasattr(apl.Axes, "primary_plot"):
            raise ImportError(
                "The anyplotlib plotting backend requires anyplotlib>=0.11.0; "
                f"anyplotlib {getattr(apl, '__version__', '?')} is installed. "
                "Upgrade with `pip install -U anyplotlib`."
            )

    # ── Figure lifecycle ──────────────────────────────────────────────────

    def create_figure(self, title=None, on_close=None, **kwargs):
        import anyplotlib as apl

        # A pre-created panel proxy (combined layout) is adopted as-is.
        fig_kwarg = kwargs.pop("fig", None)
        if isinstance(fig_kwarg, _AplFigureProxy):
            if on_close is not None:
                fig_kwarg.on_close.append(on_close)
            return fig_kwarg

        figsize = _figsize_px(kwargs.pop("figsize", None), (640, 480))
        fig, _ = apl.subplots(1, 1, figsize=figsize)
        if on_close is not None:
            fig.on_close(_quiet(on_close))
        return fig

    def close_figure(self, fig):
        if fig is None:
            return
        if isinstance(fig, _AplFigureProxy):
            # A panel proxy runs its callbacks but never closes the shared
            # figure.
            callbacks, fig.on_close = fig.on_close, []
            for fn in callbacks:
                _quiet(fn)()
            return
        try:
            fig.close()  # fires the on_close handlers
        except Exception:
            pass

    def create_combined_figure_panels(self, figsize=None, n=2):
        """Create an *n*-panel Figure and return one proxy per panel.

        The proxies share one Figure widget, which ``draw_idle`` displays
        only once every panel has drawn.
        """
        import anyplotlib as apl

        fig, axes = apl.subplots(1, n, figsize=_figsize_px(figsize, (640 * n, 640)))
        if n == 1:
            axes = [axes]
        layout = _AplCombinedLayout(fig, n)
        _FIGURE_LAYOUTS[fig] = weakref.ref(layout)
        return tuple(_AplFigureProxy(layout, ax) for ax in axes)

    @staticmethod
    def _display(fig):
        if isinstance(fig, _AplFigureProxy):
            fig._layout.panels_remaining = 0
            fig = fig._real_fig
        try:
            fig.display()  # idempotent
        except ImportError:  # pragma: no cover - IPython missing
            pass

    def ensure_displayed(self, fig):
        """Display *fig* now, even if some panels have not drawn yet.

        Called after ``plot()`` so a figure is shown even when the navigator
        panel was skipped (slider / None).
        """
        if fig is not None:
            self._display(fig)

    def draw_idle(self, fig):
        if fig is None:
            return
        if isinstance(fig, _AplFigureProxy):
            layout = fig._layout
        else:
            ref = _FIGURE_LAYOUTS.get(fig)
            layout = ref() if ref is not None else None
        if layout is not None:
            if layout.fig.displayed:
                return
            if layout.panels_remaining > 1:
                # Wait for the other panels so the figure appears whole.
                layout.panels_remaining -= 1
                return
        elif fig.displayed:
            return
        self._display(fig)

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
        if isinstance(fig, _AplFigureProxy):
            return fig.ax
        return fig.get_axes()[0]

    # Labels set before a plot is attached are kept by anyplotlib's Axes.

    def set_xlabel(self, ax, label):
        ax.set_xlabel(label)

    def set_ylabel(self, ax, label):
        ax.set_ylabel(label)

    def set_title(self, ax, title):
        ax.set_title(title)

    def set_xlim(self, ax, xmin, xmax):
        if ax.primary_plot is not None:
            ax.primary_plot.set_xlim(xmin, xmax)

    def set_ylim(self, ax, ymin, ymax):
        if ax.primary_plot is not None:
            ax.primary_plot.set_ylim(ymin, ymax)

    def get_xlim(self, ax):
        plot = ax.primary_plot
        if plot is None:
            return (0.0, 1.0)
        for name in ("get_xlim", "get_xbound"):
            fn = getattr(plot, name, None)
            if fn is not None:
                return fn()
        return (0.0, 1.0)

    def get_ylim(self, ax):
        plot = ax.primary_plot
        if plot is not None and hasattr(plot, "get_ylim"):
            return plot.get_ylim()
        return (0.0, 1.0)

    def get_xbound(self, ax):
        if ax.primary_plot is not None:
            return ax.primary_plot.get_xbound()
        return (0.0, 1.0)

    def set_axis_off(self, ax):
        if ax.primary_plot is not None:
            ax.primary_plot.set_axis_off()

    def set_aspect(self, ax, ratio):
        plot = ax.primary_plot
        if plot is not None and hasattr(plot, "set_aspect"):
            plot.set_aspect(ratio)

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
        """Translate matplotlib line properties to anyplotlib kwargs.

        Only keys present in *props* are returned, so callers can tell
        "not specified" from "specified as the default".
        """
        out = {}

        if "color" in props and props["color"] is not None:
            out["color"] = props["color"]
        elif props.get("markeredgecolor") is not None:
            # Signal1DLine.color moves the colour of scatter lines here.
            out["color"] = props["markeredgecolor"]

        if "linewidth" in props and props["linewidth"] is not None:
            out["linewidth"] = float(props["linewidth"])
        if "alpha" in props and props["alpha"] is not None:
            out["alpha"] = float(props["alpha"])

        if "marker" in props:
            out["marker"] = cls._norm_marker(props["marker"])
        if props.get("markersize") is not None:
            out["markersize"] = float(props["markersize"])

        # A non-empty label makes the line a native legend entry.
        if props.get("label"):
            out["label"] = str(props["label"])

        linestyle = props.get("linestyle")
        drawstyle = props.get("drawstyle")
        if linestyle is not None:
            if str(linestyle) in ("None", "none", "", " "):
                # Markers only.
                out["linestyle"] = "none"
                out.setdefault("marker", cls._norm_marker(props.get("marker", "o")))
            else:
                out["linestyle"] = str(linestyle)
        # anyplotlib expresses matplotlib's drawstyle="steps-*" as a linestyle,
        # which overrides the plain linestyle hyperspy sends alongside it.
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

        existing = ax.primary_plot

        if existing is None:
            return ax.plot(
                y_arr,
                axes=[x_arr] if x_arr is not None else None,
                color=color,
                linewidth=linewidth,
                **style,
            )

        if ax.image_plot is not None or not hasattr(existing, "add_line"):
            # Image panel: overlay drawn as a native 'lines' marker group.
            return existing.markers.add(
                "lines",
                segments=self._polyline_segments(x_arr, y_arr),
                edgecolors=color,
                linewidths=linewidth,
                coords="data",
            )

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
        elif hasattr(handle, "marker_type"):  # 'lines' MarkerGroup
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
        if hasattr(handle, "marker_type"):  # 'lines' MarkerGroup
            return handle.get("edgecolors", "#4fc3f7")
        return "#4fc3f7"

    def line_get_linewidth(self, handle):
        if hasattr(handle, "marker_type"):  # 'lines' MarkerGroup
            return float(handle.get("linewidths", 1.5))
        import anyplotlib as apl

        if isinstance(handle, apl.Plot1D):  # primary line: no public getter yet
            return float(handle._state.get("line_linewidth", 1.5))
        if hasattr(handle, "linewidth"):  # Line1D property
            return float(handle.linewidth)
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
        space = self._SPACE_MAP.get(transform, "data")
        if space == "data" and hasattr(plot, "data_to_pixel"):
            # Plot2D.add_text places data-space text in image pixels.
            x, y = plot.data_to_pixel((float(x), float(y)))
        return plot.add_text(
            float(x),
            float(y),
            str(s),
            color=color,
            fontsize=float(fontsize),
            transform=space,
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
        group = getattr(handle, "_group", None)  # no public getter yet
        if group is not None:
            return group.get("color", "white")
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
        # extent uses matplotlib's (left, right, bottom, top) edge convention.
        plot = ax.imshow(
            np.asarray(data),
            extent=None if extent is None else tuple(float(v) for v in extent),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        if aspect not in (None, "auto", "equal", 1, 1.0):
            plot.set_aspect(float(aspect))
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
        handle.set_extent_edges(*(float(v) for v in extent))

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
        return ax.image_plot or ax.primary_plot

    def add_colorbar(self, fig, im_handle, ax, divider=None, size=None, pad=None):
        im_handle.set_colorbar_visible(True)
        if pad is not None and hasattr(im_handle, "set_colorbar_pad"):
            im_handle.set_colorbar_pad(pad)
        return _AplColorbar(im_handle)

    def colorbar_set_label(self, cb, label):
        cb._im.set_colorbar_label(label)

    def colorbar_remove(self, cb):
        cb._im.set_colorbar_visible(False)

    def colorbar_redraw(self, cb, fig):
        pass

    # ── Events ───────────────────────────────────────────────────────────

    # Handlers receive the neutral events of ``backends._events``.  anyplotlib
    # reports browser key names ("ArrowLeft"), 0-based buttons and a modifier
    # list; they are translated here so the generic layer sees canonical
    # values.  ``xdata``/``ydata`` are already calibrated by anyplotlib.

    @staticmethod
    def _event_ax(fig_or_ax):
        """The anyplotlib Axes for a figure, panel proxy or axes."""
        if isinstance(fig_or_ax, _AplFigureProxy):
            return fig_or_ax.ax
        get_axes = getattr(fig_or_ax, "get_axes", None)
        if callable(get_axes):  # a Figure
            return get_axes()[0]
        return fig_or_ax

    @staticmethod
    def _to_key_event(ax, e):
        inside = e.xdata is not None and e.ydata is not None
        return KeyEvent(
            key=canonical_key(e.key, e.modifiers),
            inaxes=ax if inside else None,
            xdata=e.xdata,
            ydata=e.ydata,
            modifiers=canonical_modifiers(e.modifiers),
            native=e,
        )

    @staticmethod
    def _to_pointer_event(kind, ax, e):
        inside = e.xdata is not None and e.ydata is not None
        return PointerEvent(
            kind=kind,
            xdata=e.xdata,
            ydata=e.ydata,
            inaxes=ax if inside else None,
            x=e.x,
            y=e.y,
            button=None if e.button is None else int(e.button) + 1,
            modifiers=canonical_modifiers(e.modifiers),
            native=e,
        )

    def _connect(self, fig_or_ax, event_type, convert, fn):
        plot = self._get_plot(fig_or_ax)
        if plot is None:
            return None
        ax = self._event_ax(fig_or_ax)

        def _handler(e):
            return fn(convert(ax, e))

        return plot.add_event_handler(_handler, event_type)

    def connect_key_press(self, fig_or_ax, fn):
        return self._connect(fig_or_ax, "key_down", self._to_key_event, fn)

    def connect_mouse_move(self, fig_or_ax, fn):
        return self._connect(
            fig_or_ax,
            "pointer_move",
            lambda ax, e: self._to_pointer_event("move", ax, e),
            fn,
        )

    def connect_mouse_press(self, fig_or_ax, fn):
        return self._connect(
            fig_or_ax,
            "pointer_down",
            lambda ax, e: self._to_pointer_event("press", ax, e),
            fn,
        )

    def connect_mouse_release(self, fig_or_ax, fn):
        return self._connect(
            fig_or_ax,
            "pointer_up",
            lambda ax, e: self._to_pointer_event("release", ax, e),
            fn,
        )

    def _get_plot(self, fig_or_ax):
        """Return the plot attached to the given figure, proxy or axes."""
        if fig_or_ax is None:
            return None
        return getattr(self._event_ax(fig_or_ax), "primary_plot", None)

    def _primary_plot(self, ax):
        """Return the primary (image or line) plot for widget creation."""
        return ax.primary_plot

    # ── Pointers / widgets ────────────────────────────────────────────────

    def create_line_pointer(self, ax, axis, pos, color="red"):
        plot = self._primary_plot(ax)
        if plot is None:
            raise RuntimeError("ax has no plot; call plot_line or plot_image first")
        style = {"color": color, "linewidth": _stroke_px(2.0), "coords": "data"}
        if axis == "x":
            if hasattr(plot, "add_vline_widget"):
                return plot.add_vline_widget(x=float(pos), **style)
            return plot.add_widget("crosshair", cx=float(pos), cy=0.0, **style)
        if hasattr(plot, "add_hline_widget"):
            return plot.add_hline_widget(y=float(pos), **style)
        return plot.add_widget("crosshair", cx=0.0, cy=float(pos), **style)

    def update_line_pointer(self, handle, axis, pos):
        if handle.kind == "crosshair":
            key = "cx" if axis == "x" else "cy"
        else:
            key = "x" if axis == "x" else "y"
        handle.set(**{key: float(pos)}, notify=False)

    def connect_widget_drag(self, handle, on_drag):
        if isinstance(handle, _AplLine2DPatch):
            handle.connect_drag(on_drag)
            return
        # Widgets are created with coords="data", so their geometry already
        # reads in calibrated units.
        kind = getattr(handle, "kind", None)
        if kind == "vline":

            def _cb(event):
                on_drag(handle.x)

        elif kind == "hline":

            def _cb(event):
                on_drag(handle.y)

        elif kind in ("crosshair", "circle", "annular"):

            def _cb(event):
                on_drag(handle.cx, handle.cy)

        elif kind == "rectangle":
            # Corner plus size; RectangleWidget._on_widget_drag converts the
            # corner to its own position convention.

            def _cb(event):
                on_drag(handle.x, handle.y, handle.w, handle.h)

        elif kind == "polygon":

            def _cb(event):
                on_drag([tuple(v) for v in handle.get("vertices", [])])

        else:
            return
        handle.add_event_handler(_cb, "pointer_move")

    def create_rect_pointer(
        self, ax, x, y, w, h, color="red", linewidth=2, pointer=False
    ):
        plot = self._primary_plot(ax)
        if plot is None or not hasattr(plot, "add_widget"):
            raise BackendCapabilityError(_NOT_YET.format("create_rect_pointer"))
        style = {"color": color, "linewidth": _stroke_px(linewidth), "coords": "data"}
        if pointer:
            # Navigation pointers are drawn as a crosshair, or a single line
            # when the navigator is degenerate along one axis.
            cx, cy = float(x) + float(w) / 2.0, float(y) + float(h) / 2.0
            if h <= 0 and hasattr(plot, "add_vline_widget"):
                return plot.add_vline_widget(x=cx, **style)
            if w <= 0 and hasattr(plot, "add_hline_widget"):
                return plot.add_hline_widget(y=cy, **style)
            return plot.add_widget("crosshair", cx=cx, cy=cy, **style)
        # Region selector (e.g. RectangularROI) with native resize handles.
        return plot.add_widget(
            "rectangle", x=float(x), y=float(y), w=float(w), h=float(h), **style
        )

    def update_rect_pointer(self, handle, x, y, w, h):
        cx, cy = float(x) + float(w) / 2.0, float(y) + float(h) / 2.0
        if handle.kind == "crosshair":
            handle.set(cx=cx, cy=cy, notify=False)
        elif handle.kind == "vline":
            handle.set(x=cx, notify=False)
        elif handle.kind == "hline":
            handle.set(y=cy, notify=False)
        else:
            handle.set(x=float(x), y=float(y), w=float(w), h=float(h), notify=False)

    def create_circle_pointer(
        self, ax, cx, cy, r_outer, r_inner=0.0, color="red", linewidth=2, alpha=1.0
    ):
        plot = self._primary_plot(ax)
        if plot is None or not hasattr(plot, "add_widget"):
            raise BackendCapabilityError(_NOT_YET.format("create_circle_pointer"))
        style = {"color": color, "linewidth": _stroke_px(linewidth), "coords": "data"}
        # One native widget covers both circle and annulus: single-item list.
        # The widget is a true circle; its radius follows the x calibration.
        if r_inner > 0:
            handle = plot.add_widget(
                "annular",
                cx=float(cx),
                cy=float(cy),
                r_outer=float(r_outer),
                r_inner=float(r_inner),
                **style,
            )
        else:
            handle = plot.add_widget(
                "circle", cx=float(cx), cy=float(cy), r=float(r_outer), **style
            )
        return [handle]

    def update_circle_pointer(self, ax, handles, cx, cy, r_outer, r_inner=0.0):
        handle = handles[0]
        is_annular = handle.kind == "annular"
        if is_annular != (r_inner > 0):
            # circle <-> annulus are distinct widget kinds, so swap.
            new = self.create_circle_pointer(
                ax,
                cx,
                cy,
                r_outer,
                r_inner,
                color=handle.get("color", "red"),
                # The stored stroke is CSS px; convert back to points.
                linewidth=handle.get("linewidth", 2.0 * _PX_PER_PT) / _PX_PER_PT,
            )
            for old in handles:
                self.remove_pointer(ax, old)
            return new
        if is_annular:
            handle.set(
                cx=float(cx),
                cy=float(cy),
                r_outer=float(r_outer),
                r_inner=float(r_inner),
                notify=False,
            )
        else:
            handle.set(cx=float(cx), cy=float(cy), r=float(r_outer), notify=False)
        return handles

    def remove_pointer(self, ax, handle):
        if handle is None:
            return
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
        # animated is a blit concept and is ignored.
        if isinstance(handle, _AplLine2DPatch):
            handle.set_style(color=color, alpha=alpha)
            return
        updates = {}
        if color is not None:
            updates["color"] = color
        if alpha is not None:
            updates["alpha"] = float(alpha)
        if updates and hasattr(handle, "set"):
            handle.set(**updates, notify=False)

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

    # hyperspy coordinate spaces -> anyplotlib's {"data", "axes", "display"}.
    _SPACE_MAP = {
        "data": "data",
        "axes": "axes",
        "display": "display",
        "xaxis": "data",
        "yaxis": "data",
        "relative": "data",
    }

    def convert_coords(self, ax, points, from_space, to_space):
        """Convert points between coordinate spaces.

        Display-space conversion is delegated to anyplotlib's
        ``data_to_display`` / ``display_to_data``, which take image pixels on
        2-D panels; the panel's ``data_to_pixel`` / ``pixel_to_data`` bridge
        those to calibrated units.
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
        to_pixel = getattr(plot, "data_to_pixel", None)  # 2-D panels only
        from_pixel = getattr(plot, "pixel_to_data", None)

        def _to_data(vals, space):
            if space in ("data", "xaxis", "yaxis", "relative"):
                return vals
            if space == "axes":
                return np.column_stack(
                    [x0 + vals[:, 0] * xspan, y0 + vals[:, 1] * yspan]
                )
            # "display"
            vals = np.atleast_2d(plot.display_to_data(vals))
            return np.atleast_2d(from_pixel(vals)) if from_pixel else vals

        def _from_data(vals, space):
            if space in ("data", "xaxis", "yaxis", "relative"):
                return vals
            if space == "axes":
                return np.column_stack(
                    [(vals[:, 0] - x0) / xspan, (vals[:, 1] - y0) / yspan]
                )
            # "display"
            if to_pixel:
                vals = np.atleast_2d(to_pixel(vals))
            return np.atleast_2d(plot.data_to_display(vals))

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
            # Plot2D has no 'points' type: draw circles sized in display px
            # so they do not rescale with zoom.
            marker_type = "circles"
            sizes = translated.pop("sizes", None)
            if sizes is not None:
                radius = np.atleast_1d(np.asarray(sizes, dtype=float)) / 2.0
                translated["radius"] = (
                    float(radius[0]) if radius.size == 1 else radius.tolist()
                )
                if size_units not in ("x", "y", "xy", "width", "height"):
                    translated["size_units"] = "px"

        try:
            # Geometry is calibrated; axes/display transforms and px sizes are
            # left alone by anyplotlib.
            return plot.markers.add(marker_type, coords="data", **translated)
        except ValueError as exc:
            raise BackendCapabilityError(
                f"anyplotlib does not support marker type '{marker_type}' "
                f"on this plot type: {exc}"
            ) from exc

    def update_markers(self, handle, **kwargs):
        """Update a ``MarkerGroup`` returned by ``create_markers``."""
        if not kwargs:
            return
        marker_type = handle.marker_type
        offset_space = handle.get("transform", "data")
        if handle.get("size_units") == "px" and "sizes" in kwargs:
            sizes = np.atleast_1d(np.asarray(kwargs.pop("sizes"), dtype=float))
            radius = sizes / 2.0
            kwargs["radius"] = float(radius[0]) if radius.size == 1 else radius.tolist()
        translated = self._translate_marker_kwargs(marker_type, offset_space, kwargs)
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
        - numpy containers → JSON-serialisable lists
        - Strips MPL-only kwargs (``units``, ``patches``, ``drawstyle``, …)
        """
        out = {}
        out["transform"] = AnyplotlibBackend._SPACE_MAP.get(offset_space, "data")
        work = dict(kwargs)  # pop freely without mutating the caller's dict

        if "colors" in work:
            val = work.pop("colors")
            if isinstance(val, (list, tuple)) and len(val) == 1:
                val = val[0]
            work["edgecolors"] = val

        if "linewidth" in work and "linewidths" not in work:
            work["linewidths"] = work.pop("linewidth")

        if marker_type == "circles":
            if "sizes" in work:
                out["radius"] = _jsonable(work.pop("sizes"))

        elif marker_type in ("vlines", "hlines"):
            # hyperspy expands positions into full [[x,0],[x,1]] segments for
            # matplotlib; anyplotlib wants [[x], ...].
            if "segments" in work:
                segs = np.asarray(work.pop("segments"), dtype=float)
                if marker_type == "vlines":
                    out["offsets"] = [[float(v)] for v in segs[:, 0, 0]]
                else:
                    out["offsets"] = [[float(v)] for v in segs[:, 0, 1]]
            out["transform"] = "data"

        elif marker_type == "polygons":
            if "verts" in work:
                verts = work.pop("verts")
                out["vertices_list"] = [
                    np.asarray(v, dtype=float).tolist() for v in verts
                ]

        if marker_type == "texts" and "facecolor" in work:
            # matplotlib's text colour kwarg is `facecolor`; the texts
            # wire format reads `color` (and would otherwise fall back
            # to red).
            work["color"] = work.pop("facecolor")

        # A singleton colour given as an RGB(A) tuple, ``((r, g, b, a),)``:
        # anyplotlib only unwraps single-string colours.
        for key in ("edgecolors", "facecolors", "hover_edgecolors", "hover_facecolors"):
            val = work.get(key)
            if (
                isinstance(val, (list, tuple, np.ndarray))
                and len(val) == 1
                and not isinstance(val[0], str)
                and np.ndim(val[0]) == 1
            ):
                work[key] = np.asarray(val[0]).tolist()

        _STRIP = {"offset_transform", "units", "patches", "drawstyle"}
        for k, v in work.items():
            if k in _STRIP:
                continue
            if hasattr(v, "tolist"):
                if isinstance(v, np.ndarray) and v.size == 0:
                    # ``.tolist()`` would flatten e.g. an empty ``(0, 2)``
                    # offsets array to ``[]`` (shape ``(0,)`` after
                    # ``np.asarray``), which the wire format rejects as not
                    # ``(N, 2)``.  Happens when all items of a live marker
                    # collection are removed.  ``to_wire`` converts the raw
                    # ndarray losslessly.
                    out[k] = v
                else:
                    out[k] = v.tolist()  # JSON-serialisable
            else:
                out[k] = v

        return out

    # ── Remaining protocol methods ────────────────────────────────────────

    def plot_step(self, ax, x, y, **props):
        props.setdefault("drawstyle", "steps-mid")
        return self.plot_line(ax, x, y, **props)

    def create_line2d_pointer(
        self,
        ax,
        x,
        y,
        *,
        color="red",
        linewidth=1.0,
        alpha=1.0,
        linestyle="-",
        interactive=True,
        pick_radius=None,
        handle_size=None,
    ):
        plot = self._primary_plot(ax)
        if plot is None:
            raise RuntimeError("ax has no plot; call plot_line or plot_image first")
        # The main segment becomes a native line widget with its own endpoint
        # handles; pick_radius/handle_size only matter to artist backends.
        line = _AplLine2DPatch(
            x, y, color=color, linewidth=linewidth, alpha=alpha, interactive=interactive
        )
        line.materialise(plot)
        return line

    def update_line2d_pointer(self, handle, x, y):
        handle.set_data(x, y)

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

    def get_figure_from_ax(self, ax):
        fig = getattr(ax, "figure", None)
        if fig is not None:
            return fig
        raise BackendCapabilityError(_NOT_YET.format("get_figure_from_ax"))

    def connect_close_event(self, fig, fn):
        """Run *fn* when the figure (or panel proxy) is closed.

        anyplotlib has no browser-side close notification, so only a
        programmatic close counts.
        """
        if fig is None:
            return None
        if isinstance(fig, _AplFigureProxy):
            fig.on_close.append(fn)
            return None
        return fig.on_close(_quiet(fn))

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
        """Enable anyplotlib's native scale bar on the Plot2D of *ax*.

        anyplotlib draws the bar itself whenever the panel has calibrated axes
        and ``units != 'px'``, so this just sets the units string.  Without a
        Plot2D, or with an explicit *pixel_size*, the generic marker-based
        ``ScaleBar`` is used instead.  With no usable units (an uncalibrated
        axis reports ``traits.Undefined``) no bar is drawn: anyplotlib already
        labels the panel in pixels.
        """
        plot = self._primary_plot(ax)
        has_plot = plot is not None and hasattr(plot, "set_extent")
        usable_units = isinstance(units, str) and units not in ("", "px")

        if not (has_plot and pixel_size is None and usable_units):
            if has_plot and pixel_size is None:
                return _AplNoScalebar()
            from hyperspy.drawing._widgets.scalebar import ScaleBar

            return ScaleBar(ax=ax, units=units, pixel_size=pixel_size, color=color)

        plot.set_extent(plot.x_axis, plot.y_axis, units=str(units))
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
        plot.set_extent(plot.x_axis, plot.y_axis, units="px")

    def get_image_cmap_name(self, handle):
        return getattr(handle, "colormap_name", "gray")


class _AplColorbar:
    """Handle returned by ``add_colorbar``; the colorbar is a flag on the plot."""

    def __init__(self, im_handle):
        self._im = im_handle


class _AplNativeScalebar:
    """Handle for anyplotlib's built-in scale bar (``remove_scalebar`` resets
    the units to ``'px'``)."""

    def __init__(self, plot):
        self.plot = plot


class _AplNoScalebar:
    """Handle for a panel without usable units, where no bar is drawn."""

    def remove(self):
        pass
