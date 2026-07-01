from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

if TYPE_CHECKING:
    from hyperspy.drawing.he import HyperExplorer


class CoordSpace:
    """Named coordinate spaces for backend-neutral coordinate conversion."""

    DATA = "data"
    AXES = "axes"
    DISPLAY = "display"
    XAXIS = "xaxis"
    YAXIS = "yaxis"
    RELATIVE = "relative"


class MarkerType:
    """Named marker types for backend-neutral marker creation."""

    POINTS = "points"
    CIRCLES = "circles"
    SQUARES = "squares"
    LINES = "lines"
    HLINES = "hlines"
    VLINES = "vlines"
    TEXTS = "texts"
    RECTANGLES = "rectangles"
    ELLIPSES = "ellipses"
    ARROWS = "arrows"
    POLYGONS = "polygons"


class BackendCapabilityError(NotImplementedError):
    """Raised when the active backend does not support a requested feature.

    Callers may catch this to degrade gracefully or emit a UserWarning.
    """


class BlitMixin(Protocol):
    """Default no-op implementations of all blit-related methods.

    Backends that do not support blitting inherit this mixin to satisfy the
    protocol without implementing anything.  Backends that *do* support
    blitting (e.g. ``MplBackend``) override the relevant methods.
    """

    def supports_blit(self, fig: Any) -> bool:
        """True when the backend can blit (skip full redraw)."""
        return False

    def copy_background(self, fig: Any) -> Any:
        """Capture the current non-animated background for blitting."""
        return None

    def restore_background(self, fig: Any, background: Any) -> None:
        """Restore a previously captured background."""

    def blit(self, fig: Any) -> None:
        """Flush the blit buffer to the screen."""

    def connect_draw_event(self, fig: Any, fn: Callable) -> Any:
        """Connect fn to the figure's post-draw event; return a cid."""
        return None

    def draw_animated_artists(self, fig: Any) -> None:
        """Redraw all animated artists in fig."""

    def render_figure_from_ax(self, ax: Any) -> None:
        """Trigger a repaint via the axes.  Uses blit when available."""
        self.draw_idle(getattr(ax, "figure", None))  # type: ignore[attr-defined]

    def invalidate_blit_background(self, ax: Any) -> None:
        """Invalidate the blit background so the next render does a full repaint."""

    def supports_blit_from_ax(self, ax: Any) -> bool:
        """True when ax is attached to a blitting-capable HyperSpy figure."""
        return False


class PointerMixin(Protocol):
    """Default implementations of navigation pointer widget methods.

    Backends inherit this mixin to get ``BackendCapabilityError`` defaults for
    all pointer primitives.  Backends that support interactive widgets override
    the relevant methods.

    Pointer methods are deliberately higher-level than raw patch manipulation:
    each ``create_*`` call returns an opaque handle that is passed back to the
    corresponding ``update_*`` and ``remove_pointer`` calls.

    Consolidated API (replaces the old per-axis / per-property methods):

    * ``create_line_pointer(ax, axis, pos, color)``
      Single method for both vertical (``axis='x'``) and horizontal
      (``axis='y'``) draggable line widgets.

    * ``update_line_pointer(handle, axis, pos)``
      Move the line to *pos* (x-value for 'x', y-value for 'y').

    * ``create_rect_pointer(ax, x, y, w, h, color, linewidth)``
      Create a draggable rectangle widget.

    * ``update_rect_pointer(handle, x, y, w, h)``
      Resize / reposition the rectangle.

    * ``remove_pointer(ax, handle)``
      Remove any pointer handle from the axes.

    * ``set_pointer_style(handle, *, color, alpha, animated)``
      Set any combination of visual style properties in one call.
    """

    # ── Pointer creation / update ─────────────────────────────────────────

    def create_line_pointer(
        self, ax: Any, axis: str, pos: float, color: str = "red"
    ) -> Any:
        """Create a draggable line widget.

        Parameters
        ----------
        ax : backend axes object
        axis : ``'x'`` for a vertical line, ``'y'`` for a horizontal line
        pos : initial position in data coordinates
        color : line color

        Returns
        -------
        opaque handle passed to ``update_line_pointer`` / ``remove_pointer``
        """
        raise BackendCapabilityError(
            "create_line_pointer not supported by this backend"
        )

    def update_line_pointer(self, handle: Any, axis: str, pos: float) -> None:
        """Move the line pointer to *pos* in data coordinates.

        Parameters
        ----------
        handle : opaque handle returned by ``create_line_pointer``
        axis : ``'x'`` for a vertical line, ``'y'`` for a horizontal line
        pos : new position in data coordinates
        """
        raise BackendCapabilityError(
            "update_line_pointer not supported by this backend"
        )

    def create_rect_pointer(
        self,
        ax: Any,
        x: float,
        y: float,
        w: float,
        h: float,
        color: str = "red",
        linewidth: float = 2,
    ) -> Any:
        """Create a draggable rectangle widget at lower-left (x, y) with size w×h."""
        raise BackendCapabilityError(
            "create_rect_pointer not supported by this backend"
        )

    def update_rect_pointer(
        self, handle: Any, x: float, y: float, w: float, h: float
    ) -> None:
        """Reposition/resize the rectangle pointer."""
        raise BackendCapabilityError(
            "update_rect_pointer not supported by this backend"
        )

    def remove_pointer(self, ax: Any, handle: Any) -> None:
        """Remove any pointer handle from the axes."""

    def set_pointer_style(
        self,
        handle: Any,
        *,
        color: str | None = None,
        alpha: float | None = None,
        animated: bool | None = None,
    ) -> None:
        """Set visual style properties on a pointer handle.

        Any ``None`` argument is left unchanged.  Replaces the old
        ``set_patch_color`` / ``set_patch_alpha`` / ``set_patch_animated``
        trio.
        """

    # ── Artist helpers ────────────────────────────────────────────────────

    def add_artist(self, ax: Any, artist: Any) -> None:
        """Add a pre-created artist to ax (used for MPL-native patches)."""

    def create_rect_patch(self, pos, w: float, h: float, **kwargs) -> Any:
        """Create a rectangle patch (for resizer handles)."""
        raise BackendCapabilityError("create_rect_patch not supported by this backend")

    def get_data_transform_inverse(self, ax: Any) -> Any:
        """Return an inverse-data transform (for pixel→data conversions)."""
        raise BackendCapabilityError(
            "get_data_transform_inverse not supported by this backend"
        )

    def transform_point(self, transform: Any, point) -> Any:
        """Apply transform to point."""
        raise BackendCapabilityError("transform_point not supported by this backend")

    # ── Widget callbacks ──────────────────────────────────────────────────

    def simulate_pick(self, ax: Any, patch: Any) -> None:
        """Simulate a pick event on *patch* to make it the active widget."""

    def connect_widget_drag(self, handle: Any, on_drag: Callable) -> None:
        """Register *on_drag* to fire when the native widget is dragged.

        *on_drag* is called with the new position: ``(x,)`` for 1-D widgets,
        ``(x, y)`` for 2-D.  MPL-based backends leave this as a no-op and
        route drag through ``_onmousemove``.
        """

    # ── Interactive selectors ─────────────────────────────────────────────

    def create_span_selector(self, ax: Any, **kwargs) -> Any:
        """Create and return an interactive span selector on ax."""
        raise BackendCapabilityError("SpanSelector requires a matplotlib-based backend")

    def create_polygon_selector(self, ax: Any, **kwargs) -> Any:
        """Create and return an interactive polygon selector on ax."""
        raise BackendCapabilityError(
            "PolygonSelector requires a matplotlib-based backend"
        )

    # ── Coordinate transforms ─────────────────────────────────────────────

    def get_ax_transform(self, ax: Any, kind: str) -> Any:
        """Return a transform for the given kind.

        *kind* is one of ``'data'``, ``'axes'``, ``'xaxis'``, ``'yaxis'``,
        ``'display'``, ``'relative'``.
        """
        raise BackendCapabilityError(
            f"Transform '{kind}' not supported by this backend"
        )

    # ── Free-form patch creation ──────────────────────────────────────────

    def create_line2d_patch(self, x, y, **kwargs) -> Any:
        """Create a free-form line artist for use as a widget patch."""
        raise BackendCapabilityError(
            "create_line2d_patch not supported by this backend"
        )

    def create_circle_patch(self, xy, radius, **kwargs) -> Any:
        """Create a circle artist for use as a widget patch."""
        raise BackendCapabilityError(
            "create_circle_patch not supported by this backend"
        )

    # ── Coordinate conversion ─────────────────────────────────────────────

    def convert_coords(
        self,
        ax: Any,
        points,
        from_space: str,
        to_space: str,
    ):
        """Convert *points* from *from_space* to *to_space*.

        Parameters
        ----------
        ax : backend axes object
        points : array-like, shape (N, 2) or (2,)
            Points to convert.
        from_space, to_space : CoordSpace string
            One of ``CoordSpace.DATA``, ``CoordSpace.AXES``,
            ``CoordSpace.DISPLAY``, ``CoordSpace.XAXIS``,
            ``CoordSpace.YAXIS``, ``CoordSpace.RELATIVE``.

        Returns
        -------
        numpy.ndarray, shape (N, 2)
        """
        raise BackendCapabilityError("convert_coords not supported by this backend")

    # ── Native marker collections ─────────────────────────────────────────

    def create_markers(self, ax: Any, marker_type: str, **kwargs) -> Any:
        """Create a marker collection on *ax* and return an opaque handle.

        Parameters
        ----------
        ax : backend axes object
        marker_type : MarkerType string
            One of the ``MarkerType.*`` constants.
        offset_space : str, optional
            CoordSpace string for position coordinates (default ``'data'``).
        transform_space : str, optional
            CoordSpace string for marker shape/size coordinates (default
            ``'display'``).
        **kwargs
            Marker-type-specific data (offsets, sizes, colors, …) exactly as
            produced by ``Markers.get_current_kwargs()``.

        Returns
        -------
        handle : any
            Passed to ``update_markers`` / ``remove_markers``.
        """
        raise BackendCapabilityError("create_markers not supported by this backend")

    def update_markers(self, handle: Any, **kwargs) -> None:
        """Update a marker collection returned by ``create_markers``."""
        raise BackendCapabilityError("update_markers not supported by this backend")

    def remove_markers(self, ax: Any, handle: Any) -> None:
        """Remove a marker collection from *ax*."""

    # ── Step plot ─────────────────────────────────────────────────────────

    def plot_step(self, ax: Any, x, y, **props) -> Any:
        """Draw a step plot; return an opaque handle."""
        raise BackendCapabilityError("plot_step not supported by this backend")

    # ── Axes control ─────────────────────────────────────────────────────

    def set_autoscale(self, ax: Any, enable: bool) -> None:
        """Enable or disable axes autoscale."""

    def set_xticklabels(self, ax: Any, labels) -> None:
        """Set the x-axis tick labels."""

    def set_yticklabels(self, ax: Any, labels) -> None:
        """Set the y-axis tick labels."""

    def set_xticks(self, ax: Any, ticks) -> None:
        """Set the x-axis tick positions (``[]`` removes the ticks entirely)."""

    def set_yticks(self, ax: Any, ticks) -> None:
        """Set the y-axis tick positions (``[]`` removes the ticks entirely)."""


@runtime_checkable
class PlottingBackend(BlitMixin, PointerMixin, Protocol):
    """Interface every HyperSpy plotting backend must satisfy.

    **Structure**

    * :class:`BlitMixin` — blit helpers with safe no-op defaults.  Inherit it
      to skip blit support entirely; override to enable it.
    * :class:`PointerMixin` — navigation widget primitives with
      ``BackendCapabilityError`` defaults.  Override the methods your backend
      supports.
    * ``PlottingBackend`` — core drawing primitives that **every** backend must
      implement (figure lifecycle, axes, lines, images, colorbars, events,
      markers, layout).

    All methods receive backend-native objects returned by earlier backend
    calls.  The generic drawing layer never imports matplotlib or anyplotlib
    directly; it calls only these methods.
    """

    # ── Figure lifecycle ──────────────────────────────────────────────────

    def create_figure(
        self, title: str | None = None, on_close: Callable | None = None, **kwargs
    ) -> Any:
        """Return a new figure object. on_close() is called when closed."""

    def close_figure(self, fig: Any) -> None:
        """Programmatically destroy a figure."""

    def draw_idle(self, fig: Any) -> None:
        """Schedule a non-blocking redraw."""

    def disconnect_event(self, fig_or_ax: Any, cid: Any) -> None:
        """Remove a previously connected event handler."""

    # ── Axes setup ───────────────────────────────────────────────────────

    def create_axes(self, fig: Any, animate_axis: bool = False, **kwargs) -> Any:
        """Create and return a primary axes inside fig."""

    def set_xlabel(self, ax: Any, label: str) -> None: ...
    def set_ylabel(self, ax: Any, label: str) -> None: ...
    def set_title(self, ax: Any, title: str) -> None: ...
    def set_xlim(self, ax: Any, xmin: float, xmax: float) -> None: ...
    def set_ylim(self, ax: Any, ymin: float, ymax: float) -> None: ...
    def get_xlim(self, ax: Any) -> tuple[float, float]: ...
    def get_ylim(self, ax: Any) -> tuple[float, float]: ...
    def get_xbound(self, ax: Any) -> tuple[float, float]: ...
    def set_axis_off(self, ax: Any) -> None: ...
    def set_aspect(self, ax: Any, ratio: float) -> None: ...
    def add_right_axis(self, ax: Any, color: str = "black") -> Any: ...
    def remove_right_axis(self, ax: Any, right_ax: Any) -> None: ...

    # ── 1-D line plotting ─────────────────────────────────────────────────

    def plot_line(self, ax: Any, x, y, **props) -> Any:
        """Draw a line; return an opaque handle for later updates."""

    def update_line(self, handle: Any, x, y) -> None: ...
    def remove_line(self, ax: Any, handle: Any) -> None: ...
    def set_line_props(self, handle: Any, **props) -> None: ...
    def line_get_xdata(self, handle: Any): ...
    def line_get_color(self, handle: Any) -> str: ...
    def line_get_linewidth(self, handle: Any) -> float: ...

    # ── Text annotations ─────────────────────────────────────────────────

    def add_text(
        self, ax: Any, x: float, y: float, s: str, transform: str = "axes", **kwargs
    ) -> Any:
        """Add a text label.

        *transform* is a string key: ``'axes'`` (default), ``'data'``,
        ``'xaxis'``, ``'yaxis'``, ``'display'``, or ``'relative'``.
        """

    def update_text(self, handle: Any, s: str) -> None: ...
    def remove_text(self, ax: Any, handle: Any) -> None: ...
    def text_set_color(self, handle: Any, color) -> None: ...
    def text_get_color(self, handle: Any) -> str: ...

    # ── Generic artist property ───────────────────────────────────────────

    def artist_set_animated(self, handle: Any, animated: bool) -> None:
        """Set the animated flag on an artist handle (used for blit pipeline)."""

    # ── 2-D image plotting ────────────────────────────────────────────────

    def plot_image(
        self,
        ax: Any,
        data,
        extent=None,
        vmin=None,
        vmax=None,
        norm=None,
        cmap: str = "gray",
        **kwargs,
    ) -> Any: ...

    def plot_mesh(self, ax: Any, x, y, data, **kwargs) -> Any: ...
    def image_set_data(self, handle: Any, data) -> None: ...
    def image_set_extent(self, handle: Any, extent) -> None: ...
    def image_set_clim(self, handle: Any, vmin, vmax) -> None: ...
    def image_set_norm(self, handle: Any, norm) -> None: ...
    def get_image_handle(self, ax: Any) -> Any | None:
        """Return the current image/mesh handle on ax, or None."""

    # ── Colorbar ─────────────────────────────────────────────────────────

    def add_colorbar(
        self,
        fig: Any,
        im_handle: Any,
        ax: Any,
        divider: bool = False,
        size: Any = "5%",
        pad: float = 0.05,
    ) -> Any: ...
    def colorbar_set_label(self, cb: Any, label: str) -> None: ...
    def colorbar_remove(self, cb: Any) -> None: ...
    def colorbar_redraw(self, cb: Any, fig: Any) -> None: ...

    # ── Event connections ─────────────────────────────────────────────────

    def connect_key_press(self, fig_or_ax: Any, fn: Callable) -> Any: ...
    def connect_mouse_move(self, fig_or_ax: Any, fn: Callable) -> Any: ...
    def connect_mouse_press(self, fig_or_ax: Any, fn: Callable) -> Any: ...
    def connect_mouse_release(self, fig_or_ax: Any, fn: Callable) -> Any: ...
    def connect_pick(self, fig_or_ax: Any, fn: Callable) -> Any: ...

    # ── Marker collections ────────────────────────────────────────────────

    def add_collection(self, ax: Any, collection) -> Any: ...
    def collection_update(self, handle: Any, **kwargs) -> None: ...
    def collection_remove(self, ax: Any, handle: Any) -> None: ...

    # ── Layout helpers ────────────────────────────────────────────────────

    def tight_layout(self, fig: Any) -> None:
        """Apply tight_layout to fig (best-effort hint; no-op by default)."""

    def get_figure_from_ax(self, ax: Any) -> Any:
        """Return the parent figure of ax."""
        raise BackendCapabilityError("get_figure_from_ax not supported by this backend")

    # ── Combined multi-panel layout ───────────────────────────────────────

    def create_combined_figure_panels(self, figsize=None) -> tuple[Any, Any] | None:
        """Return (nav_fig, signal_fig) for a combined single-widget layout.

        Return ``None`` to use two separate figures (the default).
        """
        return None

    def ensure_displayed(self, fig: Any) -> None:
        """Called after plot() completes.

        Backends that defer display use this to force the final render.
        Default is a no-op.
        """

    def connect_close_event(self, fig: Any, fn: Callable) -> Any:
        """Connect *fn* to the figure close/destroy event; return a cid."""
        return None

    def get_explorer(self, signal_dim: int) -> type[HyperExplorer]:
        """Return the HyperExplorer subclass for *signal_dim* (0, 1, or 2).

        **Every backend must override this method.**
        """
        from hyperspy.drawing.he import HyperExplorer

        return HyperExplorer

    # ── Figure manager factories ──────────────────────────────────────────

    def create_signal1d_figure(self, title: str = "", on_close=None, **kwargs) -> Any:
        """Create and return an :class:`~hyperspy.drawing.figure.AbstractSignal1DFigure`.

        Parameters
        ----------
        title : str
            Window / figure title.
        on_close : callable or None
            Zero-argument callback fired when the figure window is closed.
        **kwargs
            Forwarded to the underlying figure constructor.
        """
        raise BackendCapabilityError(
            "create_signal1d_figure not implemented by this backend"
        )

    def create_image_figure(self, title: str = "", **kwargs) -> Any:
        """Create and return an :class:`~hyperspy.drawing.figure.AbstractImageFigure`.

        Parameters
        ----------
        title : str
            Window / figure title.
        **kwargs
            Forwarded to the underlying figure constructor.
        """
        raise BackendCapabilityError(
            "create_image_figure not implemented by this backend"
        )

    # ── Scale bar ─────────────────────────────────────────────────────────

    def create_scalebar(self, ax: Any, units: str, **kwargs) -> Any:
        """Overlay a calibrated scale bar on *ax*; return an opaque handle.

        Parameters
        ----------
        ax : backend axes object
        units : str
            Physical units label (e.g. ``'nm'``, ``'μm'``).
        pixel_size : float or None
            Physical size of one pixel.  ``None`` uses calibrated axis values.
        color : str
            Bar and label colour.  Default ``'white'``.
        animated : bool
            Whether to use animated (blit) rendering.

        Returns
        -------
        handle : any
            Passed to :meth:`remove_scalebar`.
        """
        raise BackendCapabilityError("create_scalebar not supported by this backend")

    def remove_scalebar(self, ax: Any, handle: Any) -> None:
        """Remove a previously created scale bar from *ax*."""

    # ── Image helpers ─────────────────────────────────────────────────────

    def get_image_cmap_name(self, handle: Any) -> str:
        """Return the colormap name string for an image handle."""
        raise BackendCapabilityError(
            "get_image_cmap_name not supported by this backend"
        )
