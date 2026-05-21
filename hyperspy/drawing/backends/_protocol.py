from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable


@runtime_checkable
class PlottingBackend(Protocol):
    """Minimal interface every hyperspy plotting backend must implement.

    All methods receive backend-native objects (figure, axes, handles)
    returned by earlier backend calls.  The generic drawing layer never
    imports matplotlib or anyplotlib directly; it only calls these methods.
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

    def supports_blit(self, fig: Any) -> bool:
        """True when the backend can blit (skip full redraw)."""

    def copy_background(self, fig: Any) -> Any:
        """Capture the current non-animated background for blitting."""

    def restore_background(self, fig: Any, background: Any) -> None:
        """Restore a previously captured background."""

    def blit(self, fig: Any) -> None:
        """Flush the blit buffer to the screen."""

    def connect_draw_event(self, fig: Any, fn: Callable) -> Any:
        """Connect fn to the figure's post-draw event; return a cid."""

    def disconnect_event(self, fig_or_ax: Any, cid: Any) -> None:
        """Remove a previously connected event handler."""

    # ── Axes setup ───────────────────────────────────────────────────────

    def create_axes(self, fig: Any, **kwargs) -> Any:
        """Create and return a primary axes inside fig."""

    def set_xlabel(self, ax: Any, label: str) -> None: ...
    def set_ylabel(self, ax: Any, label: str) -> None: ...
    def set_title(self, ax: Any, title: str) -> None: ...
    def set_xlim(self, ax: Any, xmin: float, xmax: float) -> None: ...
    def set_ylim(self, ax: Any, ymin: float, ymax: float) -> None: ...
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

    # ── Text annotations ─────────────────────────────────────────────────

    def add_text(
        self, ax: Any, x: float, y: float, s: str, transform: str = "axes", **kwargs
    ) -> Any:
        """Add a text label.  transform='axes' means (0,0)=bottom-left."""

    def update_text(self, handle: Any, s: str) -> None: ...
    def remove_text(self, ax: Any, handle: Any) -> None: ...

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

    def add_colorbar(self, fig: Any, im_handle: Any, ax: Any) -> Any: ...
    def colorbar_set_label(self, cb: Any, label: str) -> None: ...
    def colorbar_remove(self, cb: Any) -> None: ...
    def colorbar_redraw(self, cb: Any, fig: Any) -> None: ...

    # ── Event connections ─────────────────────────────────────────────────

    def connect_key_press(self, fig_or_ax: Any, fn: Callable) -> Any: ...
    def connect_mouse_move(self, fig_or_ax: Any, fn: Callable) -> Any: ...
    def connect_mouse_press(self, fig_or_ax: Any, fn: Callable) -> Any: ...
    def connect_mouse_release(self, fig_or_ax: Any, fn: Callable) -> Any: ...
    def connect_pick(self, fig_or_ax: Any, fn: Callable) -> Any: ...

    # ── Draw animated artists (blit support) ─────────────────────────────

    def draw_animated_artists(self, fig: Any) -> None:
        """Redraw all animated artists in fig (no-op for backends with
        per-object blitting like anyplotlib)."""

    # ── Navigation pointer widgets ────────────────────────────────────────

    def add_vline_widget(self, ax: Any, x: float, color: str = "red") -> Any: ...
    def update_vline(self, handle: Any, x: float) -> None: ...
    def add_rect_widget(
        self, ax: Any, x: float, y: float, w: float, h: float, color: str = "red"
    ) -> Any: ...
    def update_rect(
        self, handle: Any, x: float, y: float, w: float, h: float
    ) -> None: ...
    def remove_widget_patch(self, ax: Any, handle: Any) -> None: ...
    def set_patch_animated(self, handle: Any, value: bool) -> None: ...
    def set_patch_color(self, handle: Any, color: str) -> None: ...
    def set_patch_alpha(self, handle: Any, alpha: float) -> None: ...
    def add_artist(self, ax: Any, artist: Any) -> None: ...
    def create_rect_patch(self, pos, w: float, h: float, **kwargs) -> Any:
        """Create a rectangle patch (for resizer handles)."""

    def get_data_transform_inverse(self, ax: Any) -> Any:
        """Return an inverse-data transform (for pixel→data conversions)."""

    def transform_point(self, transform: Any, point) -> Any: ...

    # ── Marker collections ────────────────────────────────────────────────

    def add_collection(self, ax: Any, collection) -> Any: ...
    def collection_update(self, handle: Any, **kwargs) -> None: ...
    def collection_remove(self, ax: Any, handle: Any) -> None: ...
