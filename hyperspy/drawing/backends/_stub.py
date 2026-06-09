"""Stub template for new HyperSpy plotting backends.

Copy this file, rename ``StubBackend`` to ``MyBackend``, and implement the
methods marked REQUIRED.  Methods marked OPTIONAL have sensible defaults via
:class:`~hyperspy.drawing.backends._protocol.BlitMixin` and
:class:`~hyperspy.drawing.backends._protocol.PointerMixin`; override them only
when your backend supports the feature.

Registration::

    from hyperspy.drawing.backends import register_backend
    register_backend("mybackend", MyBackend)

Then users can do::

    import hyperspy.api as hs
    hs.preferences.Plot.backend = "mybackend"

Signature conventions
---------------------
* ``fig``      — whatever ``create_figure`` returned
* ``ax``       — whatever ``create_axes`` returned
* ``handle``   — whatever the corresponding ``plot_*`` / ``create_*`` returned
* ``cid``      — whatever the ``connect_*`` call returned (for disconnection)

All arguments are backend-native; HyperSpy never imports your library
directly.
"""

from __future__ import annotations

from hyperspy.drawing.backends._protocol import (
    BlitMixin,
    PointerMixin,
)


class StubBackend(BlitMixin, PointerMixin):
    """Minimal backend skeleton.  Every method marked REQUIRED must be implemented."""

    # =========================================================================
    # Figure lifecycle                                               [REQUIRED]
    # =========================================================================

    def create_figure(self, title=None, on_close=None, **kwargs):
        """Create and return a new figure.

        Parameters
        ----------
        title : str or None
            Window / figure title string.
        on_close : callable or None
            Zero-argument callback; call it when the user closes the figure.
        **kwargs
            Forwarded to the underlying figure constructor (e.g. figsize).

        Returns
        -------
        fig : any
            Opaque figure handle passed back to all other ``fig`` parameters.
        """
        raise NotImplementedError

    def close_figure(self, fig):
        """Programmatically destroy *fig*."""
        raise NotImplementedError

    def draw_idle(self, fig):
        """Schedule a non-blocking repaint of *fig*."""
        raise NotImplementedError

    def disconnect_event(self, fig_or_ax, cid):
        """Remove the event handler registered under *cid*.

        *cid* is whatever the corresponding ``connect_*`` call returned.
        """
        raise NotImplementedError

    # =========================================================================
    # Axes                                                           [REQUIRED]
    # =========================================================================

    def create_axes(self, fig, **kwargs):
        """Create and return a primary axes inside *fig*.

        Returns
        -------
        ax : any
            Opaque axes handle passed to all ``ax`` parameters below.
        """
        raise NotImplementedError

    def set_xlabel(self, ax, label: str) -> None:
        raise NotImplementedError

    def set_ylabel(self, ax, label: str) -> None:
        raise NotImplementedError

    def set_title(self, ax, title: str) -> None:
        raise NotImplementedError

    def set_xlim(self, ax, xmin: float, xmax: float) -> None:
        raise NotImplementedError

    def set_ylim(self, ax, ymin: float, ymax: float) -> None:
        raise NotImplementedError

    def get_ylim(self, ax) -> tuple:
        """Return (ymin, ymax) for *ax*."""
        raise NotImplementedError

    def get_xbound(self, ax) -> tuple:
        """Return (xmin, xmax) for *ax*."""
        raise NotImplementedError

    def set_axis_off(self, ax) -> None:
        raise NotImplementedError

    def set_aspect(self, ax, ratio: float) -> None:
        raise NotImplementedError

    def add_right_axis(self, ax, color: str = "black"):
        """Add a secondary y-axis on the right of *ax*; return its handle."""
        raise NotImplementedError

    def remove_right_axis(self, ax, right_ax) -> None:
        raise NotImplementedError

    # =========================================================================
    # 1-D line plotting                                              [REQUIRED]
    # =========================================================================

    def plot_line(self, ax, x, y, **props):
        """Draw a line from arrays *x*, *y*.

        Parameters
        ----------
        x, y : array-like
        **props : style keywords (color, linewidth, linestyle, label, …)

        Returns
        -------
        handle : any
            Passed to ``update_line`` / ``remove_line`` / ``set_line_props``.
        """
        raise NotImplementedError

    def update_line(self, handle, x, y) -> None:
        """Replace the data on an existing line handle."""
        raise NotImplementedError

    def remove_line(self, ax, handle) -> None:
        raise NotImplementedError

    def set_line_props(self, handle, **props) -> None:
        """Update visual properties of a line (color, linewidth, …)."""
        raise NotImplementedError

    def line_get_xdata(self, handle):
        """Return the x-data array of *handle*."""
        raise NotImplementedError

    def line_get_color(self, handle) -> str:
        """Return the color of *handle* as a string."""
        raise NotImplementedError

    # =========================================================================
    # Text annotations                                               [REQUIRED]
    # =========================================================================

    def add_text(
        self, ax, x: float, y: float, s: str, transform: str = "axes", **kwargs
    ):
        """Add a text annotation.

        Parameters
        ----------
        x, y : float
            Position in the coordinate system selected by *transform*.
        s : str
            Text content.
        transform : str
            One of ``'axes'``, ``'data'``, ``'xaxis'``, ``'yaxis'``,
            ``'display'``, ``'relative'``.

        Returns
        -------
        handle : any
            Passed to ``update_text`` / ``remove_text``.
        """
        raise NotImplementedError

    def update_text(self, handle, s: str) -> None:
        raise NotImplementedError

    def remove_text(self, ax, handle) -> None:
        raise NotImplementedError

    # =========================================================================
    # 2-D image plotting                                             [REQUIRED]
    # =========================================================================

    def plot_image(
        self,
        ax,
        data,
        extent=None,
        vmin=None,
        vmax=None,
        norm=None,
        cmap="gray",
        **kwargs,
    ):
        """Display a 2-D array as an image.

        Parameters
        ----------
        data : 2-D array-like
        extent : [xmin, xmax, ymin, ymax] or None
        vmin, vmax : float or None
        norm : normalisation object or None
        cmap : str

        Returns
        -------
        handle : any
            Passed to ``image_set_data`` / ``image_set_clim`` / etc.
        """
        raise NotImplementedError

    def plot_mesh(self, ax, x, y, data, **kwargs):
        """Display non-uniform 2-D data on a mesh (pcolormesh equivalent)."""
        raise NotImplementedError

    def image_set_data(self, handle, data) -> None:
        raise NotImplementedError

    def image_set_extent(self, handle, extent) -> None:
        raise NotImplementedError

    def image_set_clim(self, handle, vmin, vmax) -> None:
        raise NotImplementedError

    def image_set_norm(self, handle, norm) -> None:
        raise NotImplementedError

    def get_image_handle(self, ax):
        """Return the image/mesh handle on *ax*, or ``None`` if absent."""
        raise NotImplementedError

    # =========================================================================
    # Colorbar                                                       [REQUIRED]
    # =========================================================================

    def add_colorbar(self, fig, im_handle, ax):
        """Add a colorbar for *im_handle* to *fig*.

        Returns
        -------
        cb : any
            Passed to ``colorbar_set_label`` / ``colorbar_remove`` / etc.
        """
        raise NotImplementedError

    def colorbar_set_label(self, cb, label: str) -> None:
        raise NotImplementedError

    def colorbar_remove(self, cb) -> None:
        raise NotImplementedError

    def colorbar_redraw(self, cb, fig) -> None:
        raise NotImplementedError

    # =========================================================================
    # Event connections                                              [REQUIRED]
    # =========================================================================
    # Each ``connect_*`` returns a *cid* token passed to ``disconnect_event``.

    def connect_key_press(self, fig_or_ax, fn):
        raise NotImplementedError

    def connect_mouse_move(self, fig_or_ax, fn):
        raise NotImplementedError

    def connect_mouse_press(self, fig_or_ax, fn):
        raise NotImplementedError

    def connect_mouse_release(self, fig_or_ax, fn):
        raise NotImplementedError

    def connect_pick(self, fig_or_ax, fn):
        raise NotImplementedError

    # =========================================================================
    # Marker collections                                             [REQUIRED]
    # =========================================================================

    def add_collection(self, ax, collection):
        """Add a pre-built marker collection to *ax*; return a handle."""
        raise NotImplementedError

    def collection_update(self, handle, **kwargs) -> None:
        """Update collection properties (offsets, colors, …)."""
        raise NotImplementedError

    def collection_remove(self, ax, handle) -> None:
        raise NotImplementedError

    # =========================================================================
    # Layout helpers                                                 [REQUIRED]
    # =========================================================================

    def tight_layout(self, fig) -> None:
        """Apply tight layout to *fig* (best-effort; no-op is fine)."""

    def get_figure_from_ax(self, ax):
        """Return the parent figure of *ax*."""
        raise NotImplementedError

    # =========================================================================
    # Explorer                                                       [REQUIRED]
    # =========================================================================

    def get_explorer(self, signal_dim: int):
        """Return the HyperExplorer subclass for *signal_dim* (0, 1, or 2).

        Re-use the built-in MPL explorers if your backend is matplotlib-
        compatible::

            from hyperspy.drawing.he import (
                HyperSignal1D_Explorer, HyperImage_Explorer,
            )
            MAP = {0: HyperImage_Explorer, 1: HyperSignal1D_Explorer,
                   2: HyperImage_Explorer}
            return MAP.get(signal_dim, HyperImage_Explorer)

        Or subclass them and override ``_plot`` / ``_update_data`` to use your
        own drawing primitives.
        """
        raise NotImplementedError

    # =========================================================================
    # BlitMixin overrides                                             [OPTIONAL]
    # =========================================================================
    # Inherit BlitMixin no-ops (supports_blit → False, etc.).
    # Override if your backend supports blitting:
    #
    #   def supports_blit(self, fig) -> bool:
    #       return fig.canvas.supports_blit
    #
    #   def copy_background(self, fig):
    #       return fig.canvas.copy_from_bbox(fig.bbox)
    #
    #   def restore_background(self, fig, background) -> None:
    #       fig.canvas.restore_region(background)
    #
    #   def blit(self, fig) -> None:
    #       fig.canvas.blit(fig.bbox)
    #
    #   def connect_draw_event(self, fig, fn):
    #       return fig.canvas.mpl_connect("draw_event", fn)
    #
    #   def draw_animated_artists(self, fig) -> None:
    #       for ax in fig.axes:
    #           for a in ax.get_children():
    #               if a.get_animated():
    #                   ax.draw_artist(a)
    #
    #   def render_figure_from_ax(self, ax) -> None:   # calls draw_idle by default
    #       ...
    #
    #   def invalidate_blit_background(self, ax) -> None:
    #       ...
    #
    #   def supports_blit_from_ax(self, ax) -> bool:
    #       ...

    # =========================================================================
    # PointerMixin overrides                                          [OPTIONAL]
    # =========================================================================
    # Default: raise BackendCapabilityError.  Override to enable interactive
    # navigation widgets.
    #
    #   def create_line_pointer(self, ax, axis, pos, color="red"):
    #       """axis='x' → vertical line; axis='y' → horizontal line."""
    #       ...
    #
    #   def update_line_pointer(self, handle, pos) -> None:
    #       """Move the line to *pos* (x for axis='x', y for axis='y')."""
    #       ...
    #
    #   def create_rect_pointer(self, ax, x, y, w, h, color="red"):
    #       """Lower-left (x, y), size w×h."""
    #       ...
    #
    #   def update_rect_pointer(self, handle, x, y, w, h) -> None:
    #       ...
    #
    #   def remove_pointer(self, ax, handle) -> None:   # default is a no-op
    #       ...
    #
    #   def set_pointer_style(self, handle, *, color=None, alpha=None, animated=None):
    #       """Set any combination; ignore None arguments."""
    #       ...
    #
    #   # For MPL-native patch-based resizer handles:
    #   def add_artist(self, ax, artist) -> None: ...
    #   def create_rect_patch(self, pos, w, h, **kwargs): ...
    #   def get_data_transform_inverse(self, ax): ...
    #   def transform_point(self, transform, point): ...
    #
    #   # For interactive selection tools:
    #   def create_span_selector(self, ax, **kwargs): ...
    #   def create_polygon_selector(self, ax, **kwargs): ...
    #
    #   # For marker transform support:
    #   def get_ax_transform(self, ax, kind: str): ...

    # =========================================================================
    # Combined layout                                                 [OPTIONAL]
    # =========================================================================
    # Return (nav_fig, signal_fig) to display navigator + signal in one window.
    # Default returns None → two separate figures.
    #
    #   def create_combined_figure_panels(self, figsize=None):
    #       ...
    #
    #   def ensure_displayed(self, fig) -> None:
    #       """Force final render after plot() completes (for deferred backends)."""
    #       ...
    #
    #   def connect_close_event(self, fig, fn):
    #       """Connect fn() to figure close; return a cid."""
    #       ...
