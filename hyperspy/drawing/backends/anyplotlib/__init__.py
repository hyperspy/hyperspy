"""anyplotlib plotting backend for hyperspy."""

from __future__ import annotations

import numpy as np

from hyperspy.drawing.backends._protocol import BackendCapabilityError

_NOT_YET = "anyplotlib does not yet support '{}'. See docs/hyperspy_parity.md."


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


class AnyplotlibBackend:
    """Maps hyperspy drawing primitives to anyplotlib API.

    Methods marked with _NOT_YET raise NotImplementedError until the
    corresponding anyplotlib feature is implemented.
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
            # Proxy: call close callback but don't close the shared real figure.
            on_close = fig._hspy_on_close
            fig._hspy_on_close = None
            if on_close is not None:
                on_close()
            return
        # Clear before calling to prevent re-entrant close loop:
        # BlittedFigure stores on_close=self.close which calls close_figure again.
        on_close = getattr(fig, "_hspy_on_close", None)
        fig._hspy_on_close = None
        try:
            fig.close()
        except Exception:
            pass
        if on_close is not None:
            on_close()

    def create_combined_figure_panels(self, figsize=None):
        """Create a 2-panel anyplotlib Figure; return (nav_proxy, signal_proxy).

        Both proxies wrap the same underlying Figure widget so ``draw_idle``
        shows the figure only once both panels have rendered (no half-drawn
        flicker).  Pass the proxies as ``fig=`` in ``navigator_kwds`` and the
        main ``kwargs`` respectively.
        """
        import anyplotlib as apl

        if figsize is None:
            figsize = (1280, 640)
        else:
            figsize = tuple(float(v) for v in figsize)
            if max(figsize) < 50:
                figsize = (int(figsize[0] * 96 * 2), int(figsize[1] * 96))
            else:
                figsize = (int(figsize[0] * 2), int(figsize[1]))

        fig, axes = apl.subplots(1, 2, figsize=figsize)
        nav_ax, signal_ax = axes[0], axes[1]

        nav_proxy = _AplFigureProxy(fig, nav_ax)
        signal_proxy = _AplFigureProxy(fig, signal_ax)

        # hyperspy widgets reach the figure via ax.figure; use the proxy so
        # draw_idle(ax.figure) goes through the panel-countdown logic.
        nav_ax.figure = nav_proxy
        signal_ax.figure = signal_proxy

        # Count down to zero as each panel calls draw_idle for the first time.
        fig._hspy_panels_remaining = 2

        return nav_proxy, signal_proxy

    def ensure_displayed(self, fig):
        """Force-display fig, bypassing the panel countdown.

        Called from signal.py after plot() completes so that a figure is always
        shown even when the navigator was skipped (slider / None).
        """
        if fig is None:
            return
        real_fig = fig._real_fig if isinstance(fig, _AplFigureProxy) else fig
        if getattr(real_fig, "_hspy_displayed", False):
            return
        real_fig._hspy_panels_remaining = 0  # clear any pending countdown
        try:
            from IPython.display import display

            display(real_fig)
            real_fig._hspy_displayed = True
        except ImportError:
            pass

    def draw_idle(self, fig):
        if fig is None:
            return
        real_fig = fig._real_fig if isinstance(fig, _AplFigureProxy) else fig
        if getattr(real_fig, "_hspy_displayed", False):
            return

        # For combined layouts: each proxy decrements _hspy_panels_remaining
        # exactly once (on its first draw_idle call).  Display only when all
        # panels have rendered so the figure never appears half-populated.
        if isinstance(fig, _AplFigureProxy) and not getattr(fig, "_hspy_drawn", False):
            fig._hspy_drawn = True
            remaining = getattr(real_fig, "_hspy_panels_remaining", 1)
            real_fig._hspy_panels_remaining = remaining - 1

        if getattr(real_fig, "_hspy_panels_remaining", 0) > 0:
            return  # still waiting for other panels

        try:
            from IPython.display import display

            display(real_fig)
            real_fig._hspy_displayed = True
        except ImportError:
            pass

    def supports_blit(self, fig) -> bool:
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
        if cid is None:
            return
        plot = self._get_plot(fig_or_ax)
        if plot is not None and hasattr(plot, "callbacks"):
            plot.callbacks.disconnect(cid)

    def draw_animated_artists(self, fig):
        pass

    # ── Axes setup ───────────────────────────────────────────────────────

    def create_axes(self, fig, **kwargs):
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

    def get_ylim(self, ax):
        if ax._plot is not None:
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

    def plot_line(self, ax, x, y, **props):
        """Draw a line; return the Plot1D handle."""
        color = props.get("color", "#4fc3f7")
        linestyle = props.get("linestyle", "solid")
        linewidth = props.get("linewidth", 1.5)
        alpha = props.get("alpha", 1.0)
        axes_arg = [np.asarray(x)] if x is not None else None
        plot = ax.plot(
            np.asarray(y),
            axes=axes_arg,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
        )
        self._apply_pending_labels(ax, plot)
        return plot

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
        handle.set_data(np.asarray(y), x_axis=np.asarray(x))

    def remove_line(self, ax, handle):
        pass  # primary Plot1D cannot be individually removed from its panel

    def set_line_props(self, handle, **props):
        pass  # anyplotlib does not expose per-property setters on Plot1D yet

    def line_get_xdata(self, handle):
        return handle.x

    def line_get_color(self, handle):
        return handle.color

    # ── Text annotations ─────────────────────────────────────────────────

    def add_text(self, ax, x, y, s, transform="axes", **kwargs):
        return None  # text annotations not yet supported; callers check for None

    def update_text(self, handle, s):
        pass  # no-op when handle is None (add_text returned None)

    def remove_text(self, ax, handle):
        pass  # no-op when handle is None

    # ── 2-D image plotting ────────────────────────────────────────────────

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
        x_axis = y_axis = None
        if extent is not None:
            x0, x1, y1, y0 = extent
            x_axis = np.linspace(x0, x1, data.shape[1])
            y_axis = np.linspace(y0, y1, data.shape[0])
        plot = ax.imshow(
            np.asarray(data),
            axes=[x_axis, y_axis] if x_axis is not None else None,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        self._apply_pending_labels(ax, plot)
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
        x0, x1, y1, y0 = extent
        h = handle._state["image_height"]
        w = handle._state["image_width"]
        x_axis = np.linspace(x0, x1, w)
        y_axis = np.linspace(y0, y1, h)
        handle.set_extent(x_axis, y_axis)

    def image_set_clim(self, handle, vmin, vmax):
        handle.set_clim(vmin, vmax)

    def image_set_norm(self, handle, norm):
        pass  # anyplotlib uses vmin/vmax; advanced norms are not yet supported

    def get_image_handle(self, ax):
        return ax._plot if ax._plot is not None else None

    # ── Colorbar ─────────────────────────────────────────────────────────

    def add_colorbar(self, fig, im_handle, ax):
        im_handle.set_colorbar_visible(True)
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
        return self.connect_mouse_press(fig_or_ax, fn)

    def _get_plot(self, fig_or_ax):
        """Return the Plot1D/Plot2D attached to the given figure or axes."""
        if hasattr(fig_or_ax, "_plot"):
            # It's an Axes object
            return fig_or_ax._plot
        if hasattr(fig_or_ax, "_hspy_ax"):
            # It's a Figure or _AplFigureProxy — use the stored hyperspy axes
            return fig_or_ax._hspy_ax._plot
        return None

    # ── Navigation pointer widgets ────────────────────────────────────────

    def add_vline_widget(self, ax, x, color="red"):
        plot = ax._plot
        if plot is None:
            raise RuntimeError("ax has no plot; call plot_line or plot_image first")
        return plot.add_vline_widget(x=float(x), color=color)

    def update_vline(self, handle, x):
        handle.x = float(x)

    def add_hline_widget(self, ax, y, color="red"):
        plot = getattr(ax, "_plot", None)
        if plot is None or not hasattr(plot, "add_widget"):
            raise BackendCapabilityError(_NOT_YET.format("add_hline_widget"))
        return plot.add_widget("crosshair", cx=0.0, cy=float(y), color=color)

    def update_hline(self, handle, y):
        handle.cy = float(y)

    def connect_widget_drag(self, handle, on_drag):
        try:
            from anyplotlib.widgets._widgets1d import VLineWidget
            from anyplotlib.widgets._widgets2d import CrosshairWidget
        except ImportError:
            return

        if isinstance(handle, VLineWidget):

            def _cb(event):
                on_drag(handle.x)

        elif isinstance(handle, CrosshairWidget):

            def _cb(event):
                on_drag(handle.cx, handle.cy)

        else:
            return
        handle.add_event_handler(self._wrap(_cb), "pointer_move")

    def add_rect_widget(self, ax, x, y, w, h, color="red"):
        plot = getattr(ax, "_plot", None)
        if plot is None or not hasattr(plot, "add_widget"):
            raise BackendCapabilityError(_NOT_YET.format("add_rect_widget"))
        # x, y are the lower-left corner; convert to center for the crosshair
        cx = float(x) + float(w) / 2.0
        cy = float(y) + float(h) / 2.0
        return plot.add_widget("crosshair", cx=cx, cy=cy, color=color)

    def update_rect(self, handle, x, y, w, h):
        # x, y are lower-left; convert to center so cx/cy equal the nav axis value
        handle.cx = float(x) + float(w) / 2.0
        handle.cy = float(y) + float(h) / 2.0

    def remove_widget_patch(self, ax, handle):
        try:
            handle.remove()
        except Exception:
            pass

    def set_patch_animated(self, handle, value):
        pass

    def set_patch_color(self, handle, color):
        handle.color = color

    def set_patch_alpha(self, handle, alpha):
        raise BackendCapabilityError(_NOT_YET.format("set_patch_alpha"))

    def add_artist(self, ax, artist):
        pass

    def create_rect_patch(self, pos, w, h, **kwargs):
        raise BackendCapabilityError(
            _NOT_YET.format("create_rect_patch (resizer handles)")
        )

    def get_data_transform_inverse(self, ax):
        raise BackendCapabilityError(_NOT_YET.format("get_data_transform_inverse"))

    def transform_point(self, transform, point):
        raise BackendCapabilityError(_NOT_YET.format("transform_point"))

    # ── Marker collections ────────────────────────────────────────────────

    def add_collection(self, ax, collection):
        raise BackendCapabilityError(_NOT_YET.format("add_collection (markers)"))

    def collection_update(self, handle, **kwargs):
        raise BackendCapabilityError(_NOT_YET.format("collection_update (markers)"))

    def collection_remove(self, ax, handle):
        raise BackendCapabilityError(_NOT_YET.format("collection_remove (markers)"))

    # ── Combined layout / lifecycle hooks ─────────────────────────────────

    def render_figure_from_ax(self, ax):
        self.draw_idle(getattr(ax, "figure", None))

    def invalidate_blit_background(self, ax):
        pass  # anyplotlib repaints natively; no blit-background cache to invalidate

    def supports_blit_from_ax(self, ax):
        return False

    def create_span_selector(self, ax, **kwargs):
        raise BackendCapabilityError(_NOT_YET.format("create_span_selector"))

    def create_polygon_selector(self, ax, **kwargs):
        raise BackendCapabilityError(_NOT_YET.format("create_polygon_selector"))

    def get_ax_transform(self, ax, kind):
        raise BackendCapabilityError(_NOT_YET.format(f"get_ax_transform({kind!r})"))

    # connect_widget_drag implemented above; declared here for protocol completeness

    def simulate_pick(self, ax, patch):
        pass  # anyplotlib handles selection natively; no MPL pick simulation needed

    def connect_close_event(self, fig, fn):
        # anyplotlib close handling is done via on_close= kwarg at figure
        # creation time; there is no post-hoc connect mechanism yet.
        return None

    def get_explorer(self, signal_dim):
        # Phase 2 will add Apl_Hyper*Explorer subclasses; for now return the
        # generic base classes which work with any backend that implements the
        # required primitives.
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


class _AplColorbar:
    """Sentinel returned by add_colorbar for the anyplotlib backend."""

    def __init__(self, im_handle):
        self._im = im_handle
