"""anyplotlib plotting backend for hyperspy."""

from __future__ import annotations

import numpy as np

_NOT_YET = "anyplotlib does not yet support '{}'. See docs/hyperspy_parity.md."


class AnyplotlibBackend:
    """Maps hyperspy drawing primitives to anyplotlib API.

    Methods marked with _NOT_YET raise NotImplementedError until the
    corresponding anyplotlib feature is implemented.
    """

    # ── Figure lifecycle ──────────────────────────────────────────────────

    def create_figure(self, title=None, on_close=None, **kwargs):
        import anyplotlib as apl

        figsize = kwargs.pop("figsize", (640, 480))
        if isinstance(figsize, (list, tuple)) and max(figsize) < 50:
            # matplotlib uses inches; convert to pixels at 96 dpi
            figsize = (int(figsize[0] * 96), int(figsize[1] * 96))
        fig, ax = apl.subplots(1, 1, figsize=figsize)
        if on_close is not None:
            fig._hspy_on_close = on_close
        return fig

    def close_figure(self, fig):
        fig.close()
        on_close = getattr(fig, "_hspy_on_close", None)
        if on_close is not None:
            on_close()

    def draw_idle(self, fig):
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
        axes_list = fig.get_axes()
        return axes_list[0] if axes_list else None

    def set_xlabel(self, ax, label):
        raise NotImplementedError(_NOT_YET.format("set_xlabel"))

    def set_ylabel(self, ax, label):
        raise NotImplementedError(_NOT_YET.format("set_ylabel"))

    def set_title(self, ax, title):
        raise NotImplementedError(_NOT_YET.format("set_title"))

    def set_xlim(self, ax, xmin, xmax):
        raise NotImplementedError(_NOT_YET.format("set_xlim"))

    def set_ylim(self, ax, ymin, ymax):
        raise NotImplementedError(_NOT_YET.format("set_ylim"))

    def get_ylim(self, ax):
        raise NotImplementedError(_NOT_YET.format("get_ylim"))

    def get_xbound(self, ax):
        raise NotImplementedError(_NOT_YET.format("get_xbound"))

    def set_axis_off(self, ax):
        raise NotImplementedError(_NOT_YET.format("set_axis_off"))

    def set_aspect(self, ax, ratio):
        raise NotImplementedError(_NOT_YET.format("set_aspect"))

    def add_right_axis(self, ax, color="black"):
        raise NotImplementedError(_NOT_YET.format("add_right_axis (twinx)"))

    def remove_right_axis(self, ax, right_ax):
        raise NotImplementedError(_NOT_YET.format("remove_right_axis"))

    # ── 1-D line plotting ─────────────────────────────────────────────────

    def plot_line(self, ax, x, y, **props):
        """Draw a line; return the Plot1D handle."""
        color = props.get("color", "#4fc3f7")
        linestyle = props.get("linestyle", "solid")
        linewidth = props.get("linewidth", 1.5)
        alpha = props.get("alpha", 1.0)
        return ax.plot(
            np.asarray(y),
            axes=[np.asarray(x)],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
        )

    def update_line(self, handle, x, y):
        # handle is a Plot1D; update its primary data
        handle.set_data(np.asarray(y), x_axis=np.asarray(x))

    def remove_line(self, ax, handle):
        raise NotImplementedError(_NOT_YET.format("remove_line"))

    def set_line_props(self, handle, **props):
        raise NotImplementedError(_NOT_YET.format("set_line_props"))

    def line_get_xdata(self, handle):
        # handle is a Plot1D; return its x-axis
        return handle._state.get("x_axis")

    def line_get_color(self, handle):
        return handle._state.get("line_color", "#4fc3f7")

    # ── Text annotations ─────────────────────────────────────────────────

    def add_text(self, ax, x, y, s, transform="axes", **kwargs):
        raise NotImplementedError(_NOT_YET.format("add_text"))

    def update_text(self, handle, s):
        raise NotImplementedError(_NOT_YET.format("update_text"))

    def remove_text(self, ax, handle):
        raise NotImplementedError(_NOT_YET.format("remove_text"))

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
        return ax.imshow(
            np.asarray(data),
            axes=[x_axis, y_axis] if x_axis is not None else None,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

    def plot_mesh(self, ax, x, y, data, **kwargs):
        return ax.pcolormesh(
            np.asarray(data),
            x_edges=np.asarray(x),
            y_edges=np.asarray(y),
        )

    def image_set_data(self, handle, data):
        handle.set_data(np.asarray(data))

    def image_set_extent(self, handle, extent):
        raise NotImplementedError(_NOT_YET.format("image_set_extent"))

    def image_set_clim(self, handle, vmin, vmax):
        handle.set_clim(vmin, vmax)

    def image_set_norm(self, handle, norm):
        raise NotImplementedError(_NOT_YET.format("image_set_norm"))

    def get_image_handle(self, ax):
        return ax._plot if ax._plot is not None else None

    # ── Colorbar ─────────────────────────────────────────────────────────

    def add_colorbar(self, fig, im_handle, ax):
        return _AplColorbar(im_handle)

    def colorbar_set_label(self, cb, label):
        raise NotImplementedError(_NOT_YET.format("colorbar_set_label"))

    def colorbar_remove(self, cb):
        raise NotImplementedError(_NOT_YET.format("colorbar_remove"))

    def colorbar_redraw(self, cb, fig):
        pass

    # ── Events ───────────────────────────────────────────────────────────

    def connect_key_press(self, fig_or_ax, fn):
        plot = self._get_plot(fig_or_ax)
        if plot is not None:
            return plot.add_event_handler(fn, "key_down")
        return None

    def connect_mouse_move(self, fig_or_ax, fn):
        plot = self._get_plot(fig_or_ax)
        if plot is not None:
            return plot.add_event_handler(fn, "pointer_move")
        return None

    def connect_mouse_press(self, fig_or_ax, fn):
        plot = self._get_plot(fig_or_ax)
        if plot is not None:
            return plot.add_event_handler(fn, "pointer_down")
        return None

    def connect_mouse_release(self, fig_or_ax, fn):
        plot = self._get_plot(fig_or_ax)
        if plot is not None:
            return plot.add_event_handler(fn, "pointer_up")
        return None

    def connect_pick(self, fig_or_ax, fn):
        return self.connect_mouse_press(fig_or_ax, fn)

    def _get_plot(self, fig_or_ax):
        """Return the Plot1D/Plot2D attached to the given figure or axes."""
        if hasattr(fig_or_ax, "_plot"):
            # It's an Axes object
            return fig_or_ax._plot
        if hasattr(fig_or_ax, "get_axes"):
            # It's a Figure object — use the first axes
            axes_list = fig_or_ax.get_axes()
            if axes_list:
                return axes_list[0]._plot
        return None

    # ── Navigation pointer widgets ────────────────────────────────────────

    def add_vline_widget(self, ax, x, color="red"):
        plot = ax._plot
        if plot is None:
            raise RuntimeError("ax has no plot; call plot_line or plot_image first")
        return plot.add_vline_widget(x=float(x), color=color)

    def update_vline(self, handle, x):
        handle.x = float(x)

    def add_rect_widget(self, ax, x, y, w, h, color="red"):
        raise NotImplementedError(
            _NOT_YET.format("add_rect_widget (RectangleWidget on 1D plot)")
        )

    def update_rect(self, handle, x, y, w, h):
        handle.x = float(x)
        handle.y = float(y)
        handle.w = float(w)
        handle.h = float(h)

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
        raise NotImplementedError(_NOT_YET.format("set_patch_alpha"))

    def add_artist(self, ax, artist):
        pass

    def create_rect_patch(self, pos, w, h, **kwargs):
        raise NotImplementedError(
            _NOT_YET.format("create_rect_patch (resizer handles)")
        )

    def get_data_transform_inverse(self, ax):
        raise NotImplementedError(_NOT_YET.format("get_data_transform_inverse"))

    def transform_point(self, transform, point):
        raise NotImplementedError(_NOT_YET.format("transform_point"))

    # ── Marker collections ────────────────────────────────────────────────

    def add_collection(self, ax, collection):
        raise NotImplementedError(_NOT_YET.format("add_collection (markers)"))

    def collection_update(self, handle, **kwargs):
        raise NotImplementedError(_NOT_YET.format("collection_update (markers)"))

    def collection_remove(self, ax, handle):
        raise NotImplementedError(_NOT_YET.format("collection_remove (markers)"))


class _AplColorbar:
    """Sentinel returned by add_colorbar for the anyplotlib backend."""

    def __init__(self, im_handle):
        self._im = im_handle
