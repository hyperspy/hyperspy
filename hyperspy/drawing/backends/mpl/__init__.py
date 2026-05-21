from __future__ import annotations

import numpy as np


class MplBackend:
    """Wraps all matplotlib calls used by the hyperspy drawing layer."""

    # ── Figure lifecycle ──────────────────────────────────────────────────

    def create_figure(self, title=None, on_close=None, **kwargs):
        from hyperspy.drawing.utils import create_figure as _create_fig

        kwargs.setdefault("_on_figure_window_close", on_close)
        return _create_fig(
            window_title=f"Figure {title}" if title else None,
            **kwargs,
        )

    def close_figure(self, fig):
        import matplotlib.pyplot as plt

        plt.close(fig)

    def draw_idle(self, fig):
        if fig is not None:
            fig.canvas.draw_idle()

    def supports_blit(self, fig) -> bool:
        return fig is not None and fig.canvas.supports_blit

    def copy_background(self, fig):
        return fig.canvas.copy_from_bbox(fig.bbox)

    def restore_background(self, fig, background):
        fig.canvas.restore_region(background)

    def blit(self, fig):
        fig.canvas.blit(fig.bbox)

    def connect_draw_event(self, fig, fn):
        return fig.canvas.mpl_connect("draw_event", fn)

    def disconnect_event(self, fig_or_ax, cid):
        try:
            self._canvas(fig_or_ax).mpl_disconnect(cid)
        except Exception:
            pass

    # ── Axes setup ───────────────────────────────────────────────────────

    def create_axes(self, fig, **kwargs):
        ax = fig.add_subplot(111, **kwargs)
        animated = fig.canvas.supports_blit
        ax.yaxis.set_animated(animated)
        ax.xaxis.set_animated(animated)
        return ax

    def set_xlabel(self, ax, label):
        ax.set_xlabel(label)

    def set_ylabel(self, ax, label):
        ax.set_ylabel(label)

    def set_title(self, ax, title):
        ax.set_title(title)

    def set_xlim(self, ax, xmin, xmax):
        ax.set_xlim(xmin, xmax)

    def set_ylim(self, ax, ymin, ymax):
        ax.set_ylim(ymin, ymax)

    def get_ylim(self, ax):
        return ax.get_ylim()

    def get_xbound(self, ax):
        return ax.get_xbound()

    def set_axis_off(self, ax):
        ax.set_axis_off()

    def set_aspect(self, ax, ratio):
        ax.set_aspect(ratio)

    def add_right_axis(self, ax, color="black"):
        right_ax = ax.twinx()
        right_ax.tick_params(axis="y", labelcolor=color)
        right_ax.yaxis.set_animated(ax.figure.canvas.supports_blit)
        ax.set_zorder(right_ax.get_zorder() + 1)
        ax.patch.set_visible(False)
        return right_ax

    def remove_right_axis(self, ax, right_ax):
        try:
            right_ax.remove()
        except Exception:
            pass

    # ── 1-D line plotting ─────────────────────────────────────────────────

    def plot_line(self, ax, x, y, **props):
        animated = ax.figure.canvas.supports_blit
        norm = props.pop("norm", "linear")
        plot_fn = ax.semilogy if norm == "log" else ax.plot
        (line,) = plot_fn(x, y, animated=animated, **props)
        return line

    def update_line(self, handle, x, y):
        if not np.array_equiv(handle.get_xdata(), x):
            handle.set_data(x, y)
        else:
            handle.set_ydata(y)

    def remove_line(self, ax, handle):
        if handle in ax.lines:
            handle.remove()

    def set_line_props(self, handle, **props):
        import matplotlib.pyplot as plt

        plt.setp(handle, **props)

    def line_get_xdata(self, handle):
        return handle.get_xdata()

    def line_get_color(self, handle):
        return handle.get_color()

    # ── Text annotations ─────────────────────────────────────────────────

    def add_text(self, ax, x, y, s, transform="axes", **kwargs):
        animated = ax.figure.canvas.supports_blit
        t = ax.transAxes if transform == "axes" else ax.transData
        return ax.text(x, y, s=s, transform=t, animated=animated, **kwargs)

    def update_text(self, handle, s):
        handle.set_text(s)

    def remove_text(self, ax, handle):
        if handle in ax.texts:
            handle.remove()

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
        animated = ax.figure.canvas.supports_blit
        args = {"animated": animated, "cmap": cmap}
        if norm is None:
            args.update({"vmin": vmin, "vmax": vmax})
        else:
            args["norm"] = norm
        if extent is not None:
            args["extent"] = extent
        args.update(kwargs)
        ax.imshow(data, **args)
        return ax.images[-1]

    def plot_mesh(self, ax, x, y, data, **kwargs):
        h = ax.pcolormesh(x, y, data, **kwargs)
        ax.invert_yaxis()
        return h

    def image_set_data(self, handle, data):
        if hasattr(handle, "set_data"):
            handle.set_data(data)
        else:
            handle.set_array(data.ravel())

    def image_set_extent(self, handle, extent):
        handle.set_extent(extent)

    def image_set_clim(self, handle, vmin, vmax):
        handle.set_clim(vmin, vmax)

    def image_set_norm(self, handle, norm):
        handle.set_norm(norm)

    def get_image_handle(self, ax):
        if ax.images:
            return ax.images[0]
        if ax.collections:
            return ax.collections[0]
        return None

    # ── Colorbar ─────────────────────────────────────────────────────────

    def add_colorbar(self, fig, im_handle, ax):
        cb = fig.colorbar(im_handle, ax=ax)
        cb.ax.yaxis.set_animated(fig.canvas.supports_blit)
        return cb

    def colorbar_set_label(self, cb, label):
        cb.set_label(label, rotation=-90, va="bottom")

    def colorbar_remove(self, cb):
        cb.remove()

    def colorbar_redraw(self, cb, fig):
        import matplotlib.figure

        if isinstance(fig, matplotlib.figure.SubFigure):
            fig.canvas.draw_idle()
        else:
            fig.draw_without_rendering()
        cb.solids.set_animated(fig.canvas.supports_blit)

    # ── Event connections ─────────────────────────────────────────────────

    def _canvas(self, fig_or_ax):
        return getattr(fig_or_ax, "canvas", fig_or_ax.figure.canvas)

    def connect_key_press(self, fig_or_ax, fn):
        return self._canvas(fig_or_ax).mpl_connect("key_press_event", fn)

    def connect_mouse_move(self, fig_or_ax, fn):
        return self._canvas(fig_or_ax).mpl_connect("motion_notify_event", fn)

    def connect_mouse_press(self, fig_or_ax, fn):
        return self._canvas(fig_or_ax).mpl_connect("button_press_event", fn)

    def connect_mouse_release(self, fig_or_ax, fn):
        return self._canvas(fig_or_ax).mpl_connect("button_release_event", fn)

    def connect_pick(self, fig_or_ax, fn):
        return self._canvas(fig_or_ax).mpl_connect("pick_event", fn)

    # ── Draw animated artists ─────────────────────────────────────────────

    def draw_animated_artists(self, fig):
        for ax in fig.axes:
            for artist in sorted(ax.get_children(), key=lambda a: a.zorder):
                if artist.get_animated():
                    ax.draw_artist(artist)

    # ── Navigation pointer widgets ────────────────────────────────────────

    def add_vline_widget(self, ax, x, color="red"):
        from hyperspy.defaults_parser import preferences
        from hyperspy.drawing.utils import picker_kwargs

        return ax.axvline(
            x, color=color, **picker_kwargs(preferences.Plot.pick_tolerance)
        )

    def update_vline(self, handle, x):
        handle.set_xdata([x])

    def add_rect_widget(self, ax, x, y, w, h, color="red"):
        import matplotlib.patches as mpatches

        from hyperspy.defaults_parser import preferences
        from hyperspy.drawing.utils import picker_kwargs

        rect = mpatches.Rectangle(
            (x, y),
            w,
            h,
            fill=False,
            color=color,
            **picker_kwargs(preferences.Plot.pick_tolerance),
        )
        ax.add_patch(rect)
        return rect

    def update_rect(self, handle, x, y, w, h):
        handle.set_xy((x, y))
        handle.set_width(w)
        handle.set_height(h)

    def remove_widget_patch(self, ax, handle):
        try:
            handle.remove()
        except Exception:
            pass

    def set_patch_animated(self, handle, value):
        handle.set_animated(value)

    def set_patch_color(self, handle, color):
        handle.set_color(color)

    def set_patch_alpha(self, handle, alpha):
        handle.set_alpha(alpha)

    def add_artist(self, ax, artist):
        ax.add_artist(artist)

    def create_rect_patch(self, pos, w, h, **kwargs):
        import matplotlib.pyplot as plt

        return plt.Rectangle(pos, w, h, **kwargs)

    def get_data_transform_inverse(self, ax):
        return ax.transData.inverted()

    def transform_point(self, transform, point):
        return transform.transform(point)

    # ── Marker collections ────────────────────────────────────────────────

    def add_collection(self, ax, collection):
        ax.add_collection(collection)
        return collection

    def collection_update(self, handle, **kwargs):
        handle.set(**kwargs)

    def collection_remove(self, ax, handle):
        try:
            handle.remove()
        except Exception:
            pass
