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

    def create_axes(self, fig, animate_axis=False, **kwargs):
        ax = fig.add_subplot(111, **kwargs)
        if animate_axis:
            # signal1d animates the axis so the y-scale can update during blit.
            # Image plots leave it un-animated (matches the base behaviour) so
            # the ticks/spine render in the static background rather than the
            # animated pass — animating them shifts their sub-pixel rendering.
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

    def get_xlim(self, ax):
        return ax.get_xlim()

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

    @staticmethod
    def _to_mpl_norm(norm):
        """Convert a HyperNorm to the equivalent matplotlib Normalize."""
        from hyperspy.drawing.norm import (
            HyperNorm,
            LinearNorm,
            LogNorm,
            PowerNorm,
            SymLogNorm,
        )

        if norm is None or not isinstance(norm, HyperNorm):
            return norm
        from matplotlib.colors import (
            LogNorm as MplLogNorm,
        )
        from matplotlib.colors import (
            Normalize as MplNormalize,
        )
        from matplotlib.colors import (
            PowerNorm as MplPowerNorm,
        )
        from matplotlib.colors import (
            SymLogNorm as MplSymLogNorm,
        )

        if isinstance(norm, LinearNorm):
            return MplNormalize(vmin=norm.vmin, vmax=norm.vmax, clip=norm.clip)
        elif isinstance(norm, LogNorm):
            return MplLogNorm(vmin=norm.vmin, vmax=norm.vmax, clip=norm.clip)
        elif isinstance(norm, PowerNorm):
            return MplPowerNorm(
                gamma=norm.gamma, vmin=norm.vmin, vmax=norm.vmax, clip=norm.clip
            )
        elif isinstance(norm, SymLogNorm):
            return MplSymLogNorm(
                linthresh=norm.linthresh,
                linscale=norm.linscale,
                vmin=norm.vmin,
                vmax=norm.vmax,
                clip=norm.clip,
                base=norm.base,
            )
        return None

    def plot_line(self, ax, x, y, **props):
        animated = props.pop("animated", ax.figure.canvas.supports_blit)
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

    def line_get_linewidth(self, handle):
        return handle.get_linewidth()

    # ── Text annotations ─────────────────────────────────────────────────

    def add_text(self, ax, x, y, s, transform="axes", **kwargs):
        animated = kwargs.pop("animated", ax.figure.canvas.supports_blit)
        t = self.get_ax_transform(ax, transform)
        return ax.text(x, y, s=s, transform=t, animated=animated, **kwargs)

    def update_text(self, handle, s):
        if handle is not None:
            handle.set_text(s)

    def remove_text(self, ax, handle):
        if handle is not None and handle in ax.texts:
            handle.remove()

    def text_set_color(self, handle, color):
        if handle is not None:
            import matplotlib.pyplot as plt

            plt.setp(handle, color=color)

    def text_get_color(self, handle):
        if handle is not None:
            return handle.get_color()
        return "black"

    # ── Generic artist ────────────────────────────────────────────────────

    def artist_set_animated(self, handle, animated):
        if handle is not None:
            handle.set_animated(animated)

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
        mpl_norm = self._to_mpl_norm(norm)
        # Interpolation is intentionally left to the caller/matplotlib default:
        # the main ImagePlot path relies on matplotlib's own "auto" choice,
        # while decomposition/loadings component plots pass
        # ``interpolation="nearest"`` explicitly via kwargs.
        args = {"animated": animated, "cmap": cmap}
        if mpl_norm is None:
            args.update({"vmin": vmin, "vmax": vmax})
        else:
            args["norm"] = mpl_norm
        if extent is not None:
            args["extent"] = extent
        args.update(kwargs)
        ax.imshow(data, **args)
        return ax.images[-1]

    def plot_mesh(self, ax, x, y, data, **kwargs):
        # Match plot_image: animate the mesh when blitting is supported so it
        # is painted via draw_artist (the blit pass) rather than the normal
        # composited Axes.draw(), which mis-renders the edge row of this mesh
        # (non-uniform-axis navigators/images) as solid black.
        kwargs.setdefault("animated", ax.figure.canvas.supports_blit)
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
        # Not handle.set_clim(vmin, vmax): that sets vmin then vmax as two
        # separate attribute writes, each of which synchronously notifies any
        # attached colorbar. On older matplotlib, the colorbar's callback
        # recomputes norm limits via Colorbar._process_values() as soon as
        # vmin is written (while vmax is still the stale/None value from the
        # just-reset norm), scrambling vmin before vmax is applied. Setting
        # vmax first, then vmin, in one tuple assignment (as the pre-refactor
        # code did) avoids that intermediate inconsistent state.
        handle.norm.vmax, handle.norm.vmin = vmax, vmin

    def image_set_norm(self, handle, norm):
        handle.set_norm(self._to_mpl_norm(norm))

    def get_image_handle(self, ax):
        if ax.images:
            return ax.images[0]
        if ax.collections:
            return ax.collections[0]
        return None

    # ── Colorbar ─────────────────────────────────────────────────────────

    def add_colorbar(self, fig, im_handle, ax, divider=False, size="5%", pad=0.05):
        if divider:
            # Size the colorbar to the image — used by the multi-panel
            # factors/loadings/cluster grids. ``fig.colorbar(ax=ax)`` would
            # instead steal space from the axes and make a full-height bar that
            # squishes each panel.
            from mpl_toolkits.axes_grid1 import make_axes_locatable

            cax = make_axes_locatable(ax).append_axes("right", size=size, pad=pad)
            cb = fig.colorbar(im_handle, cax=cax)
        else:
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
                if artist.get_animated() and artist.axes is not None:
                    ax.draw_artist(artist)

    # ── Navigation pointer widgets ────────────────────────────────────────

    def create_line_pointer(self, ax, axis, pos, color="red"):
        from hyperspy.defaults_parser import preferences
        from hyperspy.drawing.utils import picker_kwargs

        kw = picker_kwargs(preferences.Plot.pick_tolerance)
        if axis == "x":
            return ax.axvline(pos, color=color, **kw)
        else:
            return ax.axhline(pos, color=color, **kw)

    def update_line_pointer(self, handle, axis, pos):
        if axis == "x":
            handle.set_xdata([pos])
        else:
            handle.set_ydata([pos])

    def create_rect_pointer(self, ax, x, y, w, h, color="red", linewidth=2):
        import matplotlib.patches as mpatches

        rect = mpatches.Rectangle(
            (x, y),
            w,
            h,
            fill=False,
            lw=linewidth,
            color=color,
            picker=True,
        )
        ax.add_patch(rect)
        return rect

    def update_rect_pointer(self, handle, x, y, w, h):
        handle.set_xy((x, y))
        handle.set_width(w)
        handle.set_height(h)

    def remove_pointer(self, ax, handle):
        try:
            handle.remove()
        except Exception:
            pass

    def set_pointer_style(self, handle, *, color=None, alpha=None, animated=None):
        if color is not None:
            handle.set_color(color)
        if alpha is not None:
            handle.set_alpha(alpha)
        if animated is not None:
            handle.set_animated(animated)

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

    # ── Combined layout / lifecycle hooks ─────────────────────────────────

    def create_combined_figure_panels(self, figsize=None):
        from hyperspy.defaults_parser import preferences

        if not preferences.Plot.use_subfigure:
            return None
        import matplotlib.pyplot as plt

        figsize = figsize or (15, 7)
        fig = plt.figure(figsize=figsize, layout="constrained")
        subfigs = fig.subfigures(1, 2)
        return subfigs[0], subfigs[1]

    def ensure_displayed(self, fig):
        pass

    def connect_close_event(self, fig, fn):
        if fig is None:
            return None
        canvas = getattr(fig, "canvas", None)
        if canvas is None:
            return None

        def _wrapper(evt):
            fn()

        return canvas.mpl_connect("close_event", _wrapper)

    def simulate_pick(self, ax, patch):
        try:
            from matplotlib.backend_bases import MouseEvent, PickEvent

            figure = ax.figure
            x, y = patch.get_transform().transform_point((0, 0))
            mouseevent = MouseEvent("pick_event", figure.canvas, x, y)
            if mouseevent.button:
                try:
                    event = PickEvent("pick_event", figure, mouseevent, patch)
                    figure.canvas.callbacks.process("pick_event", event)
                except Exception:
                    figure.canvas.pick_event(mouseevent, patch)
        except (ImportError, AttributeError):
            pass

    def render_figure_from_ax(self, ax):
        hspy_fig = getattr(ax, "hspy_fig", None)
        if hspy_fig is not None:
            hspy_fig.render_figure()
        elif getattr(ax, "figure", None) is not None:
            ax.figure.canvas.draw_idle()

    def invalidate_blit_background(self, ax):
        hspy_fig = getattr(ax, "hspy_fig", None)
        if hspy_fig is not None:
            hspy_fig._background = None

    def supports_blit_from_ax(self, ax):
        return getattr(ax, "hspy_fig", None) is not None and self.supports_blit(
            getattr(ax, "figure", None)
        )

    def create_span_selector(self, ax, **kwargs):
        from matplotlib.widgets import SpanSelector

        return SpanSelector(ax, **kwargs)

    def create_polygon_selector(self, ax, **kwargs):
        from matplotlib.widgets import PolygonSelector

        return PolygonSelector(ax, **kwargs)

    def connect_widget_drag(self, handle, on_drag):
        pass  # MPL widgets fire drag via _onmousemove in the widget base class

    def get_ax_transform(self, ax, kind):
        transforms = {
            "data": ax.transData,
            "axes": ax.transAxes,
            "display": None,  # resolved by caller with IdentityTransform
            "yaxis": ax.get_yaxis_transform(),
            "xaxis": ax.get_xaxis_transform(),
            "relative": ax.transData,
        }
        if kind not in transforms:
            raise ValueError(f"Unknown transform kind: {kind!r}")
        if kind == "display":
            from matplotlib.transforms import IdentityTransform

            return IdentityTransform()
        return transforms[kind]

    def _space_transform(self, ax, space):
        """Return the MPL transform corresponding to a CoordSpace string."""
        from matplotlib.transforms import IdentityTransform

        mapping = {
            "data": ax.transData,
            "relative": ax.transData,
            "axes": ax.transAxes,
            "xaxis": ax.get_xaxis_transform(),
            "yaxis": ax.get_yaxis_transform(),
            "display": IdentityTransform(),
        }
        if space not in mapping:
            raise ValueError(f"Unknown coordinate space: {space!r}")
        return mapping[space]

    def convert_coords(self, ax, points, from_space, to_space):
        import numpy as np

        from_trans = self._space_transform(ax, from_space)
        to_trans = self._space_transform(ax, to_space)
        composite = from_trans + to_trans.inverted()
        return composite.transform(np.atleast_2d(points))

    # ── Native marker collections ─────────────────────────────────────────

    _MARKER_COLLECTION_MAP = None

    def _marker_collection_map(self):
        if self._MARKER_COLLECTION_MAP is None:
            from matplotlib.collections import LineCollection

            from hyperspy.external.matplotlib.collections import (
                CircleCollection,
                EllipseCollection,
                RectangleCollection,
                SquareCollection,
                TextCollection,
            )

            MplBackend._MARKER_COLLECTION_MAP = {
                "points": CircleCollection,
                "circles": CircleCollection,
                "squares": SquareCollection,
                "lines": LineCollection,
                "hlines": LineCollection,
                "vlines": LineCollection,
                "texts": TextCollection,
                "rectangles": RectangleCollection,
                "ellipses": EllipseCollection,
            }
        return self._MARKER_COLLECTION_MAP

    def create_markers(self, ax, marker_type, **kwargs):
        from hyperspy.drawing.backends._protocol import BackendCapabilityError

        offset_space = kwargs.pop("offset_space", "data")
        transform_space = kwargs.pop("transform_space", "display")

        cmap = self._marker_collection_map()
        collection_cls = cmap.get(marker_type)
        if collection_cls is None:
            raise BackendCapabilityError(
                f"Marker type {marker_type!r} not supported by the MPL backend"
            )
        offset_transform = self._space_transform(ax, offset_space)
        transform = self._space_transform(ax, transform_space)
        collection = collection_cls(offset_transform=offset_transform, **kwargs)
        collection.set_transform(transform)
        ax.add_collection(collection)
        # BlittedFigure draws markers as animated artists; without this the
        # collection is never rendered (it is neither baked into the blit
        # background nor drawn in the animated pass). Mirrors the fallback path.
        self.artist_set_animated(
            collection, self.supports_blit(self.get_figure_from_ax(ax))
        )
        return collection

    def update_markers(self, handle, **kwargs):
        handle.set(**kwargs)

    def remove_markers(self, ax, handle):
        try:
            handle.remove()
        except Exception:
            pass

    # ── Step plot ─────────────────────────────────────────────────────────

    def plot_step(self, ax, x, y, **props):
        lines = ax.step(x, y, **props)
        return lines[0]

    # ── Patch creation ────────────────────────────────────────────────────

    def create_line2d_patch(self, x, y, **kwargs):
        import matplotlib.pyplot as plt

        return plt.Line2D(x, y, **kwargs)

    def create_circle_patch(self, xy, radius, **kwargs):
        import matplotlib.pyplot as plt

        return plt.Circle(xy, radius=radius, **kwargs)

    # ── Axes control ─────────────────────────────────────────────────────

    def set_autoscale(self, ax, enable):
        ax.autoscale(enable)

    def set_xticklabels(self, ax, labels):
        ax.set_xticklabels(labels)

    def set_yticklabels(self, ax, labels):
        ax.set_yticklabels(labels)

    def set_xticks(self, ax, ticks):
        ax.set_xticks(ticks)

    def set_yticks(self, ax, ticks):
        ax.set_yticks(ticks)

    # ── Layout helpers ────────────────────────────────────────────────────

    def tight_layout(self, fig):
        try:
            fig.tight_layout()
        except Exception:
            pass

    def get_figure_from_ax(self, ax):
        return ax.figure

    # ── Figure manager factories ──────────────────────────────────────────

    def create_signal1d_figure(self, title="", on_close=None, fig=None, **kwargs):
        from hyperspy.drawing.signal1d import Signal1DFigure

        return Signal1DFigure(
            title=title, _on_figure_window_close=on_close, fig=fig, **kwargs
        )

    def create_image_figure(self, title="", **kwargs):
        from hyperspy.drawing.image import ImagePlot

        return ImagePlot(title=title)

    # ── Scale bar ─────────────────────────────────────────────────────────

    def create_scalebar(self, ax, units, **kwargs):
        from hyperspy.drawing._widgets.scalebar import ScaleBar

        return ScaleBar(ax=ax, units=units, **kwargs)

    def remove_scalebar(self, ax, handle):
        handle.remove()

    # ── Image helpers ─────────────────────────────────────────────────────

    def get_image_cmap_name(self, handle):
        return handle.get_cmap().name

    def get_explorer(self, signal_dim):
        if signal_dim == 0:
            from hyperspy.drawing.backends.mpl.mpl_he import MPL_HyperExplorer

            return MPL_HyperExplorer
        elif signal_dim == 1:
            from hyperspy.drawing.backends.mpl.mpl_hse import (
                MPL_HyperSignal1D_Explorer,
            )

            return MPL_HyperSignal1D_Explorer
        elif signal_dim == 2:
            from hyperspy.drawing.backends.mpl.mpl_hie import MPL_HyperImage_Explorer

            return MPL_HyperImage_Explorer
        raise ValueError(
            f"Plotting is not supported for signal_dim={signal_dim}. "
            "Try s.transpose(signal_axes=1).plot() for 1D or "
            "s.transpose(signal_axes=(1,2)).plot() for 2D."
        )
