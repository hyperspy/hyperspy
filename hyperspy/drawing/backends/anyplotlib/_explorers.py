"""anyplotlib HyperExplorer subclasses."""

from __future__ import annotations

import numpy as np
from traits.api import Undefined

from hyperspy.drawing import image, signal1d, widgets
from hyperspy.drawing.backends import get_backend
from hyperspy.drawing.backends._protocol import BackendCapabilityError
from hyperspy.drawing.he import HyperExplorer
from hyperspy.drawing.hie import HyperImage_Explorer
from hyperspy.drawing.hse import HyperSignal1D_Explorer


class Apl_HyperExplorer(HyperExplorer):
    """Base anyplotlib explorer — implements navigator creation and pointer
    assignment shared by both 1-D and 2-D anyplotlib subclasses."""

    def assign_pointer(self):
        if self.navigator_data_function is None:
            nav_dim = 0
        elif self.navigator_data_function == "slider":
            nav_dim = 0
        else:
            nav_dim = len(self.navigator_data_function().shape)

        if nav_dim == 2:
            if self.axes_manager.navigation_dimension > 1:
                Pointer = widgets.SquareWidget
            else:
                Pointer = widgets.HorizontalLineWidget
        elif nav_dim == 1:
            Pointer = widgets.VerticalLineWidget
        else:
            Pointer = None
        self._pointer_nav_dim = nav_dim
        return Pointer

    def _connect_pointer(self, pointer, figure):
        pointer.set_mpl_ax(figure.ax)

    def _connect_key_nav(self, figure):
        if figure.figure is not None and self.axes_manager.navigation_axes:
            get_backend().connect_key_press(
                figure.figure, self.axes_manager.key_navigator
            )

    def _create_1d_nav_figure(self, title, **kwargs):
        fig = kwargs.pop("fig", None)
        sf = signal1d.Signal1DFigure(
            title=title,
            _on_figure_window_close=self.close,
            fig=fig,
        )
        axis = self.axes_manager.navigation_axes[0]
        sf.xlabel = "%s" % str(axis)
        if axis.units is not Undefined:
            sf.xlabel += " (%s)" % axis.units
        sf.ylabel = r"$\Sigma\mathrm{data\,over\,all\,other\,axes}$"
        sf.axis = axis
        sf.axes_manager = self.axes_manager

        sl = signal1d.Signal1DLine()
        sl.data_function = self.navigator_data_function
        for key in list(kwargs.keys()):
            if hasattr(sl, key):
                setattr(sl, key, kwargs.pop(key))
        sl.set_line_properties(
            color="blue",
            type="step" if axis.is_uniform else "line",
        )
        sf.add_line(sl)
        sf.plot()
        return sf

    def _create_2d_nav_figure(self, title, **kwargs):
        from hyperspy.defaults_parser import preferences

        imf = image.ImagePlot(title=title)
        imf.data_function = self.navigator_data_function

        for key, value in list(kwargs.items()):
            if hasattr(imf, key):
                setattr(imf, key, kwargs.pop(key))

        if self.axes_manager.navigation_dimension == 1:
            imf.yaxis = self.axes_manager.navigation_axes[0]
            imf.xaxis = self.axes_manager.signal_axes[0]
        elif self.axes_manager.navigation_dimension >= 2:
            imf.yaxis = self.axes_manager.navigation_axes[1]
            imf.xaxis = self.axes_manager.navigation_axes[0]

        if "cmap" not in kwargs or kwargs.get("cmap") is None:
            kwargs["cmap"] = preferences.Plot.cmap_navigator
        imf.plot(_on_figure_window_close=self.close, **kwargs)
        return imf


class Apl_HyperSignal1D_Explorer(HyperSignal1D_Explorer, Apl_HyperExplorer):
    """1-D signal explorer for the anyplotlib backend."""

    def _make_signal_figure(self, **kwargs):
        fig = kwargs.pop("fig", None)
        sf = signal1d.Signal1DFigure(
            title=self.signal_title + " Signal",
            _on_figure_window_close=self.close,
            fig=fig,
        )
        sf.axis = self.axis
        if sf.ax is None:
            sf.create_axis()
        sf.axes_manager = self.axes_manager
        sf.xlabel = self.xlabel
        sf.ylabel = self.ylabel

        sl = signal1d.Signal1DLine()
        is_complex = np.iscomplexobj(self.signal_data_function())
        sl.data_function = self.signal_data_function
        kwargs["data_function_kwargs"] = self.signal_data_function_kwargs
        sl.plot_indices = True
        sl.set_line_properties(
            color=self.pointer.color if self.pointer is not None else "red",
            type="step",
        )
        sf.add_line(sl)
        if is_complex:
            sl2 = signal1d.Signal1DLine()
            sl2.data_function = self.signal_data_function
            sl2.plot_coordinates = True
            sl2._plot_imag = True
            sl2.set_line_properties(color="blue", type="step")
            sf.add_line(sl2)
        sf.plot(**kwargs)
        return sf

    def _connect_key_handler(self, figure, fn):
        if figure.figure is not None:
            get_backend().connect_key_press(figure.figure, fn)

    def _add_right_line(self, **kwargs):
        raise BackendCapabilityError(
            "The anyplotlib backend does not support twin-y axes. "
            "The right-pointer feature is unavailable."
        )

    def _redraw_signal_figure(self):
        if self.signal_plot is not None and self.signal_plot.figure is not None:
            get_backend().draw_idle(self.signal_plot.figure)


class Apl_HyperImage_Explorer(HyperImage_Explorer, Apl_HyperExplorer):
    """2-D image explorer for the anyplotlib backend."""

    def _make_image_figure(self, **kwargs):
        from hyperspy.defaults_parser import preferences

        imf = image.ImagePlot()
        imf.axes_manager = self.axes_manager
        imf.data_function = self.signal_data_function
        imf.title = self.signal_title + " Signal"
        imf.xaxis, imf.yaxis = self.axes_manager.signal_axes

        for key, value in list(kwargs.items()):
            if hasattr(imf, key):
                setattr(imf, key, kwargs.pop(key))

        imf.quantity_label = self.quantity_label
        kwargs["data_function_kwargs"] = self.signal_data_function_kwargs
        if "cmap" not in kwargs or kwargs.get("cmap") is None:
            kwargs["cmap"] = preferences.Plot.cmap_signal
        imf.plot(_on_figure_window_close=self.close, **kwargs)
        return imf
