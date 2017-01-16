# Copyright 2007-2016 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.

from functools import partial

from traits.api import Undefined

from hyperspy.drawing import widgets, signal1d, image
from hyperspy.gui.axes import navigation_sliders


class MPL_HyperExplorer(object):

    """

    """

    def __init__(self):
        self.signal_data_function = None
        self.navigator_data_function = None
        self.axes_manager = None
        self.signal_title = ''
        self.navigator_title = ''
        self.quantity_label = ''
        self.signal_plot = None
        self.navigator_plot = None
        self.axis = None
        self.pointer = None
        self._pointer_nav_dim = None

    def plot_signal(self):
        # This method should be implemented by the subclasses.
        # Doing nothing is good enough for signal_dimension==0 though.
        return

    def plot_navigator(self,
                       colorbar=True,
                       scalebar=True,
                       scalebar_color="white",
                       axes_ticks=None,
                       saturated_pixels=0,
                       vmin=None,
                       vmax=None,
                       no_nans=False,
                       centre_colormap="auto",
                       title=None,
                       **kwds):
        if self.axes_manager.navigation_dimension == 0:
            return
        if self.navigator_data_function is None:
            return
        if self.navigator_data_function is "slider":
            navigation_sliders(
                self.axes_manager.navigation_axes,
                title=self.signal_title + " navigation sliders")
            return
        title = title or self.signal_title +" Navigator" if self.signal_title else ""
        if self.navigator_plot is not None:
            self.navigator_plot.plot()
            return
        elif len(self.navigator_data_function().shape) == 1:
            # Create the figure
            sf = signal1d.Signal1DFigure(title=title)
            axis = self.axes_manager.navigation_axes[0]
            sf.xlabel = '%s' % str(axis)
            if axis.units is not Undefined:
                sf.xlabel += ' (%s)' % axis.units
            sf.ylabel = r'$\Sigma\mathrm{data\,over\,all\,other\,axes}$'
            sf.axis = axis
            sf.axes_manager = self.axes_manager
            self.navigator_plot = sf
            # Create a line to the left axis with the default
            # indices
            sl = signal1d.Signal1DLine()
            sl.data_function = self.navigator_data_function
            sl.set_line_properties(color='blue',
                                   type='step')
            # Add the line to the figure
            sf.add_line(sl)
            sf.plot()
            self.pointer.set_mpl_ax(sf.ax)
            if self.axes_manager.navigation_dimension > 1:
                navigation_sliders(
                    self.axes_manager.navigation_axes,
                    title=self.signal_title + " navigation sliders")
                for axis in self.axes_manager.navigation_axes[:-2]:
                    axis.events.index_changed.connect(sf.update, [])
                    sf.events.closed.connect(
                        partial(axis.events.index_changed.disconnect,
                                sf.update), [])
            self.navigator_plot = sf
        elif len(self.navigator_data_function().shape) >= 2:
            imf = image.ImagePlot()
            imf.data_function = self.navigator_data_function
            imf.colorbar = colorbar
            imf.scalebar = scalebar
            imf.scalebar_color = scalebar_color
            imf.axes_ticks = axes_ticks
            imf.saturated_pixels = saturated_pixels
            imf.vmin = vmin
            imf.vmax = vmax
            imf.no_nans = no_nans
            imf.centre_colormap = centre_colormap
            # Navigator labels
            if self.axes_manager.navigation_dimension == 1:
                imf.yaxis = self.axes_manager.navigation_axes[0]
                imf.xaxis = self.axes_manager.signal_axes[0]
            elif self.axes_manager.navigation_dimension >= 2:
                imf.yaxis = self.axes_manager.navigation_axes[1]
                imf.xaxis = self.axes_manager.navigation_axes[0]
                if self.axes_manager.navigation_dimension > 2:
                    navigation_sliders(
                        self.axes_manager.navigation_axes,
                        title=self.signal_title + " navigation sliders")
                    for axis in self.axes_manager.navigation_axes[2:]:
                        axis.events.index_changed.connect(imf.update, [])
                        imf.events.closed.connect(
                            partial(axis.events.index_changed.disconnect,
                                    imf.update), [])

            imf.title = title
            imf.plot(**kwds)
            self.pointer.set_mpl_ax(imf.ax)
            self.navigator_plot = imf

    def close_navigator_plot(self):
        if self.navigator_plot:
            self.navigator_plot.close()

    def is_active(self):
        return True if self.signal_plot.figure else False

    def plot(self, **kwargs):
        if self.pointer is None:
            pointer = self.assign_pointer()
            if pointer is not None:
                self.pointer = pointer(self.axes_manager)
                self.pointer.color = 'red'
                self.pointer.connect_navigate()
            self.plot_navigator(**kwargs.pop('navigator_kwds', {}))
        self.plot_signal(**kwargs)

    def assign_pointer(self):
        if self.navigator_data_function is None:
            nav_dim = 0
        elif self.navigator_data_function is "slider":
            nav_dim = 0
        else:
            nav_dim = len(self.navigator_data_function().shape)

        if nav_dim == 2:  # It is an image
            if self.axes_manager.navigation_dimension > 1:
                Pointer = widgets.SquareWidget
            else:  # It is the image of a "spectrum stack"
                Pointer = widgets.HorizontalLineWidget
        elif nav_dim == 1:  # It is a spectrum
            Pointer = widgets.VerticalLineWidget
        else:
            Pointer = None
        self._pointer_nav_dim = nav_dim
        return Pointer

    def _on_navigator_plot_closing(self):
        self.navigator_plot = None

    def close(self):
        if self.signal_plot:
            self.signal_plot.close()
        if self.navigator_plot:
            self.navigator_plot.close()
