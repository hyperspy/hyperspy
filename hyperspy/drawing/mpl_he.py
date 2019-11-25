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
import numpy as np
import logging

from traits.api import Undefined

from hyperspy.drawing import widgets, signal1d, image, utils


_logger = logging.getLogger(__name__)


class MPL_HyperExplorer(object):

    """

    """

    def __init__(self):
        self.signal_data_function = None
        self.navigator_data_function = None
        # args to pass to `__call__`
        self.signal_data_function_kwargs = {}
        self.axes_manager = None
        self.signal_title = ''
        self.navigator_title = ''
        self.quantity_label = ''
        self.signal_plot = None
        self.navigator_plot = None
        self.axis = None
        self.pointer = None
        self._pointer_nav_dim = None
        self._pointer_size = None
        self._resizable_pointer = False
        self._pointer_operation = None

    def plot_signal(self):
        # This method should be implemented by the subclasses.
        # Doing nothing is good enough for signal_dimension==0 though.
        if self.axes_manager.signal_dimension == 0:
            return
        if self.signal_data_function_kwargs.get('fft_shift', False):
            self.axes_manager = self.axes_manager.deepcopy()
            for axis in self.axes_manager.signal_axes:
                axis.offset = -axis.high_value / 2.

    def plot_navigator(self,
                       colorbar=True,
                       scalebar=True,
                       scalebar_color="white",
                       axes_ticks=None,
                       saturated_pixels=None,
                       vmin=None,
                       vmax=None,
                       no_nans=False,
                       centre_colormap="auto",
                       title=None,
                       min_aspect=0.1,
                       **kwds):
        if self.axes_manager.navigation_dimension == 0:
            return
        if self.navigator_data_function is None:
            return
        if self.navigator_data_function == "slider":
            self._get_navigation_sliders()
            return
        title = title or self.signal_title + " Navigator" if self.signal_title else ""
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
                self._get_navigation_sliders()
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
            imf.min_aspect = min_aspect
            # Navigator labels
            if self.axes_manager.navigation_dimension == 1:
                imf.yaxis = self.axes_manager.navigation_axes[0]
                imf.xaxis = self.axes_manager.signal_axes[0]
            elif self.axes_manager.navigation_dimension >= 2:
                imf.yaxis = self.axes_manager.navigation_axes[1]
                imf.xaxis = self.axes_manager.navigation_axes[0]
                if self.axes_manager.navigation_dimension > 2:
                    self._get_navigation_sliders()
                    for axis in self.axes_manager.navigation_axes[2:]:
                        axis.events.index_changed.connect(imf.update, [])
                        imf.events.closed.connect(
                            partial(axis.events.index_changed.disconnect,
                                    imf.update), [])

            imf.title = title
            imf.plot(**kwds)
            self.pointer.set_mpl_ax(imf.ax)
            self.navigator_plot = imf
            self.navigator_plot.resizable_pointer = self._resizable_pointer

    def _get_navigation_sliders(self):
        try:
            self.axes_manager.gui_navigation_sliders(
                title=self.signal_title + " navigation sliders")
        except (ValueError, ImportError) as e:
            _logger.warning("Navigation sliders not available. " + str(e))

    def close_navigator_plot(self):
        if self.navigator_plot:
            self.navigator_plot.close()

    @property
    def is_active(self):
        if self.signal_plot and self.signal_plot.figure is not None:
            return True
        else:
            return False

    def plot(self, resizable_pointer=False, pointer_operation=None, 
             picker_tolerance=10.0, **kwargs):
        self._resizable_pointer = resizable_pointer
        # Parse the kwargs for plotting complex data
        for key in ['power_spectrum', 'fft_shift']:
            if key in kwargs:
                self.signal_data_function_kwargs[key] = kwargs.pop(key)
        if self.pointer is None:
            pointer, param_dict = self.assign_pointer()
            self._pointer_operation = utils.get_pointer_operation(
                pointer_operation)
            if pointer is not None:
                self.pointer = pointer(self.axes_manager, **param_dict)
                self.pointer.color = 'red'
                self.pointer.connect_navigate()
            self.plot_navigator(**kwargs.pop('navigator_kwds', {}))
            if pointer is not None:
                self.pointer.set_picker(picker_tolerance)
        self.plot_signal(**kwargs)
        if self.pointer is not None:
            if self._resizable_pointer:
                self.pointer.events.resized_am.connect(
                        self.signal_plot.update, [])

    def assign_pointer(self):
        param_dict = {}
        if self.navigator_data_function is None:
            nav_dim = 0
        elif self.navigator_data_function == "slider":
            nav_dim = 0
        else:
            nav_dim = len(self.navigator_data_function().shape)

        if nav_dim == 2:  # It is an image
            if self.axes_manager.navigation_dimension > 1:
                if self._resizable_pointer:
                    Pointer = widgets.RectangleWidget
                else:
                    Pointer = widgets.SquareWidget
            else:  # It is the image of a "spectrum stack"
                if self._resizable_pointer:
                    # Is Matplotlib SpanSelector compatible with imshow?
                    # TODO: Need to check which version of matplotlib are 
                    # supporting this
                    Pointer = widgets.RangeWidget
                    param_dict['direction'] = 'vertical'
                else:
                    Pointer = widgets.HorizontalLineWidget
        elif nav_dim == 1:  # It is a spectrum
            if self._resizable_pointer:
                Pointer = widgets.RangeWidget
                param_dict['direction'] = 'horizontal'
            else:
                Pointer = widgets.VerticalLineWidget
        else:
            Pointer = None
            self._resizable_pointer = False
        self._pointer_nav_dim = nav_dim
        return Pointer, param_dict

    def _on_navigator_plot_closing(self):
        self.navigator_plot = None
        if self.pointer is not None:
            # backup the pointer_size to restore it when the plot is reopened
            if self._resizable_pointer:
                self._pointer_size = self.pointer.get_size_in_indices()
                self.pointer.events.resized_am.disconnect(
                        self.signal_plot.update)

    def close(self):
        if self.is_active:
            if self.signal_plot:
                self.signal_plot.close()
            if self.navigator_plot:
                self.navigator_plot.close()
