# -*- coding: utf-8 -*-
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


import copy

import numpy as np
from traits.api import Undefined
import matplotlib.pyplot as plt

from hyperspy.drawing.mpl_he import MPL_HyperExplorer

from hyperspy.drawing import  signal1d, utils

from hyperspy import interactive,roi

class MPL_HyperSignal1D_Explorer(MPL_HyperExplorer):

    """Plots the current spectrum to the screen and a map with a cursor
    to explore the SI.

    """

    def __init__(self,number_of_rois=1,signal=None):
        super(MPL_HyperSignal1D_Explorer, self).__init__(number_of_rois=number_of_rois,signal=signal)
        self.xlabel = ''
        self.ylabel = ''
        #self.right_pointer = None
        #self._right_pointer_on = False
        self._auto_update_plot = True
        self.signal=signal
        self.number_of_rois=number_of_rois
        self.number_of_slices=self.number_of_rois# the idea here is that you may want 1 filtered map per spatio-spectral feature
        self.ROIS=[]# a list of the ROIS pertaining to the hyperimage
        self.ROI2DS=[]# a list of the sub-hyperimages pertaining to the hyperimage
        self.LINES=[]#a list of all the Hyperspy lines. ROIs and Lines should be kept synchronized
        self.ROIS1DSIGNALS=[]#  a list of the summed signal from the ROIs
        self.DATAFUNCTIONS=[]# a list of callback function for the lines (aka data_functions)
        self.MAPROIS=[]# a list of filtered maps
        self.FILTERED_MAPS_PLOT=[]#list of plots with the filtered maps
        self.colors=utils.ColorCycle()._color_cycle#colors will have 7 values, so if you want more than 7

    @property
    def auto_update_plot(self):
        return self._auto_update_plot

    @auto_update_plot.setter
    def auto_update_plot(self, value):
        if self._auto_update_plot is value:
            return
        for line in self.signal_plot.ax_lines + \
                self.signal_plot.right_ax_lines:
            line.auto_update = value
        # if self.pointer is not None:
        #     if value is True:
        #         self.pointer.set_mpl_ax(self.navigator_plot.ax)
        #     else:
        #         self.pointer.disconnect()

    # @property
    # def right_pointer_on(self):
    #     """I'm the 'x' property."""
    #     return self._right_pointer_on
    #
    # @right_pointer_on.setter
    # def right_pointer_on(self, value):
    #     if value == self._right_pointer_on:
    #         return
    #     self._right_pointer_on = value
    #     if value is True:
    #         self.add_right_pointer()
    #     else:
    #         self.remove_right_pointer()

    def plot_signal(self, **kwargs):
        super().plot_signal()
        if self.signal_plot is not None:
            self.signal_plot.plot(**kwargs)
            return
        # Create the figure
        self.axis = self.axes_manager.signal_axes[0]
        sf = signal1d.Signal1DFigure(title=self.signal_title +
                                     " Signal")
        sf.axis = self.axis
        if sf.ax is None:
            sf.create_axis()
        sf.axes_manager = self.axes_manager
        self.xlabel = '{}'.format(self.axes_manager.signal_axes[0])
        if self.axes_manager.signal_axes[0].units is not Undefined:
            self.xlabel += ' ({})'.format(
                self.axes_manager.signal_axes[0].units)
        self.ylabel = self.quantity_label if self.quantity_label != '' \
            else 'Intensity'
        sf.xlabel = self.xlabel
        sf.ylabel = self.ylabel



        #self.signal_plot = sf
        # Create a line to the left axis with the default indices
        # sl = signal1d.Signal1DLine()
        # is_complex = np.iscomplexobj(self.signal_data_function())
        # sl.autoscale = True if not is_complex else False
        # sl.data_function = self.signal_data_function
        # kwargs['data_function_kwargs'] = self.signal_data_function_kwargs
        # sl.plot_indices = True
        # if self.pointer is not None:
        #     color = self.pointer.color
        # else:
        #     color = 'red'
        # sl.set_line_properties(color=color, type='step')
        # # Add the line to the figure:
        # sf.add_line(sl)
        # # If the data is complex create a line in the left axis with the
        # # default coordinates
        # if is_complex:
        #     sl = signal1d.Signal1DLine()
        #     sl.autoscale = True
        #     sl.data_function = self.signal_data_function
        #     sl.plot_coordinates = True
        #     sl._plot_imag = True
        #     sl.set_line_properties(color="blue", type='step')
        #     # Add extra line to the figure
        #     sf.add_line(sl)

        self.signal_plot = sf
        # Create a line to the left axis with the default indices
        if self.signal.data.ndim == 1:

            sl = signal1d.Signal1DLine()
            is_complex = np.iscomplexobj(self.signal_data_function())
            sl.autoscale = True if not is_complex else False
            sl.data_function = self.signal_data_function
            kwargs['data_function_kwargs'] = self.signal_data_function_kwargs
            sl.plot_indices = True
            #if self.pointer is not None:
             #    color = self.pointer.color
            # else:
            #     color = 'red'
            sl.set_line_properties(color='red', type='step')
             # Add the line to the figure:
            sf.add_line(sl)

        # if sf.figure is not None:
        #     if self.axes_manager.navigation_axes:
        #         self.signal_plot.figure.canvas.mpl_connect(
        #             'key_press_event', self.axes_manager.key_navigator)
        #     if self.navigator_plot is not None:
        #         self.navigator_plot.events.closed.connect(
        #             self._on_navigator_plot_closing, [])
        #         sf.events.closed.connect(self.close_navigator_plot, [])
        #         self.signal_plot.figure.canvas.mpl_connect(
        #             'key_press_event', self.key2switch_right_pointer)
        #         self.navigator_plot.figure.canvas.mpl_connect(
        #             'key_press_event', self.key2switch_right_pointer)
        #         self.navigator_plot.figure.canvas.mpl_connect(
        #             'key_press_event', self.axes_manager.key_navigator)
        if self.signal.data.ndim == 3:
            self.create_rois_and_lines()
        sf.plot(**kwargs)
        if self.signal.data.ndim == 3:
            self.create_slices()
        #sf.plot(**kwargs)

    def create_slices(self):

        spectrumleft=self.signal.axes_manager[2].offset
        spectrumright=self.axes_manager[2].size*self.axes_manager[2].scale+self.axes_manager[2].offset
        slicewidth=(spectrumright-spectrumleft)/(5*self.number_of_slices)
        #toto=self.signal.inav[0:1,0:1].squeeze().squeeze()
        #toto.plot()
        #NB: at the moment, only one slice

        #in the following, we are creating a "fake" MPLhse
        #it is an hack to make possible to draw and use a SpanROI to select spectral ranges
        #probably the good solution would be to subclass MPLhse or MPhe...
        #we use the last of the roisignal in the list, because its ax is the one of the figure
        #and this is the one that owns the widgets of the rois, so if we want them to be active
        #we need to put them on the last ax created!
        self.ROIS1DSIGNALS[-1]._plot=MPL_HyperSignal1D_Explorer(number_of_rois=0,signal=self.ROIS1DSIGNALS[-1])
        self.ROIS1DSIGNALS[-1]._plot.signal_plot=self.signal_plot
        self.ROIS1DSIGNALS[-1]._plot.signal_plot.ax=self.LINES[-1].ax
        def toto():
            return self.ROIS1DSIGNALS[-1].data
        self.ROIS1DSIGNALS[-1]._plot.signal_data_function=toto
        self.ROIS1DSIGNALS[-1]._plot.plot_navigation=self

        for i in range(self.number_of_slices):
            left=spectrumleft+i*slicewidth
            right=left+slicewidth
            Roi=roi.SpanROI(left=left,right=right)
            Roi1D=Roi.interactive(self.signal.as_signal2D((0,1)),navigation_signal=self.ROIS1DSIGNALS[-1],color=self.colors[i])
            maproi=interactive.interactive(Roi1D.sum, axis = (0))
            self.MAPROIS.append(maproi)
            maproi.signal_title="Filtered map "+self.signal_title
            maproi.plot()
            self.FILTERED_MAPS_PLOT.append(maproi._plot.signal_plot)
            for key,item in plt.gca().spines.items():
                item.set_color(self.colors[i])
                item.set_linewidth(5)

        for i in range(self.number_of_rois):
            for maproi in self.MAPROIS:
                self.ROIS[i].interactive(self.signal, navigation_signal = maproi,color=self.colors[i])


    def create_rois_and_lines(self):
        spimleft=self.axes_manager[0].offset
        spimright=self.axes_manager[0].size*self.axes_manager[0].scale+self.axes_manager[0].offset
        spimtop=self.axes_manager[1].offset
        spimbottom=self.axes_manager[1].size*self.axes_manager[1].scale+self.axes_manager[1].offset

        roiswidth=(spimright-spimleft)/(2*self.number_of_rois)
        roisheight=(spimbottom-spimtop)/(2*self.number_of_rois)
        def get_data(signal):
            def local_get_data(axes_manager,**kwargs):
                return signal.data
                #return self.signal.inav[0:0].data
            return local_get_data
        #for debugging purpose only
        def print_result():
            print(self.ROIS1DSIGNALS[0].data.sum())

        for i in range(self.number_of_rois):

            left=spimleft+i*roiswidth
            right=left+roiswidth
            top=spimtop+i*roisheight
            bottom=top+roisheight
            Roi = roi.RectangularROI(left=left, right=right, top=top, bottom=bottom)
            self.ROIS.append(Roi)
            Roi2D = Roi.interactive(self.signal, navigation_signal = self.signal,color=self.colors[i])
            self.ROI2DS.append(Roi2D)
            SpectrumRoi = interactive.interactive(Roi2D.sum, axis = (0,1))

            self.ROIS1DSIGNALS.append(SpectrumRoi)
            theline=signal1d.Signal1DLine()
            theline.line_properties={'type':'line','color':self.colors[i]}
            #for some reason (wrong variable bounding???? anyways sounds like a scope issue), the next commented line does not work
            #DATAFUNCTIONS.append(lambda axes_manager: spimroisignal[i].data)
            #the following lines works
            self.DATAFUNCTIONS.append(get_data(self.ROIS1DSIGNALS[i]))
            theline.autoscale = True
            self.LINES.append(theline)
            self.LINES[i].data_function=self.DATAFUNCTIONS[i]
            self.signal_plot.add_line(theline)
            theline.plot_indices=False
            theline.plot()
            SpectrumRoi.events.data_changed.connect(self.LINES[i].update,kwargs=[])

    # def key2switch_right_pointer(self, event):
    #     if event.key == "e":
    #         self.right_pointer_on = not self.right_pointer_on
    #
    # def add_right_pointer(self, **kwargs):
    #     if self.signal_plot.right_axes_manager is None:
    #         self.signal_plot.right_axes_manager = \
    #             copy.deepcopy(self.axes_manager)
    #     if self.right_pointer is None:
    #         pointer = self.assign_pointer()
    #         self.right_pointer = pointer(
    #             self.signal_plot.right_axes_manager)
    #         # The following is necessary because e.g. a line pointer does not
    #         # have size
    #         if hasattr(self.pointer, "size"):
    #             self.right_pointer.size = self.pointer.size
    #         self.right_pointer.color = 'blue'
    #         self.right_pointer.connect_navigate()
    #         self.right_pointer.set_mpl_ax(self.navigator_plot.ax)
    #
    #     if self.right_pointer is not None:
    #         for axis in self.axes_manager.navigation_axes[
    #                 self._pointer_nav_dim:]:
    #             self.signal_plot.right_axes_manager._axes[
    #                 axis.index_in_array] = axis
    #     rl = signal1d.Signal1DLine()
    #     rl.autoscale = True
    #     rl.data_function = self.signal_data_function
    #     rl.set_line_properties(color=self.right_pointer.color,
    #                            type='step')
    #     self.signal_plot.create_right_axis()
    #     self.signal_plot.add_line(rl, ax='right')
    #     rl.plot_indices = True
    #     rl.text_position = (1., 1.05,)
    #     rl.plot(**kwargs)
    #     self.right_pointer_on = True
    #     if hasattr(self.signal_plot.figure, 'tight_layout'):
    #         try:
    #             self.signal_plot.figure.tight_layout()
    #         except BaseException:
    #             # tight_layout is a bit brittle, we do this just in case it
    #             # complains
    #             pass
    #
    # def remove_right_pointer(self):
    #     for line in self.signal_plot.right_ax_lines:
    #         self.signal_plot.right_ax_lines.remove(line)
    #         line.close()
    #     self.right_pointer.close()
    #     self.right_pointer = None
