# -*- coding: utf-8 -*-
# Copyright © 2007 Francisco Javier de la Peña
#
# This file is part of EELSLab.
#
# EELSLab is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# EELSLab is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with EELSLab; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  
# USA

import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
# Set the matplotlib cmap to gray (the default is jet)
plt.rcParams['image.cmap'] = 'gray' 

from utils import on_window_close
import widgets

class MPL_HyperSpectrum_Explorer():
    def __init__(self):
        
        # Data attributes
        self.spectrum_data_function = None
        self.spectrum_data2_function = None
        self.axis = None
        self.axis_switches = None
        self.axis_switches2 = None
        self.image = None
        self.pointers = None
        
        # Plotting attributes
        self.spectrum_figure = None
        self.spectrum_ax1 = None
        self.spectrum_ax2 = None
        self.data_explorer_figure = None
        self.data_explorer_ax = None
        self.autoscale = True
        
        self._line_d1c1 = None
        self._line_d1c2 = None
        self._line_d2c1 = None
        self._line_d2c2 = None
        
        # Labels
        self.xlabel = ''
        self.ylabel = ''
        self.image_title = ''
        self.spectrum_title = ''
        self.data_explorer_window_title = 'Data explorer'
        
        # Lines options
        
        self.lines_options = {
            'data1' : {
                'type' : None,
                'switches' : None,},
            'data2' : {
                'type' : None,
                'switches' : None,}}
        # Type options:'scatter', 'step', 'line'
        self.line_type_d1 = None
        self.line_type_d2 = None
        
        # Switches options: 'nans' , 'transparency'
        self.line_switches_d1 = None
        self.line_switches_d2 = None
        
        # Colors options: any valid color
        self.line_color_d1c1 = 'blue'
        self.line_color_d1c2 = 'green'
        self.line_color_d2c1 = 'red'
        self.line_color_d2c2 = 'black'
        
        
    def _does_figure_object_exists(self, fig_obj):
        if fig_obj is None:
            return False
        else:
            # Test if the figure really exists. If not call the reset function 
            # and start again. This is necessary because with some backends 
            # EELSLab fails to connect the close event to the function.
            try:
                fig_obj.show()
                return True
            except:
                fig_obj = None
                return False
                
    def _create_spectrum_figure(self):
        if self._does_figure_object_exists(self.spectrum_figure) is True:
            return            
        else:
            self.spectrum_figure = plt.figure()
            self.spectrum_figure.canvas.set_window_title(self.spectrum_title)
            on_window_close(self.spectrum_figure, self._on_spectrum_close)
            
    def _on_spectrum_close(self, *arg):
        if self._does_figure_object_exists(self.data_explorer_figure) is True:
            plt.close(self.data_explorer_figure)
        widgets.cursor.disconnect(self._update_spectrum_lines_cursor1)
        widgets.cursor2.disconnect(self._update_spectrum_lines_cursor2)
        self.data_explorer_figure = None
        self.data_explorer_ax = None
#        self.pointers = None
        self.autoscale = True
        self.spectrum_figure = None
        self.spectrum_ax1 = None
        self.spectrum_ax2 = None
        
    def _create_data_explorer_figure(self):
        if self._does_figure_object_exists(self.data_explorer_figure) is True:
            return
        self.data_explorer_figure = plt.figure()
        self.data_explorer_figure.canvas.set_window_title(
        self.data_explorer_window_title)


    def plot_data_explorer(self):
        if self.image is None:
            return False
        self._create_data_explorer_figure()
        if not self.data_explorer_figure.axes:
            self.data_explorer_ax = self.data_explorer_figure.add_subplot(111)
        else:
            # Remove the former images
            if self.data_explorer_ax.images:
                self.data_explorer_ax.images = []
        self.data_explorer_ax.imshow(self.image.T, interpolation='nearest')
        self.data_explorer_figure.canvas.draw()
                
    def plot_spectrum_lines(self, cursor = 1):
        self._create_spectrum_figure()
        if cursor == 1:
            cursor_position = tuple(widgets.cursor.coordinates)
            if self.spectrum_ax1 is not None:
                self.spectrum_figure.axes.remove(self.spectrum_ax1)
            self.spectrum_ax1 = self.spectrum_figure.add_subplot(111)
            ax = self.spectrum_ax1
            
            line1_type = self.line_type_d1
            line2_type = self.line_type_d2
            
            # Switches options: 'nans' , 'transparency'
            line1_switches = self.line_switches_d1
            line2_switches = self.line_switches_d2
            
            # Colors options: any valid color
            line1_color = self.line_color_d1c1
            line2_color = self.line_color_d2c1
        if cursor == 2:
            cursor_position = tuple(widgets.cursor2.coordinates)
            if self.spectrum_ax2 is not None:
                self.spectrum_figure.axes.remove(self.spectrum_ax2)
            self.spectrum_ax2 = self.spectrum_ax1.twinx()
            ax = self.spectrum_ax2
            
            line1_type = self.line_type_d1
            line2_type = self.line_type_d2
            
            # Switches options: 'nans' , 'transparency'
            line1_switches = self.line_switches_d1
            line2_switches = self.line_switches_d2
            
            # Colors options: any valid color
            line1_color = self.line_color_d1c2
            line2_color = self.line_color_d2c2

        if self.spectrum_data_function is not None:
            if line1_type == 'scatter':
                line1, = ax.plot(self.axis, self.spectrum_data_function(cursor = cursor),
                        color = line1_color, marker = 'o')
            if line1_type == 'step':
                line1, = ax.step(self.axis, self.spectrum_data_function(cursor = cursor),
                        color = line1_color)
            if line1_type == 'line':
                line1, = ax.step(self.axis, self.spectrum_data_function(cursor = cursor),
                        color = line1_color)
            if cursor == 1:
                self._line_d1c1 = line1
            elif cursor == 2:
                self._line_d1c2 = line1
            
        if self.spectrum_data2_function is not None:
            if line2_type == 'scatter':
                line2, = ax.plot(
                self.axis, self.spectrum_data2_function(cursor = cursor),
                        color = line2_color, marker = 'o')
            if line2_type == 'step':
                line2, = ax.step(
                self.axis, self.spectrum_data2_function(cursor = cursor),
                        color = line2_color)
            if line2_type == 'line':
                line2, = ax.step(
                self.axis, self.spectrum_data2_function(cursor = cursor),
                        color = line2_color)
            if cursor == 1:
                self._line_d2c1 = line2
            elif cursor == 2:
                self._line_d2c2 = line2
                
        # Fixing the xlim is necessary because if the data_cube contains NaNs 
        # the x autoscaling will change the limits for the first figures and
        # leave it like that for the rest
        ax.set_xlim(self.axis[0], self.axis[-1])
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        
        self.spectrum_ax1.set_title('(%i,%i)' % cursor_position)
        if self.pointers is not None:
            self._previous_cursor2_ON = self.pointers.cursor2_ON
            if self.pointers.cursor2_ON and cursor == 1:
                self.plot_spectrum_lines(cursor = 2)
    
    def _remove_cursor2(self):
        self.spectrum_figure.axes.remove(self.spectrum_ax2)
        self.spectrum_ax2 = None        

    def _update_spectrum_lines(self, cursor):
        '''Update the current spectrum figure'''
        
        if self._does_figure_object_exists(self.spectrum_figure) is False:
            self._on_spectrum_close()
            return
        if self.pointers and (self._previous_cursor2_ON is not 
        self.pointers.cursor2_ON):
            if self.pointers.cursor2_ON:
                self.plot_spectrum_lines(cursor = 2)
            else:
                self._remove_cursor2()
            self._previous_cursor2_ON = self.pointers.cursor2_ON
            #  _update_spectrum_line will be called by the cursor2 functions so 
            # there is no need to carry on
        else:
            if cursor == 1:
                ax = self.spectrum_ax1
                line1 = self._line_d1c1
                line2 = self._line_d2c1
            elif cursor == 2:
                ax = self.spectrum_ax2
                line1 = self._line_d1c2
                line2 = self._line_d2c2
            if ax is None:
                self.spectrum_figure.canvas.draw()
                return
            if self.spectrum_data_function is not None:
                ydata = self.spectrum_data_function(cursor = cursor)
            if self.spectrum_data2_function is not None:
                ydata2 = self.spectrum_data2_function(cursor = cursor)
            if line1 is not None:
                line1.set_ydata(ydata)
            if line2 is not None:
                line2.set_ydata(ydata2)
            if self.autoscale is True:
                ax.relim()
                y1, y2 = np.searchsorted(self.axis, 
                ax.get_xbound())
                y2 += 2
                y1, y2 = np.clip((y1,y2),0,len(ydata-1))
                clipped_ydata = ydata[y1:y2]
                y_max, y_min = clipped_ydata.max(), clipped_ydata.min()
                ax.set_ylim(y_min, y_max)
            if self.pointers.cursor2_ON is not True:
                self.spectrum_ax1.set_title(
                'Cursor (%i,%i) ' % 
                tuple(widgets.cursor.coordinates))   
            else:
                self.spectrum_ax1.set_title(
                'Cursor 1 (%i,%i) Cursor 2 (%i,%i)' % 
                (tuple(widgets.cursor.coordinates) + 
                tuple(widgets.cursor2.coordinates)))
                
            self.spectrum_figure.canvas.draw()
        
    def _update_spectrum_lines_cursor1(self):
        self._update_spectrum_lines(cursor = 1)
    def _update_spectrum_lines_cursor2(self):
        self._update_spectrum_lines(cursor = 2)
        
    def plot(self):
        if self.image is not None:
            # It is not an 1D HSI, so we plot the data explorer
            self.plot_data_explorer()
            self.pointers.add_axes(self.data_explorer_ax)
            plt.connect('key_press_event', widgets.cursor.key_navigator)
        
        self.plot_spectrum_lines(cursor = 1)
        if self.image is not None:
            # It is not an 1D HSI, so we plot the data explorer, we connect the 
            # coordinates changes to the functions that update de lines
            widgets.cursor.connect(self._update_spectrum_lines_cursor1)
            widgets.cursor2.connect(self._update_spectrum_lines_cursor2)
            plt.connect('key_press_event', widgets.cursor.key_navigator)
            self.data_explorer_figure.canvas.draw()
plt.show()