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
        self.image = None
        self.pixel_size = None
        self.pixel_units = None
        self.plot_scale_bar = False
        self.right_pointer = None
        self.left_pointer = None
        
        # Plotting attributes
        self.spectrum_figure = None
        self.spectrum_ax1 = None
        self.spectrum_ax2 = None
        self.data_explorer_figure = None
        self.data_explorer_ax = None
        self.autoscale = True
        
        # Labels
        self.xlabel = ''
        self.ylabel = ''
        self.image_title = ''
        self.spectrum_title = ''
        
        
        # Lines options
        
        self.line_options = {
            'left_axis' : {
                'data1' : {
                    'line' : None,
                    'type' : None,  # 'scatter', 'step', 'line'
                    'switches' : None, # 'nans' , 'transparency'
                    'color' : None},
                'data2' : {
                    'line' : None,
                    'type' : None,
                    'switches' : None,
                    'color' : None},},
            'right_axis' : {
                'data1' : {
                    'line' : None,
                    'type' : None,
                    'switches' : None,
                    'color' : None},
                'data2' : {
                    'line' : None,
                    'type' : None,
                    'switches' : None,
                    'color' : None},}}         
            
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
            self.spectrum_figure.canvas.set_window_title(self.spectrum_title 
            + ' Spectrum')
            on_window_close(self.spectrum_figure, self._on_spectrum_close)
            self.spectrum_figure.canvas.mpl_connect(
                'draw_event', self.clear)

            
    def _on_spectrum_close(self, *arg):
        
        # First we close the explorer
        if self._does_figure_object_exists(self.data_explorer_figure) is True:
            plt.close(self.data_explorer_figure)
        self.left_pointer.coordinates.disconnect(self._update_spectrum_lines_cursor1)
        self.right_pointer.coordinates.disconnect(self._update_spectrum_lines_cursor2)
        self._remove_cursor(cursor = 2)
        self._remove_cursor(cursor = 1)
        self.data_explorer_figure = None
        self.data_explorer_ax = None
        self.autoscale = True
        self.spectrum_figure = None
        self.spectrum_ax1 = None
        self.spectrum_ax2 = None
        
    def _create_data_explorer_figure(self):
        if self._does_figure_object_exists(self.data_explorer_figure) is True:
            return
        self.data_explorer_figure = plt.figure()
        self.data_explorer_figure.canvas.set_window_title(
        self.image_title + ' Data explorer')


    def plot_data_explorer(self):
        if self.image is None:
            return False
        self._create_data_explorer_figure()
        if not self.data_explorer_figure.axes:
            self.data_explorer_ax = self.data_explorer_figure.add_subplot(111)
        else:
            # Remove the old images
            if self.data_explorer_ax.images:
                self.data_explorer_ax.images = []
        self.data_explorer_ax.imshow(self.image.T, interpolation='nearest')
        self.data_explorer_ax.set_axis_off()
        if self.pixel_size is not None and self.plot_scale_bar is True:
            self.data_explorer_ax.scale_bar = widgets.Scale_Bar(
             ax = self.data_explorer_ax, units = self.pixel_units, 
             pixel_size = self.pixel_size)
        self.data_explorer_figure.subplots_adjust(0,0,1,1)
        # Adjust the size of the window
        size = [ 6,  6.* self.image.shape[1] / self.image.shape[0]]
        self.data_explorer_figure.set_size_inches(size, forward = True)        
        self.data_explorer_figure.canvas.draw()
                
    def plot_spectrum_lines(self, cursor = 1, data = 1):
        '''Plot lines in the spectrum figure
        
        This function will plot all the lines in this order from the one 
        defined by the cursor and data parameters in this order:
            (cursor,data) (1,1),(1,2),(2,1),(2,2).
        '''
        self._create_spectrum_figure()
        if cursor == 1:
            cursor_position = tuple(self.left_pointer.coordinates.coordinates)
            if self.spectrum_ax1 is None:
                self.spectrum_ax1 = self.spectrum_figure.add_subplot(111)
                
            elif not self._does_figure_object_exists(self.spectrum_ax1.figure):
                self._on_spectrum_close()
                self.spectrum_ax1 = self.spectrum_figure.add_subplot(111)
            ax = self.spectrum_ax1
            line_options = self.line_options['left_axis']
            
        if cursor == 2:
            cursor_position = tuple(self.right_pointer.coordinates.coordinates)
            if self.spectrum_ax2 is None:
                self.spectrum_ax2 = self.spectrum_ax1.twinx()
            elif not self._does_figure_object_exists(self.spectrum_ax2.figure):
                self._on_spectrum_close()
                self.plot_spectrum_lines(cursor = 1, data = 1)
            ax = self.spectrum_ax2
            line_options = self.line_options['right_axis']
            
        if data == 1:
            f = self.spectrum_data_function
            lo = line_options['data1']
        elif data == 2:
            f = self.spectrum_data2_function
            lo = line_options['data2']
            
        if f is not None:
            if lo['type'] == 'scatter':
                line, = ax.plot(self.axis, f(cursor = cursor), 
                color = lo['color'], marker = 'o', markersize = 1, 
                linestyle = 'None', markeredgecolor = lo['color'])
            if lo['type'] == 'step':
                line, = ax.step(self.axis, f(cursor = cursor), 
                color = lo['color'])
            if lo['type'] == 'line':
                line, = ax.plot(self.axis, f(cursor = cursor),
                        color = lo['color'])
            line.set_animated(True)
            lo['line'] = line
                
            # Fixing the xlim is necessary because if the data_cube contains  
            # NaNs the x autoscaling will change the limits for the first 
            # figures and leave it like that for the rest
            ax.set_xlim(self.axis[0], self.axis[-1])
            if self.left_pointer is not None and data == 1 and cursor == 1:
                ax.set_xlabel(self.xlabel)
                ax.set_ylabel(self.ylabel)
                ax.set_title('(%i,%i)' % cursor_position)
                ax.title.set_animated(True)
        if data == 1:
            self.plot_spectrum_lines(cursor = cursor, data = 2)
            return
        if data == 2 and cursor == 1 and self.right_pointer.is_on():
            self.plot_spectrum_lines(cursor = 2, data = 1)
            return
            
        # We only draw at the end of the process
        self.spectrum_figure.canvas.draw()
    
    def _remove_cursor(self, cursor = 1):
        if cursor == 1:
            ax = self.spectrum_ax1
            ld1 = self.line_options['left_axis']['data1']['line']
            ld2 = self.line_options['left_axis']['data2']['line']
        elif cursor == 2:
            ax = self.spectrum_ax2
            ld1 = self.line_options['right_axis']['data1']['line']
            ld2 = self.line_options['right_axis']['data2']['line']
        if ax is not None:
            for line in (ld1, ld2):
                if line in ax.lines:
                    ax.lines.remove(line)
            if ax in self.spectrum_figure.axes:
                self.spectrum_figure.axes.remove(ax)
            if cursor == 2:
                self.spectrum_ax2 = None
                self.line_options['right_axis']['data1']['line'] = None
                self.line_options['right_axis']['data2']['line'] = None
                if self.spectrum_ax1 is not None:
                    self.spectrum_ax1.figure.canvas.draw_idle()
            if cursor == 1:
                self.spectrum_ax1 = None
                self.line_options['left_axis']['data1']['line'] = None
                self.line_options['left_axis']['data2']['line'] = None
        

    def _update_spectrum_lines(self, cursor, data = 1):
        '''Update the current spectrum figure'''
        if self._does_figure_object_exists(self.spectrum_figure) is False:
            self._on_spectrum_close()
            return
            
        if cursor == 1:
            ax = self.spectrum_ax1
            line_options = self.line_options['left_axis'] 
        elif cursor == 2:
            # When cursor2 is turned on/off a coordiantes change is triggered 
            # calls this function, so it is here were we have to call the
            # machinery to plot or erase the lines
            
            # If it was just turned on, there will be no lines but the switcher
            # will be on, so we must draw it
            l = self.line_options['right_axis']['data1']['line']
            if l is None and self.right_pointer.is_on() is True:
                self.plot_spectrum_lines(cursor = cursor)
                return
            # If the switcher is off but there is a line we have to remove the 
            # cursor2
            elif l is not None and self.right_pointer.is_on() is False:
                self._remove_cursor(cursor = 2)
                return
            
            # The other cases are: 
            # l is not None and cursor2_ON is True : we proceed to update
            # l is None and cursor2_ON is True : This should never happen
            
            line_options = self.line_options['right_axis']
            ax = self.spectrum_ax2
        
        if ax is None and cursor == 2:
            return
            
        if data == 1:
            f = self.spectrum_data_function
            l = line_options['data1']['line']
        elif data == 2:
            f = self.spectrum_data2_function
            l = line_options['data2']['line']            
        if f is not None:
            ydata = f(cursor = cursor)
            l.set_ydata(ydata)
        
            if self.autoscale is True:
                ax.relim()
                y1, y2 = np.searchsorted(self.axis, 
                ax.get_xbound())
                y2 += 2
                y1, y2 = np.clip((y1,y2),0,len(ydata-1))
                clipped_ydata = ydata[y1:y2]
                y_max, y_min = np.nanmax(clipped_ydata), np.nanmin(clipped_ydata)
                ax.set_ylim(y_min, y_max)
            
            if self.right_pointer.is_on() is not True:
                self.spectrum_ax1.title.set_text(
                'Cursor (%i,%i) ' % 
                tuple(self.left_pointer.coordinates.coordinates))   
            else:
                self.spectrum_ax1.title.set_text(
                'Cursor 1 (%i,%i) Cursor 2 (%i,%i)' % 
                (tuple(self.left_pointer.coordinates.coordinates) + 
                tuple(self.right_pointer.coordinates.coordinates)))
            if data == 1:
                self._update_spectrum_lines(cursor = cursor, data = 2)
                
            # Update the drawing using blit
            background = ax.old_bbox
            ax.figure.canvas.restore_region(background)
#            for line in ax.lines:
#                ax.draw_artist(line)
#                # We must also draw the ax2 lines if active
#                if self.spectrum_ax1 == ax and self.spectrum_ax2 is not None:
#                    for line in self.spectrum_ax2.lines:
#                        self.spectrum_ax2.draw_artist(line)
            ax.figure.canvas.draw_idle()
            ax.figure.canvas.blit(ax.bbox)
            

    def _update_spectrum_lines_cursor1(self):
        self._update_spectrum_lines(cursor = 1)
    def _update_spectrum_lines_cursor2(self):
        self._update_spectrum_lines(cursor = 2)
        
    def plot(self):
        if self.image is not None:
            # It is not an 1D HSI, so we plot the data explorer
            self.plot_data_explorer()
            self.left_pointer.add_axes(self.data_explorer_ax)
            self.right_pointer.add_axes(self.data_explorer_ax)
            self.data_explorer_figure.canvas.mpl_connect(
            'key_press_event', self.left_pointer.coordinates.key_navigator)
        
        self.plot_spectrum_lines(cursor = 1)
#        self.spectrum_figure.canvas.connect(
#            'key_press_event', widgets.cursor.key_navigator)
        if self.image is not None:
            # It is not an 1D HSI, so we plot the data explorer, we connect the 
            # coordinates changes to the functions that update de lines
            self.left_pointer.coordinates.connect(
            self._update_spectrum_lines_cursor1)
            self.right_pointer.coordinates.connect(
            self._update_spectrum_lines_cursor2)
            self.spectrum_figure.canvas.mpl_connect('key_press_event', 
                                self.left_pointer.coordinates.key_navigator)
            self.data_explorer_figure.canvas.draw()
            self.left_pointer.update_patch_position()
            self.right_pointer.update_patch_position()
            
                                    
    def clear(self, event = None):
        'clear the cursor'
        # By using blitz the drawing time /spectrum is 30 ms
        # Without it 80 ms
        canvas = event.canvas
        for ax in canvas.figure.axes:
            ax.old_bbox = canvas.copy_from_bbox(ax.bbox)
            for line in ax.lines:
                ax.draw_artist(line)
            canvas.blit(ax.bbox)
plt.show()
