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

import math

import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

import file_io
import messages
import coordinates
import drawing.utils

class Image():
    '''
    '''    
    
    def __init__(self, dictionary):
        self.title = ''
        self.image_figure = None
        self.image_ax = None
        self.load(dictionary)
        shape = self.data_cube.squeeze().shape
        if len(shape) <= 2:
            shape = (1,1)
        elif len(shape) == 3:
            shape = (shape[2],1)
        elif len(shape) == 4:
            shape = (shape[2],shape[3])
        else:
            messages.warning_exit(
            'Image stacks of more than 4 dimensions are not supported')  

        self.coordinates = coordinates.Coordinates(shape)
    
    def to_spectrum(self):
        from spectrum import Spectrum
        s = Spectrum(
        {'calibration' : {'data_cube' : np.rollaxis(self.data_cube,-1)}})
        return s

    def save(self, filename, **kwds):
        ''''''
        file_io.save(filename, self, **kwds)        
                
    def load(self, dictionary):
        ''''''
        calibration_dict = dictionary['calibration']
        for key in calibration_dict:
            exec('self.%s = calibration_dict[\'%s\']' % (key, key))
        print "Shape: ", self.data_cube.shape
                            
    def crop(self, ix1, iy1, ix2, iy2):
        print "Cropping the image from (%s, %s) to (%s, %s)" % \
        (ix1, iy1, ix2, iy2)
        self.data_cube = self.data_cube[ix1:ix2, iy1:iy2]
        

    def plot(self, z = 0):
        shape = self.data_cube.squeeze().shape
        if len(shape) == 2:
            try:
                if 'FFT' in self.mode:
                    # display the log of the norm
                    dc = np.log(np.abs(self.data_cube))
                else:
                    dc = self.data_cube
            except:
                dc = self.data_cube
        elif len(shape) == 3:
            dc = self.data_cube[...,self.coordinates.ix]
        elif len(shape) == 4:
            dc = self.data_cube[...,self.coordinates.ix,self.coordinates.iy]
        elif len(shape) == 1:
            s = self.to_spectrum()
            s.plot()
        elif len(shape) > 4:
            messages.warning(
            'Image stacks of more than 4 dimensions are not supported')
            return
        self.create_image_figure(dc)
                    
    def change_to_frame(self, i1 = 0, i2 = 0):
        shape = self.data_cube.squeeze().shape
        if len(shape) == 3:
            self.image_ax.images[0].set_array(self.data_cube[...,i1].T)
        elif len(shape) == 4:
            self.image_ax.images[0].set_array(self.data_cube[...,i1,i2].T)
        if self.auto_contrast is True:
            self.image_ax.images[0].autoscale()
        plt.draw()
        
    def update_figure(self):
        self.image_ax.set_title(self.get_title())
        self.change_to_frame(self.coordinates.ix, self.coordinates.iy)
        
        
    def _on_figure_close(self):
        self.image_figure = None
        self.image_ax = None
        self.coordinates.reset()
    def get_title(self):
        title = '%i/%i %i/%i' % (
        self.coordinates.ix, self.coordinates.shape[0] - 1, 
        self.coordinates.iy, self.coordinates.shape[1] - 1,)
        return title
    def create_image_figure(self, dc):
        if self.image_figure is not None:
            # Test if the figure really exists. If not call the reset function 
            # and start again. This is necessary because with some backends 
            # EELSLab fails to connect the close event to the function.
            try:
                self.image_figure.show()
            except:
                self._on_figure_close()
                self.create_image_figure(dc)
            return True
            
        self.image_figure = plt.figure()
        drawing.utils.on_window_close(self.image_figure, self._on_figure_close)
        if hasattr(self, 'title'):
            title = self.title
        else:
            title = 'Image'
        self.image_figure.canvas.set_window_title(title)
        self.image_ax = self.image_figure.add_subplot(111)
        self.image_ax.imshow(dc.T, interpolation = 'nearest')
        self.image_ax.set_title(self.get_title())
        self.image_figure.canvas.draw()
        self.coordinates.on_coordinates_change.append(self.update_figure)
        plt.connect('key_press_event', self.coordinates.key_navigator)
       
        return True
    def auto_contrast(self, perc = 0.01):
        dc = self.data_cube.copy().ravel()
        dc = dc[np.isnan(dc) == False]
        dc.sort()
        i = int(round(len(dc)*perc/100.))
        i = i if i > 0 else 1
        print "i = ", i
        vmin = dc[i]
        vmax = dc[-i]
        print "Automatically setting the constrast values"
        self.vmin = vmin
        self.vmax = vmax
        print "Min = ", vmin
        print "Max = ", vmax
        
plt.show()
