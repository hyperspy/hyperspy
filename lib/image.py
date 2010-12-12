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

import file_io
import utils
import messages
import coordinates

class Image():
    '''
    '''    
    
    def __init__(self, dictionary):
        self.title = ''
        self.image_figure = None
        self.image_ax = None
        self.load(dictionary)
        self.coordinates = coordinates.Coordinates(self.data_cube.shape)
        self.auto_contrast = True
        shape = self.data_cube.squeeze().shape
        if len(shape) <= 2:
            self.coordinates.shape = (0,0)
        elif len(shape) == 3:
            self.coordinates.shape = (shape[2],0)
        elif len(shape) == 4:
            self.coordinates.shape = (shape[2],shape[3])
        else:
            messages.warning_exit(
            'Image stacks of more than 4 dimensions are not supported')  

#        try:
#            self.auto_contrast()
#            self.optimize_colorbar()
#        except:
#            self.vmin = None
#            self.vmax = None
#            self.colorbar_vmin = None
#            self.colorbar_vmax = None
    
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