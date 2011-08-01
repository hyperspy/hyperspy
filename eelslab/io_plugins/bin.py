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

# Plugin characteristics
# ----------------------
format_name = 'Plain bin'
description = 'Can be useful to exchange information with propietary software'
full_suport = False
# Recognised file extension
file_extensions = ['bin',]
default_extension = 0
# Reading capabilities
reads_images = True
reads_spectrum = False
reads_spectrum_image = True
# Writing capabilities
writes_images = True
writes_spectrum = False
writes_spectrum_image = True
# ----------------------

dtype2dm3 = {
'L' : 'Integer 4 byte unsigned',
'l' : 'Integer 4 byte signed',
'f' : 'Real 4 byte',
'd' : 'Real 8 byte',
'F' : 'Complex 8',
'D' : 'Complex 16'
}

def file_reader(filename, shape, data_type = None, **kwds):
    # TODO: write the reader...
    
    dc = np.fromfile(filename, **kwds)
    if len(shape) == 3:
        dc = dc.reshape((shape[2], shape[1], shape[0])).swapaxes(1,2)
        if data_type is None:
            data_type = 'SI'
    elif len(shape) == 2:
        dc = dc.reshape(shape).T
        if data_type is None:
            data_type = 'Image'
    return [{'mapped_parameters':{
				'data_type' : data_type,
				'name' : filename,
				}, 
			'data':dc,
			},]  
    
def file_writer(filename, object2save, *args, **kwds):
    from eelslab import spectrum
    from eelslab import image
    if isinstance(object2save,spectrum.Spectrum):
        dc = object2save.data_cube
        depth, width, height = dc.shape
        par = '(%s,%s, %s)' % (width, height, depth)
        filename = filename[:-4] + par + filename[-4:]

        if object2save.is_spectrum_image():
            dc = dc.swapaxes(1,2)
        dc.ravel().tofile(filename)
        dm3dtype = dtype2dm3[dc.dtype.char]

        print "Parameters to import in Digital Micrograph"
        print "------------------------------------------"
        print "Data type: %s" % dm3dtype
        print "Width : ", width
        print "Height : ", height
        print "Depth : ", depth
        print "Format: binary" 
        print "Swap data bytes : False"
    
    elif isinstance(object2save, image.Image):
        dc = object2save.data_cube.T
        width = dc.shape[1]
        height = dc.shape[0]
        par = '(%s,%s)' % (width, height)
        filename = filename[:-4] + par + filename[-4:]
        dc.tofile(filename)
        dm3dtype = dtype2dm3[dc.dtype.char]
        print "Parameters to import in Digital Micrograph"
        print "------------------------------------------"
        print "Data type: %s" % dm3dtype
        print "Width : ", width
        print "Height : ", height
        print "Format: binary" 
        print "Swap data bytes : False"
   
