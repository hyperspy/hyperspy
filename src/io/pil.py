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

import scipy as sp

# Plugin characteristics
# ----------------------
format_name = 'image'
description = 'Import/Export standard image formats using PIL'
full_suport = False
file_extensions = ['bmp', 'dib', 'gif', 'jpeg', 'jpe', 'jpg', 'msp', 'pcx', 
'png', 'ppm', "pbm", "pgm", 'tiff', 'tif', 'xbm', ]
default_extension = -2 # tif

# Reading features
reads_images = True
reads_spectrum = False
reads_spectrum_image = True
# Writing features
writes_images = True
writes_spectrum = False
writes_spectrum_image = False
# ----------------------

# TODO Extend it to support SI
def file_writer(filename, object2save, **kwds):
    dc = object2save.data_cube.squeeze()
    im = sp.misc.imsave(filename, dc.T)
    print "Image saved"
    
def file_reader(filename, **kwds):
    dc = sp.misc.imread(filename).T
    calibration_dict, acquisition_dict , treatments_dict= {}, {}, {}
    calibration_dict['data_cube'] = dc
    if len(dc.shape) == 2:
        dt = 'Image'
    elif len(dc.shape) == 3:
        dt = 'SI'
    return [{'data_type' : dt, 'calibration' : calibration_dict, 
'acquisition' : acquisition_dict},]

