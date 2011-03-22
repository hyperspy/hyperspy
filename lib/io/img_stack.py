# -*- coding: utf-8 -*-
# Copyright © 2011 Michael Sarahan, Francisco Javier de la Peña
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

import Image
import numpy as np
from glob import glob

# Plugin characteristics
# ----------------------
format_name = 'image'
description = 'Import/Export standard image format stack using PIL'
full_suport = False
file_extensions = ['bmp', 'dib', 'gif', 'jpeg', 'jpe', 'jpg', 'msp', 'pcx', 
                   'png', 'ppm', "pbm", "pgm", 'tiff', 'tif', 'xbm', 'spi']
default_extension = -2 # tif

# Reading features
reads_images = True
reads_spectrum = False
reads_spectrum_image = False
# Writing features
writes_images = True
writes_spectrum = False
writes_spectrum_image = False
# ----------------------

def file_reader(filename, **kwds):
    flist=glob(filename)
    imsample=Image.open(flist[0])
    w=imsample.size[0]
    h=imsample.size[1]
    d=len(flist)
    dc=np.zeros((w,h,d))
    for i in xrange(d):
        dc[:,:,i]=np.array(Image.open(flist[i]))
    calibration_dict, acquisition_dict , treatments_dict= {}, {}, {}
    calibration_dict['data_cube'] = dc
    dt = 'Image'
    return [{'data_type' : dt, 'calibration' : calibration_dict, 
             'acquisition' : acquisition_dict},]

def file_writer(filename, object2save, file_format='tif', rescale=False, **kwds):
    if not file_format in extensions:
        print "Warning! file format %s is not supported by PIL!"%file_format
        return -1
    if rescale:
        arr=arr-arr.min()
        arr=arr/arr.max()
        if rescale==8:
            arr=np.array(arr*255,dtype=np.uint8)
        elif rescale==16:
            arr=np.array(arr*65535,dtype=np.uint16)
        elif rescale==32:
            arr=np.array(arr*(2**32),dtype=np.uint32)
    for i in xrange(array.shape[2]):
        img=Image.fromarray(array[:,:,i])
        img.save(filename+'%05i'%i+'.'+file_format)
    print "%i images saved."%array.shape[2]
