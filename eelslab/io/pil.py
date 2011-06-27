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
import Image
import numpy as np

# Plugin characteristics
# ----------------------
format_name = 'image'
description = 'Import/Export standard image formats using PIL'
full_suport = False
file_extensions = ['bmp', 'dib', 'gif', 'jpeg', 'jpe', 'jpg', 'msp', 'pcx', 
                   'png', 'ppm', "pbm", "pgm", 'tiff', 'tif', 'xbm', 'spi',]
default_extension = -3 # tif

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
def file_writer(filename, object2save, rescale=None, file_format='tif',**kwds):
    dc = object2save.data_cube.squeeze()
    if rescale:
        dc=dc-dc.min()
        dc=dc/dc.max()
        if rescale==8:
            dc=np.array(dc*255,dtype=np.uint8)
        elif rescale==16:
            dc=np.array(dc*65535,dtype=np.uint16)
        elif rescale==32:
            dc=np.array(dc*(2**32 - 1),dtype=np.uint32)
    if len(dc.shape)==3:
        for i in xrange(dc.shape[2]):
            img=Image.fromarray(dc[:,:,i])
            img.save(filename+'%05i'%i+'.'+file_format)
    else:
        im = sp.misc.imsave(filename, dc.T)
    print "Image saved"
    
def file_reader(filename, **kwds):
    if '*' in filename:
        from glob import glob
        flist=glob(filename)
        flist.sort()
        imsample=Image.open(flist[0])
        w=imsample.size[0]
        h=imsample.size[1]
        d=len(flist)
        dc=np.zeros((w,h,d))
        for i in xrange(d):
            dc[:,:,i]=np.array(Image.open(flist[i]))
        dt='Image'
    else:
        dc = sp.misc.imread(filename).T
        if len(dc.shape) == 2:
            dt = 'Image'
        elif len(dc.shape) == 3:
            dt = 'SI'

    return [{'data_type' : dt, 'data':dc}]

