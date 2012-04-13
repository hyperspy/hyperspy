# -*- coding: utf-8 -*-
# Copyright 2007-2011 The Hyperspy developers
#
# This file is part of  Hyperspy.
#
#  Hyperspy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  Hyperspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  Hyperspy.  If not, see <http://www.gnu.org/licenses/>.


import numpy as np

from scipy.misc import imread, imsave

# Plugin characteristics
# ----------------------
format_name = 'image'
description = 'Import/Export standard image formats using PIL or freeimage'
full_suport = False
file_extensions = ['png', 'bmp', 'dib', 'gif', 'jpeg', 'jpe', 'jpg', 
                   'msp', 'pcx', 'ppm', "pbm", "pgm", 'xbm', 'spi',]
default_extension = 0 # png

# Reading features
reads_2d = True
reads_1d = False
reads_3d = True
reads_xd = False
# Writing features
writes_2d = True
writes_1d = False
writes_3d = False
writes_xd = False
# ----------------------


def rescale(data, bits):
    # First check if rescaling makes sense
    dtn = data.dtype.name
    if 'int' in dtn:
        if bits == 8 and ('16' not in dtn or '32' not in dtn):
            return data
        elif bits == 16 and '32' not in dtn:
            return data
    data = data - data.min()
    data = data / data.max()
    if bits == 8:
        data = 255 * data
        data = data.astype(np.uint8)
    elif bits == 16:
        data = 65535 * data
        data = data.astype(np.uint16)
    return data
        
# TODO Extend it to support SI
def file_writer(filename, signal, _rescale=True, file_format='png', **kwds):
    '''Writes data to any format supported by PIL
        
        Parameters
        ----------
        filename: str
        signal: a Signal instance
        rescale: bool
            Rescales the data to use the full dynamic range available in the 
            chosen encoding. Note that this operation (obviously) affects the 
            scale of the data what might not always be a good idea
        file_format : str
            The fileformat defined by its extension that is any one supported by 
            PIL.  
    '''
                
    if _rescale is True:
        dc = rescale(dc, bits)

    imsave(filename, dc.astype('uint%s' % bits))
    
def file_reader(filename, **kwds):
    '''Read data from any format supported by PIL.
    
    Parameters
    ----------
    filename: str
        By using '*' it is possible to load a collection of images of the same 
        size into a three dimensional dataset.
    '''
    if '*' in filename:
        from glob import glob
        flist=glob(filename)
        flist.sort()
        imsample = imread(flist[0])
        w=imsample.shape[0]
        h=imsample.shape[1]
        d=len(flist)
        dc=np.zeros((d,w,h))
        for i in xrange(d):
            dc[i,:,:] = imread(flist[i])
    else:
        dc = imread(filename)
    dt = 'image'    
    return [{'data':dc, 
             'mapped_parameters': {
                'original_filename' : filename,
                'record_by': dt,
                'signal_type' : None,
                }
             }]

