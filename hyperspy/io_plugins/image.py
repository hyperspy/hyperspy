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
try:
    from mahotas import imread, imsave
    mahotas_installed = True
except:
    from scipy.misc import imread, imsave
    mahotas_installed = False

# Plugin characteristics
# ----------------------
format_name = 'image'
description = 'Import/Export standard image formats using PIL or freeimage'
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
writes_spectrum_image = True
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
def file_writer(filename, signal, _rescale = True, file_format='tif', 
                only_view = False, **kwds):
    '''Writes data to any format supported by PIL or freeimage if mahotas is 
        installed
        
        Note that only when mahotas and freeimage are installed it is possible 
        to write 16-bit tiff files.
        
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
            PIL of mahotas if installed.  
    '''
    if only_view is True and signal.axes_manager.signal_dimension == 2:
        dc = signal()
    elif only_view is False and len(signal.data.squeeze().shape) == 2:
        dc = signal.data.squeeze()
    else:
        raise IOError("This format only supports writing of 2D data")
        
    if file_format in ('tif', 'tiff') and mahotas_installed is True:
        
        bits = 16    
    else:
        # Only tiff supports 16-bits
        bits = 8
        
    if _rescale is True:
        dc = rescale(dc, bits)

    imsave(filename, dc.astype('uint%s' % bits))
    print "Image saved"
    
def file_reader(filename, **kwds):
    '''Read data from any format supported by PIL or freeimage if mahotas is 
    installed
    
    Note that only when mahotas and freeimage are installed it is possible 
    to read 16-bit tiff files.
    
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

