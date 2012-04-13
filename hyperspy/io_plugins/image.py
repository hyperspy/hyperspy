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
format_name = 'Image'
description = 'Import/Export standard image formats using PIL or freeimage'
full_suport = False
file_extensions = ['png', 'bmp', 'dib', 'gif', 'jpeg', 'jpe', 'jpg', 
                   'msp', 'pcx', 'ppm', "pbm", "pgm", 'xbm', 'spi',]
default_extension = 0 # png


# Writing features
writes = [(2,0),]
# ----------------------



        
# TODO Extend it to support SI
def file_writer(filename, signal, file_format='png', **kwds):
    '''Writes data to any format supported by PIL
        
        Parameters
        ----------
        filename: str
        signal: a Signal instance
        file_format : str
            The fileformat defined by its extension that is any one supported by 
            PIL.  
    '''
    imsave(filename, signal.data)
    
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

