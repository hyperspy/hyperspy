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

from hyperspy.misc.tifffile import imsave, imread

# Plugin characteristics
# ----------------------
format_name = 'image'
description = 'Import/Export standard image formats using PIL or freeimage'
full_suport = False
file_extensions = ['tif', 'tiff', 'TIF', 'TIFF']
default_extension = 0 # tif

# Reading features
reads_2d = True
reads_1d = False
reads_3d = True
reads_xd = False
# Writing features
writes_2d = True
writes_1d = False
writes_3d = True
writes_xd = False
# ----------------------


def file_writer(filename, signal, _rescale = True,  **kwds):
    '''Writes data to tif using Christoph Gohlke's tifffile library
        
        Parameters
        ----------
        filename: str
        signal: a Signal instance
    '''
    
    imsave(filename, signal.data.squeeze(), **kwds)
    
def file_reader(filename, output_level=0, record_by='image',**kwds):
    '''Read data from tif files using Christoph Gohlke's tifffile
    library
    
    Parameters
    ----------
    filename: str
    output_level : int
        Has no effect
    record_by: {'image'}
        Has no effect because this format only supports recording by
        image.
    
    '''
    dc = imread(filename, **kwds)
    dt = 'image'    
    return [{'data':dc, 
             'mapped_parameters': { 'original_filename' : filename,
                                    'record_by': dt,
                                    'signal_type' : None,}
             }]

