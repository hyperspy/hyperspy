# -*- coding: utf-8 -*-
# Copyright 2007-2015 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.

import mrcz as _mrcz

import os
from datetime import datetime, timedelta
from dateutil import tz
from traits.api import Undefined
import numpy as np
import logging
import warnings

from hyperspy._signals import signal2d
# from hyperspy.misc.array_tools import sarray2dict, dict2sarray

_logger = logging.getLogger(__name__)
# Plugin characteristics
# ----------------------
format_name = 'MRCZ'
description = 'Compressed MRC file format extension with blosc meta-compression'
full_support = False
# Recognised file extension
file_extensions = ['mrc', 'MRC', 'mrcz', 'MRCZ']
default_extension = 2

# Writing capabilities:
writes = [(2, 2), (2, 1), (2, 0)]  # TODO
magics = [0x0102] # TODO


mapping = {
    'mrcz_header.voltage':
    ("Acquisition_instrument.TEM.beam_energy", lambda x: x),
    # There is no metadata field for detector gain
    #'mrcz_header.gain':
    #("Acquisition_instrument.TEM.Detector.gain", lambda x: x),
    # There is no metadata field for spherical aberration
    #'mrcz_header.C3':
    #("Acquisition_instrument.TEM.C3", lambda x: x),
}


def file_reader(filename, endianess='<', load_to_memory=True, mmap_mode='c',
                **kwds):
    _logger.debug("Reading MRCZ file: %s" % filename)
    metadata = {}
    
    useMemmap = not load_to_memory
    mrcz_endian = 'le' if endianess == '<' else 'be'
    data, mrcz_header = _mrcz.readMRC( endian=mrcz_endian, useMemmap=useMemmap,
                                  **kwds )

    # Create the axis objects for each axis
    dim = data.ndim
    names = ['z','y','x']
    units = [ mrcz_header['pixelunits'] ] * 3
    axes = [
        {   'size': data.shape[i],
            'index_in_array': i,
            'name': names[i],
            'scale': mrcz_header['pixelsize'][i],
            'offset': 0.0,
            'units': units[i], }
        for i in range(dim)]

    dictionary = {'data': data,
                  'axes': axes,
                  'metadata': metadata,
                  'original_metadata': {'mrcz_header':mrcz_header},
                  'mapping': mapping, }

    return [dictionary, ]


def file_writer(filename, signal, **kwds):
    if isinstance(signal, signal2d.Signal2D ):
        raise TypeError( "MRCZ supports 2D and 3D data only." )

    endianess = kwds.pop('endianess', '<')
    doAsync = kwds.pop('do_async', False)

    blosc_clevel = kwds.pop('clevel', 1)
    blosc_compressor = kwds.pop('compressor', None)

    # Get pixelsize and pixelunits from the axes
    pixelunits = signal.axes['units'][0]
    pixelsize = signal.axes['scale']
    # Strip out voltage, C3, and gain from signal.original_metadata
    copy_meta = signal.original_metadata['mrcz_header'].copy()
    voltage = copy_meta.pop('voltage', 0.0)

    # There aren't hyperspy fields for spherical aberration or detector gain
    C3 = copy_meta.pop('C3',0.0)
    gain = copy_meta.pop('gain', 1.0)
    # And compression meta-info 
    blosc_compressor = copy_meta.pop('compressor', None)
    blosc_clevel = copy_meta.pop('clevel', 1)
    n_threads = copy_meta.pop('n_threads', None)

    mrcz_endian = 'le' if endianess == '<' else 'be'
    _mrcz.writeMRC( signal['data'], filename, meta=signal.metadata, endian=mrcz_endian,
                    pixelsize=pixelsize, pixelunits=pixelunits,
                    voltage=voltage, C3=C3, gain=gain,
                    compressor=blosc_compressor, clevel=blosc_clevel, n_threads=n_threads )

