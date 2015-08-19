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

# The details of the format were taken from
# http://www.biochem.mpg.de/doc_tom/TOM_Release_2008/IOfun/tom_mrcread.html
# and http://ami.scripps.edu/software/mrctools/mrc_specification.php


import numpy as np
import scipy
from datetime import datetime


# Plugin characteristics
# ----------------------
format_name = 'DENS'
description = ''
full_support = False
# Recognised file extension
file_extensions = ['dens']
default_extension = 0

# Writing capabilities
writes = False


def _cnv_time(timestr):
    t = datetime.strptime(timestr, '%H:%M:%S.%f')
    dt = t - datetime(t.year, t.month, t.day)
    return float(dt.seconds) + float(dt.microseconds) * 1e-6


def file_reader(filename, *args, **kwds):
    rawdata = np.loadtxt(filename, skiprows=5,
                         converters={1: _cnv_time},
                         usecols=(1, 3), unpack=True)

    times = rawdata[0]
    dt = np.diff(times).mean()
    temp = rawdata[1]
    interp = scipy.interpolate.interp1d(times, temp, copy=False,
                                        assume_sorted=True, bounds_error=False)
    interp_axis = times[0] + dt * np.array(range(len(times)))
    print len(interp_axis), interp_axis[-1], times[-1]
    temp_interp = interp(interp_axis)

    units = ['s']
    names = ['Time']
    offsets = [times[0]]
    scales = [dt]
    navigate = [False]

    axes = [
        {
            'size': len(rawdata[i]),
            'index_in_array': i,
            'name': names[i],
            'scale': scales[i],
            'offset': offsets[i],
            'units': units[i],
            'navigate': navigate[i], }
        for i in xrange(1)]

    dictionary = {'data': temp_interp,
                  'axes': axes,
                  'metadata': {},
                  'original_metadata': {}, }

    return [dictionary, ]
