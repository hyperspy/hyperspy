# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
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


import numpy as np
import os
import scipy
from datetime import datetime


# Plugin characteristics
# ----------------------
format_name = 'DENS'
description = 'Reads heater log from a DENS heating holder'
version = '3.1'
full_support = False
# Recognised file extension
file_extensions = ['dens', 'DENS']
default_extension = 0
# Writing capabilities
writes = False


def _cnv_time(timestr):
    try:
        t = datetime.strptime(timestr.decode(), '%H:%M:%S.%f')
        dt = t - datetime(t.year, t.month, t.day)
        r = float(dt.seconds) + float(dt.microseconds) * 1e-6
    except ValueError:
        r = float(timestr)
    return r


def _bad_file(filename):
    raise AssertionError("Cannot interpret as DENS heater log: %s" % filename)


def file_reader(filename, *args, **kwds):
    with open(filename, 'rt') as f:
        # Strip leading, empty lines
        line = str(f.readline())
        while line.strip() == '' and not f.closed:
            line = str(f.readline())
        try:
            date, version = line.split('\t')
        except ValueError:
            _bad_file(filename)
        if version.strip() != 'Digiheater 3.1':
            _bad_file(filename)
        calib = str(f.readline()).split('\t')
        str(f.readline())       # delta_t
        header_line = str(f.readline())
        try:
            R0, a, b, c = [float(v.split('=')[1]) for v in calib]
            date0 = datetime.strptime(date, "%d/%m/'%y %H:%M")
            date = '%s' % date0.date()
            time = '%s' % date0.time()
        except ValueError:
            _bad_file(filename)
        original_metadata = dict(R0=R0, a=a, b=b, c=c, date=date0,
                                 version=version)

        if header_line.strip() != (
                'sample\ttime\tTset[C]\tTmeas[C]\tRheat[ohm]\tVheat[V]\t'
                'Iheat[mA]\tPheat [mW]\tc'):
            _bad_file(filename)
        try:
            rawdata = np.loadtxt(f, converters={1: _cnv_time}, usecols=(1, 3),
                                 unpack=True)
        except ValueError:
            _bad_file(filename)

    times = rawdata[0]
    # Add a day worth of seconds to any values after a detected rollover
    # Hopefully unlikely that there is more than one, but we can handle it
    for rollover in 1 + np.where(np.diff(times) < 0)[0]:
        times[rollover:] += 60 * 60 * 24
    # Raw data is not necessarily grid aligned. Interpolate onto grid.
    dt = np.diff(times).mean()
    temp = rawdata[1]
    interp = scipy.interpolate.interp1d(times, temp, copy=False,
                                        assume_sorted=True, bounds_error=False)
    interp_axis = times[0] + dt * np.array(range(len(times)))
    temp_interp = interp(interp_axis)

    metadata = {'General': {'original_filename': os.path.split(filename)[1],
                            'date': date,
                            'time': time},
                "Signal": {'signal_type': "",
                           'quantity': "Temperature (Celsius)"}, }

    axes = [{
        'size': len(temp_interp),
            'index_in_array': 0,
            'name': 'Time',
            'scale': dt,
            'offset': times[0],
            'units': 's',
            'navigate': False,
            }]

    dictionary = {'data': temp_interp,
                  'axes': axes,
                  'metadata': metadata,
                  'original_metadata': {'DENS_header': original_metadata},
                  }

    return [dictionary, ]
