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

import os
import logging

import numpy as np

_logger = logging.getLogger(__name__)

no_netcdf = False
try:
    from netCDF4 import Dataset
    which_netcdf = 'netCDF4'
except BaseException:
    try:
        from netCDF3 import Dataset
        which_netcdf = 'netCDF3'
    except BaseException:
        try:
            from Scientific.IO.NetCDF import NetCDFFile as Dataset
            which_netcdf = 'Scientific Python'
        except BaseException:
            no_netcdf = True

# Plugin characteristics
# ----------------------
format_name = 'netCDF'
description = ''
full_support = True
file_extensions = ('nc', 'NC')
default_extension = 0
# Writing features
writes = False

# ----------------------


attrib2netcdf = \
    {
        'energyorigin': 'energy_origin',
        'energyscale': 'energy_scale',
        'energyunits': 'energy_units',
        'xorigin': 'x_origin',
        'xscale': 'x_scale',
        'xunits': 'x_units',
        'yorigin': 'y_origin',
        'yscale': 'y_scale',
        'yunits': 'y_units',
        'zorigin': 'z_origin',
        'zscale': 'z_scale',
        'zunits': 'z_units',
        'exposure': 'exposure',
        'title': 'title',
        'binning': 'binning',
        'readout_frequency': 'readout_frequency',
        'ccd_height': 'ccd_height',
        'blanking': 'blanking'
    }

acquisition2netcdf = \
    {
        'exposure': 'exposure',
        'binning': 'binning',
        'readout_frequency': 'readout_frequency',
        'ccd_height': 'ccd_height',
        'blanking': 'blanking',
        'gain': 'gain',
        'pppc': 'pppc',
    }

treatments2netcdf = \
    {
        'dark_current': 'dark_current',
        'readout': 'readout',
    }


def file_reader(filename, *args, **kwds):
    if no_netcdf is True:
        raise ImportError("No netCDF library installed. "
                          "To read EELSLab netcdf files install "
                          "one of the following packages:"
                          "netCDF4, netCDF3, netcdf, scientific")

    ncfile = Dataset(filename, 'r')

    if hasattr(ncfile, 'file_format_version'):
        if ncfile.file_format_version == 'EELSLab 0.1':
            dictionary = nc_hyperspy_reader_0dot1(
                ncfile,
                filename,
                *args,
                **kwds)
    else:
        ncfile.close()
        raise IOError('Unsupported netCDF file')

    return dictionary,


def nc_hyperspy_reader_0dot1(ncfile, filename, *args, **kwds):
    calibration_dict, acquisition_dict, treatments_dict = {}, {}, {}
    dc = ncfile.variables['data_cube']
    data = dc[:]
    if 'history' in calibration_dict:
        calibration_dict['history'] = eval(ncfile.history)
    for attrib in attrib2netcdf.items():
        if hasattr(dc, attrib[1]):
            value = eval('dc.' + attrib[1])
            if isinstance(value, np.ndarray):
                calibration_dict[attrib[0]] = value[0]
            else:
                calibration_dict[attrib[0]] = value
        else:
            _logger.warning("Warning: the attribute '%s' is not defined in "
                            "the file '%s'", attrib[0], filename)
    for attrib in acquisition2netcdf.items():
        if hasattr(dc, attrib[1]):
            value = eval('dc.' + attrib[1])
            if isinstance(value, np.ndarray):
                acquisition_dict[attrib[0]] = value[0]
            else:
                acquisition_dict[attrib[0]] = value
        else:
            _logger.warning("Warning: the attribute '%s' is not defined in "
                            "the file '%s'", attrib[0], filename)
    for attrib in treatments2netcdf.items():
        if hasattr(dc, attrib[1]):
            treatments_dict[attrib[0]] = eval('dc.' + attrib[1])
        else:
            _logger.warning("Warning: the attribute '%s' is not defined in "
                            "the file '%s'", attrib[0], filename)
    original_metadata = {'record_by': ncfile.type,
                         'calibration': calibration_dict,
                         'acquisition': acquisition_dict,
                         'treatments': treatments_dict}
    ncfile.close()
    # Now we'll map some parameters
    record_by = 'image' if original_metadata[
        'record_by'] == 'image' else 'spectrum'
    if record_by == 'image':
        dim = len(data.shape)
        names = ['Z', 'Y', 'X'][3 - dim:]
        scaleskeys = ['zscale', 'yscale', 'xscale']
        originskeys = ['zorigin', 'yorigin', 'xorigin']
        unitskeys = ['zunits', 'yunits', 'xunits']

    elif record_by == 'spectrum':
        dim = len(data.shape)
        names = ['Y', 'X', 'Energy'][3 - dim:]
        scaleskeys = ['yscale', 'xscale', 'energyscale']
        originskeys = ['yorigin', 'xorigin', 'energyorigin']
        unitskeys = ['yunits', 'xunits', 'energyunits']

    # The images are recorded in the Fortran order
    data = data.T.copy()
    try:
        scales = [calibration_dict[key] for key in scaleskeys[3 - dim:]]
    except KeyError:
        scales = [1, 1, 1][3 - dim:]
    try:
        origins = [calibration_dict[key] for key in originskeys[3 - dim:]]
    except KeyError:
        origins = [0, 0, 0][3 - dim:]
    try:
        units = [calibration_dict[key] for key in unitskeys[3 - dim:]]
    except KeyError:
        units = ['', '', '']
    axes = [
        {
            'size': int(data.shape[i]),
            'index_in_array': i,
            'name': names[i],
            'scale': scales[i],
            'offset': origins[i],
            'units': units[i], }
        for i in range(dim)]
    metadata = {'General': {}, 'Signal': {}}
    metadata['General']['original_filename'] = os.path.split(filename)[1]
    metadata["Signal"]['record_by'] = record_by
    metadata["General"]['signal_type'] = ""
    dictionary = {
        'data': data,
        'axes': axes,
        'metadata': metadata,
        'original_metadata': original_metadata,
    }

    return dictionary
