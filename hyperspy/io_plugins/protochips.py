# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy.  If not, see <http://www.gnu.org/licenses/>.


import numpy as np
import os
from datetime import datetime
import warnings

# Plugin characteristics
# ----------------------
format_name = 'Protochips'
description = 'Reads Protochips log files (heating/baising and gas cell)'
full_support = False
# Recognised file extension
file_extensions = ['csv', 'CSV']
default_extension = 0
# Reading capabilities
reads_images = False
reads_spectrum = False
reads_spectrum_image = False
# Writing capabilities
writes = False

def file_reader(filename, *args, **kwds):
    csv_file = ProtochipsCSV(filename)
    return protochips_log_reader(csv_file)

def protochips_log_reader(csv_file):
    csvs = []
    for key in csv_file.logged_quantity_name_list:
        csvs.append(csv_file.get_dictionary(key))
    return csvs

class ProtochipsCSV(object):
    def __init__(self, filename, header_line_number=10):
        self.filename = filename
        self.raw_header = self._read_header(header_line_number)
        self.column_name = self._read_column_name()
        self._read_all_metadata_header()
        self.logged_quantity_name_list = self.column_name[2:]

        self._read_data(header_line_number)
        
        self.calibration_file = None

    def get_dictionary(self, quantity):
        return {'data': self._data_dictionary[quantity],
                'axes': self._get_axes(),
                'metadata': self._get_metadata(quantity),
                'original_metadata': {'Protochips_header':
                    self._get_original_metadata()},
                'mapping': self._get_mapping()}

    def _get_original_metadata(self):
        d = {'Start time':self.start_datetime}
        d['Time units'] = self.time_units
        for quantity in self.logged_quantity_name_list:
            d['%s_units'%quantity] = self._get_quantity_units(quantity)
        d['User'] = self.user
        d['Calibration file'] = self._parse_calibration_file()
        return d

    def _get_mapping(self):
        return {'Protochips_header.date':("General.time", None),}

    def _get_metadata(self, quantity):
        return {'General': {'original_filename': os.path.split(self.filename)[1],
                            'title': '%s (%s)'%(quantity,
                                self._get_quantity_units(quantity)),
                            'user': self.user,
                            'start_time': self.start_datetime,
                            'notes': self._parse_notes(),
                            'calibration_file':self._parse_calibration_file()},
                "Signal": {'signal_type': quantity,
                           'time_axis': self._get_metadata_time_axis()}}

    def _get_metadata_time_axis(self):
        return {'value':self.time_axis,
                'units':self.time_units}
                           
    def _read_data(self, header_line_number):
        names = [name.replace(' ', '_') for name in self.column_name]
        data = np.genfromtxt(self.filename, delimiter=',', dtype=None,
                             names=names,
                             skip_header=header_line_number,
                             unpack=True)

        self._data_dictionary = dict()
        for i, name, name_dtype in zip(range(len(names)), self.column_name,
                                       names):
            if name == 'Notes':
                self.notes = data[name_dtype].astype(str)
            elif name == 'Time':
                self.time_axis = data[name_dtype]
            else:
                self._data_dictionary[name] = data[name_dtype]

    def _parse_notes(self):
        arr = np.vstack((self.time_axis, self.notes))
        return np.compress(arr[1] != '', arr, axis=1)

    def _parse_calibration_file(self):
         # for the gas cell, the calibration is saved in the notes colunm
        if self.calibration_file is None:
            calibration_file = "The calibration files are saved in "\
                "metadata.General.notes"
        else:
            calibration_file = self.calibration_file            
        return calibration_file
        
    def _get_axes(self):
        scale = np.diff(self.time_axis).mean()
        units = 's'
        offset = 0
        if self.time_units == 'Milliseconds':
            scale = scale/1000            
        else:
            warnings.warn("Time units not recognised, assuming second.")

        return [{'size': self.time_axis.shape[0],
                 'index_in_array': 0,
                 'name': 'Time',
                 'scale': scale,
                 'offset': offset,
                 'units': units,
                 'navigate': False
                 }]

    def _get_quantity_units(self, quantity):
        quantity = quantity.split(' ')[-1].lower()
        return self.__dict__['%s_units'%quantity]

    def _read_header(self, header_line_number):
        with open(self.filename, 'r') as f:
            raw_header = [f.readline() for i in range(header_line_number)]
        return raw_header

    def _read_all_metadata_header(self):
        i = 1
        param, value = self._parse_metadata_header(self.raw_header[i])
        while 'User' not in param: # user should be the last of the header
            if 'Calibration file' in param:
                self.calibration_file = value
            elif 'Date (yyyy.mm.dd)' in param:
                date = value
            elif 'Time (hh:mm:ss.ms)' in param:
                time = value
            else:
                attr_name = param.replace(' ', '_').lower()
                self.__dict__[attr_name] = value
            i += 1
            param, value = self._parse_metadata_header(self.raw_header[i])

        self.user = self._parse_metadata_header(self.raw_header[i])[1]
        self.start_datetime = datetime.strptime(date+time,
                                                "%Y.%m.%d%H:%M:%S.%f")

    def _parse_metadata_header(self, line):
        return line.replace(', ', ',').split(',')[1].split(' = ')

    def _read_column_name(self):
        string = self.raw_header[0]
        return string.replace(', ', ',').replace('\n', '').split(',')

if __name__ == '__main__':
    filename = os.path.join('protochips_gas_cell.csv')
    p = ProtochipsCSV(filename=filename)
    import hyperspy.api as hs
    s = hs.load('protochips_gas_cell.csv')
    a = s[0]