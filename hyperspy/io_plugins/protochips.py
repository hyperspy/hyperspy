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

    def get_dictionary(self, quantity):
        return {'data': self._data_dictionary[quantity],
                'axes': self._get_axes(),
                'metadata': self._get_metadata(quantity),
                'original_metadata': {'Protochips_header': self._get_original_metadata()},
                'mapping': self._get_mapping()}

    def _get_original_metadata(self):
        return {'Start time':self.start_datetime,
                self.time_unit:self.time_unit,
                self.temperature_unit:self.temperature_unit,
                self.pressure_unit:self.pressure_unit,
                self.current_unit:self.current_unit,
                self.voltage_unit:self.voltage_unit,
                self.resistance_unit:self.resistance_unit,
                self.user:self.user}

    def _get_mapping(self):
        return {'Protochips_header.date':("General.time", None),}

    def _get_metadata(self, quantity):
        return {'General': {'original_filename': os.path.split(self.filename)[1],
                            'title':quantity+' ('+self._get_quantity_unit(quantity)+')',
                            'user':self.user,
                            'start_time':self.start_datetime},
                "Signal": {'signal_type':quantity}}

    def _read_data(self, header_line_number):
        usecols = [i for i in range(len(self.column_name))]
        usecols.pop(1)
        data = np.loadtxt(self.filename, delimiter=',',
                          skiprows=header_line_number, 
                          unpack=True, usecols=usecols)
        self.time = data[0]
        
        keys = self.column_name[:]
        keys.remove(' Notes')
        self._data_dictionary = dict(zip(keys, data))

    def _get_axes(self):
        scale = np.diff(self.time).mean()
        units = 's'
        offset = 0
        if self.time_unit == 'Milliseconds':
            scale = scale/1000            
        else:
            warnings.warn("Time unit not recognised, assuming second.")

        return [{'size': self.time.shape[0],
                 'index_in_array': 0,
                 'name': 'Time',
                 'scale': scale,
                 'offset': offset,
                 'units': units,
                 'navigate': False
                 }]

    def _get_quantity_unit(self, quantity):
        if 'Temperature' in quantity:
            return self.temperature_unit
        if 'Pressure' in quantity:
            return self.pressure_unit
        if 'Current' in quantity:
            return self.current_unit
        if 'Voltage' in quantity:
            return self.voltage_unit
        if 'Resistance' in quantity:
            return self.resistance_unit            

    def _read_header(self, header_line_number):
        with open(self.filename, 'r') as f:
            raw_header = [f.readline() for i in range(header_line_number)]
        return raw_header

    def _read_all_metadata_header(self):
        # Read all metadata from the header
        self.start_datetime = self._read_start_datetime()
        self.time_unit = self._read_metadata_header(self.raw_header[3])
        self.temperature_unit = self._read_metadata_header(self.raw_header[4])
        self.pressure_unit = self._read_metadata_header(self.raw_header[5])
        self.current_unit = self._read_metadata_header(self.raw_header[6])
        self.voltage_unit = self._read_metadata_header(self.raw_header[7])
        self.resistance_unit = self._read_metadata_header(self.raw_header[8])
        self.user = self._read_metadata_header(self.raw_header[9])  

    def _read_metadata_header(self, line):
        return line.split(',')[1].split('= ')[1]

    def _read_column_name(self):
        string = self.raw_header[0]
        for st in ['\r', '\n']:
            string = string.replace(st, '')
        return string.split(',')

    def _read_start_datetime(self):
        date = self._read_metadata_header(self.raw_header[1])
        time = self._read_metadata_header(self.raw_header[2])
        return datetime.strptime(date+time, "%Y.%m.%d%H:%M:%S.%f")

if __name__ == '__main__':
    filename = os.path.join('protochips_gas_cell.csv')
    p = ProtochipsCSV(filename=filename)
    import hyperspy.api as hs
    hs.plot.plot_spectra(p)