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
from datetime import datetime as dt
import warnings
import logging
from distutils.version import LooseVersion


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

_logger = logging.getLogger(__name__)


# At some point, if there is another readerw, whith also use csv file, it will
# be necessary to mention the other reader in this message (and to add an
# argument in the load function to specify the correct reader)
invalid_file_error = "The Protochips csv reader can't import the file, please"\
    " make sure, this is a valid Protochips log file."


def file_reader(filename, *args, **kwds):
    csv_file = ProtochipsCSV(filename)
    return _protochips_log_reader(csv_file)


def _protochips_log_reader(csv_file):
    csvs = []
    for key in csv_file.logged_quantity_name_list:
        try:
            csvs.append(csv_file.get_dictionary(key))
        except BaseException:
            raise IOError(invalid_file_error)
    return csvs


class ProtochipsCSV(object):

    def __init__(self, filename, ):
        self.filename = filename
        self._parse_header()
        self._read_data()

    def _parse_header(self):
        with open(self.filename, 'r') as f:
            s = f.readline()
            self.column_name = s.replace(', ', ',').replace('\n', '').split(',')
            if not self._is_protochips_csv_file():
                raise IOError(invalid_file_error)
            self._read_all_metadata_header(f)
        self.logged_quantity_name_list = self.column_name[2:]

    def _is_protochips_csv_file(self):
        # This check is not great, but it's better than nothing...
        if 'Time' in self.column_name and 'Notes' in self.column_name and len(
                self.column_name) >= 3:
            return True
        else:
            return False

    def get_dictionary(self, quantity):
        return {'data': self._data_dictionary[quantity],
                'axes': self._get_axes(),
                'metadata': self._get_metadata(quantity),
                'mapping': self._get_mapping(),
                'original_metadata': {'Protochips_header':
                                      self._get_original_metadata()}}

    def _get_original_metadata(self):
        d = {'Start time': self.start_datetime}
        d['Time units'] = self.time_units
        for quantity in self.logged_quantity_name_list:
            d['%s_units' % quantity] = self._parse_quantity_units(quantity)
        if self.user:
            d['User'] = self.user
        d['Calibration file path'] = self._parse_calibration_filepath()
        d['Time axis'] = self._get_metadata_time_axis()
        # Add the notes here, because there are not well formatted enough to
        # go in metadata
        d['Original notes'] = self._parse_notes()
        return d

    def _get_metadata(self, quantity):
        date, time = np.datetime_as_string(self.start_datetime).split('T')
        return {'General': {'original_filename': os.path.split(self.filename)[1],
                            'title': '%s (%s)' % (quantity,
                                                  self._parse_quantity_units(quantity)),
                            'date': date,
                            'time': time},
                "Signal": {'signal_type': '',
                           'quantity': self._parse_quantity(quantity)}}

    def _get_mapping(self):
        mapping = {
            "Protochips_header.Calibration file path": (
                "General.notes",
                self._parse_calibration_file_name),
            "Protochips_header.User": (
                "General.authors",
                None),
        }
        return mapping

    def _get_metadata_time_axis(self):
        return {'value': self.time_axis,
                'units': self.time_units}

    def _read_data(self):
        names = [name.replace(' ', '_') for name in self.column_name]
        # Necessary for numpy >= 1.14
        kwargs = {'encoding': 'latin1'} if np.__version__ >= LooseVersion("1.14") else {
        }
        data = np.genfromtxt(self.filename, delimiter=',', dtype=None,
                             names=names,
                             skip_header=self.header_last_line_number,
                             unpack=True, **kwargs)

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

    def _parse_calibration_filepath(self):
         # for the gas cell, the calibration is saved in the notes colunm
        if hasattr(self, "calibration_file"):
            calibration_file = self.calibration_file
        else:
            calibration_file = "The calibration files names are saved in the"\
                " 'Original notes' array of the original metadata."
        return calibration_file

    def _parse_calibration_file_name(self, path):
        basename = os.path.basename(path)
        return "Calibration file name: %s" % basename.split('\\')[-1]

    def _get_axes(self):
        scale = np.diff(self.time_axis[1:-1]).mean()
        max_diff = np.diff(self.time_axis[1:-1]).max()
        units = 's'
        offset = 0
        if self.time_units == 'Milliseconds':
            scale /= 1000
            max_diff /= 1000
            # Once we support non-linear axis, don't forgot to update the
            # documentation of the protochips reader
            _logger.warning("The time axis is not linear, the time step is "
                            "thus extrapolated to {0} {1}. The maximal step in time step is {2} {1}".format(
                                scale, units, max_diff))
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

    def _parse_quantity(self, quantity):
        quantity_name = quantity.split(' ')[-1]
        return '%s (%s)' % (quantity_name,
                            self._parse_quantity_units(quantity))

    def _parse_quantity_units(self, quantity):
        quantity = quantity.split(' ')[-1].lower()
        return self.__dict__['%s_units' % quantity]

    def _read_all_metadata_header(self, f):
        param, value = self._parse_metadata_header(f.readline())
        i = 2
        while 'User' not in param:  # user should be the last of the header
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
            try:
                param, value = self._parse_metadata_header(f.readline())
            except ValueError:
                # when the last line of header does not contain 'User',
                # possibly some old file.
                self.user = None
                break
            except IndexError:
                _logger.warning("The metadata may not be parsed properly.")
                break
        else:
            self.user = value
        self.header_last_line_number = i
        self.start_datetime = np.datetime64(dt.strptime(date + time,
                                                        "%Y.%m.%d%H:%M:%S.%f"))

    def _parse_metadata_header(self, line):
        return line.replace(', ', ',').split(',')[1].split(' = ')
