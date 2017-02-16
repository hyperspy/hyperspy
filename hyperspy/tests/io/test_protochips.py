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

import os

import numpy as np
import pytest

from numpy.testing import assert_allclose

import hyperspy.api as hs
from hyperspy.io_plugins.protochips import ProtochipsCSV, invalid_file_error

testdirpath = os.path.dirname(__file__)
dirpath = os.path.join(testdirpath, 'protochips_data')

# To generate a new reference numpy file
generate_numpy_file = False


def create_numpy_file(filename, obj):
    gen = (obj._data_dictionary[key] for key in obj.logged_quantity_name_list)
    data = np.vstack(gen)
    np.save(filename, data.T)

#######################
# Protochips gas cell #
#######################


def test_read_protochips_gas_cell():
    filename = os.path.join(dirpath, 'protochips_gas_cell.csv')
    s = hs.load(filename)
    assert len(s) == 5
    assert (s[0].metadata.General.title ==
            'Holder Temperature (Degrees C)')
    assert s[0].metadata.Signal.signal_type == ''
    assert s[0].metadata.Signal.quantity == 'Temperature (Degrees C)'
    assert s[1].metadata.General.title == 'Holder Pressure (Torr)'
    assert s[1].metadata.Signal.signal_type == ''
    assert s[1].metadata.Signal.quantity == 'Pressure (Torr)'
    assert s[2].metadata.General.title == 'Tank1 Pressure (Torr)'
    assert s[2].metadata.Signal.signal_type == ''
    assert s[2].metadata.Signal.quantity == 'Pressure (Torr)'
    assert s[3].metadata.General.title == 'Tank2 Pressure (Torr)'
    assert s[3].metadata.Signal.signal_type == ''
    assert s[3].metadata.Signal.quantity == 'Pressure (Torr)'
    assert s[4].metadata.General.title == 'Vacuum Tank Pressure (Torr)'
    assert s[4].metadata.Signal.signal_type == ''
    assert s[4].metadata.Signal.quantity == 'Pressure (Torr)'


def get_datetime(dt):
    dt_np = np.datetime64(dt)
    dt_str = np.datetime_as_string(dt_np)
    date, time = dt_str.split('T')
    return date, time, dt_np


datetime_gas_cell = get_datetime('2014-12-15T19:07:04.165000')
datetime_gas_cell_no_user = get_datetime('2016-10-17T16:59:20.391000')


def test_loading_random_csv_file():
    filename = os.path.join(dirpath, 'random_csv_file.csv')
    with pytest.raises(IOError) as cm:
        ProtochipsCSV(filename)
        cm.match(invalid_file_error)


def test_loading_invalid_protochips_file():
    filename = os.path.join(dirpath, 'invalid_protochips_file.csv')
    with pytest.raises(IOError) as cm:
        hs.load(filename)
        cm.match(invalid_file_error)


class TestProtochipsGasCellCSV():

    def setup_method(self, method):
        filename = os.path.join(dirpath, 'protochips_gas_cell.csv')
        self.s_list = hs.load(filename)

    def test_read_metadata(self):
        date, time, dt_np = datetime_gas_cell
        for s in self.s_list:
            assert s.metadata.General.date == date
            assert s.metadata.General.time == time
            assert s.axes_manager[0].units == 's'
            assert_allclose(s.axes_manager[0].scale, 0.25995, atol=1E-5)
            assert s.axes_manager[0].offset == 0

    def test_read_original_metadata(self):
        om = self.s_list[0].original_metadata.Protochips_header
        assert (om.Calibration_file_path == "The calibration files names"
                " are saved in the 'Original notes' array of the "
                "original metadata.")
        assert om.Holder_Pressure_units == 'Torr'
        assert om.Holder_Temperature_units == 'Degrees C'
        assert om.Start_time == datetime_gas_cell[2]
        assert om.Holder_Pressure_units == 'Torr'
        assert om.Tank1_Pressure_units == 'Torr'
        assert om.Tank2_Pressure_units == 'Torr'
        assert om.Vacuum_Tank_Pressure_units == 'Torr'
        assert om.Time_units == 'Milliseconds'
        assert om.User == 'eric'


class TestProtochipsGasCellCSVNoUser():

    def setup_method(self, method):
        filename = os.path.join(dirpath, 'protochips_gas_cell_no_user.csv')
        self.s_list = hs.load(filename)

    def test_read_metadata(self):
        date, time, dt_np = datetime_gas_cell_no_user
        for s in self.s_list:
            assert s.metadata.General.date == date
            assert s.metadata.General.time == time
            assert s.axes_manager[0].units == 's'
            assert_allclose(s.axes_manager[0].scale, 0.26029, atol=1E-5)
            assert s.axes_manager[0].offset == 0

    def test_read_original_metadata(self):
        om = self.s_list[0].original_metadata.Protochips_header
        assert (om.Calibration_file_path == "The calibration files names"
                        " are saved in the 'Original notes' array of the "
                        "original metadata.")
        assert om.Holder_Pressure_units == 'Torr'
        assert om.Holder_Temperature_units == 'Degrees C'
        assert om.Start_time == datetime_gas_cell_no_user[2]
        assert om.Holder_Pressure_units == 'Torr'
        assert om.Tank1_Pressure_units == 'Torr'
        assert om.Tank2_Pressure_units == 'Torr'
        assert om.Vacuum_Tank_Pressure_units == 'Torr'
        assert om.Time_units == 'Milliseconds'


class TestProtochipsGasCellCSVReader():

    def setup_method(self, method):
        self.filename = os.path.join(dirpath, 'protochips_gas_cell.csv')
        self.pgc = ProtochipsCSV(self.filename)
        if generate_numpy_file:
            create_numpy_file(self.filename.replace('.csv', '.npy'), self.pgc)

    def test_read_column_name(self):
        assert self.pgc.column_name == ['Time', 'Notes',
                                        'Holder Temperature',
                                        'Holder Pressure',
                                        'Tank1 Pressure',
                                        'Tank2 Pressure',
                                        'Vacuum Tank Pressure']

    def test_read_start_datetime(self):
        assert self.pgc.start_datetime == datetime_gas_cell[2]

    def test_read_data(self):
        gen = (self.pgc._data_dictionary[key]
               for key in self.pgc.logged_quantity_name_list)
        data = np.vstack(gen)
        expected_data = np.load(os.path.join(
            dirpath, 'protochips_gas_cell.npy'))
        np.testing.assert_allclose(data.T, expected_data)

    def test_read_metadata_header(self):
        assert self.pgc.time_units == 'Milliseconds'
        assert self.pgc.time_units == 'Milliseconds'
        assert self.pgc.temperature_units == 'Degrees C'
        assert self.pgc.pressure_units == 'Torr'
        assert self.pgc.current_units == 'Amps'
        assert self.pgc.voltage_units == 'Volts'
        assert self.pgc.resistance_units == 'Ohms'
        assert self.pgc.user == 'eric'


#########################
# Protochips electrical #
#########################

def test_read_protochips_electrical():
    filename = os.path.join(dirpath, 'protochips_electrical.csv')
    s = hs.load(filename)
    assert len(s) == 6
    assert s[0].metadata.General.title == 'Channel A Current (Amps)'
    assert s[0].metadata.Signal.signal_type == ''
    assert s[0].metadata.Signal.quantity == 'Current (Amps)'
    assert s[1].metadata.General.title == 'Channel A Voltage (Volts)'
    assert s[1].metadata.Signal.signal_type == ''
    assert s[1].metadata.Signal.quantity == 'Voltage (Volts)'
    assert s[2].metadata.General.title == 'Channel A Resistance (Ohms)'
    assert s[2].metadata.Signal.signal_type == ''
    assert s[2].metadata.Signal.quantity == 'Resistance (Ohms)'
    assert s[3].metadata.General.title == 'Channel B Current (Amps)'
    assert s[3].metadata.Signal.signal_type == ''
    assert s[3].metadata.Signal.quantity == 'Current (Amps)'
    assert s[4].metadata.General.title == 'Channel B Voltage (Volts)'
    assert s[4].metadata.Signal.signal_type == ''
    assert s[4].metadata.Signal.quantity == 'Voltage (Volts)'
    assert s[5].metadata.General.title == 'Channel B Resistance (Ohms)'
    assert s[5].metadata.Signal.signal_type == ''
    assert s[5].metadata.Signal.quantity == 'Resistance (Ohms)'


class TestProtochipsElectricalCSVReader():

    def setup_method(self, method):
        self.filename = os.path.join(dirpath, 'protochips_electrical.csv')
        self.pa = ProtochipsCSV(self.filename)
        if generate_numpy_file:
            create_numpy_file(self.filename.replace('.csv', '.npy'), self.pa)

    def test_read_column_name(self):
        assert self.pa.column_name == ['Time', 'Notes',
                                       'Channel A Current',
                                       'Channel A Voltage',
                                       'Channel A Resistance',
                                       'Channel B Current',
                                       'Channel B Voltage',
                                       'Channel B Resistance']

    def test_read_start_datetime(self):
        dt = np.datetime64('2014-10-08T16:26:51.738000')
        assert self.pa.start_datetime == dt

    def test_read_data(self):
        gen = (self.pa._data_dictionary[key]
               for key in self.pa.logged_quantity_name_list)
        data = np.vstack(gen)
        expected_data = np.load(os.path.join(
            dirpath, 'protochips_electrical.npy'))
        np.testing.assert_allclose(data.T, expected_data)


######################
# Protochips thermal #
######################

def test_read_protochips_thermal():
    filename = os.path.join(dirpath, 'protochips_thermal.csv')
    s = hs.load(filename)
    assert (s.metadata.General.title ==
            'Channel A Temperature (Degrees C)')
    assert s.metadata.Signal.signal_type == ''
    assert s.metadata.Signal.quantity == 'Temperature (Degrees C)'
    assert (s.metadata.General.notes ==
            'Calibration file name: AD21013_8.cal')


class TestProtochipsThermallCSVReader():

    def setup_method(self, method):
        self.filename = os.path.join(dirpath, 'protochips_thermal.csv')
        self.pt = ProtochipsCSV(self.filename)
        if generate_numpy_file:
            create_numpy_file(self.filename.replace('.csv', '.npy'), self.pt)

    def test_read_column_name(self):
        assert self.pt.column_name == [
            'Time', 'Notes', 'Channel A Temperature']

    def test_read_start_datetime(self):
        dt = np.datetime64('2014-12-03T17:15:37.192000')
        np.testing.assert_equal(self.pt.start_datetime, dt)

    def test_read_data(self):
        gen = (self.pt._data_dictionary[key]
               for key in self.pt.logged_quantity_name_list)
        data = np.vstack(gen)
        expected_data = np.load(os.path.join(
            dirpath, 'protochips_thermal.npy'))
        np.testing.assert_allclose(data.T, expected_data)


#############################
# Protochips electrothermal #
#############################

def test_read_protochips_electrothermal():
    filename = os.path.join(dirpath, 'protochips_electrothermal.csv')
    s = hs.load(filename)
    assert len(s) == 4
    assert (s[0].metadata.General.title ==
            'Channel A Temperature (Degrees C)')
    assert s[0].metadata.Signal.signal_type == ''
    assert s[0].metadata.Signal.quantity == 'Temperature (Degrees C)'
    assert s[1].metadata.General.title == 'Channel B Current (Amps)'
    assert s[1].metadata.Signal.signal_type == ''
    assert (s[0].metadata.General.notes ==
            'Calibration file name: AD21018_4.cal')
    assert s[1].metadata.Signal.quantity == 'Current (Amps)'
    assert s[2].metadata.General.title == 'Channel B Voltage (Volts)'
    assert s[2].metadata.Signal.signal_type == ''
    assert s[2].metadata.Signal.quantity == 'Voltage (Volts)'
    assert s[3].metadata.General.title == 'Channel B Resistance (Ohms)'
    assert s[3].metadata.Signal.signal_type == ''
    assert s[3].metadata.Signal.quantity == 'Resistance (Ohms)'


class TestProtochipsElectrothermalCSVReader():

    def setup_method(self, method):
        self.filename = os.path.join(dirpath, 'protochips_electrothermal.csv')
        self.pet = ProtochipsCSV(self.filename)
        if generate_numpy_file:
            create_numpy_file(self.filename.replace('.csv', '.npy'), self.pet)

    def test_read_column_name(self):
        assert self.pet.column_name == ['Time', 'Notes',
                                        'Channel A Temperature',
                                        'Channel B Current',
                                        'Channel B Voltage',
                                        'Channel B Resistance']

    def test_read_start_datetime(self):
        dt = np.datetime64('2014-11-05T14:42:51.369000')
        np.testing.assert_equal(self.pet.start_datetime, dt)

    def test_read_data(self):
        gen = (self.pet._data_dictionary[key]
               for key in self.pet.logged_quantity_name_list)
        data = np.vstack(gen)
        expected_data = np.load(os.path.join(dirpath,
                                             self.filename.replace('.csv', '.npy')))
        np.testing.assert_allclose(data.T, expected_data)
