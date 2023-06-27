# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy.  If not, see <https://www.gnu.org/licenses/#GPL>.

import os

import numpy as np
import pytest

import hyperspy.api as hs
from hyperspy.io_plugins.impulse import (
    ImpulseCSV,
    invalid_file_error,
    invalid_filenaming_error
)

testdirpath = os.path.dirname(__file__)
dirpath = os.path.join(testdirpath, "impulse_data")


# Load a synchronized data log file

def test_read_sync_file():
    filename = os.path.join(dirpath, 'StubExperiment_Synchronized data.csv')
    s = hs.load(filename, reader='impulse')
    assert len(s) == 13
    assert (s[0].metadata.General.title ==
            'Temperature Measured (degC)')
    assert s[0].metadata.Signal.quantity == 'degC'
    assert s[1].metadata.General.title == 'MixValve'
    assert s[1].metadata.Signal.quantity == ''
    assert s[2].metadata.General.title == 'Pin Measured'
    assert s[2].metadata.Signal.quantity == ''
    assert s[3].metadata.General.title == 'Pout Measured'
    assert s[3].metadata.Signal.quantity == ''
    assert s[4].metadata.General.title == 'Pnr Measured'
    assert s[4].metadata.Signal.quantity == ''
    assert s[5].metadata.General.title == 'Fnr'
    assert s[5].metadata.Signal.quantity == ''
    assert s[6].metadata.General.title == '% Gas1 Measured'
    assert s[6].metadata.Signal.quantity == ''
    assert s[7].metadata.General.title == '% Gas2 Measured'
    assert s[7].metadata.Signal.quantity == ''
    assert s[8].metadata.General.title == '% Gas3 Measured'
    assert s[8].metadata.Signal.quantity == ''
    assert s[9].metadata.General.title == 'Channel#1'
    assert s[9].metadata.Signal.quantity == ''
    assert s[10].metadata.General.title == 'Channel#2'
    assert s[10].metadata.Signal.quantity == ''
    assert s[11].metadata.General.title == 'Channel#3'
    assert s[11].metadata.Signal.quantity == ''
    assert s[12].metadata.General.title == 'Channel#4'
    assert s[12].metadata.Signal.quantity == ''
    

class testSyncFile:
    def setup_method(self, method):
        filename = os.path.join(dirpath, "StubExperiment_Synchronized data.csv")
        self.s_list = hs.load(filename, reader="impulse")

    def test_read_metadata(self):
        for s in self.s_list:
            assert s.metadata.General.date == "06-08-2021"
            assert s.metadata.General.time == "13:40:19"
            assert s.axes_manager[0].units == "s"
            np.testing.assert_allclose(
                s.axes_manager[0].scale, 0.5114766839378239, atol=1e-5
            )
            assert s.axes_manager[0].offset == 0

    def test_read_original_metadata(self):
        om = self.s_list[0].original_metadata.Impulse_header
        assert om.Calibration_value == 0.25
        assert om.Description == "Experiment with stub data"
        assert om.Experiment_date == "06-08-2021"
        assert om.Experiment_duration == "00.00:06:45.416"
        assert om.Experiment_time == "13:40:19"
        assert om.Gas1 == "H2"
        assert om.Gas2 == "CH4"
        assert om.Gas3 == "Ar"
        assert om.MassSpec_File_Location == ""
        assert om.MassSpec_channel_1 == "28 (Nitrogen)"
        assert om.MassSpec_channel_2 == "32 (Oxygen)"
        assert om.MassSpec_channel_3 == "18 (Water)"
        assert om.MassSpec_channel_4 == "44 (Carbon dioxide)"
        assert om.Max_temperature == 1100
        assert om.Notes == [
            "13:46:46: Live note one",
            "13:46:52: Live note two",
            "13:46:56: Live note three",
        ]
        assert om.Room_temperature == 21
        assert om.Sample == "Gold"
        assert om.System_version == "G+"
    
    def test_read_data(self):
        expected_data = np.load(
            os.path.join(dirpath, "StubExperiment_Synchronized data.npy")
        )
        np.testing.assert_allclose(self.s_list.T, expected_data)


class testSyncFileCSVreader:
    def setup_method(self, method):
        self.filename = os.path.join(dirpath, "StubExperiment_Synchronized data.csv")
        self.isf = ImpulseCSV(self.filename)

    def test_read_column_name(self):
        assert self.isf.column_name == [
            "Temperature Measured",
            "MixValve",
            "Pin Measured",
            "Pout Measured",
            "Pnr Measured",
            "Fnr",
            "% Gas1 Measured",
            "% Gas2 Measured",
            "% Gas3 Measured",
            "Channel#1",
            "Channel#2",
            "Channel#3",
            "Channel#4",
        ]


    def test_read_data(self):
        dicts = (
            self.isf._data_dictionary[key] for key in self.isf.logged_quantity_name_list
        )
        data = np.vstack(list(dicts))
        expected_data = np.load(
            os.path.join(dirpath, "StubExperiment_Synchronized data.npy")
        )
        np.testing.assert_allclose(data.T, expected_data)





# Loading a random csv file


def test_loading_random_csv_file():
    filename = os.path.join(dirpath, "random_csv_file.csv")
    with pytest.raises(IOError) as cm:
        ImpulseCSV(filename)
        cm.match(invalid_file_error)


# Loading a csv file with an incorrect filename


def test_loading_invalid_impulse_filename():
    filename = os.path.join(dirpath, "changed_file_name.csv")
    with pytest.raises(IOError) as cm:
        hs.load(filename, reader="impulse")
        cm.match(invalid_filenaming_error)


# Test raw data file


class testRawFile():
    def setup_method(self, method):
        filename = os.path.join(dirpath, "StubExperiment_Heat raw.csv")
        self.s_list = hs.load(filename, reader="impulse")

    def test_read_metadata(self):
        for s in self.s_list:
            assert s.metadata.General.date == "06-08-2021"
            assert s.metadata.General.time == "13:40:19"
            assert s.axes_manager[0].units == "s"
            np.testing.assert_allclose(
                s.axes_manager[0].scale, 0.5114766839378239, atol=1e-5
            )
            assert s.axes_manager[0].offset == 0

    def test_read_original_metadata(self):
        om = self.s_list[0].original_metadata.Impulse_header
        assert om.Calibration_value == 0.25
        assert om.Description == "Experiment with stub data"
        assert om.Experiment_date == "06-08-2021"
        assert om.Experiment_duration == "00.00:06:45.416"
        assert om.Experiment_time == "13:40:19"
        assert om.Gas1 == "H2"
        assert om.Gas2 == "CH4"
        assert om.Gas3 == "Ar"
        assert om.MassSpec_File_Location == ""
        assert om.MassSpec_channel_1 == "28 (Nitrogen)"
        assert om.MassSpec_channel_2 == "32 (Oxygen)"
        assert om.MassSpec_channel_3 == "18 (Water)"
        assert om.MassSpec_channel_4 == "44 (Carbon dioxide)"
        assert om.Max_temperature == 1100
        assert om.Notes == [
            "13:46:46: Live note one",
            "13:46:52: Live note two",
            "13:46:56: Live note three",
        ]
        assert om.Room_temperature == 21
        assert om.Sample == "Gold"
        assert om.System_version == "G+"


class testRawFileCSVreader:
    def setup_method(self, method):
        self.filename = os.path.join(dirpath, "StubExperiment_Heat raw.csv")
        self.isf = ImpulseCSV(self.filename)

    def test_read_column_name(self):
        assert self.isf.column_name == ["Temperature Measured"]

    def test_read_start_datetime(self):
        assert self.isf.start_datetime == "2021-08-06T13:40:19.000000"
