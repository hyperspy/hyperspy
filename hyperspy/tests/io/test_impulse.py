# -*- coding: utf-8 -*-
# Copyright 2007-2021 The HyperSpy developers
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

import hyperspy.api as hs
from hyperspy.io_plugins.impulse import (
    impulseCSV,
    invalid_file_error,
    invalid_filenaming_error,
    no_metadata_file_error,
)

testdirpath = os.path.dirname(__file__)
dirpath = os.path.join(testdirpath, "impulse_data")


# Load a synchronized data log file


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


class testSyncFileCSVreader:
    def setup_method(self, method):
        self.filename = os.path.join(dirpath, "StubExperiment_Synchronized data.csv")
        self.isf = impulseCSV(self.filename)

    def test_read_column_name(self):
        assert self.isf.column_name == [
            "Temperature Measured",
            "MixValve",
            "Pin Measured",
            "Pout Measured",
            "Pnr Measured",
            "Fnr,% Gas1 Measured",
            "% Gas2 Measured",
            "% Gas3 Measured",
            "Channel#1",
            "Channel#2",
            "Channel#3",
            "Channel#4",
        ]

    def test_read_start_datetime(self):
        assert self.isf.start_datetime == "2021-08-06T13:40:19.000000"

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
        impulseCSV(filename)
        cm.match(invalid_file_error)


# Loading a csv file with an incorrect filename


def test_loading_invalid_impulse_filename():
    filename = os.path.join(dirpath, "changed_file_name.csv")
    with pytest.raises(IOError) as cm:
        hs.load(filename, reader="impulse")
        cm.match(invalid_filenaming_error)


# Loading a csv file without a corresponding metadata file


def test_loading_file_without_metadata():
    filename = os.path.join(dirpath, "NoMetadata_Synchronized data.csv")
    with pytest.raises(IOError) as cm:
        hs.load(filename, reader="impulse")
        cm.match(no_metadata_file_error)


# Test raw data file


class testRawFile:
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
        self.isf = impulseCSV(self.filename)

    def test_read_column_name(self):
        assert self.isf.column_name == ["Temperature Measured"]

    def test_read_start_datetime(self):
        assert self.isf.start_datetime == "2021-08-06T13:40:19.000000"
