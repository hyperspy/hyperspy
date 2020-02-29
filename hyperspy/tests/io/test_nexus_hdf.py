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

import os.path
from os import remove
import sys
import gc
import tempfile
import numpy as np
import dask
import pytest
from distutils.version import LooseVersion
from hyperspy.io import load
import hyperspy.api as hs
from hyperspy.signal import BaseSignal
from hyperspy._signals.signal1d import Signal1D


dirpath = os.path.dirname(__file__)

file1 = os.path.join(dirpath, 'nexus_files', 'file1.nxs')
file2 = os.path.join(dirpath, 'nexus_files', 'file2.nxs')

my_path = os.path.dirname(__file__)


#
# Test you can read a hyperspy hdf file and
# capture the original metadata
#
class TestExample1_12():

    def setup_method(self, method):
        self.s = load(os.path.join(
            my_path,
            "hdf5_files","example1_v1.2.hdf5"),preferred_format="Nexus")

    def test_filename(self):
        assert (
            self.s.original_metadata.Experiments.NIO_EELS_OK_SHELL.\
                metadata.General.original_filename
                == "example1.msa" )

    def test_thickness(self):
        assert self.s.original_metadata.Experiments.NIO_EELS_OK_SHELL.\
                metadata.Sample.thickness == 50.0


class TestExample1_10():

    def setup_method(self, method):
        self.s = load(os.path.join(
            my_path,
            "hdf5_files",
            "example1_v1.0.hdf5"),preferred_format="Nexus")


class TestExample1_11():

    def setup_method(self, method):
        self.s = load(os.path.join(
            my_path,
            "hdf5_files",
            "example1_v1.1.hdf5"),preferred_format="Nexus")


@pytest.fixture()
def tmpfilepath():
    with tempfile.TemporaryDirectory() as tmp:
        yield os.path.join(tmp, "test.nxs")
        gc.collect()        # Make sure any memmaps are closed first!


class TestSavingMetadataContainers:

    def setup_method(self, method):
        self.s = BaseSignal([0.1,0.2,0.3])

    def test_save_unicode(self, tmpfilepath):
        s = self.s
        s.metadata.set_item('test1',44.0)
        s.metadata.set_item('test2',54.0)
        s.metadata.set_item('test3',64.0)        
        s.save(tmpfilepath)
        l = load(tmpfilepath)
        print(l.original_metadata)
        assert isinstance(l.original_metadata.hyperspy_metadata.test1, float)
        assert isinstance(l.original_metadata.hyperspy_metadata.test1, float)
        assert isinstance(l.original_metadata.hyperspy_metadata.test2, float)
        assert l.original_metadata.hyperspy_metadata.test2 == 54.0


    def test_general_metadata(self, tmpfilepath):
        s = self.s
        notes = "Dummy notes"
        authors = "Author 1, Author 2"
        doi = "doi"
        s.metadata.General.notes = notes
        s.metadata.General.authors = authors
        s.metadata.General.doi = doi
        #print(s.metadata)
        s.save(tmpfilepath)
        l = load(tmpfilepath)
        assert l.original_metadata.hyperspy_metadata.General.notes == notes
        assert l.original_metadata.hyperspy_metadata.General.authors == authors
        assert l.original_metadata.hyperspy_metadata.General.doi == doi

#
# read 3 varieties of nexus file
#
def test_read1():
    s = hs.load(file1)
    assert s[0].metadata.General.title == "/entry/xsp3_addetector"
    assert len(s) == 4

def test_read1_search_keys():
    # should only find 2 data sets
    s = hs.load(file1,dset_search_keys=["xsp3"])
    assert len(s) == 2

def test_read2():
    s = hs.load(file2)
    assert len(s) == 21
    
def test_read2_data_search_keys():
    # should only find 2 data sets
    s = hs.load(file2,dset_search_keys=["Fe"])
    assert len(s) == 1
            
def test_read2_meta_search_keys():
    # should only find 2 data sets
    s = hs.load(file1,meta_search_keys=["dcm_energy"])
    assert s[0].original_metadata.entry.instrument.beamline.\
              DCM.dcm_energy.value == 17.256800000000002
    assert "M1" not in s[0].original_metadata.entry.instrument.beamline

def test_preffered_format_rgba16():
    s = load(os.path.join(
        my_path,
        "hdf5_files",
        "test_rgba16.hdf5"),preferred_format="Nexus")
    data = np.load(os.path.join(
        my_path,
        "npy_files",
        "test_rgba16.npy"))
    assert (s.data == data).all()


