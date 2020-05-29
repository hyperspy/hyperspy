# -*- coding: utf-8 -*-
# Copyright 2007-2020 The HyperSpy developers
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
import gc
import tempfile
import numpy as np
import pytest
from hyperspy.io import load
import hyperspy.api as hs
from hyperspy.signal import BaseSignal


dirpath = os.path.dirname(__file__)

file1 = os.path.join(dirpath, 'nexus_files', 'single_nxdata.nxs')
file2 = os.path.join(dirpath, 'nexus_files', 'multi_signal.nxs')
file3 = os.path.join(dirpath, 'nexus_files', 'dls_nexus.nxs')


my_path = os.path.dirname(__file__)


#
# Test nexus loading..external dls file
#
class TestDLSNexus():

    def setup_method(self, method):
        self.s = load(file3)
   
    def test_string(self):
        assert self.s.original_metadata.entry.instrument.\
            beamline.M1.m1_y.attrs.units == "mm"

    def test_value(self):
        assert self.s.original_metadata.entry.instrument.\
            beamline.M1.m1_y.value == -4.0

    def test_class(self):
        assert self.s.original_metadata.entry.instrument.\
            beamline.M1.attrs.NX_class == "NXmirror"

    def test_string_array(self):
        np.testing.assert_array_equal(self.s.original_metadata.entry.testdata.nexustest.attrs.axes\
            ,np.array([b'y',b'x',b'.']))

    def test_signal_loaded(self):
        assert self.s.metadata.Signal.signal_type == "Signal1D"


    def test_axes_names(self):
        assert self.s.axes_manager[0].name == "x"
        assert self.s.axes_manager[1].name == "y"
        

#
# Test nexus loading..external dls file
#
class TestSavedSignalLoad():

    def setup_method(self, method):
        self.s = load(file1)
   
    def test_string(self):
        assert self.s.original_metadata.instrument.\
            energy.attrs.units == "keV"

    def test_value(self):
        assert self.s.original_metadata.instrument.\
            energy.value == 12.0

    def test_string_array(self):
        np.testing.assert_array_equal(self.s.original_metadata.instrument.energy.attrs.test\
            ,np.array([b"a",b"1.0",b"c"]) )

    def test_class(self):
        assert self.s.original_metadata.instrument.\
            energy.attrs.NX_class == "NXmonochromater"

    def test_signal_loaded(self):
        assert self.s.metadata.Signal.signal_type == "Signal2D"


    def test_axes_names(self):
        assert self.s.axes_manager[0].name == "xaxis"
        assert self.s.axes_manager[1].name == "yaxis"
 
#
# Test nexus loading..external dls file
#
class TestSavedMultiSignalLoad():

    def setup_method(self, method):
        self.s = load(file2)
    
    def test_signals(self):
        assert len(self.s) == 2
        
    def test_signal1_string(self):
        assert self.s[0].original_metadata.instrument.\
            energy.attrs.units == "keV"

    def test_signal1_value(self):
        assert self.s[0].original_metadata.instrument.\
            energy.value == 12.0

    def test_signal1_string_array(self):
        np.testing.assert_array_equal(self.s[0].original_metadata.instrument.energy.attrs.test\
            ,np.array([b"a",b"1.0",b"c"]) )

    def test_signal1_class(self):
        assert self.s[0].original_metadata.instrument.\
            energy.attrs.NX_class == "NXmonochromater"

    def test_signal1_signal_loaded(self):
        assert self.s[0].metadata.Signal.signal_type == "Signal2D"


    def test_signal1_axes_names(self):
        assert self.s[0].axes_manager[0].name == "xaxis"
        assert self.s[0].axes_manager[1].name == "yaxis"


    def test_signal2_string(self):
        assert self.s[1].original_metadata.instrument.processing.\
            window_size.value == 20.0

    def test_signal2_string_array(self):
        np.testing.assert_array_equal(self.s[1].original_metadata.instrument.processing.lines.value\
            ,np.array([b"Fe_Ka",b"Cu_Ka",b"Compton"]) )

    def test_signal2_class(self):
        assert self.s[1].original_metadata.instrument.scantype.value\
             == "XRF"

    def test_signal2_signal_loaded(self):
        assert self.s[1].metadata.Signal.signal_type == "BaseSignal"


    def test_signal2_axes_names(self):
        assert self.s[1].axes_manager[0].name == "energy"



# class TestExample1_10():

#     def setup_method(self, method):
#         self.s = load(os.path.join(
#             my_path,
#             "hdf5_files",
#             "example1_v1.0.hdf5"),file_format="Nexus",
#             nxdata_only=False)


# @pytest.fixture()
# def tmpfilepath():
#     with tempfile.TemporaryDirectory() as tmp:
#         yield os.path.join(tmp, "test.nxs")
#         gc.collect()        # Make sure any memmaps are closed first!


# class TestSavingMetadataContainers:

#     def setup_method(self, method):
#         self.s = BaseSignal([0.1,0.2,0.3])

#     def test_save_unicode(self, tmpfilepath):
#         s = self.s
#         s.original_metadata.set_item('test1',44.0)
#         s.original_metadata.set_item('test2',54.0)
#         s.original_metadata.set_item('test3',64.0)        
#         s.save(tmpfilepath)
#         l = load(tmpfilepath)
#         assert isinstance(l.original_metadata.test1.value, float)
#         assert isinstance(l.original_metadata.test2.value, float)
#         assert isinstance(l.original_metadata.test3.value, float)
#         assert l.original_metadata.test2.value == 54.0

# #
# # test keywords from loading nexus file
# #
def test_read_file2_signal1_meta_exception():
    s = hs.load(file2,dataset_keys=["unnamed__0"])
    assert not isinstance(s,list)

def test_read_file2_signal1():
    s = hs.load(file2,dataset_keys=["unnamed__0"])
    assert s.metadata.Signal.signal_type == "Signal2D"

def test_read_file2_signal2():
    s = hs.load(file2,dataset_keys=["unnamed__1"])
    assert s.metadata.Signal.signal_type == "BaseSignal"

def test_read_file2_meta():
    s = hs.load(file2,dataset_keys=["unnamed__0"],metadata_keys=["energy"])
    assert s.metadata.Signal.signal_type == "Signal2D"
    assert s.original_metadata.instrument.\
            energy.value == 12.0
            
def test_read_file2_signal1_meta_exception():
    s = hs.load(file2,dataset_keys=["unnamed__0"],metadata_keys=["energy"])
    with pytest.raises(AttributeError):
        s.original_metadata.instrument.stages



# def test_read1_search_keys():
#     # should only find 2 data sets
#     s = hs.load(file1,dset_search_keys=["xsp3"])
#     assert len(s) == 2

# def test_read2():
#     s = hs.load(file2)
#     assert len(s) == 20
    
# def test_read2_data_search_keys():
#     # should only find 2 data sets
#     s = hs.load(file2,dset_search_keys=["Fe"])
#     assert len(s) == 1
            

# =============================================================================
