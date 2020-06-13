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
import gc
import tempfile
import numpy as np
import pytest
from hyperspy.io import load
from hyperspy.io_plugins.nexus import read_metadata_from_file,\
    list_datasets_in_file,file_writer,_is_int,_is_numeric_data,\
        _fix_exclusion_keys,_byte_to_string
import hyperspy.api as hs
from hyperspy.signal import BaseSignal
import traits.api as t


dirpath = os.path.dirname(__file__)

file1 = os.path.join(dirpath, 'nexus_files', 'simple_signal.nxs')
file2 = os.path.join(dirpath, 'nexus_files', 'saved_multi_signal.nxs')
file3 = os.path.join(dirpath, 'nexus_files', 'nexus_dls_example.nxs')
file4 = os.path.join(dirpath, 'nexus_files', 'nexus_dls_example_no_axes.nxs')


my_path = os.path.dirname(__file__)


@pytest.fixture()
def tmpfilepath():
    with tempfile.TemporaryDirectory() as tmp:
         yield os.path.join(tmp, "test.nxs")
         gc.collect()        # Make sure any memmaps are closed first!

#
# Test nexus loading..external dls file
#
class TestDLSNexus():

    def setup_method(self, method):
        self.file = file3
        self.s = load(file3,metadata_keys="all",dataset_keys="hardlinks",
                      nxdata_only=True)

    @pytest.mark.parametrize("lazy", [True,False])
    @pytest.mark.parametrize("small_metadata_only", [True,False])
    @pytest.mark.parametrize("nxdata_only",[True,False])    
    @pytest.mark.parametrize("dataset_keys" ,["all","hardlinks"])
    def test_general_keys(self,lazy,small_metadata_only,nxdata_only,
                          dataset_keys):
        s=load(self.file,lazy=lazy,small_metadata_only=small_metadata_only,
               nxdata_only=nxdata_only,dataset_keys=dataset_keys)
        if isinstance(s,list):
            assert s[0].original_metadata.entry.instrument.\
                scannables.m1.m1_y.attrs.units == "mm"
        else:
            assert s.original_metadata.entry.instrument.\
                scannables.m1.m1_y.attrs.units == "mm"

    @pytest.mark.parametrize("nxdata_only", [True,False])
    @pytest.mark.parametrize("dataset_keys", ["all","hardlinks"])
    def test_nxdata_only(self,nxdata_only,dataset_keys):
        s=load(self.file,nxdata_only=nxdata_only,dataset_keys=dataset_keys)
        if nxdata_only and dataset_keys == "all":
            assert len(s) == 2
        if nxdata_only and dataset_keys == "hardlinks":
            assert  not isinstance(s,list)
        if nxdata_only == False and dataset_keys == "hardlinks":
            assert len(s) == 12
        if nxdata_only == False and dataset_keys == "all":
            assert len(s) == 16

    @pytest.mark.parametrize("metadata_keys", ["all","xxxx"])
    def test_hard_links(self,metadata_keys):
        s=load(file3,nxdata_only=True,metadata_keys=metadata_keys)
        # hardlinks are false - soft linked data is loaded
        if metadata_keys=="all":
            assert s[0].original_metadata.alias_metadata.\
            m1_y.attrs.units == "mm"
        else:
            with pytest.raises(AttributeError):
                assert s[0].original_metadata.alias_metadata.\
                m1_y.attrs.units == "mm"

    def test_value(self):
        assert self.s.original_metadata.entry.instrument.\
             beamline.M1.m1_y.value == -4.0

    def test_class(self):
        assert self.s.original_metadata.entry.instrument.\
            beamline.M1.attrs.NX_class == "NXmirror"

    def test_string_array(self):
        np.testing.assert_array_equal(self.s.original_metadata.entry.arraytest,
                                      np.array([b"a",b"1.0",b"c"]))
        
    def test_signal_loaded(self):
        assert self.s.metadata.General.title == "nexustest"


    def test_axes_names(self):
        assert self.s.axes_manager[0].name == "x"
        assert self.s.axes_manager[1].name == "y"
        

    def test_string(self):
        assert self.s.original_metadata.entry.instrument.\
                 beamline.M1.m1_y.attrs.units == "mm"

    

#
# Test nexus loading..external dls file
#
class TestDLSNexusNoAxes():

    def setup_method(self, method):
        self.file = file4
        self.s = load(file4,metadata_keys="all",dataset_keys="hardlinks",
                      nxdata_only=True)

    @pytest.mark.parametrize("lazy", [True,False])
    @pytest.mark.parametrize("nxdata_only",[True,False])    
    @pytest.mark.parametrize("dataset_keys" ,["all","hardlinks"])
    def test_general_keys(self,lazy,nxdata_only,
                          dataset_keys):
        s=load(self.file,lazy=lazy,
               nxdata_only=nxdata_only,dataset_keys=dataset_keys)
        if isinstance(s,list):
            assert s[0].original_metadata.entry.instrument.\
                scannables.m1.m1_y.attrs.units == "mm"
        else:
            assert s.original_metadata.entry.instrument.\
                scannables.m1.m1_y.attrs.units == "mm"

    @pytest.mark.parametrize("metadata_keys", ["all","xxxx"])
    def test_hard_links(self,metadata_keys):
        s=load(file3,nxdata_only=True,metadata_keys=metadata_keys)
        # hardlinks are false - soft linked data is loaded
        if metadata_keys=="all":
            assert s[0].original_metadata.alias_metadata.\
            m1_y.attrs.units == "mm"
        else:
            with pytest.raises(AttributeError):
                assert s[0].original_metadata.alias_metadata.\
                m1_y.attrs.units == "mm"

    def test_value(self):
        assert self.s.original_metadata.entry.instrument.\
             beamline.M1.m1_y.value == -4.0

    def test_class(self):
        assert self.s.original_metadata.entry.instrument.\
            beamline.M1.attrs.NX_class == "NXmirror"

    def test_string_array(self):
        np.testing.assert_array_equal(self.s.original_metadata.entry.arraytest,
                                      np.array([b"a",b"1.0",b"c"]))
        
    def test_signal_loaded(self):
        assert self.s.metadata.General.title == "nexustest"


    def test_axes_names(self):
        assert self.s.axes_manager[0].name == t.Undefined
        assert self.s.axes_manager[1].name == t.Undefined
        

    def test_string(self):
        assert self.s.original_metadata.entry.instrument.\
                 beamline.M1.m1_y.attrs.units == "mm"
#


# Test nexus loading..external dls file
#
class TestSavedSignalLoad():

    def setup_method(self, method):
        self.s = load(file1,nxdata_only=True)
   
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
        assert self.s.metadata.General.title == "unnamed__0"


    def test_axes_names(self):
        assert self.s.axes_manager[0].name == "xaxis"
        assert self.s.axes_manager[1].name == "yaxis"
 
#
# Test nexus loading..external dls file
#
class TestSavedMultiSignalLoad():

    def setup_method(self, method):
        self.s = load(file2,nxdata_only=True)
    
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
        assert self.s[0].metadata.General.title == "unnamed__0"


    def test_signal1_axes_names(self):
        assert self.s[0].axes_manager[0].name == "xaxis"
        assert self.s[0].axes_manager[1].name == "yaxis"


    def test_signal2_string(self):
        assert self.s[1].original_metadata.instrument.processing.\
            window_size == 20.0

    def test_signal2_string_array(self):
        np.testing.assert_array_equal(self.s[1].original_metadata.instrument.processing.lines\
            ,np.array([b"Fe_Ka",b"Cu_Ka",b"Compton"]) )

    def test_signal2_class(self):
        assert self.s[1].original_metadata.instrument.scantype\
              == "XRF"

    def test_signal2_signal_loaded(self):
        assert self.s[1].metadata.General.title == "unnamed__1"        


    def test_signal2_axes_names(self):
        assert self.s[1].axes_manager[0].name == "energy"


class TestSavingMetadataContainers:

    def setup_method(self, method):
        self.s = BaseSignal([0.1,0.2,0.3])

    def test_save_scalers(self, tmpfilepath):
        s = self.s
        s.original_metadata.set_item('test1',44.0)
        s.original_metadata.set_item('test2',54.0)
        s.original_metadata.set_item('test3',64.0)        
        s.save(tmpfilepath)
        l = load(tmpfilepath,nxdata_only=True)
        print(l.original_metadata)
        assert isinstance(l.original_metadata.test1, float)
        assert isinstance(l.original_metadata.test2, float)
        assert isinstance(l.original_metadata.test3, float)
        assert l.original_metadata.test2 == 54.0

    def test_save_arrays(self, tmpfilepath):
        s = self.s
        s.original_metadata.set_item("testarray1",["a",2,"b",4,5])
        s.original_metadata.set_item("testarray2",(1,2,3,4,5))
        s.original_metadata.set_item("testarray3",np.array([1,2,3,4,5]))        
        s.save(tmpfilepath)
        l = load(tmpfilepath,nxdata_only=True)
        np.testing.assert_array_equal(l.original_metadata.testarray1,np.array([b"a",b'2',b'b',b'4',b'5']))
        np.testing.assert_array_equal(l.original_metadata.testarray2,np.array([1,2,3,4,5]))
        np.testing.assert_array_equal(l.original_metadata.testarray3,np.array([1,2,3,4,5]))




class TestSavingMultiSignals:

    def setup_method(self, method):
        data = np.zeros((15,1,40,40))
        self.sig = hs.signals.Signal2D(data)
        self.sig.axes_manager[0].name = "stage_y_axis"

        self.sig.original_metadata.set_item("stage_y.value",4.0)
        self.sig.original_metadata.set_item("stage_y.attrs.units","mm")
        
        data = np.zeros((30,30,10))
        self.sig2 = hs.signals.Signal1D(data)
        self.sig2.axes_manager[0].name = "axis1"
        self.sig2.axes_manager[1].name = "axis2"        
        self.sig2.original_metadata.set_item("stage_x.value",8.0)
        self.sig2.original_metadata.set_item("stage_x.attrs.units","mm")


    def test_save_signal_list(self, tmpfilepath):
        file_writer(tmpfilepath,[self.sig,self.sig2])
        l = load(tmpfilepath,nxdata_only=True)
        assert len(l) == 2
        assert l[0].original_metadata.stage_y.value == 4.0
        assert l[0].axes_manager[0].name == "stage_y_axis"
        assert l[1].original_metadata.stage_x.value == 8.0
        assert l[1].original_metadata.stage_x.attrs.units == "mm"
        # test the metadata haven't merged..
        with pytest.raises(AttributeError):
            l[1].original_metadata.stage_y.value


# # #
# # # test keywords from loading nexus file
# # #
def test_read_file2_dataset_key_test():
    s = hs.load(file2,nxdata_only=True,dataset_keys=["unnamed__0"])
    assert not isinstance(s,list)

def test_read_file2_signal1():
    s = hs.load(file2,nxdata_only=True,dataset_keys=["unnamed__0"])
    assert s.metadata.General.title == "unnamed__0"

def test_read_file2_signal2():
    s = hs.load(file2,nxdata_only=True,dataset_keys=["unnamed__1"])
    assert s.metadata.General.title == "unnamed__1"

def test_read_file2_meta():
    s = hs.load(file2,nxdata_only=True,
                dataset_keys=["unnamed__0"],metadata_keys=["energy"])
    assert s.original_metadata.instrument.\
            energy.value == 12.0

@pytest.mark.parametrize("verbose", [True,False])
@pytest.mark.parametrize("dataset_keys", ["testdata","nexustest"])
def test_read_datasets(verbose,dataset_keys):
      s = list_datasets_in_file(file3,verbose=verbose,\
                                  dataset_keys=dataset_keys)
      if dataset_keys == "testdata":
         assert len(s[1]) == 3
      else:
         assert len(s[1]) == 6


@pytest.mark.parametrize("metadata_keys", ["all","xxxxx"])
def test_read_metdata(metadata_keys):
    s = read_metadata_from_file(file3,\
                                  metadata_keys=metadata_keys)
    # hardlinks are false - soft linked data is loaded
    if metadata_keys=="all":
        assert s["alias_metadata"]["m1_y"]["attrs"]["units"] == "mm"
    else:
        with pytest.raises(KeyError):
            assert s["alias_metadata"]["m1_y"]["attrs"]["units"] == "mm"          
    
def test_is_int():
    assert _is_int("a")  == False

def test_is_numeric_data():
    assert _is_numeric_data(np.array(["a","b"]))== False
    
def test_exclusion_keys():
    assert _fix_exclusion_keys("keys") == "fix_keys"
    
def test_unicode_error():
    assert _byte_to_string(b'\xff\xfeW[') == "ÿþW["
