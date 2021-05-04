# -*- coding: utf-8 -*-
# Copyright 2007-2021 The HyperSpy developers
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

import numpy as np
import pytest
import traits.api as t

import hyperspy.api as hs
from hyperspy._signals.signal1d import Signal1D
from hyperspy._signals.signal2d import Signal2D
from hyperspy.io import load
from hyperspy.io_plugins.nexus import (_byte_to_string, _fix_exclusion_keys,
                                       _is_int, _is_numeric_data, file_writer,
                                       list_datasets_in_file,
                                       read_metadata_from_file)
from hyperspy.signal import BaseSignal

dirpath = os.path.dirname(__file__)

file1 = os.path.join(dirpath, 'nexus_files', 'simple_signal.nxs')
file2 = os.path.join(dirpath, 'nexus_files', 'saved_multi_signal.nxs')
file3 = os.path.join(dirpath, 'nexus_files', 'nexus_dls_example.nxs')
file4 = os.path.join(dirpath, 'nexus_files', 'nexus_dls_example_no_axes.nxs')


my_path = os.path.dirname(__file__)


class TestDLSNexus():

    def setup_method(self, method):
        self.file = file3
        self.s = load(file3, metadata_keys=None, dataset_keys=None,
                      nxdata_only=True, hardlinks_only=True)

    @pytest.mark.parametrize("nxdata_only", [True, False])
    @pytest.mark.parametrize("hardlinks_only", [True, False])
    def test_nxdata_only(self, nxdata_only, hardlinks_only):
        s = load(self.file, nxdata_only=nxdata_only,
                 hardlinks_only=hardlinks_only)
        if nxdata_only and not hardlinks_only:
            assert len(s) == 2
        if nxdata_only and hardlinks_only:
            assert not isinstance(s, list)
        if nxdata_only is False and hardlinks_only:
            assert len(s) == 12
        if nxdata_only is False and not hardlinks_only:
            assert len(s) == 16

    @pytest.mark.parametrize("metadata_keys", ["m1_y", "xxxx"])
    def test_metadata_keys(self, metadata_keys):
        s = load(file3, nxdata_only=True, metadata_keys=metadata_keys)
        # hardlinks are false - soft linked data is loaded
        if metadata_keys == "m1_y":
            assert s[1].original_metadata.alias_metadata.\
                m1_y.attrs.units == "mm"
        else:
            with pytest.raises(AttributeError):
                assert s[0].original_metadata.alias_metadata.\
                    m1_y.attrs.units == "mm"

    def test_value(self):
        assert self.s.original_metadata.instrument.\
            beamline.M1.m1_y.value == -4.0

    def test_class(self):
        assert self.s.original_metadata.instrument.\
            beamline.M1.attrs.NX_class == "NXmirror"

    def test_axes_names(self):
        assert self.s.axes_manager[0].name == "x"
        assert self.s.axes_manager[1].name == "y"

    def test_string(self):
        assert self.s.original_metadata.instrument.\
            beamline.M1.m1_y.attrs.units == "mm"

    def test_save_hspy(self, tmp_path):
        try:
            self.s.save(tmp_path / 'test.hspy')
        except:
            pytest.fail("unexpected error saving hdf5")


class TestDLSNexusNoAxes():

    def setup_method(self, method):
        self.file = file4
        self.s = load(file4, hardlinks_only=True,
                      nxdata_only=True)

    @pytest.mark.parametrize("metadata_keys", [None, "m1_x"])
    def test_meta_keys(self, metadata_keys):
        s = load(file3, nxdata_only=True, metadata_keys=metadata_keys)
        # hardlinks are false - soft linked data is loaded
        if metadata_keys is None:
            assert s[1].original_metadata.instrument.beamline.M1.\
                m1_y.attrs.units == "mm"
        else:
            with pytest.raises(AttributeError):
                assert s[1].original_metadata.instrument.beamline.M1.\
                    m1_y.attrs.units == "mm"

    def test_value(self):
        assert self.s.original_metadata.instrument.\
             beamline.M1.m1_y.value == -4.0

    def test_class(self):
        assert self.s.original_metadata.instrument.\
            beamline.M1.attrs.NX_class == "NXmirror"

    def test_signal_loaded(self):
        assert self.s.metadata.General.title == "nexustest"

    def test_axes_names(self):
        assert self.s.axes_manager[0].name == t.Undefined
        assert self.s.axes_manager[1].name == t.Undefined

    def test_string(self):
        assert self.s.original_metadata.instrument.\
                 beamline.M1.m1_y.attrs.units == "mm"

    def test_save_hspy(self, tmp_path):
        try:
            self.s.save(tmp_path / 'test.hspy')
        except:
            pytest.fail("unexpected error saving hdf5")


class TestSavedSignalLoad():

    def setup_method(self, method):
        self.s = load(file1, nxdata_only=True)

    def test_string(self):
        assert self.s.original_metadata.instrument.\
            energy.attrs.units == "keV"

    def test_value(self):
        assert self.s.original_metadata.instrument.\
            energy.value == 12.0

    def test_string_array(self):
        np.testing.assert_array_equal(
            self.s.original_metadata.instrument.energy.attrs.test,
            np.array([b"a", b"1.0", b"c"]))

    def test_class(self):
        assert self.s.original_metadata.instrument.\
            energy.attrs.NX_class == "NXmonochromater"

    def test_signal_loaded(self):
        assert self.s.metadata.General.title == "rocks"

    def test_axes_names(self):
        assert self.s.axes_manager[0].name == "xaxis"
        assert self.s.axes_manager[1].name == "yaxis"


class TestSavedMultiSignalLoad():

    def setup_method(self, method):
        self.s = load(file2, nxdata_only=True,
                      hardlinks_only=True, use_default=False)

    def test_signals(self):
        assert len(self.s) == 2

    def test_signal1_string(self):
        assert self.s[0].original_metadata.instrument.\
            energy.attrs.units == "keV"

    def test_signal1_value(self):
        assert self.s[0].original_metadata.instrument.\
            energy.value == 12.0

    def test_signal1_string_array(self):
        np.testing.assert_array_equal(
            self.s[0].original_metadata.instrument.energy.attrs.test,
            np.array([b"a", b"1.0", b"c"]))

    def test_signal1_class(self):
        assert self.s[0].original_metadata.instrument.\
            energy.attrs.NX_class == "NXmonochromater"

    def test_signal1_signal_loaded(self):
        assert self.s[0].metadata.General.title == "rocks"

    def test_signal1_axes_names(self):
        assert self.s[0].axes_manager[0].name == "xaxis"
        assert self.s[0].axes_manager[1].name == "yaxis"

    def test_signal2_float(self):
        assert (
            self.s[1].original_metadata.instrument.processing.window_size
            == 20.0)

    def test_signal2_string_array(self):
        np.testing.assert_array_equal(
            self.s[1].original_metadata.instrument.processing.lines,
            np.array([b"Fe_Ka", b"Cu_Ka", b"Compton"]))

    def test_signal2_string(self):
        assert self.s[1].original_metadata.instrument.scantype\
              == "XRF"

    def test_signal2_signal_loaded(self):
        assert self.s[1].metadata.General.title == "unnamed__1"

    def test_signal2_axes_names(self):
        assert self.s[1].axes_manager[2].name == "energy"


class TestSavingMetadataContainers:

    def setup_method(self, method):
        self.s = BaseSignal([0.1, 0.2, 0.3])

    def test_save_scalers(self, tmp_path):
        s = self.s
        s.original_metadata.set_item('test1', 44.0)
        s.original_metadata.set_item('test2', 54.0)
        s.original_metadata.set_item('test3', 64.0)
        fname = tmp_path / 'test.nxs'
        s.save(fname)
        lin = load(fname, nxdata_only=True)
        assert isinstance(lin.original_metadata.test1, float)
        assert isinstance(lin.original_metadata.test2, float)
        assert isinstance(lin.original_metadata.test3, float)
        assert lin.original_metadata.test2 == 54.0

    def test_save_arrays(self, tmp_path):
        s = self.s
        s.original_metadata.set_item("testarray1", ["a", 2, "b", 4, 5])
        s.original_metadata.set_item("testarray2", (1, 2, 3, 4, 5))
        s.original_metadata.set_item("testarray3", np.array([1, 2, 3, 4, 5]))
        fname = tmp_path / 'test.nxs'
        s.save(fname)
        lin = load(fname, nxdata_only=True)
        np.testing.assert_array_equal(lin.original_metadata.testarray1,
                                      np.array([b"a", b'2', b'b', b'4', b'5']))
        np.testing.assert_array_equal(lin.original_metadata.testarray2,
                                      np.array([1, 2, 3, 4, 5]))
        np.testing.assert_array_equal(lin.original_metadata.testarray3,
                                      np.array([1, 2, 3, 4, 5]))

    def test_save_original_metadata(self, tmp_path):
        s = self.s
        s.original_metadata.set_item("testarray1", ["a", 2, "b", 4, 5])
        s.original_metadata.set_item("testarray2", (1, 2, 3, 4, 5))
        s.original_metadata.set_item("testarray3", np.array([1, 2, 3, 4, 5]))
        fname = tmp_path / 'test.nxs'
        s.save(fname, save_original_metadata=False)
        lin = load(fname, nxdata_only=True)
        with pytest.raises(AttributeError):
            lin.original_metadata.testarray1


def test_saving_multi_signals(tmp_path):

    sig = hs.signals.Signal2D(np.zeros((15, 1, 40, 40)))
    sig.axes_manager[0].name = "stage_y_axis"
    sig.original_metadata.set_item("stage_y.value", 4.0)
    sig.original_metadata.set_item("stage_y.attrs.units", "mm")

    sig2 = hs.signals.Signal1D(np.zeros((30, 30, 10)))
    sig2.axes_manager[0].name = "axis1"
    sig2.axes_manager[1].name = "axis2"
    sig2.original_metadata.set_item("stage_x.value", 8.0)
    sig2.original_metadata.set_item("stage_x.attrs.units", "mm")

    fname = tmp_path / 'test.nxs'
    sig.save(fname)
    file_writer(fname, [sig, sig2])
    lin = load(fname, nxdata_only=True)
    assert len(lin) == 2
    assert lin[0].original_metadata.stage_y.value == 4.0
    assert lin[0].axes_manager[0].name == "stage_y_axis"
    assert lin[1].original_metadata.stage_x.value == 8.0
    assert lin[1].original_metadata.stage_x.attrs.units == "mm"
    assert isinstance(lin[0], Signal2D)
    assert isinstance(lin[1], Signal1D)
    # test the metadata haven't merged
    with pytest.raises(AttributeError):
        lin[1].original_metadata.stage_y.value


def test_read_file2_dataset_key_test():
    s = hs.load(file2, nxdata_only=True, dataset_keys=["rocks"])
    assert not isinstance(s, list)


def test_read_file2_signal1():
    s = hs.load(file2, nxdata_only=True, dataset_keys=["rocks"])
    assert s.metadata.General.title == "rocks"


def test_read_file2_default():
    s = hs.load(file2, use_default=False, nxdata_only=True,
                hardlinks_only=True, dataset_keys=["unnamed__1"])
    assert s.metadata.General.title == "unnamed__1"
    s = hs.load(file2, use_default=True, nxdata_only=True,
                hardlinks_only=True, dataset_keys=["unnamed__1"])
    assert s.metadata.General.title == "rocks"


def test_read_file2_metadata_keys():
    s = hs.load(file2, nxdata_only=True,
                dataset_keys=["rocks"], metadata_keys=["energy"])
    assert s.original_metadata.instrument.energy.value == 12.0

def test_read_lazy_file():
    s = hs.load(file3, nxdata_only=True, lazy=True)
    assert s[0]._lazy and s[1]._lazy

@pytest.mark.parametrize("verbose", [True, False])
@pytest.mark.parametrize("dataset_keys", ["testdata", "nexustest"])
def test_list_datasets(verbose, dataset_keys):
    s = list_datasets_in_file(file3, verbose=verbose,
                              dataset_keys=dataset_keys)
    if dataset_keys == "testdata":
        assert len(s[1]) == 3
    else:
        assert len(s[1]) == 6


@pytest.mark.parametrize("metadata_keys", [None, "xxxxx"])
def test_read_metdata(metadata_keys):
    s = read_metadata_from_file(file3,
                                metadata_keys=metadata_keys)
    # hardlinks are false - soft linked data is loaded
    if metadata_keys is None:
        assert s["alias_metadata"]["m1_y"]["attrs"]["units"] == "mm"
    else:
        with pytest.raises(KeyError):
            assert s["alias_metadata"]["m1_y"]["attrs"]["units"] == "mm"


def test_is_int():
    assert _is_int("a") is False


def test_is_numeric_data():
    assert _is_numeric_data(np.array(["a", "b"])) is False


def test_exclusion_keys():
    assert _fix_exclusion_keys("keys") == "fix_keys"


def test_unicode_error():
    assert _byte_to_string(b'\xff\xfeW[') == "ÿþW["
