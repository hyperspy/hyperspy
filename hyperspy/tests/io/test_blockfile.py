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


import warnings
import os
import gc
import tempfile

import numpy as np
from numpy.testing import assert_allclose
import pytest

from hyperspy.io_plugins.blockfile import get_default_header
from hyperspy.misc.array_tools import sarray2dict
import hyperspy.api as hs
from hyperspy.misc.test_utils import assert_deep_almost_equal
from hyperspy.misc.date_time_tools import serial_date_to_ISO_format


try:
    WindowsError
except NameError:
    WindowsError = None


DIRPATH = os.path.dirname(__file__)
FILE1 = os.path.join(DIRPATH, 'blockfile_data', 'test1.blo')
FILE2 = os.path.join(DIRPATH, 'blockfile_data', 'test2.blo')


@pytest.fixture()
def save_path():
    with tempfile.TemporaryDirectory() as tmp:
        filepath = os.path.join(tmp, 'blockfile_data', 'save_temp.blo')
        yield filepath
        # Force files release (required in Windows)
        gc.collect()


ref_data2 = np.array(
    [[[[20, 23, 25, 25, 27],
       [29, 23, 23, 0, 29],
       [24, 0, 0, 22, 18],
       [0, 14, 19, 17, 26],
       [19, 21, 22, 27, 20]],

      [[28, 25, 29, 15, 29],
       [12, 15, 12, 25, 24],
       [25, 26, 26, 18, 27],
       [19, 18, 20, 23, 28],
       [28, 18, 22, 25, 0]],

      [[21, 29, 25, 19, 18],
       [30, 15, 20, 22, 26],
       [23, 18, 26, 15, 25],
       [22, 25, 24, 15, 20],
       [22, 15, 15, 21, 23]]],


     [[[28, 25, 26, 24, 26],
       [26, 17, 0, 24, 12],
       [17, 18, 21, 19, 21],
       [21, 24, 19, 17, 0],
       [17, 14, 25, 15, 26]],

      [[25, 18, 20, 15, 24],
       [19, 13, 23, 18, 11],
       [0, 25, 0, 0, 14],
       [26, 22, 22, 11, 14],
       [21, 0, 15, 13, 19]],

      [[24, 18, 20, 22, 21],
       [13, 25, 20, 28, 29],
       [15, 17, 24, 23, 23],
       [22, 21, 21, 22, 18],
       [24, 25, 18, 18, 27]]]], dtype=np.uint8)

axes1 = {
    'axis-0': {
        'name': 'y', 'navigate': True, 'offset': 0.0,
        'scale': 12.8, 'size': 3, 'units': 'nm'},
    'axis-1': {
        'name': 'x', 'navigate': True, 'offset': 0.0,
        'scale': 12.8, 'size': 2, 'units': 'nm'},
    'axis-2': {
        'name': 'dy', 'navigate': False, 'offset': 0.0,
        'scale': 0.016061676839061997, 'size': 144, 'units': 'cm'},
    'axis-3': {
        'name': 'dx', 'navigate': False, 'offset': 0.0,
        'scale': 0.016061676839061997, 'size': 144, 'units': 'cm'}}

axes2 = {
    'axis-0': {
        'name': 'y', 'navigate': True, 'offset': 0.0,
        'scale': 64.0, 'size': 2, 'units': 'nm'},
    'axis-1': {
        'name': 'x', 'navigate': True, 'offset': 0.0,
        'scale': 64.0, 'size': 3, 'units': 'nm'},
    'axis-2': {
        'name': 'dy', 'navigate': False, 'offset': 0.0,
        'scale': 0.016061676839061997, 'size': 5, 'units': 'cm'},
    'axis-3': {
        'name': 'dx', 'navigate': False, 'offset': 0.0,
        'scale': 0.016061676839061997, 'size': 5, 'units': 'cm'}}

axes2_converted = {
    'axis-0': {
        'name': 'y', 'navigate': True, 'offset': 0.0,
        'scale': 64.0, 'size': 2, 'units': 'nm'},
    'axis-1': {
        'name': 'x', 'navigate': True, 'offset': 0.0,
        'scale': 64.0, 'size': 3, 'units': 'nm'},
    'axis-2': {
        'name': 'dy', 'navigate': False, 'offset': 0.0,
        'scale': 160.61676839061997, 'size': 5, 'units': 'um'},
    'axis-3': {
        'name': 'dx', 'navigate': False, 'offset': 0.0,
        'scale': 160.61676839061997, 'size': 5, 'units': 'um'}}


def test_load1():
    s = hs.load(FILE1)
    assert s.data.shape == (3, 2, 144, 144)
    assert s.axes_manager.as_dictionary() == axes1


@pytest.mark.parametrize(("convert_units"), (True, False))
def test_load2(convert_units):
    s = hs.load(FILE2, convert_units=convert_units)
    assert s.data.shape == (2, 3, 5, 5)
    axes = axes2_converted if convert_units else axes2
    np.testing.assert_equal(s.axes_manager.as_dictionary(), axes)
    np.testing.assert_allclose(s.data, ref_data2)


@pytest.mark.parametrize(("convert_units"), (True, False))
def test_save_load_cycle(save_path, convert_units):
    sig_reload = None
    signal = hs.load(FILE2, convert_units=convert_units)
    serial = signal.original_metadata['blockfile_header']['Acquisition_time']
    date, time, timezone = serial_date_to_ISO_format(serial)
    assert signal.metadata.General.original_filename == 'test2.blo'
    assert signal.metadata.General.date == date
    assert signal.metadata.General.time == time
    assert signal.metadata.General.time_zone == timezone
    assert (
        signal.metadata.General.notes ==
        "Precession angle : \r\nPrecession Frequency : \r\nCamera gamma : on")
    signal.save(save_path, overwrite=True)
    sig_reload = hs.load(save_path, convert_units=convert_units)
    np.testing.assert_equal(signal.data, sig_reload.data)
    assert (signal.axes_manager.as_dictionary() ==
            sig_reload.axes_manager.as_dictionary())
    assert (signal.original_metadata.as_dictionary() ==
            sig_reload.original_metadata.as_dictionary())
    # change original_filename to make the metadata of both signals equals
    sig_reload.metadata.General.original_filename = signal.metadata.General.original_filename
    assert_deep_almost_equal(signal.metadata.as_dictionary(),
                             sig_reload.metadata.as_dictionary())
    assert (
        signal.metadata.General.date ==
        sig_reload.metadata.General.date)
    assert (
        signal.metadata.General.time ==
        sig_reload.metadata.General.time)
    assert isinstance(signal, hs.signals.Signal2D)
    # Delete reference to close memmap file!
    del sig_reload


def test_different_x_y_scale_units(save_path):
    # perform load and save cycle with changing the scale on y
    signal = hs.load(FILE2)
    signal.axes_manager[0].scale = 50.0
    signal.save(save_path, overwrite=True)
    sig_reload = hs.load(save_path)
    assert_allclose(sig_reload.axes_manager[0].scale, 50.0,
                    rtol=1E-5)
    assert_allclose(sig_reload.axes_manager[1].scale, 64.0,
                    rtol=1E-5)
    assert_allclose(sig_reload.axes_manager[2].scale, 0.0160616,
                    rtol=1E-5)


def test_default_header():
    # Simply check that no exceptions are raised
    header = get_default_header()
    assert header is not None


def test_non_square(save_path):
    signal = hs.signals.Signal2D((255 * np.random.rand(10, 3, 5, 6)
                                  ).astype(np.uint8))
    with pytest.raises(ValueError):
        signal.save(save_path, overwrite=True)


def test_load_lazy():
    from dask.array import Array
    s = hs.load(FILE2, lazy=True)
    assert isinstance(s.data, Array)


def test_load_to_memory():
    s = hs.load(FILE2, lazy=False)
    assert isinstance(s.data, np.ndarray)
    assert not isinstance(s.data, np.memmap)


def test_load_readonly():
    s = hs.load(FILE2, lazy=True)
    k = next(filter(lambda x: isinstance(x, str) and
                    x.startswith("array-original"),
                    s.data.dask.keys()))
    mm = s.data.dask[k]
    assert isinstance(mm, np.memmap)
    assert not mm.flags["WRITEABLE"]
    with pytest.raises(NotImplementedError):
        s.data[:] = 23


def test_load_inplace():
    with pytest.raises(ValueError):
        hs.load(FILE2, lazy=True, mmap_mode='r+')


def test_write_fresh(save_path):
    signal = hs.signals.Signal2D((255 * np.random.rand(10, 3, 5, 5)
                                  ).astype(np.uint8))
    signal.save(save_path, overwrite=True)
    sig_reload = hs.load(save_path)
    np.testing.assert_equal(signal.data, sig_reload.data)
    header = sarray2dict(get_default_header())
    header.update({
        'NX': 3, 'NY': 10,
        'DP_SZ': 5,
        'SX': 1, 'SY': 1,
        'SDP': 100,
        'Data_offset_2': 10 * 3 + header['Data_offset_1'],
        'Note': '',
    })
    header['Data_offset_2'] += header['Data_offset_2'] % 16
    assert (
        sig_reload.original_metadata.blockfile_header.as_dictionary() ==
        header)


def test_write_data_line(save_path):
    signal = hs.signals.Signal2D((255 * np.random.rand(3, 5, 5)
                                  ).astype(np.uint8))
    signal.save(save_path, overwrite=True)
    sig_reload = hs.load(save_path)
    np.testing.assert_equal(signal.data, sig_reload.data)


def test_write_data_single(save_path):
    signal = hs.signals.Signal2D((255 * np.random.rand(5, 5)
                                  ).astype(np.uint8))
    signal.save(save_path, overwrite=True)
    sig_reload = hs.load(save_path)
    np.testing.assert_equal(signal.data, sig_reload.data)


def test_write_data_am_mismatch(save_path):
    signal = hs.signals.Signal2D((255 * np.random.rand(10, 3, 5, 5)
                                  ).astype(np.uint8))
    signal.axes_manager.navigation_axes[1].size = 4
    with pytest.raises(ValueError):
        signal.save(save_path, overwrite=True)


def test_write_cutoff(save_path):
    signal = hs.signals.Signal2D((255 * np.random.rand(10, 3, 5, 5)
                                  ).astype(np.uint8))
    signal.axes_manager.navigation_axes[0].size = 20
    # Test that it raises a warning
    signal.save(save_path, overwrite=True)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sig_reload = hs.load(save_path)
        # There can be other warnings so >=
        assert len(w) >= 1
        warning_blockfile = ["Blockfile header" in str(warning.message)
                             for warning in w]
        assert True in warning_blockfile
        assert issubclass(w[warning_blockfile.index(True)].category,
                          UserWarning)
    cut_data = signal.data.flatten()
    pw = [(0, 17 * 10 * 5 * 5)]
    cut_data = np.pad(cut_data, pw, mode='constant')
    cut_data = cut_data.reshape((10, 20, 5, 5))
    np.testing.assert_equal(cut_data, sig_reload.data)


def test_crop_notes(save_path):
    note_len = 0x1000 - 0xF0
    note = 'test123' * 1000     # > note_len
    signal = hs.signals.Signal2D((255 * np.random.rand(2, 3, 2, 2)
                                  ).astype(np.uint8))
    signal.original_metadata.add_node('blockfile_header.Note')
    signal.original_metadata.blockfile_header.Note = note
    signal.save(save_path, overwrite=True)
    sig_reload = hs.load(save_path)
    assert (sig_reload.original_metadata.blockfile_header.Note ==
            note[:note_len])
