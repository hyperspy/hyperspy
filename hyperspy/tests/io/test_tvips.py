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
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.


import gc
import os
import tempfile
import traits.api as traits
from packaging import version as pversion

import numpy as np
import pytest
import dask

import hyperspy.api as hs
from hyperspy.io_plugins.tvips import (
    _guess_image_mode,
    _get_main_header_from_signal,
    _get_frame_record_dtype_from_signal,
    _is_valid_first_tvips_file,
    _find_auto_scan_start_stop,
    _guess_scan_index_grid,
    TVIPS_RECORDER_GENERAL_HEADER,
    TVIPS_RECORDER_FRAME_HEADER,
    file_writer,
    file_reader,
)
from hyperspy.misc.utils import DictionaryTreeBrowser
from hyperspy.misc.utils import dummy_context_manager


try:
    WindowsError
except NameError:
    WindowsError = None


DIRPATH = os.path.join(os.path.dirname(__file__), "tvips_files")


@pytest.fixture()
def fake_metadata_diffraction():
    metadata = {
        "Acquisition_instrument": {
            "TEM" : {
                "beam_current": 23,
                "beam_energy": 200,
                "camera_length": 80,
            }
        },
        "General": {
            "date": "1993-06-18",
            "time": "12:34:56",
            "time_zone": "CET",
        }
    }
    return DictionaryTreeBrowser(metadata)


@pytest.fixture()
def fake_metadata_imaging():
    metadata = {
        "Acquisition_instrument": {
            "TEM" : {
                "beam_current": 23,
                "beam_energy": 200,
                "magnification": 3000,
            }
        },
        "General": {
            "date": "1993-06-18",
            "time": "12:34:56",
            "time_zone": "CET",
        }
    }
    return DictionaryTreeBrowser(metadata)


@pytest.fixture()
def fake_metadata_confused():
    metadata = {
        "Acquisition_instrument": {
            "TEM" : {
                "beam_current": 23,
                "beam_energy": 200,
                "camera_length": 80,
                "magnification": 3000,
            }
        },
        "General": {
            "date": "1993-06-18",
            "time": "12:34:56",
            "time_zone": "CET",
        }
    }
    return DictionaryTreeBrowser(metadata)


@pytest.fixture()
def fake_metadatas(fake_metadata_diffraction, fake_metadata_imaging, fake_metadata_confused):
    return {"diffraction": fake_metadata_diffraction,
            "imaging": fake_metadata_imaging,
            "confused": fake_metadata_confused,
            }


@pytest.fixture()
def fake_signal_3D():
    fake_data = np.arange(120).reshape(4, 5, 6).astype(np.uint16)
    fake_signal = hs.signals.Signal2D(fake_data)
    fake_signal.axes_manager[0].scale_as_quantity = "1 nm"
    fake_signal.axes_manager[1].scale_as_quantity = "1 1/nm"
    fake_signal.axes_manager[2].scale_as_quantity = "1 1/nm"
    return fake_signal


@pytest.fixture()
def fake_signal_4D():
    fake_data = np.arange(360).reshape(3, 4, 5, 6).astype(np.uint32)
    fake_signal = hs.signals.Signal2D(fake_data)
    fake_signal.axes_manager[0].scale_as_quantity = "1 nm"
    fake_signal.axes_manager[1].scale_as_quantity = "1 nm"
    fake_signal.axes_manager[2].scale_as_quantity = "1 1/nm"
    fake_signal.axes_manager[3].scale_as_quantity = "1 1/nm"
    return fake_signal


@pytest.fixture()
def fake_signal_5D():
    fake_data = np.arange(720).reshape(2, 3, 4, 5, 6).astype(np.uint64)
    fake_signal = hs.signals.Signal2D(fake_data)
    fake_signal.axes_manager[0].scale_as_quantity = "1 s"
    fake_signal.axes_manager[1].scale_as_quantity = "1 nm"
    fake_signal.axes_manager[2].scale_as_quantity = "1 nm"
    fake_signal.axes_manager[3].scale_as_quantity = "1 1/nm"
    fake_signal.axes_manager[4].scale_as_quantity = "1 1/nm"
    return fake_signal


@pytest.fixture()
def fake_signals(fake_signal_3D, fake_signal_4D, fake_signal_5D):
    return {
        "fake_signal_3D": fake_signal_3D,
        "fake_signal_4D": fake_signal_4D,
        "fake_signal_5D": fake_signal_5D,
    }


@pytest.mark.parametrize(
    "unit, expected_mode",
    [
        ("1/nm", 2),
        ("1/m", 2),
        ("nm", 1),
        ("m", 1),
        (traits.Undefined, None),
        ("foo", None),
        ("", None),
        ("s", None),
    ]
)
@pytest.mark.parametrize(
    "sig",
    ["fake_signal_3D", "fake_signal_4D", "fake_signal_5D"]
)
def test_guess_image_mode(unit, expected_mode, sig, fake_signals):
    signal = fake_signals[sig]
    signal.axes_manager[-1].units = unit
    mode = _guess_image_mode(signal)
    assert mode == expected_mode


@pytest.mark.parametrize(
    "unit, expected_scale_factor, version, fheb",
    [
        ("1/pm", 1e3, 2, 0),
        ("um", 1e3, 1, 60),
        ("foo", 1, 2, 12),
    ]
)
@pytest.mark.parametrize(
    "sig",
    ["fake_signal_3D", "fake_signal_4D", "fake_signal_5D"]
)
@pytest.mark.parametrize(
    "metadata",
    ["diffraction", "imaging", "confused", None]
)
def test_main_header_from_signal(unit, expected_scale_factor, version, fheb,
                                 sig, fake_signals, metadata, fake_metadatas):
    signal = fake_signals[sig]
    if metadata is not None:
        signal._metadata = fake_metadatas[metadata]
    signal.axes_manager[-1].units = unit
    signal.axes_manager[-2].units = unit
    original_scale_x = signal.axes_manager[-2].scale
    if unit == "foo":
        cm = pytest.warns(UserWarning)
    else:
        cm = dummy_context_manager()
    with cm:
        header = _get_main_header_from_signal(signal, version, fheb)
    assert header["size"] == np.dtype(TVIPS_RECORDER_GENERAL_HEADER).itemsize
    assert header["version"] == version
    assert header["dimx"] == signal.axes_manager[-2].size
    assert header["dimy"] == signal.axes_manager[-1].size
    assert header["offsetx"] == 0
    assert header["offsety"] == 0
    assert header["pixelsize"] == original_scale_x * expected_scale_factor
    assert header["frameheaderbytes"] == np.dtype(TVIPS_RECORDER_FRAME_HEADER).itemsize + fheb
    if metadata == "diffraction" and unit == "1/pm":
        assert header["magtotal"] == signal.metadata.Acquisition_instrument.TEM.camera_length
    elif metadata == "imaging" and unit == "um":
        assert header["magtotal"] == signal.metadata.Acquisition_instrument.TEM.magnification
    else:
        assert header["magtotal"] == 0
    if metadata is None:
        assert header["ht"] == 0
    else:
        assert header["ht"] == signal.metadata.Acquisition_instrument.TEM.beam_energy


@pytest.mark.parametrize(
    "extra_bytes",
    [0, 20]
)
@pytest.mark.parametrize(
    "sig",
    ["fake_signal_3D", "fake_signal_4D", "fake_signal_5D"]
)
def test_get_frame_record_dtype(sig, fake_signals, extra_bytes):
    signal = fake_signals[sig]
    dt_fh = _get_frame_record_dtype_from_signal(signal, extra_bytes=extra_bytes)
    dimx = signal.axes_manager[-2].size
    dimy = signal.axes_manager[-1].size
    total_size = np.dtype(TVIPS_RECORDER_FRAME_HEADER).itemsize + extra_bytes + dimx * dimy * signal.data.itemsize
    assert dt_fh.itemsize == total_size


@pytest.mark.parametrize(
    "filename",
    [
        pytest.param("foo_000.bla", marks=pytest.mark.xfail(raises=ValueError)),
        pytest.param("foo_0.tvips", marks=pytest.mark.xfail(raises=ValueError)),
        pytest.param("foo_001.tvips", marks=pytest.mark.xfail(raises=ValueError)),
        pytest.param("foo_0000.TVIPS", marks=pytest.mark.xfail(raises=ValueError)),
        ("foo_000.tvips"),
        ("foo_000.TVIPS"),
    ]
)
def test_valid_tvips_file(filename):
    isvalid = _is_valid_first_tvips_file(filename)
    assert isvalid


@pytest.mark.parametrize(
    "rotators, expected",
    [
        (np.array([0, 0, 1, 2, 3, 4, 5, 6, 0, 0]), (2, 7)),
        (np.array([0, 1, 2, 3, 4, 5, 6, 0, 0]), (1, 6)),
        (np.array([1, 2, 3, 4, 5, 6, 0, 0]), (0, 5)),
        (np.array([0, 0, 1, 2, 3, 4, 5]), (2, 6)),
        (np.array([0, 0, 1, 1, 3, 3, 4, 0]), (2, 6)),
        (np.array([1, 2, 3, 4, 5]), (0, 4)),
        (np.array([0, 0, 0, 0]), (None, None)),
    ]
)
def test_auto_scan_start_stop(rotators, expected):
    start, stop = _find_auto_scan_start_stop(rotators)
    assert start == expected[0]
    assert stop == expected[1]


@pytest.mark.parametrize(
    "rotators, startstop, expected",
    [
        (np.array([0, 0, 1, 2, 3, 4, 5, 6, 0, 0]),
            None,
            np.array([2, 3, 4, 5, 6, 7])),
        (np.array([0, 1, 2, 3, 4, 5, 6, 0, 0]),
            (2, 5),
            np.array([2, 2, 3, 4, 5])),
        (np.array([1, 3, 3, 4, 5, 5, 9, 0, 0]),
            None,
            np.array([0, 0, 1, 3, 4, 5, 5, 5, 6])),
        (np.array([0, 0, 1, 3, 3, 4, 5, 5, 9, 0, 0]),
            None,
            np.array([2, 2, 3, 5, 6, 7, 7, 7, 8])),
    ]
)
def test_guess_scan_index_grid(rotators, startstop, expected):
    if startstop is None:
        startstop = _find_auto_scan_start_stop(rotators)
    # non jit
    indices = _guess_scan_index_grid.py_func(rotators, startstop[0], startstop[1])
    assert np.all(indices == expected)
    # jit compiled
    indices = _guess_scan_index_grid(rotators, startstop[0], startstop[1])
    assert np.all(indices == expected)


def _dask_supports_assignment():
    # direct assignment as follows is possible in newer versions (>2021.04.1) of dask, for backward compatibility we use workaround
    return pversion.parse(dask.__version__) >= pversion.parse("2021.04.1")


@pytest.mark.parametrize(
    "filename, kwargs",
    [
        (
            os.path.join(DIRPATH, "test_tvips_2233_000.tvips"),
            {
                "scan_shape": "auto",
                "scan_start_frame": 20,
                "hysteresis": 1,
                "rechunking": False,
            }
        ),
        (
            os.path.join(DIRPATH, "test_tvips_2345_000.tvips"),
            {
                "scan_shape": (2, 3),
                "scan_start_frame": 0,
                "hysteresis": 0,
                "rechunking": "auto",
            }
        ),
        (
            os.path.join(DIRPATH, "test_tvips_2345_split_000.tvips"),
            {
                "scan_shape": (2, 2),
                "scan_start_frame": 2,
                "hysteresis": -1,
                "rechunking": {0: 1, 1: 1, 2: None, 3: None},
            }
        ),
    ]
)
@pytest.mark.parametrize("wsa", ["x", "y", None])
@pytest.mark.parametrize("lazy", [True, False])
def test_tvips_file_reader(filename, lazy, kwargs, wsa):
    signal = hs.load(filename, lazy=lazy, **kwargs, winding_scan_axis=wsa)
    signal_test = hs.load(filename, lazy=lazy)
    scs = kwargs.get("scan_shape", "auto")
    ssf = kwargs.get("scan_start_frame", 0)
    hyst = kwargs.get("hysteresis", 0)
    sigshape = signal_test.data.shape[-2:]
    if scs == "auto":
        scan_dim = int(np.sqrt(signal_test.data.shape[0]))
        scs = (scan_dim, scan_dim)
        ssf = 0
    signal_test.data = signal_test.data[ssf:].reshape((*scs, *sigshape))

    if not _dask_supports_assignment() and lazy:
        signal_test.compute()
        signal_test.data = signal_test.data.copy()
    if wsa == "x":
        signal_test.data[..., ::2, :, :, :] = signal_test.data[..., ::2, :, :, :][..., :, ::-1, :, :]
        signal_test.data[..., ::2, :, :, :] = np.roll(signal_test.data[..., ::2, :, :, :], hyst, axis=-3)
    elif wsa == "y":
        signal_test.data[..., ::2, :, :] = signal_test.data[..., ::2, :, :][..., ::-1, :, :, :]
        signal_test.data[..., ::2, :, :] = np.roll(signal_test.data[..., ::2, :, :], hyst, axis=-4)
    assert np.allclose(signal_test.data, signal.data)


@pytest.mark.xfail(raises=ValueError)
def test_read_fail_version():
    file = os.path.join(DIRPATH, "test_tvips_2345_split_000.tvips"),
    hs.load(file, scan_shape="auto")


@pytest.mark.xfail(raises=ValueError)
def test_read_fail_wind_axis():
    file = os.path.join(DIRPATH, "test_tvips_2345_split_000.tvips"),
    hs.load(file, scan_shape=(2, 3), winding_scan_axis="z")


@pytest.mark.xfail(raises=ValueError)
def test_read_fail_scan_shape():
    file = os.path.join(DIRPATH, "test_tvips_2345_split_000.tvips"),
    hs.load(file, scan_shape=(3, 3))


@pytest.mark.xfail(raises=ValueError)
def test_write_fail_signal_type():
    with tempfile.TemporaryDirectory() as tdir:
        fake_signal = hs.signals.BaseSignal(np.zeros((1, 2, 3, 4)))
        path = os.path.join(tdir, "test_000.tvips")
        fake_signal.save(path)


@pytest.mark.parametrize(
    "sig, meta, max_file_size, fheb",
    [
        ("fake_signal_3D", "diffraction", None, 0),
        ("fake_signal_4D", "imaging", None, 0),
        ("fake_signal_5D", "diffraction", None, 0),
        ("fake_signal_4D", "diffraction", 100, 0),
        ("fake_signal_5D", "imaging", 100, 20),
        ("fake_signal_5D", None, 700, 10),
    ]
)
@pytest.mark.parametrize("lazy", [True, False])
def test_file_writer(sig, meta, max_file_size, fheb, fake_signals, fake_metadatas, lazy):
    signal = fake_signals[sig]
    if lazy:
        signal = signal.as_lazy()
    if meta is not None:
        metadata = fake_metadatas[meta]
        signal.metadata.add_dictionary(metadata.as_dictionary())
    metadata = signal.metadata
    with tempfile.TemporaryDirectory() as tmp:
        filepath = os.path.join(tmp, "test_tvips_save_000.tvips")
        scan_shape = signal.axes_manager.navigation_shape
        file_writer(filepath, signal, max_file_size=max_file_size, frame_header_extra_bytes=fheb)
        if max_file_size is None:
            assert len(os.listdir(tmp)) == 1
        else:
            assert len(os.listdir(tmp)) >= 1
        filepath_load = os.path.join(tmp, "test_tvips_save_000.tvips")
        dtc = file_reader(filepath_load, scan_shape=scan_shape[::-1], lazy=False)[0]
        np.testing.assert_allclose(signal.data, dtc['data'])
        assert signal.data.dtype == dtc['data'].dtype
        if metadata and meta is not None:
            assert dtc["metadata"]["General"]["date"] == metadata.General.date
            assert dtc["metadata"]["General"]["time"] == metadata.General.time
            assert dtc["metadata"]["Acquisition_instrument"]["TEM"]['beam_energy'] == metadata.Acquisition_instrument.TEM.beam_energy
            assert dtc["metadata"]["Acquisition_instrument"]["TEM"]['beam_current'] == metadata.Acquisition_instrument.TEM.beam_current
        gc.collect()


@pytest.mark.xfail(raises=ValueError)
def test_file_writer_fail():
    signal = hs.signals.Signal1D(np.array([1, 2, 3]))
    with tempfile.TemporaryDirectory() as tmp:
        filepath = os.path.join(tmp, "test.tvips")
        file_writer(filepath, signal)
