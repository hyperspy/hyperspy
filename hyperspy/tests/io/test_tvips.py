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


import gc
import os
import tempfile
import warnings
import traits.api as traits

import numpy as np
from skimage.exposure import rescale_intensity
import pytest

import hyperspy.api as hs
import hyperspy.io_plugins.tvips as tvips
from hyperspy.misc.utils import DictionaryTreeBrowser
from hyperspy.misc.utils import dummy_context_manager


try:
    WindowsError
except NameError:
    WindowsError = None


DIRPATH = os.path.dirname(__file__)
FILE1 = os.path.join(DIRPATH, "tvips_data", "test1.tvips")
FILE2 = os.path.join(DIRPATH, "tvips_data", "test2.tvips")


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
    fake_data = np.arange(120).reshape(4, 5, 6)
    fake_signal = hs.signals.Signal2D(fake_data)
    fake_signal.axes_manager[0].scale_as_quantity = "1 nm"
    fake_signal.axes_manager[1].scale_as_quantity = "1 1/nm"
    fake_signal.axes_manager[2].scale_as_quantity = "1 1/nm"
    return fake_signal


@pytest.fixture()
def fake_signal_4D():
    fake_data = np.arange(360).reshape(3, 4, 5, 6)
    fake_signal = hs.signals.Signal2D(fake_data)
    fake_signal.axes_manager[0].scale_as_quantity = "1 nm"
    fake_signal.axes_manager[1].scale_as_quantity = "1 nm"
    fake_signal.axes_manager[2].scale_as_quantity = "1 1/nm"
    fake_signal.axes_manager[3].scale_as_quantity = "1 1/nm"
    return fake_signal


@pytest.fixture()
def fake_signal_5D():
    fake_data = np.arange(720).reshape(2, 3, 4, 5, 6)
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


@pytest.fixture()
def save_path():
    with tempfile.TemporaryDirectory() as tmp:
        filepath = os.path.join(tmp, "tvips_data", "save_temp.tvips")
        yield filepath
        # Force files release (required in Windows)
        gc.collect()


@pytest.mark.parametrize("unit, expected_mode",
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
@pytest.mark.parametrize("sig",
        ["fake_signal_3D", "fake_signal_4D", "fake_signal_5D"])
def test_guess_image_mode(unit, expected_mode, sig, fake_signals):
    signal = fake_signals[sig]
    signal.axes_manager[-1].units = unit
    mode = tvips._guess_image_mode(signal)
    assert mode == expected_mode


@pytest.mark.parametrize("unit, expected_scale_factor, version, fheb",
                        [
                            ("1/pm", 1e3, 2, 0),
                            ("um", 1e3, 1, 60),
                            ("foo", 1, 2, 12),
                        ]
                        )
@pytest.mark.parametrize("sig",
        ["fake_signal_3D", "fake_signal_4D", "fake_signal_5D"])
@pytest.mark.parametrize("metadata",
        ["diffraction", "imaging", "confused", None])
def test_main_header_from_signal(unit, expected_scale_factor, version, fheb,
                                 sig, fake_signals, metadata, fake_metadatas):
    signal = fake_signals[sig]
    if metadata is not None:
        signal.metadata = fake_metadatas[metadata]
    signal.axes_manager[-1].units = unit
    signal.axes_manager[-2].units = unit
    original_scale_x = signal.axes_manager[-2].scale
    original_scale_y = signal.axes_manager[-1].scale
    if unit == "foo":
        cm = pytest.warns(UserWarning)
    else:
        cm = dummy_context_manager()
    with cm:
        header = tvips._get_main_header_from_signal(signal, version, fheb)
    assert header["size"] == np.dtype(tvips.TVIPS_RECORDER_GENERAL_HEADER).itemsize
    assert header["version"] == version
    assert header["dimx"] == signal.axes_manager[-2].size
    assert header["dimy"] == signal.axes_manager[-1].size
    assert header["offsetx"] == 0
    assert header["offsety"] == 0
    assert header["pixelsize"] == original_scale_x * expected_scale_factor
    assert header["frameheaderbytes"] == np.dtype(tvips.TVIPS_RECORDER_FRAME_HEADER).itemsize + fheb
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


@pytest.mark.parametrize("extra_bytes",
        [0, 20])
@pytest.mark.parametrize("sig",
        ["fake_signal_3D", "fake_signal_4D", "fake_signal_5D"])
def test_get_frame_record_dtype(sig, fake_signals, extra_bytes):
    signal = fake_signals[sig]
    dt_fh = tvips._get_frame_record_dtype_from_signal(signal, extra_bytes=extra_bytes)
    dimx = signal.axes_manager[-2].size
    dimy = signal.axes_manager[-1].size
    total_size = np.dtype(tvips.TVIPS_RECORDER_FRAME_HEADER).itemsize + extra_bytes + dimx * dimy * signal.data.itemsize
    assert dt_fh.itemsize == total_size


@pytest.mark.parametrize("filename",
                        [
                            pytest.param("foo_000.bla", marks=pytest.mark.xfail(raises=ValueError)),
                            pytest.param("foo_0.tvips", marks=pytest.mark.xfail(raises=ValueError)),
                            pytest.param("foo_001.tvips", marks=pytest.mark.xfail(raises=ValueError)),
                            pytest.param("foo_0000.TVIPS", marks=pytest.mark.xfail(raises=ValueError)),
                            ("foo_000.tvips"),
                            ("foo_000.TVIPS"),
                        ])
def test_valid_tvips_file(filename):
    isvalid = tvips._is_valid_first_tvips_file(filename)
    assert isvalid


@pytest.mark.parametrize("rotators, expected",
                        [
                            (np.array([0, 0, 1, 2, 3, 4, 5, 6, 0, 0]), (2, 7)),
                            (np.array([0, 1, 2, 3, 4, 5, 6, 0, 0]), (1, 6)),
                            (np.array([1, 2, 3, 4, 5, 6, 0, 0]), (0, 5)),
                            (np.array([0, 0, 1, 2, 3, 4, 5]), (2, 6)),
                            (np.array([0, 0, 1, 1, 3, 3, 4, 0]), (2, 6)),
                            (np.array([1, 2, 3, 4, 5]), (0, 4)),
                            (np.array([0, 0, 0, 0]), (None, None)),
                        ])
def test_auto_scan_start_stop(rotators, expected):
    start, stop = tvips._find_auto_scan_start_stop(rotators)
    assert start == expected[0]
    assert stop == expected[1]


@pytest.mark.parametrize("rotators, startstop, expected",
                        [
                            (np.array([0, 0, 1, 2, 3, 4, 5, 6, 0, 0]),
                             None,
                             np.array([2, 3, 4, 5, 6, 7])),
                            (np.array([0, 1, 2, 3, 4, 5, 6, 0, 0]),
                             (2, 5),
                             np.array([2, 3, 4, 5, 6])),
                            (np.array([1, 3, 3, 4, 5, 5, 9, 0, 0]),
                             None,
                             np.array([0, 0, 1, 2, 2, 3, 4, 5, 6])),
                        ])
def test_guess_scan_index_grid(rotators, startstop, expected):
    pass
    # If startstop is None:
    #    startstop = tvips._find_auto_scan_start_stop(rotators)
    # ndices = tvips._guess_scan_index_grid(rotators, startstop[0], startstop[1])
    # ssert np.all(indices == expected)
