# Copyright 2007-2024 The HyperSpy developers
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

import numpy as np

from hyperspy.axes import AxesManager
from hyperspy.signal import BaseSignal, _obj_in_dict2hspy


def test_obj_in_dict2hspy_signal():
    s_dict = BaseSignal([0, 1, 2])._to_dictionary()
    d = {"_sig_signal": s_dict}
    _obj_in_dict2hspy(d, False)
    assert isinstance(d["signal"], BaseSignal)
    np.testing.assert_allclose(d["signal"].data, [0, 1, 2])


def test_obj_in_dict2hspy_axes_manager():
    axes_list = [
        {
            "name": "a",
            "navigate": True,
            "offset": 0.0,
            "scale": 1.3,
            "size": 2,
            "units": "aa",
        },
        {
            "name": "b",
            "navigate": False,
            "offset": 1.0,
            "scale": 6.0,
            "size": 3,
            "units": "bb",
        },
    ]

    am = AxesManager(axes_list)
    d = {"_hspy_AxesManager_am": am._get_axes_dicts()}
    _obj_in_dict2hspy(d, False)
    assert isinstance(d["am"], AxesManager)
    assert am[0].scale == d["am"][0].scale


def test_signal_containers():
    s = BaseSignal([0, 1, 2])
    s2 = BaseSignal([4, 5, 6])
    s3 = BaseSignal([7, 8, 9])
    s.metadata.signal_container = [s2, s3]
    s_dict = s.metadata.as_dictionary()
    _obj_in_dict2hspy(s_dict, False)
    signal_container = s_dict["signal_container"]
    isinstance(signal_container[0], BaseSignal)
    np.testing.assert_allclose(signal_container[0].data, [4, 5, 6])
    isinstance(signal_container[1], BaseSignal)
    np.testing.assert_allclose(signal_container[1].data, [7, 8, 9])
