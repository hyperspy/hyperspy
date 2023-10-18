# -*- coding: utf-8 -*-
# Copyright 2007-2023 The HyperSpy developers
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

from pathlib import Path

import numpy as np
import pytest

import hyperspy.api as hs
from hyperspy import __version__ as hs_version


PATH = Path(__file__).resolve()



def test_load_save_filereader_metadata(tmp_path):
    # tests that original FileReader metadata is correctly persisted and
    # appended through a save and load cycle

    fname = PATH.parent / "drawing" / "data" / "Cr_L_cl.hspy"
    s = hs.load(fname)
    assert s.metadata.General.FileIO.Number_0.io_plugin == \
           'rsciio.hspy'
    assert s.metadata.General.FileIO.Number_0.operation == 'load'
    assert s.metadata.General.FileIO.Number_0.hyperspy_version == hs_version

    f = tmp_path / "temp"
    s.save(f)
    expected = {
        '0': {
            'io_plugin': 'rsciio.hspy',
            'operation': 'load',
            'hyperspy_version': hs_version
        },
        '1': {
            'io_plugin': 'rsciio.hspy',
            'operation': 'save',
            'hyperspy_version': hs_version
        },
        '2': {
            'io_plugin': 'rsciio.hspy',
            'operation': 'load',
            'hyperspy_version': hs_version
        },
    }
    del s.metadata.General.FileIO.Number_0.timestamp  # runtime dependent
    del s.metadata.General.FileIO.Number_1.timestamp  # runtime dependent
    assert \
        s.metadata.General.FileIO.Number_0.as_dictionary() == expected['0']
    assert \
        s.metadata.General.FileIO.Number_1.as_dictionary() == expected['1']

    t = hs.load(tmp_path / "temp.hspy")
    del t.metadata.General.FileIO.Number_0.timestamp  # runtime dependent
    del t.metadata.General.FileIO.Number_1.timestamp  # runtime dependent
    del t.metadata.General.FileIO.Number_2.timestamp  # runtime dependent
    assert t.metadata.General.FileIO.as_dictionary() == expected
