# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
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

import pytest

import hyperspy.api as hs
from hyperspy.external.progressbar import progressbar


@pytest.mark.parametrize("nb_progressbar", [False, True])
def test_progressbar(nb_progressbar):
    hs.preferences.General.nb_progressbar = nb_progressbar

    for i in progressbar(range(10), desc="Testing progressbar"):
        print(i)
