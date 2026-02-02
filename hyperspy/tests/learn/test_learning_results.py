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


import importlib

import numpy as np
import pytest

from hyperspy.signals import Signal1D

sklearn = importlib.util.find_spec("sklearn")
skip_sklearn = pytest.mark.skipif(sklearn is None, reason="sklearn not installed")


def test_learning_results_decom():
    rng = np.random.default_rng(123)

    s1 = Signal1D(rng.random(size=(20, 100)))
    s1.decomposition(output_dimension=2)

    out = str(s1.learning_results)
    assert "Decomposition parameters" in out
    assert "algorithm=SVD" in out
    assert "output_dimension=2" in out
    assert "Demixing parameters" not in out


@skip_sklearn
def test_learning_results_bss():
    rng = np.random.default_rng(123)

    s1 = Signal1D(rng.random(size=(20, 100)))
    s1.decomposition(output_dimension=2)
    s1.blind_source_separation(number_of_components=2)

    out = str(s1.learning_results)
    assert "Decomposition parameters" in out
    assert "Demixing parameters" in out
    assert "algorithm=sklearn_fastica" in out
    assert "n_components=2" in out
