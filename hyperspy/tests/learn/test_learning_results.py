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


import numpy as np
import pytest

from hyperspy.misc.machine_learning.import_sklearn import sklearn_installed
from hyperspy.signals import Signal1D


def test_learning_results_decom():
    rng = np.random.RandomState(123)

    s1 = Signal1D(rng.random_sample(size=(20, 100)))
    s1.decomposition(output_dimension=2)

    out = str(s1.learning_results)
    assert "Decomposition parameters" in out
    assert "algorithm=SVD" in out
    assert "output_dimension=2" in out
    assert "Demixing parameters" not in out


@pytest.mark.skipif(not sklearn_installed, reason="sklearn not installed")
def test_learning_results_bss():
    rng = np.random.RandomState(123)

    s1 = Signal1D(rng.random_sample(size=(20, 100)))
    s1.decomposition(output_dimension=2)
    s1.blind_source_separation(number_of_components=2)

    out = str(s1.learning_results)
    assert "Decomposition parameters" in out
    assert "Demixing parameters" in out
    assert "algorithm=sklearn_fastica" in out
    assert "n_components=2" in out
