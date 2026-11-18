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


import numpy as np

from hyperspy.signals import BaseSignal


def test_deepcopy():
    s = BaseSignal([0])
    s.metadata.test = [0]
    s.original_metadata.test = [0]
    s_deepcopy = s.deepcopy()
    s.metadata.test.append(1)
    s.original_metadata.test.append(1)
    assert s.metadata.test == [0, 1]
    assert s.original_metadata.test == [0, 1]
    assert s_deepcopy.metadata.test == [0]
    assert s_deepcopy.original_metadata.test == [0]


_COPIED_LR_ARRAYS = (
    "factors",
    "explained_variance",
    "V",
    "cluster_sum_signals",
    "cluster_centroid_signals",
    "cluster_distances",
    "cluster_centroids",
)


def _seed_learning_results(s):
    shared = {"payload": [1, 2, 3]}
    lr = s.learning_results
    lr.factors = np.arange(6.0).reshape(3, 2)
    lr.explained_variance = np.array([2.5, 1.5])
    lr.decomposition_algorithm = "SVD"
    lr.algorithm = "legacy"
    lr.V = np.array([9.5, 8.5])
    lr.cluster_sum_signals = np.arange(3.0)
    lr.cluster_centroid_signals = np.arange(4.0)
    lr.cluster_distances = np.arange(5.0)
    lr.cluster_centroids = np.arange(6.0)
    lr.custom_public = shared
    lr._object = shared
    return shared


def _assert_copied_learning_results(copied, original, shared):
    c = copied.learning_results
    o = original.learning_results
    for name in _COPIED_LR_ARRAYS:
        assert getattr(c, name) is not getattr(o, name), name
        np.testing.assert_allclose(getattr(c, name), getattr(o, name))
    assert c.custom_public is not shared
    assert c.custom_public == shared
    # The two fields shared one input object before copying; the copies must
    # preserve that relationship internally and still be new objects.
    assert c.custom_public is c._object
    assert c.algorithm == "legacy"
    assert c.decomposition_algorithm == "SVD"
    assert "V" in c.__dict__
    assert "explained_variance" in c.__dict__


def test_deepcopy_learning_results_control():
    s = BaseSignal(np.arange(12 * 25 * 48, dtype=float).reshape(12, 25, 48))
    shared = _seed_learning_results(s)
    copied = s.deepcopy()
    _assert_copied_learning_results(copied, s, shared)


def test_as_lazy_learning_results_control():
    s = BaseSignal(np.arange(12 * 25 * 48, dtype=float).reshape(12, 25, 48))
    shared = _seed_learning_results(s)
    copied = s.as_lazy(copy_learning_results=True)
    assert copied._lazy
    _assert_copied_learning_results(copied, s, shared)
