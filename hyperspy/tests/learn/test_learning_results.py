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
from pathlib import Path

import numpy as np
import pytest

from hyperspy.learn._mva import LearningResults
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


def test_learning_results_save_load_round_trip(tmp_path):
    rng = np.random.default_rng(123)

    s1 = Signal1D(rng.random(size=(6, 12, 25)))
    s1.decomposition(output_dimension=2)

    path = tmp_path / "learning_results.npz"
    s1.learning_results.save(path, overwrite=True)

    lr = LearningResults()
    lr.load(path)
    np.testing.assert_allclose(lr.factors, s1.learning_results.factors)
    np.testing.assert_allclose(lr.loadings, s1.learning_results.loadings)
    np.testing.assert_allclose(
        lr.explained_variance, s1.learning_results.explained_variance
    )
    assert lr.decomposition_algorithm == "SVD"
    # Object-valued members are not loaded (pickle is never enabled)
    for name in ("mean", "centre", "unmixing_matrix", "cluster_labels"):
        assert getattr(lr, name) is None, name


class _MarkerWrite:
    """Reducer-bearing object equivalent to ``_pickle_marker``.

    Unpickling it writes the marker file; pickling it (storing it in an
    NPZ) does NOT execute the reduction.
    """

    def __init__(self, path):
        self.path = str(path)

    def __reduce__(self):
        return (Path(self.path).write_text, ("hs3699",))


def test_load_npz_benign_object_member_skipped(tmp_path):
    # The object member unpickles to a payload that writes a marker file;
    # loading must never unpickle it.
    marker_path = tmp_path / "marker.txt"
    obj = np.array([_MarkerWrite(marker_path)], dtype=object)
    factors = np.arange(12.0).reshape(6, 2)
    path = tmp_path / "learning_results.npz"
    assert not marker_path.exists()
    np.savez(path, factors=factors, mean=np.array([7.0]), unmixing_matrix=obj)
    # Pickling the object during savez must not execute its reduction
    assert not marker_path.exists()

    # Armed control: NPZ members load lazily, so the marker appears only
    # when the object member is accessed with pickle enabled.
    with np.load(path, allow_pickle=True) as f:
        assert not marker_path.exists()
        f["unmixing_matrix"]
    assert marker_path.exists()
    marker_path.unlink()

    lr = LearningResults()
    lr.load(path)
    np.testing.assert_array_equal(lr.factors, factors)
    assert lr.mean == 7.0
    assert lr.unmixing_matrix is None
    assert not marker_path.exists()


def test_load_ignores_unknown_and_dunder_keys(tmp_path):
    factors = np.arange(6.0).reshape(2, 3)
    path = tmp_path / "learning_results.npz"
    np.savez(
        path,
        __class__=np.array([1, 2]),
        unknown_field=np.array([3.0]),
        factors=factors,
    )

    lr = LearningResults()
    with pytest.warns(UserWarning, match="unknown field"):
        lr.load(path)
    assert lr.__class__ is LearningResults
    assert not hasattr(lr, "unknown_field")
    np.testing.assert_array_equal(lr.factors, factors)


def test_load_keeps_dynamic_instance_attribute(tmp_path):
    path = tmp_path / "learning_results.npz"
    np.savez(path, extra=np.array([9.0]), explained_variance=np.array([1.0, 2.0]))

    lr = LearningResults()
    lr.extra = "keep"
    with pytest.warns(UserWarning, match="unknown field 'extra'"):
        lr.load(path)
    assert lr.extra == "keep"
    np.testing.assert_allclose(lr.explained_variance, [1.0, 2.0])


def test_load_legacy_alias_V(tmp_path):
    path = tmp_path / "learning_results.npz"
    np.savez(path, V=np.array([1.5, 2.5]))

    lr = LearningResults()
    lr.load(path)
    np.testing.assert_allclose(lr.explained_variance, [1.5, 2.5])
    assert not hasattr(lr, "V")
