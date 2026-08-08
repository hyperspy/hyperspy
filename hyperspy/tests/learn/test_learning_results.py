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


def test_learning_results_bH_attribute():
    """bH is a proper class attribute and survives save/load round-trip."""
    from hyperspy.learn._mva import LearningResults

    lr = LearningResults()
    rng = np.random.default_rng(42)
    lr.bH = rng.random(size=(25,))
    assert lr.bH is not None
    np.testing.assert_array_equal(lr.bH, lr.__dict__["bH"])


def test_populate_from_result_decomposition():
    """_populate_from_result transfers all decomposition attributes."""
    from hyperspy.learn._mva import LearningResults

    rng = np.random.default_rng(42)
    components = rng.random(size=(3, 25))
    scores = rng.random(size=(12, 3))
    explained_variance = rng.random(size=(3,))
    bH = rng.random(size=(25,))

    class FakeDecompositionResult:
        def __init__(self):
            self.components = components
            self.scores = scores
            self.explained_variance = explained_variance
            self.explained_variance_ratio = None
            self.mean = None
            self.bH = bH
            self._source = {"decomposition_algorithm": "SVD", "unfolded": False}
            self.params = {
                "algorithm": "SVD",
                "output_dimension": 3,
                "centre": None,
                "normalize_poissonian_noise": False,
            }

    lr = LearningResults()
    lr._populate_from_result(FakeDecompositionResult())
    np.testing.assert_allclose(lr.components, components)
    np.testing.assert_allclose(lr.scores, scores)
    np.testing.assert_allclose(lr.explained_variance, explained_variance)
    np.testing.assert_allclose(lr.bH, bH)
    assert lr.decomposition_algorithm == "SVD"
    assert lr.output_dimension == 3


def test_write_to_signal_classmethod():
    """write_to_signal classmethod populates signal.learning_results."""
    from hyperspy.learn._mva import LearningResults

    rng = np.random.default_rng(42)
    components = rng.random(size=(3, 25))
    scores = rng.random(size=(5, 3))

    class FakeDecompositionResult:
        def __init__(self):
            self.components = components
            self.scores = scores
            self.explained_variance = None
            self.explained_variance_ratio = None
            self.mean = None
            self.bH = None
            self._source = {}
            self.params = {}

    s = Signal1D(rng.random(size=(5, 25)))
    s.learning_results = None

    LearningResults.write_to_signal(FakeDecompositionResult(), s)
    assert s.learning_results is not None
    np.testing.assert_allclose(s.learning_results.components, components)
    np.testing.assert_allclose(s.learning_results.scores, scores)


def test_learning_results_bH_save_load():
    """bH survives an .npz save/load round-trip."""
    import warnings
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from hyperspy.exceptions import VisibleDeprecationWarning
    from hyperspy.learn._mva import LearningResults

    rng = np.random.default_rng(42)
    original_bH = rng.random(size=(25,))

    lr = LearningResults()
    lr.bH = original_bH
    lr.components = rng.random(size=(25, 3))
    lr.scores = rng.random(size=(12, 3))
    lr.decomposition_algorithm = "SVD"

    with TemporaryDirectory() as tmp:
        fname = Path(tmp, "bH_test.npz")
        with warnings.catch_warnings():
            warnings.simplefilter("error", VisibleDeprecationWarning)
            lr.save(fname)
        lr2 = LearningResults()
        lr2.load(fname)

    np.testing.assert_allclose(lr2.bH, original_bH)
    np.testing.assert_allclose(lr2.components, lr.components)
    np.testing.assert_allclose(lr2.scores, lr.scores)
