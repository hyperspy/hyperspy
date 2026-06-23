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

"""Tests that the deprecated factors/loadings names emit
VisibleDeprecationWarning and return correct values via the new names."""

import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from hyperspy import signals
from hyperspy.api import load as hs_load
from hyperspy.exceptions import VisibleDeprecationWarning


def generate_low_rank_matrix(m=20, n=100, rank=5, random_seed=123):
    rng = np.random.RandomState(random_seed)
    U = rng.randn(m, rank)
    V = rng.randn(n, rank)
    X = abs(U @ V.T)
    X /= np.linalg.norm(X)
    return X


class TestLearningResultsDeprecation:
    """LearningResults property-level deprecations."""

    def setup_method(self, method):
        s = signals.Signal1D(generate_low_rank_matrix())
        s.decomposition(output_dimension=3)
        self.s = s

    def test_factors_property_warns_and_returns_components(self):
        s = self.s
        with pytest.warns(VisibleDeprecationWarning, match="deprecated"):
            result = s.learning_results.factors
        np.testing.assert_allclose(result, s.learning_results.components)

    def test_loadings_property_warns_and_returns_scores(self):
        s = self.s
        with pytest.warns(VisibleDeprecationWarning, match="deprecated"):
            result = s.learning_results.loadings
        np.testing.assert_allclose(result, s.learning_results.scores)

    def test_on_loadings_property_warns_and_returns_on_scores(self):
        s = self.s
        s.learning_results.on_scores = True
        with pytest.warns(VisibleDeprecationWarning, match="deprecated"):
            result = s.learning_results.on_loadings
        assert result is True

    def test_factors_setter_warns_and_sets_components(self):
        s = self.s
        new = np.random.random((100, 3))
        with pytest.warns(VisibleDeprecationWarning, match="deprecated"):
            s.learning_results.factors = new
        np.testing.assert_allclose(s.learning_results.components, new)

    def test_loadings_setter_warns_and_sets_scores(self):
        s = self.s
        new = np.random.random((20, 3))
        with pytest.warns(VisibleDeprecationWarning, match="deprecated"):
            s.learning_results.loadings = new
        np.testing.assert_allclose(s.learning_results.scores, new)


class TestBSSLearningResultsDeprecation:
    """LearningResults BSS property-level deprecations."""

    def setup_method(self, method):
        pytest.importorskip("sklearn", reason="sklearn required for BSS")
        rng = np.random.default_rng(123)
        S = rng.laplace(size=(3, 500))
        A = rng.random(size=(3, 3))
        s = signals.Signal1D(A @ S)
        s.decomposition()
        s.blind_source_separation(3)
        self.s = s

    def test_bss_factors_property_warns_and_returns_bss_components(self):
        s = self.s
        with pytest.warns(VisibleDeprecationWarning, match="deprecated"):
            result = s.learning_results.bss_factors
        np.testing.assert_allclose(result, s.learning_results.bss_components)

    def test_bss_loadings_property_warns_and_returns_bss_scores(self):
        s = self.s
        with pytest.warns(VisibleDeprecationWarning, match="deprecated"):
            result = s.learning_results.bss_loadings
        np.testing.assert_allclose(result, s.learning_results.bss_scores)

    def test_bss_factors_setter_warns_and_sets_bss_components(self):
        s = self.s
        new = np.random.random((100, 3))
        with pytest.warns(VisibleDeprecationWarning, match="deprecated"):
            s.learning_results.bss_factors = new
        np.testing.assert_allclose(s.learning_results.bss_components, new)

    def test_bss_loadings_setter_warns_and_sets_bss_scores(self):
        s = self.s
        new = np.random.random((3, 3))
        with pytest.warns(VisibleDeprecationWarning, match="deprecated"):
            s.learning_results.bss_loadings = new
        np.testing.assert_allclose(s.learning_results.bss_scores, new)


class TestGetMethodDeprecation:
    """get_decomposition_* and get_bss_* method deprecations."""

    def setup_method(self, method):
        pytest.importorskip("sklearn", reason="sklearn required for BSS")
        rng = np.random.default_rng(123)
        S = rng.laplace(size=(3, 500))
        A = rng.random(size=(3, 3))
        s = signals.Signal1D(A @ S)
        s.decomposition()
        s.blind_source_separation(3)
        self.s = s

    def test_get_decomposition_factors_warns(self):
        s = self.s
        with pytest.warns(VisibleDeprecationWarning, match="deprecated"):
            result = s.get_decomposition_factors()
        canonical = s.get_decomposition_components()
        np.testing.assert_allclose(result.data, canonical.data)

    def test_get_decomposition_loadings_warns(self):
        s = self.s
        with pytest.warns(VisibleDeprecationWarning, match="deprecated"):
            result = s.get_decomposition_loadings()
        canonical = s.get_decomposition_scores()
        np.testing.assert_allclose(result.data, canonical.data)

    def test_get_bss_factors_warns(self):
        s = self.s
        with pytest.warns(VisibleDeprecationWarning, match="deprecated"):
            result = s.get_bss_factors()
        canonical = s.get_bss_components()
        np.testing.assert_allclose(result.data, canonical.data)

    def test_get_bss_loadings_warns(self):
        s = self.s
        with pytest.warns(VisibleDeprecationWarning, match="deprecated"):
            result = s.get_bss_loadings()
        canonical = s.get_bss_scores()
        np.testing.assert_allclose(result.data, canonical.data)


class TestPlotMethodDeprecation:
    """plot_decomposition_* and plot_bss_* method deprecations."""

    def setup_method(self, method):
        pytest.importorskip("sklearn", reason="sklearn required for BSS")
        s = signals.Signal1D(generate_low_rank_matrix())
        s.decomposition(output_dimension=3)
        s.blind_source_separation(3)
        self.s = s

    def test_plot_decomposition_factors_warns(self):
        s = self.s
        with pytest.warns(VisibleDeprecationWarning, match="deprecated"):
            s.plot_decomposition_factors(0)

    def test_plot_decomposition_loadings_warns(self):
        s = self.s
        with pytest.warns(VisibleDeprecationWarning, match="deprecated"):
            s.plot_decomposition_loadings(0)

    def test_plot_bss_factors_warns(self):
        s = self.s
        with pytest.warns(VisibleDeprecationWarning, match="deprecated"):
            s.plot_bss_factors(0)

    def test_plot_bss_loadings_warns(self):
        s = self.s
        with pytest.warns(VisibleDeprecationWarning, match="deprecated"):
            s.plot_bss_loadings(0)


class TestNormalizeAndReverseDeprecation:
    """Deprecated string values for normalize/reverse target/criterion."""

    def setup_method(self, method):
        pytest.importorskip("sklearn", reason="sklearn required for BSS")
        self.s = signals.Signal1D(generate_low_rank_matrix())
        self.s.decomposition(output_dimension=3)
        self.s.blind_source_separation(3)

    def test_normalize_decomposition_components_target_factors_warns(self):
        s = self.s
        with pytest.warns(VisibleDeprecationWarning, match="deprecated"):
            s.normalize_decomposition_components(target="factors", function=np.sum)

    def test_normalize_decomposition_components_target_loadings_warns(self):
        s = self.s
        with pytest.warns(VisibleDeprecationWarning, match="deprecated"):
            s.normalize_decomposition_components(target="loadings", function=np.sum)

    def test_reverse_component_criterion_factors_warns(self):
        s = self.s
        with pytest.warns(VisibleDeprecationWarning, match="deprecated"):
            s.blind_source_separation(3, reverse_component_criterion="factors")

    def test_reverse_component_criterion_loadings_warns(self):
        s = self.s
        with pytest.warns(VisibleDeprecationWarning, match="deprecated"):
            s.blind_source_separation(3, reverse_component_criterion="loadings")


class TestBSSKwargDeprecation:
    """blind_source_separation kwarg deprecation."""

    def setup_method(self, method):
        pytest.importorskip("sklearn", reason="sklearn required for BSS")
        rng = np.random.default_rng(123)
        S = rng.laplace(size=(3, 500))
        A = rng.random(size=(3, 3))
        s = signals.Signal1D(A @ S)
        s.decomposition(output_dimension=3)
        self.s = s

    def test_on_loadings_kwarg_warns(self):
        s = self.s
        with pytest.warns(VisibleDeprecationWarning, match="deprecated"):
            s.blind_source_separation(3, on_loadings=True)


class TestExportKwargDeprecation:
    """Deprecated kwargs for export functions."""

    def setup_method(self, method):
        pytest.importorskip("sklearn", reason="sklearn required for BSS")
        rng = np.random.default_rng(123)
        S = rng.laplace(size=(3, 500))
        A = rng.random(size=(3, 3))
        s = signals.Signal1D(A @ S)
        s.decomposition(output_dimension=3)
        s.blind_source_separation(3)
        self.s = s

    def test_export_decomposition_results_factor_prefix_warns(self):
        s = self.s
        with TemporaryDirectory() as tmp:
            with pytest.warns(VisibleDeprecationWarning, match="deprecated"):
                s.export_decomposition_results(factor_prefix="f", folder=tmp)

    def test_export_decomposition_results_loading_prefix_warns(self):
        s = self.s
        with TemporaryDirectory() as tmp:
            with pytest.warns(VisibleDeprecationWarning, match="deprecated"):
                s.export_decomposition_results(loading_prefix="l", folder=tmp)

    def test_export_bss_results_factor_prefix_warns(self):
        s = self.s
        with TemporaryDirectory() as tmp:
            with pytest.warns(VisibleDeprecationWarning, match="deprecated"):
                s.export_bss_results(factor_prefix="f", folder=tmp)

    def test_export_bss_results_loading_prefix_warns(self):
        s = self.s
        with TemporaryDirectory() as tmp:
            with pytest.warns(VisibleDeprecationWarning, match="deprecated"):
                s.export_bss_results(loading_prefix="l", folder=tmp)


class TestPlotResultsNavigatorKwargDeprecation:
    """Deprecated navigator kwargs for plot_*_results."""

    def setup_method(self, method):
        pytest.importorskip("sklearn", reason="sklearn required for BSS")
        s = signals.Signal1D(generate_low_rank_matrix())
        s.decomposition(output_dimension=3)
        s.blind_source_separation(3)
        self.s = s

    def test_plot_decomposition_results_factors_navigator_warns(self):
        s = self.s
        with pytest.warns(VisibleDeprecationWarning, match="deprecated"):
            s.plot_decomposition_results(factors_navigator="auto")

    def test_plot_decomposition_results_loadings_navigator_warns(self):
        s = self.s
        with pytest.warns(VisibleDeprecationWarning, match="deprecated"):
            s.plot_decomposition_results(loadings_navigator="auto")

    def test_plot_bss_results_factors_navigator_warns(self):
        s = self.s
        with pytest.warns(VisibleDeprecationWarning, match="deprecated"):
            s.plot_bss_results(factors_navigator="auto")

    def test_plot_bss_results_loadings_navigator_warns(self):
        s = self.s
        with pytest.warns(VisibleDeprecationWarning, match="deprecated"):
            s.plot_bss_results(loadings_navigator="auto")


class TestInternalSilenceRegression:
    """Internal code MUST NOT emit VisibleDeprecationWarning alerts during
    normal use. This is the most important acceptance criterion.

    Using warnings.simplefilter("error", VisibleDeprecationWarning) means
    ANY internal use of deprecated names will raise instead of warn.
    """

    def test_decomposition_does_not_raise_internally(self):
        s = signals.Signal1D(generate_low_rank_matrix())
        with warnings.catch_warnings():
            warnings.simplefilter("error", VisibleDeprecationWarning)
            s.decomposition(output_dimension=3)

    def test_blind_source_separation_does_not_raise_internally(self):
        pytest.importorskip("sklearn", reason="sklearn required for BSS")
        rng = np.random.default_rng(123)
        S = rng.laplace(size=(3, 500))
        A = rng.random(size=(3, 3))
        s = signals.Signal1D(A @ S)
        with warnings.catch_warnings():
            warnings.simplefilter("error", VisibleDeprecationWarning)
            s.decomposition(output_dimension=3)
            s.blind_source_separation(3)

    def test_get_decomposition_model_does_not_raise_internally(self):
        s = signals.Signal1D(generate_low_rank_matrix())
        with warnings.catch_warnings():
            warnings.simplefilter("error", VisibleDeprecationWarning)
            s.decomposition(output_dimension=3)
            s.get_decomposition_model(3)

    def test_get_bss_model_does_not_raise_internally(self):
        pytest.importorskip("sklearn", reason="sklearn required for BSS")
        rng = np.random.default_rng(123)
        S = rng.laplace(size=(3, 500))
        A = rng.random(size=(3, 3))
        s = signals.Signal1D(A @ S)
        with warnings.catch_warnings():
            warnings.simplefilter("error", VisibleDeprecationWarning)
            s.decomposition(output_dimension=3)
            s.blind_source_separation(3)
            s.get_bss_model()

    def test_get_components_and_scores_do_not_raise_internally(self):
        pytest.importorskip("sklearn", reason="sklearn required for BSS")
        s = signals.Signal1D(generate_low_rank_matrix())
        with warnings.catch_warnings():
            warnings.simplefilter("error", VisibleDeprecationWarning)
            s.decomposition(output_dimension=3)
            s.blind_source_separation(3)
            s.get_decomposition_components()
            s.get_decomposition_scores()
            s.get_bss_components()
            s.get_bss_scores()

    def test_plot_methods_do_not_raise_internally(self):
        pytest.importorskip("sklearn", reason="sklearn required for BSS")
        s = signals.Signal1D(generate_low_rank_matrix())
        with warnings.catch_warnings():
            warnings.simplefilter("error", VisibleDeprecationWarning)
            s.decomposition(output_dimension=3)
            s.blind_source_separation(3)
            s.plot_decomposition_components()
            s.plot_decomposition_scores()
            s.plot_bss_components()
            s.plot_bss_scores()

    def test_export_methods_do_not_raise_internally(self):
        pytest.importorskip("sklearn", reason="sklearn required for BSS")
        rng = np.random.default_rng(123)
        S = rng.laplace(size=(3, 500))
        A = rng.random(size=(3, 3))
        s = signals.Signal1D(A @ S)
        with warnings.catch_warnings():
            warnings.simplefilter("error", VisibleDeprecationWarning)
            s.decomposition(output_dimension=3)
            s.blind_source_separation(3)
            with TemporaryDirectory() as tmp:
                s.export_decomposition_results(folder=tmp)
                s.export_bss_results(folder=tmp)

    def test_plot_results_do_not_raise_internally(self):
        pytest.importorskip("sklearn", reason="sklearn required for BSS")
        s = signals.Signal1D(generate_low_rank_matrix())
        with warnings.catch_warnings():
            warnings.simplefilter("error", VisibleDeprecationWarning)
            s.decomposition(output_dimension=3)
            s.blind_source_separation(3)
            s.plot_decomposition_results()
            s.plot_bss_results()

    def test_normalize_and_reverse_do_not_raise_internally(self):
        s = signals.Signal1D(generate_low_rank_matrix())
        with warnings.catch_warnings():
            warnings.simplefilter("error", VisibleDeprecationWarning)
            s.decomposition(output_dimension=3)
            s.normalize_decomposition_components(target="components", function=np.sum)
            s.normalize_decomposition_components(target="scores", function=np.sum)
            s.reverse_decomposition_component(0)


class TestSaveLoadRoundTrip:
    """Save/load round-trip preserves data and handles old-format keys."""

    def setup_method(self, method):
        s = signals.Signal1D(generate_low_rank_matrix())
        s.decomposition(output_dimension=3)
        self.s = s

    def test_save_load_new_names_round_trip_no_warnings(self):
        """Save with canonical names, load back — no warnings, data matches."""
        s = self.s
        original_components = s.learning_results.components.copy()
        original_scores = s.learning_results.scores.copy()

        with TemporaryDirectory() as tmp:
            fname = Path(tmp, "results.npz")
            with warnings.catch_warnings():
                warnings.simplefilter("error", VisibleDeprecationWarning)
                s.learning_results.save(fname)
            with warnings.catch_warnings():
                warnings.simplefilter("error", VisibleDeprecationWarning)
                s.learning_results.load(fname)

        np.testing.assert_allclose(s.learning_results.components, original_components)
        np.testing.assert_allclose(s.learning_results.scores, original_scores)

    def test_save_new_names_load_and_export_no_warnings(self):
        """Save → load → full signal save should work cleanly (#2093 regression)."""
        s = self.s
        with TemporaryDirectory() as tmp:
            fname1 = Path(tmp, "results.npz")
            s.learning_results.save(fname1)
            s.learning_results.load(fname1)
            fname2 = Path(tmp, "signal.hspy")
            with warnings.catch_warnings():
                warnings.simplefilter("error", VisibleDeprecationWarning)
                s.save(fname2)

    def test_load_old_format_keys_silently_migrates_to_new_names(self):
        """Loading an .npz with old-format keys silently migrates data to
        canonical names (no warning — load() bypasses property setters)."""
        s = self.s
        with TemporaryDirectory() as tmp:
            fname = Path(tmp, "oldformat.npz")
            # Manually craft an .npz with old-format key names
            np.savez(
                fname,
                factors=s.learning_results.components,
                loadings=s.learning_results.scores,
                explained_variance=s.learning_results.explained_variance,
                output_dimension=s.learning_results.output_dimension,
                mean=s.learning_results.mean,
                decomposition_algorithm=s.learning_results.decomposition_algorithm,
                poissonian_noise_normalized=s.learning_results.poissonian_noise_normalized,
                centre=s.learning_results.centre,
            )
            # Load — should NOT warn because load() bypasses property setters
            with warnings.catch_warnings():
                warnings.simplefilter("error", VisibleDeprecationWarning)
                s.learning_results.load(fname)
            # Data should be accessible via canonical names
            assert s.learning_results.components is not None
            assert s.learning_results.scores is not None
            np.testing.assert_allclose(
                s.learning_results.components, s.learning_results.components
            )


class TestCrossVersionFileCompat:
    """Verify that .hspy/.zspy files remain loadable across the
    factors/loadings → components/scores rename."""

    def setup_method(self, method):
        # Use different dimensions so axis reversals are verifiable
        data = generate_low_rank_matrix(m=12, n=25, rank=5, random_seed=42)
        self.s = signals.Signal1D(data)
        self.s.axes_manager[0].name = "x"
        self.s.axes_manager[1].name = "E"
        self.s.decomposition(output_dimension=3)
        self.s.learning_results.on_scores = True

    def test_load_dictionary_migrates_old_keys_to_new_names(self):
        """_load_dictionary with old factors/loadings keys sets canonical names."""
        s = self.s
        # Simulate an old-format file dict
        file_dict = {
            "data": s.data,
            "axes": [],
            "metadata": {},
            "original_metadata": {},
            "learning_results": {
                "factors": s.learning_results.components.copy(),
                "loadings": s.learning_results.scores.copy(),
                "explained_variance": s.learning_results.explained_variance,
                "output_dimension": 3,
                "mean": s.learning_results.mean,
                "decomposition_algorithm": "SVD",
                "poissonian_noise_normalized": False,
                "centre": None,
                "on_loadings": True,
            },
        }
        # Create fresh signal and load
        s2 = signals.Signal1D(np.zeros((12, 25)))
        with warnings.catch_warnings():
            warnings.simplefilter("error", VisibleDeprecationWarning)
            s2._load_dictionary(file_dict)
        # Data must be accessible via canonical names
        np.testing.assert_allclose(
            s2.learning_results.components, s.learning_results.components
        )
        np.testing.assert_allclose(
            s2.learning_results.scores, s.learning_results.scores
        )
        assert s2.learning_results.on_scores is True
        # Old keys must NOT be left as raw __dict__ entries
        lr_dict = s2.learning_results.__dict__
        assert "factors" not in lr_dict
        assert "loadings" not in lr_dict
        assert "on_loadings" not in lr_dict

    def test_to_dictionary_dual_writes_old_names(self):
        """_to_dictionary includes both canonical and deprecated key names."""
        s = self.s
        dic = s._to_dictionary()
        lr = dic["learning_results"]
        # Canonical names
        assert "components" in lr
        assert "scores" in lr
        assert "on_scores" in lr
        # Deprecated names written for backward compat
        assert "factors" in lr
        assert "loadings" in lr
        assert "on_loadings" in lr
        # Both reference the same data
        np.testing.assert_array_equal(lr["components"], lr["factors"])
        np.testing.assert_array_equal(lr["scores"], lr["loadings"])
        assert lr["on_scores"] == lr["on_loadings"]

    def test_hspy_round_trip_preserves_learning_results(self):
        """Save + load hspy keeps learning_results intact via canonical names."""
        s = self.s
        with TemporaryDirectory() as tmp:
            fpath = Path(tmp, "roundtrip.hspy")
            s.save(fpath)
            s2 = hs_load(fpath)
        # Core decomposition results
        np.testing.assert_allclose(
            s2.learning_results.components, s.learning_results.components
        )
        np.testing.assert_allclose(
            s2.learning_results.scores, s.learning_results.scores
        )
        np.testing.assert_allclose(
            s2.learning_results.explained_variance,
            s.learning_results.explained_variance,
        )
        assert s2.learning_results.output_dimension == 3
        assert s2.learning_results.decomposition_algorithm == "SVD"
        assert s2.learning_results.on_scores is True
