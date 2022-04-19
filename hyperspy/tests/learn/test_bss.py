# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
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

from packaging.version import Version

import numpy as np
import pytest

from hyperspy._signals.signal1d import Signal1D
from hyperspy._signals.signal2d import Signal2D
from hyperspy.datasets import artificial_data
from hyperspy.decorators import lazifyTestClass
from hyperspy.misc.machine_learning.import_sklearn import sklearn_installed
from hyperspy.misc.machine_learning.tools import amari
from hyperspy.signals import BaseSignal


skip_sklearn = pytest.mark.skipif(not sklearn_installed, reason="sklearn not installed")


def are_bss_components_equivalent(c1_list, c2_list, atol=1e-4):
    """Check if two list of components are equivalent.

    To be equivalent they must differ by a max of `atol` except
    for an arbitrary -1 factor.

    Parameters
    ----------
    c1_list, c2_list: list of Signal instances.
        The components to check.
    atol: float
        Absolute tolerance for the amount that they can differ.

    Returns
    -------
    bool

    """
    matches = 0
    for c1 in c1_list:
        for c2 in c2_list:
            if np.allclose(c2.data, c1.data, atol=atol) or np.allclose(
                c2.data, -c1.data, atol=atol
            ):
                matches += 1
    return matches == len(c1_list)


def test_amari_distance(n=16, tol=1e-6):
    """Amari distance between matrix and its inverse should be 0."""
    rng = np.random.RandomState(123)

    A = rng.randn(n, n)
    W = np.linalg.inv(A)
    X = np.linalg.pinv(A)

    np.testing.assert_allclose(amari(W, A), 0.0, rtol=tol)
    np.testing.assert_allclose(amari(A, A), 2.912362, rtol=tol)
    np.testing.assert_allclose(amari(X, A), 0.0, rtol=tol)


@skip_sklearn
def test_bss_FastICA_object():
    """Tests that a simple sklearn pipeline is an acceptable algorithm."""
    rng = np.random.RandomState(123)
    S = rng.laplace(size=(3, 1000))
    A = rng.random_sample(size=(3, 3))
    s = Signal1D(A @ S)
    s.decomposition()

    from sklearn.decomposition import FastICA

    out = s.blind_source_separation(
        3, algorithm=FastICA(algorithm="deflation"), return_info=True
    )

    assert hasattr(out, "components_")


@skip_sklearn
def test_bss_pipeline():
    """Tests that a simple sklearn pipeline is an acceptable algorithm."""
    rng = np.random.RandomState(123)
    S = rng.laplace(size=(3, 1000))
    A = rng.random_sample(size=(3, 3))
    s = Signal1D(A @ S)
    s.decomposition()

    from sklearn.decomposition import FastICA
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    est = Pipeline(
        [("scaler", StandardScaler()), ("ica", FastICA(algorithm="deflation"))]
    )
    out = s.blind_source_separation(3, algorithm=est, return_info=True)

    assert hasattr(out, "steps")
    assert hasattr(out.named_steps["ica"], "components_")


@pytest.mark.parametrize("whiten_method", ["PCA", "ZCA"])
def test_orthomax(whiten_method):
    rng = np.random.RandomState(123)
    S = rng.laplace(size=(3, 500))
    A = rng.random_sample(size=(3, 3))
    s = Signal1D(A @ S)
    s.decomposition()
    s.blind_source_separation(3, algorithm="orthomax", whiten_method=whiten_method)

    W = s.learning_results.unmixing_matrix
    assert amari(W, A) < 0.5

    # Verify that we can change gamma for orthomax method
    s = artificial_data.get_core_loss_eels_line_scan_signal()
    s.decomposition()
    s.blind_source_separation(2, algorithm="orthomax", gamma=2)


def test_no_decomposition_error():
    s = artificial_data.get_core_loss_eels_line_scan_signal()

    with pytest.raises(AttributeError, match="A decomposition must be performed"):
        s.blind_source_separation(2)


def test_factors_error():
    s = artificial_data.get_core_loss_eels_line_scan_signal()
    s.decomposition()

    factors = s.get_decomposition_factors().data

    with pytest.raises(TypeError, match="`factors` must be a BaseSignal instance"):
        s.blind_source_separation(2, factors=factors)

    factors = BaseSignal(s.get_decomposition_factors().data)

    with pytest.raises(ValueError, match="`factors` must have navigation dimension"):
        s.blind_source_separation(2, factors=factors)


@skip_sklearn
@pytest.mark.parametrize("num_components", [None, 2])
def test_num_components(num_components):
    s = artificial_data.get_core_loss_eels_line_scan_signal()
    s.decomposition(output_dimension=2)
    s.blind_source_separation(number_of_components=num_components)


@skip_sklearn
def test_components_list():
    s = artificial_data.get_core_loss_eels_line_scan_signal()
    s.decomposition(output_dimension=3)
    s.blind_source_separation(comp_list=[0, 2])
    assert s.learning_results.unmixing_matrix.shape == (2, 2)


@skip_sklearn
def test_num_components_error():
    s = artificial_data.get_core_loss_eels_line_scan_signal()
    s.decomposition()
    s.learning_results.output_dimension = None

    with pytest.raises(
        ValueError, match="No `number_of_components` or `comp_list` provided"
    ):
        s.blind_source_separation(number_of_components=None)


def test_algorithm_error():
    s = artificial_data.get_core_loss_eels_line_scan_signal()
    s.decomposition()

    with pytest.raises(ValueError, match="'algorithm' not recognised"):
        s.blind_source_separation(2, algorithm="uniform")


@skip_sklearn
def test_normalize_components_errors():
    rng = np.random.RandomState(123)
    s = Signal1D(rng.random_sample(size=(20, 100)))
    s.decomposition()

    with pytest.raises(ValueError, match="called after s.blind_source_separation"):
        s.normalize_bss_components(target="loadings")

    s.blind_source_separation(2)

    with pytest.raises(ValueError, match="target must be"):
        s.normalize_bss_components(target="uniform")


@skip_sklearn
def test_sklearn_convergence_warning():
    # Import here to avoid error if sklearn missing
    from sklearn.exceptions import ConvergenceWarning

    rng = np.random.RandomState(123)
    ics = rng.laplace(size=(3, 1000))
    mixing_matrix = rng.random_sample(size=(100, 3))
    s = Signal1D(mixing_matrix @ ics)
    s.decomposition()

    with pytest.warns(ConvergenceWarning):
        s.blind_source_separation(
            number_of_components=10,
            algorithm="sklearn_fastica",
            diff_order=1,
            on_loadings=True,
            tol=1e-15,
            max_iter=3,
        )


@skip_sklearn
@pytest.mark.parametrize("whiten_method", [None, "PCA", "ZCA"])
def test_fastica_whiten_method(whiten_method):
    rng = np.random.RandomState(123)
    S = rng.laplace(size=(3, 1000))
    A = rng.random_sample(size=(3, 3))
    s = Signal1D(A @ S)
    s.decomposition()
    s.blind_source_separation(
        3, algorithm="sklearn_fastica", whiten_method=whiten_method
    )
    assert s.learning_results.unmixing_matrix.shape == A.shape


@skip_sklearn
@lazifyTestClass
class TestReverseBSS:
    def setup_method(self, method):
        rng = np.random.RandomState(123)
        S = rng.laplace(size=(3, 500))
        S -= 2 * S.min()  # Required to give us a positive dataset
        A = rng.random_sample(size=(3, 3))
        s = Signal1D(A @ S)
        s.decomposition()
        s.blind_source_separation(2)
        self.s = s

    def test_autoreverse_default(self):
        self.s.learning_results.bss_factors[:, 0] *= -1
        self.s._auto_reverse_bss_component("loadings")
        np.testing.assert_array_less(self.s.learning_results.bss_factors[:, 0], 0)
        np.testing.assert_array_less(0, self.s.learning_results.bss_factors[:, 1])
        self.s._auto_reverse_bss_component("factors")
        np.testing.assert_array_less(0, self.s.learning_results.bss_factors)

    def test_autoreverse_on_loading(self):
        self.s._auto_reverse_bss_component("loadings")
        np.testing.assert_array_less(0, self.s.learning_results.bss_factors)

    def test_reverse_wrong_parameter(self):
        with pytest.raises(ValueError):
            self.s.blind_source_separation(2, reverse_component_criterion="toto")


@skip_sklearn
@lazifyTestClass
class TestBSS1D:
    def setup_method(self, method):
        rng = np.random.RandomState(123)
        ics = rng.laplace(size=(3, 500))
        mixing_matrix = rng.random_sample(size=(100, 3))
        s = Signal1D(mixing_matrix @ ics)
        s.decomposition()

        mask_sig = s._get_signal_signal(dtype="bool")
        mask_sig.isig[5] = True

        mask_nav = s._get_navigation_signal(dtype="bool")
        mask_nav.isig[5] = True

        self.s = s
        self.mask_nav = mask_nav
        self.mask_sig = mask_sig

    def test_mask_error(self):
        with pytest.raises(ValueError):
            self.s.blind_source_separation(3, mask=self.mask_sig.data)

    def test_mask_shape_error(self):
        with pytest.raises(ValueError):
            self.s.blind_source_separation(3, mask=self.mask_nav)

    def test_on_loadings(self):
        self.s.blind_source_separation(3, diff_order=0, fun="exp", on_loadings=False)
        s2 = self.s.as_signal1D(0)
        s2.decomposition()
        s2.blind_source_separation(3, diff_order=0, fun="exp", on_loadings=True)
        assert are_bss_components_equivalent(
            self.s.get_bss_factors(), s2.get_bss_loadings()
        )

    @pytest.mark.filterwarnings("ignore:FastICA did not converge")
    @pytest.mark.parametrize("on_loadings", [True, False])
    @pytest.mark.parametrize("diff_order", [0, 1])
    def test_mask_diff_order(self, on_loadings, diff_order):
        if on_loadings:
            mask = self.mask_nav
            self.s.learning_results.loadings[5, :] = np.nan
        else:
            mask = self.mask_sig
            self.s.learning_results.factors[5, :] = np.nan

        # This preserves the old test behaviour. Without it,
        # we get "ConvergenceWarning: FastICA did not converge."
        # We test for convergence warnings separately
        n_components = 2 if diff_order == 1 and on_loadings else 3

        self.s.blind_source_separation(
            n_components, diff_order=diff_order, mask=mask, on_loadings=on_loadings
        )


@skip_sklearn
@lazifyTestClass
class TestBSS2D:
    def setup_method(self, method):
        rng = np.random.RandomState(123)
        ics = rng.laplace(size=(3, 256))
        mixing_matrix = rng.random_sample(size=(100, 3))
        s = Signal2D((mixing_matrix @ ics).reshape((100, 16, 16)))
        for (axis, name) in zip(s.axes_manager._axes, ("z", "y", "x")):
            axis.name = name
        s.decomposition()

        mask_sig = s._get_signal_signal(dtype="bool")
        mask_sig.unfold()
        mask_sig.data[:] = False
        mask_sig.isig[5] = True
        mask_sig.fold()

        mask_nav = s._get_navigation_signal(dtype="bool")
        mask_nav.unfold()
        mask_nav.isig[5] = True
        mask_nav.fold()

        self.s = s
        self.mask_nav = mask_nav
        self.mask_sig = mask_sig

    def test_diff_axes_string_with_mask(self):
        self.s.learning_results.factors[5, :] = np.nan
        factors = self.s.get_decomposition_factors().inav[:3]
        if self.mask_sig._lazy:
            self.mask_sig.compute()
        self.s.blind_source_separation(
            3,
            diff_order=0,
            fun="exp",
            on_loadings=False,
            factors=factors.diff(axis="x", order=1),
            mask=self.mask_sig.diff(axis="x", order=1),
        )
        matrix = self.s.learning_results.unmixing_matrix.copy()
        self.s.blind_source_separation(
            3,
            diff_order=1,
            fun="exp",
            on_loadings=False,
            diff_axes=["x"],
            mask=self.mask_sig,
        )
        matrix = self.s.learning_results.unmixing_matrix.copy()
        self.mask_sig.change_dtype("float")
        self.mask_sig.data[self.mask_sig.data == 1] = np.nan
        self.mask_sig.fold()
        self.mask_sig = self.mask_sig.derivative(axis="x")
        self.mask_sig.data[np.isnan(self.mask_sig.data)] = 1
        self.mask_sig.change_dtype("bool")
        self.s.blind_source_separation(
            3, diff_order=0, fun="exp", on_loadings=False,
            factors=factors.derivative(axis="x", order=1),
            mask=self.mask_sig)
        np.testing.assert_allclose(
            matrix, self.s.learning_results.unmixing_matrix, atol=1e-5
        )

    def test_diff_axes_string_without_mask(self):
        factors = self.s.get_decomposition_factors().inav[:3].derivative(
            axis="x", order=1)
        self.s.blind_source_separation(
            3, diff_order=0, fun="exp", on_loadings=False, factors=factors
        )
        matrix = self.s.learning_results.unmixing_matrix.copy()
        self.s.blind_source_separation(
            3, diff_order=1, fun="exp", on_loadings=False, diff_axes=["x"],
        )
        np.testing.assert_allclose(
            matrix, self.s.learning_results.unmixing_matrix, atol=1e-3
        )

    def test_diff_axes_without_mask(self):
        factors = self.s.get_decomposition_factors().inav[:3].derivative(
            axis="y", order=1)
        self.s.blind_source_separation(
            3, diff_order=0, fun="exp", on_loadings=False, factors=factors
        )
        matrix = self.s.learning_results.unmixing_matrix.copy()
        self.s.blind_source_separation(
            3, diff_order=1, fun="exp", on_loadings=False, diff_axes=[2],
        )
        np.testing.assert_allclose(
            matrix, self.s.learning_results.unmixing_matrix, atol=1e-3
        )

    def test_on_loadings(self):
        self.s.blind_source_separation(3, diff_order=0, fun="exp", on_loadings=False)
        s2 = self.s.as_signal1D(0)
        s2.decomposition()
        s2.blind_source_separation(3, diff_order=0, fun="exp", on_loadings=True)
        assert are_bss_components_equivalent(
            self.s.get_bss_factors(), s2.get_bss_loadings()
        )

    @pytest.mark.parametrize("diff_order", [0, 1])
    def test_mask_diff_order_0(self, diff_order):
        self.s.learning_results.factors[5, :] = np.nan
        self.s.blind_source_separation(3, diff_order=diff_order, mask=self.mask_sig)

    def test_mask_diff_order_1_diff_axes(self):
        self.s.learning_results.factors[5, :] = np.nan
        self.s.blind_source_separation(
            3, diff_order=1, mask=self.mask_sig, diff_axes=["x"]
        )

    def test_mask_diff_order_0_on_loadings(self):
        self.s.learning_results.loadings[5, :] = np.nan
        self.s.blind_source_separation(
            3, diff_order=0, mask=self.mask_nav, on_loadings=True
        )

    def test_mask_diff_order_1_on_loadings(self):
        s = self.s.to_signal1D()
        s.decomposition()
        if hasattr(s.learning_results.loadings, "compute"):
            s.learning_results.loadings = s.learning_results.loadings.compute()
        s.learning_results.loadings[5, :] = np.nan
        s.blind_source_separation(3, diff_order=1, mask=self.mask_sig, on_loadings=True)

    def test_mask_diff_order_1_on_loadings_diff_axes(self):
        s = self.s.to_signal1D()
        s.decomposition()
        if hasattr(s.learning_results.loadings, "compute"):
            s.learning_results.loadings = s.learning_results.loadings.compute()
        s.learning_results.loadings[5, :] = np.nan
        s.blind_source_separation(
            3, diff_order=1, mask=self.mask_sig, on_loadings=True, diff_axes=["x"]
        )


class TestPrintInfo:
    def setup_method(self, method):
        rng = np.random.RandomState(123)
        self.s = Signal1D(rng.random_sample(size=(20, 100)))
        self.s.decomposition(output_dimension=2)

    def test_bss(self, capfd):
        self.s.blind_source_separation(2, algorithm="orthomax")
        captured = capfd.readouterr()
        assert "Blind source separation info:" in captured.out

    @skip_sklearn
    def test_bss_sklearn(self, capfd):
        self.s.blind_source_separation(2)
        captured = capfd.readouterr()
        assert "Blind source separation info:" in captured.out
        assert "scikit-learn estimator:" in captured.out


class TestReturnInfo:
    def setup_method(self, method):
        rng = np.random.RandomState(123)
        self.s = Signal1D(rng.random_sample(size=(20, 100)))
        self.s.decomposition(output_dimension=2)

    def test_bss_not_supported(self):
        assert (
            self.s.blind_source_separation(algorithm="orthomax", return_info=True)
            is None
        )

    @skip_sklearn
    def test_bss_supported_return_true(self):
        assert self.s.blind_source_separation(return_info=True) is not None

    @skip_sklearn
    def test_bss_supported_return_false(self):
        assert self.s.blind_source_separation(return_info=False) is None
