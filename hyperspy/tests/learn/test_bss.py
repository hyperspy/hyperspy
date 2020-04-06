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

from hyperspy._signals.signal1d import Signal1D
from hyperspy._signals.signal2d import Signal2D
from hyperspy.datasets import artificial_data
from hyperspy.decorators import lazifyTestClass
from hyperspy.misc.machine_learning.import_sklearn import sklearn_installed


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


@pytest.mark.parametrize("whiten_method", ["pca", "zca"])
def test_orthomax(whiten_method):
    s = artificial_data.get_core_loss_eels_line_scan_signal()
    s.decomposition()
    s.blind_source_separation(2, algorithm="orthomax", whiten_method=whiten_method)

    s.learning_results.bss_factors[:, 0] *= -1
    s._auto_reverse_bss_component("loadings")
    np.testing.assert_array_less(s.learning_results.bss_factors[:, 0], 0)
    np.testing.assert_array_less(0, s.learning_results.bss_factors[:, 1])
    s._auto_reverse_bss_component("factors")
    np.testing.assert_array_less(0, s.learning_results.bss_factors)

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


def test_algorithm_error():
    s = artificial_data.get_core_loss_eels_line_scan_signal()
    s.decomposition()

    with pytest.raises(ValueError, match="'algorithm' not recognised"):
        s.blind_source_separation(2, algorithm="uniform")


@pytest.mark.skipif(not sklearn_installed, reason="sklearn not installed")
@lazifyTestClass
class TestReverseBSS:
    def setup_method(self, method):
        s = artificial_data.get_core_loss_eels_line_scan_signal()
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


@pytest.mark.skipif(not sklearn_installed, reason="sklearn not installed")
@lazifyTestClass
class TestBSS1D:
    def setup_method(self, method):
        ics = np.random.laplace(size=(3, 1000))
        np.random.seed(1)
        mixing_matrix = np.random.random((100, 3))
        s = Signal1D(np.dot(mixing_matrix, ics))
        s.decomposition()

        mask_sig = s._get_signal_signal(dtype="bool")
        mask_sig.isig[5] = True

        mask_nav = s._get_navigation_signal(dtype="bool")
        mask_nav.isig[5] = True

        self.s = s
        self.mask_nav = mask_nav
        self.mask_sig = mask_sig

    def test_on_loadings(self):
        self.s.blind_source_separation(3, diff_order=0, fun="exp", on_loadings=False)
        s2 = self.s.as_signal1D(0)
        s2.decomposition()
        s2.blind_source_separation(3, diff_order=0, fun="exp", on_loadings=True)
        assert are_bss_components_equivalent(
            self.s.get_bss_factors(), s2.get_bss_loadings()
        )

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
        n_components = 2 if diff_order == 1 and on_loadings else 3

        self.s.blind_source_separation(
            n_components, diff_order=diff_order, mask=mask, on_loadings=on_loadings
        )


@pytest.mark.skipif(not sklearn_installed, reason="sklearn not installed")
@lazifyTestClass
class TestBSS2D:
    def setup_method(self, method):
        ics = np.random.laplace(size=(3, 1024))
        np.random.seed(1)
        mixing_matrix = np.random.random((100, 3))
        s = Signal2D(np.dot(mixing_matrix, ics).reshape((100, 32, 32)))
        for (axis, name) in zip(s.axes_manager._axes, ("z", "y", "x")):
            axis.name = name
        s.decomposition()

        mask_sig = s._get_signal_signal(dtype="bool")
        mask_sig.unfold()
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
        np.testing.assert_allclose(
            matrix, self.s.learning_results.unmixing_matrix, atol=1e-6
        )

    def test_diff_axes_string_without_mask(self):
        factors = self.s.get_decomposition_factors().inav[:3].diff(axis="x", order=1)
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
        factors = self.s.get_decomposition_factors().inav[:3].diff(axis="y", order=1)
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
