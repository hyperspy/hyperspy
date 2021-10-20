import pytest
import numpy as np

from hyperspy._signals.signal1d import Signal1D
from hyperspy._signals.signal2d import Signal2D
from hyperspy.misc.machine_learning.import_sklearn import sklearn_installed


def are_bss_components_equivalent(c1_list, c2_list, atol=1e-4):
    """Check if two list of components are equivalent.

    To be equivalent they must differ by a max of `atol` except
    for an arbitraty -1 factor.

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
            if (np.allclose(c2.data, c1.data, atol=atol) or
                    np.allclose(c2.data, -c1.data, atol=atol)):
                matches += 1
    return matches == len(c1_list)


class TestBSS1D:

    def setup_method(self, method):
        ics = np.random.laplace(size=(3, 1000))
        np.random.seed(1)
        mixing_matrix = np.random.random((100, 3))
        self.s = Signal1D(np.dot(mixing_matrix, ics))
        self.s.decomposition()

    @pytest.mark.skipif(not sklearn_installed, reason="sklearn not installed")
    def test_on_loadings(self):
        self.s.blind_source_separation(
            3, diff_order=0, fun="exp", on_loadings=False)
        s2 = self.s.as_signal1D(0)
        s2.decomposition()
        s2.blind_source_separation(
            3, diff_order=0, fun="exp", on_loadings=True)
        assert are_bss_components_equivalent(
            self.s.get_bss_factors(), s2.get_bss_loadings())

    @pytest.mark.skipif(not sklearn_installed, reason="sklearn not installed")
    def test_mask_diff_order_0(self):
        # This test, unlike most other tests, either passes or raises an error.
        # It is designed to test if the mask is correctly dilated inside the
        # `blind_source_separation_method`. If the mask is not correctely
        # dilated the nan in the loadings should raise an error.
        mask = self.s._get_signal_signal(dtype="bool")
        mask.isig[5] = True
        self.s.learning_results.factors[5, :] = np.nan
        self.s.blind_source_separation(3, diff_order=0, mask=mask)

    @pytest.mark.skipif(not sklearn_installed, reason="sklearn not installed")
    def test_mask_diff_order_1(self):
        # This test, unlike most other tests, either passes or raises an error.
        # It is designed to test if the mask is correctly dilated inside the
        # `blind_source_separation_method`. If the mask is not correctely
        # dilated the nan in the loadings should raise an error.
        mask = self.s._get_signal_signal(dtype="bool")
        mask.isig[5] = True
        self.s.learning_results.factors[5, :] = np.nan
        self.s.blind_source_separation(3, diff_order=1, mask=mask)

    @pytest.mark.skipif(not sklearn_installed, reason="sklearn not installed")
    def test_mask_diff_order_0_on_loadings(self):
        # This test, unlike most other tests, either passes or raises an error.
        # It is designed to test if the mask is correctly dilated inside the
        # `blind_source_separation_method`. If the mask is not correctely
        # dilated the nan in the loadings should raise an error.
        mask = self.s._get_navigation_signal(dtype="bool")
        mask.isig[5] = True
        self.s.learning_results.loadings[5, :] = np.nan
        self.s.blind_source_separation(3, diff_order=0, mask=mask,
                                       on_loadings=True)

    @pytest.mark.skipif(not sklearn_installed, reason="sklearn not installed")
    def test_mask_diff_order_1_on_loadings(self):
        # This test, unlike most other tests, either passes or raises an error.
        # It is designed to test if the mask is correctly dilated inside the
        # `blind_source_separation_method`. If the mask is not correctely
        # dilated the nan in the loadings should raise an error.
        mask = self.s._get_navigation_signal(dtype="bool")
        mask.isig[5] = True
        self.s.learning_results.loadings[5, :] = np.nan
        self.s.blind_source_separation(3, diff_order=1, mask=mask,
                                       on_loadings=True)


class TestBSS2D:

    def setup_method(self, method):
        ics = np.random.laplace(size=(3, 1024))
        np.random.seed(1)
        mixing_matrix = np.random.random((100, 3))
        s = Signal2D(np.dot(mixing_matrix, ics).reshape((100, 32, 32)))
        for (axis, name) in zip(s.axes_manager._axes, ("z", "y", "x")):
            axis.name = name
        s.decomposition()
        self.s = s

    @pytest.mark.skipif(not sklearn_installed, reason="sklearn not installed")
    def test_diff_axes_string_with_mask(self):
        mask = self.s._get_signal_signal(dtype="bool")
        mask.unfold()
        mask.isig[5] = True
        mask.fold()
        self.s.learning_results.factors[5, :] = np.nan
        factors = self.s.get_decomposition_factors().inav[:3]
        self.s.blind_source_separation(
            3, diff_order=0, fun="exp", on_loadings=False,
            factors=factors.diff(axis="x", order=1),
            mask=mask.diff(axis="x", order=1))
        matrix = self.s.learning_results.unmixing_matrix.copy()
        self.s.blind_source_separation(
            3, diff_order=1, fun="exp", on_loadings=False,
            diff_axes=["x"], mask=mask
        )
        assert np.allclose(matrix, self.s.learning_results.unmixing_matrix,
                           atol=1e-6)

    @pytest.mark.skipif(not sklearn_installed, reason="sklearn not installed")
    def test_diff_axes_string_without_mask(self):
        factors = self.s.get_decomposition_factors().inav[:3].diff(
            axis="x", order=1)
        self.s.blind_source_separation(
            3, diff_order=0, fun="exp", on_loadings=False, factors=factors)
        matrix = self.s.learning_results.unmixing_matrix.copy()
        self.s.blind_source_separation(
            3, diff_order=1, fun="exp", on_loadings=False,
            diff_axes=["x"],
        )
        assert np.allclose(matrix, self.s.learning_results.unmixing_matrix,
                           atol=1e-3)

    @pytest.mark.skipif(not sklearn_installed, reason="sklearn not installed")
    def test_diff_axes_without_mask(self):
        factors = self.s.get_decomposition_factors().inav[:3].diff(
            axis="y", order=1)
        self.s.blind_source_separation(
            3, diff_order=0, fun="exp", on_loadings=False, factors=factors)
        matrix = self.s.learning_results.unmixing_matrix.copy()
        self.s.blind_source_separation(
            3, diff_order=1, fun="exp", on_loadings=False, diff_axes=[2],)
        assert np.allclose(matrix, self.s.learning_results.unmixing_matrix,
                           atol=1e-3)

    @pytest.mark.skipif(not sklearn_installed, reason="sklearn not installed")
    def test_on_loadings(self):
        self.s.blind_source_separation(
            3, diff_order=0, fun="exp", on_loadings=False)
        s2 = self.s.as_signal1D(0)
        s2.decomposition()
        s2.blind_source_separation(
            3, diff_order=0, fun="exp", on_loadings=True)
        assert are_bss_components_equivalent(
            self.s.get_bss_factors(), s2.get_bss_loadings())

    @pytest.mark.skipif(not sklearn_installed, reason="sklearn not installed")
    def test_mask_diff_order_0(self):
        # This test, unlike most other tests, either passes or raises an error.
        # It is designed to test if the mask is correctly dilated inside the
        # `blind_source_separation_method`. If the mask is not correctely
        # dilated the nan in the loadings should raise an error.
        mask = self.s._get_signal_signal(dtype="bool")
        mask.unfold()
        mask.isig[5] = True
        mask.fold()
        self.s.learning_results.factors[5, :] = np.nan
        self.s.blind_source_separation(3, diff_order=0, mask=mask)

    @pytest.mark.skipif(not sklearn_installed, reason="sklearn not installed")
    def test_mask_diff_order_1(self):
        # This test, unlike most other tests, either passes or raises an error.
        # It is designed to test if the mask is correctly dilated inside the
        # `blind_source_separation_method`. If the mask is not correctely
        # dilated the nan in the loadings should raise an error.
        mask = self.s._get_signal_signal(dtype="bool")
        mask.unfold()
        mask.isig[5] = True
        mask.fold()
        self.s.learning_results.factors[5, :] = np.nan
        self.s.blind_source_separation(3, diff_order=1, mask=mask)

    @pytest.mark.skipif(not sklearn_installed, reason="sklearn not installed")
    def test_mask_diff_order_1_diff_axes(self):
        # This test, unlike most other tests, either passes or raises an error.
        # It is designed to test if the mask is correctly dilated inside the
        # `blind_source_separation_method`. If the mask is not correctely
        # dilated the nan in the loadings should raise an error.
        mask = self.s._get_signal_signal(dtype="bool")
        mask.unfold()
        mask.isig[5] = True
        mask.fold()
        self.s.learning_results.factors[5, :] = np.nan
        self.s.blind_source_separation(3, diff_order=1, mask=mask,
                                       diff_axes=["x", ])

    @pytest.mark.skipif(not sklearn_installed, reason="sklearn not installed")
    def test_mask_diff_order_0_on_loadings(self):
        # This test, unlike most other tests, either passes or raises an error.
        # It is designed to test if the mask is correctly dilated inside the
        # `blind_source_separation_method`. If the mask is not correctely
        # dilated the nan in the loadings should raise an error.
        mask = self.s._get_navigation_signal(dtype="bool")
        mask.unfold()
        mask.isig[5] = True
        mask.fold()
        self.s.learning_results.loadings[5, :] = np.nan
        self.s.blind_source_separation(3, diff_order=0, mask=mask,
                                       on_loadings=True)

    @pytest.mark.skipif(not sklearn_installed, reason="sklearn not installed")
    def test_mask_diff_order_1_on_loadings(self):
        # This test, unlike most other tests, either passes or raises an error.
        # It is designed to test if the mask is correctly dilated inside the
        # `blind_source_separation_method`. If the mask is not correctely
        # dilated the nan in the loadings should raise an error.
        s = self.s.to_signal1D()
        s.decomposition()
        mask = s._get_navigation_signal(dtype="bool")
        mask.unfold()
        mask.isig[5] = True
        mask.fold()
        s.learning_results.loadings[5, :] = np.nan
        s.blind_source_separation(3, diff_order=1, mask=mask,
                                  on_loadings=True)

    @pytest.mark.skipif(not sklearn_installed, reason="sklearn not installed")
    def test_mask_diff_order_1_on_loadings_diff_axes(self):
        # This test, unlike most other tests, either passes or raises an error.
        # It is designed to test if the mask is correctly dilated inside the
        # `blind_source_separation_method`. If the mask is not correctely
        # dilated the nan in the loadings should raise an error.
        s = self.s.to_signal1D()
        s.decomposition()
        mask = s._get_navigation_signal(dtype="bool")
        mask.unfold()
        mask.isig[5] = True
        mask.fold()
        s.learning_results.loadings[5, :] = np.nan
        s.blind_source_separation(3, diff_order=1, mask=mask,
                                  on_loadings=True, diff_axes=["x"])
