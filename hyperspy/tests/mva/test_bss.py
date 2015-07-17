import nose.tools
import numpy as np

from hyperspy.signals import Spectrum, Image


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

    def setUp(self):
        ics = np.random.laplace(size=(3, 1000))
        np.random.seed(1)
        mixing_matrix = np.random.random((100, 3))
        self.s = Spectrum(np.dot(mixing_matrix, ics))
        self.s.decomposition()

    def test_on_loadings(self):
        self.s.blind_source_separation(
            3, diff_order=0, fun="exp", on_loadings=False)
        s2 = self.s.as_spectrum(0)
        s2.decomposition()
        s2.blind_source_separation(
            3, diff_order=0, fun="exp", on_loadings=True)
        nose.tools.assert_true(are_bss_components_equivalent(
            self.s.get_bss_factors(), s2.get_bss_loadings()))

    def test_mask_diff_order_0(self):
        mask = self.s._get_signal_signal(dtype="bool")
        mask[5] = True
        self.s.learning_results.factors[5, :] = np.nan
        self.s.blind_source_separation(3, diff_order=0, mask=mask)

    def test_mask_diff_order_1(self):
        mask = self.s._get_signal_signal(dtype="bool")
        mask[5] = True
        self.s.learning_results.factors[5, :] = np.nan
        self.s.blind_source_separation(3, diff_order=1, mask=mask)

    def test_mask_diff_order_0_on_loadings(self):
        mask = self.s._get_navigation_signal(dtype="bool")
        mask[5] = True
        self.s.learning_results.loadings[5, :] = np.nan
        self.s.blind_source_separation(3, diff_order=0, mask=mask,
                                       on_loadings=True)

    def test_mask_diff_order_1_on_loadings(self):
        mask = self.s._get_navigation_signal(dtype="bool")
        mask[5] = True
        self.s.learning_results.loadings[5, :] = np.nan
        self.s.blind_source_separation(3, diff_order=1, mask=mask,
                                       on_loadings=True)


class TestBSS2D:

    def setUp(self):
        ics = np.random.laplace(size=(3, 1024))
        np.random.seed(1)
        mixing_matrix = np.random.random((100, 3))
        self.s = Image(np.dot(mixing_matrix, ics).reshape((100, 32, 32)))
        self.s.decomposition()

    def test_on_loadings(self):
        self.s.blind_source_separation(
            3, diff_order=0, fun="exp", on_loadings=False)
        s2 = self.s.as_spectrum(0)
        s2.decomposition()
        s2.blind_source_separation(
            3, diff_order=0, fun="exp", on_loadings=True)
        nose.tools.assert_true(are_bss_components_equivalent(
            self.s.get_bss_factors(), s2.get_bss_loadings()))

    def test_mask_diff_order_0(self):
        mask = self.s._get_signal_signal(dtype="bool")
        mask.unfold()
        mask[5] = True
        mask.fold()
        self.s.learning_results.factors[5, :] = np.nan
        self.s.blind_source_separation(3, diff_order=0, mask=mask)

    def test_mask_diff_order_1(self):
        mask = self.s._get_signal_signal(dtype="bool")
        mask.unfold()
        mask[5] = True
        mask.fold()
        self.s.learning_results.factors[5, :] = np.nan
        self.s.blind_source_separation(3, diff_order=1, mask=mask)

    def test_mask_diff_order_0_on_loadings(self):
        mask = self.s._get_navigation_signal(dtype="bool")
        mask.unfold()
        mask[5] = True
        mask.fold()
        self.s.learning_results.loadings[5, :] = np.nan
        self.s.blind_source_separation(3, diff_order=0, mask=mask,
                                       on_loadings=True)

    def test_mask_diff_order_1_on_loadings(self):
        s = self.s.to_spectrum()
        s.decomposition()
        mask = s._get_navigation_signal(dtype="bool")
        mask.unfold()
        mask[5] = True
        mask.fold()
        s.learning_results.loadings[5, :] = np.nan
        s.blind_source_separation(3, diff_order=1, mask=mask,
                                  on_loadings=True)
