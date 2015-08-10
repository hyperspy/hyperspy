import numpy as np

import nose.tools
from nose.tools import assert_true, raises
from hyperspy import signals


class TestNdAxes:

    def setUp(self):
        # Create three signals with dimensions:
        # s1 : <Signal, title: , dimensions: (4, 3, 2|2, 3)>
        # s2 : <Signal, title: , dimensions: (2, 3|4, 3, 2)>
        # s12 : <Signal, title: , dimensions: (2, 3|4, 3, 2)>
        # Where s12 data is transposed in respect to s2
        dc1 = np.random.random((2, 3, 4, 3, 2))
        dc2 = np.rollaxis(np.rollaxis(dc1, -1), -1)
        s1 = signals.Signal(dc1.copy())
        s2 = signals.Signal(dc2)
        s12 = signals.Signal(dc1.copy())
        for i, axis in enumerate(s1.axes_manager._axes):
            if i < 3:
                axis.navigate = True
            else:
                axis.navigate = False
        for i, axis in enumerate(s2.axes_manager._axes):
            if i < 2:
                axis.navigate = True
            else:
                axis.navigate = False
        for i, axis in enumerate(s12.axes_manager._axes):
            if i < 3:
                axis.navigate = False
            else:
                axis.navigate = True
        self.s1 = s1
        self.s2 = s2
        self.s12 = s12

    def test_consistensy(self):
        s1 = self.s1
        s2 = self.s2
        s12 = self.s12
        s1.decomposition()
        s2.decomposition()
        s12.decomposition()
        np.testing.assert_array_almost_equal(s2.learning_results.loadings,
                                             s12.learning_results.loadings)
        np.testing.assert_array_almost_equal(s2.learning_results.factors,
                                             s12.learning_results.factors)
        np.testing.assert_array_almost_equal(s1.learning_results.loadings,
                                             s2.learning_results.factors)
        np.testing.assert_array_almost_equal(s1.learning_results.factors,
                                             s2.learning_results.loadings)

    def test_consistensy_poissonian(self):
        s1 = self.s1
        s2 = self.s2
        s12 = self.s12
        s1.decomposition(normalize_poissonian_noise=True)
        s2.decomposition(normalize_poissonian_noise=True)
        s12.decomposition(normalize_poissonian_noise=True)
        np.testing.assert_array_almost_equal(s2.learning_results.loadings,
                                             s12.learning_results.loadings)
        np.testing.assert_array_almost_equal(s2.learning_results.factors,
                                             s12.learning_results.factors)
        np.testing.assert_array_almost_equal(s1.learning_results.loadings,
                                             s2.learning_results.factors)
        np.testing.assert_array_almost_equal(s1.learning_results.factors,
                                             s2.learning_results.loadings)


class TestGetExplainedVarinaceRation:

    def setUp(self):
        s = signals.Signal(np.empty(1))
        s.learning_results.explained_variance_ratio = np.empty(10)
        self.s = s

    def test_data(self):
        assert_true((self.s.get_explained_variance_ratio().data ==
                     self.s.learning_results.explained_variance_ratio).all())

    @raises(AttributeError)
    def test_no_evr(self):
        self.s.get_explained_variance_ration()


class TestReverseDecompositionComponent:

    def setUp(self):
        s = signals.Signal(np.empty(1))
        self.factors = np.ones([2, 3])
        self.loadings = np.ones([2, 3])
        s.learning_results.factors = self.factors.copy()
        s.learning_results.loadings = self.loadings.copy()
        self.s = s

    def test_reversal_factors_one_component_reversed(self):
        self.s.reverse_decomposition_component(0)
        assert_true((self.s.learning_results.factors[:, 0] ==
                     self.factors[:, 0] * -1).all())

    def test_reversal_loadings_one_component_reversed(self):
        self.s.reverse_decomposition_component(0)
        assert_true((self.s.learning_results.loadings[:, 0] ==
                     self.loadings[:, 0] * -1).all())

    def test_reversal_factors_one_component_not_reversed(self):
        self.s.reverse_decomposition_component(0)
        assert_true((self.s.learning_results.factors[:, 1:] ==
                     self.factors[:, 1:]).all())

    def test_reversal_loadings_one_component_not_reversed(self):
        self.s.reverse_decomposition_component(0)
        assert_true((self.s.learning_results.loadings[:, 1:] ==
                     self.loadings[:, 1:]).all())

    def test_reversal_factors_multiple_components_reversed(self):
        self.s.reverse_decomposition_component((0, 2))
        assert_true((self.s.learning_results.factors[:, (0, 2)] ==
                     self.factors[:, (0, 2)] * -1).all())

    def test_reversal_loadings_multiple_components_reversed(self):
        self.s.reverse_decomposition_component((0, 2))
        assert_true((self.s.learning_results.loadings[:, (0, 2)] ==
                     self.loadings[:, (0, 2)] * -1).all())

    def test_reversal_factors_multiple_components_not_reversed(self):
        self.s.reverse_decomposition_component((0, 2))
        assert_true((self.s.learning_results.factors[:, 1] ==
                     self.factors[:, 1]).all())

    def test_reversal_loadings_multiple_components_not_reversed(self):
        self.s.reverse_decomposition_component((0, 2))
        assert_true((self.s.learning_results.loadings[:, 1] ==
                     self.loadings[:, 1]).all())


class TestNormalizeComponents():

    def setUp(self):
        s = signals.Signal(np.empty(1))
        self.factors = np.ones([2, 3])
        self.loadings = np.ones([2, 3])
        s.learning_results.factors = self.factors.copy()
        s.learning_results.loadings = self.loadings.copy()
        s.learning_results.bss_factors = self.factors.copy()
        s.learning_results.bss_loadings = self.loadings.copy()
        self.s = s

    def test_normalize_bss_factors(self):
        s = self.s
        s.normalize_bss_components(target="factors",
                                   function=np.sum)
        nose.tools.assert_true(
            (s.learning_results.bss_factors == self.factors / 2.).all())
        nose.tools.assert_true(
            (s.learning_results.bss_loadings == self.loadings * 2.).all())

    def test_normalize_bss_loadings(self):
        s = self.s
        s.normalize_bss_components(target="loadings",
                                   function=np.sum)
        nose.tools.assert_true(
            (s.learning_results.bss_factors == self.factors * 2.).all())
        nose.tools.assert_true(
            (s.learning_results.bss_loadings == self.loadings / 2.).all())

    def test_normalize_decomposition_factors(self):
        s = self.s
        s.normalize_decomposition_components(target="factors",
                                             function=np.sum)
        nose.tools.assert_true(
            (s.learning_results.factors ==
             self.factors / 2.).all())
        nose.tools.assert_true(
            (s.learning_results.loadings ==
             self.loadings * 2.).all())

    def test_normalize_decomposition_loadings(self):
        s = self.s
        s.normalize_decomposition_components(target="loadings",
                                             function=np.sum)
        nose.tools.assert_true(
            (s.learning_results.factors ==
             self.factors * 2.).all())
        nose.tools.assert_true(
            (s.learning_results.loadings ==
             self.loadings / 2.).all())
