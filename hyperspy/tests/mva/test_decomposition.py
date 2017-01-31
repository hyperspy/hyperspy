import numpy as np
import pytest

from hyperspy import signals
from hyperspy.misc.machine_learning.import_sklearn import sklearn_installed


class TestNdAxes:

    def setup_method(self, method):
        # Create three signals with dimensions:
        # s1 : <BaseSignal, title: , dimensions: (4, 3, 2|2, 3)>
        # s2 : <BaseSignal, title: , dimensions: (2, 3|4, 3, 2)>
        # s12 : <BaseSignal, title: , dimensions: (2, 3|4, 3, 2)>
        # Where s12 data is transposed in respect to s2
        dc1 = np.random.random((2, 3, 4, 3, 2))
        dc2 = np.rollaxis(np.rollaxis(dc1, -1), -1)
        s1 = signals.BaseSignal(dc1.copy())
        s2 = signals.BaseSignal(dc2)
        s12 = signals.BaseSignal(dc1.copy())
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

    def test_consistency(self):
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

    def test_consistency_poissonian(self):
        s1 = self.s1
        s1n000 = self.s1.inav[0, 0, 0]
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
        # Check that views of the data don't change. See #871
        np.testing.assert_array_equal(s1.inav[0, 0, 0].data, s1n000.data)


class TestGetExplainedVarinaceRatio:

    def setup_method(self, method):
        s = signals.BaseSignal(np.empty(1))
        self.s = s

    def test_data(self):
        self.s.learning_results.explained_variance_ratio = np.asarray([2, 4])
        np.testing.assert_array_equal(
            self.s.get_explained_variance_ratio().data,
            np.asarray([2, 4]))

    def test_no_evr(self):
        with pytest.raises(AttributeError):
            self.s.get_explained_variance_ratio()


class TestReverseDecompositionComponent:

    def setup_method(self, method):
        s = signals.BaseSignal(np.zeros(1))
        self.factors = np.ones([2, 3])
        self.loadings = np.ones([2, 3])
        s.learning_results.factors = self.factors.copy()
        s.learning_results.loadings = self.loadings.copy()
        self.s = s

    def test_reversal_factors_one_component_reversed(self):
        self.s.reverse_decomposition_component(0)
        np.testing.assert_array_equal(self.s.learning_results.factors[:, 0],
                                      self.factors[:, 0] * -1)

    def test_reversal_loadings_one_component_reversed(self):
        self.s.reverse_decomposition_component(0)
        np.testing.assert_array_equal(self.s.learning_results.loadings[:, 0],
                                      self.loadings[:, 0] * -1)

    def test_reversal_factors_one_component_not_reversed(self):
        self.s.reverse_decomposition_component(0)
        np.testing.assert_array_equal(self.s.learning_results.factors[:, 1:],
                                      self.factors[:, 1:])

    def test_reversal_loadings_one_component_not_reversed(self):
        self.s.reverse_decomposition_component(0)
        np.testing.assert_array_equal(self.s.learning_results.loadings[:, 1:],
                                      self.loadings[:, 1:])

    def test_reversal_factors_multiple_components_reversed(self):
        self.s.reverse_decomposition_component((0, 2))
        np.testing.assert_array_equal(self.s.learning_results.factors[:, (0, 2)],
                                      self.factors[:, (0, 2)] * -1)

    def test_reversal_loadings_multiple_components_reversed(self):
        self.s.reverse_decomposition_component((0, 2))
        np.testing.assert_array_equal(self.s.learning_results.loadings[:, (0, 2)],
                                      self.loadings[:, (0, 2)] * -1)

    def test_reversal_factors_multiple_components_not_reversed(self):
        self.s.reverse_decomposition_component((0, 2))
        np.testing.assert_array_equal(self.s.learning_results.factors[:, 1],
                                      self.factors[:, 1])

    def test_reversal_loadings_multiple_components_not_reversed(self):
        self.s.reverse_decomposition_component((0, 2))
        np.testing.assert_array_equal(self.s.learning_results.loadings[:, 1],
                                      self.loadings[:, 1])


class TestNormalizeComponents():

    def setup_method(self, method):
        s = signals.BaseSignal(np.zeros(1))
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
        np.testing.assert_array_equal(s.learning_results.bss_factors,
                                      self.factors / 2.)
        np.testing.assert_array_equal(s.learning_results.bss_loadings,
                                      self.loadings * 2.)

    def test_normalize_bss_loadings(self):
        s = self.s
        s.normalize_bss_components(target="loadings",
                                   function=np.sum)
        np.testing.assert_array_equal(s.learning_results.bss_factors,
                                      self.factors * 2.)
        np.testing.assert_array_equal(s.learning_results.bss_loadings,
                                      self.loadings / 2.)

    def test_normalize_decomposition_factors(self):
        s = self.s
        s.normalize_decomposition_components(target="factors",
                                             function=np.sum)
        np.testing.assert_array_equal(s.learning_results.factors,
                                      self.factors / 2.)
        np.testing.assert_array_equal(s.learning_results.loadings,
                                      self.loadings * 2.)

    def test_normalize_decomposition_loadings(self):
        s = self.s
        s.normalize_decomposition_components(target="loadings",
                                             function=np.sum)
        np.testing.assert_array_equal(s.learning_results.factors,
                                      self.factors * 2.)
        np.testing.assert_array_equal(s.learning_results.loadings,
                                      self.loadings / 2.)


class TestReturnInfo:

    def setup_method(self, method):
        self.s = signals.Signal1D(np.random.random((20, 100)))

    def test_decomposition_not_supported(self):
        # Not testing MLPCA, takes too long
        for algorithm in ["svd", "fast_svd"]:
            print(algorithm)
            assert self.s.decomposition(
                algorithm=algorithm, return_info=True, output_dimension=1) is None

    @pytest.mark.skipif(not sklearn_installed, reason="sklearn not installed")
    def test_decomposition_supported_return_true(self):
        for algorithm in ["RPCA_GoDec", "ORPCA"]:
            assert self.s.decomposition(
                algorithm=algorithm,
                return_info=True,
                output_dimension=1) is not None
        for algorithm in ["sklearn_pca", "nmf",
                          "sparse_pca", "mini_batch_sparse_pca", ]:
            assert self.s.decomposition(
                algorithm=algorithm,
                return_info=True,
                output_dimension=1) is not None

    @pytest.mark.skipif(not sklearn_installed, reason="sklearn not installed")
    def test_decomposition_supported_return_false(self):
        for algorithm in ["RPCA_GoDec", "ORPCA"]:
            assert self.s.decomposition(
                algorithm=algorithm,
                return_info=False,
                output_dimension=1) is None
        for algorithm in ["sklearn_pca", "nmf",
                          "sparse_pca", "mini_batch_sparse_pca", ]:
            assert self.s.decomposition(
                algorithm=algorithm,
                return_info=False,
                output_dimension=1) is None
