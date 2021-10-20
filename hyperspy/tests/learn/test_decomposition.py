# -*- coding: utf-8 -*-
# Copyright 2007-2021 The HyperSpy developers
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

from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
import pytest

from hyperspy import signals
from hyperspy.decorators import lazifyTestClass
from hyperspy.exceptions import VisibleDeprecationWarning
from hyperspy.misc.machine_learning.import_sklearn import sklearn_installed


def generate_low_rank_matrix(m=20, n=100, rank=5, random_seed=123):
    """Generate a low-rank matrix with specified size and rank for testing."""
    rng = np.random.RandomState(random_seed)
    U = rng.randn(m, rank)
    V = rng.randn(n, rank)
    X = np.abs(U @ V.T)
    X /= np.linalg.norm(X)
    return X


def test_error_axes():
    s = signals.BaseSignal(generate_low_rank_matrix())

    with pytest.raises(AttributeError, match="not possible to decompose a dataset"):
        s.decomposition()


class TestNdAxes:
    def setup_method(self, method):
        """Create three signals with dimensions:

        s1 : <BaseSignal, title: , dimensions: (4, 3, 2|2, 3)>
        s2 : <BaseSignal, title: , dimensions: (2, 3|4, 3, 2)>
        s12 : <BaseSignal, title: , dimensions: (2, 3|4, 3, 2)>

        Where s12 data is transposed in respect to s2
        """
        rng = np.random.RandomState(123)
        dc1 = rng.random_sample(size=(2, 3, 4, 3, 2))
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
        np.testing.assert_array_almost_equal(
            s2.learning_results.loadings, s12.learning_results.loadings
        )
        np.testing.assert_array_almost_equal(
            s2.learning_results.factors, s12.learning_results.factors
        )
        np.testing.assert_array_almost_equal(
            s1.learning_results.loadings, s2.learning_results.factors
        )
        np.testing.assert_array_almost_equal(
            s1.learning_results.factors, s2.learning_results.loadings
        )

    def test_consistency_poissonian(self):
        s1 = self.s1
        s1n000 = self.s1.inav[0, 0, 0]
        s2 = self.s2
        s12 = self.s12
        s1.decomposition(normalize_poissonian_noise=True)
        s2.decomposition(normalize_poissonian_noise=True)
        s12.decomposition(normalize_poissonian_noise=True)
        np.testing.assert_array_almost_equal(
            s2.learning_results.loadings, s12.learning_results.loadings
        )
        np.testing.assert_array_almost_equal(
            s2.learning_results.factors, s12.learning_results.factors
        )
        np.testing.assert_array_almost_equal(
            s1.learning_results.loadings, s2.learning_results.factors
        )
        np.testing.assert_array_almost_equal(
            s1.learning_results.factors, s2.learning_results.loadings
        )
        # Check that views of the data don't change. See #871
        np.testing.assert_array_equal(s1.inav[0, 0, 0].data, s1n000.data)


@lazifyTestClass
class TestGetModel:
    def setup_method(self, method):
        rng = np.random.RandomState(100)
        sources = signals.Signal1D(rng.standard_t(0.5, size=(3, 100)))
        maps = signals.Signal2D(rng.standard_t(0.5, size=(3, 4, 5)))
        self.s = (
            sources.inav[0] * maps.inav[0].T
            + sources.inav[1] * maps.inav[1].T
            + sources.inav[2] * maps.inav[2].T
        )

    @pytest.mark.parametrize("centre", [None, "signal"])
    def test_get_decomposition_model(self, centre):
        s = self.s
        s.decomposition(algorithm="SVD", centre=centre)
        sc = self.s.get_decomposition_model(3)
        rms = np.sqrt(((sc.data - s.data) ** 2).sum())
        assert rms < 5e-7

    @pytest.mark.skipif(not sklearn_installed, reason="sklearn not installed")
    def test_get_bss_model(self):
        s = self.s
        s.decomposition(algorithm="SVD")
        s.blind_source_separation(3)
        sc = self.s.get_bss_model()
        rms = np.sqrt(((sc.data - s.data) ** 2).sum())
        assert rms < 5e-7


@lazifyTestClass
class TestGetExplainedVarinaceRatio:
    def setup_method(self, method):
        s = signals.BaseSignal(np.empty(1))
        self.s = s

    def test_data(self):
        self.s.learning_results.explained_variance_ratio = np.asarray([2, 4])
        np.testing.assert_array_equal(
            self.s.get_explained_variance_ratio().data, np.asarray([2, 4])
        )

    def test_no_evr(self):
        with pytest.raises(AttributeError):
            self.s.get_explained_variance_ratio()


class TestEstimateElbowPosition:
    def setup_method(self, method):
        s = signals.BaseSignal(np.empty(1))
        s.learning_results.explained_variance_ratio = np.asarray(
            [
                10e-1,
                5e-2,
                9e-3,
                1e-3,
                9e-5,
                5e-5,
                3.0e-5,
                2.2e-5,
                1.9e-5,
                1.8e-5,
                1.7e-5,
                1.6e-5,
            ]
        )
        self.s = s

    def test_elbow_position_array(self):
        variance = self.s.learning_results.explained_variance_ratio
        elbow = self.s.estimate_elbow_position(variance)
        assert elbow == 4

    def test_elbow_position_log(self):
        variance = self.s.learning_results.explained_variance_ratio
        elbow = self.s.estimate_elbow_position(variance, log=False)
        assert elbow == 1

    def test_elbow_position_none(self):
        _ = self.s.learning_results.explained_variance_ratio
        elbow = self.s.estimate_elbow_position(None)
        assert elbow == 4

    def test_elbow_position_error(self):
        self.s.learning_results.explained_variance_ratio = None
        with pytest.raises(
            ValueError, match="decomposition must be performed before calling"
        ):
            _ = self.s.estimate_elbow_position(None)

    # Should be removed in scikit-learn 0.26
    # https://scikit-learn.org/dev/whats_new/v0.24.html#sklearn-cross-decomposition
    @pytest.mark.filterwarnings("ignore:The 'init' value, when 'init=None':FutureWarning")
    @pytest.mark.skipif(not sklearn_installed, reason="sklearn not installed")
    def test_store_number_significant_components(self):
        s = signals.Signal1D(generate_low_rank_matrix())
        s.decomposition()
        assert s.learning_results.number_significant_components == 2

        # Check that number_significant_components is reset properly
        s.decomposition(algorithm="NMF")
        assert s.learning_results.number_significant_components is None


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
        np.testing.assert_array_equal(
            self.s.learning_results.factors[:, 0], self.factors[:, 0] * -1
        )

    def test_reversal_loadings_one_component_reversed(self):
        self.s.reverse_decomposition_component(0)
        np.testing.assert_array_equal(
            self.s.learning_results.loadings[:, 0], self.loadings[:, 0] * -1
        )

    def test_reversal_factors_one_component_not_reversed(self):
        self.s.reverse_decomposition_component(0)
        np.testing.assert_array_equal(
            self.s.learning_results.factors[:, 1:], self.factors[:, 1:]
        )

    def test_reversal_loadings_one_component_not_reversed(self):
        self.s.reverse_decomposition_component(0)
        np.testing.assert_array_equal(
            self.s.learning_results.loadings[:, 1:], self.loadings[:, 1:]
        )

    def test_reversal_factors_multiple_components_reversed(self):
        self.s.reverse_decomposition_component((0, 2))
        np.testing.assert_array_equal(
            self.s.learning_results.factors[:, (0, 2)], self.factors[:, (0, 2)] * -1
        )

    def test_reversal_loadings_multiple_components_reversed(self):
        self.s.reverse_decomposition_component((0, 2))
        np.testing.assert_array_equal(
            self.s.learning_results.loadings[:, (0, 2)], self.loadings[:, (0, 2)] * -1
        )

    def test_reversal_factors_multiple_components_not_reversed(self):
        self.s.reverse_decomposition_component((0, 2))
        np.testing.assert_array_equal(
            self.s.learning_results.factors[:, 1], self.factors[:, 1]
        )

    def test_reversal_loadings_multiple_components_not_reversed(self):
        self.s.reverse_decomposition_component((0, 2))
        np.testing.assert_array_equal(
            self.s.learning_results.loadings[:, 1], self.loadings[:, 1]
        )


class TestNormalizeComponents:
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
        s.normalize_bss_components(target="factors", function=np.sum)
        np.testing.assert_array_equal(
            s.learning_results.bss_factors, self.factors / 2.0
        )
        np.testing.assert_array_equal(
            s.learning_results.bss_loadings, self.loadings * 2.0
        )

    def test_normalize_bss_loadings(self):
        s = self.s
        s.normalize_bss_components(target="loadings", function=np.sum)
        np.testing.assert_array_equal(
            s.learning_results.bss_factors, self.factors * 2.0
        )
        np.testing.assert_array_equal(
            s.learning_results.bss_loadings, self.loadings / 2.0
        )

    def test_normalize_decomposition_factors(self):
        s = self.s
        s.normalize_decomposition_components(target="factors", function=np.sum)
        np.testing.assert_array_equal(s.learning_results.factors, self.factors / 2.0)
        np.testing.assert_array_equal(s.learning_results.loadings, self.loadings * 2.0)

    def test_normalize_decomposition_loadings(self):
        s = self.s
        s.normalize_decomposition_components(target="loadings", function=np.sum)
        np.testing.assert_array_equal(s.learning_results.factors, self.factors * 2.0)
        np.testing.assert_array_equal(s.learning_results.loadings, self.loadings / 2.0)


class TestDecompositionAlgorithm:
    def setup_method(self, method):
        self.s = signals.Signal1D(generate_low_rank_matrix())

    @pytest.mark.parametrize("algorithm", ["SVD", "MLPCA"])
    def test_decomposition(self, algorithm):
        self.s.decomposition(algorithm=algorithm, output_dimension=2)

    @pytest.mark.skipif(not sklearn_installed, reason="sklearn not installed")
    @pytest.mark.parametrize("algorithm", ["RPCA", "ORPCA", "ORNMF", "MLPCA"])
    def test_decomposition_output_dimension_not_given(self, algorithm):
        with pytest.raises(ValueError, match="`output_dimension` must be specified"):
            self.s.decomposition(algorithm=algorithm, return_info=False)

    @pytest.mark.skipif(not sklearn_installed, reason="sklearn not installed")
    @pytest.mark.parametrize("algorithm", ["fast_svd", "fast_mlpca"])
    def test_fast_deprecation_warning(self, algorithm):
        with pytest.warns(
            VisibleDeprecationWarning,
            match="argument `svd_solver='randomized'` instead.",
        ):
            self.s.decomposition(algorithm=algorithm, output_dimension=2)

    @pytest.mark.skipif(not sklearn_installed, reason="sklearn not installed")
    @pytest.mark.parametrize("algorithm", ["RPCA_GoDec", "svd", "mlpca", "nmf",])
    def test_name_deprecation_warning(self, algorithm):
        with pytest.warns(
            VisibleDeprecationWarning,
            match="has been deprecated and will be removed in HyperSpy 2.0.",
        ):
            self.s.decomposition(algorithm=algorithm, output_dimension=2)

    def test_algorithm_error(self):
        with pytest.raises(ValueError, match="not recognised. Expected"):
            self.s.decomposition(algorithm="uniform")


class TestPrintInfo:
    def setup_method(self, method):
        self.s = signals.Signal1D(generate_low_rank_matrix())

    @pytest.mark.parametrize("algorithm", ["SVD", "MLPCA"])
    def test_decomposition(self, algorithm, capfd):
        self.s.decomposition(algorithm=algorithm, output_dimension=2)
        captured = capfd.readouterr()
        assert "Decomposition info:" in captured.out

    # Should be removed in scikit-learn 0.26
    # https://scikit-learn.org/dev/whats_new/v0.24.html#sklearn-cross-decomposition
    @pytest.mark.filterwarnings("ignore:The 'init' value, when 'init=None':FutureWarning")
    @pytest.mark.skipif(not sklearn_installed, reason="sklearn not installed")
    @pytest.mark.parametrize(
        "algorithm", ["sklearn_pca", "NMF", "sparse_pca", "mini_batch_sparse_pca"]
    )
    def test_decomposition_sklearn(self, capfd, algorithm):
        self.s.decomposition(algorithm=algorithm, output_dimension=5)
        captured = capfd.readouterr()
        assert "Decomposition info:" in captured.out
        assert "scikit-learn estimator:" in captured.out

    @pytest.mark.parametrize("algorithm", ["SVD"])
    def test_no_print(self, algorithm, capfd):
        self.s.decomposition(algorithm=algorithm, output_dimension=2, print_info=False)
        captured = capfd.readouterr()
        assert "Decomposition info:" not in captured.out


class TestReturnInfo:
    def setup_method(self, method):
        self.s = signals.Signal1D(generate_low_rank_matrix())

    @pytest.mark.parametrize("algorithm", ["SVD", "MLPCA"])
    def test_decomposition_not_supported(self, algorithm):
        assert (
            self.s.decomposition(
                algorithm=algorithm, return_info=True, output_dimension=2
            )
            is None
        )

    # Should be removed in scikit-learn 0.26
    # https://scikit-learn.org/dev/whats_new/v0.24.html#sklearn-cross-decomposition
    @pytest.mark.filterwarnings("ignore:The 'init' value, when 'init=None':FutureWarning")
    @pytest.mark.skipif(not sklearn_installed, reason="sklearn not installed")
    @pytest.mark.parametrize("return_info", [True, False])
    @pytest.mark.parametrize(
        "algorithm",
        [
            "RPCA",
            "ORPCA",
            "ORNMF",
            "sklearn_pca",
            "NMF",
            "sparse_pca",
            "mini_batch_sparse_pca",
        ],
    )
    def test_decomposition_supported(self, algorithm, return_info):
        out = self.s.decomposition(
                algorithm=algorithm, return_info=return_info, output_dimension=2
            )
        assert (out is not None) is return_info


class TestNonFloatTypeError:
    def setup_method(self, method):
        mat = generate_low_rank_matrix()
        self.s_int = signals.Signal1D((mat * 20).astype("int"))
        self.s_float = signals.Signal1D(mat)

    def test_decomposition_error(self):
        self.s_float.decomposition()
        with pytest.raises(TypeError):
            self.s_int.decomposition()


class TestLoadDecompositionResults:
    def setup_method(self, method):
        self.s = signals.Signal1D([[1.1, 1.2, 1.4, 1.3], [1.5, 1.5, 1.4, 1.2]])

    def test_load_decomposition_results(self):
        """Test whether the sequence of loading learning results and then
        saving the signal causes errors. See #2093.
        """
        with TemporaryDirectory() as tmpdir:
            self.s.decomposition()
            fname1 = Path(tmpdir, "results.npz")
            self.s.learning_results.save(fname1)
            self.s.learning_results.load(fname1)
            fname2 = Path(tmpdir, "output.hspy")
            self.s.save(fname2)
            assert isinstance(self.s.learning_results.decomposition_algorithm, str)


class TestComplexSignalDecomposition:
    def setup_method(self, method):
        rng = np.random.RandomState(123)
        real = rng.random_sample(size=(8, 8))
        imag = rng.random_sample(size=(8, 8))
        s_complex_dtype = signals.ComplexSignal1D(real + 1j * imag - 1j * imag)
        s_real_dtype = signals.ComplexSignal1D(real)
        s_complex_dtype.decomposition()
        s_real_dtype.decomposition()
        self.s_complex_dtype = s_complex_dtype
        self.s_real_dtype = s_real_dtype

    def test_imaginary_is_zero(self):
        np.testing.assert_allclose(self.s_complex_dtype.data.imag, 0.0)

    def test_decomposition_independent_of_complex_dtype(self):
        complex_pca = self.s_complex_dtype.get_decomposition_model(5)
        real_pca = self.s_real_dtype.get_decomposition_model(5)
        np.testing.assert_almost_equal((complex_pca - real_pca).data.max(), 0.0)

    def test_first_r_values_of_scree_non_zero(self):
        """For low-rank matrix by creating a = RandomComplex(m, r) and
        b = RandomComplex(n, r), then performing PCA on the result of a.b^T
        (i.e. an m x n matrix).
        The first r values of scree plot / singular values should be non-zero.
        """
        m, n, r = 32, 32, 3
        rng = np.random.RandomState(123)

        A = rng.random_sample(size=(m, r)) + 1j * rng.random_sample(size=(m, r))
        B = rng.random_sample(size=(n, r)) + 1j * rng.random_sample(size=(n, r))

        s = signals.ComplexSignal1D(A @ B.T)
        s.decomposition()
        np.testing.assert_almost_equal(
            s.get_explained_variance_ratio().data[r:].sum(), 0
        )


@pytest.mark.parametrize("reproject", [None, "navigation", "signal", "both"])
def test_decomposition_reproject(reproject):
    s = signals.Signal1D(generate_low_rank_matrix())
    s.decomposition(reproject=reproject)


@pytest.mark.skipif(not sklearn_installed, reason="sklearn not installed")
@pytest.mark.parametrize("reproject", ["signal", "both"])
def test_decomposition_reproject_warning(reproject):
    s = signals.Signal1D(generate_low_rank_matrix())
    with pytest.warns(
        UserWarning, match="Reprojecting the signal is not yet supported"
    ):
        s.decomposition(algorithm="NMF", reproject=reproject)


@pytest.mark.skipif(not sklearn_installed, reason="sklearn not installed")
def test_decomposition_pipeline():
    """Tests that a simple sklearn pipeline is an acceptable algorithm."""
    s = signals.Signal1D(generate_low_rank_matrix())

    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    est = Pipeline([("scaler", StandardScaler()), ("PCA", PCA(n_components=2))])
    out = s.decomposition(algorithm=est, output_dimension=2, return_info=True)

    assert hasattr(out, "steps")
    assert hasattr(out.named_steps["PCA"], "explained_variance_")


@pytest.mark.skipif(not sklearn_installed, reason="sklearn not installed")
def test_decomposition_gridsearchcv():
    """Tests that a simple sklearn GridSearchCV is an acceptable algorithm."""
    s = signals.Signal1D(generate_low_rank_matrix())

    from sklearn.decomposition import PCA
    from sklearn.model_selection import GridSearchCV

    est = GridSearchCV(PCA(), {"n_components": (2, 3)}, cv=5)
    out = s.decomposition(algorithm=est, output_dimension=2, return_info=True)

    assert hasattr(out, "best_estimator_")
    assert hasattr(out, "cv_results_")
    np.testing.assert_allclose(out.best_score_, 268.700158)


def test_decomposition_mlpca_var_func():
    s = signals.Signal1D(generate_low_rank_matrix())
    s.decomposition(output_dimension=2, algorithm="MLPCA", var_func=lambda x: x)


def test_decomposition_mlpca_warnings_errors():
    s = signals.Signal1D(generate_low_rank_matrix())

    with pytest.warns(
        VisibleDeprecationWarning, match="`polyfit` argument has been deprecated"
    ):
        s.decomposition(output_dimension=2, algorithm="MLPCA", polyfit=[1, 2, 3])

    with pytest.raises(
        ValueError, match="`var_func` and `var_array` cannot both be defined"
    ):
        s.decomposition(
            output_dimension=2,
            algorithm="MLPCA",
            var_func=[1, 2, 3],
            var_array=s.data.copy(),
        )

    with pytest.raises(ValueError, match="`var_array` must have the same shape"):
        s.decomposition(
            output_dimension=2, algorithm="MLPCA", var_array=s.data.copy()[:-3, :-3],
        )

    with pytest.raises(
        ValueError, match="`var_func` must be either a function or an array"
    ):
        s.decomposition(output_dimension=2, algorithm="MLPCA", var_func="func")

    with pytest.warns(
        UserWarning, match="does not make sense to normalize Poisson noise",
    ):
        s.decomposition(
            normalize_poissonian_noise=True, algorithm="MLPCA", output_dimension=2
        )


def test_negative_values_error():
    x = generate_low_rank_matrix()
    x[0, 0] = -1.0
    s = signals.Signal1D(x)
    with pytest.raises(ValueError, match="Negative values found in data!"):
        s.decomposition(normalize_poissonian_noise=True)


def test_undo_treatments_error():
    s = signals.Signal1D(generate_low_rank_matrix())
    s.decomposition(output_dimension=2, copy=False)

    with pytest.raises(AttributeError, match="Unable to undo data pre-treatments!"):
        s.undo_treatments()


def test_normalize_components_errors():
    s = signals.Signal1D(generate_low_rank_matrix())

    with pytest.raises(ValueError, match="can only be called after s.decomposition"):
        s.normalize_decomposition_components(target="loadings")

    s.decomposition()

    with pytest.raises(ValueError, match="target must be"):
        s.normalize_decomposition_components(target="uniform")


def test_centering_error():
    s = signals.Signal1D(generate_low_rank_matrix())

    with pytest.raises(
        ValueError, match="normalize_poissonian_noise=True is only compatible"
    ):
        s.decomposition(normalize_poissonian_noise=True, centre="navigation")

    with pytest.raises(ValueError, match="'centre' must be one of"):
        s.decomposition(centre="random")

    for centre in ["variables", "trials"]:
        with pytest.warns(
            VisibleDeprecationWarning,
            match="centre='{}' has been deprecated".format(centre),
        ):
            s.decomposition(centre=centre)


@pytest.mark.parametrize('mask_as_array', [True, False])
def test_decomposition_navigation_mask(mask_as_array):
    s = signals.Signal1D(generate_low_rank_matrix())
    navigation_mask = (s.sum(-1) < 1.5)
    if mask_as_array:
        navigation_mask = navigation_mask
    s.decomposition(navigation_mask=navigation_mask)
    data = s.get_decomposition_loadings().inav[0].data
    # Use np.argwhere(np.isnan(data)) to get the indices
    np.testing.assert_allclose(data[[2, 5, 7, 8, 12, 13, 15, 19]],
                               np.full(8, np.nan))
    assert not np.isnan(s.get_decomposition_factors().data).any()


@pytest.mark.parametrize('mask_as_array', [True, False])
def test_decomposition_signal_mask(mask_as_array):
    s = signals.Signal1D(generate_low_rank_matrix())
    signal_mask = (s.sum(0) < 0.25)
    if mask_as_array:
        signal_mask = signal_mask.data
    s.decomposition(signal_mask=signal_mask)
    data = s.get_decomposition_factors().inav[0].data
    # Use np.argwhere(np.isnan(data)) to get the indices
    np.testing.assert_allclose(data[[ 5, 12, 14, 15, 18, 21, 27, 28, 32, 43,
                                     52, 55, 57, 59, 62, 79, 83]],
                                np.full(17, np.nan))
    assert not np.isnan(s.get_decomposition_loadings().data).any()


@pytest.mark.parametrize('normalise_poissonian_noise', [True, False])
def test_decomposition_mask_all_data(normalise_poissonian_noise):
    with pytest.raises(ValueError, match='All the data are masked'):
        s = signals.Signal1D(generate_low_rank_matrix())
        signal_mask = (s.sum(0) >= s.sum(0).min())
        s.decomposition(normalise_poissonian_noise, signal_mask=signal_mask)

    with pytest.raises(ValueError, match='All the data are masked'):
        s = signals.Signal1D(generate_low_rank_matrix())
        navigation_mask = (s.sum(-1) >= 0)
        s.decomposition(normalise_poissonian_noise,
                        navigation_mask=navigation_mask)
