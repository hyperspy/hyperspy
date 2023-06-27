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

import numpy as np
import pytest

from hyperspy import signals
from hyperspy.misc.machine_learning import import_sklearn

sklearn = pytest.importorskip("sklearn", reason="sklearn not installed")

if import_sklearn.sklearn_installed:
    # Create the data once, since the parametrizations
    # will repeat the decomposition and BSS unnecessarily
    rng1 = np.random.RandomState(123)
    signal1 = signals.Signal1D(rng1.uniform(size=(7, 5, 7)))
    signal1.decomposition()
    signal1.blind_source_separation(number_of_components=3)

    rng2 = np.random.RandomState(123)
    signal2 = signals.Signal2D(rng2.uniform(size=(7, 5, 7)))
    signal2.decomposition()
    signal2.blind_source_separation(number_of_components=3)
else:
    # No need to create the data, since BSS will fail
    # if sklearn is missing, and the pytest.importorskip
    # will skip the rest of the file anyway
    pass


class TestCluster1D:
    def setup_method(self):
        self.signal = signal1.deepcopy()
        self.navigation_mask = np.zeros((7, 5), dtype=bool)
        self.navigation_mask[4:6, 1:4] = True
        self.signal_mask = np.zeros((7,), dtype=bool)
        self.signal_mask[2:6] = True

    @pytest.mark.parametrize("algorithm", (None, "agglomerative", "spectralclustering"))
    @pytest.mark.parametrize("cluster_source", ("signal", "bss", "decomposition"))
    @pytest.mark.parametrize(
        "source_for_centers", (None, "signal", "bss", "decomposition")
    )
    @pytest.mark.parametrize("preprocessing", (None, "standard", "norm", "minmax"))
    @pytest.mark.parametrize("use_masks", (True, False))
    def test_combinations(
        self, algorithm, cluster_source, preprocessing, source_for_centers, use_masks
    ):
        if use_masks:
            navigation_mask = self.navigation_mask
            signal_mask = self.signal_mask
        else:
            navigation_mask = None
            signal_mask = None
        self.signal.cluster_analysis(
            cluster_source,
            n_clusters=3,
            source_for_centers=source_for_centers,
            preprocessing=preprocessing,
            navigation_mask=navigation_mask,
            signal_mask=signal_mask,
            algorithm=algorithm,
        )
        np.testing.assert_array_equal(
            self.signal.learning_results.cluster_labels.shape, (3, 35)
        )
        np.testing.assert_array_equal(
            self.signal.learning_results.cluster_centroid_signals.shape, (3, 7)
        )
        np.testing.assert_array_equal(
            self.signal.learning_results.cluster_sum_signals.shape, (3, 7)
        )
        self.signal.get_cluster_labels()
        self.signal.get_cluster_signals()

    def test_custom_algorithm(self):
        self.signal.cluster_analysis(
            "signal", n_clusters=3, preprocessing="norm",
        )
        np.testing.assert_array_equal(
            self.signal.learning_results.cluster_labels.shape, (3, 35)
        )
        np.testing.assert_array_equal(
            self.signal.learning_results.cluster_centroid_signals.shape, (3, 7)
        )
        np.testing.assert_array_equal(
            self.signal.learning_results.cluster_sum_signals.shape, (3, 7)
        )

    def test_custom_preprocessing(self):
        custom_method = import_sklearn.sklearn.preprocessing.Normalizer()
        self.signal.cluster_analysis(
            "signal", n_clusters=3, preprocessing=custom_method, algorithm="kmeans"
        )
        np.testing.assert_array_equal(
            self.signal.learning_results.cluster_labels.shape, (3, 35)
        )
        np.testing.assert_array_equal(
            self.signal.learning_results.cluster_centroid_signals.shape, (3, 7)
        )
        np.testing.assert_array_equal(
            self.signal.learning_results.cluster_sum_signals.shape, (3, 7)
        )


class TestClusterSignalSources:
    def setup_method(self):
        self.signal = signal2.deepcopy()
        self.navigation_mask = np.zeros((7,), dtype=bool)
        self.navigation_mask[4:6] = True
        self.signal_mask = np.zeros((5, 7), dtype=bool)
        self.signal_mask[1:4, 2:6] = True

    @pytest.mark.parametrize("use_masks", (True, False))
    def test_cluster_source(self, use_masks):
        if use_masks:
            navigation_mask = self.navigation_mask
            signal_mask = self.signal_mask
        else:
            navigation_mask = None
            signal_mask = None
        # test using cluster source centre is a signal
        signal_copy = self.signal.deepcopy()
        self.signal.cluster_analysis(
            signal_copy,
            n_clusters=3,
            source_for_centers="signal",
            preprocessing="norm",
            navigation_mask=navigation_mask,
            signal_mask=signal_mask,
            algorithm="kmeans",
        )

        np.testing.assert_array_equal(
            self.signal.learning_results.cluster_labels.shape, (3, 7)
        )
        np.testing.assert_array_equal(
            self.signal.learning_results.cluster_centroid_signals.shape, (3, 35)
        )
        np.testing.assert_array_equal(
            self.signal.learning_results.cluster_sum_signals.shape, (3, 35)
        )

    @pytest.mark.parametrize("use_masks", (True, False))
    def test_source_center(self, use_masks):
        if use_masks:
            navigation_mask = self.navigation_mask
            signal_mask = self.signal_mask
        else:
            navigation_mask = None
            signal_mask = None
        # test using cluster source centre is a signal
        signal_copy = self.signal.deepcopy()
        self.signal.cluster_analysis(
            "signal",
            n_clusters=3,
            source_for_centers=signal_copy,
            preprocessing="norm",
            navigation_mask=navigation_mask,
            signal_mask=signal_mask,
            algorithm="kmeans",
        )

        np.testing.assert_array_equal(
            self.signal.learning_results.cluster_labels.shape, (3, 7)
        )
        np.testing.assert_array_equal(
            self.signal.learning_results.cluster_centroid_signals.shape, (3, 35)
        )
        np.testing.assert_array_equal(
            self.signal.learning_results.cluster_sum_signals.shape, (3, 35)
        )


@pytest.mark.filterwarnings("ignore:FastICA did not converge")
class TestClusterEstimate:
    def setup_method(self):
        rng = np.random.RandomState(123)
        # Use prime numbers to avoid fluke equivalences
        # create 3 random clusters
        n_samples = [100] * 3
        std = [0.05] * 3
        X = []
        centers = np.array(
            [[-1.0, -1.0, 1, 1], [1.0, -1.0, -1.0, -1], [-1.0, 1.0, 1.0, -1.0]]
        )
        for i, (n, std) in enumerate(zip(n_samples, std)):
            X.append(centers[i] + rng.normal(scale=std, size=(n, 4)))
        X = np.concatenate(X)
        rng.shuffle(X)
        self.signal = signals.Signal1D(X)
        self.signal.decomposition()
        self.signal.blind_source_separation(number_of_components=3)

    @pytest.mark.parametrize("metric", ("elbow", "silhouette", "gap"))
    def test_metric(self, metric):
        max_clusters = 6
        self.signal.estimate_number_of_clusters(
            "signal",
            max_clusters=max_clusters,
            preprocessing="norm",
            algorithm="kmeans",
            metric=metric,
        )
        k_range = self.signal.learning_results.cluster_metric_index
        best_k = self.signal.learning_results.estimated_number_of_clusters
        if isinstance(best_k, list):
            best_k = best_k[0]

        test_k_range = list(range(1, max_clusters + 1))
        if metric == "silhouette":
            test_k_range = list(range(2, max_clusters + 1))

        np.testing.assert_allclose(k_range, test_k_range)
        np.testing.assert_allclose(best_k, 3)

    @pytest.mark.parametrize("algorithm", ("kmeans", "agglomerative"))
    def test_cluster_algorithm(self, algorithm):
        max_clusters = 6
        self.signal.estimate_number_of_clusters(
            "signal",
            max_clusters=max_clusters,
            preprocessing="norm",
            algorithm=algorithm,
            metric="elbow",
        )
        k_range = self.signal.learning_results.cluster_metric_index
        best_k = self.signal.learning_results.estimated_number_of_clusters
        if isinstance(best_k, list):
            best_k = best_k[0]

        test_k_range = list(range(1, max_clusters + 1))
        if algorithm == "agglomerative":
            test_k_range = list(range(2, max_clusters + 1))

        np.testing.assert_allclose(k_range, test_k_range)
        np.testing.assert_allclose(best_k, 3)


class DummyClusterAlgorithm:
    def __init__(self):
        self.test = None

    def fit(self, X):
        pass


class DummyScalingAlgorithm:
    def __init__(self):
        self.test = None

    def fit(self, X):
        pass


class TestClusterExceptions:
    def setup_method(self):
        self.rng = np.random.RandomState(123)
        self.s = signals.Signal1D(self.rng.random_sample(size=(20, 100)))

    def test_cluster_source_error(self):
        with pytest.raises(
            ValueError,
            match="cluster source needs to be set "
            "to `decomposition` , `signal` , `bss` "
            "or a suitable Signal",
        ):
            self.s.cluster_analysis("randtest", n_clusters=2)

    def test_cluster_source_size_error(self):
        x2 = self.rng.random_sample(size=(10, 80))
        s2 = signals.Signal1D(x2)
        with pytest.raises(
            ValueError,
            match="cluster_source does not have the same "
            "navigation size as the this signal",
        ):
            self.s.cluster_analysis(s2, n_clusters=2)

    def test_cluster_source_center_size_error(self):
        x2 = self.rng.random_sample(size=(10, 80))
        s2 = signals.Signal1D(x2)
        with pytest.raises(
            ValueError,
            match="cluster_source does not have the same "
            "navigation size as the this signal",
        ):
            self.s.cluster_analysis("signal", n_clusters=2, source_for_centers=s2)

    def test_cluster_bss_error(self):
        with pytest.raises(
            ValueError,
            match="A cluster source has been set to bss "
            " but no blind source separation results found. "
            " Please run blind source separation method first",
        ):
            self.s.cluster_analysis("bss", n_clusters=2)

    def test_cluster_decomposition_error(self):
        with pytest.raises(
            ValueError,
            match="A cluster source has been set to "
            "decomposition but no decomposition results found. "
            "Please run decomposition method first",
        ):
            self.s.cluster_analysis("decomposition", n_clusters=2)

    def test_cluster_nav_mask_error(self):
        nav_mask = np.zeros((11,), dtype=bool)
        with pytest.raises(
            ValueError,
            match="Navigation mask size does not match " "signal navigation size",
        ):
            self.s.cluster_analysis("signal", n_clusters=2, navigation_mask=nav_mask)

    def test_cluster_sig_mask_error(self):
        sig_mask = np.zeros((11,), dtype=bool)
        with pytest.raises(
            ValueError,
            match="signal mask size does not match your " "cluster source signal size",
        ):
            self.s.cluster_analysis("signal", n_clusters=2, signal_mask=sig_mask)

    def test_cluster_basesig_mask_error(self):
        sig_mask = np.zeros((11,), dtype=bool)
        with pytest.raises(
            ValueError,
            match="signal mask size does not match your " "cluster source signal size",
        ):
            self.s.cluster_analysis(
                self.s.deepcopy(), n_clusters=2, signal_mask=sig_mask
            )

    def test_max_cluster_error(self):
        max_clusters = 1
        with pytest.raises(
            ValueError,
            match="The max number of clusters, max_clusters, "
            "must be specified and be >= 2.",
        ):

            self.s.estimate_number_of_clusters(
                "signal",
                max_clusters=max_clusters,
                preprocessing=None,
                algorithm="kmeans",
                metric="elbow",
            )

    def test_cluster_preprocessing_object_error(self):
        preprocessing = object()
        with pytest.raises(
            ValueError, match=r"The cluster preprocessing method should be either \w*"
        ):
            self.s.cluster_analysis("signal", n_clusters=2, preprocessing=preprocessing)

    def test_clustering_object_error(self):
        empty_object = object()
        with pytest.raises(
            ValueError, match=r"The clustering method should be either \w*"
        ):
            self.s.cluster_analysis("signal", n_clusters=2, algorithm=empty_object)

    def test_estimate_alg_error(self):
        with pytest.raises(
            ValueError,
            match="Estimate number of clusters only works with "
            "supported clustering algorithms",
        ):
            self.s.estimate_number_of_clusters("signal", algorithm="orange")

    def test_estimate_pre_error(self):
        with pytest.raises(
            ValueError,
            match="Estimate number of clusters only works with "
            "supported preprocessing algorithms",
        ):
            self.s.estimate_number_of_clusters("signal", preprocessing="orange")

    def test_sklearn_exception(self):
        import_sklearn.sklearn_installed = False
        with pytest.raises(ImportError):
            self.s.cluster_analysis("signal", n_clusters=2)
        import_sklearn.sklearn_installed = True

    def test_sklearn_exception2(self):
        import_sklearn.sklearn_installed = False
        with pytest.raises(ImportError):
            self.s._get_cluster_algorithm("kmeans")
        import_sklearn.sklearn_installed = True

    def test_sklearn_exception3(self):
        import_sklearn.sklearn_installed = False
        with pytest.raises(ImportError):
            self.s._get_cluster_preprocessing_algorithm("norm")
        import_sklearn.sklearn_installed = True

    def test_preprocess_alg_exception(self):
        sc = DummyScalingAlgorithm()
        with pytest.raises(
            ValueError, match=r"The cluster preprocessing method should be \w*"
        ):
            self.s.cluster_analysis("signal", n_clusters=2, preprocessing=sc)

    def test_cluster_alg_exception(self):
        sc = DummyClusterAlgorithm()
        with pytest.raises(AttributeError, match=r"Fited cluster estimator \w*"):
            self.s.cluster_analysis("signal", n_clusters=2, algorithm=sc)


def test_get_methods():
    rng = np.random.RandomState(123)
    signal = signals.Signal1D(rng.random_sample(size=(11, 5, 7)))
    signal.decomposition()
    signal.cluster_analysis("signal", n_clusters=2)
    signal.unfold()
    cl = signal.get_cluster_labels(merged=True)
    np.testing.assert_array_equal(
        cl.data,
        (signal.learning_results.cluster_labels * np.arange(2)[:, np.newaxis]).sum(0),
    )

    cl = signal.get_cluster_labels(merged=False)
    np.testing.assert_array_equal(cl.data, signal.learning_results.cluster_labels)

    cl = signal.get_cluster_signals(signal="sum")
    np.testing.assert_array_equal(cl.data, signal.learning_results.cluster_sum_signals)

    cl = signal.get_cluster_signals(signal="centroid")
    np.testing.assert_array_equal(
        cl.data, signal.learning_results.cluster_centroid_signals
    )
    cl = signal.get_cluster_signals(signal="mean")
    np.testing.assert_array_equal(
        cl.data,
        signal.learning_results.cluster_sum_signals
        / signal.learning_results.cluster_labels.sum(1, keepdims=True),
    )

    cl = signal.get_cluster_distances()
    np.testing.assert_array_equal(cl.data, signal.learning_results.cluster_distances)
