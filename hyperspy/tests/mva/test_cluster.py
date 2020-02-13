# -*- coding: utf-8 -*-
# Copyright 2007-2019 The HyperSpy developers
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

from hyperspy import signals
from hyperspy.misc.machine_learning.import_sklearn import sklearn_installed


pytestmark = pytest.mark.skipif(not sklearn_installed,
                                reason="sklearn not installed")

import hyperspy.misc.machine_learning.import_sklearn  as import_sklearn

class TestCluster1d:

    def setup_method(self):
        # Use prime numbers to avoid fluke equivalences
        self.signal = signals.Signal1D(np.random.rand(11, 5, 7))
        self.signal.decomposition()
        self.navigation_mask = np.zeros((11, 5), dtype=bool)
        self.navigation_mask[4:6, 1:4] = True
        self.signal_mask = np.zeros((7,), dtype=bool)
        self.signal_mask[2:6] = True


    @pytest.mark.parametrize("algorithm", ("kmeans", "agglomerative"))
    @pytest.mark.parametrize("use_decomposition_results", (True, False))
    @pytest.mark.parametrize("scaling", ("standard", "norm", "minmax",None))
    @pytest.mark.parametrize("use_decomposition_for_centers", (True, False))
    @pytest.mark.parametrize("use_masks", (True, False))
    def test_parameters(self, algorithm, use_decomposition_results,
                        scaling, use_decomposition_for_centers,
                        use_masks):
        if use_masks:
            navigation_mask = self.navigation_mask
            signal_mask = self.signal_mask
        else:
            navigation_mask = None
            signal_mask = None
        self.signal.cluster_analysis(3,
                                     scaling=scaling,
                                     use_decomposition_results=\
                                         use_decomposition_results,
                                     use_decomposition_for_centers=\
                                         use_decomposition_for_centers,
                                     navigation_mask=navigation_mask,
                                     signal_mask=signal_mask,
                                     algorithm=algorithm,
                                     )
        np.testing.assert_array_equal(
            self.signal.learning_results.cluster_labels.shape, (3, 55))
        np.testing.assert_array_equal(
            self.signal.learning_results.cluster_centers.shape, (3, 7))
        self.signal.get_cluster_labels()
        self.signal.get_cluster_centers()


class TestCluster2d:

    def setup_method(self):
        self.signal = signals.Signal2D(np.random.rand(11, 5, 7))
        self.signal.decomposition()
        self.navigation_mask = np.zeros((11,), dtype=bool)
        self.navigation_mask[4:6] = True
        self.signal_mask = np.zeros((5, 7), dtype=bool)
        self.signal_mask[1:4, 2:6] = True

    @pytest.mark.parametrize("algorithm", ("kmeans", "agglomerative"))
    @pytest.mark.parametrize("use_decomposition_results", (True, False))
    @pytest.mark.parametrize("scaling", ("standard", "norm","minmax",None))
    @pytest.mark.parametrize("use_decomposition_for_centers", (True, False))
    @pytest.mark.parametrize("use_masks", (True, False))
    def test_parameters(self, algorithm, use_decomposition_results,
                        scaling, use_decomposition_for_centers,
                        use_masks):
        if use_masks:
            navigation_mask = self.navigation_mask
            signal_mask = self.signal_mask
        else:
            navigation_mask = None
            signal_mask = None
        self.signal.cluster_analysis(3,
                                     scaling=scaling,
                                     use_decomposition_results=\
                                         use_decomposition_results,
                                     use_decomposition_for_centers=\
                                         use_decomposition_for_centers,
                                     navigation_mask=navigation_mask,
                                     signal_mask=signal_mask,
                                     algorithm=algorithm,
                                     )
        np.testing.assert_array_equal(
            self.signal.learning_results.cluster_labels.shape, (3, 11))
        np.testing.assert_array_equal(
            self.signal.learning_results.cluster_centers.shape, (3, 35))
        self.signal.get_cluster_labels()
        self.signal.get_cluster_centers()



class TestClusterEvaluate:

    def setup_method(self):
        np.random.seed(1)
        # Use prime numbers to avoid fluke equivalences
        # create 3 random clusters
        n_samples=[400]*3
        std = [1.0]*3
        X = []
        centers = np.random.uniform(-20,20,size=(3, 5))
        for i, (n, std) in enumerate(zip(n_samples, std)):
            X.append(centers[i] + np.random.normal(scale=std,size=(n, 5)))
        X = np.concatenate(X)
        np.random.shuffle(X)
        self.signal = signals.Signal1D(X)
        self.signal.decomposition()

    @pytest.mark.parametrize("algorithm", ("kmeans", "agglomerative"))
    @pytest.mark.parametrize("use_decomposition_results", (True, False))
    @pytest.mark.parametrize("scaling", ("standard", "norm", "minmax",None))
    @pytest.mark.parametrize("use_decomposition_for_centers", (True, False))
    @pytest.mark.parametrize("metric", ("elbow","silhouette","gap"))
    def test_scores(self, algorithm, use_decomposition_results,
                    scaling, use_decomposition_for_centers,metric):

        self.signal.evaluate_number_of_clusters(
            8,
            scaling=scaling,
            use_decomposition_results=use_decomposition_results,
            use_decomposition_for_centers=use_decomposition_for_centers,
            algorithm=algorithm,
            metric=metric)
        k_range = self.signal.learning_results.cluster_metric_index
        best_k = self.signal.learning_results.number_of_clusters
        if isinstance(best_k,list):
            best_k = best_k[0]

        test_k_range=list(range(1,9))
        if(algorithm == "agglomerative"):
            test_k_range   = list(range(2, 9))
        elif(algorithm == "kmeans"):
            if metric == "silhouette":
                test_k_range   = list(range(2,9))

        np.testing.assert_allclose(k_range,test_k_range)
        np.testing.assert_allclose(best_k, 3)




class TestClusterCustomScaling:

    def setup_method(self):
        np.random.seed(1)
        # Use prime numbers to avoid fluke equivalences
        # create 3 random clusters
        n_samples=[400] * 3
        std = [1.0] * 3
        X = []
        centers = np.random.uniform(-20, 20, size=(3, 5))
        for i, (n, std) in enumerate(zip(n_samples, std)):
            X.append(centers[i] + np.random.normal(scale=std, size=(n, 5)))
        X = np.concatenate(X)
        np.random.shuffle(X)
        self.signal = signals.Signal1D(X)
        self.signal.decomposition()


    @pytest.mark.parametrize("algorithm", ("kmeans", "agglomerative"))
    @pytest.mark.parametrize("use_decomposition_results", (True, False))
    @pytest.mark.parametrize("use_decomposition_for_centers", (True, False))
    @pytest.mark.parametrize("metric", ("elbow", "silhouette", "gap"))
    def test_custom(self,  use_decomposition_results,
                algorithm,use_decomposition_for_centers,metric):
            custom_scaling = import_sklearn.sklearn.preprocessing.MinMaxScaler
            self.signal.evaluate_number_of_clusters(
                8,
                scaling=custom_scaling,
                use_decomposition_results=use_decomposition_results,
                use_decomposition_for_centers=use_decomposition_for_centers,
                algorithm=algorithm,
                metric=metric)
            best_k = self.signal.learning_results.number_of_clusters
            if isinstance(best_k, list):
                best_k = best_k[0]

            np.testing.assert_allclose(best_k, 3)