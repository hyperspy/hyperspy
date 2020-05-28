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
        self.signal.blind_source_separation(number_of_components=3)
        self.navigation_mask = np.zeros((11, 5), dtype=bool)
        self.navigation_mask[4:6, 1:4] = True
        self.signal_mask = np.zeros((7,), dtype=bool)
        self.signal_mask[2:6] = True


    @pytest.mark.parametrize("algorithm", ("kmeans", "agglomerative",
                                            "spectralclustering",
                                            "minibatchkmeans"))
    @pytest.mark.parametrize("cluster_source", ("signal","bss","decomposition"))
    @pytest.mark.parametrize("source_for_centers", (None,"signal","bss","decomposition"))
    @pytest.mark.parametrize("scaling", ("standard", "norm", "minmax",None))
    @pytest.mark.parametrize("use_masks", (True, False))
    def test_parameters(self, algorithm, cluster_source,
                        scaling, source_for_centers,
                        use_masks):
        if use_masks:
            navigation_mask = self.navigation_mask
            signal_mask = self.signal_mask
        else:
            navigation_mask = None
            signal_mask = None
        self.signal.cluster_analysis(cluster_source,n_clusters=3,
                                      source_for_centers=\
                                          source_for_centers,
                                      scaling=scaling,
                                      navigation_mask=navigation_mask,
                                      signal_mask=signal_mask,
                                      algorithm=algorithm)
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
        self.signal.blind_source_separation(number_of_components=3)
        self.navigation_mask = np.zeros((11,), dtype=bool)
        self.navigation_mask[4:6] = True
        self.signal_mask = np.zeros((5, 7), dtype=bool)
        self.signal_mask[1:4, 2:6] = True
    @pytest.mark.parametrize("algorithm", ("kmeans", "agglomerative",
                                            "spectralclustering",
                                            "minibatchkmeans"))
    @pytest.mark.parametrize("cluster_source", ("signal","bss","decomposition"))
    @pytest.mark.parametrize("source_for_centers", (None,"signal","bss","decomposition"))
    @pytest.mark.parametrize("scaling", ("standard", "norm","minmax",None))
    @pytest.mark.parametrize("use_masks", (True, False))
    def test_parameters(self, algorithm, cluster_source,
                        scaling, source_for_centers,
                        use_masks):
        if use_masks:
            navigation_mask = self.navigation_mask
            signal_mask = self.signal_mask
        else:
            navigation_mask = None
            signal_mask = None
        self.signal.cluster_analysis(cluster_source=\
                                          cluster_source, n_clusters=3,
                                      source_for_centers=\
                                          source_for_centers,
                                      scaling=scaling,
                                      navigation_mask=navigation_mask,
                                      signal_mask=signal_mask,
                                      algorithm=algorithm)
        np.testing.assert_array_equal(
            self.signal.learning_results.cluster_labels.shape, (3, 11))
        np.testing.assert_array_equal(
            self.signal.learning_results.cluster_centers.shape, (3, 35))
        self.signal.get_cluster_labels()
        self.signal.get_cluster_centers()



class TestClusterSource2d:

    def setup_method(self):
        self.signal = signals.Signal2D(np.random.rand(11, 5, 7))
        self.signal.decomposition()
        self.signal.blind_source_separation(number_of_components=3)
        self.navigation_mask = np.zeros((11,), dtype=bool)
        self.navigation_mask[4:6] = True
        self.signal_mask = np.zeros((5, 7), dtype=bool)
        self.signal_mask[1:4, 2:6] = True
    @pytest.mark.parametrize("algorithm", ("kmeans", "agglomerative"))
    @pytest.mark.parametrize("source_for_centers", ("signal","bss","decomposition"))
    @pytest.mark.parametrize("scaling", ("standard", "norm","minmax",None))
    @pytest.mark.parametrize("use_masks", (True, False))
    def test_parameters(self, algorithm, source_for_centers,
                        scaling, use_masks):
        if use_masks:
            navigation_mask = self.navigation_mask
            signal_mask = self.signal_mask
        else:
            navigation_mask = None
            signal_mask = None
        # test using cluster source centre is a signal
        signal_copy = self.signal.deepcopy()
        self.signal.cluster_analysis(cluster_source=\
                                          signal_copy, n_clusters=3,
                                      source_for_centers=\
                                          source_for_centers,
                                      scaling=scaling,
                                      navigation_mask=navigation_mask,
                                      signal_mask=signal_mask,
                                      algorithm=algorithm)
            
        np.testing.assert_array_equal(
            self.signal.learning_results.cluster_labels.shape, (3, 11))
        np.testing.assert_array_equal(
            self.signal.learning_results.cluster_centers.shape, (3, 35))



class TestClusterSignalAsSource2d:

    def setup_method(self):
        self.signal = signals.Signal2D(np.random.rand(11, 5, 7))
        self.signal.decomposition()
        self.signal.blind_source_separation(number_of_components=3)
        self.navigation_mask = np.zeros((11,), dtype=bool)
        self.navigation_mask[4:6] = True
        self.signal_mask = np.zeros((5, 7), dtype=bool)
        self.signal_mask[1:4, 2:6] = True
    @pytest.mark.parametrize("algorithm", ("kmeans", "agglomerative"))
    @pytest.mark.parametrize("scaling", ("standard", "norm","minmax",None))
    @pytest.mark.parametrize("use_masks", (True, False))
    def test_parameters(self, algorithm,
                        scaling, use_masks):
        if use_masks:
            navigation_mask = self.navigation_mask
            signal_mask = self.signal_mask
        else:
            navigation_mask = None
            signal_mask = None
        # test using cluster source centre is a signal
        signal_copy = self.signal.deepcopy()
        self.signal.cluster_analysis(cluster_source=\
                                          signal_copy, n_clusters=3,
                                      source_for_centers=\
                                          signal_copy,
                                      scaling=scaling,
                                      navigation_mask=navigation_mask,
                                      signal_mask=signal_mask,
                                      algorithm=algorithm)
            
        np.testing.assert_array_equal(
            self.signal.learning_results.cluster_labels.shape, (3, 11))
        np.testing.assert_array_equal(
            self.signal.learning_results.cluster_centers.shape, (3, 35))


class TestClusterCenterSource2d:

    def setup_method(self):
        self.signal = signals.Signal2D(np.random.rand(11, 5, 7))
        self.signal.decomposition()
        self.signal.blind_source_separation(number_of_components=3)
        self.navigation_mask = np.zeros((11,), dtype=bool)
        self.navigation_mask[4:6] = True
        self.signal_mask = np.zeros((5, 7), dtype=bool)
        self.signal_mask[1:4, 2:6] = True
        self.signal_two = signals.Signal2D(np.random.rand(11, 5, 10))

    @pytest.mark.parametrize("algorithm", ("kmeans", "agglomerative"))
    @pytest.mark.parametrize("cluster_source", ("signal","bss","decomposition"))
    @pytest.mark.parametrize("scaling", ("standard", "norm","minmax",None))
    @pytest.mark.parametrize("use_masks", (True, False))
    def test_parameters(self, algorithm, cluster_source,
                        scaling, use_masks):
        if use_masks:
            navigation_mask = self.navigation_mask
            signal_mask = self.signal_mask
        else:
            navigation_mask = None
            signal_mask = None
        # test using cluster source centre is a signal
        signal_copy = self.signal_two
        self.signal.cluster_analysis(cluster_source=\
                                          cluster_source, n_clusters=3,
                                      source_for_centers=\
                                          signal_copy,
                                      scaling=scaling,
                                      navigation_mask=navigation_mask,
                                      signal_mask=signal_mask,
                                      algorithm=algorithm)
            
        np.testing.assert_array_equal(
            self.signal.learning_results.cluster_labels.shape, (3, 11))
        np.testing.assert_array_equal(
            self.signal.learning_results.cluster_centers.shape, (3, 50))



class TestClusterEstimate:

    def setup_method(self):
        np.random.seed(1)
        # Use prime numbers to avoid fluke equivalences
        # create 3 random clusters
        n_samples=[400]*3
        std = [0.05]*3
        X = []
        centers = np.array([[-1.,-1.,1,1],[1.,-1.,-1.,-1],[-1.,1.,1.,-1.]])
        for i, (n, std) in enumerate(zip(n_samples, std)):
            X.append(centers[i] + np.random.normal(scale=std,size=(n, 4)))
        X = np.concatenate(X)
        np.random.shuffle(X)
        self.signal = signals.Signal1D(X)
        self.signal.decomposition()
        self.signal.blind_source_separation(number_of_components=3)

    @pytest.mark.parametrize("algorithm", ("kmeans", "agglomerative"))
    @pytest.mark.parametrize("cluster_source", ("signal","bss","decomposition"))
    @pytest.mark.parametrize("scaling", ("standard", "norm", "minmax",None))
    @pytest.mark.parametrize("metric", ("elbow","silhouette","gap"))
    def test_scores(self, algorithm, cluster_source,
                    scaling,metric):
        max_clusters = 8
        self.signal.estimate_number_of_clusters(
            cluster_source,
            max_clusters=max_clusters,
            scaling=scaling,
            algorithm=algorithm,
            metric=metric)
        k_range = self.signal.learning_results.cluster_metric_index
        best_k = self.signal.learning_results.estimated_number_of_clusters
        if isinstance(best_k,list):
            best_k = best_k[0]

        test_k_range=list(range(1,max_clusters+1))
        if(algorithm == "agglomerative"):
            test_k_range   = list(range(2, max_clusters+1))
        elif(algorithm == "kmeans"):
            if metric == "silhouette":
                test_k_range   = list(range(2,max_clusters+1))

        np.testing.assert_allclose(k_range,test_k_range)
        np.testing.assert_allclose(best_k, 3)




class TestClusterCustomScaling:

    def setup_method(self):
        np.random.seed(1)
        # Use prime numbers to avoid fluke equivalences
        # create 3 random clusters
        n_samples=[400]*3
        std = [0.05]*3
        X = []
        centers = np.array([[-1.,-1.,1,1],[1.,-1.,-1.,-1],[-1.,1.,1.,-1.]])

        for i, (n, std) in enumerate(zip(n_samples, std)):
            X.append(centers[i] + np.random.normal(scale=std,size=(n, 4)))
        X = np.concatenate(X)
        np.random.shuffle(X)
        self.signal = signals.Signal1D(X)
        self.signal.decomposition()
        self.signal.blind_source_separation(number_of_components=3)

    @pytest.mark.parametrize("algorithm", ("kmeans", "agglomerative"))
    @pytest.mark.parametrize("cluster_source", ("signal","bss","decomposition"))
    @pytest.mark.parametrize("metric", ("elbow", "silhouette", "gap"))
    def test_custom(self,  cluster_source,
                algorithm,metric):
            custom_scaling = import_sklearn.sklearn.preprocessing.MinMaxScaler()
            max_clusters = 8
            self.signal.estimate_number_of_clusters(
                cluster_source,
                max_clusters=max_clusters,
                scaling=custom_scaling,
                algorithm=algorithm,
                metric=metric)
            best_k = self.signal.learning_results.estimated_number_of_clusters
            if isinstance(best_k, list):
                best_k = best_k[0]

            np.testing.assert_allclose(best_k, 3)
            
            

class TestCustomClusterAlgorithm:
    
    def setup_method(self):
        self.signal = signals.Signal2D(np.random.rand(11, 5, 7))
        self.signal.decomposition()
        self.signal.blind_source_separation(number_of_components=3)
        self.navigation_mask = np.zeros((11,), dtype=bool)
        self.navigation_mask[4:6] = True
        self.signal_mask = np.zeros((5, 7), dtype=bool)
        self.signal_mask[1:4, 2:6] = True
        self.signal.decomposition()
        self.signal.blind_source_separation(number_of_components=3)

    @pytest.mark.parametrize("cluster_source", ("signal","bss","decomposition"))
    @pytest.mark.parametrize("scaling", ("standard", "norm", "minmax",None))
    def test_custom(self, cluster_source, scaling):
        custom_method =  import_sklearn.sklearn.cluster.KMeans(3)
        self.signal.cluster_analysis(cluster_source=\
                                          cluster_source,n_clusters=3,
                                      scaling=scaling,
                                      algorithm=custom_method,
                                      )
        np.testing.assert_array_equal(
            self.signal.learning_results.cluster_labels.shape, (3, 11))
        np.testing.assert_array_equal(
            self.signal.learning_results.cluster_centers.shape, (3, 35))



def test_cluster_source_error():
    rng = np.random.RandomState(123)
    x = rng.random((20, 100))
    s = signals.Signal1D(x)
    with pytest.raises(ValueError, match="cluster source needs to be set "
                                     "to `decomposition` , `signal` , `bss` "
                                     "or a suitable Signal"):
        s.cluster_analysis("randtest")



def test_cluster_source_size_error():
    rng = np.random.RandomState(123)
    x = rng.random((20, 100))
    s = signals.Signal1D(x)
    x2 = rng.random((10, 80))
    s2 = signals.Signal1D(x2)
    with pytest.raises(ValueError, match="cluster_source does not have the same "
                                 "navigation size as the this signal"):
        s.cluster_analysis(s2)

def test_cluster_source_center_size_error():
    rng = np.random.RandomState(123)
    x = rng.random((20, 100))
    s = signals.Signal1D(x)
    x2 = rng.random((10, 80))
    s2 = signals.Signal1D(x2)
    with pytest.raises(ValueError, match="cluster_source does not have the same "
                                 "navigation size as the this signal"):
        s.cluster_analysis("signal",source_for_centers=s2)


def test_cluster_bss_error():
    rng = np.random.RandomState(123)
    x = rng.random((20, 100))
    s = signals.Signal1D(x)
    with pytest.raises(ValueError, match="A cluster source has been set to bss "
                         " but no blind source separation results found. "
                         " Please run blind source separation method first"):
        s.cluster_analysis("bss")

def test_cluster_decomposition_error():
    rng = np.random.RandomState(123)
    x = rng.random((20, 100))
    s = signals.Signal1D(x)
    with pytest.raises(ValueError, match="A cluster source has been set to "
                       "decomposition but no decomposition results found. "
                       "Please run decomposition method first"):
        s.cluster_analysis("decomposition")


def test_cluster_nav_mask_error():
    rng = np.random.RandomState(123)
    x = rng.random((20, 100))
    nav_mask = np.zeros((11,), dtype=bool)
    s = signals.Signal1D(x)
    with pytest.raises(ValueError, match="Navigation mask size does not match "
                       "signal navigation size"):
        s.cluster_analysis("signal",navigation_mask=nav_mask)

def test_cluster_sig_mask_error():
    rng = np.random.RandomState(123)
    x = rng.random((20, 100))
    sig_mask = np.zeros((11,), dtype=bool)
    s = signals.Signal1D(x)
    with pytest.raises(ValueError, match="signal mask size does not match your "
                                     "cluster source signal size"):
        s.cluster_analysis("signal",signal_mask=sig_mask)

def test_cluster_basesig_mask_error():
    rng = np.random.RandomState(123)
    x = rng.random((20, 100))
    sig_mask = np.zeros((11,), dtype=bool)
    s = signals.Signal1D(x)
    with pytest.raises(ValueError, match="signal mask size does not match your "
                                     "cluster source signal size"):
        s.cluster_analysis(s.copy(),signal_mask=sig_mask)


def test_cluster_ncluster_error():
    rng = np.random.RandomState(123)
    x = rng.random((20, 100))
    s = signals.Signal1D(x)
    with pytest.raises(ValueError, match="The number of clusters, n_clusters "
                             "must be specified and be >= 2."):
        s.cluster_analysis("signal",n_clusters=1)



def test_max_cluster_error():
    np.random.seed(1)
    # Use prime numbers to avoid fluke equivalences
    # create 3 random clusters
    n_samples=[400]*3
    std = [0.05]*3
    X = []
    centers = np.array([[-1.,-1.,1,1],[1.,-1.,-1.,-1],[-1.,1.,1.,-1.]])
    for i, (n, std) in enumerate(zip(n_samples, std)):
        X.append(centers[i] + np.random.normal(scale=std,size=(n, 4)))
    X = np.concatenate(X)
    np.random.shuffle(X)
    signal = signals.Signal1D(X)
    signal.decomposition()
    max_clusters = 1
    with pytest.raises(ValueError, match="The max number of clusters, max_clusters, "
                             "must be specified and be >= 2."):

        signal.estimate_number_of_clusters(
                "signal",
                max_clusters=max_clusters,
                scaling=None,
                algorithm="kmeans",
                metric="elbow")

def test_cluster_scaling_object_error():
    rng = np.random.RandomState(123)
    x = rng.random((20, 100))
    s = signals.Signal1D(x)
    scaling = object()
    with pytest.raises(ValueError, match="The cluster scaling method should be either \w*"):
        s.cluster_analysis("signal",scaling=scaling)
    

def test_clustering_object_error():
    rng = np.random.RandomState(123)
    x = rng.random((20, 100))
    s = signals.Signal1D(x)
    empty_object = object()
    with pytest.raises(ValueError, match="The clustering method should be either \w*"):
        s.cluster_analysis("signal",algorithm=empty_object)
    
