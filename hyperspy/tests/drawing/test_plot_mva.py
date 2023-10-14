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

import hyperspy
from hyperspy import signals
from hyperspy.misc.machine_learning.import_sklearn import sklearn_installed

baseline_dir = 'plot_mva'
default_tol = 2.0


class TestPlotDecomposition:

    def setup_method(self, method):
        rng = np.random.default_rng(1)
        sources = rng.random(size=(5, 100))
        mixmat = rng.random((100, 5))
        self.s = signals.Signal1D(np.dot(mixmat, sources))
        self.s.add_gaussian_noise(.1, random_state=rng)
        self.s.decomposition()
        self.s2 = signals.Signal1D(self.s.data.reshape(10, 10, 100))
        self.s2.decomposition()

    def _generate_parameters():
        parameters = []
        for n in [10, 50]:
            for xaxis_type in ['index', 'number']:
                for threshold in [0, 0.001]:
                    for xaxis_labeling in ['ordinal', 'cardinal']:
                        parameters.append([n, threshold, xaxis_type,
                                           xaxis_labeling])
        return parameters

    @pytest.mark.parametrize(("n", "threshold", "xaxis_type", "xaxis_labeling"),
                             _generate_parameters())
    @pytest.mark.mpl_image_compare(
        baseline_dir=baseline_dir, tolerance=default_tol)
    def test_plot_explained_variance_ratio(self, n, threshold, xaxis_type,
                                           xaxis_labeling):
        ax = self.s.plot_explained_variance_ratio(n, threshold=threshold,
                                                  xaxis_type=xaxis_type,
                                                  xaxis_labeling=xaxis_labeling)
        return ax.get_figure()

    @pytest.mark.mpl_image_compare(
        baseline_dir=baseline_dir, tolerance=default_tol)
    def test_plot_cumulative_explained_variance_ratio(self):
        ax = self.s.plot_cumulative_explained_variance_ratio()
        return ax.get_figure()

    @pytest.mark.parametrize("n", [3, [3, 4]])
    @pytest.mark.mpl_image_compare(
        baseline_dir=baseline_dir, tolerance=default_tol)
    def test_plot_decomposition_loadings_nav1(self, n):
        return self.s.plot_decomposition_loadings(n)

    @pytest.mark.parametrize("n", (3, [3, 4]))
    @pytest.mark.mpl_image_compare(
        baseline_dir=baseline_dir, tolerance=default_tol)
    def test_plot_decomposition_factors_nav1(self, n):
        return self.s.plot_decomposition_factors(n)

    @pytest.mark.parametrize(("n", "per_row", "axes_decor"),
                             ((6, 3, 'all'), (8, 4, None),
                              ([3, 4, 5, 6], 2, 'ticks')))
    @pytest.mark.mpl_image_compare(
        baseline_dir=baseline_dir, tolerance=default_tol)
    def test_plot_decomposition_loadings_nav2(self, n, per_row, axes_decor):
        return self.s2.plot_decomposition_loadings(n, per_row=per_row,
                                                   title='Loading',
                                                   axes_decor=axes_decor)


@pytest.mark.skipif(not sklearn_installed, reason="sklearn not installed")
class TestPlotClusterAnalysis:

    def setup_method(self, method):
        rng = np.random.default_rng(1)
        # Use prime numbers to avoid fluke equivalences
        # create 3 random clusters
        n_samples=[250,100,50]
        std = [1.0,2.0,0.5]
        X = []
        centers = np.array([[-15.0, -15.0,-15.0], [1.0, 1.0,1.0],
                            [15.0, 15.0, 15.0]])
        for i, (n, std) in enumerate(zip(n_samples, std)):
            X.append(centers[i] + rng.normal(scale=std, size=(n, 3)))

        data = np.concatenate(X)

        # nav1, sig1
        s = signals.Signal1D(data.reshape(400, 3))
        # nav2, sig1
        s2 = signals.Signal1D(data.reshape(40, 10, 3))

        import sklearn
        n_init = "auto" if Version(sklearn.__version__) >= Version('1.3') else 10

        # Run decomposition and cluster analysis
        s.decomposition()
        s.cluster_analysis("decomposition", n_clusters=3, algorithm='kmeans',
                           preprocessing="minmax", random_state=0,
                           n_init=n_init)
        s.estimate_number_of_clusters(
            "decomposition", metric="elbow", n_init=n_init,
            )

        s2.decomposition()
        s2.cluster_analysis("decomposition", n_clusters=3, algorithm='kmeans',
                            preprocessing="minmax", random_state=0,
                            n_init=n_init)

        data = np.zeros((2000, 5))
        data[:250*5:5, :] = 10
        data[2 + 250*5:350*5:5, :] = 2
        data[350*5:400*5, 4] = 20

        # nav2, sig2
        s3 = signals.Signal2D(data.reshape(20, 20, 5, 5))
        s3.decomposition()
        s3.cluster_analysis("decomposition", n_clusters=3, algorithm='kmeans',
                            preprocessing="minmax", random_state=0,
                            n_init=n_init)

        self.s = s
        self.s2 = s2
        self.s3 = s3

    @pytest.mark.mpl_image_compare(
        baseline_dir=baseline_dir, tolerance=default_tol)
    def test_plot_cluster_labels_nav1_sig1(self):
        return self.s.plot_cluster_labels()

    @pytest.mark.mpl_image_compare(
        baseline_dir=baseline_dir, tolerance=default_tol)
    def test_plot_cluster_signals_nav1_sig1(self):
        return self.s.plot_cluster_signals()

    @pytest.mark.mpl_image_compare(
        baseline_dir=baseline_dir, tolerance=default_tol)
    def test_plot_cluster_distances_nav1_sig1(self):
        return self.s.plot_cluster_distances()

    @pytest.mark.mpl_image_compare(
        baseline_dir=baseline_dir, tolerance=default_tol)
    def test_plot_cluster_labels_nav2_sig1(self):
        return self.s2.plot_cluster_labels()

    @pytest.mark.mpl_image_compare(
        baseline_dir=baseline_dir, tolerance=default_tol)
    def test_plot_cluster_signals_nav2_sig1(self):
        return self.s2.plot_cluster_signals()

    @pytest.mark.mpl_image_compare(
        baseline_dir=baseline_dir, tolerance=default_tol)
    def test_plot_cluster_labels_nav2_sig2(self):
        return self.s3.plot_cluster_labels()

    @pytest.mark.mpl_image_compare(
        baseline_dir=baseline_dir, tolerance=default_tol*5)
    def test_plot_cluster_distances_nav2_sig2(self):
        return self.s3.plot_cluster_distances()

    @pytest.mark.mpl_image_compare(
        baseline_dir=baseline_dir, tolerance=default_tol)
    def test_plot_cluster_signals_nav2_sig2(self):
        return self.s3.plot_cluster_signals()

    @pytest.mark.skipif(Version(hyperspy.__version__) < Version("2.0"),
                        reason="Failing on CI for 1.7.6.dev0")
    @pytest.mark.mpl_image_compare(
        baseline_dir=baseline_dir, tolerance=default_tol*2)
    def test_plot_cluster_metric(self):
        ax = self.s.plot_cluster_metric()
        return ax.get_figure()

    def test_except_nocluster_metric(self):
        with pytest.raises(ValueError):
            self.s2.plot_cluster_metric()


def test_plot_signal_dimension3():
    rng = np.random.default_rng(1)
    sources = rng.random(size=(5, 100))
    mixmat = rng.random((100, 5))
    s = signals.Signal1D(np.dot(mixmat, sources))
    s.add_gaussian_noise(.1, random_state=rng)
    s2 = signals.Signal1D(s.data.reshape(2, 5, 10, 100))

    s3 = s2.transpose(signal_axes=3)
    s3.decomposition()
    s3.plot_decomposition_results()

    s4 = s2.transpose(signal_axes=1)
    s4.decomposition()
    s4.plot_decomposition_results()


def test_plot_without_decomposition():
    rng = np.random.default_rng(1)
    sources = rng.random(size=(5, 100))
    mixmat = rng.random((100, 5))
    s = signals.Signal1D(np.dot(mixmat, sources))
    with pytest.raises(RuntimeError):
        s.plot_decomposition_factors()
    with pytest.raises(RuntimeError):
        s.plot_decomposition_loadings()
    with pytest.raises(RuntimeError):
        s.plot_decomposition_results()
    s.decomposition()
    with pytest.raises(RuntimeError):
        s.plot_bss_factors()
    with pytest.raises(RuntimeError):
        s.plot_bss_loadings()
