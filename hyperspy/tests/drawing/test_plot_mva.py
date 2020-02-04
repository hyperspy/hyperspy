# -*- coding: utf-8 -*-
# Copyright 2007-2015 The HyperSpy developers
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


baseline_dir = 'plot_mva'
default_tol = 2.0


class TestPlotDecomposition:

    def setup_method(self, method):
        np.random.seed(1)
        sources = np.random.random(size=(5, 100))
        np.random.seed(1)
        mixmat = np.random.random((100, 5))
        self.s = signals.Signal1D(np.dot(mixmat, sources))
        np.random.seed(1)
        self.s.add_gaussian_noise(.1)
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
        from sklearn.datasets import make_blobs
        np.random.seed(1)
        data = make_blobs(n_samples=200, n_features=10, 
                          shuffle=False,random_state=1)[0]
        # nav1, sig1
        s = signals.Signal1D(data.reshape(200, 10))
        # nav2, sig1
        s2 = signals.Signal1D(data.reshape(20, 10, 10))
        # nav2, sig2
        s3 = signals.Signal2D(data.reshape(4, 5, 10, 10))

        # Run decomposition and cluster analysis
        s.decomposition()
        s.cluster_analysis(3, algorithm='kmeans',
                           scaling="minmax", random_state=1)
        s2.decomposition()
        s2.cluster_analysis(3, algorithm='kmeans',
                            scaling="minmax", random_state=1)
        s3.decomposition()
        s3.cluster_analysis(3, algorithm='kmeans',
                            scaling="minmax", random_state=1)

        self.s = s
        self.s2 = s2
        self.s3 = s3

    @pytest.mark.mpl_image_compare(
        baseline_dir=baseline_dir, tolerance=default_tol)
    def test_plot_cluster_labels_nav1_sig1(self):
        return self.s.plot_cluster_labels()

    @pytest.mark.mpl_image_compare(
        baseline_dir=baseline_dir, tolerance=default_tol)
    def test_plot_cluster_centers_nav1_sig1(self):
        return self.s.plot_cluster_centers()

    @pytest.mark.mpl_image_compare(
        baseline_dir=baseline_dir, tolerance=default_tol)
    def test_plot_cluster_labels_nav2_sig1(self):
        return self.s2.plot_cluster_labels()

    @pytest.mark.mpl_image_compare(
        baseline_dir=baseline_dir, tolerance=default_tol)
    def test_plot_cluster_centers_nav2_sig1(self):
        return self.s2.plot_cluster_centers()

    @pytest.mark.mpl_image_compare(
        baseline_dir=baseline_dir, tolerance=default_tol)
    def test_plot_cluster_labels_nav2_sig2(self):
        return self.s3.plot_cluster_labels()

    @pytest.mark.mpl_image_compare(
        baseline_dir=baseline_dir, tolerance=default_tol)
    def test_plot_cluster_centers_nav2_sig2(self):
        return self.s3.plot_cluster_centers()

def test_plot_without_decomposition():
    sources = np.random.random(size=(5, 100))
    mixmat = np.random.random((100, 5))
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
