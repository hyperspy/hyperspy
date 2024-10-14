# -*- coding: utf-8 -*-
# Copyright 2007-2024 The HyperSpy developers
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

import pytest

import hyperspy.api as hs
from hyperspy.utils.baseline_removal_tool import (
    ALGORITHMS_MAPPING_POLYNOMIAL,
    PARAMETERS_ALGORITHMS,
    BaselineRemoval,
)

pytest.importorskip("pybaselines")


def test_remove_baseline():
    s = hs.data.two_gaussians().inav[:5, :5]

    assert s.isig[:10].data.mean() > 20
    s2 = s.remove_baseline(method="aspls", lam=1e7, inplace=False)
    assert s.isig[:10].data.mean() > 20
    assert s2.isig[:10].data.mean() < 5

    s.remove_baseline(method="aspls", lam=1e7)
    assert s.isig[:10].data.mean() < 5


def test_remove_baseline_apply_close():
    s = hs.data.two_gaussians().inav[:5, :5]
    assert s.isig[:10].data.mean() > 20

    # open/close cycle
    br = BaselineRemoval(s)
    assert br.estimator_line is not None
    br.close()
    assert br.estimator_line is None
    assert s.isig[:10].data.mean() > 20

    br = BaselineRemoval(s)
    assert br.estimator_line is not None
    br.algorithm = "Adaptive Smoothness Penalized Least Squares"
    br.lam = 1e7
    # move to different index to call update line
    s.axes_manager.indices = (2, 4)
    br.apply()
    assert br.estimator_line is None
    assert s.isig[:10].data.mean() < 5


def test_baseline_removal_tool_enable():
    s = hs.data.two_gaussians().inav[:5, :5]

    br = BaselineRemoval(s)

    # Whittaker
    assert br.algorithm == "Adaptive Smoothness Penalized Least Squares"
    for parameter in PARAMETERS_ALGORITHMS.keys():
        # only "lam" and "diff_order" are enable
        if parameter in ["lam", "diff_order"]:
            result = True
        else:
            result = False
        assert getattr(br, f"_enable_{parameter}") is result

    for algorithm in [
        "Asymmetric Least Squares",
        "Peaked Signal's Asymmetric Least Squares Algorithm",
        "Derivative Peak-Screening Asymmetric Least Squares Algorithm",
    ]:
        br.algorithm = algorithm
        assert br.algorithm is algorithm
        assert br._enable_p is True

    br.algorithm = "Improved Asymmetric Least Squares"
    assert br._enable_lam_1 is True

    br.algorithm = "Doubly Reweighted Penalized Least Squares"
    assert br._enable_lam_1 is False
    assert br._enable_p is False
    assert br._enable_eta is True

    # Polynomial
    for algorithm in ALGORITHMS_MAPPING_POLYNOMIAL.keys():
        br.algorithm = algorithm
        assert br._enable_p is False
        assert br._enable_lam is False
        assert br._enable_lam_1 is False
        assert br._enable_eta is False
        assert br._enable_poly_order is True
        assert br._enable_diff_order is False
        assert br._enable_penalized_spline is False

    # Splines
    for algorithm in [
        "Mixture Model",
        "Iterative Reweighted Spline Quantile Regression",
    ]:
        br.algorithm = algorithm
        assert br._enable_p is (algorithm == "Mixture Model")
        assert br._enable_lam is True
        assert br._enable_lam_1 is False
        assert br._enable_eta is False
        assert br._enable_poly_order is False
        assert br._enable_diff_order is True
        assert br._enable_penalized_spline is False
