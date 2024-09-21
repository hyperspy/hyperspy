# -*- coding: utf-8 -*-
# Copyright 2007-2023 The HyperSpy developers
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

import itertools
import time

import traits.api as t

import hyperspy
from hyperspy.ui_registry import add_gui_method

# Whittaker
algorithms_mapping_whittaker = {
    "Asymmetric Least Squares": "asls",
    "Improved Asymmetric Least Squares": "iasls",
    "Adaptive Iteratively Reweighted Penalized Least Squares": "airpls",
    "Asymmetrically Reweighted Penalized Least Squares": "arpls",
    "Doubly Reweighted Penalized Least Squares": "drpls",
    "Improved Asymmetrically Reweighted Penalized Least Squares": "iarpls",
    "Adaptive Smoothness Penalized Least Squares": "aspls",
    "Peaked Signal's Asymmetric Least Squares Algorithm": "psalsa",
    "Derivative Peak-Screening Asymmetric Least Squares Algorithm": "derpsalsa",
}
# Polynomial
algorithms_mapping_polynomial = {
    "Regular Polynomial": "poly",
    "Modified Polynomial": "modpoly",
    "Improved Modified Polynomial": "imodpoly",
    "Locally Estimated Scatterplot Smoothing": "loess",
}
# Splines
algorithms_mapping_splines = {
    "Mixture Model": "mixture_model",
    "Iterative Reweighted Spline Quantile Regression": "irsqr",
}  # + Penalized splines version
algorithms_mapping = dict(algorithms_mapping_whittaker)
algorithms_mapping.update(algorithms_mapping_splines)
algorithms_mapping.update(algorithms_mapping_polynomial)

algorithms_parameters = {
    # Whittaker
    "asls": ("lam", "p", "diff_order"),
    "iasls": ("lam", "lam_1", "p", "diff_order"),
    "airpls": ("lam", "diff_order"),
    "arpls": ("lam", "diff_order"),
    "drpls": ("lam", "eta", "diff_order"),
    "iarpls": ("lam", "diff_order"),
    "aspls": ("lam", "diff_order"),
    "psalsa": ("lam", "p", "diff_order"),
    "derpsalsa": ("lam", "p", "diff_order"),
    # Polynomial
    "poly": ("poly_order",),
    "modpoly": ("poly_order",),
    "imodpoly": ("poly_order",),
    "loess": ("poly_order",),
    # Splines
    "mixture_model": (
        "lam",
        "p",
        "num_knots",
        "spline_degree",
        "diff_order",
        "symmetric",
    ),
    "irsqr": ("lam", "quantile", "num_knots", "spline_degree", "diff_order"),
}

algorithms_mapping_inverse = {v: k for k, v in algorithms_mapping.items()}
# Get the mapping of parameters to algorithms
parameters_algorithms = {}
for parameter in set(itertools.chain(*algorithms_parameters.values())):
    algorithm_list = [
        algorithms_mapping_inverse[algorithm]
        for algorithm, parameters in algorithms_parameters.items()
        if parameter in parameters
    ]
    parameters_algorithms[parameter] = algorithm_list


def _baseline_fitting(data, baseline_fitter, **kwargs):
    return data - baseline_fitter(data, **kwargs)[0]


@add_gui_method(toolkey="hyperspy.Signal1D.remove_baseline")
class BaselineRemoval(t.HasTraits):
    algorithm = t.Enum(
        *algorithms_mapping.keys(),
        default="Adaptive Smoothness Penalized Least Squares",
    )
    # Whittaker parameters
    lam = t.Range(1.0, 1e15, value=1e6)
    lam_1 = t.Range(1e-10, 1.0, value=1e-4)
    p = t.Range(0.0, 1.0, value=0.5, exclude_low=True, exclude_high=True)
    eta = t.Range(0.0, 1.0, value=0.5)
    diff_order = t.Range(1, 3, value=2)
    penalized_spline = t.Bool()
    # Polynomial parameters
    poly_order = t.Range(1, 10, value=2)
    # Spline parameters
    num_knots = t.Range(10, 10000, value=100)
    spline_degree = t.Range(1, 5, value=3)
    symmetric = t.Bool()
    quantile = t.Range(0.001, 0.5, value=0.05)

    # these are used to know if parameters needs to be enable or not
    # Whittaker parameters
    _enable_p = t.Bool()
    _enable_lam = t.Bool()
    _enable_lam_1 = t.Bool()
    _enable_eta = t.Bool()
    _enable_diff_order = t.Bool()
    _enable_penalized_spline = t.Bool()
    # Polynomial parameters
    _enable_poly_order = t.Bool()
    # Splines parameters
    _enable_spline_parameters = t.Bool()

    # To display the time per pixel
    _time_per_pixel = t.Float(value=0)

    def __init__(self, signal):
        super().__init__()
        self.signal = signal
        # Plot the signal if not already plotted
        if signal._plot is None or not signal._plot.is_active:
            self.signal.plot()
        self.bl_line = None  # The baseline line
        self.estimator = None
        self.estimator_line = None
        # Use as default, good speed and baseline estimation
        self.algorithm = "Adaptive Smoothness Penalized Least Squares"
        self.set_estimator()

    def set_estimator(self):
        from pybaselines import Baseline

        algorithm = self._get_algorithm_abbreviation()
        self.estimator = getattr(
            Baseline(
                self.signal.axes_manager[-1].axis,
                check_finite=False,
            ),
            self._get_algorithm_abbreviation(),
        )

        # get algorithm full name
        algorithm = algorithms_mapping_inverse[self._get_algorithm_abbreviation(False)]

        self._enable_p = algorithm in parameters_algorithms["p"]
        self._enable_lam = algorithm in parameters_algorithms["lam"]
        self._enable_lam_1 = algorithm in parameters_algorithms["lam_1"]
        self._enable_eta = algorithm in parameters_algorithms["eta"]
        self._enable_diff_order = algorithm in parameters_algorithms["diff_order"]
        self._enable_poly_order = algorithm in algorithms_mapping_polynomial.keys()
        self._enable_penalized_spline = algorithm in algorithms_mapping_whittaker.keys()
        self._enable_spline_parameters = (
            algorithm in parameters_algorithms["spline_degree"] or self.penalized_spline
        )
        self._update_lines()

    @t.observe(
        [
            "penalized_spline",
            "algorithm",
        ],
        post_init=True,
    )
    def _update_estimator(self, event):
        self.set_estimator()
        self._update_lines()

    def _get_algorithm_abbreviation(self, with_prefix=True):
        algorithm = algorithms_mapping[self.algorithm]
        if (
            self.penalized_spline
            and with_prefix
            and self.algorithm not in algorithms_mapping_splines
        ):
            algorithm = "pspline_" + algorithm

        return algorithm

    def _get_kwargs(self):
        args_name = algorithms_parameters[self._get_algorithm_abbreviation(False)]
        if self.penalized_spline:
            args_name += ("num_knots", "spline_degree")
        kwargs = {key: getattr(self, key) for key in args_name}
        return kwargs

    def _baseline_to_plot(self, *args, **kwargs):
        start = time.perf_counter_ns()
        out = self.estimator(self.signal._get_current_data(), **self._get_kwargs())[0]
        end = time.perf_counter_ns()
        self._time_per_pixel = (end - start) / 1e6
        return out

    def _create_lines(self):
        self.estimator_line = hyperspy.drawing.signal1d.Signal1DLine()
        self.estimator_line.data_function = self._baseline_to_plot
        self.estimator_line.set_line_properties(color="blue", type="line", scaley=False)
        self.signal._plot.signal_plot.add_line(self.estimator_line)
        self.estimator_line.autoscale = ""
        self.estimator_line.plot()

    @t.observe(
        [
            "lam",
            "lam_1",
            "p",
            "eta",
            "diff_order",
            "num_knots",
            "spline_degree",
            "symmetric",
            "poly_order",
            "quantile",
        ],
        post_init=True,
    )
    def _update_lines(self, event=None):
        if self.estimator_line is None:
            self._create_lines()
        try:
            self.estimator_line.update(render_figure=True, update_ylimits=True)
        except AttributeError:
            # in case the figure is closed
            # to fix this, the callback should be disconnected correctly
            # when closing the figure
            pass

    def apply(self):
        self.signal.remove_baseline(
            algorithm=algorithms_mapping[self.algorithm],
            inplace=True,
            **self._get_kwargs(),
        )
        self.close()

    def close(self):
        if self.signal._plot.is_active and self.estimator_line is not None:
            self.estimator_line.close()
            self.estimator_line = None
