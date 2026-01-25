# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
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
import traits.api as t

from hyperspy.drawing._markers.circles import Circles
from hyperspy.drawing.markers import convert_positions
from hyperspy.exceptions import SignalDimensionError
from hyperspy.ui_registry import add_gui_method


@add_gui_method(toolkey="hyperspy.Signal2D.find_peaks")
class PeaksFinder2D(t.HasTraits):
    method = t.Enum(
        "Local max",
        "Max",
        "Minmax",
        "Zaefferer",
        "Stat",
        "Laplacian of Gaussian",
        "Difference of Gaussian",
        "Template matching",
        default="Local Max",
    )
    # For "Local max" method
    local_max_distance = t.Range(1, 20, value=3)
    local_max_threshold = t.Range(0, 20.0, value=10)
    # For "Max" method
    max_alpha = t.Range(0, 6.0, value=3)
    max_distance = t.Range(1, 20, value=10)
    # For "Minmax" method
    minmax_distance = t.Range(0, 6.0, value=3)
    minmax_threshold = t.Range(0, 20.0, value=10)
    # For "Zaefferer" method
    zaefferer_grad_threshold = t.Range(0, 0.2, value=0.1)
    zaefferer_window_size = t.Range(2, 80, value=40)
    zaefferer_distance_cutoff = t.Range(1, 100.0, value=50)
    # For "Stat" method
    stat_alpha = t.Range(0, 2.0, value=1)
    stat_window_radius = t.Range(5, 20, value=10)
    stat_convergence_ratio = t.Range(0, 0.1, value=0.05)
    # For "Laplacian of Gaussian" method
    log_min_sigma = t.Range(0, 2.0, value=1)
    log_max_sigma = t.Range(0, 100.0, value=50)
    log_num_sigma = t.Range(0, 20, value=10)
    log_threshold = t.Range(0, 0.4, value=0.2)
    log_overlap = t.Range(0, 1.0, value=0.5)
    log_log_scale = t.Bool(False)
    # For "Difference of Gaussian" method
    dog_min_sigma = t.Range(0, 2.0, value=1)
    dog_max_sigma = t.Range(0, 100.0, value=50)
    dog_sigma_ratio = t.Range(0, 3.2, value=1.6)
    dog_threshold = t.Range(0, 0.4, value=0.2)
    dog_overlap = t.Range(0, 1.0, value=0.5)
    # For "Cross correlation" method
    xc_template = None
    xc_distance = t.Range(0, 100.0, value=5.0)
    xc_threshold = t.Range(0, 10.0, value=0.5)

    random_navigation_position = t.Button()
    compute_over_navigation_axes = t.Button()

    show_navigation_sliders = t.Bool(False)

    def __init__(self, signal, method, peaks=None, **kwargs):
        self._attribute_argument_mapping_local_max = {
            "local_max_distance": "min_distance",
            "local_max_threshold": "threshold_abs",
        }
        self._attribute_argument_mapping_max = {
            "max_alpha": "alpha",
            "max_distance": "distance",
        }
        self._attribute_argument_mapping_local_minmax = {
            "minmax_distance": "distance",
            "minmax_threshold": "threshold",
        }
        self._attribute_argument_mapping_local_zaefferer = {
            "zaefferer_grad_threshold": "grad_threshold",
            "zaefferer_window_size": "window_size",
            "zaefferer_distance_cutoff": "distance_cutoff",
        }
        self._attribute_argument_mapping_local_stat = {
            "stat_alpha": "alpha",
            "stat_window_radius": "window_radius",
            "stat_convergence_ratio": "convergence_ratio",
        }
        self._attribute_argument_mapping_local_log = {
            "log_min_sigma": "min_sigma",
            "log_max_sigma": "max_sigma",
            "log_num_sigma": "num_sigma",
            "log_threshold": "threshold",
            "log_overlap": "overlap",
            "log_log_scale": "log_scale",
        }
        self._attribute_argument_mapping_local_dog = {
            "dog_min_sigma": "min_sigma",
            "dog_max_sigma": "max_sigma",
            "dog_sigma_ratio": "sigma_ratio",
            "dog_threshold": "threshold",
            "dog_overlap": "overlap",
        }
        self._attribute_argument_mapping_local_xc = {
            "xc_template": "template",
            "xc_distance": "distance",
            "xc_threshold": "threshold",
        }

        self._attribute_argument_mapping_dict = {
            "local_max": self._attribute_argument_mapping_local_max,
            "max": self._attribute_argument_mapping_max,
            "minmax": self._attribute_argument_mapping_local_minmax,
            "zaefferer": self._attribute_argument_mapping_local_zaefferer,
            "stat": self._attribute_argument_mapping_local_stat,
            "laplacian_of_gaussian": self._attribute_argument_mapping_local_log,
            "difference_of_gaussian": self._attribute_argument_mapping_local_dog,
            "template_matching": self._attribute_argument_mapping_local_xc,
        }

        if signal.axes_manager.signal_dimension != 2:
            raise SignalDimensionError(signal.axes.signal_dimension, 2)

        self._set_parameters_observer()
        self.on_trait_change(
            self.set_random_navigation_position, "random_navigation_position"
        )

        self.signal = signal
        self.peaks = peaks
        self.markers = None
        if self.signal._plot is None or not self.signal._plot.is_active:
            self.signal.plot()
        if self.signal.axes_manager.navigation_size > 0:
            self.show_navigation_sliders = True
            self.signal.axes_manager.events.indices_changed.connect(
                self._update_peak_finding, []
            )
            self.signal._plot.signal_plot.events.closed.connect(self.disconnect, [])
        # Set initial parameters:
        # As a convenience, if the template argument is provided, we keep it
        # even if the method is different, to be able to use it later.
        if "template" in kwargs.keys():
            self.xc_template = kwargs["template"]
        if method is not None:
            method_dict = {
                "local_max": "Local max",
                "max": "Max",
                "minmax": "Minmax",
                "zaefferer": "Zaefferer",
                "stat": "Stat",
                "laplacian_of_gaussian": "Laplacian of Gaussian",
                "difference_of_gaussian": "Difference of Gaussian",
                "template_matching": "Template matching",
            }
            self.method = method_dict[method]
        self._parse_paramaters_initial_values(**kwargs)
        self._update_peak_finding()

    def _parse_paramaters_initial_values(self, **kwargs):
        # Get the attribute to argument mapping for the current method
        arg_mapping = self._attribute_argument_mapping_dict[
            self._normalise_method_name(self.method)
        ]
        for attr, arg in arg_mapping.items():
            if arg in kwargs.keys():
                setattr(self, attr, kwargs[arg])

    def _update_peak_finding(self, method=None):
        if method is None:
            method = self.method
        self._find_peaks_current_index(method=method)
        self._plot_markers()

    def _method_changed(self, old, new):
        if new == "Template matching" and self.xc_template is None:
            raise RuntimeError('The "template" argument is required.')
        self._update_peak_finding(method=new)

    def _parameter_changed(self, old, new):
        self._update_peak_finding()

    def _set_parameters_observer(self):
        for parameters_mapping in self._attribute_argument_mapping_dict.values():
            for parameter in list(parameters_mapping.keys()):
                self.on_trait_change(self._parameter_changed, parameter)

    def _get_parameters(self, method):
        # Get the attribute to argument mapping for the given method
        arg_mapping = self._attribute_argument_mapping_dict[method]
        # return argument and values as kwargs
        return {arg: getattr(self, attr) for attr, arg in arg_mapping.items()}

    def _normalise_method_name(self, method):
        return method.lower().replace(" ", "_")

    def _find_peaks_current_index(self, method):
        method = self._normalise_method_name(method)
        self.peaks.data = self.signal.find_peaks(
            method,
            current_index=True,
            interactive=False,
            **self._get_parameters(method),
        )

    def _plot_markers(self):
        offsets = self.peaks.data
        offsets = convert_positions(offsets, self.signal.axes_manager.signal_axes)
        if self.markers is None:
            self.markers = Circles(
                offsets=offsets,
                edgecolor="red",
                facecolors="none",
                sizes=20,
                units="points",
            )
        else:
            self.markers.offsets = offsets

    def compute_navigation(self):
        method = self._normalise_method_name(self.method)
        with self.signal.axes_manager.events.indices_changed.suppress():
            self.peaks.data = self.signal.find_peaks(
                method,
                interactive=False,
                current_index=False,
                **self._get_parameters(method),
            )

    def close(self):
        # remove markers
        if self.signal._plot is not None and self.signal._plot.is_active:
            self.signal._plot.signal_plot.remove_markers(render_figure=True)
        self.disconnect()

    def disconnect(self):
        # disconnect event
        am = self.signal.axes_manager
        if self._update_peak_finding in am.events.indices_changed.connected:
            am.events.indices_changed.disconnect(self._update_peak_finding)

    def set_random_navigation_position(self):
        index = self._rng.integers(0, self.signal.axes_manager._max_index)
        self.signal.axes_manager.indices = np.unravel_index(
            index, tuple(self.signal.axes_manager._navigation_shape_in_array)
        )[::-1]
