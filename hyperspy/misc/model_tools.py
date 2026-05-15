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


import numbers
from collections import defaultdict
from collections.abc import Iterable

import numpy as np
from prettytable import PrettyTable

from hyperspy.misc._utils import _parse_percentile_value


def _format_string(val, format_string=".5g", max_length=None, add_ellipsis=True):
    """
    Returns formatted string for a value unless it equals None,
    then empty string is returned.

    Parameters
    ----------
    val : any
        Value to format
    format_string : str, optional
        For numeric types only: the format string to use. Default is ".5g".
    max_length : int or None, optional
        Maximum length of the returned string. If None, no maximum length
        is applied. Default is None.
    add_ellipsis : bool, optional
        Whether to add ellipsis when truncating the string.
        Default is True.
    """
    if val is None:
        to_return = ""
    elif isinstance(val, str):
        to_return = val
    elif isinstance(val, Iterable):
        to_return = ", ".join(
            f"{v:{format_string}}" if isinstance(v, numbers.Number) else str(v)
            for v in val
        )
        to_return = f"({to_return})"
    else:
        to_return = f"{val:{format_string}}"

    if max_length is not None and len(to_return) > max_length:
        if add_ellipsis:
            # Add ellipsis to indicate truncation
            to_return = to_return[: max_length - 3] + "..."
        else:
            to_return = to_return[:max_length]

    return to_return


class CurrentComponentValues:
    """
    Convenience class that makes use of __repr__ methods for nice printing in
    the notebook of the properties of parameters of a component.

    Parameters
    ----------
    component : hyperspy component instance
    only_free : bool, default False
        If True: Only include the free parameters in the view
    only_active : bool, default False
        If True: Helper for ``CurrentModelValues``. Only include active
        components in the view. Always shows values if used on an individual
        component.
    """

    def __init__(self, component, only_free=False, only_active=False):
        self.name = component.name
        self.component_type = component.__class__.__name__
        self.active = component.active
        self.parameters = component.parameters
        self._id_name = component._id_name
        self.only_free = only_free
        self.only_active = only_active

    def _build_table(self):
        """Build and return a PrettyTable with parameter data."""

        def _num_fmt(f, v):
            return "" if v is None else "%.5g" % v

        table = PrettyTable()
        table.field_names = [
            "Parameter",
            "Free",
            "Value",
            "Std",
            "Min",
            "Max",
            "Linear",
        ]
        table.align["Parameter"] = "r"
        table.align["Free"] = "r"
        table.align["Value"] = "r"
        table.align["Std"] = "r"
        table.align["Min"] = "r"
        table.align["Max"] = "r"
        table.align["Linear"] = "r"
        table.custom_format = {
            "Value": _num_fmt,
            "Std": _num_fmt,
            "Min": _num_fmt,
            "Max": _num_fmt,
        }
        _widths = {
            "Parameter": 14,
            "Free": 7,
            "Value": 10,
            "Std": 10,
            "Min": 10,
            "Max": 10,
            "Linear": 6,
        }
        table.min_width = _widths
        table.max_width = _widths

        # Add rows
        for para in self.parameters:
            if not self.only_free or self.only_free and para.free:
                free = para.free if para.twin is None else "Twinned"
                ln = para._linear
                value = (
                    _format_string(para.value)
                    if isinstance(para.value, Iterable)
                    else para.value
                )
                table.add_row(
                    [
                        para.name,
                        str(free),
                        value,
                        para.std,
                        para.bmin,
                        para.bmax,
                        str(ln),
                    ]
                )
        return table

    def __repr__(self):
        if self.only_active:
            header = "{0}: {1}".format(self.component_type, self.name)
        else:
            header = "{0}: {1}\nActive: {2}".format(
                self.component_type, self.name, self.active
            )

        table = self._build_table()
        return header + "\n" + str(table)

    def _repr_html_(self):
        if self.only_active:
            header = "<p><b>{0}: {1}</b></p>".format(self.component_type, self.name)
        else:
            header = "<p><b>{0}: {1}</b><br />Active: {2}</p>".format(
                self.component_type, self.name, self.active
            )

        table = self._build_table()
        table_html = table.get_html_string(
            attributes={
                "style": "width:100%; border-collapse:collapse; text-align:center;",
                "border": "1",
            }
        )

        return header + table_html


class CurrentModelValues:
    """
    Convenience class that makes use of __repr__ methods for nice printing in
    the notebook of the properties of parameters in components in a model.

    Parameters
    ----------
    component : hyperspy component instance
    only_free : bool, default False
        If True: Only include the free parameters in the view
    only_active : bool, default False
        If True: Only include active parameters in the view
    """

    def __init__(self, model, only_free=False, only_active=False, component_list=None):
        self.model = model
        self.only_free = only_free
        self.only_active = only_active
        self.component_list = model if component_list is None else component_list

    def __repr__(self):
        text = "{}: {}\n".format(
            self.model.__class__.__name__, self.model.signal.metadata.General.title
        )
        for comp in self.component_list:
            if not self.only_active or self.only_active and comp.active:
                if not self.only_free or comp.free_parameters and self.only_free:
                    text += (
                        CurrentComponentValues(
                            component=comp,
                            only_free=self.only_free,
                            only_active=self.only_active,
                        ).__repr__()
                        + "\n"
                    )
        return text

    def _repr_html_(self):
        html = "<h4>{}: {}</h4>".format(
            self.model.__class__.__name__, self.model.signal.metadata.General.title
        )
        for comp in self.component_list:
            if not self.only_active or self.only_active and comp.active:
                if not self.only_free or comp.free_parameters and self.only_free:
                    html += CurrentComponentValues(
                        component=comp,
                        only_free=self.only_free,
                        only_active=self.only_active,
                    )._repr_html_()
        return html


def _calculate_covariance(
    target_signal, coefficients, component_data, residual=None, lazy=False
):
    """
    Calculate covariance matrix after having performed Linear Regression.

    Parameters
    ----------

    target_signal : array-like, shape (N,) or (M, N)
        The signal array to be fit to.
    coefficients : array-like, shape C or (M, C)
        The fitted coefficients.
    component_data : array-like, shape N or (C, N)
        The component data.
    residual : array-like, shape (0,) or (M,)
        The residual sum of squares, optional. Calculated if None.
    lazy : bool
        Whether the signal is lazy.

    Notes
    -----
    Explanation of the array shapes in HyperSpy terms:
    N : flattened signal shape
    M : flattened navigation shape
    C : number of components

    See https://stats.stackexchange.com/questions/62470 for more info on the
    algorithm
    """
    if target_signal.ndim > 1:
        fit = coefficients[..., None, :] * component_data.T[None]
    else:
        fit = coefficients * component_data.T

    if residual is None:
        residual = ((target_signal - fit.sum(-1)) ** 2).sum(-1)

    fit_dot = np.matmul(fit.swapaxes(-2, -1), fit)

    # A coefficient that is (numerically) exactly zero -- e.g. a component
    # that doesn't contribute for a given pixel -- makes the corresponding
    # row/column of fit_dot structurally zero, i.e. genuinely singular
    # rather than merely ill-conditioned. np.linalg.inv() cannot invert
    # that, so fall back to the Moore-Penrose pseudo-inverse, which is
    # well-defined for singular matrices.
    #
    # Try inv() first rather than always using pinv() because the two are
    # not interchangeable: pinv() truncates singular values below its
    # rcond threshold, so on a merely ill-conditioned (but invertible)
    # fit_dot it silently returns a different, regularised answer instead
    # of the exact one inv() gives -- and it does so ~6-12x slower (SVD
    # vs. LU), which matters here since this runs per-pixel over a whole
    # navigation map when lazy. Only fall back to pinv()'s approximation
    # when the matrix is actually singular and inv() has no answer at all.
    def _safe_inv(matrix):
        try:
            return np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(matrix)

    # Always go through map_blocks (rather than a direct dask.array.linalg.inv)
    # when lazy: dask's own inv() uses a QR-based solve that is more prone to
    # hitting exact singularities than numpy's LU-based inv()/pinv(), and
    # map_blocks lets the LinAlgError fallback above run per-chunk on
    # materialised numpy arrays. This also sidesteps dask.array.linalg.inv's
    # lack of support for the batched (3-D) case.
    if lazy:
        import dask.array as da

        inv_fit_dot = da.map_blocks(
            _safe_inv, fit_dot, chunks=fit_dot.chunks, dtype=float, meta=fit_dot
        )
    else:
        inv_fit_dot = _safe_inv(fit_dot)

    n = fit.shape[-2]  # the signal axis length
    k = coefficients.shape[-1]  # the number of components
    covariance = (1 / (n - k)) * (residual * inv_fit_dot.T).T
    return covariance


def _calculate_parameter_uncertainty_from_fisher_information(fisher_information_matrix):
    """
    Calculate parameter uncertainties from Fisher Information Matrix.

    For maximum likelihood estimation, parameter uncertainties are given by
    the Cramér-Rao bound: Var(θ) ≥ [I(θ)]^(-1), where I(θ) is the Fisher
    Information Matrix (the Hessian of the negative log-likelihood).

    Parameters
    ----------
    fisher_information_matrix : ndarray
        The Fisher Information Matrix (Hessian of negative log-likelihood)

    Returns
    -------
    uncertainties : ndarray
        Parameter standard deviations (square root of diagonal of covariance matrix)
    covariance : ndarray
        Full covariance matrix (inverse of Fisher Information Matrix)
    """
    try:
        # Calculate covariance matrix as inverse of Fisher Information Matrix
        covariance = np.linalg.inv(fisher_information_matrix)

        # Parameter uncertainties are square root of diagonal elements
        uncertainties = np.sqrt(np.diag(covariance))

        # Check for invalid results
        if (
            np.any(np.isnan(uncertainties))
            or np.any(np.isinf(uncertainties))
            or np.any(uncertainties < 0)
        ):
            raise np.linalg.LinAlgError("Invalid uncertainties computed")

        return uncertainties, covariance

    except np.linalg.LinAlgError:
        # Handle singular matrix case - use pseudo-inverse
        try:
            covariance = np.linalg.pinv(fisher_information_matrix)
            uncertainties = np.sqrt(np.diag(covariance))

            # Check if pseudo-inverse gives reasonable results
            if (
                np.any(np.isnan(uncertainties))
                or np.any(np.isinf(uncertainties))
                or np.any(uncertainties < 0)
            ):
                # If pseudo-inverse also fails, return NaN
                uncertainties = np.full(fisher_information_matrix.shape[0], np.nan)
                covariance = np.full_like(fisher_information_matrix, np.nan)

            return uncertainties, covariance

        except Exception:
            # If all else fails, return NaN
            uncertainties = np.full(fisher_information_matrix.shape[0], np.nan)
            covariance = np.full_like(fisher_information_matrix, np.nan)
            return uncertainties, covariance


class ModelStatistics:
    """
    Display-class for showing mean, std, min, max of each parameter
    in each model component in a clean text or HTML table.

    Parameters
    ----------
    model : hyperspy model instance
    thresholds : dict, optional
        Same structure as in print_model_statistics().
    """

    def __init__(self, model, thresholds=None, component_list=None):
        self.model = model
        self.thresholds = thresholds
        self.component_list = model if component_list is None else component_list
        self.stats = self._compute_statistics()

    def _compute_statistics(self):
        """Compute statistics exactly like print_model_statistics(),
        but return them as a nested dictionary for display."""
        collected_values = []
        for i, comp in enumerate(self.component_list):
            comp_name = f"{i} - {comp.name}"
            for param in comp.parameters:
                if hasattr(param, "map") and param.map is not None:
                    arr = np.array(param.map)
                    values = np.array([float(arr[j][0]) for j in range(len(arr))])
                    collected_values.append(
                        {
                            "component": comp_name,
                            "parameter": param.name,
                            "values": values,
                        }
                    )

        # Apply thresholds if given
        if self.thresholds is not None:
            for entry in collected_values:
                th = self.thresholds.get(entry["parameter"], {"min": None, "max": None})
                values = np.array(entry["values"], dtype=float)
                if th.get("min") is not None:
                    if not isinstance(th.get("min"), (float, int)):
                        th["min"] = np.nanpercentile(
                            values, _parse_percentile_value(th.get("min"), "min")
                        )
                    values = values[values >= th["min"]]
                if th.get("max") is not None:
                    if not isinstance(th.get("max"), (float, int)):
                        th["max"] = np.nanpercentile(
                            values, _parse_percentile_value(th.get("max"), "max")
                        )
                    values = values[values <= th["max"]]
                entry["values"] = values

        # Aggregate by component and parameter
        aggregated = defaultdict(lambda: defaultdict(list))

        for entry in collected_values:
            comp_type = entry["component"].split(" - ")[1]
            values = np.array(entry["values"], dtype=float)

            if len(values) > 0:
                aggregated[comp_type][entry["parameter"]].extend(values.tolist())

        statistics = defaultdict(lambda: defaultdict(dict))

        for comp_type, params in aggregated.items():
            for pname, values in params.items():
                arr = np.array(values, dtype=float)
                if len(arr) > 0:
                    statistics[comp_type][pname] = {
                        "mean": np.mean(arr),
                        "std": np.std(arr),
                        "min": np.min(arr),
                        "max": np.max(arr),
                    }
        return statistics

    # --- Table Output ---
    def _build_table(self, params):
        """Build and return a PrettyTable for a component type's statistics."""

        def _num_fmt(f, v):
            return "%.3e" % v

        table = PrettyTable()
        table.field_names = ["Parameter", "Mean", "Std", "Min", "Max"]
        table.align["Parameter"] = "l"
        table.align["Mean"] = "r"
        table.align["Std"] = "r"
        table.align["Min"] = "r"
        table.align["Max"] = "r"
        table.custom_format = {
            "Mean": _num_fmt,
            "Std": _num_fmt,
            "Min": _num_fmt,
            "Max": _num_fmt,
        }
        _widths = {
            "Parameter": 14,
            "Mean": 12,
            "Std": 12,
            "Min": 12,
            "Max": 12,
        }
        table.min_width = _widths
        table.max_width = _widths

        for pname, stats in params.items():
            table.add_row(
                [
                    pname,
                    stats["mean"],
                    stats["std"],
                    stats["min"],
                    stats["max"],
                ]
            )
        return table

    def __repr__(self):
        text = ""
        for comp_type, params in self.stats.items():
            text += f"{comp_type}:\n"
            table = self._build_table(params)
            text += str(table) + "\n\n"
        return text

    def _repr_html_(self):
        html = ""
        for comp_type, params in self.stats.items():
            html += f"<h4>{comp_type}</h4>"
            table = self._build_table(params)
            html += table.get_html_string(
                attributes={
                    "style": "width:100%; border-collapse:collapse; text-align:center;",
                    "border": "1",
                }
            )
            html += "<br>"
        return html


def _intervals_to_tuples(intervals):
    """Normalize and validate the ``intervals`` parameter for
    :meth:`~.api.model.components.Component.estimate_parameters`.

    Parameters
    ----------
    intervals
        One of:
        - A bare ``(left, right)`` tuple (auto-wrapped to a single-interval
          list).
        - A tuple or list of ``(left, right)`` tuples.
        - A tuple or list of :class:`~.api.roi.SpanROI` objects.

    Returns
    -------
    list of tuple
        ``[(left_0, right_0), (left_1, right_1), ...]``

    Raises
    ------
    ValueError
        If ``intervals`` is not a list or tuple, or if any element has an
        invalid format.
    """
    if not isinstance(intervals, (list, tuple)):
        raise ValueError("`intervals` must be a list of tuples or SpanROI objects.")
    if isinstance(intervals, tuple):
        if len(intervals) == 0:
            intervals = []
        else:
            first = intervals[0]
            if (
                isinstance(first, (tuple, list))
                and len(first) == 2
                and not isinstance(first[0], (tuple, list))
            ):
                # tuple of intervals — use as-is
                pass
            elif hasattr(first, "left") and hasattr(first, "right"):
                # tuple of SpanROIs — use as-is
                pass
            else:
                # bare (left, right) pair — wrap in list
                intervals = [intervals]
    interval_tuples = []
    for interval in intervals:
        if hasattr(interval, "left") and hasattr(interval, "right"):
            interval_tuples.append((interval.left, interval.right))
        elif isinstance(interval, (tuple, list)) and len(interval) == 2:
            interval_tuples.append(tuple(interval))
        else:
            raise ValueError(
                f"Invalid interval format: {interval}. "
                "Expected tuple (left, right) or SpanROI object."
            )
    return interval_tuples


class SummaryStatistics:
    """
    Display class for the five-number summary statistics of a signal.

    Parameters
    ----------
    mean, std, min, q1, median, q3, max : float
        The statistics to display.
    formatter : str, optional
        Printf-style format string for numeric values. Default is ``"%.3g"``.
    """

    def __init__(self, mean, std, min, q1, median, q3, max, formatter="%.3g"):
        self.stats = [
            ("mean", mean),
            ("std", std),
            ("min", min),
            ("Q1", q1),
            ("median", median),
            ("Q3", q3),
            ("max", max),
        ]
        self.formatter = formatter

    def _build_table(self):
        table = PrettyTable()
        table.field_names = ["Statistic", "Value"]
        table.align["Statistic"] = "r"
        table.align["Value"] = "r"
        _fmt = self.formatter
        table.custom_format = {
            "Value": lambda f, v: "" if v is None else _fmt % v,
        }
        table.min_width = {"Statistic": 12, "Value": 10}
        table.max_width = {"Statistic": 12, "Value": 10}
        for name, val in self.stats:
            table.add_row([name, val])
        return table

    def __repr__(self):
        return "Summary statistics\n" + str(self._build_table())

    def _repr_html_(self):
        return "<h4>Summary statistics</h4>" + self._build_table().get_html_string(
            attributes={
                "style": "width:100%; border-collapse:collapse; text-align:center;",
                "border": "1",
            }
        )
