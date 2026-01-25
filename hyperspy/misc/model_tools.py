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

        # Add rows
        for para in self.parameters:
            if not self.only_free or self.only_free and para.free:
                free = para.free if para.twin is None else "Twinned"
                ln = para._linear
                table.add_row(
                    [
                        _format_string(para.name, max_length=14),
                        _format_string(str(free), max_length=7),
                        _format_string(para.value, max_length=10),
                        _format_string(para.std, max_length=10),
                        _format_string(para.bmin, max_length=10),
                        _format_string(para.bmax, max_length=10),
                        _format_string(str(ln), max_length=6),
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

    # Prefer to find another way than matrix inverse
    # if target_signal shape is 1D, then fit_dot is 2D and numpy going to dask.linalg.inv is fine.
    # If target_signal shape is 2D, then dask.linalg.inv will fail because fit_dot is 3D.
    if lazy and target_signal.ndim > 1:
        import dask.array as da

        inv_fit_dot = da.map_blocks(
            np.linalg.inv, fit_dot, chunks=fit_dot.chunks, dtype=float, meta=fit_dot
        )
    else:
        inv_fit_dot = np.linalg.inv(fit_dot)

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
        table = PrettyTable()
        table.field_names = ["Parameter", "Mean", "Std", "Min", "Max"]
        table.align["Parameter"] = "l"
        table.align["Mean"] = "r"
        table.align["Std"] = "r"
        table.align["Min"] = "r"
        table.align["Max"] = "r"

        for pname, stats in params.items():
            table.add_row(
                [
                    _format_string(pname, max_length=14),
                    _format_string(stats["mean"], format_string=".3e", max_length=12),
                    _format_string(stats["std"], format_string=".3e", max_length=12),
                    _format_string(stats["min"], format_string=".3e", max_length=12),
                    _format_string(stats["max"], format_string=".3e", max_length=12),
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
