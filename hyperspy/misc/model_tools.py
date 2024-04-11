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

import dask.array as da
import numpy as np


def _format_string(val):
    """
    Returns formatted string for a value unless it equals None, then blank
    """
    return "{:6g}".format(val) if val is not None else ""


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
        self.active = component.active
        self.parameters = component.parameters
        self._id_name = component._id_name
        self.only_free = only_free
        self.only_active = only_active

    def __repr__(self):
        # Number of digits for each label for the terminal-style view.
        size = {
            "name": 14,
            "free": 7,
            "value": 10,
            "std": 10,
            "bmin": 10,
            "bmax": 10,
            "linear": 6,
        }
        # Using nested string formatting for flexibility in future updates
        signature = "{{:>{name}}} | {{:>{free}}} | {{:>{value}}} | {{:>{std}}} | {{:>{bmin}}} | {{:>{bmax}}} | {{:>{linear}}}".format(
            **size
        )

        if self.only_active:
            text = "{0}: {1}".format(self.__class__.__name__, self.name)
        else:
            text = "{0}: {1}\nActive: {2}".format(
                self.__class__.__name__, self.name, self.active
            )
        text += "\n"
        text += signature.format(
            "Parameter Name", "Free", "Value", "Std", "Min", "Max", "Linear"
        )
        text += "\n"
        text += signature.format(
            "=" * size["name"],
            "=" * size["free"],
            "=" * size["value"],
            "=" * size["std"],
            "=" * size["bmin"],
            "=" * size["bmax"],
            "=" * size["linear"],
        )
        text += "\n"
        for para in self.parameters:
            if not self.only_free or self.only_free and para.free:
                free = para.free if para.twin is None else "Twinned"
                ln = para._linear
                text += signature.format(
                    para.name[: size["name"]],
                    str(free)[: size["free"]],
                    str(para.value)[: size["value"]],
                    str(para.std)[: size["std"]],
                    str(para.bmin)[: size["bmin"]],
                    str(para.bmax)[: size["bmax"]],
                    str(ln)[: size["linear"]],
                )
                text += "\n"
        return text

    def _repr_html_(self):
        if self.only_active:
            text = "<p><b>{0}: {1}</b></p>".format(self.__class__.__name__, self.name)
        else:
            text = "<p><b>{0}: {1}</b><br />Active: {2}</p>".format(
                self.__class__.__name__, self.name, self.active
            )

        para_head = """<table style="width:100%"><tr><th>Parameter Name</th><th>Free</th>
            <th>Value</th><th>Std</th><th>Min</th><th>Max</th><th>Linear</th></tr>"""
        text += para_head
        for para in self.parameters:
            if not self.only_free or self.only_free and para.free:
                free = para.free if para.twin is None else "Twinned"
                linear = para._linear
                value = _format_string(para.value)
                std = _format_string(para.std)
                bmin = _format_string(para.bmin)
                bmax = _format_string(para.bmax)

                text += """<tr><td>{0}</td><td>{1}</td><td>{2}</td>
                    <td>{3}</td><td>{4}</td><td>{5}</td><td>{6}</td></tr>""".format(
                    para.name, free, value, std, bmin, bmax, linear
                )
        text += "</table>"
        return text


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
        self.model_type = str(self.model.__class__).split("'")[1].split(".")[-1]

    def __repr__(self):
        text = "{}: {}\n".format(
            self.model_type, self.model.signal.metadata.General.title
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
            self.model_type, self.model.signal.metadata.General.title
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
        inv_fit_dot = da.map_blocks(np.linalg.inv, fit_dot, chunks=fit_dot.chunks)
    else:
        inv_fit_dot = np.linalg.inv(fit_dot)

    n = fit.shape[-2]  # the signal axis length
    k = coefficients.shape[-1]  # the number of components
    covariance = (1 / (n - k)) * (residual * inv_fit_dot.T).T
    return covariance
