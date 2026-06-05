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

import logging

import numpy as np

from hyperspy._components.expression import Expression
from hyperspy.misc.utils import ordinal

_logger = logging.getLogger(__name__)


class Polynomial(Expression):
    """n-order polynomial component.

    Polynomial component consisting of order + 1 parameters.
    The parameters are named "a" followed by the corresponding order,
    i.e.

    .. math::

        f(x) = a_{2} x^{2} + a_{1} x^{1} + a_{0}

    Zero padding is used for polynomial of order > 10.

    Parameters
    ----------
    order : int
        Order of the polynomial, must be different from 0.
    **kwargs
        Keyword arguments can be used to initialise the value of the
        parameters, i.e. a2=2, a1=3, a0=1. Extra keyword arguments are passed
        to the :class:`~.api.model.components1D.Expression` component.

    """

    def __init__(self, order=2, module=None, **kwargs):
        if order == 0:
            raise ValueError("Polynomial of order 0 is not supported.")
        coeff_list = [
            "{}".format(o).zfill(len(list(str(order)))) for o in range(order, -1, -1)
        ]
        expr = "+".join(
            ["a{}*x**{}".format(c, o) for c, o in zip(coeff_list, range(order, -1, -1))]
        )
        name = "{} order Polynomial".format(ordinal(order))
        super().__init__(
            expression=expr, name=name, module=module, autodoc=False, **kwargs
        )
        # Need to save order to be able to reload component after being saved
        self._whitelist["order"] = ("init", order)

    def get_polynomial_order(self):
        return len(self.parameters) - 1

    def estimate_parameters(
        self,
        signal,
        x1=None,
        x2=None,
        intervals=None,
        only_current=False,
    ):
        """Estimate the parameters by polynomial fitting.

        Parameters
        ----------
        signal : :class:`~.api.signals.Signal1D`
        x1 : float, optional
            Defines the left limit of the spectral range to use for the
            estimation. Deprecated, use ``intervals`` instead.
        x2 : float, optional
            Defines the right limit of the spectral range to use for the
            estimation. Deprecated, use ``intervals`` instead.
        intervals : list of tuples or :class:`~.api.roi.SpanROI`, optional
            List of intervals for estimation. Each interval can be a tuple
            ``(left, right)`` or a :class:`~.api.roi.SpanROI` instance.
            Data from all intervals is concatenated for the fit.
            If ``None``, the ``x1``, ``x2`` arguments are used for backward
            compatibility.
        only_current : bool
            If False estimates the parameters for the full dataset.

        Returns
        -------
        bool

        """
        super()._estimate_parameters(signal)
        axis = signal.axes_manager.signal_axes[0]

        # Backward compat: positional callers pass only_current as 4th arg
        if isinstance(intervals, bool):
            only_current = intervals
            intervals = None

        if intervals is not None:
            if hasattr(intervals, "left") and hasattr(intervals, "right"):
                intervals = [intervals]
            if isinstance(intervals, tuple) and len(intervals) == 2:
                intervals = [intervals]
            if not isinstance(intervals, (list, tuple)):
                raise ValueError(
                    "`intervals` must be a list of tuples or SpanROI objects."
                )
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
            indices = [axis.value_range_to_indices(a, b) for a, b in interval_tuples]
        elif x1 is not None:
            if x2 is None:
                raise ValueError("x2 must be provided when using x1.")
            i1, i2 = axis.value_range_to_indices(x1, x2)
            indices = [(i1, i2)]
        else:
            indices = [(axis.low_index, axis.high_index + 1)]

        if axis.is_binned:
            scaling_factor = (
                axis.scale
                if axis.is_uniform
                else np.mean(np.gradient(axis.axis), axis=-1)
            )

        def _get_concatenated_data(sig, indices_list):
            x_parts = []
            y_parts = []
            for idx_start, idx_end in indices_list:
                x_parts.append(axis.axis[idx_start:idx_end])
                if sig._lazy:
                    y_parts.append(sig.isig[idx_start:idx_end].data)
                else:
                    y_parts.append(
                        sig._get_current_data()[idx_start:idx_end]
                        if only_current
                        else sig.data[..., idx_start:idx_end]
                    )
            return np.concatenate(x_parts), np.concatenate(y_parts, axis=-1)

        if only_current is True:
            s = signal
            x_data, y_data = _get_concatenated_data(s, indices)
            estimation = np.polyfit(x_data, y_data, self.get_polynomial_order())
            if axis.is_binned:
                for para, estim in zip(self.parameters[::-1], estimation):
                    para.value = estim / scaling_factor
            else:
                for para, estim in zip(self.parameters[::-1], estimation):
                    para.value = estim
            return True
        else:
            if self.parameters[0].map is None:
                self._create_arrays()

            nav_shape = signal.axes_manager._navigation_shape_in_array
            with signal.unfolded():
                data = signal.data
                if axis.index_in_array > 0:
                    data = data.T
                x_parts = []
                y_parts = []
                for idx_start, idx_end in indices:
                    x_parts.append(axis.axis[idx_start:idx_end])
                    y_parts.append(data[idx_start:idx_end, ...])
                x_data = np.concatenate(x_parts)
                y_data = np.concatenate(y_parts, axis=0)
                fit = np.polyfit(x_data, y_data, self.get_polynomial_order())
                if axis.index_in_array > 0:
                    fit = fit.T
                cmap_shape = nav_shape + (self.get_polynomial_order() + 1,)
                fit = fit.reshape(cmap_shape)

                if axis.is_binned:
                    for i, para in enumerate(self.parameters[::-1]):
                        para.map["values"][:] = fit[..., i] / scaling_factor
                        para.map["is_set"][:] = True
                else:
                    for i, para in enumerate(self.parameters[::-1]):
                        para.map["values"][:] = fit[..., i]
                        para.map["is_set"][:] = True
            self.fetch_stored_values()
            return True


def convert_to_polynomial(poly_dict):
    """
    Convert the dictionary from the old to the new polynomial definition
    """
    _logger.info("Converting the polynomial to the new definition.")
    coeff_list = [
        "{}".format(o).zfill(len(list(str(poly_dict["order"]))))
        for o in range(poly_dict["order"], -1, -1)
    ]
    poly2_dict = dict(poly_dict)
    coefficient_dict = poly_dict["parameters"][0]
    poly2_dict["parameters"] = []
    for i, coeff in enumerate(coeff_list):
        param_dict = dict(coefficient_dict)
        param_dict["_id_name"] = "a{}".format(coeff)
        for v in ["value", "_bounds"]:
            param_dict[v] = coefficient_dict[v][i]
        poly2_dict["parameters"].append(param_dict)

    return poly2_dict
