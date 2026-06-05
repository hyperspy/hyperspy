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

_logger = logging.getLogger(__name__)


class PowerLaw(Expression):
    r"""Power law component.

    .. math::

        f(x) = A\cdot(x-x_0)^{-r}

    ============= =============
     Variable      Parameter
    ============= =============
     :math:`A`     A
     :math:`r`     r
     :math:`x_0`   origin
    ============= =============


    Parameters
    ----------
    A : float
        Height parameter.
    r : float
        Power law coefficient.
    origin : float
        Location parameter.
    **kwargs
        Extra keyword arguments are passed to the
        :class:`~.api.model.components1D.Expression` component.

    Attributes
    ----------
    left_cutoff : float
        For x <= left_cutoff, the function returns 0. Default value is 0.0.
    """

    def __init__(
        self,
        A=10e5,
        r=3.0,
        origin=0.0,
        left_cutoff=0.0,
        module=None,
        compute_gradients=False,
        **kwargs,
    ):
        super().__init__(
            expression="where(left_cutoff<x, A*(-origin + x)**-r, 0)",
            name="PowerLaw",
            A=A,
            r=r,
            origin=origin,
            left_cutoff=left_cutoff,
            position="origin",
            module=module,
            autodoc=False,
            compute_gradients=compute_gradients,
            linear_parameter_list=["A"],
            check_parameter_linearity=False,
            **kwargs,
        )

        self.origin.free = False
        self.left_cutoff.free = False

        # Boundaries
        self.A.bmin = 0.0
        self.A.bmax = None
        self.r.bmin = 1.0
        self.r.bmax = 5.0

        self.isbackground = True
        self.convolved = False

    def estimate_parameters(
        self,
        signal,
        x1=None,
        x2=None,
        only_current=False,
        out=False,
        intervals=None,
    ):
        """Estimate the parameters for the power law component

        The two area method is used to estimate the parameters.

        Parameters
        ----------
        signal : :class:`~.api.signals.Signal1D`
        x1 : float, optional
            The left endpoint of the signal interval.
        x2 : float, optional
            The right endpoint of the signal interval.
        only_current : bool
            If False, estimates the parameters for the full dataset.
        out : bool
            If True, returns the result arrays directly without storing in the
            parameter maps/values. The returned order is (A, r).
        intervals : list of tuple or :class:`~.api.roi.SpanROI`, optional
            List of intervals for estimation. Each interval can be a tuple
            ``(left, right)`` or a :class:`~.api.roi.SpanROI` instance.
            The two-area method requires exactly 2 intervals. If a single
            interval is provided, it will be split in two for the estimation.
            If ``None``, the ``x1``, ``x2`` arguments are used for backward
            compatibility.

        Returns
        -------
        bool
            Exit status required for the :meth:`~.api.signals.Signal1D.remove_background` function.

        """
        super()._estimate_parameters(signal)
        axis = signal.axes_manager.signal_axes[0]

        two_intervals = False
        if intervals is not None:
            if not isinstance(intervals, (list, tuple)):
                raise ValueError(
                    "`intervals` must be a list of tuples or SpanROI objects."
                )
            if isinstance(intervals, tuple) and len(intervals) == 2:
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
            if len(interval_tuples) == 1:
                left, right = interval_tuples[0]
                mid = (left + right) / 2
                interval_tuples = [(left, mid), (mid, right)]
            elif len(interval_tuples) != 2:
                raise ValueError(
                    "Power law estimation requires exactly 2 intervals "
                    f"for the two-area method, got {len(interval_tuples)}."
                )
            x1, x2 = interval_tuples[0]
            x3, x4 = interval_tuples[1]
            two_intervals = True
        elif x1 is not None:
            if x2 is None:
                raise ValueError("x2 must be provided when using x1.")
        else:
            raise ValueError("Either `intervals` or `x1` and `x2` must be provided.")

        if x1 is not None and x2 <= x1:
            raise ValueError("x2 must be greater than x1")
        if not two_intervals:
            i1, i4 = axis.value_range_to_indices(x1, x2)
            # Ensure that i1 and i4 are odd to split the interval in two
            if not (i4 + i1) % 2 == 0:
                i4 -= 1
            if i4 == i1:
                i4 += 2
            i3 = (i4 + i1) // 2
            i2 = i3
        else:
            if x3 < x2:
                raise ValueError("x3 must be greater than x2")
            if x4 <= x3:
                raise ValueError("x4 must be greater than x3")
            i1, i2 = axis.value_range_to_indices(x1, x2)
            i3, i4 = axis.value_range_to_indices(x3, x4)
            if i1 == i2 or i3 == i4:
                raise ValueError(
                    "The estimation intervals must contain at least 2 points"
                )
        x1, x2, x3, x4 = axis.index2value([i1, i2, i3, i4])
        if only_current is True:
            s = signal.get_current_signal()
        else:
            s = signal
        if s._lazy:
            I1 = s.isig[i1:i2].integrate1D(2j).data
            I2 = s.isig[i3:i4].integrate1D(2j).data
        else:
            from hyperspy.signal import BaseSignal

            shape = s.data.shape[:-1]
            I1_s = BaseSignal(np.empty(shape, dtype="float", like=s.data))
            I2_s = BaseSignal(np.empty(shape, dtype="float", like=s.data))
            # Use the `out` parameters to avoid doing the deepcopy
            s.isig[i1:i2].integrate1D(2j, out=I1_s)
            s.isig[i3:i4].integrate1D(2j, out=I2_s)

            I1 = I1_s.data
            I2 = I2_s.data
        with np.errstate(divide="raise"):
            try:
                r = (
                    2
                    * (np.log(I1 / I2 * (x4 - x3) / (x2 - x1)))
                    / (np.log(x4 * x3 / x2 / x1))
                )
                k = 1 - r
                A2 = k * I2 / (x4**k - x3**k)
                A1 = k * I1 / (x2**k - x1**k)
                A = (A1 * I1 + A2 * I2) / (I1 + I2)
                if s._lazy:
                    r = r.map_blocks(np.nan_to_num)
                    A = A.map_blocks(np.nan_to_num)
                else:
                    r = np.nan_to_num(r)
                    A = np.nan_to_num(A)
            except (RuntimeWarning, FloatingPointError):
                _logger.warning(
                    "Power-law parameter estimation failed "
                    'because of a "divide-by-zero" error.'
                )
                return False

        if only_current is True:
            self.r.value = r[0]
            self.A.value = A[0]
            return True

        if out:
            return A, r
        else:
            self.A.map["values"][:] = A
            self.A.map["is_set"][:] = True
            self.r.map["values"][:] = r
            self.r.map["is_set"][:] = True
            self.origin.map["is_set"] = True
            self.left_cutoff.map["is_set"] = True
            self.fetch_stored_values()
            return True

    def grad_A(self, x):
        return self.function(x) / self.A.value

    def grad_r(self, x):
        return np.where(
            x > self.left_cutoff.value,
            -self.A.value
            * np.log(x - self.origin.value)
            * (x - self.origin.value) ** (-self.r.value),
            0,
        )

    def grad_origin(self, x):
        return np.where(
            x > self.left_cutoff.value,
            self.r.value
            * (x - self.origin.value) ** (-self.r.value - 1)
            * self.A.value,
            0,
        )
