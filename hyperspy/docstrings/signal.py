# -*- coding: utf-8 -*-
"""Common docstring snippets for signal.

"""

ONE_AXIS_PARAMETER = \
    """: {int | string | axis}
            The axis can be passed directly, or specified using the index of
            the axis in `axes_manager` or the axis name."""

MANY_AXIS_PARAMETER = \
    """: {int | string | axis | tuple | None}
            Either one on its own, or many axes in a tuple can be passed. In
            both cases the axes can be passed directly, or specified using the 
            index in `axes_manager` or the name of the axis.  Any duplicates 
            are removed. If string, it can also be `signal` or `navigation` 
            and performed over all signal or navigation axes. If None, the 
            operation is performed over all navigation axes (default)."""

OUT_ARG = \
    """out : {Signal, None}
            If None, a new Signal is created with the result of the operation
            and returned (default). If a Signal is passed, it is used to
            receive the output of the operation, and nothing is returned."""

NAN_FUNC = \
    """Identical to {0} except ignores missing (NaN) values.
       The full documentation follows:

       -------------------- {0} --------------------
        {1} """

ROI_ARG = \
    """roi : {None | True | roi | list}
            If None, the operation is performed over the whole dataset.
            If True, a roi is added to the plot.
            If a roi or a list of is passed, the operation is performed only 
            over the roi(s). The dimension of the roi should be compatible 
            with the axis argument (if provided)."""