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
            index in `axes_manager` or the name of the axis. Any duplicates are
            removed. If None, the operation is performed over all navigation
            axes (default)."""

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

OPTIMIZE_ARG = \
    """optimize : bool, default is False
            If True, the location of the data in memory is optimised for the
            fastest iteration over the navigation axes. This operation can
            cause a peak of memory usage and requires considerable processing
            times for large datasets and/or low specification hardware.
            See the `transposing` section of the HyperSpy user guide
            for more information. When operating on lazy signals, if `True`
            the chunks are optimised for the new axes configuration."""

RECHUNK_ARG = \
    """rechunk: bool
           Only has effect when operating on lazy signal. If `True` (default),
           the data may be automatically rechunked before performing this operation."""
