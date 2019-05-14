# -*- coding: utf-8 -*-
"""Common docstring snippets for signal.

"""

ONE_AXIS_PARAMETER = \
    """: int, str or `axis`
            The axis can be passed directly, or specified using the index of
            the axis in `axes_manager` or the axis name."""

MANY_AXIS_PARAMETER = \
    """: int, str, `axis`, tuple of `axis` or None
            Either one on its own, or many axes in a tuple can be passed. In
            both cases the axes can be passed directly, or specified using the
            index in `axes_manager` or the name of the axis. Any duplicates are
            removed. If None, the operation is performed over all navigation
            axes (default)."""

OUT_ARG = \
    """out : `Signal` or None
            If None, a new Signal is created with the result of the operation
            and returned (default). If a Signal is passed, it is used to
            receive the output of the operation, and nothing is returned."""

NAN_FUNC = \
    """Identical to {0} except ignores missing (NaN) values.
       The full documentation follows:

       -------------------- {0} --------------------
        {1} """

OPTIMIZE_ARG = \
    """optimize : bool
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

SHOW_PROGRESSBAR_ARG = \
    """show_progressbar : None or bool
           If True, display a progress bar. If None the default from the 
           preferences settings is used."""

PARALLEL_ARG = \
    """parallel : None or bool
           If True, perform computation in parallel using multiple cores. If 
           None the default from the preferences settings is used."""

PARALLEL_INT_ARG = \
    """parallel : None, bool or int
           If True, perform computation in parallel using multiple cores. If 
           int, use as many cores as specified. If None the default from 
           the preferences settings is used."""
