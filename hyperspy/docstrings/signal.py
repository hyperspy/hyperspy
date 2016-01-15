# -*- coding: utf-8 -*-
"""Common docstring snippets for signal.

"""

ONE_AXIS_PARAMETER = \
    """: {int | string | axis}
            The axis can be passed directly, or specified using the index of the
            axis in `axes_manager` or the axis name."""
MANY_AXIS_PARAMETER = \
    """: {int | string | axis | tuple | None}
            Either one on its own, or many axes in a tuple can be passed. In
            both cases the axes can be passed directly, or specified using the
            index in `axes_manager` or the name of the axis. Any duplicates are
            removed. If None, the operation is performed over all navigation
            axes (default)."""
