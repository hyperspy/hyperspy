# -*- coding: utf-8 -*-
# Copyright 2007-2020 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.

"""Common docstring snippets for signal.

"""

ONE_AXIS_PARAMETER = \
    """: :py:class:`int`, :py:class:`str`, or :py:class:`~hyperspy.axes.DataAxis`
            The axis can be passed directly, or specified using the index of
            the axis in the Signal's `axes_manager` or the axis name."""

MANY_AXIS_PARAMETER = \
    """: :py:class:`int`, :py:class:`str`, :py:class:`~hyperspy.axes.DataAxis`, tuple (of DataAxis) or :py:data:`None`
            Either one on its own, or many axes in a tuple can be passed. In
            both cases the axes can be passed directly, or specified using the
            index in `axes_manager` or the name of the axis. Any duplicates are
            removed. If ``None``, the operation is performed over all navigation
            axes (default)."""

OUT_ARG = \
    """out : :py:class:`~hyperspy.signal.BaseSignal` (or subclasses) or :py:data:`None`
            If ``None``, a new Signal is created with the result of the
            operation and returned (default). If a Signal is passed,
            it is used to receive the output of the operation, and nothing is
            returned."""

NAN_FUNC = \
    """Identical to :py:meth:`~hyperspy.signal.BaseSignal.{0}`, except ignores
    missing (NaN) values. See that method's documentation for details.
    """

OPTIMIZE_ARG = \
    """optimize : bool
            If ``True``, the location of the data in memory is optimised for the
            fastest iteration over the navigation axes. This operation can
            cause a peak of memory usage and requires considerable processing
            times for large datasets and/or low specification hardware.
            See the :ref:`signal.transpose` section of the HyperSpy user guide
            for more information. When operating on lazy signals, if ``True``,
            the chunks are optimised for the new axes configuration."""

RECHUNK_ARG = \
    """rechunk: bool
           Only has effect when operating on lazy signal. If ``True`` (default),
           the data may be automatically rechunked before performing this
           operation."""

SHOW_PROGRESSBAR_ARG = \
    """show_progressbar : None or bool
           If ``True``, display a progress bar. If ``None``, the default from
           the preferences settings is used."""

PARALLEL_ARG = \
    """parallel : None or bool
           If ``True``, perform computation in parallel using multithreading. If
           ``None``, the default from the preferences settings is used. The number
           of threads is controlled by the ``max_workers`` argument."""

MAX_WORKERS_ARG = \
    """max_workers : None or int
           Maximum number of threads used when ``parallel=True``. If None, defaults
           to ``min(32, os.cpu_count())``."""
