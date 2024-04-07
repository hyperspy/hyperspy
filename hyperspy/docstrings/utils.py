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

"""Common docstring snippets for utils."""

STACK_METADATA_ARG = """stack_metadata : {bool, int}
        If integer, this value defines the index of the signal in the signal
        list, from which the ``metadata`` and ``original_metadata`` are taken.
        If ``True``, the ``original_metadata`` and ``metadata`` of each signals
        are stacked and saved in ``original_metadata.stack_elements`` of the
        returned signal. In this case, the ``metadata`` are copied from the
        first signal in the list.
        If False, the ``metadata`` and ``original_metadata`` are not copied."""


REBIN_ARGS = """new_shape : list (of float or int) or None
            For each dimension specify the new_shape. This will internally be
            converted into a ``scale`` parameter.
        scale : list (of float or int) or None
            For each dimension, specify the new:old pixel ratio, e.g. a ratio
            of 1 is no binning and a ratio of 2 means that each pixel in the new
            spectrum is twice the size of the pixels in the old spectrum.
            The length of the list should match the dimension of the
            Signal's underlying data array.
            *Note : Only one of ``scale`` or ``new_shape`` should be specified,
            otherwise the function will not run*
        crop : bool
            Whether or not to crop the resulting rebinned data (default is
            ``True``). When binning by a non-integer number of
            pixels it is likely that the final row in each dimension will
            contain fewer than the full quota to fill one pixel. For example,
            a 5*5 array binned by 2.1 will produce two rows containing
            2.1 pixels and one row containing only 0.8 pixels. Selection of
            ``crop=True`` or ``crop=False`` determines whether or not this
            `"black"` line is cropped from the final binned array or not.
            *Please note that if ``crop=False`` is used, the final row in each
            dimension may appear black if a fractional number of pixels are left
            over. It can be removed but has been left to preserve total counts
            before and after binning.*
        dtype : {None, numpy.dtype, "same"}
            Specify the dtype of the output. If None, the dtype will be
            determined by the behaviour of :func:`numpy.sum`, if ``"same"``,
            the dtype will be kept the same. Default is None."""
