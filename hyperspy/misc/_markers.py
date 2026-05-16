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

import numpy as np

from hyperspy.misc import utils


def dict2vector(data, keys, return_size=True, dtype=float):
    """Take some dictionary of values and create offsets based on the input keys.
    For instances like creating a horizontal or vertical line then some key is duplicated.

    Multiple keys can be passed as well. For example to define a rectangle:

    >>> dict2vector(data, keys= [[["x1", "y1"], ["x2", "y2"]]]) # doctest: +SKIP

    In this example the keys will be unpacked to create a line segment
    """
    keys = np.array(keys)
    # check to see if the array should be ragged
    unique_keys = np.unique(keys)
    is_key_iter = [
        utils.isiterable(data[key]) and not isinstance(data[key], str)
        for key in unique_keys
    ]
    if not any(is_key_iter):  # no iterable keys
        if dtype is str:
            dtype = object
        vector = np.empty(keys.shape, dtype=dtype)
        for i in np.ndindex(
            keys.shape
        ):  # iterate through keys and create resulting vector
            vector[i] = data[keys[i]]
        if dtype is object:
            vector = np.asarray(vector, dtype=str)
    else:
        iter_key = unique_keys[is_key_iter][0]
        nav_shape = data[iter_key].shape
        if not all(is_key_iter):  # only some values are iterating
            non_iterkeys = unique_keys[np.logical_not(is_key_iter)]
            for k in non_iterkeys:
                data[k] = np.full(shape=nav_shape, fill_value=data[k])
        vector = np.empty(nav_shape, dtype=object)  # Create ragged array
        for i in np.ndindex(nav_shape):
            if dtype is str:
                vect = []
                for j in np.ndindex(keys.shape):
                    vect.append(data[keys[j]][i])
                vect = np.array(vect)
            else:
                vect = np.empty(keys.shape, dtype=dtype)
                for j in np.ndindex(keys.shape):
                    vect[j] = data[keys[j]][i]
            vector[i] = vect
    if return_size:
        if not utils.isiterable(data["size"]):
            size = data["size"]
        else:
            size = np.empty(data["size"].shape, dtype=object)
            for i in np.ndindex(data["size"].shape):
                size[i] = data["size"][i]
        return vector, size
    else:
        return vector


def markers_dict_to_markers(marker_dict):
    """
    This function maps a marker dict to a Markers object. It supports parsing
    old markers API, typically for file saved with hyperspy < 2.0.
    """
    # hyperspy 1.x markers uses `marker_type`, 2.x uses name
    markers_class = marker_dict.pop("class", marker_dict.pop("marker_type", None))
    if markers_class is None:
        raise ValueError("Not a valid marker dictionary.")

    kwargs = {
        # in hyperspy >= 2.0, all data and properties are in kwargs
        **marker_dict.pop("kwargs", {}),
        # in hyperspy < 2.0, "markers properties" are saved in `marker_properties`
        **marker_dict.pop("marker_properties", {}),
    }
    # Parse old markers API: add relevant "data" to kwargs
    if "data" in marker_dict:
        if "Point" in markers_class:
            kwargs["offsets"], kwargs["sizes"] = dict2vector(
                marker_dict["data"], keys=["x1", "y1"], return_size=True
            )
            kwargs["facecolor"] = kwargs["color"]
            kwargs["units"] = "dots"
            if "size" not in kwargs:
                kwargs["size"] = 20
            kwargs["size"] = kwargs["size"] / np.pi
            markers_class = "Points"
        elif "HorizontalLineSegment" in markers_class:
            kwargs["segments"] = dict2vector(
                marker_dict["data"],
                keys=[[["x1", "y1"], ["x2", "y1"]]],
                return_size=False,
            )
            markers_class = "Lines"

        elif "HorizontalLine" in markers_class:
            kwargs["offsets"] = dict2vector(
                marker_dict["data"], keys=["y1"], return_size=False
            )
            markers_class = "HorizontalLines"

        elif "VerticalLineSegment" in markers_class:
            kwargs["segments"] = dict2vector(
                marker_dict["data"],
                keys=[[["x1", "y1"], ["x1", "y2"]]],
                return_size=False,
            )
            markers_class = "Lines"
        elif "VerticalLine" in markers_class:
            kwargs["offsets"] = dict2vector(
                marker_dict["data"], keys=["x1"], return_size=False
            )
            markers_class = "VerticalLines"
        elif "Line" in markers_class:
            kwargs["segments"] = dict2vector(
                marker_dict["data"],
                keys=[[["x1", "y1"], ["x2", "y2"]]],
                return_size=False,
            )
            markers_class = "Lines"

        elif "Arrow" in markers_class:
            # check if dx == x2 or dx == x2 - x1, etc.
            vectors = dict2vector(
                marker_dict["data"], keys=["x1", "y1", "x2", "y2"], return_size=False
            )
            if vectors.dtype == object:
                offsets = np.empty(vectors.shape, dtype=object)
                U = np.empty(vectors.shape, dtype=object)
                V = np.empty(vectors.shape, dtype=object)
                for i in np.ndindex(vectors.shape):
                    offsets[i] = np.array(
                        [
                            [vectors[i][0], vectors[i][1]],
                        ]
                    )
                    U[i] = np.array([vectors[i][0] - vectors[i][2]])
                    V[i] = np.array([vectors[i][1] - vectors[i][3]])
            else:
                offsets = np.array(
                    [
                        [vectors[0], vectors[1]],
                    ]
                )
                U = np.array([vectors[2] - vectors[0]])
                V = np.array([vectors[3] - vectors[1]])

            kwargs["offsets"] = offsets
            kwargs["U"] = U
            kwargs["V"] = V
            markers_class = "Arrows"

        elif "Rectangle" in markers_class:
            # check if dx == x2 or dx == x2 - x1, etc.
            vectors = dict2vector(
                marker_dict["data"], keys=["x1", "y1", "x2", "y2"], return_size=False
            )
            if vectors.dtype == object:
                offsets = np.empty(vectors.shape, dtype=object)
                widths = np.empty(vectors.shape, dtype=object)
                heights = np.empty(vectors.shape, dtype=object)
                for i in np.ndindex(vectors.shape):
                    offsets[i] = [
                        [
                            (vectors[i][0] + vectors[i][2]) / 2,
                            (vectors[i][1] + vectors[i][3]) / 2,
                        ],
                    ]
                    widths[i] = [
                        np.abs(vectors[i][0] - vectors[i][2]),
                    ]
                    heights[i] = [
                        np.abs(vectors[i][1] - vectors[i][3]),
                    ]
            else:
                offsets = [
                    [((vectors[0] + vectors[2]) / 2), ((vectors[1] + vectors[3]) / 2)],
                ]
                widths = [
                    np.abs(vectors[0] - vectors[2]),
                ]
                heights = [
                    np.abs(vectors[1] - vectors[3]),
                ]
            kwargs["offsets"] = offsets
            kwargs["widths"] = widths
            kwargs["heights"] = heights
            if not kwargs.pop("fill", False):
                kwargs["facecolor"] = "none"
            markers_class = "Rectangles"

        elif "Ellipse" in markers_class:
            kwargs["offsets"] = dict2vector(
                marker_dict["data"],
                keys=[
                    ["x1", "y1"],
                ],
                return_size=False,
            )
            kwargs["widths"] = dict2vector(
                marker_dict["data"], keys=["x2"], return_size=False
            )
            kwargs["heights"] = dict2vector(
                marker_dict["data"], keys=["y2"], return_size=False
            )
            if not kwargs.pop("fill", False):
                kwargs["facecolor"] = "none"
            markers_class = "Ellipses"

        elif "Text" in markers_class:
            kwargs["offsets"] = dict2vector(
                marker_dict["data"], keys=[["x1", "y1"]], return_size=False
            )
            kwargs["texts"] = dict2vector(
                marker_dict["data"], keys=["text"], return_size=False, dtype=str
            )
            kwargs["verticalalignment"] = "bottom"
            kwargs["horizontalalignment"] = "left"
            markers_class = "Texts"

        # remove "data" key:value
        del marker_dict["data"]
    if "size" in kwargs:
        kwargs["sizes"] = kwargs.pop("size")

    from hyperspy.utils import markers

    return getattr(markers, markers_class)(**marker_dict, **kwargs)
