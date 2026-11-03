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

"""Unit tests for hyperspy.drawing.marker_collection.HyperMarkerCollection."""

import matplotlib.collections
import pytest

from hyperspy.drawing.marker_collection import (
    ArrowsCollection,
    CirclesCollection,
    EllipsesCollection,
    HLinesCollection,
    HyperMarkerCollection,
    LinesCollection,
    PointsCollection,
    PolygonsCollection,
    RectanglesCollection,
    SquaresCollection,
    TextsCollection,
    VLinesCollection,
)

ALL_COLLECTIONS = [
    PointsCollection,
    CirclesCollection,
    SquaresCollection,
    LinesCollection,
    VLinesCollection,
    HLinesCollection,
    TextsCollection,
    RectanglesCollection,
    EllipsesCollection,
    PolygonsCollection,
    ArrowsCollection,
]


@pytest.mark.parametrize("collection_cls", ALL_COLLECTIONS)
def test_mpl_collection_returns_a_collection_subclass(collection_cls):
    mpl_cls = collection_cls.mpl_collection()
    assert isinstance(mpl_cls, type)
    assert issubclass(
        mpl_cls, (matplotlib.collections.Collection, matplotlib.quiver.Quiver)
    )


@pytest.mark.parametrize("collection_cls", ALL_COLLECTIONS)
def test_from_marker_type_roundtrip(collection_cls):
    assert (
        HyperMarkerCollection.from_marker_type(collection_cls._marker_type)
        is collection_cls
    )


def test_from_marker_type_unknown_raises():
    with pytest.raises(ValueError, match="Unknown marker type"):
        HyperMarkerCollection.from_marker_type("not-a-real-marker-type")


def test_is_hyper_collection_true_for_subclass():
    assert HyperMarkerCollection.is_hyper_collection(PointsCollection) is True


@pytest.mark.parametrize("obj", [object(), PointsCollection(), "points", None, 42])
def test_is_hyper_collection_false_for_non_subclass(obj):
    assert HyperMarkerCollection.is_hyper_collection(obj) is False


def test_position_key_to_set_defaults_to_position_key():
    assert PointsCollection._position_key_to_set == PointsCollection._position_key


def test_position_key_to_set_overridden_for_vlines_and_hlines():
    assert VLinesCollection._position_key_to_set == "segments"
    assert HLinesCollection._position_key_to_set == "segments"


def test_base_mpl_collection_raises_not_implemented():
    class _Unimplemented(HyperMarkerCollection):
        pass

    with pytest.raises(NotImplementedError, match="_Unimplemented"):
        _Unimplemented.mpl_collection()
