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

"""Backend-agnostic marker collection descriptors.

Each ``HyperMarkerCollection`` subclass describes one marker geometry:

1. ``_marker_type`` — passed to ``PlottingBackend.create_markers`` for
   native rendering.
2. ``_position_key`` / ``_position_key_to_set`` — the kwarg keys holding
   position data.
3. ``mpl_collection()`` — the matplotlib Collection class used on the
   matplotlib fallback path.

Pass the *subclass* (not an instance) to ``Markers(collection=...)``.
"""

from __future__ import annotations

# Registry mapping _marker_type string → HyperMarkerCollection subclass.
# Populated automatically via __init_subclass__.
_REGISTRY: dict[str, type[HyperMarkerCollection]] = {}


class HyperMarkerCollection:
    """Base class for backend-agnostic marker collection descriptors.

    Each subclass represents one geometry type.  Pass the *subclass* (not an
    instance) as the ``collection`` argument to
    :class:`~hyperspy.drawing.markers.Markers`.

    Subclasses must define:

    ``_marker_type``
        One of the :class:`~hyperspy.drawing.backends._protocol.MarkerType`
        string constants.  Used for native backend dispatch.
    ``_position_key``
        The kwarg key that holds primary positional data (``"offsets"``,
        ``"segments"``, or ``"verts"``).  Defaults to ``"offsets"``.
    ``_position_key_to_set``
        The key used when *updating* an existing collection.  Defaults to
        ``_position_key``; differs for ``VLinesCollection`` and
        ``HLinesCollection`` which accept positions as ``"offsets"`` but
        construct the full ``"segments"`` internally.

    Subclasses must also implement :meth:`mpl_collection` to support the
    ``MplBackend`` fallback path.
    """

    _marker_type: str = ""
    _position_key: str = "offsets"
    _position_key_to_set: str | None = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Default _position_key_to_set to _position_key when not overridden.
        if cls._position_key_to_set is None:
            cls._position_key_to_set = cls._position_key
        if cls._marker_type:
            _REGISTRY[cls._marker_type] = cls

    @classmethod
    def mpl_collection(cls):
        """Return the MPL Collection class for the fallback rendering path.

        Returns
        -------
        type
            A subclass of :class:`matplotlib.collections.Collection`.
        """
        raise NotImplementedError(
            f"{cls.__name__} must implement mpl_collection() "
            "to support the MplBackend fallback path."
        )

    @staticmethod
    def is_hyper_collection(obj) -> bool:
        """Return True when *obj* is a HyperMarkerCollection subclass (not instance)."""
        return isinstance(obj, type) and issubclass(obj, HyperMarkerCollection)

    @staticmethod
    def from_marker_type(marker_type: str) -> type[HyperMarkerCollection]:
        """Return the HyperMarkerCollection subclass registered for *marker_type*.

        Parameters
        ----------
        marker_type : str
            One of the ``MarkerType.*`` string constants (e.g. ``"points"``).

        Raises
        ------
        ValueError
            When *marker_type* does not match any registered subclass.
        """
        try:
            return _REGISTRY[marker_type]
        except KeyError:
            raise ValueError(
                f"Unknown marker type '{marker_type}'. Valid types: {sorted(_REGISTRY)}"
            )


# ── Concrete collection descriptors ──────────────────────────────────────────


class PointsCollection(HyperMarkerCollection):
    """Descriptor for point markers (filled circle glyphs at offsets)."""

    _marker_type = "points"
    _position_key = "offsets"

    @classmethod
    def mpl_collection(cls):
        from hyperspy.external.matplotlib.collections import CircleCollection

        return CircleCollection


class CirclesCollection(HyperMarkerCollection):
    """Descriptor for circle markers with explicit radii (data-space sized)."""

    _marker_type = "circles"
    _position_key = "offsets"

    @classmethod
    def mpl_collection(cls):
        from hyperspy.external.matplotlib.collections import CircleCollection

        return CircleCollection


class SquaresCollection(HyperMarkerCollection):
    """Descriptor for square markers with explicit widths."""

    _marker_type = "squares"
    _position_key = "offsets"

    @classmethod
    def mpl_collection(cls):
        from hyperspy.external.matplotlib.collections import SquareCollection

        return SquareCollection


class LinesCollection(HyperMarkerCollection):
    """Descriptor for arbitrary line segment markers (segments key)."""

    _marker_type = "lines"
    _position_key = "segments"

    @classmethod
    def mpl_collection(cls):
        from matplotlib.collections import LineCollection

        return LineCollection


class VLinesCollection(HyperMarkerCollection):
    """Descriptor for vertical line markers spanning the full axes height.

    Positions are stored as x-values in ``offsets`` and converted to full
    ``[[x,0],[x,1]]`` segments when passed to the collection.
    """

    _marker_type = "vlines"
    _position_key = "offsets"
    _position_key_to_set = "segments"

    @classmethod
    def mpl_collection(cls):
        from matplotlib.collections import LineCollection

        return LineCollection


class HLinesCollection(HyperMarkerCollection):
    """Descriptor for horizontal line markers spanning the full axes width.

    Positions are stored as y-values in ``offsets`` and converted to full
    ``[[0,y],[1,y]]`` segments when passed to the collection.
    """

    _marker_type = "hlines"
    _position_key = "offsets"
    _position_key_to_set = "segments"

    @classmethod
    def mpl_collection(cls):
        from matplotlib.collections import LineCollection

        return LineCollection


class TextsCollection(HyperMarkerCollection):
    """Descriptor for text annotation markers."""

    _marker_type = "texts"
    _position_key = "offsets"

    @classmethod
    def mpl_collection(cls):
        from hyperspy.external.matplotlib.collections import TextCollection

        return TextCollection


class RectanglesCollection(HyperMarkerCollection):
    """Descriptor for rectangle markers with explicit widths and heights."""

    _marker_type = "rectangles"
    _position_key = "offsets"

    @classmethod
    def mpl_collection(cls):
        from hyperspy.external.matplotlib.collections import RectangleCollection

        return RectangleCollection


class EllipsesCollection(HyperMarkerCollection):
    """Descriptor for ellipse markers with explicit widths, heights, and angles."""

    _marker_type = "ellipses"
    _position_key = "offsets"

    @classmethod
    def mpl_collection(cls):
        from hyperspy.external.matplotlib.collections import EllipseCollection

        return EllipseCollection


class PolygonsCollection(HyperMarkerCollection):
    """Descriptor for polygon markers defined by explicit vertex lists."""

    _marker_type = "polygons"
    _position_key = "verts"

    @classmethod
    def mpl_collection(cls):
        from matplotlib.collections import PolyCollection

        return PolyCollection


class ArrowsCollection(HyperMarkerCollection):
    """Descriptor for arrow / quiver markers."""

    _marker_type = "arrows"
    _position_key = "offsets"

    @classmethod
    def mpl_collection(cls):
        from hyperspy.external.matplotlib.quiver import Quiver

        return Quiver
