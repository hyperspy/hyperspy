"""anyplotlib HyperExplorer subclasses."""

from __future__ import annotations

from hyperspy.drawing.hie import HyperImage_Explorer
from hyperspy.drawing.hse import HyperSignal1D_Explorer


class Apl_HyperSignal1D_Explorer(HyperSignal1D_Explorer):
    """1-D signal explorer for the anyplotlib backend."""


class Apl_HyperImage_Explorer(HyperImage_Explorer):
    """2-D image explorer for the anyplotlib backend."""
