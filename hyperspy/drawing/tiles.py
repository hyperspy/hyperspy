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


from __future__ import division

import importlib.util
import logging

import numpy as np

from hyperspy.drawing.backends import get_backend
from hyperspy.drawing.backends._protocol import BackendCapabilityError
from hyperspy.drawing.figure import BlittedFigure

_logger = logging.getLogger(__name__)


class HistogramTilePlot(BlittedFigure):
    """SAMFire histogram debug plot.

    This class uses matplotlib bar/patch primitives directly and therefore
    only works with the matplotlib backend.  Calling :meth:`plot` or
    :meth:`update` while a different backend is active raises
    :class:`~hyperspy.drawing.backends._protocol.BackendCapabilityError`.
    """

    def __init__(self):
        super().__init__()  # initialises events, ax_markers, _background, etc.

    def _require_mpl(self):
        backend = get_backend()
        if importlib.util.find_spec("matplotlib") is None:
            raise BackendCapabilityError("HistogramTilePlot requires matplotlib.")
        backend_name = type(backend).__name__
        if "Mpl" not in backend_name and "matplotlib" not in backend_name.lower():
            raise BackendCapabilityError(
                f"HistogramTilePlot uses matplotlib bar/patch primitives and "
                f"cannot run under the '{backend_name}' backend."
            )

    def create_axis(self, ncols=1, nrows=1, number=1, title=""):
        ax = self.figure.add_subplot(ncols, nrows, number)
        ax.set_title(title)
        ax.hspy_fig = self
        return ax

    def plot(self, db, **kwargs):
        self._require_mpl()
        if self.figure is None:
            self.create_figure()
        if len(db):
            self.update(db, **kwargs)

    def update(self, db, **kwargs):
        self._require_mpl()
        ncomps = len(db)
        i = -1
        for c_n, v in db.items():
            i += 1
            ncols = len(v)
            istart = ncols * i
            j = 0
            for p_n, (hist, bin_edges) in v.items():
                j += 1
                mask = hist > 0
                if np.any(mask):
                    title = c_n + " " + p_n
                    ax = self.create_axis(ncomps, ncols, istart + j, title)
                    self.ax = ax
                    while ax.patches:
                        ax.patches[0].remove()
                    ax.bar(
                        bin_edges[:-1][mask],
                        hist[mask],
                        np.diff(bin_edges)[mask],
                        **kwargs,
                    )
                    width = bin_edges[-1] - bin_edges[0]
                    ax.set_xlim(bin_edges[0] - width * 0.1, bin_edges[-1] + width * 0.1)
                    ax.set_ylim(0, np.max(hist) * 1.1)
        self.render_figure()  # routes through backend (draw_idle or blit)
