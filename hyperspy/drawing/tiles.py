# -*- coding: utf-8 -*-
# Copyright 2007-2011 The HyperSpy developers
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


from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from hyperspy.drawing.figure import BlittedFigure
from hyperspy.drawing import utils


class HistogramTilePlot(BlittedFigure):

    def __init__(self):
        self.figure = None
        self.title = ''
        self.ax = None

    def create_axis(self, ncols=1, nrows=1, number=1, title=''):
        ax = self.figure.add_subplot(ncols, nrows, number)
        ax.set_title(title)
        ax.hspy_fig = self
        return ax

    def plot(self, db, **kwargs):
        if self.figure is None:
            self.create_figure()
        ncomps = len(db)

        if not ncomps:
            return
        else:
            self.update(db, **kwargs)

    def update(self, db, **kwargs):
        ncomps = len(db)
        # get / set axes
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
                    title = c_n + ' ' + p_n
                    ax = self.create_axis(ncomps, ncols, istart + j, title)
                    self.ax = ax
                    # remove previous
                    while ax.patches:
                        ax.patches[0].remove()
                    # set new; only draw non-zero height bars
                    ax.bar(
                        bin_edges[
                            :-1][mask],
                        hist[mask],
                        np.diff(bin_edges)[mask],
                        # animated=True,
                        **kwargs)
                    width = bin_edges[-1] - bin_edges[0]
                    ax.set_xlim(
                        bin_edges[0] - width * 0.1, bin_edges[-1] + width * 0.1)
                    ax.set_ylim(0, np.max(hist) * 1.1)
                    # ax.set_title(c_n + ' ' + p_n)
        self.figure.canvas.draw_idle()

    def close(self):
        try:
            plt.close(self.figure)
        except BaseException:
            pass
        self.figure = None
