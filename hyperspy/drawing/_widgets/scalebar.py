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

from hyperspy.drawing.backends import get_backend
from hyperspy.misc.math_tools import closest_nice_number


class ScaleBar(object):
    def __init__(
        self,
        ax,
        units,
        pixel_size=None,
        color="white",
        position=None,
        max_size_ratio=0.25,
        lw=2,
        length=None,
        animated=False,
    ):
        """Add a scale bar to an image.

        Parameters
        ----------
        ax : matplotlib axes
            The axes where to draw the scale bar.
        units : str
        pixel_size : {None, float}
            If None the axes of the image are supposed to be calibrated.
            Otherwise the pixel size must be specified.
        color : a valid matplotlib color
        position {None, (float, float)}
            If None the position is automatically determined.
        max_size_ratio : float
            The maximum size of the scale bar in respect to the
            length of the x axis
        lw : int
            The line width
        length : {None, float}
            If None the length is automatically calculated using the
            max_size_ratio.

        """

        self.animated = animated
        self.ax = ax
        self.units = units
        self.pixel_size = pixel_size
        backend = get_backend()
        self.xmin, self.xmax = backend.get_xlim(ax)
        self.ymin, self.ymax = backend.get_ylim(ax)
        self.text = None
        self.line = None
        self.tex_bold = False
        if length is None:
            self.calculate_size(max_size_ratio=max_size_ratio)
        else:
            self.length = length
        if position is None:
            self.position = self.calculate_line_position()
        else:
            self.position = position
        self.calculate_text_position()
        self.plot_scale(line_width=lw)
        self.set_color(color)

    def get_units_string(self):
        if self.tex_bold is True:
            if (self.units[0] and self.units[-1]) == "$":
                return r"$\mathbf{%g\,%s}$" % (self.length, self.units[1:-1])
            else:
                return r"$\mathbf{%g\,}$\textbf{%s}" % (self.length, self.units)
        else:
            return r"$%g\,$%s" % (self.length, self.units)

    def calculate_line_position(self, pad=0.05):
        return (
            (1 - pad) * self.xmin + pad * self.xmax,
            (1 - pad) * self.ymin + pad * self.ymax,
        )

    def calculate_text_position(self, pad=1 / 100.0):
        ps = self.pixel_size if self.pixel_size is not None else 1
        x1, y1 = self.position
        x2, y2 = x1 + self.length / ps, y1

        self.text_position = ((x1 + x2) / 2.0, y2 + (self.ymax - self.ymin) / ps * pad)

    def calculate_size(self, max_size_ratio=0.25):
        ps = self.pixel_size if self.pixel_size is not None else 1
        size = closest_nice_number(
            np.abs(ps * (self.xmax - self.xmin) * max_size_ratio)
        )
        self.length = size

    def remove(self):
        backend = get_backend()
        if self.line is not None:
            backend.remove_line(self.ax, self.line)
        if self.text is not None:
            backend.remove_text(self.ax, self.text)

    def plot_scale(self, line_width=1):
        self.remove()
        backend = get_backend()
        ps = self.pixel_size if self.pixel_size is not None else 1
        x1, y1 = self.position
        x2, y2 = x1 + self.length / ps, y1
        self.line = backend.plot_line(
            self.ax,
            [x1, x2],
            [y1, y2],
            linestyle="-",
            lw=line_width,
            animated=self.animated,
        )
        self.text = backend.add_text(
            self.ax,
            *self.text_position,
            s=self.get_units_string(),
            ha="center",
            size="medium",
            transform="data",
            animated=self.animated,
        )
        backend.set_xlim(self.ax, self.xmin, self.xmax)
        backend.set_ylim(self.ax, self.ymin, self.ymax)
        backend.draw_idle(backend.get_figure_from_ax(self.ax))

    def _set_position(self, x, y):
        self.position = x, y
        self.calculate_text_position()
        self.plot_scale(line_width=self.line.get_linewidth())

    def set_color(self, c):
        backend = get_backend()
        backend.set_line_props(self.line, color=c)
        self.text.set_color(c)
        backend.draw_idle(backend.get_figure_from_ax(self.ax))

    def set_length(self, length):
        color = self.line.get_color()
        self.length = length
        self.calculate_scale_size()
        self.calculate_text_position()
        self.plot_scale(line_width=self.line.get_linewidth())
        self.set_color(color)

    def set_tex_bold(self):
        self.tex_bold = True
        self.text.set_text(self.get_units_string())
        backend = get_backend()
        backend.draw_idle(backend.get_figure_from_ax(self.ax))
