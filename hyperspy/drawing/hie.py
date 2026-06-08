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

"""Backend-agnostic HyperImage_Explorer base."""

from abc import abstractmethod

from hyperspy.drawing.he import HyperExplorer


class HyperImage_Explorer(HyperExplorer):
    """Base for 2-D image explorers.  Backend subclasses implement
    _make_image_figure()."""

    def plot_signal(self, **kwargs):
        super().plot_signal()
        imf = self._make_image_figure(**kwargs)
        self.signal_plot = imf
        self._connect_key_nav(imf)
        if self.navigator_plot is not None:
            self._connect_key_nav(self.navigator_plot)

    @abstractmethod
    def _make_image_figure(self, **kwargs):
        raise NotImplementedError
