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

from hyperspy.samfire_utils.segmenters.histogram import HistogramSegmenter
from hyperspy.samfire_utils.strategy import GlobalStrategy


class HistogramStrategy(GlobalStrategy):
    def __init__(self, bins="fd"):
        super().__init__("Histogram global strategy")
        self.segmenter = HistogramSegmenter(bins)


__all__ = [
    "GlobalStrategy",
    "HistogramStrategy",
]


def __dir__():
    return sorted(__all__)
