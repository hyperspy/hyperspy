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

import importlib

# ruff: noqa: F822

__all__ = [
    "BackgroundRemoval",
    "_get_background_estimator",
    "Signal2DCalibration",
    "Signal1DCalibration",
    "ImageContrastEditor",
    "IMAGE_CONTRAST_EDITOR_HELP_IPYWIDGETS",
    "Load",
    "LineInSignal2D",
    "LineInSignal1D",
    "SimpleMessage",
    "PeaksFinder2D",
    "SpanSelectorInSignal1D",
    "Signal1DRangeSelector",
    "Smoothing",
    "SmoothingSavitzkyGolay",
    "SmoothingLowess",
    "SmoothingTV",
    "ButterworthFilter",
    "SpikesRemoval",
    "SpikesRemovalInteractive",
    "SPIKES_REMOVAL_INSTRUCTIONS",
]


# mapping following the pattern: from value import key
_import_mapping = {
    "BackgroundRemoval": "_background_removal",
    "_get_background_estimator": "_background_removal",
    "Signal2DCalibration": "_calibration",
    "Signal1DCalibration": "_calibration",
    "ImageContrastEditor": "_image_contrast_editor",
    "IMAGE_CONTRAST_EDITOR_HELP_IPYWIDGETS": "_image_contrast_editor",
    "Load": "_io",
    "LineInSignal2D": "_line",
    "LineInSignal1D": "_line",
    "SimpleMessage": "_message",
    "PeaksFinder2D": "_peaks_finder2d",
    "SpanSelectorInSignal1D": "_selector",
    "Signal1DRangeSelector": "_selector",
    "Smoothing": "_smoothing",
    "SmoothingSavitzkyGolay": "_smoothing",
    "SmoothingLowess": "_smoothing",
    "SmoothingTV": "_smoothing",
    "ButterworthFilter": "_smoothing",
    "SpikesRemoval": "_spikes_removal",
    "SpikesRemovalInteractive": "_spikes_removal",
    "SPIKES_REMOVAL_INSTRUCTIONS": "_spikes_removal",
}


def __dir__():
    return sorted(__all__)


def __getattr__(name):
    if name in __all__:
        import_path = "hyperspy.signal_tools." + _import_mapping.get(name)
        return getattr(importlib.import_module(import_path), name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
