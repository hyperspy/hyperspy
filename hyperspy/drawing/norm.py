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

"""Backend-agnostic normalisation objects for image display.

These classes describe *how* to map data values to the [0, 1] display range.
They carry no matplotlib dependency.  The active plotting backend is
responsible for converting them to its own internal norm representation
(e.g. :class:`matplotlib.colors.Normalize` for the MPL backend).

Usage::

    from hyperspy.drawing.norm import LogNorm, PowerNorm
    s.plot(norm=LogNorm())
    s.plot(norm=PowerNorm(gamma=0.5))
"""

from __future__ import annotations


class HyperNorm:
    """Base class for all HyperSpy normalisation descriptors.

    Subclasses carry the parameters needed to construct a backend norm object.
    They do **not** perform any data mapping themselves.
    """

    def __repr__(self) -> str:
        attrs = ", ".join(
            f"{k}={v!r}" for k, v in self.__dict__.items() if not k.startswith("_")
        )
        return f"{type(self).__name__}({attrs})"

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and self.__dict__ == other.__dict__


class LinearNorm(HyperNorm):
    """Linear (identity) normalisation — identical to the default behaviour.

    Parameters
    ----------
    vmin, vmax : float or None
        Explicit clip bounds.  ``None`` means "auto".
    clip : bool
        Whether to clip values outside [vmin, vmax].
    """

    def __init__(
        self,
        vmin: float | None = None,
        vmax: float | None = None,
        clip: bool = False,
    ) -> None:
        self.vmin = vmin
        self.vmax = vmax
        self.clip = clip


class LogNorm(HyperNorm):
    """Logarithmic normalisation.

    Parameters
    ----------
    vmin, vmax : float or None
        Explicit clip bounds.  ``None`` means "auto".
    clip : bool
        Whether to clip values outside [vmin, vmax].
    """

    def __init__(
        self,
        vmin: float | None = None,
        vmax: float | None = None,
        clip: bool = False,
    ) -> None:
        self.vmin = vmin
        self.vmax = vmax
        self.clip = clip


class PowerNorm(HyperNorm):
    """Power-law (gamma) normalisation.

    Parameters
    ----------
    gamma : float
        Exponent.  Values < 1 brighten dark regions; > 1 brighten bright ones.
    vmin, vmax : float or None
    clip : bool
    """

    def __init__(
        self,
        gamma: float,
        vmin: float | None = None,
        vmax: float | None = None,
        clip: bool = False,
    ) -> None:
        self.gamma = gamma
        self.vmin = vmin
        self.vmax = vmax
        self.clip = clip


class SymLogNorm(HyperNorm):
    """Symmetric logarithmic normalisation — handles data that spans zero.

    Parameters
    ----------
    linthresh : float
        The range within which the plot is linear (to avoid log(0)).
    linscale : float
        Factor that stretches the linear range relative to the log range.
    vmin, vmax : float or None
    clip : bool
    base : float
        Logarithm base.  Default is 10.
    """

    def __init__(
        self,
        linthresh: float,
        linscale: float = 1.0,
        vmin: float | None = None,
        vmax: float | None = None,
        clip: bool = False,
        base: float = 10.0,
    ) -> None:
        self.linthresh = linthresh
        self.linscale = linscale
        self.vmin = vmin
        self.vmax = vmax
        self.clip = clip
        self.base = base
