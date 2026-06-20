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

"""Functions for generating artificial data.

For use in things like docstrings or to test HyperSpy functionalities.

"""

import numpy as np
import scipy

from hyperspy import components1d, components2d, signals
from hyperspy.axes import UniformDataAxis
from hyperspy.decorators import jit_ifnumba
from hyperspy.misc.math_tools import check_random_state

try:
    from numba import prange
except ImportError:
    # Numba not installed
    prange = range


ADD_NOISE_DOCSTRING = """add_noise : bool
        If True, add noise to the signal. Use ``random_state`` to seed
        the noise to generate reproducible noise.
    random_state : None, int or numpy.random.Generator, default None
        Random seed used to generate the data.
    """


@jit_ifnumba(cache=True, parallel=True)
def _create_array_with_gaussian(spacing_x, spacing_y, size_x, size_y, gaussian):
    array = np.zeros((size_x, size_y), dtype=np.float32)
    for i in prange(int(size_x / spacing_x)):
        for j in prange(int(size_y / spacing_y)):
            array[
                i * spacing_x : (i + 1) * spacing_x, j * spacing_y : (j + 1) * spacing_y
            ] += gaussian

    return array * 1e3


def atomic_resolution_image(
    size=512,
    spacing=0.2,
    column_radius=0.05,
    rotation_angle=0,
    pixel_size=0.015,
    add_noise=False,
    random_state=None,
):
    """
    Make an artificial atomic resolution image. The atomic columns
    are modelled with Gaussian functions.

    Parameters
    ----------
    size : int or tuple of int, default=512
        The number of pixels of the image in horizontal and vertical
        directions. If int, the size is the same in both directions.
    spacing : float or tuple of float, default=14
        The spacing between the atomic columns in horizontal
        and vertical directions in nanometer.
    column_radius : float, default=0.05
        The radius of the atomic column, i. e. the width of the Gaussian
        in nanometer.
    rotation_angle : int or float, default=0
        The rotation of the lattice in degree.
    pixel_size : float, default=0.015
        The pixel size in nanometer.
    %s

    Returns
    -------
    :class:`~.api.signals.Signal2D`

    Examples
    --------
    >>> import hyperspy.api as hs
    >>> s = hs.data.atomic_resolution_image()
    >>> s.plot()

    """
    if isinstance(size, int):
        size = (size,) * 2
    elif not isinstance(size, tuple):
        raise ValueError("`size` must be an integer or tuple of int.")

    if isinstance(spacing, float):
        spacing = (spacing,) * 2
    elif not isinstance(spacing, tuple):
        raise ValueError("`spacing` must be an integer or tuple of int.")

    size_x, size_y = size
    spacing_x, spacing_y = tuple([int(round(v / pixel_size)) for v in spacing])

    gaussian = components2d.Gaussian2D(
        sigma_x=column_radius / pixel_size,
        sigma_y=column_radius / pixel_size,
        centre_x=spacing_x / 2,
        centre_y=spacing_y / 2,
    )

    gaussian_values = gaussian.function(*np.mgrid[:spacing_x, :spacing_y])

    array = _create_array_with_gaussian(
        spacing_x, spacing_y, size_x, size_y, gaussian_values
    )

    if add_noise:
        random_state = check_random_state(random_state)
        array += random_state.poisson(array)

    s = signals.Signal2D(array)

    if rotation_angle != 0:
        s.map(scipy.ndimage.rotate, angle=rotation_angle, reshape=False)

        w, h = s.axes_manager.signal_axes[0].size, s.axes_manager.signal_axes[1].size
        wr, hr = _get_largest_rectangle_from_rotation(w, h, rotation_angle)
        w_remove, h_remove = (w - wr), (h - hr)
        s.crop_signal(
            int(w_remove / 2),
            int(w - w_remove / 2),
            int(h_remove / 2),
            int(h - h_remove / 2),
        )

    for axis in s.axes_manager.signal_axes:
        axis.scale = pixel_size
        axis.units = "nm"

    return s


atomic_resolution_image.__doc__ %= ADD_NOISE_DOCSTRING


def luminescence_signal(
    navigation_dimension=0,
    uniform=False,
    add_baseline=False,
    add_noise=True,
    random_state=None,
):
    """
    Get an artificial luminescence signal in wavelength scale (nm, uniform) or
    energy scale (eV, non-uniform), simulating luminescence data recorded with a
    diffracting spectrometer. Some random noise is also added to the spectrum,
    to simulate experimental noise.

    Parameters
    ----------
    navigation_dimension : int
        The navigation dimension(s) of the signal. 0 = single spectrum,
        1 = linescan, 2 = spectral map etc...
    uniform : bool
        return uniform (wavelength) or non-uniform (energy) spectrum
    add_baseline : bool
        If true, adds a constant baseline to the spectrum. Conversion to
        energy representation will turn the constant baseline into inverse
        powerlaw.
    %s

    Examples
    --------
    >>> import hyperspy.api as hs
    >>> s = hs.data.luminescence_signal()
    >>> s.plot()

    With constant baseline

    >>> s = hs.data.luminescence_signal(uniform=True, add_baseline=True)
    >>> s.plot()

    To make the noise the same for multiple spectra, which can
    be useful for testing fitting routines

    >>> s1 = hs.data.luminescence_signal(random_state=10)
    >>> s2 = hs.data.luminescence_signal(random_state=10)
    >>> print((s1.data == s2.data).all())
    True

    2D map

    >>> s = hs.data.luminescence_signal(navigation_dimension=2)
    >>> s.plot()

    See Also
    --------
    atomic_resolution_image

    Returns
    -------
    :class:`~.api.signals.Signal1D`
    """

    # Initialisation of random number generator
    random_state = check_random_state(random_state)

    # Creating a uniform data axis, roughly similar to Horiba iHR320 with a 150 mm-1 grating
    nm_axis = UniformDataAxis(
        index_in_array=None,
        name="Wavelength",
        units="nm",
        navigate=False,
        size=1024,
        scale=0.54,
        offset=222.495,
        is_binned=False,
    )

    # Artificial luminescence peak
    gaussian_peak = components1d.Gaussian(A=5000, centre=375, sigma=25)

    if navigation_dimension >= 0:
        # Generate empty data (ones)
        data = np.ones([10 for i in range(navigation_dimension)] + [nm_axis.size])
        # Generate spatial axes
        spaxes = [
            UniformDataAxis(
                index_in_array=None,
                name="X{:d}".format(i),
                units="um",
                navigate=False,
                size=10,
                scale=2.1,
                offset=0,
                is_binned=False,
            )
            for i in range(navigation_dimension)
        ]
        # Generate empty signal
        sig = signals.Signal1D(data, axes=spaxes + [nm_axis])
        sig.metadata.General.title = "{:d}d-map Artificial Luminescence Signal".format(
            navigation_dimension
        )
    else:
        raise ValueError(
            "Value {:d} invalid as navigation dimension.".format(navigation_dimension)
        )

    # Populating data array, possibly with noise and baseline
    sig.data *= gaussian_peak.function(nm_axis.axis)
    if add_noise:
        sig.data += (random_state.uniform(size=sig.data.shape) - 0.5) * 1.4
    if add_baseline:
        data += 350.0

    # if not uniform, transformation into non-uniform axis
    if not uniform:
        hc = 1239.84198  # nm/eV
        # converting to non-uniform axis
        sig.axes_manager.signal_axes[0].convert_to_functional_data_axis(
            expression="a/x",
            name="Energy",
            units="eV",
            a=hc,
        )
        # Reverting the orientation of signal axis to have increasing Energy
        sig = sig.isig[::-1]
        # Jacobian transformation
        Eax = sig.axes_manager.signal_axes[0].axis
        sig *= hc / Eax**2
    return sig


luminescence_signal.__doc__ %= ADD_NOISE_DOCSTRING


def wave_image(
    angle=45, wavelength=10, shape=(256, 256), add_noise=True, random_state=None
):
    """
    Returns a wave image generated using the sinus function.

    Parameters
    ----------
    angle : float, optional
        The angle in degree.
    wavelength : float, optional
        The wavelength the wave in pixel. The default is 10
    shape : tuple of float, optional
        The shape of the data. The default is (256, 256).
    %s

    Returns
    -------
    :class:`~.api.signals.Signal2D`
    """

    x = np.arange(0, shape[0], 1)
    y = np.arange(0, shape[1], 1)
    X, Y = np.meshgrid(x, y)

    angle = np.deg2rad(angle)
    grating = np.sin(2 * np.pi * (X * np.cos(angle) + Y * np.sin(angle)) / wavelength)
    if add_noise:
        random_state = check_random_state(random_state)

        grating += random_state.random(grating.shape)

    s = signals.Signal2D(grating)
    for axis in s.axes_manager.signal_axes:
        axis.units = "nm"
        axis.scale = 0.01

    return s


wave_image.__doc__ %= ADD_NOISE_DOCSTRING


def _get_largest_rectangle_from_rotation(width, height, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    degrees), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    from: http://stackoverflow.com/a/16778797/1018861
    In hyperspy, it is centered around centre coordinate of the signal.
    """
    import math

    angle = math.radians(angle)
    if width <= 0 or height <= 0:
        return 0, 0

    width_is_longer = width >= height
    side_long, side_short = (width, height) if width_is_longer else (height, width)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2.0 * sin_a * cos_a * side_long:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (
            (width * cos_a - height * sin_a) / cos_2a,
            (height * cos_a - width * sin_a) / cos_2a,
        )

    return wr, hr
