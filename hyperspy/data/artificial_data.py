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

"""Functions for generating artificial data.

For use in things like docstrings or to test HyperSpy functionalities.

"""

import numpy as np

from hyperspy import components1d, components2d, signals
from hyperspy.axes import UniformDataAxis
from hyperspy.misc.math_tools import check_random_state

ADD_NOISE_DOCSTRING = """add_noise : bool
        If True, add noise to the signal. See note to seed the noise to
        generate reproducible noise.
    random_state : None, int or numpy.random.Generator, default None
        Random seed used to generate the data.
    """


def atomic_resolution_image():
    """
    Get an artificial atomic resolution image.

    Returns
    -------
    :class:`~.api.signals.Signal2D`

    Examples
    --------
    >>> s = hs.data.atomic_resolution_image()
    >>> s.plot()

    """
    sX, sY = 2, 2
    x_array, y_array = np.mgrid[0:200, 0:200]
    image = np.zeros_like(x_array, dtype=np.float32)
    gaussian2d = components2d.Gaussian2D(sigma_x=sX, sigma_y=sY)
    for x in range(10, 195, 20):
        for y in range(10, 195, 20):
            gaussian2d.centre_x.value = x
            gaussian2d.centre_y.value = y
            image += gaussian2d.function(x_array, y_array)

    s = signals.Signal2D(image)
    return s


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
    >>> (s1.data == s2.data).all()
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
