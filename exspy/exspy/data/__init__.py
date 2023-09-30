# -*- coding: utf-8 -*-
# Copyright 2007-2023 The exSpy developers
#
# This file is part of exSpy.
#
# exSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# exSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with exSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

import functools
from pathlib import Path
import warnings

import numpy as np
from scipy import interpolate

import hyperspy.api as hs
from hyperspy.misc.math_tools import check_random_state

import exspy
from exspy.misc.eels.eelsdb import eelsdb


__all__ = [
    "EDS_SEM_TM002",
    "EDS_TEM_FePt_nanoparticles",
    "eelsdb",
    "EELS_low_loss",
    "EELS_MnFe",
]


def __dir__():
    return sorted(__all__)


_ADD_NOISE_DOCSTRING = \
"""add_noise : bool
        If True, add noise to the signal. See note to seed the noise to
        generate reproducible noise.
    random_state : None or int or RandomState instance, default None
        Random seed used to generate the data.
    """


_RETURNS_DOCSTRING = \
"""Returns
    -------
    :py:class:`~hyperspy._signals.eels.EELSSpectrum`
    """


def _resolve_dir():
    """Returns the absolute path to this file's directory."""
    return Path(__file__).resolve().parent


def EDS_SEM_TM002():
    """
    Load an EDS-SEM spectrum of a EDS-TM002 standard supplied by the
    Bundesanstalt für Materialforschung und -prüfung (BAM).
    The sample consists of an approximately 6 µm thick layer containing
    the elements C, Al, Mn, Cu and Zr on a silicon substrate.

    Notes
    -----
    - Sample: EDS-TM002 provided by BAM (www.webshop.bam.de)
    - SEM Microscope: Nvision40 Carl Zeiss
    - EDS Detector: X-max 80 from Oxford Instrument
    """
    file_path = _resolve_dir().joinpath("EDS_SEM_TM002.hspy")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        # load "read-only" to ensure data access regardless of install location
        return hs.load(file_path, mode="r", reader="hspy")


def EDS_TEM_FePt_nanoparticles():
    """
    Load an EDS-TEM spectrum of FePt bimetallic nanoparticles.

    Notes
    -----
    - TEM Microscope: Tecnai Osiris 200 kV D658 AnalyticalTwin
    - EDS Detector: Super-X 4 detectors Brucker
    """
    file_path = _resolve_dir().joinpath("EDS_TEM_FePt_nanoparticles.hspy")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        return hs.load(file_path, mode="r", reader="hspy")


def EELS_low_loss(add_noise=True, random_state=None, navigation_shape=(10, )):
    """
    Get an artificial low loss electron energy loss spectrum.

    The zero loss peak is offset by 4.1 eV.

    Parameters
    ----------
    %s
    navigation_shape : tuple
        The shape of the navigation space.

    %s

    Example
    -------
    >>> s = exspy.data.low_loss(navigation_shape=(10, 10))
    >>> s.plot()

    See also
    --------
    EELS_MnFe

    """
    random_state = check_random_state(random_state)

    x = np.arange(-10, 40, 0.2)
    zero_loss = hs.model.components1D.Gaussian(A=500, centre=0, sigma=0.8)
    plasmon = hs.model.components1D.Gaussian(A=100, centre=15, sigma=20)

    if navigation_shape:
        s = hs.signals.Signal1D(np.ones(navigation_shape[::-1] + x.shape))
        plasmon._axes_manager = s.axes_manager
        plasmon._create_arrays()
        plasmon.centre.map['values'][:] = random_state.uniform(
            low=14.5, high=15.5, size=navigation_shape[::-1]
            )
        plasmon.centre.map['is_set'][:] = True
        plasmon.A.map['values'][:] = random_state.uniform(
            low=50, high=70, size=navigation_shape[::-1]
            )
        plasmon.A.map['is_set'][:] = True
        plasmon.sigma.map['values'][:] = random_state.uniform(
            low=1.8, high=2.2, size=navigation_shape[::-1]
            )
        plasmon.sigma.map['is_set'][:] = True

    data = np.broadcast_to(zero_loss.function(x), navigation_shape[::-1] + x.shape)
    data = data + plasmon.function_nd(x)

    if add_noise:
        data = data + random_state.uniform(size=len(x))

    from exspy.signals import EELSSpectrum
    s = EELSSpectrum(data)
    s.axes_manager[-1].offset = x[0]
    s.axes_manager[-1].scale = x[1] - x[0]
    s.metadata.General.title = 'Artifical low loss EEL spectrum'
    s.axes_manager[-1].name = 'Electron energy loss'
    s.axes_manager[-1].units = 'eV'
    s.set_microscope_parameters(
        beam_energy=200, convergence_angle=26, collection_angle=20)

    return s


EELS_low_loss.__doc__ %= (_ADD_NOISE_DOCSTRING, _RETURNS_DOCSTRING)


def EELS_MnFe(
        add_powerlaw=True,
        add_noise=True,
        random_state=None,
        navigation_shape=(10, )
        ):
    """
    Get an artificial core loss electron energy loss spectrum.

    Similar to a Mn-L32 edge from a perovskite oxide.

    Some random noise is also added to the spectrum, to simulate
    experimental noise.

    Parameters
    ----------
    add_powerlaw : bool
        If True, adds a powerlaw background to the spectrum.
        Default is False.
    %s
    navigation_shape : tuple
        The shape of the navigation space. Must be of length 1.

    %s

    Example
    -------
    >>> import exspy
    >>> s = exspy.data.EELS_Mn()
    >>> s.plot()

    With the powerlaw background

    >>> s = exspy.data.EELS_Mn(add_powerlaw=True)
    >>> s.plot()

    See also
    --------
    EELS_low_loss

    """
    if len(navigation_shape) > 1:
        raise ValueError("`navigation_shape` must be of length 1.")

    random_state = check_random_state(random_state)

    x = np.arange(400, 800, 0.25)
    arctan_Mn = exspy.components.EELSArctan(A=1, k=0.2, x0=640)
    arctan_Fe = exspy.components.EELSArctan(A=1, k=0.2, x0=708)
    Mn_l3 = hs.model.components1D.Gaussian(A=150, centre=640, sigma=4)
    Mn_l2 = hs.model.components1D.Gaussian(A=75, centre=655, sigma=4)
    Fe_l3 = hs.model.components1D.Gaussian(A=150, centre=708, sigma=4)
    Fe_l2 = hs.model.components1D.Gaussian(A=50, centre=730, sigma=3)

    if len(navigation_shape) == 0 or navigation_shape == (1, ):
        Mn = 0.5
        Fe = 0.5
    else:
        Mn = np.array([1, 1, 0.75, 0.5, 0])
        Fe = np.array([0, 0, 0.25, 0.5, 1])
        Mn_interpolate = interpolate.interp1d(np.arange(0, len(Mn)), Mn)
        Fe_interpolate = interpolate.interp1d(np.arange(0, len(Fe)), Fe)
        Mn = Mn_interpolate(np.linspace(0, len(Mn)-1, navigation_shape[0]))
        Fe = Fe_interpolate(np.linspace(0, len(Fe)-1, navigation_shape[0]))

    def get_data(component, element_distribution):
        data_ = np.broadcast_to(component.function(x), navigation_shape + x.shape)
        return (data_.T * element_distribution).T

    arctan_Mn_data = get_data(arctan_Mn, Mn)
    Mn_l3_data = get_data(Mn_l3, Mn)
    Mn_l2_data = get_data(Mn_l2, Mn)
    arctan_Fe_data = get_data(arctan_Fe, Fe)
    Fe_l3_data = get_data(Fe_l3, Fe)
    Fe_l2_data = get_data(Fe_l2, Fe)

    data = arctan_Mn_data + Mn_l3_data + Mn_l2_data + arctan_Fe_data + Fe_l3_data + Fe_l2_data

    if add_noise:
        data += random_state.uniform(size=navigation_shape + x.shape) * 0.7

    if add_powerlaw:
        powerlaw = hs.model.components1D.PowerLaw(A=10e8, r=2.9, origin=0)
        data = data + np.broadcast_to(
            powerlaw.function(x), navigation_shape + x.shape
            )

    s = exspy.signals.EELSSpectrum(data)
    s.axes_manager[-1].offset = x[0]
    s.axes_manager[-1].scale = x[1] - x[0]
    s.metadata.General.title = 'Artifical core loss EEL spectrum'
    s.axes_manager[-1].name = 'Electron energy loss'
    s.axes_manager[-1].units = 'eV'
    s.set_microscope_parameters(
        beam_energy=200, convergence_angle=26, collection_angle=20
        )
    s.add_elements(["Fe", "Mn"])
    return s.squeeze()


EELS_MnFe.__doc__ %= (_ADD_NOISE_DOCSTRING, _RETURNS_DOCSTRING)
