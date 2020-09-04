"""Functions for generating artificial data.

For use in things like docstrings or to test HyperSpy functionalities.

"""

import numpy as np

from hyperspy.misc.math_tools import check_random_state


ADD_POWERLAW_DOCSTRING = \
"""add_powerlaw : bool
        If True, adds a powerlaw background to the spectrum. Default is False.
    """


ADD_NOISE_DOCSTRING = \
"""add_noise : bool
        If True, add noise to the signal. See note to seed the noise to
        generate reproducible noise.
    random_state : None or int or RandomState instance, default None
        Random seed used to generate the data.
    """

RETURNS_DOCSTRING = \
"""Returns
    -------
    :py:class:`~hyperspy._signals.eels.EELSSpectrum`
    """


def get_low_loss_eels_signal(add_noise=True, random_state=None):
    """Get an artificial low loss electron energy loss spectrum.

    The zero loss peak is offset by 4.1 eV.

    Parameters
    ----------
    %s
    %s

    Returns
    -------
    artificial_low_loss_signal : :py:class:`~hyperspy._signals.eels.EELSSpectrum`

    Example
    -------
    >>> s = hs.datasets.artificial_data.get_low_loss_eels_signal()
    >>> s.plot()

    See also
    --------
    get_core_loss_eels_signal, get_core_loss_eels_model,
    get_low_loss_eels_line_scan_signal, get_core_loss_eels_line_scan_signal

    """

    from hyperspy.signals import EELSSpectrum
    from hyperspy import components1d

    random_state = check_random_state(random_state)

    x = np.arange(-100, 400, 0.5)
    zero_loss = components1d.Gaussian(A=100, centre=4.1, sigma=1)
    plasmon = components1d.Gaussian(A=100, centre=60, sigma=20)

    data = zero_loss.function(x)
    data += plasmon.function(x)
    if add_noise:
        data += random_state.uniform(size=len(x)) * 0.7

    s = EELSSpectrum(data)
    s.axes_manager[0].offset = x[0]
    s.axes_manager[0].scale = x[1] - x[0]
    s.metadata.General.title = 'Artifical low loss EEL spectrum'
    s.axes_manager[0].name = 'Electron energy loss'
    s.axes_manager[0].units = 'eV'
    s.set_microscope_parameters(
        beam_energy=200, convergence_angle=26, collection_angle=20)
    return s

get_low_loss_eels_signal.__doc__ %= (ADD_NOISE_DOCSTRING,
                                     RETURNS_DOCSTRING)


def get_core_loss_eels_signal(add_powerlaw=False, add_noise=True, random_state=None):
    """Get an artificial core loss electron energy loss spectrum.

    Similar to a Mn-L32 edge from a perovskite oxide.

    Some random noise is also added to the spectrum, to simulate
    experimental noise.

    Parameters
    ----------
    %s
    %s

    %s

    Example
    -------
    >>> import hs.datasets.artifical_data as ad
    >>> s = ad.get_core_loss_eels_signal()
    >>> s.plot()

    With the powerlaw background

    >>> s = ad.get_core_loss_eels_signal(add_powerlaw=True)
    >>> s.plot()

    To make the noise the same for multiple spectra, which can
    be useful for testing fitting routines

    >>> s1 = ad.get_core_loss_eels_signal(random_state=10)
    >>> s2 = ad.get_core_loss_eels_signal(random_state=10)
    >>> (s1.data == s2.data).all()
    True

    See also
    --------
    get_core_loss_eels_line_scan_signal, get_low_loss_eels_line_scan_signal,
    get_core_loss_eels_model

    """

    from hyperspy.signals import EELSSpectrum
    from hyperspy import components1d

    random_state = check_random_state(random_state)

    x = np.arange(400, 800, 1)
    arctan = components1d.EELSArctan(A=1, k=0.2, x0=688)
    mn_l3_g = components1d.Gaussian(A=100, centre=695, sigma=4)
    mn_l2_g = components1d.Gaussian(A=20, centre=720, sigma=4)

    data = arctan.function(x)
    data += mn_l3_g.function(x)
    data += mn_l2_g.function(x)
    if add_noise:
        data += random_state.uniform(size=len(x)) * 0.7

    if add_powerlaw:
        powerlaw = components1d.PowerLaw(A=10e8, r=3, origin=0)
        data += powerlaw.function(x)

    s = EELSSpectrum(data)
    s.axes_manager[0].offset = x[0]
    s.metadata.General.title = 'Artifical core loss EEL spectrum'
    s.axes_manager[0].name = 'Electron energy loss'
    s.axes_manager[0].units = 'eV'
    s.set_microscope_parameters(
        beam_energy=200, convergence_angle=26, collection_angle=20)
    return s

get_core_loss_eels_signal.__doc__ %= (ADD_POWERLAW_DOCSTRING,
                                      ADD_NOISE_DOCSTRING,
                                      RETURNS_DOCSTRING)


def get_low_loss_eels_line_scan_signal(add_noise=True, random_state=None):
    """Get an artificial low loss electron energy loss line scan spectrum.

    The zero loss peak is offset by 4.1 eV.

    Parameters
    ----------
    %s

    %s

    Example
    -------
    >>> s = hs.datasets.artificial_data.get_low_loss_eels_signal()
    >>> s.plot()

    See also
    --------
    artificial_low_loss_line_scan_signal : :py:class:`~hyperspy._signals.eels.EELSSpectrum`


    """

    from hyperspy.signals import EELSSpectrum
    from hyperspy import components1d

    random_state = check_random_state(random_state)

    x = np.arange(-100, 400, 0.5)
    zero_loss = components1d.Gaussian(A=100, centre=4.1, sigma=1)
    plasmon = components1d.Gaussian(A=100, centre=60, sigma=20)

    data_signal = zero_loss.function(x)
    data_signal += plasmon.function(x)
    data = np.zeros((12, len(x)))
    for i in range(12):
        data[i] += data_signal
        if add_noise:
            data[i] += random_state.uniform(size=len(x)) * 0.7

    s = EELSSpectrum(data)
    s.axes_manager.signal_axes[0].offset = x[0]
    s.axes_manager.signal_axes[0].scale = x[1] - x[0]
    s.metadata.General.title = 'Artifical low loss EEL spectrum'
    s.axes_manager.signal_axes[0].name = 'Electron energy loss'
    s.axes_manager.signal_axes[0].units = 'eV'
    s.axes_manager.navigation_axes[0].name = 'Probe position'
    s.axes_manager.navigation_axes[0].units = 'nm'
    s.set_microscope_parameters(
        beam_energy=200, convergence_angle=26, collection_angle=20)
    return s

get_low_loss_eels_line_scan_signal.__doc__ %= (ADD_NOISE_DOCSTRING,
                                               RETURNS_DOCSTRING)


def get_core_loss_eels_line_scan_signal(add_powerlaw=False, add_noise=True, random_state=None):
    """Get an artificial core loss electron energy loss line scan spectrum.

    Similar to a Mn-L32 and Fe-L32 edge from a perovskite oxide.

    Parameters
    ----------
    %s
    %s

    %s

    Example
    -------
    >>> s = hs.datasets.artificial_data.get_core_loss_eels_line_scan_signal()
    >>> s.plot()

    See also
    --------
    get_low_loss_eels_line_scan_signal, get_core_loss_eels_model

    """

    from hyperspy.signals import EELSSpectrum
    from hyperspy import components1d

    random_state = check_random_state(random_state)

    x = np.arange(400, 800, 1)
    arctan_mn = components1d.EELSArctan(A=1, k=0.2, x0=688)
    arctan_fe = components1d.EELSArctan(A=1, k=0.2, x0=612)
    mn_l3_g = components1d.Gaussian(A=100, centre=695, sigma=4)
    mn_l2_g = components1d.Gaussian(A=20, centre=720, sigma=4)
    fe_l3_g = components1d.Gaussian(A=100, centre=605, sigma=4)
    fe_l2_g = components1d.Gaussian(A=10, centre=630, sigma=3)

    mn_intensity = [1, 1, 1, 1, 1, 1, 0.8, 0.5, 0.2, 0, 0, 0]
    fe_intensity = [0, 0, 0, 0, 0, 0, 0.2, 0.5, 0.8, 1, 1, 1]
    data = np.zeros((len(mn_intensity), len(x)))
    for i in range(len(mn_intensity)):
        data[i] += arctan_mn.function(x) * mn_intensity[i]
        data[i] += mn_l3_g.function(x) * mn_intensity[i]
        data[i] += mn_l2_g.function(x) * mn_intensity[i]
        data[i] += arctan_fe.function(x) * fe_intensity[i]
        data[i] += fe_l3_g.function(x) * fe_intensity[i]
        data[i] += fe_l2_g.function(x) * fe_intensity[i]
        if add_noise:
            data[i] += random_state.uniform(size=len(x)) * 0.7

    if add_powerlaw:
        powerlaw = components1d.PowerLaw(A=10e8, r=3, origin=0)
        data += powerlaw.function_nd(np.stack([x]*len(mn_intensity)))

    if add_powerlaw:
        powerlaw = components1d.PowerLaw(A=10e8, r=3, origin=0)
        data += powerlaw.function(x)

    s = EELSSpectrum(data)
    s.axes_manager.signal_axes[0].offset = x[0]
    s.metadata.General.title = 'Artifical core loss EEL spectrum'
    s.axes_manager.signal_axes[0].name = 'Electron energy loss'
    s.axes_manager.signal_axes[0].units = 'eV'
    s.axes_manager.navigation_axes[0].name = 'Probe position'
    s.axes_manager.navigation_axes[0].units = 'nm'
    s.set_microscope_parameters(
        beam_energy=200, convergence_angle=26, collection_angle=20)
    return s

get_core_loss_eels_line_scan_signal.__doc__ %= (ADD_POWERLAW_DOCSTRING,
                                                ADD_NOISE_DOCSTRING,
                                                RETURNS_DOCSTRING)


def get_core_loss_eels_model(add_powerlaw=False, add_noise=True, random_state=None):
    """Get an artificial core loss electron energy loss model.

    Similar to a Mn-L32 edge from a perovskite oxide.

    Parameters
    ----------
    %s
    %s

    Returns
    -------
    :py:class:`~hyperspy.models.eelsmodel.EELSModel`

    Example
    -------
    >>> import hs.datasets.artifical_data as ad
    >>> s = ad.get_core_loss_eels_model()
    >>> s.plot()

    With the powerlaw background

    >>> s = ad.get_core_loss_eels_model(add_powerlaw=True)
    >>> s.plot()

    See also
    --------
    get_core_loss_eels_signal

    """
    s = get_core_loss_eels_signal(add_powerlaw=add_powerlaw,
                                  add_noise=add_noise,
                                  random_state=random_state)
    m = s.create_model(auto_background=False, GOS='hydrogenic')
    return m

get_core_loss_eels_model.__doc__ %= (ADD_POWERLAW_DOCSTRING,
                                     ADD_NOISE_DOCSTRING)


def get_atomic_resolution_tem_signal2d():
    """Get an artificial atomic resolution TEM Signal2D.

    Returns
    -------
    :py:class:`~hyperspy._signals.signal2d.Signal2D`

    Example
    -------
    >>> s = hs.datasets.artificial_data.get_atomic_resolution_tem_signal2d()
    >>> s.plot()

    """
    from hyperspy.signals import Signal2D
    from hyperspy import components2d

    sX, sY = 2, 2
    x_array, y_array = np.mgrid[0:200, 0:200]
    image = np.zeros_like(x_array, dtype=np.float32)
    gaussian2d = components2d.Gaussian2D(sigma_x=sX, sigma_y=sY)
    for x in range(10, 195, 20):
        for y in range(10, 195, 20):
            gaussian2d.centre_x.value = x
            gaussian2d.centre_y.value = y
            image += gaussian2d.function(x_array, y_array)

    s = Signal2D(image)
    return s
