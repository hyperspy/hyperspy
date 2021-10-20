"""Functions for generating artificial data.

For use in things like docstrings or to test HyperSpy functionalities.

"""

import numpy as np
from hyperspy import components1d, components2d
from hyperspy.signals import EELSSpectrum, Signal2D


def get_low_loss_eels_signal():
    """Get an artificial low loss electron energy loss spectrum.

    The zero loss peak is offset by 4.1 eV.

    Returns
    -------
    artificial_low_loss_signal : HyperSpy EELSSpectrum

    Example
    -------
    >>> s = hs.datasets.artificial_data.get_low_loss_eels_signal()
    >>> s.plot()

    See also
    --------
    get_core_loss_eels_signal : get a core loss signal
    get_core_loss_eels_model : get a core loss model
    get_low_loss_eels_line_scan_signal : get EELS low loss line scan
    get_core_loss_eels_line_scan_signal : get EELS core loss line scan

    """
    x = np.arange(-100, 400, 0.5)
    zero_loss = components1d.Gaussian(A=100, centre=4.1, sigma=1)
    plasmon = components1d.Gaussian(A=100, centre=60, sigma=20)

    data = zero_loss.function(x)
    data += plasmon.function(x)
    data += np.random.random(size=len(x)) * 0.7

    s = EELSSpectrum(data)
    s.axes_manager[0].offset = x[0]
    s.axes_manager[0].scale = x[1] - x[0]
    s.metadata.General.title = 'Artifical low loss EEL spectrum'
    s.axes_manager[0].name = 'Electron energy loss'
    s.axes_manager[0].units = 'eV'
    s.set_microscope_parameters(
        beam_energy=200, convergence_angle=26, collection_angle=20)
    return s


def get_core_loss_eels_signal(add_powerlaw=False):
    """Get an artificial core loss electron energy loss spectrum.

    Similar to a Mn-L32 edge from a perovskite oxide.

    Some random noise is also added to the spectrum, to simulate
    experimental noise.

    Parameters
    ----------
    add_powerlaw : bool
        If True, adds a powerlaw background to the spectrum.
        Default False.

    Returns
    -------
    artificial_core_loss_signal : HyperSpy EELSSpectrum

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

    >>> np.random.seed(seed=10)
    >>> s1 = ad.get_core_loss_eels_signal()
    >>> np.random.seed(seed=10)
    >>> s2 = ad.get_core_loss_eels_signal()
    >>> (s1.data == s2.data).all()
    True

    See also
    --------
    get_low_loss_eels_model : get a low loss signal
    get_core_loss_eels_model : get a model instead of a signal
    get_low_loss_eels_line_scan_signal : get EELS low loss line scan
    get_core_loss_eels_line_scan_signal : get EELS core loss line scan

    """
    x = np.arange(400, 800, 1)
    arctan = components1d.Arctan(A=1, k=0.2, x0=688)
    arctan.minimum_at_zero = True
    mn_l3_g = components1d.Gaussian(A=100, centre=695, sigma=4)
    mn_l2_g = components1d.Gaussian(A=20, centre=720, sigma=4)

    data = arctan.function(x)
    data += mn_l3_g.function(x)
    data += mn_l2_g.function(x)
    data += np.random.random(size=len(x)) * 0.7

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


def get_low_loss_eels_line_scan_signal():
    """Get an artificial low loss electron energy loss line scan spectrum.

    The zero loss peak is offset by 4.1 eV.

    Returns
    -------
    artificial_low_loss_line_scan_signal : HyperSpy EELSSpectrum

    Example
    -------
    >>> s = hs.datasets.artificial_data.get_low_loss_eels_signal()
    >>> s.plot()

    See also
    --------
    get_core_loss_eels_signal : get a core loss signal
    get_core_loss_eels_model : get a core loss model
    get_core_loss_eels_line_scan_signal : core loss signal with the same size

    """
    x = np.arange(-100, 400, 0.5)
    zero_loss = components1d.Gaussian(A=100, centre=4.1, sigma=1)
    plasmon = components1d.Gaussian(A=100, centre=60, sigma=20)

    data_signal = zero_loss.function(x)
    data_signal += plasmon.function(x)
    data = np.zeros((12, len(x)))
    for i in range(12):
        data[i] += data_signal
        data[i] += np.random.random(size=len(x)) * 0.7

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


def get_core_loss_eels_line_scan_signal():
    """Get an artificial core loss electron energy loss line scan spectrum.

    Similar to a Mn-L32 and Fe-L32 edge from a perovskite oxide.

    Returns
    -------
    artificial_core_loss_line_scan_signal : HyperSpy EELSSpectrum

    Example
    -------
    >>> s = hs.datasets.artificial_data.get_core_loss_eels_line_scan_signal()
    >>> s.plot()

    See also
    --------
    get_low_loss_eels_model : get a low loss signal
    get_core_loss_eels_model : get a model instead of a signal
    get_low_loss_eels_line_scan_signal : get low loss signal with the same size

    """
    x = np.arange(400, 800, 1)
    arctan_mn = components1d.Arctan(A=1, k=0.2, x0=688)
    arctan_mn.minimum_at_zero = True
    arctan_fe = components1d.Arctan(A=1, k=0.2, x0=612)
    arctan_fe.minimum_at_zero = True
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
        data[i] += np.random.random(size=len(x)) * 0.7

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


def get_core_loss_eels_model(add_powerlaw=False):
    """Get an artificial core loss electron energy loss model.

    Similar to a Mn-L32 edge from a perovskite oxide.

    Parameters
    ----------
    add_powerlaw : bool
        If True, adds a powerlaw background to the spectrum.
        Default False.

    Returns
    -------
    artificial_core_loss_model : HyperSpy EELSModel

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
    get_low_loss_eels_model : get a low loss signal
    get_core_loss_eels_signal : get a model instead of a signal

    """
    s = get_core_loss_eels_signal(add_powerlaw=add_powerlaw)
    m = s.create_model(auto_background=False, GOS='hydrogenic')
    return m


def get_atomic_resolution_tem_signal2d():
    """Get an artificial atomic resolution TEM Signal2D.

    Returns
    -------
    artificial_tem_image : HyperSpy Signal2D

    Example
    -------
    >>> s = hs.datasets.artificial_data.get_atomic_resolution_tem_signal2d()
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
    s = Signal2D(image)
    return s
