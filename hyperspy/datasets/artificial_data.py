import numpy as np
from hyperspy import components1d, components2d
from hyperspy.signals import EELSSpectrum, Signal2D


def get_core_loss_eel_signal():
    """Get an artificial core loss electron energy loss spectrum.

    Similar to a Mn-L32 edge from a perovskite oxide.

    Returns
    -------
    artificial_core_loss_signal : HyperSpy EELSSpectrum

    Example
    -------
    >>> s = hs.datasets.artificial_data.get_core_loss_eel_signal()
    >>> s.plot()
artifical
    See also
    --------
    get_core_loss_eel_model : get a model instead of a signal

    """
    x = np.arange(400, 800, 1)
    arctan = components1d.Arctan(A=1, k=0.2, x0=688)
    arctan.minimum_at_zero = True
    mn_l3_g = components1d.Gaussian(A=100, centre=695, sigma=4)
    mn_l2_g = components1d.Gaussian(A=20, centre=720, sigma=4)

    data = arctan.function(x)
    data += mn_l3_g.function(x)
    data += mn_l2_g.function(x)
    data += np.random.random(size=len(x))*0.7

    s = EELSSpectrum(data)
    s.axes_manager[0].offset = x[0]
    s.metadata.General.title = 'Artifical core loss EEL spectrum'
    s.axes_manager[0].name = 'Electron energy loss'
    s.axes_manager[0].units = 'eV'
    s.set_microscope_parameters(
            beam_energy=200, convergence_angle=26, collection_angle=20)
    return s


def get_core_loss_eel_model():
    """Get an artificial core loss electron energy loss model.

    Similar to a Mn-L32 edge from a perovskite oxide.

    Returns
    -------
    artificial_core_loss_model : HyperSpy EELSModel

    Example
    -------
    >>> s = hs.datasets.artificial_data.get_core_loss_eel_model()
    >>> s.plot()

    See also
    --------
    get_core_loss_eel_signal : get a model instead of a signal

    """
    s = get_core_loss_eel_signal()
    m = s.create_model(auto_background=False, GOS='hydrogenic')
    m.fit()
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
