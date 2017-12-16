import numpy as np
from hyperspy import components1d
from hyperspy.signals import EELSSpectrum


def get_core_loss_eel_signal():
    """Get an artifical core loss electron energy loss spectrum.

    Similar to a Mn-L32 edge from a perovskite oxide.

    Returns
    -------
    artifical_core_loss_signal : HyperSpy EELSSpectrum

    Example
    -------
    >>> s = hs.datasets.artifical_data.get_core_loss_eel_signal()
    >>> s.plot()

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
    """Get an artifical core loss electron energy loss model.

    Similar to a Mn-L32 edge from a perovskite oxide.

    Returns
    -------
    artifical_core_loss_model : HyperSpy EELSModel

    Example
    -------
    >>> s = hs.datasets.artifical_data.get_core_loss_eel_model()
    >>> s.plot()

    See also
    --------
    get_core_loss_eel_signal : get a model instead of a signal

    """
    s = get_core_loss_eel_signal()
    m = s.create_model(auto_background=False, GOS='hydrogenic')
    m.fit()
    return m
