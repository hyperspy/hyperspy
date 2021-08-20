"""Functions for generating artificial data.

For use in things like docstrings or to test HyperSpy functionalities.

"""

import numpy as np

from hyperspy import components1d, components2d
from hyperspy import signals
from hyperspy.misc.math_tools import check_random_state
from hyperspy.axes import UniformDataAxis


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

    Example
    -------
    >>> s = hs.datasets.artificial_data.get_low_loss_eels_signal()
    >>> s.plot()

    See also
    --------
    get_core_loss_eels_signal, get_core_loss_eels_model,
    get_low_loss_eels_line_scan_signal, get_core_loss_eels_line_scan_signal

    """

    random_state = check_random_state(random_state)

    x = np.arange(-100, 400, 0.5)
    zero_loss = components1d.Gaussian(A=100, centre=4.1, sigma=1)
    plasmon = components1d.Gaussian(A=100, centre=60, sigma=20)

    data = zero_loss.function(x)
    data += plasmon.function(x)
    if add_noise:
        data += random_state.uniform(size=len(x)) * 0.7

    s = signals.EELSSpectrum(data)
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


def get_core_loss_eels_signal(add_powerlaw=False, add_noise=True,
                              random_state=None):
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

    s = signals.EELSSpectrum(data)
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

    s = signals.EELSSpectrum(data)
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


def get_core_loss_eels_line_scan_signal(add_powerlaw=False, add_noise=True,
                                        random_state=None):
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

    s = signals.EELSSpectrum(data)
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


def get_core_loss_eels_model(add_powerlaw=False, add_noise=True,
                             random_state=None):
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


def get_luminescence_signal(navigation_dimension=0,
                            uniform=False,
                            add_baseline=False,
                            add_noise=True,
                            random_state=None):
    """Get an artificial luminescence signal in wavelength scale (nm, uniform) or
    energy scale (eV, non-uniform), simulating luminescence data recorded with a
    diffracting spectrometer. Some random noise is also added to the spectrum,
    to simulate experimental noise.

    Parameters
    ----------
    navigation_dimension: positive int.
        The navigation dimension(s) of the signal. 0 = single spectrum,
        1 = linescan, 2 = spectral map etc...
    uniform: bool.
        return uniform (wavelength) or non-uniform (energy) spectrum
    add_baseline : bool
        If true, adds a constant baseline to the spectrum. Conversion to
        energy representation will turn the constant baseline into inverse
        powerlaw.
    %s

    Example
    -------
    >>> import hyperspy.datasets.artificial_data as ad
    >>> s = ad.get_luminescence_signal()
    >>> s.plot()

    With constant baseline

    >>> s = ad.get_luminescence_signal(uniform=True, add_baseline=True)
    >>> s.plot()

    To make the noise the same for multiple spectra, which can
    be useful for testing fitting routines

    >>> s1 = ad.get_luminescence_signal(random_state=10)
    >>> s2 = ad.get_luminescence_signal(random_state=10)
    >>> (s1.data == s2.data).all()
    True

    2D map

    >>> s = ad.get_luminescence_signal(navigation_dimension=2)
    >>> s.plot()

    See also
    --------
    get_low_loss_eels_signal,
    get_core_loss_eels_signal,
    get_low_loss_eels_line_scan_signal,
    get_core_loss_eels_line_scan_signal,
    get_core_loss_eels_model,
    get_atomic_resolution_tem_signal2d,

    """

    #Initialisation of random number generator
    random_state = check_random_state(random_state)

    #Creating a uniform data axis, roughly similar to Horiba iHR320 with a 150 mm-1 grating
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

    #Artificial luminescence peak
    gaussian_peak = components1d.Gaussian(A=5000, centre=375, sigma=25)

    if navigation_dimension>=0:
        #Generate empty data (ones)
        data = np.ones([10 for i in range(navigation_dimension)]+[nm_axis.size])
        #Generate spatial axes
        spaxes = [UniformDataAxis(index_in_array=None,
                name="X{:d}".format(i),
                units="um",
                navigate=False,
                size=10,
                scale=2.1,
                offset=0,
                is_binned=False,
                )  for i in range(navigation_dimension)]
        #Generate empty signal
        sig = signals.Signal1D(data,axes = spaxes + [nm_axis])
        sig.metadata.General.title = '{:d}d-map Artificial Luminescence Signal'\
                                        .format(navigation_dimension)
    else:
        raise ValueError("Value {:d} invalid as navigation dimension.".format(\
                            navigation_dimension))

    #Populating data array, possibly with noise and baseline
    sig.data *= gaussian_peak.function(nm_axis.axis)
    if add_noise:
        sig.data += (random_state.uniform(size=sig.data.shape) - 0.5)*1.4
    if add_baseline:
        data += 350.

    #if not uniform, transformation into non-uniform axis
    if not uniform:
        hc = 1239.84198 #nm/eV
        #converting to non-uniform axis
        sig.axes_manager.signal_axes[0].convert_to_functional_data_axis(\
                                                              expression="a/x",
                                                              name='Energy',
                                                              units='eV',
                                                              a=hc,
                                                          )
        #Reverting the orientation of signal axis to have increasing Energy
        sig = sig.isig[::-1]
        #Jacobian transformation
        Eax = sig.axes_manager.signal_axes[0].axis
        sig *= hc/Eax**2
    return sig

get_luminescence_signal.__doc__ %= (ADD_NOISE_DOCSTRING)
