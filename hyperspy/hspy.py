"""

This module is deprecated and will be removed in HyperSpy 0.10,
plese use :mod:`~hyperspy.api` instead.

"""
# -*- coding: utf-8 -*-

from hyperspy.Release import version as __version__
from hyperspy import components
from hyperspy import signals
from hyperspy.io import load
from hyperspy.defaults_parser import preferences
from hyperspy import utils
from hyperspy.datasets import example_signals
from hyperspy.misc.hspy_warnings import VisibleDeprecationWarning


def get_configuration_directory_path():
    import hyperspy.misc.config_dir
    return hyperspy.misc.config_dir.config_path


def create_model(signal, *args, **kwargs):
    """Create a model object

    Any extra argument is passes to the Model constructor.

    Parameters
    ----------
    signal: A signal class

    If the signal is an EELS signal the following extra parameters
    are available:

    auto_background : boolean
        If True, and if spectrum is an EELS instance adds automatically
        a powerlaw to the model and estimate the parameters by the
        two-area method.
    auto_add_edges : boolean
        If True, and if spectrum is an EELS instance, it will
        automatically add the ionization edges as defined in the
        Spectrum instance. Adding a new element to the spectrum using
        the components.EELSSpectrum.add_elements method automatically
        add the corresponding ionisation edges to the model.
    ll : {None, EELSSpectrum}
        If an EELSSPectrum is provided, it will be assumed that it is
        a low-loss EELS spectrum, and it will be used to simulate the
        effect of multiple scattering by convolving it with the EELS
        spectrum.
    GOS : {'hydrogenic', 'Hartree-Slater', None}
        The GOS to use when auto adding core-loss EELS edges.
        If None it will use the Hartree-Slater GOS if
        they are available, otherwise it will use the hydrogenic GOS.

    Returns
    -------

    A Model class

    """
    import warnings
    warnings.warn(
        "This function is deprecated and will be removed in Hyperspy 1.0. "
        "Please use the equivalent `Signal.create_model` method "
        "instead.", VisibleDeprecationWarning)
    return signal.create_model(*args, **kwargs)
