"""
The :mod:`hyperspy.datasets` module includes access to local and remote
datasets.

Functions:

    eelsdb
        Download spectra from the EELS data base https://eelsdb.eu

Submodules:

The :mod:`~hyperspy.api.datasets` module contains the following submodules:

    :mod:`~hyperspy.api.datasets.example_signals`
        Example datasets distributed with HyperSpy.


"""

from hyperspy.misc.eels.eelsdb import eelsdb
from hyperspy.datasets import artificial_data, example_signals


__all__ = [
    'artificial_data',
    'eelsdb',
    'example_signals',
    ]


def __dir__():
    return sorted(__all__)
