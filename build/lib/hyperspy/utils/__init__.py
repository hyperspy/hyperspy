"""

Functions that operate on Signal instances and other goodies.

    stack
        Stack Signal instances.

Subpackages:

    material
        Tools related to the material under study.
    plot
        Tools for plotting.
    eds
        Tools for energy-dispersive X-ray data analysis.
    example_signals
        A few example of signal


"""
import hyperspy.utils.material
import hyperspy.utils.eds
import hyperspy.utils.plot
import hyperspy.datasets.example_signals
import hyperspy.utils.model
from hyperspy.misc.utils import (stack, transpose)
from hyperspy.interactive import interactive
import hyperspy.utils.roi
import hyperspy.utils.samfire
