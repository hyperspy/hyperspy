"""Plotting funtions.

Functions:

plot_spectra, plot_images
    Plot multiple spectra/images in the same figure.
plot_signals
    Plot multiple signals at the same time.
plot_histograms
    Compute and plot the histograms of multiple signals in the same figure.

The :mod:`~hyperspy.api.plot` module contains the following submodules:

:mod:`~hyperspy.api.markers`
        Markers that can be added to `Signal` plots.

"""

from hyperspy.drawing.utils import plot_spectra
from hyperspy.drawing.utils import plot_images
from hyperspy.drawing.utils import plot_signals
from hyperspy.drawing.utils import plot_histograms
from hyperspy.utils import markers
