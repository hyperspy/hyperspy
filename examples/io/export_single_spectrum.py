"""
Export single spectrum
======================

Creates a single spectrum image, saves it and plots it:

1. Create a single sprectrum using `Signal1D` signal.
2. Save signal as a msa file
3. Plot the signal using the `plot` method
4. Save the figure as a png file

.. note::
    The figure below cannot be made live in the browser: the in-browser
    interpreter has no file system to save the ``.msa`` to, and ships without
    RosettaSciIO's format plugins. The figure is still an anyplotlib widget,
    so panning and zooming work as usual.

"""

import hyperspy.api as hs
import numpy as np

# Render with anyplotlib so the figures below stay live in the browser.
hs.preferences.Plot.backend = "anyplotlib"

s = hs.signals.Signal1D(np.random.rand(1024))

# Export as msa file, very similar to a csv file but containing standardised
# metadata
s.save('testSpectrum.msa', overwrite=True)

# Plot it
s.plot()
