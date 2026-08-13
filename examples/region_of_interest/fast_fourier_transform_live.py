"""
Live FFT
========

Get interactive fast Fourier transform (FFT) from a subset of a Signal2D
using RectangularROI.

"""

import hyperspy.api as hs
import numpy as np

# Render with anyplotlib so the figures below stay live in the browser.
hs.preferences.Plot.backend = "anyplotlib"

#%%
# Create a signal:
s = hs.data.atomic_resolution_image()

#%%
# Add noise to the signal to make it more realistic
s.data *= 1E3
s.data += np.random.default_rng().poisson(s.data)

#%%
# Create the ROI, here a :py:class:`~.api.roi.RectangularROI`:
roi = hs.roi.RectangularROI()

#%%
# Slice the signal with the ROI and take the FFT of the slice. Both use the
# :func:`~.api.interactive` function, so ``sliced_signal`` and its FFT
# recompute whenever the ROI moves. Apodization smoothens the edge of the
# image before taking the FFT, which removes streaks from it — see the
# :ref:`signal.fft` section of the user guide for more details.
#
# The two plots are drawn together so you can watch the FFT follow the ROI:
# activate the figures, then drag or resize the green rectangle on the left.
s.plot()
sliced_signal = roi.interactive(s, recompute_out_event=None)

s_fft = hs.interactive(sliced_signal.fft, apodization=True, shift=True, recompute_out_event=None)
s_fft.plot(power_spectrum=True)  # Interactive

# sphinx_gallery_thumbnail_number = 1
