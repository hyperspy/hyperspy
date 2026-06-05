"""
Live FFT
========

Get interactive fast Fourier transform (FFT) from a subset of a Signal2D
using RectangularROI.

"""

import hyperspy.api as hs
import numpy as np

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
# Slice signal with the ROI. By using the `interactive` function, the
# output signal ``sliced_signal`` will update automatically.
# The ROI will be added automatically on the signal plot.
s.plot()
sliced_signal = roi.interactive(s, recompute_out_event=None)

# Choose the second figure as gallery thumbnail:
# sphinx_gallery_thumbnail_number = 2

#%%
# Get the FFT of this sliced signal, and plot it
# Apodization is used to smoothen the edge of the image before taking the FFT
# to remove streaks from the FFT - see the :ref:`signal.fft` section of the
# user guide for more details:
s_fft = hs.interactive(sliced_signal.fft, apodization=True, shift=True, recompute_out_event=None)
s_fft.plot(power_spectrum=True)
