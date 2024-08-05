"""
Live FFT
========

Get interactive fast Fourier transform (FFT) from a subset of a Signal2D
using RectangularROi.

Note: the FFT of the example signal is different from what you would expect
in an atomic resolution Transmission Electron Microscopy image.

"""

import hyperspy.api as hs

#%%
# Create a signal:
s = hs.data.atomic_resolution_image()

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
s_fft = hs.interactive(sliced_signal.fft, apodization=True, shift=True, recompute_out_event=None)
s_fft.plot(power_spectrum=True)
