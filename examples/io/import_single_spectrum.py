"""
Import single spectrum
======================

Import a single from a csv file and save it:

1. Read the data using `numpy.loadtxt`
2. Create a single sprectrum using `Signal1D` signal
3. Calibrate the data and set the corresponding axis

"""

# Set the matplotlib backend of your choice, for example
# %matploltib qt
import hyperspy.api as hs
import numpy as np

x, y = np.loadtxt('EDS_spectrum.csv', delimiter=',', skiprows=3, unpack=True)

s = hs.signals.EDSSEMSpectrum(y)

# Get the scale of the x axis by computing the difference between two 
# consecutive values
scale = x[1] - x[0]

# Get the offset of the x-axis:
offset = x[0]

# Set the scale and the offset to the signal axis:
s.axes_manager.signal_axes[0].scale = scale
s.axes_manager.signal_axes[0].offset = offset

# Set the name and the unit of the energy axis
s.axes_manager.signal_axes[0].units = 'keV'
s.axes_manager.signal_axes[0].name = 'X-rays Energy'
