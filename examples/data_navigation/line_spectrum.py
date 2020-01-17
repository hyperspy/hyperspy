"""
Create a line spectrum
======================

Creates a line spectrum and plots it:

1. Create a `Signal1D` signal.
2. Set the properties of the 2 different axes
3. Plot the signal using the `plot` method

"""

# Set the matplotlib backend of your choice, for example
# %matploltib qt
import numpy as np
import hyperspy.api as hs
import matplotlib.pyplot as plt

# Create a line spectrum with random data with navigation shape of 100
# and signal shape of 1024
s = hs.signals.Signal1D(np.random.random((100, 1024)))

# Define the axis properties
s.axes_manager.signal_axes[0].name = 'Energy'
s.axes_manager.signal_axes[0].units = 'eV'
s.axes_manager.signal_axes[0].scale = 0.3
s.axes_manager.signal_axes[0].offset = 100

s.axes_manager.navigation_axes[0].name = 'time'
s.axes_manager.navigation_axes[0].units = 'fs'
s.axes_manager.navigation_axes[0].scale = 0.3
s.axes_manager.navigation_axes[0].offset = 100

# Give a title
s.metadata.General.title = 'Random line spectrum'

# Plot it
s.plot()

#%%
# 
# ####################################################################
#
#

plt.show()  # No necessary when running in the HyperSpy's IPython profile
