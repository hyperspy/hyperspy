"""
Create a spectrum image
=======================

Creates a spectrum image and plots it:

1. Create a `Signal1D` signal.
2. Set the properties of the 2 different axes
3. Plot the signal using the `plot` method

"""

# Set the matplotlib backend of your choice, for example
# %matploltib qt
import numpy as np
import hyperspy.api as hs
import matplotlib.pyplot as plt

# Create a spectrum image with random data with navigation shape of 64 x 64
# and signal shape of 1024
s = hs.signals.Signal1D(np.random.random((64, 64, 1024)))

# Define the axis properties
s.axes_manager.signal_axes[0].name = 'Energy'
s.axes_manager.signal_axes[0].units = 'eV'
s.axes_manager.signal_axes[0].scale = 0.3
s.axes_manager.signal_axes[0].offset = 100

s.axes_manager.navigation_axes[0].name = 'X'
s.axes_manager.navigation_axes[0].units = 'nm'
s.axes_manager.navigation_axes[0].scale = 0.1
s.axes_manager.navigation_axes[0].offset = 100

s.axes_manager.navigation_axes[1].name = 'Y'
s.axes_manager.navigation_axes[1].units = 'nm'
s.axes_manager.navigation_axes[1].scale = 0.1
s.axes_manager.navigation_axes[1].offset = 100

# Give a title
s.metadata.General.title = 'Random spectrum image'

# Plot it
s.plot()

#%%
# 
# ####################################################################
#
#

# Not necessary when the matplotlib has been set, but we need it here to 
# generate the documentation
plt.show()
