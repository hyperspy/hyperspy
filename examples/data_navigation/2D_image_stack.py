"""
Create a 4D image
=================

Creates a 4D image and plots it:

1. Create a `Signal2D` signal.
2. Set the properties of the 3 different axes
3. Plot the signal using the `plot` method

"""

# Set the matplotlib backend of your choice, for example
# %matploltib qt
import numpy as np
import hyperspy.api as hs
import matplotlib.pyplot as plt

# Create a 2D image stack of random data with navigation shape of 16 x 16
# and signal shape of 32 x 32
im = hs.signals.Signal2D(np.random.random((16, 16, 32, 32)))

# Define the axis properties
im.axes_manager.signal_axes[0].name = ''
im.axes_manager.signal_axes[0].units = '1/nm'
im.axes_manager.signal_axes[0].scale = 0.1
im.axes_manager.signal_axes[0].offset = 0

im.axes_manager.signal_axes[1].name = ''
im.axes_manager.signal_axes[1].units = '1/nm'
im.axes_manager.signal_axes[1].scale = 0.1
im.axes_manager.signal_axes[1].offset = 0

im.axes_manager.navigation_axes[0].name = 'X'
im.axes_manager.navigation_axes[0].units = 'nm'
im.axes_manager.navigation_axes[0].scale = 0.3
im.axes_manager.navigation_axes[0].offset = 100

im.axes_manager.navigation_axes[1].name = 'Y'
im.axes_manager.navigation_axes[1].units = 'nm'
im.axes_manager.navigation_axes[1].scale = 0.3
im.axes_manager.navigation_axes[1].offset = 100

# Give a title
im.metadata.General.title = 'Random 2D image stack'

# Plot it
im.plot()

#%%
# 
# ####################################################################
#
#

# Not necessary when the matplotlib has been set, but we need it here to 
# generate the documentation
plt.show()
