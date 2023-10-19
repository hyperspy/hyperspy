"""
Creates a 4D image
==================

This example creates a 4D dataset, i.e. 2 navigation dimension and
2 signal dimension and plots it.
"""

import numpy as np
import hyperspy.api as hs

# Create a 2D image stack with random data
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

im.plot()