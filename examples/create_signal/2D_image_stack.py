"""
Creates a 4D image
==================

This example creates a 4D dataset, i.e. 2 navigation dimension and
2 signal dimension and plots it.
"""

import numpy as np
import hyperspy.api as hs

# %%
# Create a 2D image stack with random data
# -----------------------------------------
# Create a 2D image stack with random data
im = hs.signals.Signal2D(np.random.random((16, 16, 32, 32)))

# %%
# Define the axis properties
# --------------------------
# Define the axis properties using batch assignment
im.axes_manager.signal_axes[0].set(name='', units='1/nm', scale=0.1, offset=0)
im.axes_manager.signal_axes[1].set(name='', units='1/nm', scale=0.1, offset=0)
im.axes_manager.navigation_axes[0].set(name='X', units='nm', scale=0.3, offset=100)
im.axes_manager.navigation_axes[1].set(name='Y', units='nm', scale=0.3, offset=100)

# %%
# Set metadata and plot
# ---------------------
# Give a title
im.metadata.General.title = 'Random 2D image stack'

im.plot()