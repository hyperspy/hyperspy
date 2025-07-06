"""
Creates a 3D image
==================

This example creates an image stack and plots it.
"""

import numpy as np
import hyperspy.api as hs

# %%
# Create an image stack with random data
# ---------------------------------------
# Create an image stack with random data
im = hs.signals.Signal2D(np.random.random((16, 32, 32)))

# %%
# Define the axis properties
# --------------------------
# Define the axis properties using batch assignment
im.axes_manager.signal_axes[0].set(name='X', units='nm', scale=0.1, offset=0)
im.axes_manager.signal_axes[1].set(name='Y', units='nm', scale=0.1, offset=0)
im.axes_manager.navigation_axes[0].set(name='time', units='fs', scale=0.3, offset=100)

# %%
# Set metadata and plot
# ---------------------
# Give a title
im.metadata.General.title = 'Random image stack'

# Plot it
im.plot()