"""
Creates a line spectrum
=======================

This example creates a line spectrum and plots it.
"""

import numpy as np
import hyperspy.api as hs

# %%
# Create a line spectrum with random data
# ----------------------------------------
# Create a line spectrum with random data
s = hs.signals.Signal1D(np.random.random((100, 1024)))

# %%
# Define the axis properties
# --------------------------
# Define the axis properties using batch assignment
s.axes_manager.signal_axes[0].set(name='Energy', units='eV', scale=0.3, offset=100)
s.axes_manager.navigation_axes[0].set(name='time', units='fs', scale=0.3, offset=100)

# %%
# Set metadata and plot
# ---------------------
# Give a title
s.metadata.General.title = 'Random line spectrum'

# Plot it
s.plot()