"""
Creates a spectrum image
========================

This example creates a spectrum image, i.e. navigation dimension 2 and
signal dimension 1, and plots it.
"""

import numpy as np
import hyperspy.api as hs

# %%
# **Creating a spectrum image with random data**
#
# A spectrum image is a fundamental data structure in multidimensional data analysis.
# It consists of a 2D spatial map where each pixel contains a complete spectrum.

# Create a spectrum image with random data
s = hs.signals.Signal1D(np.random.random((64, 64, 1024)))

# **Data structure:**
# - Navigation dimensions: (64, 64) - 2D spatial scan
# - Signal dimension: (1024,) - spectrum at each position  
# - Total spectra: 4,096 individual measurements

# %%
# **Defining axis properties**
#
# Proper axis calibration is essential for meaningful analysis. We'll set up
# both the spatial (navigation) and spectral (signal) axes with physical units.

# Signal axis (spectral dimension) using batch assignment
s.axes_manager.signal_axes[0].name = 'Energy'
s.axes_manager.signal_axes[0].units = 'eV'
s.axes_manager.signal_axes[0].scale = 0.3
s.axes_manager.signal_axes[0].offset = 100  # 0.3 eV per channel, starting at 100 eV

# Navigation axes (spatial dimensions) using batch assignment
s.axes_manager.navigation_axes[0].name = 'X'
s.axes_manager.navigation_axes[0].units = 'nm'
s.axes_manager.navigation_axes[0].scale = 0.1
s.axes_manager.navigation_axes[0].offset = 100  # 0.1 nm per pixel
s.axes_manager.navigation_axes[1].name = 'Y'
s.axes_manager.navigation_axes[1].units = 'nm'
s.axes_manager.navigation_axes[1].scale = 0.1
s.axes_manager.navigation_axes[1].offset = 100

# %%
# Set metadata and plot
# ---------------------
# Give a title
s.metadata.General.title = 'Random spectrum image'

# Plot it
s.plot()
