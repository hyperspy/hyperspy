"""
Creating Signal2D from numpy array
===================================

This example demonstrates how to create a HyperSpy Signal2D from a numpy array
and understand different signal arrangements (image stack vs map of images).
"""

# %%
# Create different Signal2D arrangements
import numpy as np
import hyperspy.api as hs

# %%
# Example 1: Image stack (signal dimension 2, navigation dimension 1)
# Shape: (30, 10, 20) -> 30 images of size 10x20
image_stack_data = np.random.random((30, 10, 20))
image_stack = hs.signals.Signal2D(image_stack_data)

# %%
# **Image stack: 1D navigation, 2D signal**
#
# This creates a sequence of 30 images, each with dimensions 10×20 pixels.
# Perfect for time series, depth series, or any sequential image data.

# %%
# Example 2: Map of images (signal dimension 2, navigation dimension 2)  
# Shape: (30, 10, 10, 20) -> 30x10 grid, each containing a 10x20 image
image_map_data = np.random.random((30, 10, 10, 20))
image_map = hs.signals.Signal2D(image_map_data)

# %%
# **Map of images: 2D navigation, 2D signal**
#
# This creates a 2D grid (30×10) where each position contains a 10×20 image.
# Useful for scanning experiments where an image is acquired at each scan position.

# %%
# Set meaningful axis properties for the image stack
image_stack.axes_manager.navigation_axes[0].name = 'Image number'
image_stack.axes_manager.navigation_axes[0].units = ''

image_stack.axes_manager.signal_axes[0].name = 'Y'
image_stack.axes_manager.signal_axes[0].units = 'px'
image_stack.axes_manager.signal_axes[1].name = 'X'  
image_stack.axes_manager.signal_axes[1].units = 'px'

image_stack.metadata.General.title = 'Random image stack'

# %%
# Plot the image stack
image_stack.plot()

# %%
# Plot the image map
image_map.metadata.General.title = 'Random image map'
image_map.plot()
