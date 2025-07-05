"""
Adding Linear Ramps to Signal2D
===============================

This example demonstrates how to add linear ramps to 2D signals using
HyperSpy's add_ramp method. Linear ramps can be useful for simulating
drift, correcting illumination gradients, or creating test patterns.

"""

import numpy as np
import hyperspy.api as hs

# Create a synthetic 2D signal - a simple Gaussian peak
size = 100
x = np.linspace(-3, 3, size)
y = np.linspace(-3, 3, size)
X, Y = np.meshgrid(x, y)

# Create a 2D Gaussian
data = np.exp(-(X**2 + Y**2) / 2)

# Create Signal2D with proper scale
s = hs.signals.Signal2D(data)
s.axes_manager[0].scale = 0.1  # Y-axis scale: 0.1 nm/pixel
s.axes_manager[1].scale = 0.1  # X-axis scale: 0.1 nm/pixel
s.axes_manager[0].units = 'nm'
s.axes_manager[1].units = 'nm'
s.axes_manager[0].name = 'y'
s.axes_manager[1].name = 'x'
s.metadata.General.title = "Original Gaussian signal"

print("Original signal shape:", s.data.shape)
print("X-axis: scale =", s.axes_manager[1].scale, s.axes_manager[1].units)
print("Y-axis: scale =", s.axes_manager[0].scale, s.axes_manager[0].units)

# Display the original signal and ramp examples using HyperSpy's plot_images

# Example 1: Add a linear ramp in X direction only
s_ramp_x = s.deepcopy()
s_ramp_x.add_ramp(ramp_x=0.5, ramp_y=0.0, offset=0.0)
s_ramp_x.metadata.General.title = "X-direction ramp"

# Example 2: Add a linear ramp in Y direction only
s_ramp_y = s.deepcopy()
s_ramp_y.add_ramp(ramp_x=0.0, ramp_y=0.3, offset=0.0)
s_ramp_y.metadata.General.title = "Y-direction ramp"

# Example 3: Add ramps in both directions
s_ramp_xy = s.deepcopy()
s_ramp_xy.add_ramp(ramp_x=0.2, ramp_y=0.3, offset=0.1)
s_ramp_xy.metadata.General.title = "XY-direction ramp with offset"

# Example 4: Strong diagonal ramp
s_diagonal = s.deepcopy()
s_diagonal.add_ramp(ramp_x=0.8, ramp_y=0.8, offset=-0.5)
s_diagonal.metadata.General.title = "Strong diagonal ramp"

# Example 5: Show the ramp itself (no original signal)
s_ramp_only = hs.signals.Signal2D(np.zeros_like(data))
s_ramp_only.axes_manager[0].scale = 0.1
s_ramp_only.axes_manager[1].scale = 0.1
s_ramp_only.axes_manager[0].units = 'nm'
s_ramp_only.axes_manager[1].units = 'nm'
s_ramp_only.add_ramp(ramp_x=0.5, ramp_y=0.3, offset=0.2)
s_ramp_only.metadata.General.title = "Pure linear ramp"

# Use HyperSpy's plot_images for cleaner visualization
hs.plot.plot_images([s, s_ramp_x, s_ramp_y, s_ramp_xy, s_diagonal, s_ramp_only],
                   label=['Original Signal', 
                          'Linear Ramp in X\n(ramp_x=0.5)',
                          'Linear Ramp in Y\n(ramp_y=0.3)', 
                          'Ramp in X and Y\n(ramp_x=0.2, ramp_y=0.3, offset=0.1)',
                          'Strong Diagonal Ramp\n(ramp_x=0.8, ramp_y=0.8, offset=-0.5)',
                          'Pure Linear Ramp\n(no original signal)'],
                   cmap='viridis',
                   colorbar=True)

# Demonstrate understanding of scale and units
print("\nUnderstanding ramp parameters with scale:")
print(f"X-axis scale: {s.axes_manager[1].scale} {s.axes_manager[1].units}/pixel")
print(f"Y-axis scale: {s.axes_manager[0].scale} {s.axes_manager[0].units}/pixel")
print("\nThe ramp slopes are given in units per axis unit.")
print("For ramp_x=0.5 with x-scale=0.1 nm/pixel:")
print("- The slope is 0.5 intensity units per nm")
print("- This corresponds to 0.05 intensity units per pixel")

# Create profile plots to show the ramp effect using HyperSpy's native plotting

# X-profile through the center
center_y = s.data.shape[0] // 2
x_profile_original = s.isig[:, center_y]
x_profile_ramp = s_ramp_x.isig[:, center_y] 

x_profile_original.metadata.General.title = "X-direction Profile: Original"
x_profile_ramp.metadata.General.title = "X-direction Profile: With X-ramp"

print("\nPlotting X-direction profiles...")
x_profile_original.plot()
x_profile_ramp.plot()

# Y-profile through the center
center_x = s.data.shape[1] // 2
y_profile_original = s.isig[center_x, :]
y_profile_ramp = s_ramp_y.isig[center_x, :]

y_profile_original.metadata.General.title = "Y-direction Profile: Original"
y_profile_ramp.metadata.General.title = "Y-direction Profile: With Y-ramp"

print("Plotting Y-direction profiles...")
y_profile_original.plot()
y_profile_ramp.plot()

print("\nLinear ramp examples completed!")
print("The add_ramp method modifies the signal in-place.")
print("Use deepcopy() if you want to preserve the original signal.")
