"""
Adding Linear Ramps to Signal2D
===============================

This example demonstrates how to add linear ramps to 2D signals using
HyperSpy's add_ramp method. Linear ramps can be useful for simulating
drift, correcting illumination gradients, or creating test patterns.

"""

import numpy as np
import hyperspy.api as hs

# %%
# Create a synthetic 2D signal
# -----------------------------
# Create a synthetic 2D signal - a simple Gaussian peak
size = 100
x = np.linspace(-3, 3, size)
y = np.linspace(-3, 3, size)
X, Y = np.meshgrid(x, y)

# Create a 2D Gaussian
data = np.exp(-(X**2 + Y**2) / 2)

# Create Signal2D with proper scale
s = hs.signals.Signal2D(data)
# Set axis properties efficiently using .set() method
s.axes_manager.signal_axes.set(
    name=['y', 'x'],
    units=['nm', 'nm'],
    scale=[0.1, 0.1]  # Y-axis and X-axis scale: 0.1 nm/pixel
)
s.metadata.General.title = "Original Gaussian signal"

# %%
# **Signal properties and axis information**
#
# The created signal has defined axis scales and units that affect how ramps are calculated.
# Understanding these properties is crucial for proper ramp parameter selection.

# Signal characteristics:
# - Shape: 64x64 pixels
# - X-axis scale: 0.1 nm/pixel  
# - Y-axis scale: 0.1 nm/pixel
# - Both axes in nanometer units

# %%
# Display examples of adding ramps
# ---------------------------------

# Example 1: Add a linear ramp in X direction only
# -------------------------------------------------
s_ramp_x = s.deepcopy()
s_ramp_x.add_ramp(ramp_x=0.5, ramp_y=0.0, offset=0.0)
s_ramp_x.metadata.General.title = "X-direction ramp"

# %%
# Example 2: Add a linear ramp in Y direction only
# -------------------------------------------------
s_ramp_y = s.deepcopy()
s_ramp_y.add_ramp(ramp_x=0.0, ramp_y=0.3, offset=0.0)
s_ramp_y.metadata.General.title = "Y-direction ramp"

# %%
# Example 3: Add ramps in both directions
# ----------------------------------------
s_ramp_xy = s.deepcopy()
s_ramp_xy.add_ramp(ramp_x=0.2, ramp_y=0.3, offset=0.1)
s_ramp_xy.metadata.General.title = "XY-direction ramp with offset"

# %%
# Example 4: Strong diagonal ramp
# --------------------------------
s_diagonal = s.deepcopy()
s_diagonal.add_ramp(ramp_x=0.8, ramp_y=0.8, offset=-0.5)
s_diagonal.metadata.General.title = "Strong diagonal ramp"

# %%
# Example 5: Show pure ramp (no original signal)
# -----------------------------------------------
s_ramp_only = hs.signals.Signal2D(np.zeros_like(data))
s_ramp_only.axes_manager.signal_axes.set(
    scale=[0.1, 0.1],
    units=['nm', 'nm']
)
s_ramp_only.add_ramp(ramp_x=0.5, ramp_y=0.3, offset=0.2)

# Display all ramp examples using plot_images
hs.plot.plot_images([s, s_ramp_x, s_ramp_y, s_ramp_xy, s_diagonal, s_ramp_only],
                    label=['Original Signal', 'Linear Ramp in X\n(ramp_x=0.5)', 
                           'Linear Ramp in Y\n(ramp_y=0.3)', 
                           'Ramp in X and Y\n(ramp_x=0.2, ramp_y=0.3, offset=0.1)',
                           'Strong Diagonal Ramp\n(ramp_x=0.8, ramp_y=0.8, offset=-0.5)',
                           'Pure Linear Ramp\n(no original signal)'],
                    per_row=3)

# %%
# **Understanding ramp parameters and axis scaling**
#
# Ramp slopes are specified in signal units per axis unit, not per pixel.
# This distinction is important for proper calibration and analysis.

# **Key concepts:**
# - X-axis scale: 0.1 nm/pixel
# - Y-axis scale: 0.1 nm/pixel  
# - Ramp slopes are given in intensity units per axis unit (nm)
#
# **Example calculation:**
# For ramp_x=0.5 with x-scale=0.1 nm/pixel:
# - The slope is 0.5 intensity units per nm
# - This corresponds to 0.05 intensity units per pixel

# Create profile plots to show the ramp effect
# Extract center row and column profiles  
center_y = s.data.shape[0] // 2
center_x = s.data.shape[1] // 2

# Create 1D signals from the profiles for easy plotting
x_profile_orig = hs.signals.Signal1D(s.data[center_y, :])
x_profile_ramp = hs.signals.Signal1D(s_ramp_x.data[center_y, :])
x_profile_orig.axes_manager.signal_axes.set(
    scale=[s.axes_manager[1].scale],
    units=[s.axes_manager[1].units],
    name=['X position']
)
x_profile_ramp.axes_manager = x_profile_orig.axes_manager.deepcopy()

y_profile_orig = hs.signals.Signal1D(s.data[:, center_x])
y_profile_ramp = hs.signals.Signal1D(s_ramp_y.data[:, center_x])
y_profile_orig.axes_manager.signal_axes.set(
    scale=[s.axes_manager[0].scale],
    units=[s.axes_manager[0].units],
    name=['Y position']
)
y_profile_ramp.axes_manager = y_profile_orig.axes_manager.deepcopy()

# Plot the profiles to show ramp effects
x_profile_orig.plot()
x_profile_ramp.plot()
y_profile_orig.plot()
y_profile_ramp.plot()

# %%
# **Important notes for linear ramp operations**
#
# Understanding the behavior and requirements of the add_ramp method is crucial for effective use.

# **Key points:**
# - The add_ramp method modifies the signal **in-place**
# - Use deepcopy() if you want to preserve the original signal
# - Ramp parameters are specified in physical units, not pixels
# - Proper axis calibration is essential for meaningful ramp addition
