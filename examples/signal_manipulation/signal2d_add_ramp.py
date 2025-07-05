"""
Adding Linear Ramps to Signal2D
===============================

This example demonstrates how to add linear ramps to 2D signals using
HyperSpy's add_ramp method. Linear ramps can be useful for simulating
drift, correcting illumination gradients, or creating test patterns.

"""

import numpy as np
import hyperspy.api as hs
import matplotlib.pyplot as plt

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
s.axes_manager[0].scale = 0.1  # Y-axis scale: 0.1 nm/pixel
s.axes_manager[1].scale = 0.1  # X-axis scale: 0.1 nm/pixel
s.axes_manager[0].units = 'nm'
s.axes_manager[1].units = 'nm'
s.axes_manager[0].name = 'y'
s.axes_manager[1].name = 'x'
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
# Display the original signal
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Plot 1: Original signal
im1 = axes[0].imshow(s.data, cmap='viridis')
axes[0].set_title('Original Signal')
axes[0].set_xlabel('X (pixels)')
axes[0].set_ylabel('Y (pixels)')
plt.colorbar(im1, ax=axes[0])

# %%
# Example 1: Add a linear ramp in X direction only
# -------------------------------------------------
s_ramp_x = s.deepcopy()
s_ramp_x.add_ramp(ramp_x=0.5, ramp_y=0.0, offset=0.0)
s_ramp_x.metadata.General.title = "X-direction ramp"

im2 = axes[1].imshow(s_ramp_x.data, cmap='viridis')
axes[1].set_title('Linear Ramp in X\n(ramp_x=0.5)')
axes[1].set_xlabel('X (pixels)')
axes[1].set_ylabel('Y (pixels)')
plt.colorbar(im2, ax=axes[1])

# %%
# Example 2: Add a linear ramp in Y direction only
# -------------------------------------------------
s_ramp_y = s.deepcopy()
s_ramp_y.add_ramp(ramp_x=0.0, ramp_y=0.3, offset=0.0)
s_ramp_y.metadata.General.title = "Y-direction ramp"

im3 = axes[2].imshow(s_ramp_y.data, cmap='viridis')
axes[2].set_title('Linear Ramp in Y\n(ramp_y=0.3)')
axes[2].set_xlabel('X (pixels)')
axes[2].set_ylabel('Y (pixels)')
plt.colorbar(im3, ax=axes[2])

# %%
# Example 3: Add ramps in both directions
# ----------------------------------------
s_ramp_xy = s.deepcopy()
s_ramp_xy.add_ramp(ramp_x=0.2, ramp_y=0.3, offset=0.1)
s_ramp_xy.metadata.General.title = "XY-direction ramp with offset"

im4 = axes[3].imshow(s_ramp_xy.data, cmap='viridis')
axes[3].set_title('Ramp in X and Y\n(ramp_x=0.2, ramp_y=0.3, offset=0.1)')
axes[3].set_xlabel('X (pixels)')
axes[3].set_ylabel('Y (pixels)')
plt.colorbar(im4, ax=axes[3])

# %%
# Example 4: Strong diagonal ramp
# --------------------------------
s_diagonal = s.deepcopy()
s_diagonal.add_ramp(ramp_x=0.8, ramp_y=0.8, offset=-0.5)
s_diagonal.metadata.General.title = "Strong diagonal ramp"

im5 = axes[4].imshow(s_diagonal.data, cmap='viridis')
axes[4].set_title('Strong Diagonal Ramp\n(ramp_x=0.8, ramp_y=0.8, offset=-0.5)')
axes[4].set_xlabel('X (pixels)')
axes[4].set_ylabel('Y (pixels)')
plt.colorbar(im5, ax=axes[4])

# %%
# Example 5: Show pure ramp (no original signal)
# -----------------------------------------------
s_ramp_only = hs.signals.Signal2D(np.zeros_like(data))
s_ramp_only.axes_manager[0].scale = 0.1
s_ramp_only.axes_manager[1].scale = 0.1
s_ramp_only.axes_manager[0].units = 'nm'
s_ramp_only.axes_manager[1].units = 'nm'
s_ramp_only.add_ramp(ramp_x=0.5, ramp_y=0.3, offset=0.2)

im6 = axes[5].imshow(s_ramp_only.data, cmap='viridis')
axes[5].set_title('Pure Linear Ramp\n(no original signal)')
axes[5].set_xlabel('X (pixels)')
axes[5].set_ylabel('Y (pixels)')
plt.colorbar(im6, ax=axes[5])

plt.tight_layout()
plt.show()

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

# Create a profile plot to show the ramp effect
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# X-profile through the center
center_y = s.data.shape[0] // 2
x_coords = np.arange(s.data.shape[1]) * s.axes_manager[1].scale

ax1.plot(x_coords, s.data[center_y, :], 'b-', label='Original', linewidth=2)
ax1.plot(x_coords, s_ramp_x.data[center_y, :], 'r--', label='With X-ramp', linewidth=2)
ax1.set_xlabel(f'X position ({s.axes_manager[1].units})')
ax1.set_ylabel('Intensity')
ax1.set_title('X-direction Profile (center row)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Y-profile through the center
center_x = s.data.shape[1] // 2
y_coords = np.arange(s.data.shape[0]) * s.axes_manager[0].scale

ax2.plot(y_coords, s.data[:, center_x], 'b-', label='Original', linewidth=2)
ax2.plot(y_coords, s_ramp_y.data[:, center_x], 'g--', label='With Y-ramp', linewidth=2)
ax2.set_xlabel(f'Y position ({s.axes_manager[0].units})')
ax2.set_ylabel('Intensity')
ax2.set_title('Y-direction Profile (center column)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# **Important notes for linear ramp operations**
#
# Understanding the behavior and requirements of the add_ramp method is crucial for effective use.

# **Key points:**
# - The add_ramp method modifies the signal **in-place**
# - Use deepcopy() if you want to preserve the original signal
# - Ramp parameters are specified in physical units, not pixels
# - Proper axis calibration is essential for meaningful ramp addition
