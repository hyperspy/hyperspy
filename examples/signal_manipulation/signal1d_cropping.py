"""
Signal1D cropping methods
=========================

This example demonstrates different methods for cropping 1D signals in HyperSpy,
including the crop_signal() method, isig[] indexing, and interactive ROI cropping.
"""

# %%
# **Creating test signal for cropping demonstration**
#
# We'll use HyperSpy's built-in test data that contains two Gaussian peaks,
# perfect for demonstrating different cropping methods.

import numpy as np
import hyperspy.api as hs

# Create a signal with two Gaussians (built-in test data)
s = hs.data.two_gaussians()

# **Signal characteristics:**
# - Contains two Gaussian peaks in a multidimensional dataset
# - Navigation dimensions for spatial scanning or other parameters
# - Signal dimension for measured values (energy, wavelength, frequency, etc.)

# %%
# **Method 1: Using crop_signal() method**
#
# The `crop_signal()` method modifies the signal in-place, permanently changing the data.
# This is useful when you want to reduce memory usage and don't need the original range.

s1 = s.deepcopy()  # Make a copy to preserve original
s1.crop_signal(5, 15)
# Signal is now permanently cropped to the 5-15 range

# %%
# **Method 2: Using isig[] indexing**  
#
# The `isig[]` indexing creates a view of the data without modifying the original.
# This is non-destructive and allows you to work with subsets while preserving the full dataset.

s2 = s.isig[5.:15.]  # Creates a "cropped view" - original data unchanged
# This method is preferred for exploration and temporary analysis

# %%
# **Method 3: Cropping with step size**
#
# You can also specify a step size to subsample the signal while cropping.
# This is useful for reducing data density or matching different sampling rates.

s3 = s.isig[5.:15.:0.5]  # Every 0.5 units - both crops and subsamples
# Combines cropping with downsampling in a single operation

# %%
# Plot comparison of original and cropped signals
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Plot the signals using matplotlib directly
axes[0,0].plot(s.axes_manager.signal_axes[0].axis, s.data[16, 16])
axes[0,0].set_title('Original signal')
axes[0,0].set_xlabel('Energy')

axes[0,1].plot(s1.axes_manager.signal_axes[0].axis, s1.data[16, 16])
axes[0,1].set_title('crop_signal(5, 15)')
axes[0,1].set_xlabel('Energy')

axes[1,0].plot(s2.axes_manager.signal_axes[0].axis, s2.data[16, 16])
axes[1,0].set_title('isig[5.:15.]')
axes[1,0].set_xlabel('Energy')

axes[1,1].plot(s3.axes_manager.signal_axes[0].axis, s3.data[16, 16])
axes[1,1].set_title('isig[5.:15.:0.5]')
axes[1,1].set_xlabel('Energy')

plt.tight_layout()
plt.show()

# %%
# Method 4: Interactive cropping with ROI
# Create a SpanROI for interactive cropping
roi = hs.roi.SpanROI(left=5, right=15)

# Apply the ROI to get cropped signal
s_roi = roi(s)
print(f"\nROI cropped signal: {s_roi}")

# Plot the original signal and demonstrate ROI
s.plot()
print("Original signal plotted for interactive ROI demonstration")

# %%
# Demonstrate the difference between in-place and view operations
print("\n--- In-place vs View operations ---")
s_original = hs.data.two_gaussians()
print(f"Original data sum: {s_original.data.sum():.2f}")

# crop_signal modifies in-place
s_inplace = s_original.deepcopy()
s_inplace.crop_signal(5, 15)
print(f"After crop_signal: {s_inplace.data.sum():.2f}")

# isig creates a view (original unchanged)
s_view = s_original.isig[5.:15.]
print(f"Original after isig operation: {s_original.data.sum():.2f}")
print(f"View data sum: {s_view.data.sum():.2f}")
