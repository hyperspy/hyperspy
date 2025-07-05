"""
Background Removal
==================

This example demonstrates how to remove backgrounds from 1D signals using the
:meth:`~.api.signals.Signal1D.remove_background` method. Various background types
are available including Power Law, Polynomial, Gaussian, and others.

The method provides both programmatic and interactive interfaces for background
estimation and removal.
"""

# %%
# Create a signal with a power law background
# ############################################
#
# We'll start by creating a synthetic signal with peaks on a power law background

import numpy as np
import hyperspy.api as hs

# Create energy axis
energy = np.arange(100, 1000, 2)

# Create a power law background
background = 1e6 * energy.astype(float)**(-3)

# Add some Gaussian peaks
peaks = (1000 * np.exp(-((energy - 300)**2) / (2 * 30**2)) +
         800 * np.exp(-((energy - 500)**2) / (2 * 50**2)) +
         600 * np.exp(-((energy - 700)**2) / (2 * 40**2)))

# Combine signal and background, add some noise
signal_data = background + peaks + np.random.normal(0, 50, len(energy))

# Create HyperSpy signal
s = hs.signals.Signal1D(signal_data)
s.axes_manager.signal_axes[0].name = 'Energy'
s.axes_manager.signal_axes[0].units = 'eV'
s.axes_manager.signal_axes[0].scale = 2.0
s.axes_manager.signal_axes[0].offset = 100.0
s.metadata.General.title = 'Spectrum with Power Law Background'

# Plot the original signal
s.plot()

# %%
# Remove power law background (default method)
# #############################################
#
# The default background type is 'PowerLaw', which is commonly used
# for analytical spectral data.

# Remove background non-interactively
# We'll specify the range for background estimation
s_no_bg = s.remove_background(
    signal_range=(400., 900.),  # Range for background estimation
    background_type='Power law',  # Note: use 'Power law' not 'PowerLaw'
    fast=True,  # Use fast analytical approximation
    plot_remainder=True  # Show the result
)

# %%
# Compare different background types
# ##################################
#
# Let's compare different background types on the same signal

# Create a subplot to compare different background removal methods
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Comparison of Background Removal Methods', fontsize=14)

# Original signal
axes[0, 0].plot(s.axes_manager.signal_axes[0].axis, s.data)
axes[0, 0].set_title('Original Signal')
axes[0, 0].set_xlabel('Energy (eV)')
axes[0, 0].set_ylabel('Intensity')

# Power law background removal
s_powerlaw = s.remove_background(
    signal_range=(400., 900.),
    background_type='Power law',
    fast=True
)
axes[0, 1].plot(s.axes_manager.signal_axes[0].axis, s_powerlaw.data)
axes[0, 1].set_title('Power Law Background Removed')
axes[0, 1].set_xlabel('Energy (eV)')
axes[0, 1].set_ylabel('Intensity')

# Polynomial background removal
s_poly = s.remove_background(
    signal_range=(400., 900.),
    background_type='Polynomial',
    polynomial_order=3,
    fast=True
)
axes[1, 0].plot(s.axes_manager.signal_axes[0].axis, s_poly.data)
axes[1, 0].set_title('Polynomial Background Removed')
axes[1, 0].set_xlabel('Energy (eV)')
axes[1, 0].set_ylabel('Intensity')

# Gaussian background removal (more stable than exponential)
s_gauss = s.remove_background(
    signal_range=(400., 900.),
    background_type='Gaussian',
    fast=True
)
axes[1, 1].plot(s.axes_manager.signal_axes[0].axis, s_gauss.data)
axes[1, 1].set_title('Gaussian Background Removed')
axes[1, 1].set_xlabel('Energy (eV)')
axes[1, 1].set_ylabel('Intensity')

plt.tight_layout()
plt.show()

# %%
# Using precise fitting (fast=False)
# ###################################
#
# For better accuracy, we can use curve fitting instead of analytical approximation

# More accurate fitting with curve fitting
s_accurate = s.remove_background(
    signal_range=(400., 900.),
    background_type='Power law',
    fast=False  # Use curve fitting for better accuracy
)

# Compare fast vs accurate fitting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(s.axes_manager.signal_axes[0].axis, s_no_bg.data, label='Fast (analytical)')
ax1.plot(s.axes_manager.signal_axes[0].axis, s_accurate.data, label='Accurate (fitted)')
ax1.set_title('Fast vs Accurate Background Removal')
ax1.set_xlabel('Energy (eV)')
ax1.set_ylabel('Intensity')
ax1.legend()

# Show the background that was removed
background_fast = s - s_no_bg
background_accurate = s - s_accurate

ax2.plot(s.axes_manager.signal_axes[0].axis, background_fast.data, 
         label='Fast background', linestyle='--')
ax2.plot(s.axes_manager.signal_axes[0].axis, background_accurate.data, 
         label='Accurate background', linestyle='-')
ax2.set_title('Estimated Backgrounds')
ax2.set_xlabel('Energy (eV)')
ax2.set_ylabel('Intensity')
ax2.legend()

plt.tight_layout()
plt.show()

# %%
# Working with spectrum images
# ############################
#
# The background removal also works on spectrum images (3D datasets)

# Create a simple spectrum image (2D navigation, 1D signal)
np.random.seed(42)
spectrum_image_data = np.zeros((5, 5, len(energy)))

# Add different backgrounds to each spectrum
for i in range(5):
    for j in range(5):
        # Vary the background parameters across the image
        bg_amplitude = 1e5 * (1 + 0.5 * np.random.random())
        bg_power = -2.5 - 0.5 * np.random.random()
        local_background = bg_amplitude * energy.astype(float)**bg_power
        
        # Add the same peaks but with varying intensities
        peak_scale = 0.5 + np.random.random()
        local_peaks = peak_scale * peaks
        
        # Combine and add noise
        spectrum_image_data[i, j, :] = (local_background + local_peaks + 
                                       np.random.normal(0, 30, len(energy)))

# Create spectrum image signal
si = hs.signals.Signal1D(spectrum_image_data)
si.axes_manager.signal_axes[0].name = 'Energy'
si.axes_manager.signal_axes[0].units = 'eV'
si.axes_manager.signal_axes[0].scale = 2.0
si.axes_manager.signal_axes[0].offset = 100.0
si.metadata.General.title = 'Spectrum Image with Power Law Backgrounds'

# Remove background from all spectra
si_no_bg = si.remove_background(
    signal_range=(400., 900.),
    background_type='Power law',
    fast=True
)

# Plot a comparison for one spectrum
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(si.axes_manager.signal_axes[0].axis, si.inav[2, 2].data, 
         label='Original')
plt.plot(si.axes_manager.signal_axes[0].axis, si_no_bg.inav[2, 2].data, 
         label='Background removed')
plt.xlabel('Energy (eV)')
plt.ylabel('Intensity')
plt.title('Single Spectrum from Image')
plt.legend()

# Show the mean spectrum from the entire image
plt.subplot(1, 2, 2)
mean_original = si.mean(axis=(0, 1))
mean_no_bg = si_no_bg.mean(axis=(0, 1))
plt.plot(mean_original.axes_manager.signal_axes[0].axis, mean_original.data, 
         label='Original (mean)')
plt.plot(mean_no_bg.axes_manager.signal_axes[0].axis, mean_no_bg.data, 
         label='Background removed (mean)')
plt.xlabel('Energy (eV)')
plt.ylabel('Intensity')
plt.title('Mean Spectrum from Image')
plt.legend()

plt.tight_layout()
plt.show()

# %%
# **Background removal summary and available models**
#
# Background removal is essential for quantitative analysis across many analytical techniques.

# **Available background types for different applications:**
# 
# **Common models:**
# - **Power law**: Default choice, excellent for many types of continuum background
# - **Polynomial**: Flexible, good for slowly varying backgrounds
# - **Exponential**: For exponentially decaying backgrounds
# - **Offset**: Simple constant background
#
# **Peak-like models (for complex backgrounds):**
# - **Gaussian**: Symmetric peak-like backgrounds
# - **Lorentzian**: Peak with broader tails
# - **Voigt**: Combination of Gaussian and Lorentzian
# - **Split Voigt**: Asymmetric Voigt profile
# - **Skew normal**: Asymmetric Gaussian-like background
# - **Doniach**: Asymmetric peak model (common in XPS)
