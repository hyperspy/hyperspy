"""
Signal1D Integration
====================

This example demonstrates various methods for integrating 1D signals in HyperSpy.
Integration is useful for calculating areas under curves, total intensities,
and reducing dimensionality of data.
"""

# %%
# Create sample data for integration
# ##################################
#
# We'll create various signals to demonstrate different integration scenarios

import numpy as np
import hyperspy.api as hs
import matplotlib.pyplot as plt

# Create an energy axis
energy = np.linspace(0, 1000, 1000)

# Create a signal with multiple peaks
peak1 = 1000 * np.exp(-((energy - 200)**2) / (2 * 30**2))
peak2 = 800 * np.exp(-((energy - 500)**2) / (2 * 50**2))
peak3 = 600 * np.exp(-((energy - 750)**2) / (2 * 40**2))

# Add a smooth background
background = 100 + 200 * np.exp(-energy / 500)

# Combine signal components
signal_data = background + peak1 + peak2 + peak3

# Create HyperSpy signal
s = hs.signals.Signal1D(signal_data)
s.axes_manager.signal_axes[0].name = 'Energy'
s.axes_manager.signal_axes[0].units = 'eV'
s.axes_manager.signal_axes[0].scale = 1.0
s.axes_manager.signal_axes[0].offset = 0.0
s.metadata.General.title = 'Multi-peak Spectrum'

# Plot the original signal
s.plot()
plt.title('Original Signal for Integration')

# %%
# Basic integration over the entire signal
# #########################################
#
# The simplest integration sums over the entire signal axis

# Integrate the entire signal
total_integral = s.integrate1D(axis=0)
print(f"Total integral (entire signal): {float(total_integral.data):.2f}")

# %%
# Integration over a specific range using isig
# #############################################
#
# Use the isig indexing to select a range before integration

# Integrate over a specific energy range (around the first peak)
range_integral_1 = s.isig[150:250].integrate1D(axis=0)
print(f"Integral from 150-250 eV: {float(range_integral_1.data):.2f}")

# Integrate over the second peak
range_integral_2 = s.isig[450:550].integrate1D(axis=0)
print(f"Integral from 450-550 eV: {float(range_integral_2.data):.2f}")

# Integrate over the third peak
range_integral_3 = s.isig[700:800].integrate1D(axis=0)
print(f"Integral from 700-800 eV: {float(range_integral_3.data):.2f}")

# %%
# Visualizing integration ranges
# ##############################
#
# Let's visualize the different integration ranges on the spectrum

fig, ax = plt.subplots(figsize=(12, 6))

# Plot the full spectrum
ax.plot(s.axes_manager.signal_axes[0].axis, s.data, 'b-', linewidth=2, label='Full spectrum')

# Highlight integration ranges
energy_axis = s.axes_manager.signal_axes[0].axis

# Range 1 (150-250 eV)
mask1 = (energy_axis >= 150) & (energy_axis <= 250)
ax.fill_between(energy_axis[mask1], 0, s.data[mask1], alpha=0.3, color='red', 
                label=f'Range 1: {float(range_integral_1.data):.0f}')

# Range 2 (450-550 eV)
mask2 = (energy_axis >= 450) & (energy_axis <= 550)
ax.fill_between(energy_axis[mask2], 0, s.data[mask2], alpha=0.3, color='green', 
                label=f'Range 2: {float(range_integral_2.data):.0f}')

# Range 3 (700-800 eV)
mask3 = (energy_axis >= 700) & (energy_axis <= 800)
ax.fill_between(energy_axis[mask3], 0, s.data[mask3], alpha=0.3, color='orange', 
                label=f'Range 3: {float(range_integral_3.data):.0f}')

ax.set_xlabel('Energy (eV)')
ax.set_ylabel('Intensity')
ax.set_title('Integration Ranges and Results')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()

# %%
# Integration with ROI for interactive selection
# ###############################################
#
# Using ROI allows for interactive selection of integration ranges

# Create a span ROI for interactive range selection
roi = hs.roi.SpanROI(left=400, right=600)

# Apply ROI to extract the region of interest
s_roi = roi(s)

# Integrate the ROI region
roi_integral = s_roi.integrate1D(axis=0)
print(f"ROI integral (400-600 eV): {float(roi_integral.data):.2f}")

# Visualize the ROI selection
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Original signal with ROI bounds
ax1.plot(s.axes_manager.signal_axes[0].axis, s.data, 'b-', linewidth=2)
ax1.axvline(x=400, color='red', linestyle='--', label='ROI left bound')
ax1.axvline(x=600, color='red', linestyle='--', label='ROI right bound')
ax1.fill_between(s.axes_manager.signal_axes[0].axis, 0, s.data, 
                 where=((s.axes_manager.signal_axes[0].axis >= 400) & 
                        (s.axes_manager.signal_axes[0].axis <= 600)),
                 alpha=0.3, color='red', label='ROI region')
ax1.set_xlabel('Energy (eV)')
ax1.set_ylabel('Intensity')
ax1.set_title('Original Signal with ROI')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Extracted ROI region
ax2.plot(s_roi.axes_manager.signal_axes[0].axis, s_roi.data, 'r-', linewidth=2)
ax2.set_xlabel('Energy (eV)')
ax2.set_ylabel('Intensity')
ax2.set_title(f'ROI Region (Integral: {float(roi_integral.data):.0f})')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Integration of spectrum images (2D navigation)
# ###############################################
#
# Create a spectrum image to demonstrate integration across navigation dimensions

# Create a 2D spectrum image (spatial map of spectra)
np.random.seed(42)
spectrum_image_data = np.zeros((10, 10, 1000))

# Create varying spectra across the image
for i in range(10):
    for j in range(10):
        # Vary peak positions and intensities spatially
        peak1_pos = 200 + 20 * np.sin(i * 0.5) * np.cos(j * 0.5)
        peak2_pos = 500 + 30 * np.cos(i * 0.3) * np.sin(j * 0.3)
        
        peak1_int = 800 + 200 * np.random.random()
        peak2_int = 600 + 150 * np.random.random()
        
        local_peak1 = peak1_int * np.exp(-((energy - peak1_pos)**2) / (2 * 35**2))
        local_peak2 = peak2_int * np.exp(-((energy - peak2_pos)**2) / (2 * 45**2))
        local_bg = 50 + 100 * np.exp(-energy / 600)
        
        spectrum_image_data[i, j, :] = local_bg + local_peak1 + local_peak2

# Create spectrum image signal
si = hs.signals.Signal1D(spectrum_image_data)
si.axes_manager.signal_axes[0].name = 'Energy'
si.axes_manager.signal_axes[0].units = 'eV'
si.axes_manager.signal_axes[0].scale = 1.0
si.axes_manager.signal_axes[0].offset = 0.0
si.axes_manager.navigation_axes[0].name = 'x'
si.axes_manager.navigation_axes[1].name = 'y'
si.metadata.General.title = 'Spectrum Image'

# Integrate the entire spectrum at each pixel to create a total intensity map
total_intensity_map = si.integrate1D(axis=-1)  # -1 refers to the signal axis

# Integrate specific energy ranges to create elemental maps
peak1_map = si.isig[175:225].integrate1D(axis=-1)  # Around first peak
peak2_map = si.isig[475:525].integrate1D(axis=-1)  # Around second peak

# Plot the results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Sample spectrum from the middle of the image
axes[0, 0].plot(si.axes_manager.signal_axes[0].axis, si.inav[5, 5].data)
axes[0, 0].set_title('Sample Spectrum (pixel 5,5)')
axes[0, 0].set_xlabel('Energy (eV)')
axes[0, 0].set_ylabel('Intensity')
axes[0, 0].grid(True, alpha=0.3)

# Mean spectrum from entire image
mean_spectrum = si.mean(axis=(0, 1))
axes[0, 1].plot(mean_spectrum.axes_manager.signal_axes[0].axis, mean_spectrum.data)
axes[0, 1].set_title('Mean Spectrum')
axes[0, 1].set_xlabel('Energy (eV)')
axes[0, 1].set_ylabel('Intensity')
axes[0, 1].grid(True, alpha=0.3)

# Sum spectrum (total signal)
sum_spectrum = si.sum(axis=(0, 1))
axes[0, 2].plot(sum_spectrum.axes_manager.signal_axes[0].axis, sum_spectrum.data)
axes[0, 2].set_title('Sum Spectrum')
axes[0, 2].set_xlabel('Energy (eV)')
axes[0, 2].set_ylabel('Intensity')
axes[0, 2].grid(True, alpha=0.3)

# Total intensity map
im1 = axes[1, 0].imshow(total_intensity_map.data, cmap='viridis')
axes[1, 0].set_title('Total Intensity Map')
plt.colorbar(im1, ax=axes[1, 0])

# Peak 1 intensity map
im2 = axes[1, 1].imshow(peak1_map.data, cmap='Reds')
axes[1, 1].set_title('Peak 1 Map (175-225 eV)')
plt.colorbar(im2, ax=axes[1, 1])

# Peak 2 intensity map
im3 = axes[1, 2].imshow(peak2_map.data, cmap='Blues')
axes[1, 2].set_title('Peak 2 Map (475-525 eV)')
plt.colorbar(im3, ax=axes[1, 2])

plt.tight_layout()
plt.show()

# %%
# Advanced integration: cumulative integration
# #############################################
#
# Sometimes we want to see how the integral accumulates along the signal axis

# Calculate cumulative integral
cumulative_integral = np.cumsum(s.data) * s.axes_manager.signal_axes[0].scale

# Plot original signal and cumulative integral
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Original signal
ax1.plot(s.axes_manager.signal_axes[0].axis, s.data, 'b-', linewidth=2)
ax1.set_ylabel('Intensity')
ax1.set_title('Original Signal')
ax1.grid(True, alpha=0.3)

# Cumulative integral
ax2.plot(s.axes_manager.signal_axes[0].axis, cumulative_integral, 'r-', linewidth=2)
ax2.set_xlabel('Energy (eV)')
ax2.set_ylabel('Cumulative Integral')
ax2.set_title('Cumulative Integration')
ax2.grid(True, alpha=0.3)

# Add vertical lines at integration boundaries from earlier
for boundary, color in [(150, 'green'), (250, 'green'), (450, 'orange'), 
                        (550, 'orange'), (700, 'purple'), (800, 'purple')]:
    ax1.axvline(x=boundary, color=color, linestyle='--', alpha=0.7)
    ax2.axvline(x=boundary, color=color, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

print("\nIntegration Summary:")
print("==================")
print(f"Total signal integral: {float(total_integral.data):.2f}")
print(f"Peak 1 (150-250 eV): {float(range_integral_1.data):.2f}")
print(f"Peak 2 (450-550 eV): {float(range_integral_2.data):.2f}")
print(f"Peak 3 (700-800 eV): {float(range_integral_3.data):.2f}")
print(f"ROI region (400-600 eV): {float(roi_integral.data):.2f}")
print(f"Final cumulative value: {cumulative_integral[-1]:.2f}")

print("\nIntegration methods available:")
print("- integrate1D(): Integrate over specified axis")
print("- isig[start:end].integrate1D(): Integrate over energy range")
print("- ROI.integrate1D(): Interactive range selection")
print("- Cumulative integration for progressive totals")
