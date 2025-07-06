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

s.plot()
ax = s._plot.signal_plot.ax

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

ax.set_title('Integration Ranges and Results')
ax.legend()

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
s.plot()
s_roi.plot()

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

# Plot the results using HyperSpy's plot_images for the maps
# Sample spectrum from the middle of the image
si.inav[5, 5].plot()

# Mean and sum spectra
mean_spectrum = si.mean(axis=(0, 1))
sum_spectrum = si.sum(axis=(0, 1))

# Plot the spectra
mean_spectrum.plot()
sum_spectrum.plot()

# Use HyperSpy's plot_images for the intensity maps
hs.plot.plot_images([total_intensity_map, peak1_map, peak2_map],
                   label=['Total Intensity Map', 'Peak 1 Map (175-225 eV)', 'Peak 2 Map (475-525 eV)'],
                   cmap=['viridis', 'Reds', 'Blues'],
                   colorbar=True)

# %%
# Advanced integration: cumulative integration
# #############################################
#
# Sometimes we want to see how the integral accumulates along the signal axis

# Calculate cumulative integral
cumulative_integral = np.cumsum(s.data) * s.axes_manager.signal_axes[0].scale

# Plot original signal and cumulative integral
s.plot()
ax1 = s._plot.signal_plot.ax
ax1.set_title('Original Signal')

# Create cumulative integral signal for proper plotting
s_cumulative = s.deepcopy()
s_cumulative.data = cumulative_integral
s_cumulative.metadata.General.title = 'Cumulative Integration'
s_cumulative.axes_manager.signal_axes[0].name = 'Energy'
s_cumulative.axes_manager.signal_axes[0].units = 'eV'

# %%
# Integration Summary
# ==================
#
# Review of the different integration methods and their results

print(f"Total signal integral: {float(total_integral.data):.2f}")
print(f"Peak 1 (150-250 eV): {float(range_integral_1.data):.2f}")
print(f"Peak 2 (450-550 eV): {float(range_integral_2.data):.2f}")
print(f"Peak 3 (700-800 eV): {float(range_integral_3.data):.2f}")
print(f"ROI region (400-600 eV): {float(roi_integral.data):.2f}")
print(f"Final cumulative value: {cumulative_integral[-1]:.2f}")

print("Integration methods available:")
print("- integrate1D(): Integrate over specified axis")
print("- isig[start:end].integrate1D(): Integrate over energy range")
print("- ROI.integrate1D(): Interactive range selection")
print("- Cumulative integration for progressive totals")
