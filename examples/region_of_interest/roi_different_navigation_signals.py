"""
Advanced ROI Examples with Different Navigation Signals
=======================================================

This example demonstrates advanced Region of Interest (ROI) operations 
with different types of navigation signals and complex ROI manipulations.
"""

import numpy as np
import hyperspy.api as hs

# %%
# ## Creating test signals for advanced ROI operations
# 
# We'll create synthetic spectrum images with spatial and spectral variation

# %%
# Create 3D spectrum image (2D navigation, 1D signal)

# Create synthetic spectrum image data
nav_size_x, nav_size_y = 20, 15
sig_size = 100

# Create base spectrum with peaks at different positions
energy_axis = np.linspace(0, 10, sig_size)
base_spectrum = np.exp(-(energy_axis - 5)**2 / 2) + 0.5 * np.exp(-(energy_axis - 7)**2 / 1)

# Create 3D data with spatial variation
data_3d = np.zeros((nav_size_y, nav_size_x, sig_size))
for y in range(nav_size_y):
    for x in range(nav_size_x):
        # Add spatial variation to peak positions and intensities
        peak1_shift = 0.2 * np.sin(2 * np.pi * x / nav_size_x)
        peak2_intensity = 0.5 + 0.3 * np.cos(2 * np.pi * y / nav_size_y)
        
        spectrum = (np.exp(-((energy_axis - 5 - peak1_shift)**2) / 2) + 
                   peak2_intensity * np.exp(-((energy_axis - 7)**2) / 1) +
                   np.random.random(sig_size) * 0.1)
        data_3d[y, x, :] = spectrum

# Create HyperSpy signal
spectrum_image = hs.signals.Signal1D(data_3d)
# Set navigation axes (spatial dimensions) using batch setting
spectrum_image.axes_manager.navigation_axes.set(
    name=['y', 'x'],
    units=['nm', 'nm'],
    scale=[1.0, 1.0]
)
# Set signal axis (energy dimension)
spectrum_image.axes_manager.signal_axes.set(
    name=['Energy'],
    units=['eV'],
    scale=[0.1]
)
spectrum_image.metadata.General.title = "Synthetic Spectrum Image"

# Display information about the created signal
print(f"Created spectrum image with shape {spectrum_image.data.shape}")

# %%
# ## Example 1: Multiple ROIs on Spectrum Image
# 
# We can apply multiple spatial ROIs to extract different regions of the spectrum image

# Create rectangular ROI for spatial selection
spatial_roi1 = hs.roi.RectangularROI(left=5, right=10, top=5, bottom=10)
spatial_roi2 = hs.roi.RectangularROI(left=12, right=17, top=8, bottom=13)

# Apply ROIs to get sub-regions
region1 = spatial_roi1(spectrum_image, axes=['x', 'y'])
region2 = spatial_roi2(spectrum_image, axes=['x', 'y'])

# Results of ROI application
print(f"Region 1 shape: {region1.data.shape}")
print(f"Region 2 shape: {region2.data.shape}")

# Get mean spectra from each region
mean_spectrum1 = region1.mean(axis=(0, 1))
mean_spectrum2 = region2.mean(axis=(0, 1))

mean_spectrum1.metadata.General.title = "Mean Spectrum - Region 1"
mean_spectrum2.metadata.General.title = "Mean Spectrum - Region 2"

# Summary of spatial ROI operations
print("Extracted mean spectra from different spatial regions")

# %%
# ## Example 2: Energy ROI for Peak Analysis
# 
# Energy ROIs allow us to extract specific spectral ranges for detailed analysis

# Create energy ROI around the first peak
energy_roi1 = hs.roi.SpanROI(left=4.5, right=5.5)
peak1_map = energy_roi1(spectrum_image, axes=['Energy'])

# Create energy ROI around the second peak  
energy_roi2 = hs.roi.SpanROI(left=6.5, right=7.5)
peak2_map = energy_roi2(spectrum_image, axes=['Energy'])

# Integrate over energy to create peak intensity maps
peak1_intensity = peak1_map.sum(axis=-1)
peak2_intensity = peak2_map.sum(axis=-1)

peak1_intensity.metadata.General.title = "Peak 1 Intensity Map"
peak2_intensity.metadata.General.title = "Peak 2 Intensity Map"

print("Created peak intensity maps using energy ROIs")
print(f"Peak 1 map shape: {peak1_intensity.data.shape}")
print(f"Peak 2 map shape: {peak2_intensity.data.shape}")

# %%
# Summary and best practices
#
# Advanced ROI operations demonstrated:
# 1. Multiple spatial ROIs for region comparison  
# 2. Energy ROIs for peak analysis and mapping
#
# Best practices:
# - Choose ROI type based on your analysis needs
# - Use appropriate axes specification
# - Consider ROI size effects on statistics

print("Advanced ROI Examples Complete")
