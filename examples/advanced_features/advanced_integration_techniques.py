"""
Advanced Integration Techniques
===============================

This example demonstrates advanced techniques for integrating HyperSpy with
external libraries, sophisticated indexing patterns, and best practices for
signal arithmetic and data manipulation.
"""

# %%
# ## Signal Arithmetic and External Library Integration
# 
# Many external libraries work directly with HyperSpy signals through the __array__ protocol

import hyperspy.api as hs
import numpy as np
from scipy import ndimage

# Create test signal
s = hs.signals.Signal2D(np.random.rand(20, 20) + 1)
s.metadata.General.title = 'Test Image'
# Configure axes using efficient .set() method
s.axes_manager.signal_axes.set(
    name=['Y', 'X'],
    units=['nm', 'nm'],
    scale=[0.1, 0.1]
)

# %%
# ### External libraries work directly with signals
# 
# Most external libraries work with signals through __array__ protocol (return numpy arrays)

# SciPy functions work directly with signals
filtered = ndimage.gaussian_filter(s, sigma=2.0)    # Returns numpy array
edges = ndimage.sobel(s)                           # Returns numpy array

# NumPy mathematical functions preserve signal structure  
log_signal = np.log(s + 0.1)                      # Returns HyperSpy signal!
sqrt_signal = np.sqrt(s)                           # Returns HyperSpy signal!

print(f"Original signal type: {type(s)}")
print(f"SciPy result type: {type(filtered)}")
print(f"NumPy math result type: {type(log_signal)}")
print(f"Log signal title: {log_signal.metadata.General.title}")

# %%
# ## Advanced HyperSpy Indexing
# 
# HyperSpy indexing is more powerful than array slicing and preserves metadata

# Create 3D signal for indexing demonstration
s_3d = hs.signals.Signal1D(np.random.randn(8, 12, 50))
s_3d.metadata.General.title = 'Spectrum Image'
# Configure axes using direct property assignment for all axes
s_3d.axes_manager.navigation_axes[0].name = 'Y'
s_3d.axes_manager.navigation_axes[0].scale = 0.1
s_3d.axes_manager.navigation_axes[0].units = 'µm'

s_3d.axes_manager.navigation_axes[1].name = 'X'
s_3d.axes_manager.navigation_axes[1].scale = 0.2
s_3d.axes_manager.navigation_axes[1].units = 'µm'

s_3d.axes_manager.signal_axes[0].name = 'Energy'
s_3d.axes_manager.signal_axes[0].scale = 0.5
s_3d.axes_manager.signal_axes[0].offset = 100
s_3d.axes_manager.signal_axes[0].units = 'eV'

# %%
# ### Value-based indexing (use physical coordinates)

# Index by actual axis values, not pixel indices
spatial_roi = s_3d.inav[0.2:0.6, 0.4:1.2]          # Physical coordinates in µm
energy_window = s_3d.isig[110.:130.]                # Energy range in eV
combined_roi = s_3d.inav[0.3:0.5, :].isig[105.:125.]  # Both spatial and spectral

print(f"Original shape: {s_3d.data.shape}")
print(f"Spatial ROI shape: {spatial_roi.data.shape}")
print(f"Energy window shape: {energy_window.data.shape}")
print(f"Combined ROI shape: {combined_roi.data.shape}")

# All indexing preserves signal structure and metadata
print(f"ROI signal type: {type(spatial_roi)}")
print(f"ROI title preserved: {spatial_roi.metadata.General.title}")

# %%
# ### Axis name-based operations (semantic clarity)

# Use axis names instead of magic numbers
intensity_map = s_3d.max(axis='Energy')             # Clear intent
mean_spectrum = s_3d.mean(axis=('X', 'Y'))          # Average over spatial dimensions
spatial_variance = s_3d.std(axis=('X', 'Y'))       # Variance across space

print(f"Intensity map shape: {intensity_map.data.shape}")
print(f"Mean spectrum shape: {mean_spectrum.data.shape}")

# %%
# ## Type Conversion with change_dtype
# 
# Use HyperSpy's change_dtype instead of manual .astype() operations

# Create signal for type conversion demo
s_float = hs.signals.Signal2D(np.random.randn(10, 10).astype(np.float64))
s_float.metadata.General.title = 'Original Float64 Signal'

print(f"Original dtype: {s_float.data.dtype}")

# Use HyperSpy's change_dtype (preserves everything)
s_converted = s_float.deepcopy()
s_converted.change_dtype(np.float32)

print(f"After change_dtype: {s_converted.data.dtype}")
print(f"Metadata preserved: {s_converted.metadata.General.title}")
print(f"Signal type preserved: {type(s_converted).__name__}")

# %%
# ## Visualization of Results
# 
# Let's visualize the effects of our integration techniques

# Create a more structured signal for visualization
np.random.seed(42)  # For reproducible results
x, y = np.mgrid[0:10:0.5, 0:10:0.5]
center_x, center_y = 5, 5
gaussian_2d = np.exp(-((x - center_x)**2 + (y - center_y)**2) / 8) 
noise = 0.1 * np.random.random(gaussian_2d.shape)
demo_signal = hs.signals.Signal2D(gaussian_2d + noise)

demo_signal.axes_manager.signal_axes.set(
    name=['Y', 'X'],
    units=['µm', 'µm'],
    scale=[0.5, 0.5],
    offset=[0, 0]
)
demo_signal.metadata.General.title = 'Test Gaussian with Noise'

# Apply external library processing
filtered_demo = ndimage.gaussian_filter(demo_signal, sigma=1.0)
edges_demo = ndimage.sobel(demo_signal)

# Convert numpy results back to HyperSpy signals for proper visualization
filtered_signal = hs.signals.Signal2D(filtered_demo)
filtered_signal.axes_manager = demo_signal.axes_manager.deepcopy()
filtered_signal.metadata.General.title = 'Gaussian Filtered'

edges_signal = hs.signals.Signal2D(edges_demo)
edges_signal.axes_manager = demo_signal.axes_manager.deepcopy()
edges_signal.metadata.General.title = 'Edge Detection (Sobel)'

# Create visualization comparing original, filtered, and edge detection
hs.plot.plot_images([demo_signal, filtered_signal, edges_signal], 
                    tight_layout=True, axes_decor='all')

# %%
# ### Visualization of ROI Operations

# Create a spectrum image for ROI demonstration
spectrum_data = np.zeros((16, 16, 100))
for i in range(16):
    for j in range(16):
        # Create position-dependent spectra
        peak_pos = 40 + 5 * np.sin(2 * np.pi * i / 16) + 3 * np.cos(2 * np.pi * j / 16)
        energy_axis = np.arange(100)
        spectrum = 1000 * np.exp(-((energy_axis - peak_pos) / 8)**2) + 50 * np.random.random(100)
        spectrum_data[i, j, :] = spectrum

roi_demo_signal = hs.signals.Signal1D(spectrum_data)
roi_demo_signal.axes_manager.navigation_axes.set(
    name=['Y', 'X'],
    units=['µm', 'µm'], 
    scale=[0.1, 0.1]
)
roi_demo_signal.axes_manager.signal_axes[0].name = 'Energy'
roi_demo_signal.axes_manager.signal_axes[0].units = 'eV'
roi_demo_signal.axes_manager.signal_axes[0].scale = 0.5
roi_demo_signal.axes_manager.signal_axes[0].offset = 100

# Extract ROI and visualize
roi_extracted = roi_demo_signal.inav[0.3:1.2, 0.4:1.0]
mean_spectrum = roi_extracted.mean(axis=('X', 'Y'))
intensity_map = roi_demo_signal.max(axis='Energy')

# Plot navigation image and mean spectrum
roi_demo_signal.plot()
intensity_map.plot()

# %%
# ### Summary of Best Practices
# 
# 1. **External libraries**: Try signal first, many work through __array__ protocol
# 2. **Indexing**: Use .inav[] and .isig[] with value-based coordinates when possible  
# 3. **Type conversion**: Use change_dtype() to preserve signal structure
# 4. **Axis operations**: Use semantic axis names for clarity
# 5. **Signal arithmetic**: NumPy math functions often preserve signal structure
# 6. **Visualization**: Always visualize results to verify processing steps
