"""
HyperSpy Advanced Features Demo
===============================

This example demonstrates corrected best practices based on comprehensive testing
of HyperSpy's capabilities with external libraries and advanced indexing.
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
s_3d.axes_manager.navigation_axes.set(
    name=['Y', 'X'],
    scale=[0.1, 0.2],
    units=['µm', 'µm']
)
s_3d.axes_manager.signal_axes.set(
    name=['Energy'],
    scale=[0.5],
    offset=[100],
    units=['eV']
)

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
# ### Summary of Best Practices
# 
# 1. **External libraries**: Try signal first, many work through __array__ protocol
# 2. **Indexing**: Use .inav[] and .isig[] with value-based coordinates when possible  
# 3. **Type conversion**: Use change_dtype() to preserve signal structure
# 4. **Axis operations**: Use semantic axis names for clarity
# 5. **Signal arithmetic**: NumPy math functions often preserve signal structure
