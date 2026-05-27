"""
Multidimensional Data Handling
===============================

This example demonstrates advanced techniques for handling complex multidimensional
datasets in HyperSpy, including axis manipulation, data reshaping, and efficient
analysis workflows for high-dimensional scientific data.
"""

import hyperspy.api as hs
import numpy as np

# %%
# Create complex multidimensional dataset
# ----------------------------------------
# Simulate a 5D dataset: temperature series of 4D-STEM data
# (temperature, scan_y, scan_x, detector_y, detector_x)

print("Creating 5D dataset simulation...")
n_temp, scan_y, scan_x, det_y, det_x = 6, 16, 16, 32, 32  # Reduced for faster computation
data_5d = np.random.random((n_temp, scan_y, scan_x, det_y, det_x))

# Add temperature-dependent structure
temperatures = np.linspace(300, 800, n_temp)  # Kelvin
for t_idx, temp in enumerate(temperatures):
    # Simulate temperature-dependent diffraction pattern changes
    center_shift = int(5 * (temp - 300) / 500)  # Center shifts with temperature
    for sy in range(scan_y):
        for sx in range(scan_x):
            # Add diffraction spots that shift with temperature
            y_center, x_center = det_y//2 + center_shift, det_x//2 + center_shift
            y, x = np.ogrid[:det_y, :det_x]
            mask = (y - y_center)**2 + (x - x_center)**2 < 100
            data_5d[t_idx, sy, sx, mask] += 1000 * np.exp(-(temp - 550)**2 / 10000)

# %%
# ## Initial Signal Creation and Axis Configuration
# 
# HyperSpy interprets the last 2 dimensions as signal (detector), others as navigation

signal_5d = hs.signals.Signal2D(data_5d)

# Configure all axes immediately using direct property assignment
signal_5d.axes_manager[0].name = 'temperature'
signal_5d.axes_manager[0].units = 'K'
signal_5d.axes_manager[0].scale = 62.5
signal_5d.axes_manager[0].offset = 300

signal_5d.axes_manager[1].name = 'scan_y'
signal_5d.axes_manager[1].units = 'nm'
signal_5d.axes_manager[1].scale = 0.1

signal_5d.axes_manager[2].name = 'scan_x'
signal_5d.axes_manager[2].units = 'nm'
signal_5d.axes_manager[2].scale = 0.1

signal_5d.axes_manager[3].name = 'detector_y'
signal_5d.axes_manager[3].units = 'px'
signal_5d.axes_manager[3].scale = 1.0

signal_5d.axes_manager[4].name = 'detector_x'
signal_5d.axes_manager[4].units = 'px'
signal_5d.axes_manager[4].scale = 1.0

signal_5d.metadata.General.title = 'Temperature-Dependent 4D-STEM Dataset'

print(f"5D Signal shape: {signal_5d}")
print(f"Navigation axes: {[ax.name for ax in signal_5d.axes_manager.navigation_axes]}")
print(f"Signal axes: {[ax.name for ax in signal_5d.axes_manager.signal_axes]}")

# %%
# ## Analysis-Driven Axis Rearrangement
# 
# Reorganize data based on analysis goals rather than arbitrary reshaping

# Temperature-dependent diffraction analysis
temp_diffraction = signal_5d.transpose(signal_axes=['temperature', 'detector_y', 'detector_x'])
print(f"Temperature-diffraction analysis shape: {temp_diffraction}")

# Spatial mapping at specific temperature
spatial_map = signal_5d.inav[5].transpose(signal_axes=['scan_y', 'scan_x'])
print(f"Spatial map at T={temperatures[5]:.0f}K shape: {spatial_map}")

# Virtual detector time series (integrate detector, analyze temperature evolution)
virtual_detector = signal_5d.transpose(signal_axes=['temperature'])
virtual_series = virtual_detector.sum(axis=('detector_y', 'detector_x'))
print(f"Virtual detector time series shape: {virtual_series}")

# %%
# ## Advanced Indexing Patterns
# 
# Use HyperSpy's sophisticated indexing for complex data selection

# Select temperature range for analysis (adjust based on actual axis order)
mid_temps = signal_5d.inav[:, :, 2:6]  # Temperatures 425-675K
print(f"Mid-temperature range axes: {[ax.name for ax in mid_temps.axes_manager.navigation_axes]}")

# Select spatial ROI across all temperatures (adjust based on actual axis order)
spatial_roi = signal_5d.inav[5:15, 8:18, :]  # Spatial ROI, all temps
print(f"Spatial ROI shape: {spatial_roi}")

# Detector center region for all conditions
detector_center = signal_5d.isig[20:44, 20:44]  # Center 24x24 pixels
print(f"Detector center region shape: {detector_center}")

# Combined selection: specific temps, spatial ROI, detector center (adjust indexing)
complex_roi = signal_5d.inav[8:24, 12:20, 3:7].isig[28:36, 28:36]
print(f"Complex ROI shape: {complex_roi}")

# %%
# ## Statistical Analysis Across Dimensions
# 
# Leverage HyperSpy's axis-aware operations for multidimensional statistics

# Temperature evolution maps (average over detector)
temp_evolution = signal_5d.mean(axis=('detector_y', 'detector_x'))
print(f"Temperature evolution maps shape: {temp_evolution}")

# Spatial variance at each temperature
spatial_variance = signal_5d.std(axis=('scan_y', 'scan_x'))
print(f"Spatial variance patterns shape: {spatial_variance}")

# Diffraction pattern stability (std across temperature)
pattern_stability = signal_5d.std(axis='temperature')
print(f"Pattern stability analysis shape: {pattern_stability}")

# Statistical summary across all navigation dimensions
overall_stats = signal_5d.mean(axis=('temperature', 'scan_y', 'scan_x'))
print(f"Overall diffraction statistics shape: {overall_stats}")

# %%
# ## Efficient Processing Pipelines
# 
# Chain operations for complex analysis workflows

# Pipeline: Select temperature range → ROI → analyze detector patterns
analysis_pipeline = (signal_5d
                     .inav[10:22, 12:20, 2:6]  # Spatial + temperature selection
                     .isig[20:44, 20:44]       # Detector center
                     .mean(axis=('scan_y', 'scan_x'))  # Average spatially
                     .max(axis=('detector_y', 'detector_x')))  # Peak intensity

print(f"Analysis pipeline result shape: {analysis_pipeline}")
print(f"Temperature-dependent peak intensities: {analysis_pipeline.data}")

# %%
# ## Data Reshaping for External Analysis
# 
# When you need to work with external libraries that require specific shapes

# Reshape for machine learning: flatten spatial dimensions
ml_data = signal_5d.transpose(signal_axes=['scan_y', 'scan_x', 'detector_y', 'detector_x'])
# Now shape is (temperature | scan_y, scan_x, detector_y, detector_x)
ml_array = ml_data.data.reshape(n_temp, -1)  # (n_temp, n_features)
print(f"ML-ready data shape: {ml_array.shape}")

# Reshape for traditional image analysis: merge detector dimensions
image_data = signal_5d.transpose(signal_axes=['detector_y', 'detector_x'])
# Shape: (temp, scan_y, scan_x | detector_y, detector_x)
print(f"Image analysis ready shape: {image_data}")

# %%
# ## Memory-Efficient Operations
# 
# Handle large datasets without unnecessary copying

# Check memory usage
data_size_mb = signal_5d.data.nbytes / (1024**2)
print(f"Dataset size: {data_size_mb:.1f} MB")

# Verify operations create views when possible
transposed = signal_5d.transpose(signal_axes=['temperature', 'detector_y', 'detector_x'])
shares_memory = np.shares_memory(signal_5d.data, transposed.data)
print(f"Transpose shares memory: {shares_memory}")

# Use lazy evaluation for very large datasets
print("For datasets too large for memory, consider:")
print("- Using Dask arrays: hs.signals.Signal2D(dask_array)")
print("- Processing in chunks with signal.map()")
print("- Using signal.get_chunk() for partial loading")

# %%
# ## Visualization of Multidimensional Results
# 
# Use HyperSpy's plotting capabilities for complex data

# Plot temperature evolution
temp_evolution.plot()

# Plot diffraction patterns at different temperatures
signal_5d.inav[0].plot()  # First temperature
signal_5d.inav[-1].plot()  # Last temperature

print("Example demonstrates:")
print("1. 5D dataset creation and configuration")
print("2. Analysis-driven axis rearrangement")
print("3. Advanced indexing for complex selections")
print("4. Statistical analysis across multiple dimensions")
print("5. Efficient processing pipelines")
print("6. Memory-conscious data handling")
