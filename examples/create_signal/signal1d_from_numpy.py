"""
Creating Signal1D from numpy array
===================================

This example demonstrates how to create a HyperSpy Signal1D from a numpy array
and understand the fundamental concept of navigation vs signal dimensions.
This is the foundation for all multidimensional data analysis in HyperSpy.
"""

# %%
# **Understanding the Data Structure**
#
# Before creating HyperSpy signals, it's crucial to understand how
# multidimensional data is organized and interpreted for analysis.

import numpy as np
import hyperspy.api as hs

# %%
# **Creating the Input Data**
#
# We'll create a 3D numpy array that represents a common scientific dataset:
# a collection of spectra measured at different spatial positions.

# Create a random numpy array with shape (10, 20, 100)
# This represents:
# - 10 × 20 spatial positions (navigation space)
# - 100 spectral channels per position (signal space)
my_np_array = np.random.random((10, 20, 100))

print("=== Original Numpy Array ===")
print(f"Shape: {my_np_array.shape}")
print(f"Data type: {my_np_array.dtype}")
print(f"Total elements: {my_np_array.size}")
print(f"Memory usage: {my_np_array.nbytes / 1024:.1f} KB")

# %%
# **Converting to HyperSpy Signal1D**
#
# HyperSpy automatically interprets the array dimensions based on the signal type.
# For Signal1D, the last dimension is treated as the signal (spectral) dimension.

s = hs.signals.Signal1D(my_np_array)

print("\n=== HyperSpy Signal1D ===")
print(f"Signal type: {type(s)}")
print(f"Data shape: {s.data.shape}")
print(f"Axes manager: {s.axes_manager}")

# %%
# **Understanding Dimensional Interpretation**
#
# HyperSpy rearranges and interprets array dimensions for multidimensional
# data analysis workflows. This is fundamental to understanding how
# HyperSpy processes your data.

print("\n=== Dimensional Analysis ===")
print(f"Navigation dimensions: {s.axes_manager.navigation_dimension}")
print(f"Signal dimensions: {s.axes_manager.signal_dimension}")
print(f"Navigation shape: {s.axes_manager.navigation_shape}")
print(f"Signal shape: {s.axes_manager.signal_shape}")

# %%
# **Detailed Axes Information**
#
# Each axis in HyperSpy has specific properties that define the data coordinates.
# Understanding these properties is essential for proper data calibration.

print("\n=== Axes Details ===")
for i, axis in enumerate(s.axes_manager._axes):
    print(f"Axis {i}: {axis.name}")
    print(f"  - Size: {axis.size}")
    print(f"  - Index in array: {axis.index_in_array}")
    print(f"  - Navigate: {axis.navigate}")
    print(f"  - Scale: {axis.scale}")
    print(f"  - Offset: {axis.offset}")
    print(f"  - Units: {axis.units}")

# %%
# **Navigation vs Signal Axes Concept**
#
# This distinction is crucial for understanding how HyperSpy processes your data:

# **Navigation axes:** Define where measurements were taken
# - These represent scanning positions, time points, or experimental conditions
# - Operations like mapping functions iterate over these dimensions
# - In this case: 2 navigation axes with sizes 20 and 10 (200 total positions)

# **Signal axes:** Define what was measured at each position
# - These represent the actual measurement (spectrum, image, etc.)
# - Mathematical operations typically apply to these dimensions
# - In this case: 1 signal axis with size 100 (100 spectral channels)

print("\n=== Conceptual Breakdown ===")
print("Navigation space represents: WHERE measurements were taken")
print("Signal space represents: WHAT was measured at each position")
print(f"Total spectra: {s.axes_manager.navigation_size}")
print(f"Channels per spectrum: {s.axes_manager.signal_size}")

# %%
# **Visualizing the Data Structure**
#
# Let's create a simple visualization to understand the data organization.
# This helps conceptualize how multidimensional data is structured.

# Create a more structured example for better visualization
x_axis = np.linspace(0, 10, 100)  # Energy or wavelength axis
y_positions = np.linspace(0, 19, 20)  # Y scan positions  
z_positions = np.linspace(0, 9, 10)   # Z scan positions

# Create synthetic spectral data with spatial variation
structured_data = np.zeros((10, 20, 100))
for i in range(10):
    for j in range(20):
        # Create a spectrum that varies with position
        spectrum = np.exp(-(x_axis - 5 - i*0.1 - j*0.05)**2 / 2) + 0.1*np.random.random(100)
        structured_data[i, j, :] = spectrum

# Create a properly structured Signal1D
s_structured = hs.signals.Signal1D(structured_data)

# Set meaningful axis information
s_structured.axes_manager[0].name = 'Y_position'
s_structured.axes_manager[0].units = 'μm'
s_structured.axes_manager[0].scale = 0.1
s_structured.axes_manager[1].name = 'X_position'  
s_structured.axes_manager[1].units = 'μm'
s_structured.axes_manager[1].scale = 0.1
s_structured.axes_manager[2].name = 'Energy'
s_structured.axes_manager[2].units = 'eV'
s_structured.axes_manager[2].scale = 0.1
s_structured.axes_manager[2].offset = 0.0

print("\n=== Structured Signal Information ===")
print(f"Signal: {s_structured}")
print(f"Axes: {s_structured.axes_manager}")

# %%
# **Interactive Visualization**
#
# Plot the signal to see the navigation vs signal concept in action.
# This creates an interactive plot with a navigator and signal viewer.

s_structured.plot()

# %%
# **Key Takeaways**
#
# 1. **Array Interpretation**: HyperSpy automatically interprets array dimensions
#    based on the signal type (Signal1D, Signal2D, etc.)
#
# 2. **Navigation vs Signal**: Understanding this distinction is fundamental
#    to effectively using HyperSpy for multidimensional data analysis
#
# 3. **Axes Management**: Each dimension has properties (scale, offset, units)
#    that define the coordinate system for your data
#
# 4. **Interactive Plotting**: HyperSpy's plotting system automatically
#    creates appropriate visualizations based on the data dimensionality
#
# 5. **Data Structure**: The signal shape determines how mathematical
#    operations and analysis methods will be applied to your data

# %%
# **Common Use Cases**
#
# This numpy-to-Signal1D conversion is commonly used for:
# - **Spectroscopy**: Converting detector arrays to spectral datasets
# - **Time series**: Converting temporal measurements to signals
# - **Hyperspectral imaging**: Converting image stacks to spectral images
# - **Electron microscopy**: Converting EELS/EDS datasets for analysis
#
# Understanding this fundamental conversion is the first step toward
# effective multidimensional data analysis with HyperSpy.
