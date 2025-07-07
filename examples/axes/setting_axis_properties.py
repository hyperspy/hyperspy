"""
Setting axis properties
=======================

This example demonstrates how to access and set axis properties in HyperSpy,
including names, scales, offsets, units, and batch operations.
"""

# %%
# ## Create a test signal
# 
# We'll create a 3D signal to demonstrate various axis operations

import numpy as np
import hyperspy.api as hs

# Create a 3D signal (2 navigation, 1 signal dimension)
s = hs.signals.Signal1D(np.random.random((10, 20, 100)))
print(f"Original signal: {s}")
print(f"Axes manager: {s.axes_manager}")

# %%
# ## Method 1: Accessing individual axes
# 
# There are several ways to access individual axes in HyperSpy

# Access by index
nav_axis_0 = s.axes_manager[0]  # First navigation axis
nav_axis_1 = s.axes_manager[1]  # Second navigation axis  
sig_axis = s.axes_manager[2]    # Signal axis (can also use [-1])

print(f"Navigation axis 0: {nav_axis_0}")
print(f"Navigation axis 1: {nav_axis_1}")
print(f"Signal axis: {sig_axis}")

# Alternative access methods
print(f"Signal axis (alternative): {s.axes_manager.signal_axes[0]}")
print(f"Navigation axes: {s.axes_manager.navigation_axes}")

# %%
# ## Method 2: Setting individual axis properties
# 
# Axis properties can be set individually using direct assignment or the .set() method

# Method 2a: Direct assignment (traditional approach)
s.axes_manager[0].name = "X"
s.axes_manager[0].units = "nm"
s.axes_manager[0].scale = 0.5
s.axes_manager[0].offset = 10

# Method 2b: Using .set() method for multiple properties (recommended)
s.axes_manager[1].set(name="Y", units="nm", scale=0.5, offset=20)

# Set properties for signal axis using .set() method
s.axes_manager[2].set(name="Energy", units="eV", scale=0.1, offset=100)

print("After setting properties:")
print(s.axes_manager)

# %%
# ## Using .get() method to retrieve properties efficiently
# 
# The .get() method allows retrieving multiple properties at once

# Get properties from navigation axes
nav_properties = s.axes_manager.navigation_axes.get('name', 'units', 'scale', 'offset')
print("Navigation axes properties:")
for prop, values in nav_properties.items():
    print(f"  {prop}: {values}")

# Get properties from signal axes  
sig_properties = s.axes_manager.signal_axes.get('name', 'units', 'scale', 'offset')
print("\nSignal axes properties:")
for prop, values in sig_properties.items():
    print(f"  {prop}: {values}")

# %%
# ## Method 3: Individual setting (less efficient but sometimes needed)
# 
# For comparison, here's the traditional approach of setting properties individually

# Create a copy to demonstrate individual setting
s_individual = s.deepcopy()

# Individual setting (less efficient for multiple axes)
s_individual.axes_manager[0].name = "X_individual"
s_individual.axes_manager[0].scale = 1.0
s_individual.axes_manager[1].name = "Y_individual" 
s_individual.axes_manager[1].scale = 1.0

print("Individual setting approach:")
print(f"X-axis: name={s_individual.axes_manager[0].name}, scale={s_individual.axes_manager[0].scale}")
print(f"Y-axis: name={s_individual.axes_manager[1].name}, scale={s_individual.axes_manager[1].scale}")

# %%
# Method 3: Access by name (after naming)
print("\n--- Access by name ---")
x_axis = s.axes_manager["X"]
energy_axis = s.axes_manager["Energy"]

print(f"X axis: {x_axis}")
print(f"Energy axis: {energy_axis}")

# Modify properties by name
s.axes_manager["X"].scale = 1.0
s.axes_manager["Energy"].offset = 200

print(f"Updated X axis scale: {s.axes_manager['X'].scale}")
print(f"Updated Energy axis offset: {s.axes_manager['Energy'].offset}")

# %%
# Method 4: Batch setting of axis properties (HyperSpy >= 2.2)
print("\n--- Batch setting of axis properties ---")

# Create a new signal for batch operations
s2 = hs.signals.Signal1D(np.random.random((15, 25, 150)))

# Set multiple properties for navigation axes at once
s2.axes_manager.navigation_axes.set(
    name=("X", "Y"), 
    units=("μm", "μm"),
    scale=(0.2, 0.2),
    offset=(5, 10)
)

# Set properties for signal axes
s2.axes_manager.signal_axes.set(
    name="Wavelength",
    units="nm", 
    scale=2.0,
    offset=400
)

print("After batch setting:")
print(s2.axes_manager)

# %%
# Method 5: Get axis properties in batch
print("\n--- Getting axis properties ---")

# Get specific properties from navigation axes
nav_props = s2.axes_manager.navigation_axes.get("name", "scale", "units")
print(f"Navigation axes properties: {nav_props}")

# Get all important properties
sig_props = s2.axes_manager.signal_axes.get("name", "scale", "offset", "units")
print(f"Signal axes properties: {sig_props}")

# %%
# Method 6: Working with axis coordinates and indices
print("\n--- Coordinates and indices ---")

# Current position in navigation space
print(f"Current navigation indices: {s.axes_manager.indices}")
print(f"Current navigation coordinates: {s.axes_manager.coordinates}")

# Set navigation position
s.axes_manager.indices = [5, 8]  # Move to position (5, 8)
print(f"New navigation indices: {s.axes_manager.indices}")
print(f"New navigation coordinates: {s.axes_manager.coordinates}")

# Set by coordinates
s.axes_manager.coordinates = (12.5, 24.0)  # Move to physical coordinates
print(f"After setting coordinates: {s.axes_manager.indices}")
print(f"Physical coordinates: {s.axes_manager.coordinates}")

# %%
# Plot the configured signal to show the axis labels
s.metadata.General.title = "Signal with configured axes"
s.plot()

print(f"\nFinal axes configuration:")
print(s.axes_manager)
