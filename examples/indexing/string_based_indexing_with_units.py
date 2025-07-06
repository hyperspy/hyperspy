"""
String-based Indexing with Units in HyperSpy
=============================================

This example demonstrates HyperSpy's powerful string-based indexing capabilities
that allow slicing signals using physical units and relative positions.

"""

import hyperspy.api as hs
import numpy as np

# %%
# ## Creating a test signal with physical units
# Let's create a 1D signal representing an energy spectrum

s = hs.signals.Signal1D(np.random.random(200) * 1000)
s.axes_manager[0].scale = 0.05  # 0.05 eV per channel
s.axes_manager[0].offset = 100  # Start at 100 eV
s.axes_manager[0].units = 'eV'
s.axes_manager[0].name = 'Energy'

print(f"Energy range: {s.axes_manager[0].axis[0]:.1f} to {s.axes_manager[0].axis[-1]:.1f} eV")
print(f"Signal shape: {s.data.shape}")

# %%
# ## String-based slicing with same units
# When axis has units, we can slice using strings with the same units

energy_subset = s.isig['105 eV':'108 eV']
print(f"Subset from 105-108 eV has shape: {energy_subset.data.shape}")
print(f"Energy range: {energy_subset.axes_manager[0].axis[0]:.2f} to {energy_subset.axes_manager[0].axis[-1]:.2f} eV")

# %%
# ## String-based slicing with compatible units  
# HyperSpy automatically converts between compatible units

# Create a distance-based signal for unit conversion demonstration
distance_signal = hs.signals.Signal1D(np.random.random(100))
distance_signal.axes_manager[0].scale = 0.001  # 1 nm per channel 
distance_signal.axes_manager[0].offset = 1     # Start at 1 µm
distance_signal.axes_manager[0].units = 'µm'
distance_signal.axes_manager[0].name = 'Distance'

print(f"\nDistance range: {distance_signal.axes_manager[0].axis[0]:.3f} to {distance_signal.axes_manager[0].axis[-1]:.3f} µm")

# Slice using micrometers
subset_um = distance_signal.isig['1.02 µm':'1.05 µm']
print(f"Slice in µm: {subset_um.data.shape} elements")

# Slice using nanometers (automatically converted)
subset_nm = distance_signal.isig['1020 nm':'1050 nm'] 
print(f"Slice in nm: {subset_nm.data.shape} elements")

# Slice using millimeters
subset_mm = distance_signal.isig['0.00102 mm':'0.00105 mm']
print(f"Slice in mm: {subset_mm.data.shape} elements")

# %%
# ## Relative indexing
# Use 'rel' prefix to specify positions as fractions of the total range

print("\nRelative indexing examples:")
rel_start = s.isig['rel0.1':'rel0.3']  # 10% to 30% of the energy range
print(f"10%-30% of range: {rel_start.data.shape} elements")
print(f"Energy range: {rel_start.axes_manager[0].axis[0]:.2f} to {rel_start.axes_manager[0].axis[-1]:.2f} eV")

rel_end = s.isig['rel0.8':'rel1.0']    # 80% to 100% of the range
print(f"80%-100% of range: {rel_end.data.shape} elements")

# %%
# ## Multi-dimensional indexing with units
# Create a 2D signal with navigation dimensions

signal_2d = hs.signals.Signal1D(np.random.random((50, 40, 80)))

# Set up navigation axes with units
signal_2d.axes_manager.navigation_axes.set(
    scale=[0.1, 0.1],
    offset=[0, 0],
    units=['µm', 'µm'],
    name=['x', 'y']
)

# Set up signal axis
signal_2d.axes_manager[2].scale = 0.1
signal_2d.axes_manager[2].offset = 500
signal_2d.axes_manager[2].units = 'nm'
signal_2d.axes_manager[2].name = 'Wavelength'

print(f"\n2D signal shape: {signal_2d.data.shape}")

# Navigation indexing with units
nav_slice = signal_2d.inav['1.0 µm':'3.0 µm', '0.5 µm':'2.5 µm']
print(f"Navigation slice shape: {nav_slice.data.shape}")

# Signal indexing with units  
sig_slice = signal_2d.isig['520 nm':'560 nm']
print(f"Signal slice shape: {sig_slice.data.shape}")

# Combined indexing
combined = signal_2d.inav['1.0 µm':'3.0 µm', :].isig['520 nm':'560 nm']
print(f"Combined slice shape: {combined.data.shape}")

# %%
# ## Unit conversion in navigation
# Use different units for the same physical dimension

# Convert navigation units
nav_nm = signal_2d.inav['1000 nm':'3000 nm', :]  # Same as 1-3 µm
print(f"Navigation with nm units: {nav_nm.data.shape}")

# %%
# ## Best practices for string-based indexing

print("\nBest practices demonstrated:")
print("✓ Use meaningful axis names and units")
print("✓ Leverage automatic unit conversion between compatible dimensions")  
print("✓ Use relative indexing for proportion-based selections")
print("✓ Combine with regular integer/float indexing as needed")
print("✓ Units must be dimensionally compatible (length with length, energy with energy)")

# %%
# ## Error handling example
# Attempting to use incompatible units will raise an error

try:
    # This will fail - cannot convert between energy and length units
    bad_slice = s.isig['105 nm':'108 nm']  # nm is length, eV is energy
except Exception as e:
    print(f"\nExpected error with incompatible units:")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")

# %%
# ## Summary
print("\nString-based indexing enables:")
print("• Intuitive slicing using physical units")
print("• Automatic unit conversion between compatible dimensions")  
print("• Relative positioning using 'rel' prefix")
print("• Works with both .isig and .inav")
print("• Essential for working with calibrated, real-world data")
