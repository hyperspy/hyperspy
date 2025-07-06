"""
Working with Physical Units
============================

This example demonstrates how to work with physical units in HyperSpy using
the Pint library. HyperSpy uses Pint to handle unit conversions and provides
convenient methods to set and manipulate axis scales and offsets with units.

"""

import numpy as np
import hyperspy.api as hs
import pint

# %%
# Introduction to Units in HyperSpy
# ---------------------------------
#
# HyperSpy uses the Pint library to handle physical units. The scale and offset
# of axes can be set and manipulated using physical quantities with units.

# Create a simple 1D signal
data = np.sin(np.linspace(0, 4*np.pi, 100))
s = hs.signals.Signal1D(data)

# Initial axis properties (default values)
print("Initial axis properties:")
print(f"Scale: {s.axes_manager[0].scale}")
print(f"Units: {s.axes_manager[0].units}")
print(f"Offset: {s.axes_manager[0].offset}")

# %%
# ## Setting Units with scale_as_quantity and offset_as_quantity
# 
# HyperSpy provides convenient properties for setting axis scales and offsets with units

s.axes_manager[0].scale_as_quantity = '0.1 nm'
s.axes_manager[0].offset_as_quantity = '5.0 nm'
s.axes_manager[0].name = 'Position'

# Updated axis properties with units applied
print("\nAfter setting scale and offset with units:")
print(f"Scale: {s.axes_manager[0].scale} {s.axes_manager[0].units}")
print(f"Offset: {s.axes_manager[0].offset} {s.axes_manager[0].units}")

# Access the pint quantity objects
scale_quantity = s.axes_manager[0].scale_as_quantity
offset_quantity = s.axes_manager[0].offset_as_quantity

# Display the quantity objects and their properties
print(f"\nPint quantity objects:")
print(f"Scale quantity: {scale_quantity}")
print(f"Offset quantity: {offset_quantity}")
print(f"Type: {type(scale_quantity)}")

# %%
# ## Working with the Pint Unit Registry
# 
# HyperSpy uses the Pint library for unit handling. You can access the registry directly.

# Get the pint unit registry used by HyperSpy
ureg = pint.get_application_registry()

# Create physical quantities
length_quantity = 2.5 * ureg.micrometer
energy_quantity = 200 * ureg.keV

# Examples of created quantities
print(f"\nCreated quantities:")
print(f"Length: {length_quantity}")
print(f"Energy: {energy_quantity}")

# Perform unit conversions
length_in_nm = length_quantity.to('nm')
energy_in_eV = energy_quantity.to('eV')

# Display converted quantities
print(f"\nConverted quantities:")
print(f"Length in nm: {length_in_nm}")
print(f"Energy in eV: {energy_in_eV}")

# %%
# ## Automatic Unit Conversion with convert_units()
# 
# HyperSpy can automatically convert axis units when requested

# Create a 2D signal with different axes
data_2d = np.random.randn(50, 100)
s2d = hs.signals.Signal2D(data_2d)

# Set initial units and scales using efficient batch setting
s2d.axes_manager.signal_axes.set(
    name=['x', 'y'],
    scale=[1.5e-9, 0.5e-9],  # meters  
    units=['m', 'm'],
    offset=[0, 0]
)

print(f"\nInitial 2D signal axes:")
print(f"X-axis: scale={s2d.axes_manager[0].scale}, units={s2d.axes_manager[0].units}")
print(f"Y-axis: scale={s2d.axes_manager[1].scale}, units={s2d.axes_manager[1].units}")

# Convert to convenient units automatically
s2d.axes_manager.convert_units()

print(f"\nAfter automatic unit conversion:")
print(f"X-axis: scale={s2d.axes_manager[0].scale}, units={s2d.axes_manager[0].units}")
print(f"Y-axis: scale={s2d.axes_manager[1].scale}, units={s2d.axes_manager[1].units}")

# %%
# Manual Unit Conversion
# =======================

# Convert specific axes to specific units
s2d_copy = s2d.deepcopy()

# Reset to original units for proper conversion using batch setting
s2d_copy.axes_manager.signal_axes.set(
    scale=[1.5e-9, 0.5e-9],
    units=['m', 'm']
)

# Convert navigation axes to micrometers individually
s2d_copy.axes_manager[0].convert_to_units('µm')
s2d_copy.axes_manager[1].convert_to_units('µm')

print(f"\nAfter converting navigation axes to µm:")
print(f"X-axis: scale={s2d_copy.axes_manager[0].scale}, units={s2d_copy.axes_manager[0].units}")
print(f"Y-axis: scale={s2d_copy.axes_manager[1].scale}, units={s2d_copy.axes_manager[1].units}")

# %%
# Working with Spectroscopic Data
# ================================

# Create an EELS-like signal
# Create 3D data: 20x20 navigation dimensions, 1000 energy channels
nav_data = np.random.randn(20, 20)
energy_profile = 100 * np.exp(-np.linspace(0, 5, 1000))
eels_data = np.random.poisson(energy_profile[np.newaxis, np.newaxis, :] * 
                             (1 + 0.1 * nav_data[:, :, np.newaxis]))
s_eels = hs.signals.Signal1D(eels_data)

# Set spatial axes (navigation axes) using batch setting
s_eels.axes_manager.navigation_axes.set(
    name=['y', 'x'],
    scale=[0.1, 0.1],  # nm per pixel
    units=['nm', 'nm'],
    offset=[0, 0]
)

# Set energy axis (signal axis)
s_eels.axes_manager.signal_axes.set(
    name=['Energy Loss'],
    scale=[0.25],  # eV per channel
    units=['eV'],
    offset=[200]  # start at 200 eV
)

print(f"\nEELS signal axes:")
for i, axis in enumerate(s_eels.axes_manager._axes):
    print(f"Axis {i} ({axis.name}): scale={axis.scale} {axis.units}, offset={axis.offset} {axis.units}")

# %%
# Unit Conversions in Different Scenarios
# ========================================

# Scenario 1: Converting energy units
s_eels_copy = s_eels.deepcopy()

# Convert energy from eV to keV
s_eels_copy.axes_manager[2].convert_to_units('keV')

print(f"\nAfter converting energy to keV:")
print(f"Energy axis: scale={s_eels_copy.axes_manager[2].scale} {s_eels_copy.axes_manager[2].units}")
print(f"Energy axis: offset={s_eels_copy.axes_manager[2].offset} {s_eels_copy.axes_manager[2].units}")

# Scenario 2: Converting spatial units to micrometers
s_eels_copy2 = s_eels.deepcopy()
s_eels_copy2.axes_manager[0].convert_to_units('µm')
s_eels_copy2.axes_manager[1].convert_to_units('µm')

print(f"\nAfter converting spatial axes to µm:")
print(f"X-axis: scale={s_eels_copy2.axes_manager[0].scale} {s_eels_copy2.axes_manager[0].units}")
print(f"Y-axis: scale={s_eels_copy2.axes_manager[1].scale} {s_eels_copy2.axes_manager[1].units}")

# %%
# Using Units with Pint Operations
# =================================

# Get HyperSpy's unit registry
ureg = pint.get_application_registry()

# Work with the axis scale as a pint quantity
original_scale = s_eels.axes_manager[2].scale_as_quantity
print(f"\nOriginal energy scale: {original_scale}")

# Add some energy to the scale using pint operations
additional_energy = 0.05 * ureg.eV
new_scale = original_scale + additional_energy
print(f"New energy scale: {new_scale}")

# Set the new scale back to the axis
s_eels.axes_manager[2].scale_as_quantity = new_scale
print(f"Updated axis scale: {s_eels.axes_manager[2].scale} {s_eels.axes_manager[2].units}")

# %%
# Visualization with Units using HyperSpy's native plotting
# =========================================================

print("\n--- Demonstrating unit conversions with plots ---")

# Plot original signal with eV units
s_plot = s_eels.inav[10, 10]  # Single spectrum
s_plot.metadata.General.title = 'Original Energy Scale (eV)'
s_plot.plot()

# Plot energy converted to keV
s_plot_kev = s_eels_copy.inav[10, 10]
s_plot_kev.metadata.General.title = 'Energy Scale in keV'
s_plot_kev.plot()

# Create spatial navigation maps by summing over energy axis
nav_map_nm = s_eels.sum(axis=-1)  # Sum over energy axis to get 2D navigation map
nav_map_nm.metadata.General.title = 'Spatial Map (nm)'
nav_map_nm.plot()

nav_map_um = s_eels_copy2.sum(axis=-1)  # Sum over energy axis to get 2D navigation map  
nav_map_um.metadata.General.title = 'Spatial Map (µm)'
nav_map_um.plot()

# Use plot_images to compare the spatial maps with different units
hs.plot.plot_images([nav_map_nm, nav_map_um],
                   label=['Spatial Map (nm)', 'Spatial Map (µm)'],
                   cmap='viridis',
                   colorbar=True)

# %%
# Practical Examples with Different Unit Systems
# ===============================================

print("\n" + "="*60)
print("PRACTICAL EXAMPLES")
print("="*60)

# Example 1: TEM/STEM imaging
print("\n1. TEM/STEM Imaging Example:")
stem_data = np.random.randn(256, 256)
stem_image = hs.signals.Signal2D(stem_data)

# Typical STEM parameters using batch setting
stem_image.axes_manager.signal_axes.set(
    name=['x', 'y'],
    scale=[0.05, 0.05],  # nm per pixel
    units=['nm', 'nm']
)

print(f"STEM image field of view: {stem_image.axes_manager[0].size * stem_image.axes_manager[0].scale} × {stem_image.axes_manager[1].size * stem_image.axes_manager[1].scale} nm²")

# Convert to micrometers for overview
stem_image.axes_manager.convert_units(units='µm')
print(f"Field of view in µm: {stem_image.axes_manager[0].size * stem_image.axes_manager[0].scale:.2f} × {stem_image.axes_manager[1].size * stem_image.axes_manager[1].scale:.2f} µm²")

# Example 2: EDS spectrum
print("\n2. EDS Spectrum Example:")
eds_data = np.random.poisson(1000 * np.exp(-0.1 * np.arange(4096)))
eds_spectrum = hs.signals.Signal1D(eds_data)

# Typical EDS parameters using batch setting
eds_spectrum.axes_manager.signal_axes.set(
    name=['Energy'],
    scale=[10],  # eV per channel
    units=['eV'],
    offset=[0]
)

print(f"EDS energy range: {eds_spectrum.axes_manager[0].offset} to {eds_spectrum.axes_manager[0].axis[-1]} eV")

# Convert to keV for display
eds_spectrum.axes_manager[0].convert_to_units('keV')
print(f"EDS energy range in keV: {eds_spectrum.axes_manager[0].offset} to {eds_spectrum.axes_manager[0].axis[-1]} keV")

# Example 3: Diffraction pattern
print("\n3. Diffraction Pattern Example:")
diffraction_data = np.random.randn(512, 512)
diffraction = hs.signals.Signal2D(diffraction_data)

# Typical diffraction units (reciprocal space) using batch setting
diffraction.axes_manager.signal_axes.set(
    name=['qx', 'qy'],
    scale=[0.01e9, 0.01e9],  # 1/m per pixel
    units=['1/m', '1/m']
)

print(f"Diffraction pattern q-range: {diffraction.axes_manager[0].axis[-1]:.2e} 1/m")

# Convert to more convenient units
diffraction.axes_manager.convert_units()
print(f"Converted q-range: {diffraction.axes_manager[0].axis[-1]:.2f} {diffraction.axes_manager[0].units}")

# %%
# Advanced: Custom Unit Operations
# =================================

print("\n" + "="*60)
print("ADVANCED UNIT OPERATIONS")
print("="*60)

# Calculate derived quantities using pint
ureg = pint.get_application_registry()

# Example: Calculate pixel area from STEM image scales
pixel_area = (stem_image.axes_manager[0].scale_as_quantity * 
              stem_image.axes_manager[1].scale_as_quantity)
print(f"\nPixel area: {pixel_area}")
print(f"Pixel area in nm²: {pixel_area.to('nm**2')}")

# Example: Energy resolution calculation
energy_per_channel = eds_spectrum.axes_manager[0].scale_as_quantity
fwhm_channels = 2.5  # typical detector resolution in channels
energy_resolution = fwhm_channels * energy_per_channel
print(f"\nEDS energy resolution: {energy_resolution}")

# Example: Real space to reciprocal space conversion
# For a typical 200 keV electron microscope
electron_wavelength = 2.51e-12 * ureg.meter  # pm for 200 keV electrons
camera_length = 0.3 * ureg.meter  # 30 cm camera length
detector_pixel_size = 14e-6 * ureg.meter  # 14 µm CCD pixel

# Calculate reciprocal space calibration
q_per_pixel = detector_pixel_size / (camera_length * electron_wavelength)
print(f"\nDiffraction calibration: {q_per_pixel.to('1/nm'):.3f} per pixel")

# %%
# Summary
# =======

print("\n" + "="*60)
print("SUMMARY")
print("="*60)

summary_text = """
HyperSpy Physical Units Features:

1. scale_as_quantity and offset_as_quantity:
   - Set and get axis scales/offsets with units
   - Returns Pint quantity objects
   - Example: axis.scale_as_quantity = '0.1 nm'

2. convert_units() method:
   - Automatic unit conversion to convenient scales
   - Manual conversion to specific units
   - Works on individual axes or groups (navigation/signal)

3. Pint integration:
   - Full access to Pint unit registry
   - Perform calculations with physical quantities
   - Unit validation and error checking

4. Common use cases:
   - Microscopy: nm, µm, mm scales
   - Spectroscopy: eV, keV energy scales  
   - Diffraction: reciprocal space units (1/nm, 1/Å)
   - Time series: s, ms, µs

5. Best practices:
   - Set units early in data processing
   - Use automatic conversion for display
   - Leverage Pint for calculations
   - Validate unit compatibility
"""

print(summary_text)

# Show final comparison using HyperSpy signals with proper units
x_nm = np.linspace(0, 10, 100)  # nm
y_data = np.sin(2 * np.pi * x_nm)

# Create HyperSpy signals with different units
signal_nm = hs.signals.Signal1D(y_data)
signal_nm.axes_manager[0].name = 'Distance'
signal_nm.axes_manager[0].units = 'nm'
signal_nm.axes_manager[0].scale = 0.1
signal_nm.axes_manager[0].offset = 0
signal_nm.metadata.General.title = 'Signal in nanometers'

signal_um = signal_nm.deepcopy()
signal_um.axes_manager[0].convert_to_units('µm')
signal_um.metadata.General.title = 'Signal in micrometers'

signal_pm = signal_nm.deepcopy()
signal_pm.axes_manager[0].convert_to_units('pm')
signal_pm.metadata.General.title = 'Signal in picometers'

# Plot using HyperSpy's native plotting - units are handled automatically
signal_nm.plot()
signal_um.plot()
signal_pm.plot()

print("\nExample completed successfully!")
print("Physical units in HyperSpy provide powerful tools for scientific data analysis.")
