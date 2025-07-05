"""
Working with Physical Units
============================

This example demonstrates how to work with physical units in HyperSpy using
the Pint library. HyperSpy uses Pint to handle unit conversions and provides
convenient methods to set and manipulate axis scales and offsets with units.

"""

import numpy as np
import matplotlib.pyplot as plt
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

# Set initial units and scales
s2d.axes_manager[0].name = 'x'
s2d.axes_manager[0].scale = 1.5e-9  # meters
s2d.axes_manager[0].units = 'm'
s2d.axes_manager[0].offset = 0

s2d.axes_manager[1].name = 'y'  
s2d.axes_manager[1].scale = 0.5e-9  # meters
s2d.axes_manager[1].units = 'm'
s2d.axes_manager[1].offset = 0

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

# Reset to original units for proper conversion
s2d_copy.axes_manager[0].scale = 1.5e-9
s2d_copy.axes_manager[0].units = 'm'
s2d_copy.axes_manager[1].scale = 0.5e-9
s2d_copy.axes_manager[1].units = 'm'

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

# Set spatial axes (navigation axes)
s_eels.axes_manager[0].name = 'y'
s_eels.axes_manager[0].scale = 0.1  # nm per pixel
s_eels.axes_manager[0].units = 'nm'
s_eels.axes_manager[0].offset = 0

s_eels.axes_manager[1].name = 'x'
s_eels.axes_manager[1].scale = 0.1  # nm per pixel  
s_eels.axes_manager[1].units = 'nm'
s_eels.axes_manager[1].offset = 0

# Set energy axis (signal axis)
s_eels.axes_manager[2].name = 'Energy Loss'
s_eels.axes_manager[2].scale = 0.25  # eV per channel
s_eels.axes_manager[2].units = 'eV'
s_eels.axes_manager[2].offset = 200  # start at 200 eV

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
# Visualization with Units
# =========================

# Create a figure showing the effect of unit conversions
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Effect of Unit Conversions on Axis Display', fontsize=14)

# Original signal with nm units
s_plot = s_eels.inav[10, 10]  # Single spectrum
axes[0, 0].plot(s_plot.axes_manager[0].axis, s_plot.data)
axes[0, 0].set_xlabel(f'{s_plot.axes_manager[0].name} ({s_plot.axes_manager[0].units})')
axes[0, 0].set_ylabel('Intensity')
axes[0, 0].set_title('Original Energy Scale (eV)')
axes[0, 0].grid(True, alpha=0.3)

# Energy converted to keV
s_plot_kev = s_eels_copy.inav[10, 10]
axes[0, 1].plot(s_plot_kev.axes_manager[0].axis, s_plot_kev.data)
axes[0, 1].set_xlabel(f'{s_plot_kev.axes_manager[0].name} ({s_plot_kev.axes_manager[0].units})')
axes[0, 1].set_ylabel('Intensity')
axes[0, 1].set_title('Energy Scale in keV')
axes[0, 1].grid(True, alpha=0.3)

# Spatial navigation in nm
nav_map = s_eels.sum(axis=-1)  # Sum over energy axis to get 2D navigation map
im1 = axes[1, 0].imshow(nav_map.data, extent=[
    s_eels.axes_manager[1].axis[0], s_eels.axes_manager[1].axis[-1],
    s_eels.axes_manager[0].axis[-1], s_eels.axes_manager[0].axis[0]
])
axes[1, 0].set_xlabel(f'{s_eels.axes_manager[1].name} ({s_eels.axes_manager[1].units})')
axes[1, 0].set_ylabel(f'{s_eels.axes_manager[0].name} ({s_eels.axes_manager[0].units})')
axes[1, 0].set_title('Spatial Map (nm)')
plt.colorbar(im1, ax=axes[1, 0])

# Spatial navigation in µm
nav_map_um = s_eels_copy2.sum(axis=-1)  # Sum over energy axis to get 2D navigation map
im2 = axes[1, 1].imshow(nav_map_um.data, extent=[
    s_eels_copy2.axes_manager[1].axis[0], s_eels_copy2.axes_manager[1].axis[-1],
    s_eels_copy2.axes_manager[0].axis[-1], s_eels_copy2.axes_manager[0].axis[0]
])
axes[1, 1].set_xlabel(f'{s_eels_copy2.axes_manager[1].name} ({s_eels_copy2.axes_manager[1].units})')
axes[1, 1].set_ylabel(f'{s_eels_copy2.axes_manager[0].name} ({s_eels_copy2.axes_manager[0].units})')
axes[1, 1].set_title('Spatial Map (µm)')
plt.colorbar(im2, ax=axes[1, 1])

plt.tight_layout()
plt.show()

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

# Typical STEM parameters
stem_image.axes_manager[0].name = 'x'
stem_image.axes_manager[0].scale = 0.05  # nm per pixel
stem_image.axes_manager[0].units = 'nm'

stem_image.axes_manager[1].name = 'y'
stem_image.axes_manager[1].scale = 0.05  # nm per pixel
stem_image.axes_manager[1].units = 'nm'

print(f"STEM image field of view: {stem_image.axes_manager[0].size * stem_image.axes_manager[0].scale} × {stem_image.axes_manager[1].size * stem_image.axes_manager[1].scale} nm²")

# Convert to micrometers for overview
stem_image.axes_manager.convert_units(units='µm')
print(f"Field of view in µm: {stem_image.axes_manager[0].size * stem_image.axes_manager[0].scale:.2f} × {stem_image.axes_manager[1].size * stem_image.axes_manager[1].scale:.2f} µm²")

# Example 2: EDS spectrum
print("\n2. EDS Spectrum Example:")
eds_data = np.random.poisson(1000 * np.exp(-0.1 * np.arange(4096)))
eds_spectrum = hs.signals.Signal1D(eds_data)

# Typical EDS parameters
eds_spectrum.axes_manager[0].name = 'Energy'
eds_spectrum.axes_manager[0].scale = 10  # eV per channel
eds_spectrum.axes_manager[0].units = 'eV'
eds_spectrum.axes_manager[0].offset = 0

print(f"EDS energy range: {eds_spectrum.axes_manager[0].offset} to {eds_spectrum.axes_manager[0].axis[-1]} eV")

# Convert to keV for display
eds_spectrum.axes_manager[0].convert_to_units('keV')
print(f"EDS energy range in keV: {eds_spectrum.axes_manager[0].offset} to {eds_spectrum.axes_manager[0].axis[-1]} keV")

# Example 3: Diffraction pattern
print("\n3. Diffraction Pattern Example:")
diffraction_data = np.random.randn(512, 512)
diffraction = hs.signals.Signal2D(diffraction_data)

# Typical diffraction units (reciprocal space)
diffraction.axes_manager[0].name = 'qx'
diffraction.axes_manager[0].scale = 0.01e9  # 1/m per pixel
diffraction.axes_manager[0].units = '1/m'

diffraction.axes_manager[1].name = 'qy'
diffraction.axes_manager[1].scale = 0.01e9  # 1/m per pixel
diffraction.axes_manager[1].units = '1/m'

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

# Show final comparison
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Create example with different unit systems
x_nm = np.linspace(0, 10, 100)  # nm
y_data = np.sin(2 * np.pi * x_nm)

x_um = x_nm / 1000  # µm
x_pm = x_nm * 1000  # pm

ax.plot(x_nm, y_data, 'b-', label='nanometers (nm)', linewidth=2)
ax.plot(x_um * 1000, y_data, 'r--', label='micrometers (×1000 for display)', linewidth=2)
ax.plot(x_pm / 1000, y_data, 'g:', label='picometers (÷1000 for display)', linewidth=2)

ax.set_xlabel('Distance (nm equivalent)')
ax.set_ylabel('Signal')
ax.set_title('Same Data with Different Unit Representations')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nExample completed successfully!")
print("Physical units in HyperSpy provide powerful tools for scientific data analysis.")
