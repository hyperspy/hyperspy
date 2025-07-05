"""
Batch Axis Property Setting
============================

This example demonstrates how to efficiently set properties for multiple axes
simultaneously using HyperSpy's batch setting capabilities. This is particularly
useful when working with multidimensional datasets where you need to configure
multiple axes with similar or related properties.

Key concepts covered:
- Batch setting of axis properties using set() method
- Batch retrieval of axis properties using get() method
- Working with navigation vs signal axes separately
- Practical examples for different data types

"""

import numpy as np
import hyperspy.api as hs
import matplotlib.pyplot as plt

# %%
# Create Test Signals with Multiple Axes
# --------------------------------------
# 
# Let's create various multidimensional signals to demonstrate batch axis
# property setting for different scenarios.

print("Creating test signals for batch axis property demonstration...")

# 1. 4D spectrum image (2D navigation, 1D signal + energy dispersive)
# This could represent spectrum imaging with energy dispersion
data_4d = np.random.rand(20, 25, 100, 150)
s_4d = hs.signals.Signal2D(data_4d)
s_4d.metadata.General.title = '4D Spectrum Image'
print(f"4D signal shape: {s_4d.data.shape}")
print(f"Navigation axes: {len(s_4d.axes_manager.navigation_axes)}")
print(f"Signal axes: {len(s_4d.axes_manager.signal_axes)}")

# 2. 3D image stack (1D navigation, 2D signal)
data_3d = np.random.rand(30, 128, 128)
s_3d = hs.signals.Signal2D(data_3d)
s_3d.metadata.General.title = '3D Image Stack'
print(f"3D signal shape: {s_3d.data.shape}")

# 3. Spectrum image (2D navigation, 1D signal)
data_spectrum = np.random.rand(15, 12, 256)
s_spectrum = hs.signals.Signal1D(data_spectrum)
s_spectrum.metadata.General.title = 'Spectrum Image'
print(f"Spectrum image shape: {s_spectrum.data.shape}")

# %%
# ## Demonstrate Basic Batch Operations
# 
# The following examples show how to efficiently set properties for multiple axes at once

# ### Example 1: Setting navigation axes properties for 4D signal
#
# First, let's see the current state of the navigation axes:
print("\n1. Setting navigation axes properties for 4D signal:")
print("   Before batch setting:")
for i, axis in enumerate(s_4d.axes_manager.navigation_axes):
    print(f"   Axis {i}: name='{axis.name}', scale={axis.scale}, offset={axis.offset}, units='{axis.units}'")

# Batch set navigation axes properties
s_4d.axes_manager.navigation_axes.set(
    name=("X", "Y"),
    scale=(0.1, 0.1),
    offset=(0, 0),
    units=("μm", "μm")
)

# After batch setting, the axes have been updated:
print("   After batch setting:")
for i, axis in enumerate(s_4d.axes_manager.navigation_axes):
    print(f"   Axis {i}: name='{axis.name}', scale={axis.scale}, offset={axis.offset}, units='{axis.units}'")

# ### Example 2: Setting signal axes properties for 4D signal
#
# Similar batch operations can be applied to signal axes:
print("\n2. Setting signal axes properties for 4D signal:")
print("   Before batch setting:")
for i, axis in enumerate(s_4d.axes_manager.signal_axes):
    print(f"   Axis {i}: name='{axis.name}', scale={axis.scale}, offset={axis.offset}, units='{axis.units}'")

# Batch set signal axes properties (for energy-dispersive detector)
s_4d.axes_manager.signal_axes.set(
    name=("Energy", "Detector_X"),
    scale=(0.5, 1.0),
    offset=(100, 0),
    units=("eV", "px")
)

# Results after batch setting:
print("   After batch setting:")
for i, axis in enumerate(s_4d.axes_manager.signal_axes):
    print(f"   Axis {i}: name='{axis.name}', scale={axis.scale}, offset={axis.offset}, units='{axis.units}'")

# %%
# ## Batch Retrieval of Properties
# 
# You can also retrieve properties from multiple axes at once

# Get all properties for navigation axes
nav_props = s_4d.axes_manager.navigation_axes.get("name", "scale", "offset", "units")
print("   Navigation axes properties:")
for prop, values in nav_props.items():
    print(f"   {prop}: {values}")

# Get all properties for signal axes
sig_props = s_4d.axes_manager.signal_axes.get("name", "scale", "offset", "units")
print("   Signal axes properties:")
for prop, values in sig_props.items():
    print(f"   {prop}: {values}")

# Get all axis names (navigation + signal)
nav_names = s_4d.axes_manager.navigation_axes.get("name")
sig_names = s_4d.axes_manager.signal_axes.get("name")
all_names = nav_names["name"] + sig_names["name"]
print(f"   All axis names: {all_names}")

# %%
# Practical Examples for Different Data Types
# -------------------------------------------

print("\n" + "="*50)
print("PRACTICAL EXAMPLES FOR DIFFERENT DATA TYPES")
print("="*50)

# Example 1: EELS Spectrum Image
print("\n1. EELS Spectrum Image Setup:")
s_eels = s_spectrum.deepcopy()

# Set up EELS-specific axes
s_eels.axes_manager.navigation_axes.set(
    name=("X", "Y"),
    scale=(2.0, 2.0),      # 2 nm per pixel
    offset=(-15, -12),     # Start positions
    units=("nm", "nm")
)

s_eels.axes_manager.signal_axes[0].name = "Energy Loss"
s_eels.axes_manager.signal_axes[0].scale = 0.25  # 0.25 eV per channel
s_eels.axes_manager.signal_axes[0].offset = 200  # Start at 200 eV
s_eels.axes_manager.signal_axes[0].units = "eV"

print("   EELS spectrum image configured:")
print(f"   Navigation: {s_eels.axes_manager.navigation_size} spectra")
print(f"   Signal: {s_eels.axes_manager.signal_size} energy channels")
print(f"   Energy range: {s_eels.axes_manager.signal_axes[0].offset:.1f} to "
      f"{s_eels.axes_manager.signal_axes[0].offset + s_eels.axes_manager.signal_axes[0].scale * s_eels.axes_manager.signal_axes[0].size:.1f} eV")

# Example 2: Time-resolved imaging
print("\n2. Time-resolved Image Series Setup:")
s_time = s_3d.deepcopy()

# Configure time-resolved imaging axes
s_time.axes_manager.navigation_axes[0].name = "Time"
s_time.axes_manager.navigation_axes[0].scale = 0.1  # 100 ms per frame
s_time.axes_manager.navigation_axes[0].offset = 0
s_time.axes_manager.navigation_axes[0].units = "s"

s_time.axes_manager.signal_axes.set(
    name=("Y", "X"),
    scale=(50e-3, 50e-3),  # 50 μm per pixel
    offset=(0, 0),
    units=("μm", "μm")
)

print("   Time-resolved imaging configured:")
print(f"   Time points: {s_time.axes_manager.navigation_axes[0].size}")
print(f"   Duration: {s_time.axes_manager.navigation_axes[0].scale * s_time.axes_manager.navigation_axes[0].size:.1f} s")
print(f"   Image size: {s_time.axes_manager.signal_axes[0].size} × {s_time.axes_manager.signal_axes[1].size} pixels")

# Example 3: Multi-energy X-ray imaging
print("\n3. Multi-energy X-ray Imaging Setup:")
s_xray = s_4d.deepcopy()

# Configure for multi-energy X-ray imaging
s_xray.axes_manager.navigation_axes.set(
    name=("Y", "X"),
    scale=(10e-3, 10e-3),  # 10 μm per pixel
    offset=(0, 0),
    units=("mm", "mm")
)

s_xray.axes_manager.signal_axes.set(
    name=("Energy", "Detector_Y"),
    scale=(1.0, 0.1),      # 1 keV energy, 0.1 mm detector
    offset=(10, -5),       # Start at 10 keV, center detector
    units=("keV", "mm")
)

print("   Multi-energy X-ray imaging configured:")
print(f"   Spatial resolution: {s_xray.axes_manager.navigation_axes[0].scale*1000:.0f} μm")
print(f"   Energy range: {s_xray.axes_manager.signal_axes[0].offset:.0f}-{s_xray.axes_manager.signal_axes[0].offset + s_xray.axes_manager.signal_axes[0].scale * s_xray.axes_manager.signal_axes[0].size:.0f} keV")

# %%
# Advanced Batch Operations
# -------------------------

print("\n" + "="*50)
print("ADVANCED BATCH OPERATIONS")
print("="*50)

# Example 1: Conditional batch setting
print("\n1. Conditional axis configuration based on signal type:")

def configure_spectroscopy_axes(signal, technique="EELS"):
    """Configure axes for different spectroscopy techniques."""
    
    if technique == "EELS":
        # Electron Energy Loss Spectroscopy
        nav_config = {
            "name": ("X", "Y"),
            "scale": (1.0, 1.0),
            "units": ("nm", "nm")
        }
        sig_config = {
            "name": "Energy Loss",
            "scale": 0.5,
            "offset": 0,
            "units": "eV"
        }
    elif technique == "EDS":
        # Energy Dispersive Spectroscopy
        nav_config = {
            "name": ("X", "Y"), 
            "scale": (5.0, 5.0),
            "units": ("nm", "nm")
        }
        sig_config = {
            "name": "X-ray Energy",
            "scale": 0.01,
            "offset": 0,
            "units": "keV"
        }
    elif technique == "CL":
        # Cathodoluminescence
        nav_config = {
            "name": ("X", "Y"),
            "scale": (0.1, 0.1),
            "units": ("μm", "μm")
        }
        sig_config = {
            "name": "Wavelength",
            "scale": 1.0,
            "offset": 400,
            "units": "nm"
        }
    else:
        # Default configuration
        nav_config = {
            "name": ("X", "Y"),
            "scale": (1.0, 1.0),
            "units": ("", "")
        }
        sig_config = {
            "name": "Signal",
            "scale": 1.0,
            "offset": 0,
            "units": ""
        }
    
    # Apply navigation axes configuration
    if signal.axes_manager.navigation_dimension >= 2:
        signal.axes_manager.navigation_axes.set(**nav_config)
    
    # Apply signal axes configuration
    if signal.axes_manager.signal_dimension >= 1:
        signal.axes_manager.signal_axes[0].name = sig_config["name"]
        signal.axes_manager.signal_axes[0].scale = sig_config["scale"]
        signal.axes_manager.signal_axes[0].offset = sig_config["offset"]
        signal.axes_manager.signal_axes[0].units = sig_config["units"]

# Test the configuration function
s_test = hs.signals.Signal1D(np.random.rand(10, 10, 100))

print("   Configuring for EELS:")
configure_spectroscopy_axes(s_test, "EELS")
nav_props = s_test.axes_manager.navigation_axes.get("name", "scale", "units")
print(f"   Navigation: {nav_props}")
print(f"   Signal: {s_test.axes_manager.signal_axes[0].name}, {s_test.axes_manager.signal_axes[0].scale} {s_test.axes_manager.signal_axes[0].units}")

print("   Configuring for EDS:")
configure_spectroscopy_axes(s_test, "EDS") 
nav_props = s_test.axes_manager.navigation_axes.get("name", "scale", "units")
print(f"   Navigation: {nav_props}")
print(f"   Signal: {s_test.axes_manager.signal_axes[0].name}, {s_test.axes_manager.signal_axes[0].scale} {s_test.axes_manager.signal_axes[0].units}")

# Example 2: Copying axis properties between signals
print("\n2. Copying axis properties between signals:")

# Create a template signal with well-configured axes
template = hs.signals.Signal1D(np.random.rand(5, 5, 50))
template.axes_manager.navigation_axes.set(
    name=("Stage_X", "Stage_Y"),
    scale=(0.5, 0.5),
    offset=(-2.5, -2.5),
    units=("mm", "mm")
)
template.axes_manager.signal_axes[0].name = "Raman Shift"
template.axes_manager.signal_axes[0].scale = 2.0
template.axes_manager.signal_axes[0].offset = 100
template.axes_manager.signal_axes[0].units = "cm⁻¹"

# Create a new signal that needs the same configuration
new_signal = hs.signals.Signal1D(np.random.rand(8, 8, 75))

print("   Template signal configuration:")
print(f"   Nav: {template.axes_manager.navigation_axes.get('name', 'scale', 'units')}")
print(f"   Sig: {template.axes_manager.signal_axes[0].name}, {template.axes_manager.signal_axes[0].scale} {template.axes_manager.signal_axes[0].units}")

# Copy properties from template (adjusting for different sizes)
new_signal.axes_manager.navigation_axes.set(
    **template.axes_manager.navigation_axes.get("name", "scale", "units")
)
new_signal.axes_manager.signal_axes[0].name = template.axes_manager.signal_axes[0].name
new_signal.axes_manager.signal_axes[0].scale = template.axes_manager.signal_axes[0].scale
new_signal.axes_manager.signal_axes[0].units = template.axes_manager.signal_axes[0].units

print("   New signal after copying configuration:")
print(f"   Nav: {new_signal.axes_manager.navigation_axes.get('name', 'scale', 'units')}")
print(f"   Sig: {new_signal.axes_manager.signal_axes[0].name}, {new_signal.axes_manager.signal_axes[0].scale} {new_signal.axes_manager.signal_axes[0].units}")

# %%
# Visualization of Configured Axes
# --------------------------------

print("\n3. Demonstration of properly configured axes in plots:")

# Demonstrate how properly configured axes improve data interpretation
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# EELS spectrum image - show sum spectrum and spatial distribution of a peak
eels_sum_spectrum = s_eels.sum()
axes[0, 0].plot(eels_sum_spectrum.axes_manager[0].axis, eels_sum_spectrum.data)
axes[0, 0].set_title('EELS Sum Spectrum')
axes[0, 0].set_xlabel(f'{eels_sum_spectrum.axes_manager[0].name} ({eels_sum_spectrum.axes_manager[0].units})')
axes[0, 0].set_ylabel('Intensity')
axes[0, 0].grid(True, alpha=0.3)

# Time-resolved imaging - show first frame
time_frame = s_time.inav[0]
im1 = axes[0, 1].imshow(time_frame.data, origin='lower')
axes[0, 1].set_title('Time-resolved Image (t=0)')
axes[0, 1].set_xlabel('X (pixels)')
axes[0, 1].set_ylabel('Y (pixels)')

# Show time series at a point
time_series = s_time.inav[:].isig[64, 64]  # Time series at center pixel
axes[1, 0].plot(s_time.axes_manager.navigation_axes[0].axis, time_series.data)
axes[1, 0].set_xlabel(f'{s_time.axes_manager.navigation_axes[0].name} ({s_time.axes_manager.navigation_axes[0].units})')
axes[1, 0].set_ylabel('Intensity')
axes[1, 0].set_title('Time Series at Center Pixel')
axes[1, 0].grid(True, alpha=0.3)

# Show example spectrum from EELS
single_spectrum = s_eels.inav[7, 6]
axes[1, 1].plot(single_spectrum.axes_manager[0].axis, single_spectrum.data)
axes[1, 1].set_xlabel(f'{single_spectrum.axes_manager[0].name} ({single_spectrum.axes_manager[0].units})')
axes[1, 1].set_ylabel('Intensity')
axes[1, 1].set_title('Individual EELS Spectrum')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("   ✓ Properly configured axes enable meaningful plots with correct labels and units")

# %%
# Best Practices and Performance Tips
# -----------------------------------

print("\n" + "="*60)
print("BEST PRACTICES FOR BATCH AXIS PROPERTY SETTING")
print("="*60)
print("""
1. EFFICIENCY TIPS:
   • Use batch operations (.set() method) instead of individual assignments
   • Batch operations are faster for multiple axes
   • Use .get() method to retrieve multiple properties at once

2. NAMING CONVENTIONS:
   • Use descriptive axis names (e.g., "Energy Loss" not "axis_2")
   • Follow consistent naming within your workflow
   • Use standard unit abbreviations (nm, eV, keV, etc.)

3. SCALE AND OFFSET SETTING:
   • Always set scale before offset when possible
   • Use physical units that make sense for your data
   • Consider the precision needed for your analysis

4. SIGNAL TYPE CONSIDERATIONS:
   • Different signal types may need different axis configurations
   • Use conditional configuration functions for flexibility
   • Create templates for common measurement setups

5. VERIFICATION:
   • Always verify axis configuration after batch setting
   • Use .get() method to check properties were set correctly
   • Plot data to visually confirm axis labels and scales
""")

# Demonstrate performance comparison
print("\n4. Performance comparison: individual vs batch setting")

import time

# Create a test signal
test_signal = hs.signals.Signal1D(np.random.rand(10, 10, 100))

# Method 1: Individual setting
start_time = time.time()
test_signal.axes_manager[0].name = "X"
test_signal.axes_manager[0].scale = 0.1
test_signal.axes_manager[0].offset = 0
test_signal.axes_manager[0].units = "μm"
test_signal.axes_manager[1].name = "Y"
test_signal.axes_manager[1].scale = 0.1
test_signal.axes_manager[1].offset = 0
test_signal.axes_manager[1].units = "μm"
individual_time = time.time() - start_time

# Method 2: Batch setting
start_time = time.time()
test_signal.axes_manager.navigation_axes.set(
    name=("X", "Y"),
    scale=(0.1, 0.1),
    offset=(0, 0),
    units=("μm", "μm")
)
batch_time = time.time() - start_time

print(f"   Individual setting time: {individual_time*1000:.3f} ms")
print(f"   Batch setting time: {batch_time*1000:.3f} ms")
print(f"   Speed improvement: {individual_time/batch_time:.1f}x faster")

print("\n5. Common axis configuration patterns:")

patterns = {
    "SEM Imaging": {
        "nav": {"name": ("Y", "X"), "scale": (1e-3, 1e-3), "units": ("μm", "μm")},
        "description": "Scanning electron microscopy with μm resolution"
    },
    "TEM Diffraction": {
        "nav": {"name": ("Y", "X"), "scale": (0.1, 0.1), "units": ("1/nm", "1/nm")},
        "description": "Transmission electron microscopy diffraction patterns"
    },
    "Raman Mapping": {
        "nav": {"name": ("Y", "X"), "scale": (0.5, 0.5), "units": ("μm", "μm")},
        "sig": {"name": "Raman Shift", "scale": 1.0, "units": "cm⁻¹"},
        "description": "Raman spectroscopy mapping"
    },
    "XPS Imaging": {
        "nav": {"name": ("Y", "X"), "scale": (10e-3, 10e-3), "units": ("mm", "mm")},
        "sig": {"name": "Binding Energy", "scale": 0.1, "units": "eV"},
        "description": "X-ray photoelectron spectroscopy imaging"
    }
}

for technique, config in patterns.items():
    print(f"\n   {technique}: {config['description']}")
    if 'nav' in config:
        print(f"     Navigation: {config['nav']}")
    if 'sig' in config:
        print(f"     Signal: {config['sig']}")

print("\nBatch axis property setting examples completed!")
print("Use .set() and .get() methods for efficient axis management!")
