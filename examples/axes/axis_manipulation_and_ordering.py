"""
Advanced Axis Manipulation in HyperSpy
======================================

This example demonstrates advanced axis manipulation techniques in HyperSpy,
building on the understanding of HyperSpy's axis conventions.

Key concepts covered:
- Sophisticated use of .T, .transpose(), .as_signal1D(), .as_signal2D()
- Practical axis manipulation workflows
- Real-world data reshaping scenarios
- Best practices for axis management

Prerequisites: Understanding of HyperSpy axis conventions (see axis_ordering_and_conventions.py)
"""

import numpy as np
import hyperspy.api as hs

# %%
# **Review: HyperSpy Axis Conventions**
# ====================================
#
# Quick reminder of the key concepts

print("=== QUICK REVIEW: HyperSpy Axis Conventions ===")

# NumPy array -> HyperSpy Signal1D
numpy_data = np.random.random((10, 20, 30))
s = hs.signals.Signal1D(numpy_data)

print(f"NumPy shape: {numpy_data.shape}")     # (10, 20, 30)
print(f"HyperSpy Signal1D: {s}")             # (20, 10|30)
print("Navigation axes are reversed: NumPy (10, 20) -> HyperSpy (20, 10)")
print("Signal axis is last: NumPy (30) -> HyperSpy (30)")

# Set meaningful names to track manipulations
s.axes_manager[0].name = 'fast_scan'    # Size 20, was NumPy index 1
s.axes_manager[1].name = 'slow_scan'    # Size 10, was NumPy index 0
s.axes_manager[2].name = 'energy'       # Size 30, was NumPy index 2

print(f"Starting signal: {s}")
print(f"Axis names: {[ax.name for ax in s.axes_manager._axes]}")

# %%
# **The .T Property: Complete Navigation-Signal Space Swap**
# ==========================================================
#
# .T is the simplest manipulation - it completely swaps navigation and signal spaces

print("\n=== The .T Property ===")

print(f"Original: {s}")
print(f"  Navigation: {[ax.name for ax in s.axes_manager.navigation_axes]}")
print(f"  Signal: {[ax.name for ax in s.axes_manager.signal_axes]}")

# Apply .T
s_transposed = s.T
print(f"\nAfter .T: {s_transposed}")
print(f"  Navigation: {[ax.name for ax in s_transposed.axes_manager.navigation_axes]}")
print(f"  Signal: {[ax.name for ax in s_transposed.axes_manager.signal_axes]}")

print("\n.T swapped ALL axes between navigation and signal spaces")

# %%
# **The .transpose() Method: Precise Control**
# ===========================================
#
# .transpose() gives you complete control over which axes go where

print("\n=== The .transpose() Method ===")

# Method 1: Specify number of signal axes
s_2sig = s.transpose(signal_axes=2)
print(f"transpose(signal_axes=2): {s_2sig}")
print(f"  Navigation: {[ax.name for ax in s_2sig.axes_manager.navigation_axes]}")  
print(f"  Signal: {[ax.name for ax in s_2sig.axes_manager.signal_axes]}")

# Method 2: Specify number of navigation axes
s_1nav = s.transpose(navigation_axes=1)
print(f"\ntranspose(navigation_axes=1): {s_1nav}")
print(f"  Navigation: {[ax.name for ax in s_1nav.axes_manager.navigation_axes]}")
print(f"  Signal: {[ax.name for ax in s_1nav.axes_manager.signal_axes]}")

# Method 3: Specify exact axes by index
s_custom = s.transpose(signal_axes=[0, 2])  # fast_scan and energy -> signal
print(f"\ntranspose(signal_axes=[0, 2]): {s_custom}")
print(f"  Navigation: {[ax.name for ax in s_custom.axes_manager.navigation_axes]}")
print(f"  Signal: {[ax.name for ax in s_custom.axes_manager.signal_axes]}")

# Method 4: Specify exact axes by name
s_named = s.transpose(signal_axes=['slow_scan', 'energy'])
print(f"\ntranspose(signal_axes=['slow_scan', 'energy']): {s_named}")
print(f"  Navigation: {[ax.name for ax in s_named.axes_manager.navigation_axes]}")
print(f"  Signal: {[ax.name for ax in s_named.axes_manager.signal_axes]}")

# %%
# **Converting Signal Types: .as_signal1D() and .as_signal2D()**
# ==============================================================
#
# These methods change the signal type while preserving data structure

print("\n=== Signal Type Conversion ===")

# Start with a 4D dataset for more interesting conversions
data_4d = np.random.random((5, 8, 10, 100))
s_4d = hs.signals.Signal1D(data_4d)

# Set meaningful names
s_4d.axes_manager[0].name = 'x_pos'      # Size 8, was NumPy index 1
s_4d.axes_manager[1].name = 'y_pos'      # Size 5, was NumPy index 0
s_4d.axes_manager[2].name = 'time'       # Size 10, was NumPy index 2
s_4d.axes_manager[3].name = 'wavelength' # Size 100, was NumPy index 3

print(f"Original Signal1D: {s_4d}")
print(f"  Nav: {[ax.name for ax in s_4d.axes_manager.navigation_axes]}")
print(f"  Sig: {[ax.name for ax in s_4d.axes_manager.signal_axes]}")

# Convert to Signal2D - specify which axes become the 2D signal
s_2d_spatial = s_4d.as_signal2D((0, 1))  # x_pos, y_pos as 2D signal
print(f"\nas_signal2D((0, 1)) - spatial mapping: {s_2d_spatial}")
print(f"  Nav: {[ax.name for ax in s_2d_spatial.axes_manager.navigation_axes]}")
print(f"  Sig: {[ax.name for ax in s_2d_spatial.axes_manager.signal_axes]}")

s_2d_spectral = s_4d.as_signal2D((2, 3))  # time, wavelength as 2D signal
print(f"\nas_signal2D((2, 3)) - spectral dynamics: {s_2d_spectral}")
print(f"  Nav: {[ax.name for ax in s_2d_spectral.axes_manager.navigation_axes]}")
print(f"  Sig: {[ax.name for ax in s_2d_spectral.axes_manager.signal_axes]}")

# Convert to Signal1D - specify which axis becomes the 1D signal
s_1d_time = s_4d.as_signal1D(2)  # time as 1D signal
print(f"\nas_signal1D(2) - time series: {s_1d_time}")
print(f"  Nav: {[ax.name for ax in s_1d_time.axes_manager.navigation_axes]}")
print(f"  Sig: {[ax.name for ax in s_1d_time.axes_manager.signal_axes]}")

# %%
# **Real-World Workflow Examples**
# ================================
#
# Practical examples of axis manipulation for common analysis tasks

print("\n=== REAL-WORLD WORKFLOWS ===")

# Workflow 1: Hyperspectral imaging analysis
print("\n1. Hyperspectral imaging workflow:")

# Simulate hyperspectral data: (time, y, x, wavelength)
hyperspectral_data = np.random.random((10, 20, 30, 400))
hyper_signal = hs.signals.Signal1D(hyperspectral_data)
hyper_signal.axes_manager[0].name = 'x_scan'      # Size 30, was index 2
hyper_signal.axes_manager[1].name = 'y_scan'      # Size 20, was index 1  
hyper_signal.axes_manager[2].name = 'time'        # Size 10, was index 0
hyper_signal.axes_manager[3].name = 'wavelength'  # Size 400, was index 3

print(f"  Original hyperspectral data: {hyper_signal}")

# Task A: Analyze spectral evolution over time
temporal_analysis = hyper_signal.transpose(signal_axes=['time', 'wavelength'])
print(f"  Temporal analysis: {temporal_analysis}")
print("    Use case: Study how spectra change over time at each spatial position")

# Task B: Create spatial maps at specific wavelengths
spatial_maps = hyper_signal.as_signal2D(('x_scan', 'y_scan'))
print(f"  Spatial mapping: {spatial_maps}")
print("    Use case: Generate maps showing spatial distribution of spectral features")

# Task C: Extract time series for region-of-interest analysis
time_series = hyper_signal.transpose(signal_axes=['time'])
print(f"  Time series extraction: {time_series}")
print("    Use case: Study temporal dynamics averaged over spectral bands")

# Workflow 2: 4D-STEM analysis
print("\n2. 4D-STEM workflow:")

# Simulate 4D-STEM data: (scan_y, scan_x, detector_y, detector_x)
stem_data = np.random.random((50, 50, 128, 128))
stem_signal = hs.signals.Signal2D(stem_data)
stem_signal.axes_manager[0].name = 'scan_x'      # Size 50, was index 1
stem_signal.axes_manager[1].name = 'scan_y'      # Size 50, was index 0
stem_signal.axes_manager[2].name = 'detector_y'  # Size 128, was index 2
stem_signal.axes_manager[3].name = 'detector_x'  # Size 128, was index 3

print(f"  Original 4D-STEM: {stem_signal}")

# Task A: Analyze diffraction patterns (keep as Signal2D)
print(f"  Diffraction analysis: {stem_signal}")
print("    Use case: Study diffraction patterns at each scan position")

# Task B: Generate virtual detector images
virtual_detector = stem_signal.transpose(signal_axes=['scan_x', 'scan_y'])
print(f"  Virtual detector: {virtual_detector}")
print("    Use case: Create images from integrated detector regions")

# Task C: Line profile analysis  
line_profiles = stem_signal.as_signal1D('detector_x')
print(f"  Line profiles: {line_profiles}")
print("    Use case: Analyze 1D profiles across detector for each scan position")

# %%
# **Advanced Manipulation Techniques**
# ===================================
#
# Sophisticated axis manipulations for complex analysis needs

print("\n=== ADVANCED TECHNIQUES ===")

# Create a complex 5D dataset
data_5d = np.random.random((2, 3, 4, 5, 6))
s_5d = hs.signals.Signal1D(data_5d)

# Name all axes
axis_names = ['temperature', 'scan_x', 'scan_y', 'time', 'energy']
for i, name in enumerate(axis_names):
    s_5d.axes_manager[i].name = name

print(f"5D signal: {s_5d}")

# Technique 1: Multi-stage transposition
stage1 = s_5d.transpose(signal_axes=['time', 'energy'])
print(f"Stage 1 - time evolution: {stage1}")

stage2 = stage1.transpose(signal_axes=['scan_x', 'scan_y', 'energy'])
print(f"Stage 2 - spatial-spectral maps: {stage2}")

# Technique 2: Selective axis grouping
grouped = s_5d.transpose(navigation_axes=['temperature'], signal_axes=['scan_x', 'scan_y', 'time', 'energy'])
print(f"Grouped analysis: {grouped}")

# Technique 3: Dynamic axis reassignment
dynamic = s_5d.deepcopy()
# Simulate analysis that changes what we consider "signal" vs "navigation"
dynamic = dynamic.transpose(signal_axes=['temperature', 'scan_x'])
print(f"Dynamic reassignment: {dynamic}")

# %%
# **Performance and Memory Considerations**
# =========================================

print("\n=== PERFORMANCE CONSIDERATIONS ===")

print("1. Axis manipulation best practices:")
print("   • .T is fastest - simple pointer swap")
print("   • .transpose() with indices is efficient")
print("   • .as_signal1D()/.as_signal2D() may copy data")
print("   • Use deepcopy() when you need independent copies")

print("\n2. Memory efficiency:")
print("   • Transpositions often create views, not copies")
print("   • Signal type conversions may require data copying")
print("   • Check data.flags.owndata to see if data is copied")

# Demonstrate memory efficiency
original = hs.signals.Signal1D(np.random.random((100, 100, 1000)))
transposed = original.T
converted = original.as_signal2D((0, 1))

print(f"\n   Original owns data: {original.data.flags.owndata}")
print(f"   Transposed owns data: {transposed.data.flags.owndata}")
print(f"   Converted owns data: {converted.data.flags.owndata}")

# %%
# **Summary and Best Practices**
# ==============================

print("\n=== SUMMARY ===")
print("Advanced axis manipulation in HyperSpy:")
print("1. .T: Complete navigation ↔ signal space swap")
print("2. .transpose(): Precise control over axis assignment")
print("3. .as_signal1D()/.as_signal2D(): Change signal interpretation")
print("4. Multiple approaches: by count, by index, by name")
print("5. Consider memory and performance implications")

print("\n=== BEST PRACTICES ===")
print("✅ Always name your axes for clarity")
print("✅ Plan your analysis workflow before manipulating axes")
print("✅ Use the most specific method for your needs")
print("✅ Verify axis assignments after complex manipulations")
print("✅ Consider memory usage for large datasets")
print("✅ Document your axis manipulations in analysis scripts")

print("\n=== COMMON PATTERNS ===")
print("• Hyperspectral: spatial navigation → wavelength signal")
print("• Time series: spatial navigation → time signal")  
print("• 4D-STEM: scan navigation → detector signal")
print("• Tomography: angle navigation → projection signal")
print("• Multi-modal: technique navigation → measurement signal")
