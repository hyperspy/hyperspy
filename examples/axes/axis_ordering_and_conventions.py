"""
Axis Ordering and Conventions in HyperSpy
==========================================

This example demonstrates the fundamental differences between HyperSpy's axis 
conventions and NumPy's, focusing on:

1. Natural order vs Array order
2. Navigation vs Signal axis classification
3. How HyperSpy reverses and reinterprets NumPy data
4. Practical implications for data manipulation

Understanding these concepts is crucial for working effectively with HyperSpy.
"""

import numpy as np
import hyperspy.api as hs

# %%
# **The Core Concept: Natural Order vs Array Order**
# ==================================================
#
# HyperSpy fundamentally reverses how we think about array axes compared to NumPy.

print("=== FUNDAMENTAL DIFFERENCE: Array Order vs Natural Order ===")

# Create a simple 3D NumPy array
data_3d = np.random.random((4, 5, 6))
print(f"NumPy array shape: {data_3d.shape}")  # (4, 5, 6)

# When HyperSpy creates a signal, it reverses the interpretation
signal_1d = hs.signals.Signal1D(data_3d)
print(f"HyperSpy Signal1D: {signal_1d}")  # <Signal1D, title: , dimensions: (5, 4|6)>

print("\nKey insights:")
print("• NumPy array order: (4, 5, 6)")
print("• HyperSpy natural order display: (5, 4|6)")
print("• Last NumPy axis (6) becomes signal axis")
print("• Earlier NumPy axes (4, 5) become navigation axes in REVERSED order")

# The actual data hasn't changed - just how HyperSpy interprets it
print(f"\nData shape is still: {signal_1d.data.shape}")  # Still (4, 5, 6)

# %%
# **Understanding Index Mappings**
# ================================
#
# Let's explore how HyperSpy maps between natural order and array order

print("\n=== INDEX MAPPINGS ===")

# Create a 4D example for clarity
data_4d = np.arange(2*3*4*5).reshape(2, 3, 4, 5)
signal_4d = hs.signals.Signal1D(data_4d)

print(f"NumPy data shape: {data_4d.shape}")  # (2, 3, 4, 5)
print(f"HyperSpy signal: {signal_4d}")      # (4, 3, 2|5)

# Set names to make the mapping clear
signal_4d.axes_manager[0].name = 'nav_axis_0'  # Natural order index 0
signal_4d.axes_manager[1].name = 'nav_axis_1'  # Natural order index 1  
signal_4d.axes_manager[2].name = 'nav_axis_2'  # Natural order index 2
signal_4d.axes_manager[3].name = 'sig_axis_0'  # Natural order index 3

print("\nAxis mapping:")
for i, axis in enumerate(signal_4d.axes_manager._axes):
    print(f"  Natural order [{i}]: {axis.name} -> Array index {axis.index_in_array}")

# Natural order gives: nav_axis_0, nav_axis_1, nav_axis_2, sig_axis_0
# Array order gives: nav_axis_2, nav_axis_1, nav_axis_0, sig_axis_0

print("\nNatural order vs Array order:")
natural_order = signal_4d.axes_manager._get_axes_in_natural_order()
array_order = signal_4d.axes_manager._axes
print(f"  Natural order names: {[ax.name for ax in natural_order]}")
print(f"  Array order names: {[ax.name for ax in array_order]}")

# %%
# **Navigation vs Signal Classification**
# =======================================
#
# HyperSpy automatically classifies axes as navigation or signal

print("\n=== NAVIGATION vs SIGNAL CLASSIFICATION ===")

# Signal1D: Last axis is signal, others are navigation
print("Signal1D classification:")
print(f"  Navigation axes: {[ax.name for ax in signal_4d.axes_manager.navigation_axes]}")
print(f"  Signal axes: {[ax.name for ax in signal_4d.axes_manager.signal_axes]}")

# Signal2D: Last TWO axes are signal, others are navigation
signal_2d = hs.signals.Signal2D(data_4d)
signal_2d.axes_manager[0].name = 'nav_0'
signal_2d.axes_manager[1].name = 'nav_1' 
signal_2d.axes_manager[2].name = 'sig_0'
signal_2d.axes_manager[3].name = 'sig_1'

print(f"\nSignal2D: {signal_2d}")
print(f"  Navigation axes: {[ax.name for ax in signal_2d.axes_manager.navigation_axes]}")
print(f"  Signal axes: {[ax.name for ax in signal_2d.axes_manager.signal_axes]}")

# BaseSignal: No automatic classification - you control it
signal_base = hs.signals.BaseSignal(data_4d)
# Use transpose() to set signal dimensions correctly
signal_base = signal_base.transpose(signal_axes=1)  # Make last 1 axis signal
signal_base.axes_manager[0].name = 'nav_0'
signal_base.axes_manager[1].name = 'nav_1'
signal_base.axes_manager[2].name = 'nav_2'
signal_base.axes_manager[3].name = 'sig_0'

print(f"\nBaseSignal (1 signal dim): {signal_base}")
print(f"  Navigation axes: {[ax.name for ax in signal_base.axes_manager.navigation_axes]}")
print(f"  Signal axes: {[ax.name for ax in signal_base.axes_manager.signal_axes]}")

# %%
# **Practical Examples: Data Access Patterns**
# ============================================
#
# Understanding how HyperSpy's ordering affects data access

print("\n=== DATA ACCESS PATTERNS ===")

# Create a simple 3D dataset for demonstration
data_simple = np.zeros((2, 3, 4))
# Fill with pattern to see ordering effects
for i in range(2):
    for j in range(3):
        for k in range(4):
            data_simple[i, j, k] = 100*i + 10*j + k

signal_simple = hs.signals.Signal1D(data_simple)
signal_simple.axes_manager[0].name = 'nav_fast'   # size 3, was index 1 in NumPy
signal_simple.axes_manager[1].name = 'nav_slow'   # size 2, was index 0 in NumPy  
signal_simple.axes_manager[2].name = 'signal'     # size 4, was index 2 in NumPy

print(f"Signal shape: {signal_simple}")
print(f"NumPy data shape: {signal_simple.data.shape}")  # Still (2, 3, 4)

# Access patterns
print("\nData access examples:")
print(f"signal_simple.inav[0, 0].data = {signal_simple.inav[0, 0].data}")  # nav_fast=0, nav_slow=0
print(f"This corresponds to NumPy data[0, 0, :] = {data_simple[0, 0, :]}")

print(f"signal_simple.inav[1, 0].data = {signal_simple.inav[1, 0].data}")  # nav_fast=1, nav_slow=0  
print(f"This corresponds to NumPy data[0, 1, :] = {data_simple[0, 1, :]}")

print(f"signal_simple.inav[0, 1].data = {signal_simple.inav[0, 1].data}")  # nav_fast=0, nav_slow=1
print(f"This corresponds to NumPy data[1, 0, :] = {data_simple[1, 0, :]}")

# %%
# **Advanced Index Manipulation**
# ===============================
#
# HyperSpy provides special indexing for different axis orders

print("\n=== ADVANCED INDEXING ===")

# Natural order indexing (default)
print("Natural order indexing:")
print(f"  signal_simple.axes_manager[0] = {signal_simple.axes_manager[0].name}")  # nav_fast
print(f"  signal_simple.axes_manager[1] = {signal_simple.axes_manager[1].name}")  # nav_slow
print(f"  signal_simple.axes_manager[2] = {signal_simple.axes_manager[2].name}")  # signal

# Array order indexing using complex numbers (real + 3j)
print("\nArray order indexing (using complex numbers):")
print(f"  signal_simple.axes_manager[0+3j] = {signal_simple.axes_manager[0+3j].name}")  # nav_slow (array index 0)
print(f"  signal_simple.axes_manager[1+3j] = {signal_simple.axes_manager[1+3j].name}")  # nav_fast (array index 1)
print(f"  signal_simple.axes_manager[2+3j] = {signal_simple.axes_manager[2+3j].name}")  # signal (array index 2)

# Navigation-only indexing (real + 1j)
print("\nNavigation-only indexing:")
print(f"  signal_simple.axes_manager[0+1j] = {signal_simple.axes_manager[0+1j].name}")  # First nav axis in natural order
print(f"  signal_simple.axes_manager[1+1j] = {signal_simple.axes_manager[1+1j].name}")  # Second nav axis in natural order

# Signal-only indexing (real + 2j)  
print("\nSignal-only indexing:")
print(f"  signal_simple.axes_manager[0+2j] = {signal_simple.axes_manager[0+2j].name}")  # First signal axis

# %%
# **Axis Manipulation and Transposition**
# =======================================
#
# How HyperSpy's axis manipulation methods work with these conventions

print("\n=== AXIS MANIPULATION ===")

# Original signal
print(f"Original: {signal_simple}")
print(f"  Navigation: {[ax.name for ax in signal_simple.axes_manager.navigation_axes]}")
print(f"  Signal: {[ax.name for ax in signal_simple.axes_manager.signal_axes]}")

# .T swaps navigation and signal spaces
transposed = signal_simple.T
print(f"\nAfter .T: {transposed}")
print(f"  Navigation: {[ax.name for ax in transposed.axes_manager.navigation_axes]}")
print(f"  Signal: {[ax.name for ax in transposed.axes_manager.signal_axes]}")

# .transpose() with explicit control
# Move 'nav_slow' to signal space
custom_transpose = signal_simple.transpose(signal_axes=[1, 2])  # nav_slow and signal -> signal
print(f"\nCustom transpose (signal_axes=[1,2]): {custom_transpose}")
print(f"  Navigation: {[ax.name for ax in custom_transpose.axes_manager.navigation_axes]}")
print(f"  Signal: {[ax.name for ax in custom_transpose.axes_manager.signal_axes]}")

# %%
# **Real-World Implications**
# ===========================
#
# Why this matters for practical data analysis

print("\n=== REAL-WORLD IMPLICATIONS ===")

# Spectrum image example
print("Spectrum Image Example:")
print("NumPy data shape (Y, X, Energy): (50, 100, 1024)")
spectrum_image_data = np.random.random((50, 100, 1024))
spectrum_image = hs.signals.Signal1D(spectrum_image_data)

# Set meaningful names
spectrum_image.axes_manager[0].name = 'X'      # Was NumPy index 1
spectrum_image.axes_manager[1].name = 'Y'      # Was NumPy index 0  
spectrum_image.axes_manager[2].name = 'Energy' # Was NumPy index 2

print(f"HyperSpy interpretation: {spectrum_image}")
print(f"Navigation order: {[ax.name for ax in spectrum_image.axes_manager.navigation_axes]}")
print("This means:")
print("  - First navigation axis (X) corresponds to NumPy's second axis")
print("  - Second navigation axis (Y) corresponds to NumPy's first axis")
print("  - Signal axis (Energy) corresponds to NumPy's third axis")

# Image stack example  
print("\nImage Stack Example:")
print("NumPy data shape (Time, Y, X): (20, 64, 64)")
image_stack_data = np.random.random((20, 64, 64))
image_stack = hs.signals.Signal2D(image_stack_data)

image_stack.axes_manager[0].name = 'Time'   # Was NumPy index 0
image_stack.axes_manager[1].name = 'Y'      # Was NumPy index 1
image_stack.axes_manager[2].name = 'X'      # Was NumPy index 2

print(f"HyperSpy interpretation: {image_stack}")
print(f"Navigation: {[ax.name for ax in image_stack.axes_manager.navigation_axes]}")
print(f"Signal: {[ax.name for ax in image_stack.axes_manager.signal_axes]}")

# %%
# **Best Practices and Common Pitfalls**
# ======================================

print("\n=== BEST PRACTICES ===")

print("1. ALWAYS name your axes immediately after creating a signal:")
example_data = np.random.random((10, 20, 30))
example_signal = hs.signals.Signal1D(example_data)
# Good practice:
example_signal.axes_manager[0].name = 'scan_y'     # Size 20, was NumPy index 1
example_signal.axes_manager[1].name = 'scan_x'     # Size 10, was NumPy index 0
example_signal.axes_manager[2].name = 'wavelength' # Size 30, was NumPy index 2
print(f"  Well-named signal: {example_signal}")

print("\n2. Remember the reversal pattern:")
print("  NumPy (a, b, c, d) -> HyperSpy (c, b, a|d) for Signal1D")
print("  NumPy (a, b, c, d) -> HyperSpy (b, a|d, c) for Signal2D")

print("\n3. Use meaningful axis names, not generic ones:")
print("  Good: 'time', 'energy', 'x_position'")
print("  Bad: 'axis_0', 'dim_1', 'coordinate_2'")

print("\n4. Check your axis mapping after loading data:")
example_signal.axes_manager[0].name = 'y_scan'
example_signal.axes_manager[1].name = 'x_scan'  
example_signal.axes_manager[2].name = 'energy'
print(f"  Check: {example_signal}")
for i, ax in enumerate(example_signal.axes_manager._axes):
    print(f"    Natural[{i}]: {ax.name} (size {ax.size}) -> Array[{ax.index_in_array}]")

print("\n=== COMMON PITFALLS ===")
print("❌ Assuming HyperSpy axes match NumPy indices directly")
print("❌ Forgetting that navigation axes are reversed from NumPy order")
print("❌ Not naming axes, making debugging impossible")
print("❌ Confusing natural order with array order when accessing axes")
print("✅ Always verify axis mapping with named axes")
print("✅ Use HyperSpy's axis manipulation methods (.T, .transpose())")
print("✅ Think in terms of 'navigation' vs 'signal' rather than array indices")

# %%
# **Summary**
# ===========

print("\n=== SUMMARY ===")
print("HyperSpy's axis conventions:")
print("1. REVERSES navigation axes compared to NumPy")
print("2. SEPARATES axes into navigation (iterate over) and signal (analyze)")
print("3. DISPLAYS axes in 'natural order' (nav_0, nav_1, ..., sig_0, sig_1, ...)")
print("4. STORES data in original NumPy order but reinterprets axis meaning")
print("5. PROVIDES multiple indexing methods for different use cases")
print("\nThe key insight: HyperSpy optimizes for human thinking (x, y, z)")
print("while NumPy optimizes for computer memory layout (z, y, x)")
