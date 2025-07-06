"""
Signal Dimensions Basics
========================

This example demonstrates the core concept of navigation vs signal dimensions 
in HyperSpy through simple, visual examples. Understanding this print("\n• Spectrum Image (2D nav + 1D signal):")
print("  - Navigation: (10, 8) - x,y spatial grid (REVERSED from NumPy!)")
print("  - Signal: (200,) - energy/spectral dimension")  
print("  - Use case: Spectroscopy mapping, hyperspectral imaging")
print("  - NumPy shape was (8, 10, 200), HyperSpy shows (10, 8|200)")inction is
fundamental to effective HyperSpy usage.

Key concepts:
- Navigation axes: where you navigate/scan (spatial positions, time points, etc.)
- Signal axes: what you measure at each navigation position (spectrum, image, etc.)
- How HyperSpy interprets multidimensional data differently from NumPy
"""

import numpy as np
import hyperspy.api as hs

# %%
# **Understanding HyperSpy's Axis Interpretation**
# ================================================
# HyperSpy interprets multidimensional data differently from NumPy

print("=== HyperSpy vs NumPy Axis Interpretation ===")

# Create a 3D NumPy array
numpy_array = np.random.random((10, 20, 100))
print(f"NumPy array shape: {numpy_array.shape}")

# When we create a HyperSpy Signal1D from this data:
signal = hs.signals.Signal1D(numpy_array)
print(f"HyperSpy Signal1D: {signal}")
print(f"Navigation shape: {signal.axes_manager.navigation_shape}")
print(f"Signal shape: {signal.axes_manager.signal_shape}")

print("\nKey insight: HyperSpy treats the LAST axis as signal, others as navigation")
print("NumPy (10, 20, 100) -> HyperSpy (20, 10|100)")
print("Navigation axes are REVERSED: NumPy (10, 20) -> HyperSpy (20, 10)")

# %%
# **Different Signal Types with Meaningful Examples**
# ===================================================

# Single spectrum: 0D navigation + 1D signal  
print("\n=== Single Spectrum (0D navigation + 1D signal) ===")
energy = np.linspace(0, 100, 200)
spectrum_data = 1000 * np.exp(-((energy - 50)**2) / 200) + 100
spectrum = hs.signals.Signal1D(spectrum_data)
spectrum.axes_manager[0].name = 'Energy'
spectrum.axes_manager[0].units = 'eV'
spectrum.axes_manager[0].scale = 0.5

print(f"Single spectrum: {spectrum}")
print("Interpretation: Just one spectrum, no scanning dimensions")

# Spectrum image: 2D navigation + 1D signal
print("\n=== Spectrum Image (2D navigation + 1D signal) ===")
spectrum_image_data = np.random.random((8, 10, 200))
# Add realistic spectral features
for i in range(8):
    for j in range(10):
        # Create peaks that vary with position
        peak_pos = 50 + 10 * np.sin(i * np.pi / 8) * np.cos(j * np.pi / 10)
        spectrum_image_data[i, j, :] += 500 * np.exp(-((energy - peak_pos)**2) / 100)

spectrum_image = hs.signals.Signal1D(spectrum_image_data)
# Important: HyperSpy reverses the navigation axes
spectrum_image.axes_manager[0].name = 'x'  # Size 10, was NumPy index 1
spectrum_image.axes_manager[0].units = 'nm'
spectrum_image.axes_manager[1].name = 'y'  # Size 8, was NumPy index 0  
spectrum_image.axes_manager[1].units = 'nm'
spectrum_image.axes_manager[2].name = 'Energy'  # Size 200, was NumPy index 2
spectrum_image.axes_manager[2].units = 'eV'
spectrum_image.axes_manager[2].scale = 0.5

print(f"Spectrum image: {spectrum_image}")
print("Interpretation: 2D spatial scan, spectrum at each position")
print("Note: Navigation axes are reversed from NumPy order!")

# Image stack: 1D navigation + 2D signal  
print("\n=== Image Stack (1D navigation + 2D signal) ===")
image_stack_data = np.random.random((15, 64, 64))
# Add time-dependent features
for t in range(15):
    # Moving feature across the image
    center_x = 32 + 10 * np.sin(t * 2 * np.pi / 15)
    center_y = 32 + 10 * np.cos(t * 2 * np.pi / 15)
    y, x = np.ogrid[:64, :64]
    mask = (x - center_x)**2 + (y - center_y)**2 < 100
    image_stack_data[t, mask] += 0.5

image_stack = hs.signals.Signal2D(image_stack_data)
image_stack.axes_manager[0].name = 'time'
image_stack.axes_manager[0].units = 's'
image_stack.axes_manager[1].name = 'y'
image_stack.axes_manager[1].units = 'μm'
image_stack.axes_manager[2].name = 'x'
image_stack.axes_manager[2].units = 'μm'

print(f"Image stack: {image_stack}")
print("Interpretation: Time series of 2D images")

# %%
# **Visualize the dimension concepts using HyperSpy's native plotting**
# =====================================================================

print("\n=== Plotting Different Signal Types ===")

# Single spectrum
spectrum.metadata.General.title = 'Single Spectrum (0D nav + 1D signal)'
spectrum.plot()

# Spectrum image 
spectrum_image.metadata.General.title = 'Spectrum Image (2D nav + 1D signal)'
spectrum_image.plot()

# Image stack
image_stack.metadata.General.title = 'Image Stack (1D nav + 2D signal)'
image_stack.plot()

# %%
# **Detailed Analysis of Signal Dimensions**
# ==========================================

print("\n=== Detailed Dimension Analysis ===")

signals = [
    ("Single spectrum", spectrum),
    ("Spectrum image", spectrum_image), 
    ("Image stack", image_stack)
]

for name, sig in signals:
    print(f"\n{name}:")
    print(f"  NumPy data shape: {sig.data.shape}")
    print(f"  HyperSpy display: {sig}")
    print(f"  Navigation shape: {sig.axes_manager.navigation_shape}")
    print(f"  Signal shape: {sig.axes_manager.signal_shape}")
    print(f"  Navigation dimension: {sig.axes_manager.navigation_dimension}")
    print(f"  Signal dimension: {sig.axes_manager.signal_dimension}")
    print(f"  Axis names (natural order): {[ax.name for ax in sig.axes_manager._axes]}")
    if sig.axes_manager.navigation_dimension > 0:
        print(f"  Navigation axes: {[ax.name for ax in sig.axes_manager.navigation_axes]}")
        print(f"  Signal axes: {[ax.name for ax in sig.axes_manager.signal_axes]}")
    print("  Key: Navigation axes are in reversed order from NumPy!")

# %%
# **Axis Navigation Examples**  
# =============================

print("\n=== Navigation Examples ===")

# For spectrum image, navigate to different positions
if spectrum_image.axes_manager.navigation_dimension > 0:
    # Get spectrum at position (2, 3) - remember navigation order is reversed!
    spectrum_at_pos = spectrum_image.inav[2, 3]  # x=2, y=3 in HyperSpy natural order
    print(f"Spectrum at position x=2, y=3: {spectrum_at_pos}")
    print(f"  This gives us: {spectrum_at_pos.axes_manager.signal_shape} spectrum")
    print(f"  Note: inav[2, 3] means x=2 (fast), y=3 (slow) in HyperSpy")

# For image stack, navigate to different time points
if image_stack.axes_manager.navigation_dimension > 0:
    # Get image at time point 5
    image_at_time = image_stack.inav[5]
    print(f"Image at time 5: {image_at_time}")
    print(f"  This gives us: {image_at_time.axes_manager.signal_shape} image")

# %%
# **Understanding Signal Space vs Navigation Space**
# ==================================================

print("\n=== Signal vs Navigation Space Concepts ===")
print("NAVIGATION SPACE:")
print("- Dimensions you 'navigate' through or scan over")
print("- Examples: spatial positions (x,y), time points, temperatures")
print("- You typically iterate through these dimensions for analysis")
print()
print("SIGNAL SPACE:")  
print("- The actual measurement/data at each navigation position")
print("- Examples: spectrum (energy), image (height×width), single value")
print("- This is what you analyze/process at each navigation point")
print()
print("SIGNAL TYPES:")
print("- Signal1D: 1D signal (e.g., spectrum, line profile)")
print("- Signal2D: 2D signal (e.g., image, 2D map)")
print("- BaseSignal: 0D signal (e.g., single value at each nav position)")

# %%
# **Signal Dimension Examples Summary**
# ====================================

print("\n=== Summary ===")
print("Signal dimensions determine how HyperSpy interprets and processes your data:")
print()
print("• Single Spectrum (0D nav + 1D signal):")
print("  - Navigation: () - no scanning")  
print("  - Signal: (200,) - energy/spectral dimension")
print("  - Use case: Single spectrum analysis")
print()
print("• Spectrum Image (2D nav + 1D signal):")
print("  - Navigation: (8, 10) - spatial y,x grid")
print("  - Signal: (200,) - energy/spectral dimension")  
print("  - Use case: Spectroscopy mapping, hyperspectral imaging")
print()
print("• Image Stack (1D nav + 2D signal):")
print("  - Navigation: (15,) - time or z-dimension")
print("  - Signal: (64, 64) - 2D image")
print("  - Use case: Time-lapse imaging, tomography")
print()
print("The key insight: HyperSpy's signal classification determines")
print("which dimensions are processed vs which are iterated over!")
print("REMEMBER: Navigation axes are in REVERSED order from NumPy!")
