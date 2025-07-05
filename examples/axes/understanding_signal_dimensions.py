"""
Signal Dimensions Basics
========================

This example demonstrates the core concept of navigation vs signal dimensions 
in HyperSpy through simple, visual examples.
"""

import numpy as np
import hyperspy.api as hs

# %%
# Create different types of signals to show dimension concepts

# Single spectrum: 0D navigation + 1D signal  
energy = np.linspace(0, 100, 200)
spectrum_data = 1000 * np.exp(-((energy - 50)**2) / 200) + 100
spectrum = hs.signals.Signal1D(spectrum_data)
spectrum.axes_manager[0].name = 'Energy'
spectrum.axes_manager[0].units = 'eV'

# Spectrum image: 2D navigation + 1D signal
spectrum_image_data = np.random.random((8, 10, 200))
spectrum_image = hs.signals.Signal1D(spectrum_image_data)
spectrum_image.axes_manager[0].name = 'y'
spectrum_image.axes_manager[1].name = 'x'
spectrum_image.axes_manager[2].name = 'Energy'

# Image stack: 1D navigation + 2D signal  
image_stack_data = np.random.random((15, 64, 64))
image_stack = hs.signals.Signal2D(image_stack_data)
image_stack.axes_manager[0].name = 'time'
image_stack.axes_manager[1].name = 'x'
image_stack.axes_manager[2].name = 'y'

# %%
# Visualize the dimension concepts using HyperSpy's native plotting

print("\n--- Demonstrating different signal dimension types ---")

# Single spectrum
spectrum.metadata.General.title = 'Single Spectrum (0D nav + 1D signal)'
spectrum.plot()

# Spectrum image 
spectrum_image.metadata.General.title = 'Spectrum Image (2D nav + 1D signal)'
spectrum_image.plot()

# Image stack
image_stack.metadata.General.title = 'Image Stack (1D nav + 2D signal)'
image_stack.plot()

# Show data shapes and navigation information
print(f"\nData shapes and dimensions:")
print(f"Single spectrum:")
print(f"  Data shape: {spectrum.data.shape}")
print(f"  Navigation shape: {spectrum.axes_manager.navigation_shape}")
print(f"  Signal shape: {spectrum.axes_manager.signal_shape}")

print(f"\nSpectrum image:")
print(f"  Data shape: {spectrum_image.data.shape}")
print(f"  Navigation shape: {spectrum_image.axes_manager.navigation_shape}")
print(f"  Signal shape: {spectrum_image.axes_manager.signal_shape}")

print(f"\nImage stack:")
print(f"  Data shape: {image_stack.data.shape}")
print(f"  Navigation shape: {image_stack.axes_manager.navigation_shape}")
print(f"  Signal shape: {image_stack.axes_manager.signal_shape}")

# %%
# **Signal Dimension Examples Summary**
#
# Understanding how HyperSpy interprets signal dimensions is fundamental to effective data analysis.

# **Spectrum (0D navigation + 1D signal):**
# - Navigation shape: () - no scanning dimensions  
# - Signal shape: (200,) - energy/spectral dimension
# - Interpretation: Single spectrum measurement

# **Spectrum Image (2D navigation + 1D signal):**  
# - Navigation shape: (8, 10) - spatial scanning grid
# - Signal shape: (200,) - energy/spectral dimension  
# - Interpretation: Spectrum acquired at each spatial position

# **Image Stack (1D navigation + 2D signal):**
# - Navigation shape: (15,) - time series or depth dimension
# - Signal shape: (32, 32) - 2D image at each navigation point
# - Interpretation: Series of images (e.g., time-lapse, depth series)
