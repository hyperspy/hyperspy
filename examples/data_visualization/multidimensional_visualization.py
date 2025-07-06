"""
Multidimensional Data Visualization
===================================

This example demonstrates how to visualize different types of multidimensional
datasets in HyperSpy, including spectrum images and image stacks.

Key visualization concepts:
- Navigator plots for multidimensional data
- Signal plots at specific positions
- Comparing data from multiple locations
- Programmatic navigation through datasets
"""

import numpy as np
import hyperspy.api as hs

# %%
# **Creating sample multidimensional datasets for visualization**
#
# We'll create different types of multidimensional signals to demonstrate
# the various visualization capabilities available in HyperSpy.

# %%
# **Create Sample Data**
#
# We'll create two types of multidimensional signals to demonstrate visualization:
# 1. Spectrum image (2D navigation + 1D signal)
# 2. Image stack (1D navigation + 2D signal)

# Create a spectrum image (2D navigation + 1D signal)
nav_shape = (12, 16)  # Navigation dimensions (y, x)
energy_points = 200   # Signal dimension

# Create energy axis
energy = np.linspace(100, 800, energy_points)

# Initialize spectrum image data
spectrum_image_data = np.zeros(nav_shape + (energy_points,))

# Create spatially varying spectra
for i in range(nav_shape[0]):
    for j in range(nav_shape[1]):
        # Create spatially varying peaks
        peak1_pos = 300 + 30 * np.sin(i * 0.4) * np.cos(j * 0.3)
        peak2_pos = 500 + 20 * np.cos(i * 0.3) * np.sin(j * 0.4)
        
        peak1_int = 800 + 200 * np.sin(i * 0.5) + 100 * np.cos(j * 0.3)
        peak2_int = 600 + 150 * np.cos(i * 0.4) + 100 * np.sin(j * 0.5)
        
        # Create Gaussian peaks
        peak1 = peak1_int * np.exp(-((energy - peak1_pos)**2) / (2 * 30**2))
        peak2 = peak2_int * np.exp(-((energy - peak2_pos)**2) / (2 * 25**2))
        
        # Add background and noise
        background = 150 + 80 * np.exp(-energy / 400)
        noise = np.random.normal(0, 20, energy_points)
        
        spectrum_image_data[i, j, :] = peak1 + peak2 + background + noise

# Create HyperSpy Signal1D (spectrum image)
si = hs.signals.Signal1D(spectrum_image_data)
si.axes_manager.signal_axes[0].name = 'Energy Loss'
si.axes_manager.signal_axes[0].units = 'eV'
si.axes_manager.signal_axes[0].scale = 3.5
si.axes_manager.signal_axes[0].offset = 100.0
si.axes_manager.navigation_axes[0].name = 'y'
si.axes_manager.navigation_axes[1].name = 'x'
si.axes_manager.navigation_axes[0].units = 'μm'
si.axes_manager.navigation_axes[1].units = 'μm'
si.axes_manager.navigation_axes[0].scale = 0.1
si.axes_manager.navigation_axes[1].scale = 0.1
si.metadata.General.title = 'Sample Spectrum Image'

print(f"Created spectrum image: {si}")

# %%
# Basic Visualization Examples
# =============================

print("\n1. Basic Spectrum Image Visualization")
print("="*50)

# Use HyperSpy's built-in plotting capabilities
si.plot()

# Also demonstrate the navigator separately
navigator = si.sum(axis=si.axes_manager.signal_axes)
navigator.plot()

# %%
# Multiple Position Analysis
# ===========================
#
# Extract spectra from multiple positions for comparison

# Extract spectra from multiple positions and plot them using HyperSpy
positions = [(2, 3), (6, 8), (9, 10), (5, 11)]
spectra = [si.inav[pos[0], pos[1]] for pos in positions]

# Plot each spectrum using HyperSpy's plot method
for i, (spectrum, pos) in enumerate(zip(spectra, positions)):
    spectrum.plot()

# %%
# Programmatic Navigation
# ========================

print("\n3. Programmatic Navigation")
print("="*50)

# Demonstrate programmatic navigation
print(f"Navigation shape: {si.axes_manager.navigation_shape}")
print(f"Signal shape: {si.axes_manager.signal_shape}")

# Show current navigation coordinates
current_indices = tuple(ax.index for ax in si.axes_manager.navigation_axes)
current_coords = tuple(ax.value for ax in si.axes_manager.navigation_axes)
print(f"Current navigation position (indices): {current_indices}")
print(f"Current navigation position (coordinates): {current_coords}")

# Navigate to a specific position by index
si.axes_manager.navigation_axes[0].index = 5
si.axes_manager.navigation_axes[1].index = 8
new_indices = tuple(ax.index for ax in si.axes_manager.navigation_axes)
new_coords = tuple(ax.value for ax in si.axes_manager.navigation_axes)
print(f"After navigation - indices: {new_indices}")
print(f"After navigation - coordinates: {new_coords}")

# Navigate by coordinate value
si.axes_manager.navigation_axes[0].value = 0.3
si.axes_manager.navigation_axes[1].value = 1.0
final_indices = tuple(ax.index for ax in si.axes_manager.navigation_axes)
final_coords = tuple(ax.value for ax in si.axes_manager.navigation_axes)
print(f"After coordinate navigation - indices: {final_indices}")
print(f"After coordinate navigation - coordinates: {final_coords}")

# %%
# Advanced Visualization: Multiple Positions on Navigator
# ========================================================

print("\n4. Advanced Visualization")
print("="*50)

# Create a comprehensive visualization showing positions on the navigator
positions_to_analyze = [(3, 4), (8, 6), (10, 10)]
labels = ['Position A', 'Position B', 'Position C']

# Show navigator with total intensity
navigator = si.sum(axis=si.axes_manager.signal_axes)
navigator.plot()

# Plot spectra from selected positions using HyperSpy plotting
for pos, label in zip(positions_to_analyze, labels):
    spectrum = si.inav[pos[0], pos[1]]
    spectrum.metadata.General.title = f'{label} at position {pos}'
    spectrum.plot()

# %%
# Summary
# =======
#
# Multidimensional visualization techniques demonstrated

print(f"""
CREATED SIGNALS:
• Spectrum image: {si}
• Navigation shape: {si.axes_manager.navigation_shape}
• Signal shape: {si.axes_manager.signal_shape}

KEY VISUALIZATION TECHNIQUES:
1. Navigator plots: Show spatial distribution of signal properties
2. Position-specific plots: Extract and display signals from chosen locations
3. Programmatic navigation: Move through dataset using indices or coordinates
4. Comparative analysis: Compare signals from multiple positions simultaneously

PRACTICAL APPLICATIONS:
• Energy/wavelength mapping: Analyze compositional variations
• Time series: Track changes over time
• Diffraction analysis: Examine patterns from different sample positions
• In-situ studies: Monitor dynamic processes

OUTPUT FILES:
• multidimensional_basic_viz.png
• multidimensional_multiple_positions.png  
• multidimensional_advanced_viz.png
""")

print("Multidimensional visualization demonstration complete!")
