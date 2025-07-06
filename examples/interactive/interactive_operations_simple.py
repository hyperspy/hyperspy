"""
Interactive Operations
====================

This example demonstrates interactive-style operations in HyperSpy
for live data analysis and responsive plotting.
"""

# %%
# **Creating test signal for interactive operations**
#
# We'll create a Signal1D with navigation dimensions to demonstrate how 
# interactive operations work in HyperSpy.

import numpy as np
import hyperspy.api as hs

# Create a 2D signal with navigation dimensions
data = np.random.random((10, 100)) * 100 + np.linspace(0, 50, 100)
test_signal = hs.signals.Signal1D(data)
test_signal.axes_manager.signal_axes[0].name = 'Energy'
test_signal.axes_manager.signal_axes[0].units = 'eV'
test_signal.axes_manager.navigation_axes[0].name = 'Position'

# **Signal characteristics:**
# - Navigation shape: (10,) - 10 different positions
# - Signal shape: (100,) - 100 energy channels per spectrum
# - Total data points: 1,000

# %%
# **Computing statistics along signal axis**
#
# These operations demonstrate how to compute statistical measures along specific axes.
# Statistics are computed along the signal axis (energy), producing navigation-shaped results.

# These operations compute statistics along the signal axis
interactive_max = test_signal.max(axis=-1)  # Maximum along signal axis
interactive_std = test_signal.std(axis=-1)  # Standard deviation along signal axis  
interactive_mean = test_signal.mean(axis=-1)  # Mean along signal axis

print(f"Max values shape: {interactive_max.data.shape}")
print(f"Mean values shape: {interactive_mean.data.shape}")
print(f"Std values shape: {interactive_std.data.shape}")

# Print some statistics
print(f"Max of max values: {interactive_max.data.max():.2f}")
print(f"Mean of mean values: {interactive_mean.data.mean():.2f}")
print(f"Mean of std values: {interactive_std.data.mean():.2f}")

# %%
# Demonstrate data changes and recomputation
print("\n--- Modifying data and recomputing ---")

# Save original statistics
original_max = interactive_max.data.copy()
original_mean = interactive_mean.data.copy()

# Modify the signal data (add noise)
test_signal += np.random.normal(0, 5, test_signal.data.shape)  # Must use .data for shape

# Recompute statistics
new_max = test_signal.max(axis=-1)
new_mean = test_signal.mean(axis=-1) 
new_std = test_signal.std(axis=-1)

print(f"Original max range: [{original_max.min():.2f}, {original_max.max():.2f}]")
print(f"New max range: [{new_max.data.min():.2f}, {new_max.data.max():.2f}]")
print(f"Original mean range: [{original_mean.min():.2f}, {original_mean.max():.2f}]")
print(f"New mean range: [{new_mean.data.min():.2f}, {new_mean.data.max():.2f}]")

# Update the interactive results for plotting
interactive_max = new_max
interactive_mean = new_mean
interactive_std = new_std

# %%
# Visualize the interactive operations using HyperSpy's native plotting
print("\n--- Visualizing interactive operations ---")

# Original signal (first spectrum)
first_spectrum = test_signal.inav[0]
first_spectrum.metadata.General.title = 'Original signal (first spectrum)'
first_spectrum.plot()

# Interactive operations results - convert to Signal1D for proper plotting
interactive_max.metadata.General.title = 'Maximum along signal axis'
interactive_max.plot()

interactive_std.metadata.General.title = 'Standard deviation along signal axis'
interactive_std.plot()

interactive_mean.metadata.General.title = 'Mean along signal axis'
interactive_mean.plot()

# %%
# Example of interactive-style functions
print("\n--- Custom interactive-style functions ---")

def compute_signal_to_noise(signal):
    """Compute signal-to-noise ratio"""
    signal_mean = signal.mean(axis=-1)
    signal_std = signal.std(axis=-1)
    return signal_mean / signal_std

def compute_peak_position(signal):
    """Find peak position for each spectrum"""
    return signal.indexmax(axis=-1)

def compute_integrated_intensity(signal, start_idx=20, end_idx=80):
    """Compute integrated intensity over a range"""
    return signal.isig[start_idx:end_idx].sum(axis=-1)

# Apply these functions
snr = compute_signal_to_noise(test_signal)
peak_pos = compute_peak_position(test_signal)
integrated = compute_integrated_intensity(test_signal)

print(f"Signal-to-noise ratio - mean: {snr.data.mean():.2f}")
print(f"Peak positions - range: [{peak_pos.data.min()}, {peak_pos.data.max()}]")
print(f"Integrated intensity - mean: {integrated.data.mean():.2f}")

# %%
# Plot the custom functions results using HyperSpy's native plotting
print("\n--- Visualizing custom analysis results ---")

snr.metadata.General.title = 'Signal-to-Noise Ratio'
snr.plot()

peak_pos.metadata.General.title = 'Peak Position'  
peak_pos.plot()

integrated.metadata.General.title = 'Integrated Intensity'
integrated.plot()

print("\nInteractive operations example completed!")
print("Key points:")
print("- Use max(), min(), mean(), std() for basic statistics")
print("- Apply custom functions for domain-specific analysis")
print("- Recompute when data changes to maintain consistency")
print("- Visualize results to understand data behavior")
