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
import matplotlib.pyplot as plt

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
test_signal.data += np.random.normal(0, 5, test_signal.data.shape)

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
# Visualize the interactive operations
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Original signal (first spectrum)
axes[0,0].plot(test_signal.inav[0].data)
axes[0,0].set_title('Original signal (first spectrum)')
axes[0,0].set_xlabel('Channel')
axes[0,0].set_ylabel('Intensity')

# Interactive operations results
axes[0,1].plot(interactive_max.data)
axes[0,1].set_title('Maximum along signal axis')
axes[0,1].set_xlabel('Navigation index')
axes[0,1].set_ylabel('Max value')

axes[1,0].plot(interactive_std.data)
axes[1,0].set_title('Standard deviation along signal axis')
axes[1,0].set_xlabel('Navigation index')
axes[1,0].set_ylabel('Std value')

axes[1,1].plot(interactive_mean.data)
axes[1,1].set_title('Mean along signal axis')
axes[1,1].set_xlabel('Navigation index')
axes[1,1].set_ylabel('Mean value')

plt.tight_layout()
plt.show()

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
# Plot the custom functions results
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(snr.data)
axes[0].set_title('Signal-to-Noise Ratio')
axes[0].set_xlabel('Position')
axes[0].set_ylabel('SNR')

axes[1].plot(peak_pos.data)
axes[1].set_title('Peak Position')
axes[1].set_xlabel('Position')
axes[1].set_ylabel('Channel')

axes[2].plot(integrated.data)
axes[2].set_title('Integrated Intensity')
axes[2].set_xlabel('Position')
axes[2].set_ylabel('Counts')

plt.tight_layout()
plt.show()

print("\nInteractive operations example completed!")
print("Key points:")
print("- Use max(), min(), mean(), std() for basic statistics")
print("- Apply custom functions for domain-specific analysis")
print("- Recompute when data changes to maintain consistency")
print("- Visualize results to understand data behavior")
