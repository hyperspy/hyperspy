"""
Simple Chained Operations in HyperSpy
======================================

This example demonstrates how to chain operations in HyperSpy by applying
multiple processing steps to signals and maintaining consistency when
data changes.
"""

import numpy as np
import hyperspy.api as hs

# %%
# **Simple Chained Operations Demonstration**
#
# This example shows how to chain operations in HyperSpy by applying multiple 
# processing steps to signals and maintaining consistency when data changes.

# %%
# **Creating test data for chained operations**
#
# We'll create a 2D spectrum image with spatial navigation and energy signal axes.

# Create a 2D spectrum image (spatial navigation, energy signal)
n_nav_x, n_nav_y = 10, 10
n_signal = 100

# Create synthetic spectrum data with different peaks at different positions
data = np.random.random((n_nav_x, n_nav_y, n_signal)) * 10

# Add some peaks that vary spatially
x_coords, y_coords = np.ogrid[:n_nav_x, :n_nav_y]
energy_axis = np.linspace(0, 50, n_signal)

for i in range(n_nav_x):
    for j in range(n_nav_y):
        # Add a peak that shifts with position
        peak_center = 20 + 5 * (i + j) / (n_nav_x + n_nav_y)
        peak_height = 50 + 20 * np.sin(i * np.pi / n_nav_x) * np.cos(j * np.pi / n_nav_y)
        peak_width = 3
        
        # Gaussian peak
        peak = peak_height * np.exp(-((energy_axis - peak_center) ** 2) / (2 * peak_width ** 2))
        data[i, j, :] += peak

# Create the signal
s = hs.signals.Signal1D(data)
# Set navigation axes (X, Y spatial dimensions)
s.axes_manager.navigation_axes.set(name=['X', 'Y'])
# Set signal axis (Energy spectrum)
s.axes_manager.signal_axes.set(
    name=['Energy'],
    units=['eV'],
    scale=[0.5]
)
s.metadata.General.title = "Synthetic spectrum image"

# Signal characteristics successfully created with spatial navigation and energy axis
# Created spectrum image with shape (10, 10, 100) for demonstration

# %%
# **Calculating basic statistics**
#
# We'll calculate various statistics along the signal axis to create maps showing
# spatial variation in the spectral data.

# Calculate statistics along the signal axis
max_intensity = s.max(axis='Energy')  # Maximum intensity at each position
mean_intensity = s.mean(axis='Energy')  # Mean intensity at each position
total_intensity = s.sum(axis='Energy')  # Total intensity at each position

# Successfully created statistical maps:
# - Maximum intensity at each spatial position
# - Mean intensity for background analysis  
# - Total intensity for integrated analysis

# %%
# **Creating derived quantities**
#
# We'll create more sophisticated analysis results by combining basic statistics
# and extracting meaningful spectral information.

# Signal-to-noise ratio (approximate)
snr_map = mean_intensity / s.std(axis='Energy')
# Created SNR map by calculating mean/standard deviation ratio

# Peak position analysis - find channel with maximum intensity at each position
peak_position_map = s.indexmax(axis='Energy')

# Peak energy conversion - convert peak positions to physical energy values
energy_axis_vals = s.axes_manager[2].axis
peak_energy_map = peak_position_map.deepcopy()
peak_energy_map.data = energy_axis_vals[peak_position_map.data]
peak_energy_map.metadata.General.title = "Peak energy map"

# %%
# **Data-dependent processing**
#
# Demonstrating how to apply processing steps that depend on the data content
# and maintain consistency across related calculations.

# Crop the signal to a specific energy range
energy_min, energy_max = 15, 35  # eV
s_cropped = s.isig[energy_min:energy_max]
# Energy range cropping applied: 15-35 eV window selected for analysis

# Energy integration in specific ranges for ratio analysis
low_energy_sum = s.isig[0:20].sum(axis='Energy')
high_energy_sum = s.isig[30:50].sum(axis='Energy')

# Calculate ratio
energy_ratio = high_energy_sum / (low_energy_sum + 1e-10)  # Add small value to avoid division by zero

# %%
# **Updating after data modification**
#
# Demonstrating how chained operations respond when the original data is modified.

# Store original values
original_max = max_intensity.data.copy()
original_mean = mean_intensity.data.copy()

# Original data ranges stored for comparison:
print(f"Original mean range: [{original_mean.min():.2f}, {original_mean.max():.2f}]")

# Modify the original signal using signal arithmetic (preserves metadata)
s = s * 1.5  # Increase intensity by 50% - creates new signal with updated title

# Recalculate statistics
new_max_intensity = s.max(axis='Energy')
new_mean_intensity = s.mean(axis='Energy')

print(f"New max range: [{np.min(new_max_intensity.data):.2f}, {np.max(new_max_intensity.data):.2f}]")
print(f"New mean range: [{np.min(new_mean_intensity.data):.2f}, {np.max(new_mean_intensity.data):.2f}]")

# Verify the scaling using signal arithmetic
ratio_max = new_max_intensity / original_max  # Signal division preserves metadata
ratio_mean = new_mean_intensity / original_mean
print(f"Max intensity scaling factor: {np.mean(ratio_max.data):.2f} (expected: 1.50)")
print(f"Mean intensity scaling factor: {np.mean(ratio_mean.data):.2f} (expected: 1.50)")

# %%
# Step 5: Visualization using HyperSpy plotting
print("\n6. VISUALIZATION")
print("-" * 20)

# Plot spectrum at center position using HyperSpy
center_spectrum = s.inav[n_nav_x//2, n_nav_y//2]
center_spectrum.plot()

# Use HyperSpy's plot_images for the analysis maps
hs.plot.plot_images([new_max_intensity, peak_energy_map, snr_map, energy_ratio],
                   label=['Max Intensity Map', 'Peak Energy Map', 'Signal-to-Noise Ratio', 'High/Low Energy Ratio'],
                   cmap=['viridis', 'plasma', 'RdYlBu', 'coolwarm'],
                   colorbar=True)

# Plot cropped spectrum using HyperSpy
cropped_spectrum_signal = s_cropped.inav[n_nav_x//2, n_nav_y//2]
cropped_spectrum_signal.plot()

# %%
# Summary
print("\n" + "=" * 70)
print("SIMPLE CHAINED OPERATIONS COMPLETED!")
print("=" * 70)
print("\nKey concepts demonstrated:")
print("✅ Basic statistical operations on signals")
print("✅ Deriving new quantities from existing signals")
print("✅ Energy/spectral range operations")
print("✅ Data-dependent processing")
print("✅ Updating calculations after data modification")
print("✅ Comprehensive visualization")
print("\nThis pattern can be extended to more complex processing chains")
print("where each step depends on the results of previous steps.")
