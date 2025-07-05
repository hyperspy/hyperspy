"""
Simple Chained Operations in HyperSpy
======================================

This example demonstrates how to chain operations in HyperSpy by applying
multiple processing steps to signals and maintaining consistency when
data changes.
"""

import numpy as np
import matplotlib.pyplot as plt
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
s.axes_manager[0].name = "X"
s.axes_manager[1].name = "Y" 
s.axes_manager[2].name = "Energy"
s.axes_manager[2].units = "eV"
s.axes_manager[2].scale = 0.5
s.metadata.General.title = "Synthetic spectrum image"

# Signal characteristics successfully created with spatial navigation and energy axis
# Created spectrum image with shape (10, 10, 100) for demonstration

# %%
# **Calculating basic statistics**
#
# We'll calculate various statistics along the signal axis to create maps showing
# spatial variation in the spectral data.

# Calculate statistics along the signal axis
max_intensity = s.max(axis=-1)  # Maximum intensity at each position
mean_intensity = s.mean(axis=-1)  # Mean intensity at each position
total_intensity = s.sum(axis=-1)  # Total intensity at each position

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
snr_map = mean_intensity / s.std(axis=-1)
# Created SNR map by calculating mean/standard deviation ratio

# Peak position analysis - find channel with maximum intensity at each position
peak_position_map = s.indexmax(axis=-1)

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
low_energy_sum = s.isig[0:20].sum(axis=-1)
high_energy_sum = s.isig[30:50].sum(axis=-1)

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

# Modify the original signal
s.data *= 1.5  # Increase intensity by 50%

# Recalculate statistics
new_max_intensity = s.max(axis=-1)
new_mean_intensity = s.mean(axis=-1)

print(f"New max range: [{new_max_intensity.data.min():.2f}, {new_max_intensity.data.max():.2f}]")
print(f"New mean range: [{new_mean_intensity.data.min():.2f}, {new_mean_intensity.data.max():.2f}]")

# Verify the scaling
ratio_max = new_max_intensity.data / original_max
ratio_mean = new_mean_intensity.data / original_mean
print(f"Max intensity scaling factor: {ratio_max.mean():.2f} (expected: 1.50)")
print(f"Mean intensity scaling factor: {ratio_mean.mean():.2f} (expected: 1.50)")

# %%
# Step 5: Visualization
print("\n6. VISUALIZATION")
print("-" * 20)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot spectrum at center position
center_spectrum = s.inav[n_nav_x//2, n_nav_y//2].data
axes[0, 0].plot(energy_axis, center_spectrum)
axes[0, 0].set_title('Spectrum at center position')
axes[0, 0].set_xlabel('Energy (eV)')
axes[0, 0].set_ylabel('Intensity')

# Plot max intensity map
im1 = axes[0, 1].imshow(new_max_intensity.data, origin='lower', cmap='viridis')
axes[0, 1].set_title('Max Intensity Map')
axes[0, 1].set_xlabel('Y position')
axes[0, 1].set_ylabel('X position')
plt.colorbar(im1, ax=axes[0, 1])

# Plot peak energy map
im2 = axes[0, 2].imshow(peak_energy_map.data, origin='lower', cmap='plasma')
axes[0, 2].set_title('Peak Energy Map')
axes[0, 2].set_xlabel('Y position')
axes[0, 2].set_ylabel('X position')
plt.colorbar(im2, ax=axes[0, 2])

# Plot SNR map
im3 = axes[1, 0].imshow(snr_map.data, origin='lower', cmap='RdYlBu')
axes[1, 0].set_title('Signal-to-Noise Ratio')
axes[1, 0].set_xlabel('Y position')
axes[1, 0].set_ylabel('X position')
plt.colorbar(im3, ax=axes[1, 0])

# Plot energy ratio map
im4 = axes[1, 1].imshow(energy_ratio.data, origin='lower', cmap='coolwarm')
axes[1, 1].set_title('High/Low Energy Ratio')
axes[1, 1].set_xlabel('Y position')
axes[1, 1].set_ylabel('X position')
plt.colorbar(im4, ax=axes[1, 1])

# Plot cropped spectrum
cropped_spectrum = s_cropped.inav[n_nav_x//2, n_nav_y//2].data
cropped_energy = s_cropped.axes_manager[2].axis
axes[1, 2].plot(cropped_energy, cropped_spectrum)
axes[1, 2].set_title('Cropped spectrum (15-35 eV)')
axes[1, 2].set_xlabel('Energy (eV)')
axes[1, 2].set_ylabel('Intensity')

plt.tight_layout()
plt.show()

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
