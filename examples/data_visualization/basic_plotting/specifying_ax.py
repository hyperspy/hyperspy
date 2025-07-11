"""
Specifying Matplotlib Axis for HyperSpy Plots
==============================================

This example demonstrates how to integrate HyperSpy plotting functions with
custom matplotlib figure layouts by specifying exact axes for plot placement.
This is essential for creating complex multi-panel figures and publications.
"""

# %%
# **Import required libraries**
#
# We'll use HyperSpy's plotting functions combined with matplotlib's
# figure management capabilities for precise control over plot layouts.

import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np

# %%
# **Understanding Custom Axis Specification**
#
# When creating complex figures, you often need to place HyperSpy plots
# in specific locations within a larger figure. This is done by:
# 1. Creating a figure with multiple subplots using matplotlib
# 2. Specifying which axis (subplot) to use for each HyperSpy plot
# 3. Using HyperSpy's ax parameter to target specific axes

# %%
# **Working with Signal2D Data**
#
# First, let's create some 2D signal data for demonstration.
# We'll create two complementary signals to show side-by-side comparison.

# Create a simple 2D signal with structured data
s = hs.signals.Signal2D(np.arange(100).reshape(10, 10))
# Create a complementary signal (negative values)
s2 = -s

print(f"Signal 1 shape: {s.data.shape}")
print(f"Signal 1 data range: {s.data.min():.1f} to {s.data.max():.1f}")
print(f"Signal 2 data range: {s2.data.min():.1f} to {s2.data.max():.1f}")

# %%
# **Single Signal with Custom Axis Placement**
#
# Create a figure with 3 subplots and place the HyperSpy plot in the middle one.
# This demonstrates precise control over where your scientific plots appear.

fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(12, 4))

# Plot the signal in the middle subplot (index 1)
hs.plot.plot_images(s, ax=axs[1], axes_decor="off", 
                   colorbar=True, label="Signal 1")

# Add text to the unused subplots for clarity
axs[0].text(0.5, 0.5, 'Empty\nSubplot', ha='center', va='center', fontsize=12)
axs[2].text(0.5, 0.5, 'Empty\nSubplot', ha='center', va='center', fontsize=12)

# Remove axes decorations from empty subplots
for i in [0, 2]:
    axs[i].set_xticks([])
    axs[i].set_yticks([])

plt.tight_layout()
plt.show()

# %%
# **Multiple Signals with Custom Axis Placement**
#
# Now let's place multiple HyperSpy plots in specific locations.
# This is particularly useful for comparative analysis and publication figures.

fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(12, 4))

# Plot both signals in the last two subplots (indices 1 and 2)
hs.plot.plot_images([s, s2], ax=axs[1:3], axes_decor="off", 
                   colorbar=True, label=['Signal 1', 'Signal 2'])

# Add descriptive text to the first subplot
axs[0].text(0.5, 0.5, 'Comparison:\nSignal 1 vs\nSignal 2', 
           ha='center', va='center', fontsize=12, weight='bold')
axs[0].set_xticks([])
axs[0].set_yticks([])

plt.tight_layout()
plt.show()

# %%
# **Working with Signal1D Data**
#
# The same axis specification principles apply to 1D signals (spectra).
# This is essential for spectral analysis and multi-spectrum comparisons.

# Create 1D signals for spectral analysis
s1d = hs.signals.Signal1D(np.arange(100))
s1d_neg = -s1d

print(f"Spectrum 1 shape: {s1d.data.shape}")
print(f"Spectrum 1 data range: {s1d.data.min():.1f} to {s1d.data.max():.1f}")

# %%
# **Complex Multi-Panel Layout for Spectra**
#
# Create a more complex figure layout with multiple rows and columns.
# This demonstrates advanced figure composition for scientific presentations.

fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(14, 8))

# Place the spectra in the bottom row, last two columns
hs.plot.plot_spectra([s1d, s1d_neg], ax=axs[1, 1:3], style="mosaic")

# Add informative content to other subplots
axs[0, 0].text(0.5, 0.5, 'Title:\nSpectral Analysis', 
               ha='center', va='center', fontsize=14, weight='bold')
axs[0, 1].text(0.5, 0.5, 'Methods:\nComparative\nSpectroscopy', 
               ha='center', va='center', fontsize=12)
axs[0, 2].text(0.5, 0.5, 'Results:\nSee spectra\nbelow', 
               ha='center', va='center', fontsize=12)
axs[1, 0].text(0.5, 0.5, 'Parameters:\nRange: 0-100\nPoints: 100', 
               ha='center', va='center', fontsize=10)

# Clean up unused subplots
for i in range(2):
    for j in range(3):
        if not (i == 1 and j > 0):  # Skip the plots with spectra
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])

plt.tight_layout()
plt.show()

# %%
# **Key Takeaways**
#
# 1. **Figure Control**: Use matplotlib's `subplots()` to create custom layouts
# 2. **Axis Targeting**: Use the `ax` parameter to specify exact plot locations
# 3. **Multiple Plots**: Pass lists of signals and axes for batch plotting
# 4. **Publication Ready**: Combine with `tight_layout()` for professional figures
# 5. **Flexibility**: Works with both Signal1D and Signal2D data types
#
# This approach gives you complete control over figure composition,
# essential for creating publication-quality scientific visualizations.
