"""
Composing Complex Scientific Figures
====================================

This example demonstrates how to create sophisticated multi-panel figures
by combining different HyperSpy plotting functions. This is essential for
creating publication-quality figures that show both images and spectra
in a single comprehensive visualization.
"""

# %%
# **Import required libraries**
#
# We'll combine HyperSpy's specialized plotting functions with matplotlib's
# figure composition capabilities to create complex scientific visualizations.

import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np

# %%
# **Creating Complementary Datasets**
#
# For demonstration, we'll create related 2D and 1D signals that represent
# different aspects of the same scientific measurement. This simulates
# real-world scenarios where you have both spatial (2D) and spectral (1D) data.

# Create structured 2D data (e.g., representing a spatial map)
s2D_0 = hs.signals.Signal2D(np.arange(100).reshape(10, 10))
s2D_1 = -s2D_0  # Complementary signal (opposite phase/polarity)

# Create corresponding 1D spectral data
s1D_0 = hs.signals.Signal1D(np.arange(100))
s1D_1 = -s1D_0  # Complementary spectrum

# Print dataset information for clarity
print("=== Dataset Information ===")
print(f"2D Signal 1: {s2D_0.data.shape}, range: {s2D_0.data.min():.1f} to {s2D_0.data.max():.1f}")
print(f"2D Signal 2: {s2D_1.data.shape}, range: {s2D_1.data.min():.1f} to {s2D_1.data.max():.1f}")
print(f"1D Signal 1: {s1D_0.data.shape}, range: {s1D_0.data.min():.1f} to {s1D_0.data.max():.1f}")
print(f"1D Signal 2: {s1D_1.data.shape}, range: {s1D_1.data.min():.1f} to {s1D_1.data.max():.1f}")

# %%
# **Understanding Figure Composition Strategy**
#
# For comprehensive scientific analysis, we often need to show:
# 1. **Spatial data** (2D images) - showing where measurements were taken
# 2. **Spectral data** (1D spectra) - showing what was measured
# 3. **Comparative analysis** - showing relationships between datasets
#
# This example shows how to arrange these elements in a single figure.

# %%
# **Creating the Multi-Panel Figure**
#
# We'll create a 2x2 grid where:
# - Left column: 2D spatial data (images)
# - Right column: 1D spectral data (spectra)
# - Top row: Primary measurements
# - Bottom row: Complementary measurements

fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 10))

# Plot images in the left column
hs.plot.plot_images([s2D_0, s2D_1], ax=axs[:, 0], axes_decor="off", 
                   colorbar=True, label=['Primary Image', 'Complementary Image'])

# Plot spectra in the right column
hs.plot.plot_spectra([s1D_0, s1D_1], ax=axs[:, 1], style="mosaic")

# Add overall title and improve layout
fig.suptitle('Multi-Modal Scientific Data Analysis', fontsize=16, fontweight='bold')

# Add column labels for clarity
axs[0, 0].set_title('Spatial Data (2D Images)', fontsize=12, pad=20)
axs[0, 1].set_title('Spectral Data (1D Spectra)', fontsize=12, pad=20)

# Adjust layout for better presentation
plt.tight_layout()
plt.show()

# %%
# **Advanced Figure Composition with Annotations**
#
# Let's create a more sophisticated version with annotations and explanatory text.
# This demonstrates how to add contextual information to your scientific figures.

fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(14, 10))

# Plot the data with enhanced styling
hs.plot.plot_images([s2D_0, s2D_1], ax=axs[:, 0], axes_decor="off", 
                   colorbar=True, label=['Dataset A', 'Dataset B'])
hs.plot.plot_spectra([s1D_0, s1D_1], ax=axs[:, 1], style="mosaic")

# Add detailed annotations
fig.suptitle('Comprehensive Multi-Modal Analysis', fontsize=16, fontweight='bold')

# Add informative subplot titles
axs[0, 0].set_title('Spatial Map A\n(Primary measurement)', fontsize=11, pad=15)
axs[1, 0].set_title('Spatial Map B\n(Complementary measurement)', fontsize=11, pad=15)
axs[0, 1].set_title('Spectrum A\n(Primary response)', fontsize=11, pad=15)
axs[1, 1].set_title('Spectrum B\n(Complementary response)', fontsize=11, pad=15)

# Add a text box with analysis notes
textstr = 'Analysis Notes:\n• Datasets A & B show complementary patterns\n• Spatial correlation visible in both maps\n• Spectral features mirror spatial distribution'
props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
fig.text(0.02, 0.02, textstr, fontsize=10, verticalalignment='bottom', bbox=props)

plt.tight_layout()
plt.show()

# %%
# **Key Design Principles for Scientific Figures**
#
# When composing complex figures, consider these best practices:
#
# 1. **Logical Organization**: Group related data types (spatial vs spectral)
# 2. **Clear Labeling**: Use descriptive titles and labels
# 3. **Consistent Scaling**: Ensure colorbars and axes are appropriate
# 4. **Contextual Information**: Add annotations explaining the analysis
# 5. **Professional Layout**: Use `tight_layout()` for clean presentation
#
# This approach creates figures that are both scientifically rigorous
# and visually appealing for publications and presentations.

# %%
# **Real-World Applications**
#
# This multi-panel approach is particularly valuable for:
# - **Electron microscopy**: Showing both STEM images and EELS spectra
# - **X-ray analysis**: Combining elemental maps with spectral data
# - **Optical spectroscopy**: Relating spatial and spectral measurements
# - **Materials characterization**: Correlating structure and properties
#
# The flexibility of HyperSpy's plotting functions makes it easy to adapt
# this template for your specific scientific application.
