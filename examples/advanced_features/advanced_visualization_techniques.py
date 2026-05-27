"""
Advanced Visualization Techniques
=================================

This example demonstrates key visualization techniques in HyperSpy for 
effective data analysis and presentation. Following AI Guide best practices 
for clear, educational examples.

Key concepts covered:
- Multi-signal plotting
- Image array visualization
- HyperSpy-native plot methods
- Publication-ready formatting
"""

import hyperspy.api as hs
import numpy as np
import matplotlib.pyplot as plt

# %%
# ## Create Sample Data for Visualization
# 
# Generate realistic test data for demonstration.

print("🎨 Creating sample data for visualization")
print("=" * 50)

# Create a simple spectrum image with two peaks
nx, ny, n_energy = 15, 20, 200
energy_axis = np.linspace(100, 300, n_energy)

# Generate test data with varying peak positions
data = np.zeros((nx, ny, n_energy))
for i in range(nx):
    for j in range(ny):
        # Peak 1: varies with position
        peak1_pos = 150 + 20 * np.sin(2 * np.pi * i / nx)
        peak1 = 1000 * np.exp(-((energy_axis - peak1_pos) / 10)**2)
        
        # Peak 2: fixed position
        peak2 = 500 * np.exp(-((energy_axis - 200) / 15)**2)
        
        # Background
        background = 50 + 20 * np.random.random()
        
        data[i, j, :] = peak1 + peak2 + background

# Create HyperSpy signal with proper axes
spectrum_image = hs.signals.Signal1D(data)
spectrum_image.axes_manager.navigation_axes[0].name = 'x'
spectrum_image.axes_manager.navigation_axes[0].units = 'μm'
spectrum_image.axes_manager.navigation_axes[0].scale = 0.1
spectrum_image.axes_manager.navigation_axes[1].name = 'y'
spectrum_image.axes_manager.navigation_axes[1].units = 'μm'
spectrum_image.axes_manager.navigation_axes[1].scale = 0.1
spectrum_image.axes_manager.signal_axes[0].name = 'energy'
spectrum_image.axes_manager.signal_axes[0].units = 'eV'
spectrum_image.axes_manager.signal_axes[0].scale = 1.0
spectrum_image.axes_manager.signal_axes[0].offset = 100

print(f"Created spectrum image: {spectrum_image}")

# %%
# ## Multi-Signal Plotting
# 
# Compare multiple spectra using HyperSpy's plot_spectra function.

print("\n📊 Multi-Signal Plotting")
print("=" * 50)

# Extract representative spectra from different regions
spectrum_left = spectrum_image.inav[0, ny//2]
spectrum_center = spectrum_image.inav[nx//2, ny//2]
spectrum_right = spectrum_image.inav[-1, ny//2]

# Set meaningful titles
spectrum_left.metadata.General.title = 'Left Region'
spectrum_center.metadata.General.title = 'Center Region'
spectrum_right.metadata.General.title = 'Right Region'

# Plot overlapping spectra for comparison
hs.plot.plot_spectra([spectrum_left, spectrum_center, spectrum_right], 
                     style='overlap', legend='auto')

print("✅ Overlapping spectra show peak position variations")

# %%
# ## Image Array Visualization
# 
# Display multiple related images in an organized layout.

print("\n🖼️ Image Array Visualization")
print("=" * 50)

# Create intensity maps from different energy ranges
peak1_map = spectrum_image.isig[140.:160.].integrate1D(axis='energy')
peak2_map = spectrum_image.isig[185.:215.].integrate1D(axis='energy')
total_intensity = spectrum_image.integrate1D(axis='energy')

# Set titles for clarity
peak1_map.metadata.General.title = 'Peak 1 Intensity'
peak2_map.metadata.General.title = 'Peak 2 Intensity'
total_intensity.metadata.General.title = 'Total Intensity'

# Display as image array
hs.plot.plot_images([peak1_map, peak2_map, total_intensity], 
                    tight_layout=True, axes_decor='ticks', colorbar='single')

print("✅ Image arrays reveal spatial intensity variations")

# %%
# ## Statistical Analysis Visualization
# 
# Create and visualize statistical maps from spectrum images.

print("\n📈 Statistical Analysis Visualization")
print("=" * 50)

# Generate statistical maps using HyperSpy methods
max_intensity = spectrum_image.max(axis='energy')
mean_intensity = spectrum_image.mean(axis='energy')

max_intensity.metadata.General.title = 'Maximum Intensity Map'
mean_intensity.metadata.General.title = 'Mean Intensity Map'

# Create ratio map to show peak variations
ratio_map = peak1_map / peak2_map
ratio_map.metadata.General.title = 'Peak Ratio Map'

# Visualize statistical maps
hs.plot.plot_images([max_intensity, mean_intensity, ratio_map], 
                    tight_layout=True, colorbar='multi')

print("✅ Statistical maps reveal data patterns and variations")

# %%
# ## Interactive Navigation
# 
# Demonstrate HyperSpy's interactive plotting capabilities.

print("\n🎮 Interactive Navigation")
print("=" * 50)

# Plot spectrum image with interactive navigation
spectrum_image.plot()

print("Interactive features available:")
print("• Click on navigator to jump to positions")
print("• Use arrow keys for step-by-step navigation")
print("• Mouse wheel for fine navigation control")
print("• Real-time spectrum updates")

# %%
# ## Region of Interest (ROI) Analysis
# 
# Use ROIs for focused analysis and visualization.

print("\n🎯 Region of Interest Analysis")
print("=" * 50)

# Create rectangular ROI for spatial selection
roi = hs.roi.RectangularROI(left=0.3, top=0.5, right=0.9, bottom=1.2)

# Extract and analyze ROI data
roi_signal = roi(spectrum_image)
roi_mean = roi_signal.mean(axis=(0, 1))
roi_mean.metadata.General.title = 'ROI Average Spectrum'

# Plot ROI results
roi_mean.plot()

print("✅ ROI enables focused analysis of specific regions")

# %%
# ## Publication-Ready Formatting
# 
# Configure plots for professional presentation.

print("\n📄 Publication-Ready Formatting")
print("=" * 50)

# Create a clean spectrum for publication
pub_spectrum = spectrum_center.deepcopy()
pub_spectrum.metadata.General.title = ''  # Remove title for clean look

# Plot with publication settings
pub_spectrum.plot()

# Customize using matplotlib for publication quality
fig = plt.gcf()
ax = plt.gca()

# Apply publication formatting
ax.set_xlabel('Energy (eV)', fontsize=12, fontweight='bold')
ax.set_ylabel('Intensity (counts)', fontsize=12, fontweight='bold')
ax.tick_params(labelsize=10)
ax.grid(True, alpha=0.3)

# Set figure properties
fig.set_size_inches(6, 4)
fig.set_dpi(150)

print("✅ Applied publication formatting:")
print("• Clean axis labels and titles")
print("• Appropriate font sizes")
print("• Grid for readability")
print("• Optimized figure size")

# %%
# ## HyperSpy-Native Plotting Methods
# 
# Leverage HyperSpy's built-in plotting capabilities.

print("\n🔧 HyperSpy-Native Plotting")
print("=" * 50)

# Use HyperSpy's built-in plot methods
print("Using HyperSpy's native plotting methods:")

# Direct signal plotting
print("• signal.plot() for interactive navigation")
print("• signal.plot_line() for line profiles")
print("• signal.plot_images() for image arrays")

# Statistical plotting
print("• signal.mean().plot() for average spectra")
print("• signal.max().plot() for intensity maps")
print("• signal.std().plot() for variability maps")

# Advanced plotting
print("• hs.plot.plot_spectra() for multi-signal comparison")
print("• hs.plot.plot_images() for organized layouts")

# %%
# ## Visualization Best Practices Summary
# 
# Key guidelines for effective HyperSpy visualization.

print("\n✅ Visualization Best Practices")
print("=" * 50)

print("1. 🎯 Choose Appropriate Plot Types:")
print("   • Use overlap style for direct comparison")
print("   • Use image arrays for spatial analysis")
print("   • Use statistical maps for pattern recognition")

print("\n2. 📊 Optimize for Audience:")
print("   • Clean formatting for publications")
print("   • Interactive plots for exploration")
print("   • Clear labels and units")

print("\n3. 🔧 Use HyperSpy Methods:")
print("   • Leverage built-in plotting functions")
print("   • Use axes_manager for proper scaling")
print("   • Apply metadata for context")

print("\n4. 🎨 Enhance Understanding:")
print("   • Add meaningful titles and labels")
print("   • Use appropriate color schemes")
print("   • Include scale bars and colorbars")

print("\nVisualization techniques demonstrated:")
print("• Multi-signal comparison plots")
print("• Image array organization")
print("• Statistical analysis visualization")
print("• Interactive navigation")
print("• ROI-based analysis")
print("• Publication-ready formatting")
print("• HyperSpy-native plotting methods")

print(f"\nExample completed successfully!")
print(f"Data shape: {spectrum_image.data.shape}")
print(f"Memory usage: {spectrum_image.data.nbytes / 1024**2:.1f} MB")