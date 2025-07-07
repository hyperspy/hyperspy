"""
Advanced Visualization Techniques
=================================

This example demonstrates sophisticated visualization techniques in HyperSpy
beyond basic plotting, including multi-signal comparisons, custom plot layouts,
interactive features, and publication-ready visualizations.
"""

import hyperspy.api as hs
import numpy as np

# %%
# Create diverse signals for visualization demonstrations
# -------------------------------------------------------

# Spectrum image with varying peak positions
nx, ny, n_energy = 20, 30, 1000
data_spectrum = np.zeros((nx, ny, n_energy))
energy_axis = np.linspace(100, 600, n_energy)

for i in range(nx):
    for j in range(ny):
        # Create varying peak positions and intensities
        peak1_pos = 200 + 50 * np.sin(2 * np.pi * i / nx)
        peak2_pos = 400 + 30 * np.cos(2 * np.pi * j / ny)
        
        peak1 = 1000 * np.exp(-((energy_axis - peak1_pos) / 25)**2)
        peak2 = 800 * np.exp(-((energy_axis - peak2_pos) / 20)**2)
        background = 100 + 50 * np.random.random()
        
        data_spectrum[i, j, :] = peak1 + peak2 + background + 10 * np.random.random(n_energy)

spectrum_image = hs.signals.Signal1D(data_spectrum)
spectrum_image.axes_manager[0].name = 'x'
spectrum_image.axes_manager[0].units = 'μm'
spectrum_image.axes_manager[0].scale = 0.1

spectrum_image.axes_manager[1].name = 'y'
spectrum_image.axes_manager[1].units = 'μm'
spectrum_image.axes_manager[1].scale = 0.1

spectrum_image.axes_manager[2].name = 'energy'
spectrum_image.axes_manager[2].units = 'eV'
spectrum_image.axes_manager[2].scale = 0.5
spectrum_image.axes_manager[2].offset = 100

spectrum_image.metadata.General.title = 'Spectrum Image - Peak Variations'

# Create multiple related signals for comparison
peak1_map = spectrum_image.isig[180.:220.].integrate1D(axis='energy')
peak2_map = spectrum_image.isig[380.:420.].integrate1D(axis='energy')
total_intensity = spectrum_image.integrate1D(axis='energy')

peak1_map.metadata.General.title = 'Peak 1 Intensity Map'
peak2_map.metadata.General.title = 'Peak 2 Intensity Map'
total_intensity.metadata.General.title = 'Total Intensity Map'

print("Created demonstration signals for advanced visualization")

# %%
# ## Multi-Signal Plotting with Different Styles
# 
# HyperSpy provides powerful multi-signal plotting capabilities

# Extract representative spectra from different regions
spectrum_corner = spectrum_image.inav[0, 0]
spectrum_center = spectrum_image.inav[nx//2, ny//2]
spectrum_edge = spectrum_image.inav[-1, ny//2]

spectrum_corner.metadata.General.title = 'Corner Spectrum'
spectrum_center.metadata.General.title = 'Center Spectrum'
spectrum_edge.metadata.General.title = 'Edge Spectrum'

# %%
# ### Overlapping Spectra Comparison

# Plot multiple spectra with overlap style
hs.plot.plot_spectra([spectrum_corner, spectrum_center, spectrum_edge], 
                     style='overlap', legend='auto')

print("Overlap style: All spectra on same axes for direct comparison")

# %%
# ### Cascade Style for Clear Separation

# Plot with cascade style for clear visual separation
hs.plot.plot_spectra([spectrum_corner, spectrum_center, spectrum_edge], 
                     style='cascade', padding=200)

print("Cascade style: Vertically offset spectra for clear separation")

# %%
# ### Mosaic Style for Subplot Layout

# Plot in mosaic style for individual subplot analysis
hs.plot.plot_spectra([spectrum_corner, spectrum_center, spectrum_edge], 
                     style='mosaic')

print("Mosaic style: Individual subplots for detailed analysis")

# %%
# ## Advanced Image Array Visualization
# 
# Visualize multiple related images in organized layouts

# Create list of intensity maps
intensity_maps = [peak1_map, peak2_map, total_intensity]

# Plot as image array with automatic layout
hs.plot.plot_images(intensity_maps, tight_layout=True, 
                    axes_decor='ticks', label='auto',
                    colorbar='single')

print("Image arrays: Multiple related images in organized grid")

# %%
# ## Statistical Map Visualizations
# 
# Create and visualize statistical maps from spectrum images

# Generate various statistical maps
max_intensity = spectrum_image.max(axis='energy')
mean_energy = spectrum_image.mean(axis='energy')
std_spectrum = spectrum_image.std(axis='energy')

max_intensity.metadata.General.title = 'Maximum Intensity'
mean_energy.metadata.General.title = 'Mean Energy'
std_spectrum.metadata.General.title = 'Spectral Standard Deviation'

# Visualize statistical maps together
statistical_maps = [max_intensity, mean_energy, std_spectrum]
hs.plot.plot_images(statistical_maps, tight_layout=True,
                    axes_decor='ticks', colorbar='multi')

print("Statistical maps: Different analytical perspectives of the same dataset")

# %%
# ## Interactive Navigation and ROI Selection
# 
# Demonstrate interactive features for data exploration

# Plot spectrum image with interactive navigation
spectrum_image.plot()

print("Interactive navigation features:")
print("- Click on navigator to jump to positions")
print("- Use arrow keys for step-by-step navigation")
print("- Ctrl+arrow keys to jump to edges")
print("- Mouse wheel for fine navigation control")

# %%
# ## Region of Interest (ROI) Visualization
# 
# Use ROIs for interactive analysis and visualization

# Create rectangular ROI for spatial selection
roi_rect = hs.roi.RectangularROI(left=0.5, top=0.8, right=1.5, bottom=1.8)

# Extract signal from ROI
roi_signal = roi_rect(spectrum_image)
roi_mean = roi_signal.mean(axis=(0, 1))  # Average over spatial dimensions
roi_mean.metadata.General.title = 'ROI Average Spectrum'

# Plot ROI analysis
roi_mean.plot()

print("ROI capabilities:")
print("- Interactive selection and modification")
print("- Real-time extraction and analysis")
print("- Multiple ROI types: rectangular, circular, line, point")

# %%
# ## Custom Visualization with Markers
# 
# Add analytical markers and annotations to plots

# Find peak positions in each spectrum using HyperSpy
peak_positions = spectrum_image.estimate_peak_width()

# Create marker for peak positions (simplified approach)
mean_spectrum = spectrum_image.mean(axis=(0, 1))
peak_indices = np.where(mean_spectrum.data > mean_spectrum.data.mean() + 2*mean_spectrum.data.std())[0]
peak_energies = mean_spectrum.axes_manager[0].axis[peak_indices]

# Add vertical line markers at peak positions
vertical_markers = []
for energy in peak_energies[:5]:  # Limit to first 5 peaks
    marker = hs.plot.markers.VerticalLine(energy, color='red', linewidth=2)
    vertical_markers.append(marker)

# Plot with markers
mean_spectrum.plot()
for marker in vertical_markers:
    mean_spectrum.add_marker(marker, plot_marker=True)

print(f"Added {len(vertical_markers)} peak markers to mean spectrum")

# %%
# ## Publication-Ready Visualization Settings
# 
# Configure plots for publication quality

# Create a clean spectrum plot for publication
pub_spectrum = spectrum_center.deepcopy()
pub_spectrum.metadata.General.title = ''  # Remove title for clean look

# Plot with publication settings
pub_spectrum.plot()

# Access the matplotlib axes for fine control
import matplotlib.pyplot as plt

# Get current figure and axes
fig = plt.gcf()
ax = plt.gca()

# Customize for publication
ax.set_xlabel('Energy (eV)', fontsize=12, fontweight='bold')
ax.set_ylabel('Intensity (counts)', fontsize=12, fontweight='bold')
ax.tick_params(labelsize=10)
ax.grid(True, alpha=0.3)

# Adjust figure size and DPI for publication
fig.set_size_inches(6, 4)
fig.set_dpi(300)

print("Publication settings applied:")
print("- Clean axis labels and titles")
print("- Appropriate font sizes")
print("- Grid for readability")
print("- High DPI for print quality")

# %%
# ## Advanced Color Mapping and Scaling
# 
# Demonstrate sophisticated color control for better data visualization

# Create intensity map with custom colormap
intensity_map = spectrum_image.max(axis='energy')

# Plot with different color schemes
intensity_map.plot(colorbar=True, scalebar=False)

print("Color mapping options:")
print("- cmap parameter for different color schemes")
print("- vmin/vmax for manual scaling")
print("- norm parameter for logarithmic scaling")
print("- colorbar control for professional appearance")

# %%
# ## Multi-Dimensional Visualization Summary
# 
# Demonstrate combined visualization approach

# Create comprehensive analysis visualization
print("\nComprehensive Visualization Summary:")
print("====================================")

# 1. Multi-signal comparison
print("1. Multi-signal comparison completed")

# 2. Image arrays
print("2. Image array visualization completed")

# 3. Statistical analysis
print("3. Statistical map analysis completed")

# 4. Interactive features
print("4. Interactive navigation demonstrated")

# 5. ROI analysis
print("5. ROI extraction and analysis completed")

# 6. Markers and annotations
print("6. Peak markers and annotations added")

# 7. Publication quality
print("7. Publication-ready formatting applied")

print("\nAdvanced visualization techniques covered:")
print("- Multi-signal plotting (overlap, cascade, mosaic)")
print("- Image array organization")
print("- Statistical map generation")
print("- Interactive navigation and ROI selection")
print("- Custom markers and annotations")
print("- Publication-ready formatting")
print("- Color mapping and scaling control")
