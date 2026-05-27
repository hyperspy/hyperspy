"""
Comprehensive Markers Guide
===========================

This example demonstrates how to create and use various marker types in HyperSpy
for annotating plots and highlighting features in signals. It covers the most
commonly used marker types and their applications.
"""

import hyperspy.api as hs
import numpy as np

# %%
# Create test signals for demonstration
# --------------------------------------
#
# We'll create both 1D and 2D signals to demonstrate different marker applications.

# Create a 2D signal with multiple navigation dimensions
rng = np.random.default_rng(0)
data_2d = np.ones((10, 5, 100, 100))
# Add some interesting features
x, y = np.meshgrid(np.arange(100), np.arange(100))
for i in range(10):
    for j in range(5):
        # Add circular features at different positions
        center_x, center_y = 20 + i * 7, 30 + j * 10
        circle = np.exp(-((x - center_x)**2 + (y - center_y)**2) / 200)
        data_2d[i, j] += circle

s2d = hs.signals.Signal2D(data_2d)
s2d.axes_manager.signal_axes[0].name = 'X'
s2d.axes_manager.signal_axes[0].units = 'nm'
s2d.axes_manager.signal_axes[0].scale = 0.1
s2d.axes_manager.signal_axes[1].name = 'Y'
s2d.axes_manager.signal_axes[1].units = 'nm'
s2d.axes_manager.signal_axes[1].scale = 0.1

# Create a 1D signal
x_axis = np.linspace(0, 10, 1000)
spectrum = np.exp(-x_axis/2) + 0.5 * np.sin(10 * x_axis) + 0.1 * np.random.randn(1000)
s1d = hs.signals.Signal1D(spectrum)
s1d.axes_manager.signal_axes[0].name = 'Energy'
s1d.axes_manager.signal_axes[0].units = 'eV'
s1d.axes_manager.signal_axes[0].scale = 0.01

print(f"Created 2D signal: {s2d}")
print(f"Created 1D signal: {s1d}")

# %%
# **Circle Markers: Highlighting circular features**
#
# Circle markers are perfect for highlighting peaks, particles, or circular features.

print("\n--- Circle Markers ---")

# Static circles at fixed positions
circle_positions = np.array([[20, 30], [50, 60], [80, 40]])
circle_sizes = [15, 20, 12]

circle_markers = hs.plot.markers.Circles(
    offsets=circle_positions,
    sizes=circle_sizes,
    facecolors='none',  # Use facecolors='none' instead of fill=False
    edgecolors=['red', 'blue', 'green'],
    linewidth=2
)

# Plot with circle markers
s2d.plot()
s2d.add_marker(circle_markers)

print("Circle markers added to highlight features")

# %%
# **Rectangle Markers: Highlighting rectangular regions**
#
# Rectangle markers are useful for marking analysis regions or defects.

print("\n--- Rectangle Markers ---")

# Define rectangle positions and sizes
rect_positions = np.array([[10, 10], [60, 70], [30, 80]])
rect_widths = [15, 20, 12]
rect_heights = [10, 15, 18]

rectangle_markers = hs.plot.markers.Rectangles(
    offsets=rect_positions,
    widths=rect_widths,
    heights=rect_heights,
    angles=[0, 15, -10],  # Rotation angles
    color=['orange', 'purple', 'cyan'],
    linewidth=2,
    facecolors='none',
    edgecolors=['orange', 'purple', 'cyan']
)

# Create a new plot to avoid marker overlap
s2d_rect = s2d.deepcopy()
s2d_rect.plot()
s2d_rect.add_marker(rectangle_markers)

print("Rectangle markers added to mark analysis regions")

# %%
# **Arrow Markers: Indicating directions or pointing to features**
#
# Arrow markers are excellent for pointing to specific features or showing directions.

print("\n--- Arrow Markers ---")

# Arrow starting points and directions
arrow_starts = np.array([[10, 90], [90, 10], [50, 50]])
arrow_directions = np.array([[20, -15], [-15, 20], [10, 10]])

arrow_markers = hs.plot.markers.Arrows(
    offsets=arrow_starts,
    U=arrow_directions[:, 0],  # X-direction components
    V=arrow_directions[:, 1],  # Y-direction components
    color=['red', 'green', 'blue'],
    linewidth=3,
    arrowstyle='->'
)

# Create a new plot for arrows
s2d_arrow = s2d.deepcopy()
s2d_arrow.plot()
s2d_arrow.add_marker(arrow_markers)

print("Arrow markers added to point to features")

# %%
# **Line Markers: Marking boundaries or measurement lines**
#
# Line markers are useful for marking boundaries, measurement lines, or linear features.

print("\n--- Line Markers ---")

# Define line start and end points
line_starts = np.array([[0, 50], [50, 0], [25, 25]])
line_ends = np.array([[100, 50], [50, 100], [75, 75]])

# Create segments for line markers
line_segments = np.array([[[0, 50], [100, 50]], 
                         [[50, 0], [50, 100]], 
                         [[25, 25], [75, 75]]])

line_markers = hs.plot.markers.Lines(
    segments=line_segments,
    color=['black', 'gray', 'brown'],
    linewidth=[2, 3, 1],
    linestyle=['-', '--', ':']
)

# Create a new plot for lines
s2d_line = s2d.deepcopy()
s2d_line.plot()
s2d_line.add_marker(line_markers)

print("Line markers added for measurement lines")

# %%
# **Text Markers: Adding labels and annotations**
#
# Text markers are essential for labeling features and adding annotations.

print("\n--- Text Markers ---")

# Text positions and labels
text_positions = np.array([[20, 30], [50, 60], [80, 40]])
text_labels = ['Peak A', 'Peak B', 'Defect']

text_markers = hs.plot.markers.Texts(
    offsets=text_positions,
    texts=text_labels,
    sizes=[12, 14, 10],
    color=['red', 'blue', 'green']
)

# Create a new plot for text
s2d_text = s2d.deepcopy()
s2d_text.plot()
s2d_text.add_marker(text_markers)

print("Text markers added for feature labeling")

# %%
# **Vertical Line Markers for 1D Signals**
#
# Vertical lines are particularly useful for 1D signals to mark specific energies or positions.

print("\n--- Vertical Line Markers for 1D Signals ---")

# Mark specific energies
energy_positions = [2.0, 4.5, 7.0]  # eV
colors = ['red', 'green', 'blue']
labels = ['Peak 1', 'Peak 2', 'Peak 3']

# Create vertical line markers
vline_markers = hs.plot.markers.VerticalLines(
    offsets=energy_positions,
    color=colors,
    linewidth=2,
    linestyle='--'
)

# Plot 1D signal with vertical lines
s1d.plot()
s1d.add_marker(vline_markers)

# Add text markers to label the lines
text_positions_1d = np.array([[pos, 0.8] for pos in energy_positions])
text_markers_1d = hs.plot.markers.Texts(
    offsets=text_positions_1d,
    texts=labels,
    sizes=10,
    color=colors
)
s1d.add_marker(text_markers_1d)

print("Vertical line markers added to 1D signal")

# %%
# **Navigation-dependent Markers: Dynamic markers that change with navigation**
#
# Navigation-dependent markers change as you navigate through the signal dimensions.

print("\n--- Navigation-dependent Markers ---")

# Create markers that change with navigation position
# For each navigation position, we'll mark different features

# Get the number of navigation positions
nav_shape = s2d.axes_manager.navigation_shape
total_nav_positions = np.prod(nav_shape)

# Create position arrays for each navigation position
nav_positions = []
for i in range(nav_shape[0]):
    for j in range(nav_shape[1]):
        # Create marker position that depends on navigation coordinates
        marker_x = 20 + i * 7
        marker_y = 30 + j * 10
        nav_positions.append([marker_x, marker_y])

nav_positions = np.array(nav_positions)

# Create navigation-dependent circle markers
nav_circle_markers = hs.plot.markers.Circles(
    offsets=nav_positions,
    sizes=10,
    color='yellow',
    linewidth=2,
    facecolors='none',
    edgecolors='yellow'
)

# Create a new plot for navigation-dependent markers
s2d_nav = s2d.deepcopy()
s2d_nav.plot()
s2d_nav.add_marker(nav_circle_markers)

print("Navigation-dependent markers added")
print("Navigate through the signal to see markers change position")

# %%
# **Combining Multiple Marker Types**
#
# Often you need to combine different marker types for comprehensive annotation.

print("\n--- Combining Multiple Marker Types ---")

# Create a comprehensive annotation example
combined_circles = hs.plot.markers.Circles(
    offsets=np.array([[25, 25], [75, 75]]),
    sizes=[15, 20],
    color=['red', 'blue'],
    linewidth=2,
    facecolors='none',
    edgecolors=['red', 'blue']
)

combined_arrows = hs.plot.markers.Arrows(
    offsets=np.array([[10, 10], [90, 90]]),
    U=np.array([10, -10]),
    V=np.array([10, -10]),
    color=['red', 'blue'],
    linewidth=2
)

combined_texts = hs.plot.markers.Texts(
    offsets=np.array([[25, 30], [75, 80]]),
    texts=['Feature A', 'Feature B'],
    sizes=12,
    color=['red', 'blue']
)

# Create a comprehensive plot
s2d_combined = s2d.deepcopy()
s2d_combined.plot()
s2d_combined.add_marker(combined_circles)
s2d_combined.add_marker(combined_arrows)
s2d_combined.add_marker(combined_texts)

print("Combined multiple marker types for comprehensive annotation")

# %%
# **Advanced Marker Features**
#
# HyperSpy markers support advanced features like scaling, units, and transformations.

print("\n--- Advanced Marker Features ---")

# Markers with physical units
# When your signal has calibrated axes, markers can use physical units
print(f"Signal axes units: {s2d.axes_manager.signal_axes[0].units}")

# Create markers with size scaling based on signal properties
adaptive_sizes = [10, 15, 20]  # Sizes that could be based on feature analysis
adaptive_positions = np.array([[30, 30], [50, 50], [70, 70]])

adaptive_markers = hs.plot.markers.Circles(
    offsets=adaptive_positions,
    sizes=adaptive_sizes,
    color=['red', 'green', 'blue'],
    linewidth=2,
    facecolors='none',
    edgecolors=['red', 'green', 'blue']
)

s2d_adaptive = s2d.deepcopy()
s2d_adaptive.plot()
s2d_adaptive.add_marker(adaptive_markers)

print("Adaptive markers created based on signal properties")

# %%
# **Marker Management: Adding, removing, and modifying markers**
#
# Learn how to manage markers dynamically.

print("\n--- Marker Management ---")

# Create a signal for marker management demonstration
s2d_mgmt = s2d.deepcopy()
s2d_mgmt.plot()

# Add initial markers
initial_markers = hs.plot.markers.Circles(
    offsets=np.array([[20, 20], [40, 40]]),
    sizes=10,
    color='red'
)

s2d_mgmt.add_marker(initial_markers)
print("Initial markers added")

# You can add more markers
additional_markers = hs.plot.markers.Rectangles(
    offsets=np.array([[60, 60], [80, 80]]),
    widths=15,
    heights=10,
    color='blue'
)

s2d_mgmt.add_marker(additional_markers)
print("Additional markers added")

# Remove all markers (if needed)
# s2d_mgmt.plot.remove_markers()  # Uncomment to remove all markers

# %%
# **Best Practices for Using Markers**
#
# Guidelines for effective marker usage.

print("\n--- Best Practices for Markers ---")

print("1. Color coding:")
print("   - Use consistent colors for similar features")
print("   - Choose colors that contrast with your signal")
print("   - Consider colorblind-friendly palettes")

print("\n2. Size and scaling:")
print("   - Make markers large enough to be visible")
print("   - Scale marker sizes with feature importance")
print("   - Consider signal zoom level")

print("\n3. Marker types:")
print("   - Circles: for point features, peaks, particles")
print("   - Rectangles: for regions of interest, analysis areas")
print("   - Arrows: for pointing to features, showing directions")
print("   - Lines: for boundaries, measurement lines")
print("   - Text: for labeling, annotations")

print("\n4. Navigation-dependent markers:")
print("   - Use when marker positions change with navigation")
print("   - Helpful for tracking features across dimensions")
print("   - Consider computational efficiency")

print("\n5. Combining markers:")
print("   - Use multiple marker types for comprehensive annotation")
print("   - Maintain visual clarity and avoid clutter")
print("   - Group related markers logically")

# %%
# **Practical Applications**
#
# Common use cases for different marker types.

print("\n--- Practical Applications ---")

print("Common applications by marker type:")

print("\n• Circle markers:")
print("  - Highlighting nanoparticles in TEM images")
print("  - Marking peak positions in spectroscopy")
print("  - Indicating regions of interest for analysis")

print("\n• Rectangle markers:")
print("  - Defining analysis regions")
print("  - Marking defects or anomalies")
print("  - Indicating cropping areas")

print("\n• Arrow markers:")
print("  - Pointing to specific features")
print("  - Showing crystal orientations")
print("  - Indicating growth directions")

print("\n• Line markers:")
print("  - Marking grain boundaries")
print("  - Indicating measurement lines")
print("  - Showing linear features")

print("\n• Text markers:")
print("  - Labeling phases or components")
print("  - Adding measurement values")
print("  - Providing feature descriptions")

print("\n• Vertical line markers (1D):")
print("  - Marking characteristic energies")
print("  - Indicating peak positions")
print("  - Showing fitting regions")

print("\nMarker demonstration complete!")
print("Explore the different plots to see various marker types in action.")
