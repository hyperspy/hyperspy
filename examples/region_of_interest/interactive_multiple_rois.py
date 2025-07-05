"""
Interactive ROI with Multiple Types
===================================

This example demonstrates how to use multiple types of Region of Interest (ROI)
widgets simultaneously on the same image. ROIs are powerful tools for selecting
specific regions of data for analysis, and different ROI types serve different
purposes.

ROI types covered:
- RectangularROI: Select rectangular regions
- Line2DROI: Select line profiles
- Point2DROI: Select specific points
- CircleROI: Select circular regions
- PolygonROI: Select arbitrary polygon regions

"""

import numpy as np
import hyperspy.api as hs

# Enable interactive plotting if running in Jupyter
# %matplotlib widget

# %%
# Create Test Image Data
# ----------------------
# 
# First, let's create a synthetic 2D image with interesting features to 
# demonstrate different ROI types.

# Create a 2D image with various features
x = np.linspace(-5, 5, 200)
y = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x, y)

# Create an image with multiple features
image_data = (
    # Central Gaussian blob
    3 * np.exp(-(X**2 + Y**2)) +
    # Off-center peak
    2 * np.exp(-((X-2)**2 + (Y-1)**2)/0.5) +
    # Linear gradient
    0.3 * X +
    # Sinusoidal pattern
    0.5 * np.sin(2*np.pi*X/3) * np.cos(2*np.pi*Y/4) +
    # Random noise
    0.1 * np.random.randn(200, 200)
)

# Create HyperSpy Signal2D
signal = hs.signals.Signal2D(image_data)
signal.axes_manager[0].name = 'Y'
signal.axes_manager[0].units = 'μm'
signal.axes_manager[0].scale = 0.05
signal.axes_manager[0].offset = -5
signal.axes_manager[1].name = 'X' 
signal.axes_manager[1].units = 'μm'
signal.axes_manager[1].scale = 0.05
signal.axes_manager[1].offset = -5
signal.metadata.General.title = 'Test Image for ROI Demonstration'

print("Created test image with shape:", signal.data.shape)
print("X-axis range: {:.1f} to {:.1f} μm".format(
    signal.axes_manager[1].offset,
    signal.axes_manager[1].offset + signal.axes_manager[1].scale * signal.axes_manager[1].size
))
print("Y-axis range: {:.1f} to {:.1f} μm".format(
    signal.axes_manager[0].offset,
    signal.axes_manager[0].offset + signal.axes_manager[0].scale * signal.axes_manager[0].size
))

# %%
# Display the Original Image
# --------------------------

signal.plot()

# %%
# Create Multiple ROI Types
# -------------------------
# 
# Now let's create different types of ROIs to select various regions of interest.

# 1. Rectangular ROI - for selecting rectangular regions
rectangular_roi = hs.roi.RectangularROI(
    left=-1, right=1,    # X coordinates
    top=-1, bottom=1     # Y coordinates
)

# 2. Line ROI - for extracting line profiles
line_roi = hs.roi.Line2DROI(
    x1=-3, y1=-3,        # Start point
    x2=3, y2=3,          # End point
    linewidth=0.2        # Width of the line
)

# 3. Point ROI - for selecting specific points
point_roi = hs.roi.Point2DROI(x=2, y=1)  # Position of the off-center peak

# 4. Circle ROI - for selecting circular regions
circle_roi = hs.roi.CircleROI(
    cx=0, cy=0,          # Center coordinates
    r=1.5                # Radius
)

# 5. Polygon ROI - for arbitrary shaped regions
# Create a triangular selection around an interesting region
# Note: PolygonROI vertices are specified as x,y pairs
polygon_roi = hs.roi.PolygonROI()

print("Created 5 different ROI types:")
print("1. Rectangular ROI: central square region")
print("2. Line ROI: diagonal line profile")
print("3. Point ROI: single point at peak location")
print("4. Circle ROI: circular region around center")
print("5. Polygon ROI: triangular region")

# %%
# Apply ROIs Interactively
# ------------------------
# 
# Now let's add these ROIs as interactive widgets to the image.

# Plot the signal first (required for interactive ROIs)
signal.plot()

# Add interactive ROIs with different colors
print("Adding interactive ROI widgets...")
roi_2d_rectangular = rectangular_roi.interactive(signal, color="red")
roi_1d_line = line_roi.interactive(signal, color="yellow") 
roi_0d_point = point_roi.interactive(signal, color="blue")
roi_2d_circle = circle_roi.interactive(signal, color="green")
roi_2d_polygon = polygon_roi.interactive(signal, color="purple")

print("""
Interactive ROI widgets added to the image:
- Red rectangle: Use for selecting rectangular regions
- Yellow line: Use for line profile extraction
- Blue point: Use for point measurements
- Green circle: Use for circular region selection
- Purple triangle: Use for arbitrary polygon selection

You can now:
1. Drag ROIs to move them
2. Resize ROIs by dragging corners/edges
3. Use the widgets to interactively explore the data
""")

# %%
# Extract Data from ROIs
# ----------------------
# 
# Let's extract and analyze data from each ROI type.

# Extract data using each ROI
print("\nExtracting data from ROIs...")

# 1. Rectangular ROI - extracts a 2D region
rect_data = rectangular_roi(signal)
print(f"Rectangular ROI data shape: {rect_data.data.shape}")
print(f"Mean intensity in rectangle: {rect_data.data.mean():.3f}")

# 2. Line ROI - extracts a 1D line profile
line_data = line_roi(signal)
print(f"Line ROI data shape: {line_data.data.shape}")
print(f"Max intensity along line: {line_data.data.max():.3f}")

# 3. Point ROI - extracts a single value
point_data = point_roi(signal)
print(f"Point ROI data: {point_data.data.item():.3f}")

# 4. Circle ROI - extracts a 2D circular region
circle_data = circle_roi(signal)
print(f"Circle ROI data shape: {circle_data.data.shape}")
print(f"Mean intensity in circle: {circle_data.data.mean():.3f}")

# 5. Polygon ROI - extracts data within polygon
polygon_data = polygon_roi(signal)
print(f"Polygon ROI data shape: {polygon_data.data.shape}")
print(f"Mean intensity in polygon: {polygon_data.data.mean():.3f}")

# %%
# Visualize Extracted Data using HyperSpy's native plotting
# ---------------------------------------------------------

print("\n--- Displaying extracted ROI data ---")

# Display the original image
signal.plot()

# Display extracted ROI data using HyperSpy's native plotting
rect_data.metadata.General.title = 'Rectangular ROI Data'
rect_data.plot()

# Line profile data
line_data.metadata.General.title = 'Line Profile'
line_data.plot()

# Circle ROI data
circle_data.metadata.General.title = 'Circle ROI Data'
circle_data.plot()

# Polygon ROI data
polygon_data.metadata.General.title = 'Polygon ROI Data'
polygon_data.plot()

# Use plot_images to compare 2D ROI extractions
hs.plot.plot_images([signal, rect_data, circle_data, polygon_data],
                   label=['Original Image', 'Rectangular ROI', 'Circle ROI', 'Polygon ROI'],
                   cmap='viridis',
                   colorbar=True)

# Display statistics comparison
print("\n--- ROI Statistics Comparison ---")
roi_stats = {
    'Rectangular': rect_data.data.mean(),
    'Circle': circle_data.data.mean(),
    'Polygon': polygon_data.data.mean(),
    'Point': point_data.data.item(),
    'Line Max': line_data.data.max()
}

for roi_type, value in roi_stats.items():
    print(f"{roi_type:12}: {value:.3f}")

print("ROI extraction and visualization completed using HyperSpy's native plotting.")

# %%
# Advanced ROI Usage
# ------------------
# 
# Let's demonstrate some advanced ROI features.

print("\n" + "="*50)
print("ADVANCED ROI FEATURES")
print("="*50)

# 1. ROI on different signals
print("\n1. Using ROI with different navigation signal:")
# Create a 3D signal (stack of images)
image_stack = np.random.rand(10, 100, 100)
stack_signal = hs.signals.Signal2D(image_stack)
stack_signal.axes_manager[0].name = 'Time'
stack_signal.axes_manager[0].units = 's'

# Use rectangular ROI from previous signal on new signal
roi_on_stack = rectangular_roi(stack_signal)
print(f"ROI applied to stack shape: {roi_on_stack.data.shape}")

# 2. ROI parameter access and modification
print("\n2. ROI parameter access:")
print(f"Rectangular ROI bounds: left={rectangular_roi.left:.2f}, right={rectangular_roi.right:.2f}")
print(f"                       top={rectangular_roi.top:.2f}, bottom={rectangular_roi.bottom:.2f}")
print(f"Line ROI endpoints: ({line_roi.x1:.2f}, {line_roi.y1:.2f}) to ({line_roi.x2:.2f}, {line_roi.y2:.2f})")
print(f"Circle ROI: center=({circle_roi.cx:.2f}, {circle_roi.cy:.2f}), radius={circle_roi.r:.2f}")

# 3. Programmatically modify ROI
print("\n3. Programmatic ROI modification:")
# Move the rectangular ROI
rectangular_roi.left = -2
rectangular_roi.right = 0
rectangular_roi.top = 1
rectangular_roi.bottom = 3
print("Moved rectangular ROI to new position")

# 4. ROI event handling
print("\n4. ROI event handling (for custom callbacks):")
def roi_changed_callback(roi):
    print(f"ROI changed! New bounds: {roi.left:.2f}, {roi.right:.2f}, {roi.top:.2f}, {roi.bottom:.2f}")

# Connect event (this would work in an interactive session)
# rectangular_roi.events.changed.connect(roi_changed_callback)

# %%
# ROI Best Practices and Tips
# ---------------------------

print("\n" + "="*60)
print("ROI BEST PRACTICES AND TIPS")
print("="*60)
print("""
1. CHOOSING THE RIGHT ROI TYPE:
   - RectangularROI: General rectangular selections, simple regions
   - Line2DROI: Line profiles, cross-sections, measuring distances
   - Point2DROI: Single pixel measurements, peak positions
   - CircleROI: Radially symmetric features, averaging around points
   - PolygonROI: Irregular shapes, complex boundaries

2. INTERACTIVE USAGE:
   - Always plot the signal before adding interactive ROIs
   - Use different colors for multiple ROIs to distinguish them
   - ROIs can be dragged and resized interactively
   - Use navigation_signal parameter to display ROI on different signal

3. PROGRAMMATIC CONTROL:
   - Access ROI parameters (left, right, top, bottom, etc.)
   - Modify ROI properties programmatically
   - Use events for real-time updates and callbacks
   - ROIs work in physical coordinates, not pixels

4. PERFORMANCE CONSIDERATIONS:
   - For large datasets, consider the computational cost of ROI operations
   - Use appropriate ROI sizes for your analysis needs
   - Multiple ROIs can be applied simultaneously

5. COORDINATE SYSTEMS:
   - ROIs use the signal's physical coordinate system
   - Proper axis calibration ensures ROIs point to correct regions
   - ROI coordinates are preserved across signal transformations
""")

print("\nInteractive ROI demonstration completed!")
print("Try manipulating the ROI widgets in the interactive plot!")
