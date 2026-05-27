"""
Signal2D cropping and ROI operations
====================================

This example demonstrates different methods for cropping 2D signals (images)
in HyperSpy, including crop_signal(), isig indexing, and interactive ROI.
"""

# %%
# Create test data
import numpy as np
import hyperspy.api as hs

# Create a test image with some structure
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

# Create an image with concentric circles
image_data = np.exp(-(X**2 + Y**2)/5) + 0.5*np.exp(-((X-2)**2 + (Y-2)**2)/2)
im = hs.signals.Signal2D(image_data)

# Set up axes using batch assignment
im.axes_manager.signal_axes[0].name = 'Y'
im.axes_manager.signal_axes[0].units = 'μm'
im.axes_manager.signal_axes[0].scale = 0.1
im.axes_manager.signal_axes[0].offset = -5
im.axes_manager.signal_axes[1].name = 'X'
im.axes_manager.signal_axes[1].units = 'μm'
im.axes_manager.signal_axes[1].scale = 0.1
im.axes_manager.signal_axes[1].offset = -5

im.metadata.General.title = 'Test image with circular features'
print(f"Original image: {im}")

# %%
# Method 1: Using crop_signal() method (modifies in-place)
im1 = im.deepcopy()
im1.crop_signal(left=0.5, top=0.7, bottom=2.0)
print(f"After crop_signal: {im1}")

# %%
# Method 2: Using isig[] indexing (creates views)
im2 = im.isig[0.5:, 0.7:2.0]  # Crop from x=0.5 to end, y=0.7 to 2.0
print(f"Using isig[0.5:, 0.7:2.0]: {im2}")

# More complex indexing
im3 = im.isig[1.0:4.0, -2.0:3.0]  # Specific region
print(f"Using isig[1.0:4.0, -2.0:3.0]: {im3}")

# %%
# Method 3: ROI-based cropping
rect_roi = hs.roi.RectangularROI(left=1.0, right=4.0, top=-1.0, bottom=2.0)
im_roi = rect_roi(im)
print(f"Rectangular ROI result: {im_roi}")

# Circle ROI
circle_roi = hs.roi.CircleROI(cx=0.0, cy=0.0, r=3.0)
im_circle = circle_roi(im)
print(f"Circle ROI result: {im_circle}")

# %%
# Plot comparison of different cropping methods using HyperSpy's plot_images
hs.plot.plot_images([im, im1, im2, im3, im_roi, im_circle],
                   label=['Original image', 'crop_signal()', 'isig[0.5:, 0.7:2.0]',
                          'isig[1.0:4.0, -2.0:3.0]', 'Rectangular ROI', 'Circle ROI'],
                   cmap='viridis',
                   colorbar=True)

# %%
# Working with image stacks
print("\n--- Working with image stacks ---")

# Create an image stack (navigation dim 1, signal dim 2)
stack_data = np.random.random((10, 50, 50))
# Add some systematic variation
for i in range(10):
    x_offset = (i - 5) * 0.5
    y_offset = (i - 5) * 0.3
    X_shifted = X[25:75, 25:75] - x_offset
    Y_shifted = Y[25:75, 25:75] - y_offset
    stack_data[i] = np.exp(-(X_shifted**2 + Y_shifted**2)/3)

stack = hs.signals.Signal2D(stack_data)
stack.axes_manager.navigation_axes[0].name = 'Frame'
stack.axes_manager.signal_axes[0].name = 'Y'
stack.axes_manager.signal_axes[1].name = 'X'

print(f"Image stack: {stack}")

# Crop the entire stack
stack_cropped = stack.isig[10:40, 10:40]
print(f"Cropped stack: {stack_cropped}")

# %%
# Apply ROI to image stack
rect_roi_stack = hs.roi.RectangularROI(left=10, right=40, top=10, bottom=40)
stack_roi = rect_roi_stack(stack)
print(f"Stack with ROI: {stack_roi}")

# %%
# Demonstrate coordinate vs pixel-based operations
print("\n--- Coordinate vs pixel operations ---")

# Check the actual coordinate ranges
print(f"Original image X range: {im.axes_manager.signal_axes[1].axis[0]:.2f} to {im.axes_manager.signal_axes[1].axis[-1]:.2f}")
print(f"Original image Y range: {im.axes_manager.signal_axes[0].axis[0]:.2f} to {im.axes_manager.signal_axes[0].axis[-1]:.2f}")

# ROI coordinates are in physical units
print(f"ROI coordinates (physical): left={rect_roi.left}, right={rect_roi.right}")
print(f"ROI coordinates (physical): top={rect_roi.top}, bottom={rect_roi.bottom}")

# Compare with pixel-based indexing
print(f"Equivalent pixel ranges would need conversion from physical coordinates")

# %%
# Interactive ROI demonstration (for interactive use)
print("\n--- Interactive ROI setup ---")
interactive_roi = hs.roi.RectangularROI()  # Will auto-center when used interactively
print("Interactive ROI created - use roi.interactive(signal) for interactive manipulation")

# Example of how to use (commented out since it requires GUI)
# im.plot()
# interactive_result = interactive_roi.interactive(im)
print("Note: Interactive features require GUI environment")
