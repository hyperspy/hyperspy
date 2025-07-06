"""
Basic ROI application
=====================

This example demonstrates how to create and apply different types of 
Regions of Interest (ROI) to HyperSpy signals.
"""

# %%
# Create test signals
import numpy as np
import hyperspy.api as hs

# %%
# **Creating test signals for ROI demonstration**
#
# We'll create both 1D and 2D signals to show how ROIs work differently 
# depending on whether they're applied to navigation or signal dimensions.

# Create a 1D signal for ROI demonstration
s1d = hs.signals.Signal1D(np.arange(2000).reshape((20, 10, 10)))
# 1D signal: (10, 10|20) - 2D navigation, 1D signal

# Create a 2D signal for ROI demonstration  
s2d = hs.signals.Signal2D(np.arange(100).reshape((10, 10)))
# 2D signal: (|10, 10) - no navigation, 2D signal

# %%
# **Example 1: RectangularROI application**
#
# ROIs can be applied to different dimensions depending on the signal type.
# The same ROI behaves differently when applied to navigation vs signal dimensions.

rectangular_roi = hs.roi.RectangularROI(left=3, right=7, top=2, bottom=5)

# Apply ROI to 1D signal (crops navigation dimensions)
s1d_cropped = rectangular_roi(s1d)
# Result: Cropped navigation space, full signal dimension retained

# Apply ROI to 2D signal (crops signal dimensions)
s2d_cropped = rectangular_roi(s2d)
# Result: Cropped 2D signal space

# %%
# **Example 2: Different ROI types demonstration**
#
# HyperSpy offers various ROI types for different analysis needs.

# Point ROI
point_roi = hs.roi.Point2DROI(x=5, y=5)
point_result = point_roi(s2d)
# Point ROI extracts data at a specific coordinate

# Circular ROI
circle_roi = hs.roi.CircleROI(cx=5, cy=5, r=2)
circle_result = circle_roi(s2d)
print(f"Circle ROI result: {circle_result}")

# Line ROI
line_roi = hs.roi.Line2DROI(x1=2, y1=2, x2=8, y2=8, linewidth=1)
line_result = line_roi(s2d)
print(f"Line ROI result: {line_result}")

# %%
# Example 3: SpanROI for 1D signals
span_roi = hs.roi.SpanROI(left=3, right=7)

# Create a simple 1D signal for demonstration
simple_1d = hs.signals.Signal1D(np.random.randn(100))
simple_1d.axes_manager.signal_axes[0].scale = 0.1
simple_1d.axes_manager.signal_axes[0].offset = -5

span_result = span_roi(simple_1d)
print(f"\n1D signal after span ROI: {span_result}")

# %%
# Plot the ROI applications using HyperSpy's plotting functions

# Use HyperSpy's plot_images for the 2D signals
hs.plot.plot_images([s2d, s2d_cropped, circle_result],
                   label=['Original 2D signal', 'Rectangular ROI', 'Circle ROI'],
                   colorbar=True)

# Plot the 1D signals using HyperSpy's native plotting
line_result.plot()
simple_1d.plot()
span_result.plot()

# %%
# Example 4: ROI properties and physical coordinates
print("\n--- ROI Properties ---")
print(f"Rectangular ROI bounds: left={rectangular_roi.left}, right={rectangular_roi.right}")
print(f"                        top={rectangular_roi.top}, bottom={rectangular_roi.bottom}")
print(f"Circle ROI: center=({circle_roi.cx}, {circle_roi.cy}), radius={circle_roi.r}")
print(f"Span ROI: left={span_roi.left}, right={span_roi.right}")

# ROIs work with physical coordinates, not pixels
print(f"\nROI coordinates are in physical units, not pixels")
print(f"2D signal axes: X={s2d.axes_manager.signal_axes[1].units}, Y={s2d.axes_manager.signal_axes[0].units}")
