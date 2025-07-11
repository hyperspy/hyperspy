"""
Signal Cropping Methods
=======================

This example demonstrates different methods for cropping both 1D and 2D signals
in HyperSpy, including the crop_signal() method, isig[] indexing, and 
interactive ROI cropping. Learn the differences between these approaches
and when to use each method.
"""

import numpy as np
import hyperspy.api as hs

# %%
# ## Create Test Signals
# 
# We'll create both 1D and 2D signals to demonstrate cropping methods.

print("🔧 Creating Test Signals for Cropping")
print("=" * 50)

# Create 1D signal with two Gaussians
s1d = hs.data.two_gaussians()
print(f"1D signal: {s1d}")

# Create 2D signal (image) with structured features
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

# Create an image with concentric patterns
image_data = np.exp(-(X**2 + Y**2)/5) + 0.5*np.exp(-((X-2)**2 + (Y-2)**2)/2)
s2d = hs.signals.Signal2D(image_data)

# Configure 2D signal axes
s2d.axes_manager.signal_axes[0].name = 'Y'
s2d.axes_manager.signal_axes[0].units = 'μm'
s2d.axes_manager.signal_axes[0].scale = 0.1
s2d.axes_manager.signal_axes[0].offset = -5
s2d.axes_manager.signal_axes[1].name = 'X'
s2d.axes_manager.signal_axes[1].units = 'μm'
s2d.axes_manager.signal_axes[1].scale = 0.1
s2d.axes_manager.signal_axes[1].offset = -5
s2d.metadata.General.title = 'Test 2D Signal'

print(f"2D signal: {s2d}")

# %%
# ## Method 1: Using crop_signal() Method
# 
# The crop_signal() method modifies the signal in-place, permanently changing the data.

print("\n✂️ Method 1: crop_signal() - In-place Cropping")
print("=" * 50)

# 1D Cropping with crop_signal()
s1d_crop1 = s1d.deepcopy()  # Create copy to preserve original
print(f"Original 1D range: {s1d_crop1.axes_manager.signal_axes[0].axis[0]:.2f} to "
      f"{s1d_crop1.axes_manager.signal_axes[0].axis[-1]:.2f}")

# Crop 1D signal between specific values
s1d_crop1.crop_signal(left_value=2.0, right_value=8.0)
print(f"Cropped 1D range: {s1d_crop1.axes_manager.signal_axes[0].axis[0]:.2f} to "
      f"{s1d_crop1.axes_manager.signal_axes[0].axis[-1]:.2f}")

# 2D Cropping with crop_signal()
s2d_crop1 = s2d.deepcopy()  # Create copy to preserve original
print(f"Original 2D shape: {s2d_crop1.data.shape}")

# Crop 2D signal using coordinate ranges
s2d_crop1.crop_signal(top=2.0, bottom=-2.0, left=-1.0, right=3.0)
print(f"Cropped 2D shape: {s2d_crop1.data.shape}")

print("✅ crop_signal() permanently modifies the signal")

# %%
# ## Method 2: Using isig[] Indexing
# 
# Indexing with isig[] creates a new signal with the cropped region.

print("\n🎯 Method 2: isig[] Indexing - Non-destructive Cropping")
print("=" * 50)

# 1D Cropping with isig[]
s1d_crop2 = s1d.isig[2.0:8.0]  # Same range as crop_signal example
print(f"1D isig cropping: {s1d_crop2.axes_manager.signal_axes[0].axis[0]:.2f} to "
      f"{s1d_crop2.axes_manager.signal_axes[0].axis[-1]:.2f}")

# 2D Cropping with isig[]
s2d_crop2 = s2d.isig[-2.0:2.0, -1.0:3.0]  # Y-axis, X-axis ranges
print(f"2D isig cropping shape: {s2d_crop2.data.shape}")

# Advanced indexing: every other pixel
s2d_crop3 = s2d.isig[::2, ::2]  # Subsample by factor of 2
print(f"2D subsampled shape: {s2d_crop3.data.shape}")

print("✅ isig[] preserves original signal and creates new cropped signal")

# %%
# ## Method 3: Interactive ROI Cropping
# 
# Use Region of Interest (ROI) for interactive cropping.

print("\n🎮 Method 3: Interactive ROI Cropping")
print("=" * 50)

# 1D ROI - SpanROI for signal range selection
roi_1d = hs.roi.SpanROI(left=3.0, right=7.0)
s1d_roi = roi_1d(s1d)
print(f"1D ROI cropping: {s1d_roi.axes_manager.signal_axes[0].axis[0]:.2f} to "
      f"{s1d_roi.axes_manager.signal_axes[0].axis[-1]:.2f}")

# 2D ROI - RectangularROI for image region selection
roi_2d = hs.roi.RectangularROI(left=-1.0, top=-1.5, right=2.0, bottom=1.5)
s2d_roi = roi_2d(s2d)
print(f"2D ROI cropping shape: {s2d_roi.data.shape}")

print("✅ ROI objects can be modified interactively and reused")

# %%
# ## Navigation vs Signal Cropping
# 
# Demonstrate cropping in navigation space vs signal space.

print("\n📊 Navigation vs Signal Space Cropping")
print("=" * 50)

# For multidimensional data, distinguish between navigation and signal cropping
if s1d.axes_manager.navigation_dimension > 0:
    # Crop navigation space using inav[]
    s1d_nav_crop = s1d.inav[2:8, 3:7]  # Crop navigation dimensions
    print(f"Navigation cropped 1D: {s1d_nav_crop}")
    
    # Crop signal space using isig[]
    s1d_sig_crop = s1d.isig[2.0:8.0]  # Crop signal dimension
    print(f"Signal cropped 1D: {s1d_sig_crop}")

print("✅ Use inav[] for navigation space, isig[] for signal space")

# %%
# ## Comparison and Visualization
# 
# Compare different cropping methods visually.

print("\n🎨 Visualization of Cropping Methods")
print("=" * 50)

# Plot original and cropped 1D signals
s1d.plot()
s1d_crop2.plot()

# Plot original and cropped 2D signals
s2d.plot()
s2d_crop2.plot()

print("✅ All cropping methods demonstrated successfully")

# %%
# ## Best Practices Summary
# 
# Guidelines for choosing the right cropping method.

print("\n✅ Cropping Method Selection Guide")
print("=" * 50)

print("1. 🔧 crop_signal() method:")
print("   • Use when: Permanently reducing data size")
print("   • Memory: Modifies original signal in-place")
print("   • Reversible: No, original data is lost")

print("\n2. 🎯 isig[] indexing:")
print("   • Use when: Creating new cropped views")
print("   • Memory: Creates new signal object")
print("   • Reversible: Yes, original signal preserved")

print("\n3. 🎮 ROI objects:")
print("   • Use when: Interactive analysis needed")
print("   • Memory: Creates new signal object")
print("   • Reversible: Yes, ROI can be modified")

print("\n4. 📊 Navigation vs Signal:")
print("   • inav[]: Crop navigation dimensions (spatial, time, etc.)")
print("   • isig[]: Crop signal dimensions (energy, wavelength, etc.)")

print("\nKey advantages:")
print("• crop_signal(): Memory efficient for permanent reduction")
print("• isig[]: Flexible and reversible")
print("• ROI: Interactive and reusable")
print("• Calibrated units: All methods work with physical units")

print(f"\nCropping examples completed successfully!")
print(f"Original 1D signal: {s1d.data.shape}, Original 2D signal: {s2d.data.shape}")
