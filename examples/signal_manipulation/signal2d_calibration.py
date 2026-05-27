"""
Signal2D calibration methods
============================

This example demonstrates how to calibrate the scale of 2D signals (images)
both interactively and programmatically.
"""

# %%
# Create a test image with known features for calibration
import numpy as np
import hyperspy.api as hs

# Create an image with calibration features (grid pattern)
def create_calibration_image(size=200, grid_spacing=20):
    """Create an image with a regular grid for calibration purposes"""
    x = np.arange(size)
    y = np.arange(size)
    X, Y = np.meshgrid(x, y)
    
    # Create grid pattern
    grid_x = np.sin(2 * np.pi * X / grid_spacing)
    grid_y = np.sin(2 * np.pi * Y / grid_spacing)
    
    # Combine patterns and add some structure
    image = (grid_x * grid_y > 0.7).astype(float)
    
    # Add some circular features
    for i in range(3):
        for j in range(3):
            cx, cy = 50 + i * 50, 50 + j * 50
            circle = np.exp(-((X - cx)**2 + (Y - cy)**2) / 100)
            image += 0.3 * circle
    
    return image

# Create the test image
image_data = create_calibration_image()
s = hs.signals.Signal2D(image_data)

# %%
# **Initial signal properties before calibration**
#
# Initially, the image has arbitrary pixel units. We'll demonstrate how to convert
# these to meaningful physical units through calibration.

# The uncalibrated signal has:
# - Scale = 1.0 (arbitrary units per pixel)
# - Units = '' (no physical meaning)
# - Offset = 0.0

# %%
# **Method 1: Non-interactive calibration with known coordinates**
#
# This approach is useful when you know the pixel coordinates and physical distance
# of features in your image, or when automating calibration procedures.

# Let's say we know that from pixel (50, 50) to pixel (150, 100) 
# the actual distance is 2.5 micrometers
s_calibrated = s.deepcopy()

# Calibrate using known coordinates
s_calibrated.calibrate(
    x0=50,      # Starting X pixel
    y0=50,      # Starting Y pixel  
    x1=150,     # Ending X pixel
    y1=100,     # Ending Y pixel
    new_length=2.5,  # Known distance
    units="μm",      # Units
    interactive=False
)

print(f"Calibrated signal: {s_calibrated}")
print(f"Calibrated X axis: scale={s_calibrated.axes_manager.signal_axes[1].scale:.4f}, units='{s_calibrated.axes_manager.signal_axes[1].units}'")
print(f"Calibrated Y axis: scale={s_calibrated.axes_manager.signal_axes[0].scale:.4f}, units='{s_calibrated.axes_manager.signal_axes[0].units}'")

# %%
# Method 2: Different calibration scenarios
print("\n--- Different calibration scenarios ---")

# Scenario A: Calibrate only X axis (isotropic scaling)
s_x_only = s.deepcopy()
s_x_only.calibrate(
    x0=0, y0=100, x1=200, y1=100,  # Horizontal line
    new_length=5.0,
    units="nm",
    interactive=False
)
print(f"X-only calibration: X scale={s_x_only.axes_manager.signal_axes[1].scale:.4f} nm/pixel")

# Scenario B: High-resolution calibration
s_hires = s.deepcopy()
s_hires.calibrate(
    x0=80, y0=80, x1=120, y1=120,  # Diagonal measurement
    new_length=0.5,  # Very small distance
    units="Å",  # Angstrom units
    interactive=False
)
print(f"High-res calibration: scale={s_hires.axes_manager.signal_axes[1].scale:.6f} Å/pixel")

# %%
# Method 3: Understanding the calibration calculation
print("\n--- Understanding calibration calculation ---")

def manual_calibration_calculation(x0, y0, x1, y1, new_length):
    """Calculate what the calibration does manually"""
    # Calculate pixel distance
    pixel_distance = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
    
    # Calculate scale (physical units per pixel)
    scale = new_length / pixel_distance
    
    return pixel_distance, scale

# Example calculation
x0, y0, x1, y1 = 50, 50, 150, 100
new_length = 2.5

pixel_dist, calculated_scale = manual_calibration_calculation(x0, y0, x1, y1, new_length)

print(f"Manual calculation:")
print(f"  Pixel distance: {pixel_dist:.2f} pixels")
print(f"  Physical distance: {new_length} μm")
print(f"  Calculated scale: {calculated_scale:.6f} μm/pixel")

# Verify against HyperSpy result
hyperspy_scale = s_calibrated.axes_manager.signal_axes[1].scale
print(f"  HyperSpy scale: {hyperspy_scale:.6f} μm/pixel")
print(f"  Match: {abs(calculated_scale - hyperspy_scale) < 1e-10}")

# %%
# Method 4: Working with different units
print("\n--- Working with different units ---")

# Create multiple calibrated versions with different units
unit_examples = [
    ("mm", 0.001),     # Millimeters
    ("μm", 1.0),       # Micrometers  
    ("nm", 1000.0),    # Nanometers
    ("Å", 10000.0),    # Angstroms
    ("pm", 1000000.0), # Picometers
]

print("Same physical calibration in different units:")
for units, scale_factor in unit_examples:
    s_temp = s.deepcopy()
    s_temp.calibrate(
        x0=50, y0=50, x1=150, y1=100,
        new_length=2.5 * scale_factor,  # Scale the length
        units=units,
        interactive=False
    )
    scale = s_temp.axes_manager.signal_axes[1].scale
    print(f"  {units}: {scale:.6f} {units}/pixel")

# %%
# Method 5: Calibration verification and visualization using HyperSpy plotting
print("\n--- Calibration verification ---")

# Plot original and calibrated images using HyperSpy's plot_images
hs.plot.plot_images([s, s_calibrated],
                   label=['Original (pixels)', 'Calibrated (μm)'],
                   colorbar=True)

# %%
# Method 6: Practical calibration workflow
print("\n--- Practical calibration workflow ---")

def calibration_workflow_example():
    """Example of a typical calibration workflow"""
    
    # Step 1: Load/create image
    signal = s.deepcopy()
    print("1. Loaded image data")
    
    # Step 2: Identify calibration features
    # (In practice, you would use s.plot() and identify features visually)
    print("2. Identified calibration features at known positions")
    
    # Step 3: Measure known distance
    # For this example, let's say we know the grid spacing is 1.2 μm
    known_spacing = 1.2  # μm
    
    # Step 4: Find pixel coordinates of the spacing
    # Grid spacing is 20 pixels in our synthetic data
    pixel_spacing = 20
    
    # Step 5: Apply calibration
    signal.calibrate(
        x0=0, y0=0, x1=pixel_spacing, y1=0,
        new_length=known_spacing,
        units="μm", 
        interactive=False
    )
    
    # Step 6: Verify calibration
    print(f"3. Applied calibration: {signal.axes_manager.signal_axes[1].scale:.4f} μm/pixel")
    
    # Step 7: Check consistency
    expected_scale = known_spacing / pixel_spacing
    actual_scale = signal.axes_manager.signal_axes[1].scale
    print(f"4. Expected scale: {expected_scale:.4f} μm/pixel")
    print(f"   Actual scale: {actual_scale:.4f} μm/pixel")
    print(f"   Calibration accurate: {abs(expected_scale - actual_scale) < 1e-6}")
    
    return signal

calibrated_workflow = calibration_workflow_example()

# %%
# Method 7: Interactive calibration setup (for GUI environments)
print("\n--- Interactive calibration ---")
print("For interactive calibration in GUI environments:")
print("1. s.plot()  # Display the image")
print("2. s.calibrate()  # Opens interactive calibration tool")
print("3. Click and drag to measure a known distance")
print("4. Enter the known length and units")
print("5. The calibration is applied automatically")
print("")
print("Note: Interactive calibration requires a GUI environment")
print("      and is best used in Jupyter notebooks or IPython with Qt backend")

# %%
# Method 8: Batch calibration for multiple images
print("\n--- Batch calibration for image stacks ---")

# Create a stack of images that need the same calibration
stack_data = np.array([create_calibration_image() for _ in range(3)])
s_stack = hs.signals.Signal2D(stack_data)

print(f"Image stack before calibration: {s_stack}")

# Apply the same calibration to all images in the stack
s_stack.calibrate(
    x0=50, y0=50, x1=150, y1=100,
    new_length=2.5,
    units="μm",
    interactive=False
)

print(f"Image stack after calibration: {s_stack}")
print("All images in the stack now have the same calibration")

# Verify that all navigation positions have the same scale
scales_x = [s_stack.inav[i].axes_manager.signal_axes[1].scale for i in range(3)]
scales_y = [s_stack.inav[i].axes_manager.signal_axes[0].scale for i in range(3)]

print(f"X scales across stack: {scales_x}")
print(f"Y scales across stack: {scales_y}")
print(f"Consistent calibration: {len(set(scales_x)) == 1 and len(set(scales_y)) == 1}")
