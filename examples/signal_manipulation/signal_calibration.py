"""
Signal calibration for 1D and 2D signals
=========================================

This example demonstrates how to calibrate the scale of both 1D spectra and 2D images
using known reference features, both interactively and programmatically.
"""

# %%
# Create test signals with known features for calibration
import numpy as np
import hyperspy.api as hs

def create_calibration_spectrum(size=1024, peaks=[200, 400, 600, 800]):
    """Create a spectrum with known peaks for calibration purposes"""
    x = np.arange(size)
    
    # Create background
    background = 50 + 10 * np.exp(-x/200) + 5 * np.random.random(size)
    
    # Add peaks at specified positions
    spectrum = background.copy()
    for peak_pos in peaks:
        # Add Gaussian peak
        peak = 100 * np.exp(-((x - peak_pos)**2) / (2 * 20**2))
        spectrum += peak
    
    return spectrum

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

# Create test signals
spectrum_data = create_calibration_spectrum()
s1d = hs.signals.Signal1D(spectrum_data)

image_data = create_calibration_image()
s2d = hs.signals.Signal2D(image_data)

print("Original signals before calibration:")
print(f"1D signal: {s1d}")
print(f"2D signal: {s2d}")

# %%
# **1D Signal Calibration: Manual calibration using known peak positions**
#
# For spectral data, we often know the energy/frequency of certain peaks.
# Here we'll calibrate based on known peak positions.

print("\n--- 1D Signal Calibration ---")

# Original axis information
axis = s1d.axes_manager.signal_axes[0]
print(f"Original axis: scale={axis.scale}, units='{axis.units}', offset={axis.offset}")

# Suppose we know that:
# - Peak at channel 200 corresponds to 1000 eV
# - Peak at channel 600 corresponds to 3000 eV
# This gives us two points to calculate the linear calibration

# Method 1: Direct axis manipulation
channel_1, energy_1 = 200, 1000  # eV
channel_2, energy_2 = 600, 3000  # eV

# Calculate calibration parameters
# energy = scale * channel + offset
# Using two points: energy_1 = scale * channel_1 + offset
#                  energy_2 = scale * channel_2 + offset
scale = (energy_2 - energy_1) / (channel_2 - channel_1)
offset = energy_1 - scale * channel_1

print(f"Calculated calibration: scale={scale:.3f} eV/channel, offset={offset:.3f} eV")

# Apply calibration
s1d.axes_manager.signal_axes[0].scale = scale
s1d.axes_manager.signal_axes[0].offset = offset
s1d.axes_manager.signal_axes[0].units = 'eV'

print(f"Calibrated 1D signal: scale={s1d.axes_manager.signal_axes[0].scale:.3f} eV/channel")

# %%
# **2D Signal Calibration: Calibration using known feature spacing**
#
# For images, we often know the physical size of certain features.
# Here we'll calibrate based on known grid spacing.

print("\n--- 2D Signal Calibration ---")

# Original axis information
x_axis = s2d.axes_manager.signal_axes[0]
y_axis = s2d.axes_manager.signal_axes[1]
print(f"Original axes: x scale={x_axis.scale}, y scale={y_axis.scale}")

# Suppose we know that the grid spacing corresponds to 50 nm
# The grid was created with spacing=20 pixels, so:
pixel_size = 50 / 20  # nm per pixel

# Method 1: Direct axis calibration
s2d.axes_manager.signal_axes[0].scale = pixel_size
s2d.axes_manager.signal_axes[0].units = 'nm'
s2d.axes_manager.signal_axes[1].scale = pixel_size
s2d.axes_manager.signal_axes[1].units = 'nm'

print(f"Calibrated 2D signal: scale={pixel_size:.3f} nm/pixel")

# %%
# **Method 2: Using calibrate method with reference points**
#
# HyperSpy provides a calibrate method for more advanced calibration scenarios.

# Reset signals for demonstration
s1d_reset = hs.signals.Signal1D(spectrum_data)
s2d_reset = hs.signals.Signal2D(image_data)

# For 1D calibration with multiple reference points
# This is useful when you have several known peaks
reference_1d = {
    200: 1000,  # channel 200 = 1000 eV
    400: 2000,  # channel 400 = 2000 eV  
    600: 3000,  # channel 600 = 3000 eV
    800: 4000   # channel 800 = 4000 eV
}

# Extract channels and energies for calibration
channels = np.array(list(reference_1d.keys()))
energies = np.array(list(reference_1d.values()))

# Perform linear fit for calibration
fit_params = np.polyfit(channels, energies, 1)
calibrated_scale = fit_params[0]
calibrated_offset = fit_params[1]

print(f"\nLinear fit calibration:")
print(f"Scale: {calibrated_scale:.3f} eV/channel")
print(f"Offset: {calibrated_offset:.3f} eV")

# Apply the calibration
s1d_reset.axes_manager.signal_axes[0].scale = calibrated_scale
s1d_reset.axes_manager.signal_axes[0].offset = calibrated_offset
s1d_reset.axes_manager.signal_axes[0].units = 'eV'

# %%
# **Verification and best practices**
#
# Always verify your calibration by checking known reference points.

print("\n--- Calibration Verification ---")

# Check 1D calibration
print("1D Calibration verification:")
axis_1d = s1d.axes_manager.signal_axes[0]
for channel, expected_energy in reference_1d.items():
    actual_energy = axis_1d.index2value(channel)
    print(f"Channel {channel}: Expected {expected_energy} eV, Got {actual_energy:.1f} eV")

# For 2D calibration, check known distances
print("\n2D Calibration verification:")
axis_2d = s2d.axes_manager.signal_axes[0]
print(f"Grid spacing: {20 * pixel_size:.1f} nm (expected: 50 nm)")
print(f"Image size: {s2d.axes_manager.signal_axes[0].size * pixel_size:.1f} nm")

# %%
# **Advanced calibration techniques**
#
# For more complex scenarios, you can use non-linear calibration or 
# interactive calibration methods.

print("\n--- Advanced Calibration Techniques ---")

# Method 3: Interactive calibration (for GUI environments)
# This allows you to click on known features and enter their values
print("Interactive calibration can be performed using:")
print("s.calibrate() - Opens an interactive calibration interface")

# Method 4: Using reference spectra or images
# You can also calibrate by comparing with a reference signal
print("\nReference-based calibration:")
print("Use s.align1D() or s.align2D() to align with reference signals")

# %%
# **Best practices for calibration**
#
# 1. Always verify your calibration with multiple reference points
# 2. Document your calibration procedure and reference values
# 3. Check for systematic errors or drift
# 4. Consider the precision and accuracy of your reference values

print("\n--- Best Practices Summary ---")
print("1. Use multiple reference points for better accuracy")
print("2. Verify calibration with independent measurements")
print("3. Document calibration parameters and procedure")
print("4. Consider measurement uncertainties")
print("5. Save calibrated signals with proper metadata")

# Final calibrated signals
print(f"\nFinal calibrated signals:")
print(f"1D signal: {s1d}")
print(f"2D signal: {s2d}")

# %%
# **Plotting the calibrated signals**
#
# Always plot your calibrated signals to verify the calibration makes sense.

# Plot calibrated 1D signal
s1d.plot()
s1d.axes_manager.signal_axes[0].name = 'Energy'

# Plot calibrated 2D signal  
s2d.plot()
s2d.axes_manager.signal_axes[0].name = 'x'
s2d.axes_manager.signal_axes[1].name = 'y'

print("\nCalibration complete! Check the plots to verify the calibration.")
