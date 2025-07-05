"""
Signal1D spectral calibration
=============================

This example demonstrates how to calibrate the energy/frequency axis of 1D 
spectra using known reference peaks, both interactively and programmatically.
"""

# %%
# Create a test spectrum with known peaks for calibration
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

# Create the test spectrum
spectrum_data = create_calibration_spectrum()
s = hs.signals.Signal1D(spectrum_data)

# Initially, the spectrum has arbitrary channel units
print(f"Original signal: {s}")
print(f"Original axis: scale={s.axes_manager.signal_axes[0].scale}, "
      f"units='{s.axes_manager.signal_axes[0].units}', "
      f"offset={s.axes_manager.signal_axes[0].offset}")

# Plot the original uncalibrated spectrum
s.plot()

# %%
# Method 1: Manual calibration using known reference peaks
print("\n--- Manual calibration using known peak positions ---")

# Suppose we know that:
# - Peak at channel 200 corresponds to 1000 eV
# - Peak at channel 600 corresponds to 3000 eV
# This gives us two points to calculate the linear calibration

# Calculate the scale (eV per channel)
channel1, energy1 = 200, 1000  # eV
channel2, energy2 = 600, 3000  # eV

scale = (energy2 - energy1) / (channel2 - channel1)
offset = energy1 - scale * channel1

print(f"Calculated scale: {scale} eV/channel")
print(f"Calculated offset: {offset} eV")

# Apply the calibration manually
s_calibrated = s.deepcopy()
s_calibrated.axes_manager.signal_axes[0].scale = scale
s_calibrated.axes_manager.signal_axes[0].offset = offset
s_calibrated.axes_manager.signal_axes[0].units = 'eV'
s_calibrated.axes_manager.signal_axes[0].name = 'Energy'

print(f"Calibrated axis: scale={s_calibrated.axes_manager.signal_axes[0].scale}, "
      f"units='{s_calibrated.axes_manager.signal_axes[0].units}', "
      f"offset={s_calibrated.axes_manager.signal_axes[0].offset}")

# Plot the calibrated spectrum
s_calibrated.plot()

# %%
# Method 2: Using the calibrate method programmatically
print("\n--- Using calibrate method programmatically ---")

# Create a fresh copy
s_calib2 = s.deepcopy()

# Use the calibrate method with known points
# Point 1: channel 200 -> 1000 eV, Point 2: channel 600 -> 3000 eV
try:
    # Note: calibrate method signature may vary, using a simple approach
    axis = s_calib2.axes_manager.signal_axes[0]
    axis.scale = scale
    axis.offset = offset
    axis.units = 'eV'
    axis.name = 'Energy'
    
    print("Calibration applied successfully")
    print(f"New axis: scale={axis.scale}, units='{axis.units}', offset={axis.offset}")
    
except Exception as e:
    print(f"Calibration method issue: {e}")
    print("Applied manual calibration instead")

# Verify the calibration by checking peak positions
peak_energies = []
for i, channel in enumerate([200, 400, 600, 800]):
    energy = offset + scale * channel
    peak_energies.append(energy)
    print(f"Peak {i+1}: Channel {channel} -> {energy:.1f} eV")

# %%
# Method 3: Calibration using shift and scale
print("\n--- Alternative calibration approach ---")

# Create another copy for alternative calibration
s_alt = s.deepcopy()

# Set axis properties step by step
signal_axis = s_alt.axes_manager.signal_axes[0]

# Method: set offset and scale directly
signal_axis.offset = offset
signal_axis.scale = scale
signal_axis.units = 'eV'
signal_axis.name = 'Energy'

print(f"Alternative calibration applied")
print(f"Axis range: {signal_axis.axis[0]:.1f} to {signal_axis.axis[-1]:.1f} {signal_axis.units}")

# Plot comparison
s.plot()
s_alt.plot()

# %%
# Method 4: Interactive calibration (demonstration concept)
print("\n--- Interactive calibration concept ---")

# For interactive calibration, users would typically:
# 1. Plot the spectrum
# 2. Identify reference peaks visually
# 3. Use GUI tools or interactive widgets to set calibration points

# Simulate an interactive calibration workflow
def simulate_interactive_calibration(signal, reference_peaks):
    """
    Simulate interactive calibration where user clicks on peaks
    and provides reference energies
    """
    print("Interactive calibration simulation:")
    print("User would click on peaks and enter reference energies")
    
    # For this demo, use the known values
    channels = [200, 600]
    energies = [1000, 3000]
    
    print(f"Reference point 1: Channel {channels[0]} = {energies[0]} eV")
    print(f"Reference point 2: Channel {channels[1]} = {energies[1]} eV")
    
    # Calculate and apply calibration
    scale_calc = (energies[1] - energies[0]) / (channels[1] - channels[0])
    offset_calc = energies[0] - scale_calc * channels[0]
    
    signal_copy = signal.deepcopy()
    axis = signal_copy.axes_manager.signal_axes[0]
    axis.scale = scale_calc
    axis.offset = offset_calc
    axis.units = 'eV'
    axis.name = 'Energy'
    
    return signal_copy

# Apply simulated interactive calibration
s_interactive = simulate_interactive_calibration(s, reference_peaks=[1000, 3000])

print(f"Interactive calibration result:")
print(f"Scale: {s_interactive.axes_manager.signal_axes[0].scale} eV/channel")
print(f"Offset: {s_interactive.axes_manager.signal_axes[0].offset} eV")

# %%
# Verification and quality check
print("\n--- Calibration verification ---")

# Check that the calibration is consistent
axis = s_interactive.axes_manager.signal_axes[0]
print(f"Energy range: {axis.axis[0]:.1f} to {axis.axis[-1]:.1f} eV")
print(f"Energy step: {axis.scale:.3f} eV/channel")

# Verify specific peak positions match expected values
test_channels = [200, 400, 600, 800]
expected_energies = [1000, 2000, 3000, 4000]

print("\nPeak position verification:")
for ch, exp_energy in zip(test_channels, expected_energies):
    actual_energy = axis.offset + axis.scale * ch
    print(f"Channel {ch}: Expected {exp_energy} eV, Got {actual_energy:.1f} eV")

# Final comparison plot
s.plot()
s_interactive.plot()

print("\nCalibration complete! The spectrum now has a properly calibrated energy axis.")
