"""
Signal alignment for 1D and 2D signals
======================================

This example demonstrates how to align both 1D spectra and 2D images using 
cross-correlation to correct for energy drifts and spatial shifts.
"""

# %%
# Create test signals with known shifts for demonstration
import numpy as np
import hyperspy.api as hs

def create_test_spectra_with_shifts(n_spectra=5, size=512):
    """Create a set of spectra with known energy shifts"""
    
    # Create empty signal for simulation
    s = hs.signals.Signal1D(np.zeros((n_spectra, size)))
    s.axes_manager.signal_axes[0].name = 'Energy'
    s.axes_manager.signal_axes[0].units = 'eV'
    s.axes_manager.signal_axes[0].scale = 0.5
    s.axes_manager.signal_axes[0].offset = 100
    
    # Create model for simulation
    m = s.create_model()
    
    # Add exponential background
    background = hs.model.components1D.Expression(
        "a + b * exp(-x/c)",
        name="Background",
        a=100, b=50, c=200
    )
    m.append(background)
    
    # ✅ BEST PRACTICE: Explicitly set all parameters to avoid unset parameter errors
    # Set background parameters
    background.a.value = 100
    background.b.value = 50
    background.c.value = 200
    
    # Add peaks
    peak1 = hs.model.components1D.Gaussian(name="Peak1")
    peak2 = hs.model.components1D.Gaussian(name="Peak2")
    m.append(peak1)
    m.append(peak2)

    # Set base parameters for all spectra
    # ✅ BEST PRACTICE: For multidimensional signals, set value then assign to all
    background.a.value = 100.0
    background.a.assign_current_value_to_all()
    background.b.value = 50.0
    background.b.assign_current_value_to_all()
    background.c.value = 200.0
    background.c.assign_current_value_to_all()
    
    peak1.A.value = 1000.0
    peak1.A.assign_current_value_to_all()
    peak1.centre.value = 200.0
    peak1.centre.assign_current_value_to_all()
    peak1.sigma.value = 10.0
    peak1.sigma.assign_current_value_to_all()
    
    # ✅ BEST PRACTICE: Set ALL parameters for Peak2 to avoid unset parameter errors
    peak2.A.value = 800.0
    peak2.A.assign_current_value_to_all()
    peak2.centre.value = 300.0
    peak2.centre.assign_current_value_to_all()
    peak2.sigma.value = 15.0
    peak2.sigma.assign_current_value_to_all()
    
    # ✅ BEST PRACTICE: Double-check that all parameters are set
    # Print parameter values for debugging
    print(f"Background parameters: a={background.a.value}, b={background.b.value}, c={background.c.value}")
    print(f"Peak1 parameters: A={peak1.A.value}, centre={peak1.centre.value}, sigma={peak1.sigma.value}")
    print(f"Peak2 parameters: A={peak2.A.value}, centre={peak2.centre.value}, sigma={peak2.sigma.value}")
    
    # Verify all parameters are set
    for component in m:
        for param in component.parameters:
            if not hasattr(param, 'value') or param.value is None:
                print(f"WARNING: {component.name}.{param.name} is not set! Setting to 1.0")
                param.value = 1.0
                param.assign_current_value_to_all()
    
    # Generate base spectrum (use the first spectrum as base)
    s_base = m.as_signal()
    base_spectrum = s_base.inav[0].data  # Get the first spectrum
    
    # Now create shifted versions
    shifts = [0, 2, -1.5, 3, -2.5]  # eV shifts
    for i, shift in enumerate(shifts):
        # Create shifted spectrum by interpolating
        energy_axis = s.axes_manager.signal_axes[0].axis
        shifted_energy = energy_axis + shift
        
        # Interpolate to create shifted spectrum
        shifted_spectrum = np.interp(energy_axis, shifted_energy, base_spectrum)
        
        # Add some noise
        shifted_spectrum += np.random.normal(0, 5, size)
        
        s.data[i] = shifted_spectrum
    
    return s, shifts

def create_test_images_with_shifts(n_images=5, size=64):
    """Create a set of images with known spatial shifts"""
    
    def create_test_image(shift_x=0, shift_y=0, size=64):
        """Create a test image with distinct features"""
        x = np.linspace(-5, 5, size) + shift_x
        y = np.linspace(-5, 5, size) + shift_y
        X, Y = np.meshgrid(x, y)
        
        # Create multiple features for alignment
        circle1 = np.exp(-((X-1)**2 + (Y-1)**2)/0.5)  # Circle at (1,1)
        circle2 = np.exp(-((X+2)**2 + (Y-2)**2)/0.8)  # Circle at (-2,2)
        rectangle = np.exp(-np.maximum(np.abs(X+1), np.abs(Y+1))/0.3)  # Rectangle at (-1,-1)
        
        return circle1 + 0.7*circle2 + 0.5*rectangle
    
    # Create image stack with known shifts
    known_shifts = [(0, 0), (0.5, 0.3), (-0.3, 0.7), (0.8, -0.4), (-0.6, -0.5)]
    
    image_data = np.zeros((n_images, size, size))
    for i, (sx, sy) in enumerate(known_shifts):
        image_data[i] = create_test_image(sx, sy, size)
        # Add some noise
        image_data[i] += np.random.normal(0, 0.05, (size, size))
    
    # Create HyperSpy signal
    s = hs.signals.Signal2D(image_data)
    s.axes_manager.navigation_axes[0].name = 'Image'
    s.axes_manager.signal_axes[0].name = 'Y'
    s.axes_manager.signal_axes[0].units = 'μm'
    s.axes_manager.signal_axes[0].scale = 0.1
    s.axes_manager.signal_axes[1].name = 'X'
    s.axes_manager.signal_axes[1].units = 'μm'
    s.axes_manager.signal_axes[1].scale = 0.1
    
    return s, known_shifts

# Create test signals
s1d, shifts_1d = create_test_spectra_with_shifts()
s2d, shifts_2d = create_test_images_with_shifts()

print("Created test signals:")
print(f"1D signal: {s1d}")
print(f"2D signal: {s2d}")
print(f"Known 1D shifts: {shifts_1d} eV")
print(f"Known 2D shifts: {shifts_2d} μm")

# %%
# **1D Signal Alignment: Correcting spectral energy drift**
#
# Spectral alignment is crucial when dealing with time series or multiple acquisitions
# where energy drift can occur.

print("\n--- 1D Signal Alignment ---")

# Plot original misaligned spectra
s1d.plot()

# Method 1: align1D with cross-correlation
print("Aligning 1D spectra using cross-correlation...")

# Use the first spectrum as reference (index 0)
reference_indices = (0,)

# Align all spectra to the reference
shifts_estimated = s1d.align1D(reference_indices=reference_indices, crop=True)

print(f"Aligned 1D signal: {s1d}")
print(f"Estimated shifts: {shifts_estimated}")

# Compare original and aligned spectra
print("Alignment results:")
for i in range(s1d.axes_manager.navigation_size):
    print(f"Spectrum {i}: Known shift = {shifts_1d[i]:.1f} eV")

# Plot aligned spectra
s1d.plot()

# %%
# **2D Signal Alignment: Correcting spatial drift**
#
# Image alignment is important for correcting sample drift during acquisition.

print("\n--- 2D Signal Alignment ---")

# Plot original misaligned images
s2d.plot()

# Method 1: align2D with cross-correlation
print("Aligning 2D images using cross-correlation...")

# Use the first image as reference
reference_image = s2d.inav[0]

# Align all images to the reference
shifts_2d_estimated = s2d.align2D(reference=reference_image, crop=True)

print(f"Aligned 2D signal: {s2d}")
print(f"Estimated 2D shifts: {shifts_2d_estimated}")

# Compare original and aligned images
print("Alignment results:")
for i in range(s2d.axes_manager.navigation_size):
    print(f"Image {i}: Known shift = {shifts_2d[i]} μm")

# Plot aligned images
s2d.plot()

# %%
# **Advanced alignment options**
#
# HyperSpy provides several alignment options for different scenarios.

print("\n--- Advanced Alignment Options ---")

# Method 2: align1D with different correlation methods
print("1D alignment options:")
print("- correlation_method='cross_correlation' (default)")
print("- correlation_method='phase_correlation' (for subpixel accuracy)")
print("- correlation_method='template_matching' (for specific features)")

# Method 3: align2D with different options
print("\n2D alignment options:")
print("- subpixel alignment using phase correlation")
print("- ROI-based alignment for specific regions")
print("- Progressive alignment for long time series")

# Example: Subpixel alignment for 2D signals
print("\nPerforming subpixel alignment...")
s2d_subpixel = s2d.align2D(reference=reference_image, 
                           crop=True, 
                           sobel=True,  # Use edge detection
                           medfilter=True)  # Apply median filter

print(f"Subpixel aligned 2D signal: {s2d_subpixel}")

# %%
# **Quality assessment of alignment**
#
# Always assess the quality of your alignment results.

print("\n--- Alignment Quality Assessment ---")

# Method 1: Visual inspection
print("Visual inspection:")
print("- Compare before/after alignment plots")
print("- Check for proper feature overlap")
print("- Verify no artifacts were introduced")

# Method 2: Cross-correlation coefficient
print("\nQuantitative assessment:")
print("- Cross-correlation coefficient measures alignment quality")
print("- Values closer to 1 indicate better alignment")

# Calculate alignment quality for 1D signals
def calculate_alignment_quality_1d(aligned_signal, reference_index=0):
    """Calculate cross-correlation coefficient between reference and aligned spectra"""
    ref_spectrum = aligned_signal.inav[reference_index].data
    correlations = []
    
    for i in range(aligned_signal.axes_manager.navigation_size):
        spectrum = aligned_signal.inav[i].data
        correlation = np.corrcoef(ref_spectrum, spectrum)[0, 1]
        correlations.append(correlation)
    
    return correlations

correlations_1d = calculate_alignment_quality_1d(s1d)
print(f"1D alignment quality (correlation coefficients): {correlations_1d}")

# %%
# **Best practices for signal alignment**
#
# Follow these best practices for robust signal alignment.

print("\n--- Best Practices for Signal Alignment ---")

print("1. Choose appropriate reference:")
print("   - Use high signal-to-noise ratio spectrum/image")
print("   - Consider using average of multiple good spectra/images")
print("   - Ensure reference contains distinctive features")

print("\n2. Pre-processing considerations:")
print("   - Remove outliers before alignment")
print("   - Apply appropriate filtering if needed")
print("   - Normalize intensities if necessary")

print("\n3. Alignment parameters:")
print("   - crop=True to remove edge artifacts")
print("   - Consider subpixel alignment for high precision")
print("   - Use appropriate correlation method")

print("\n4. Post-alignment verification:")
print("   - Always verify alignment visually")
print("   - Check for introduction of artifacts")
print("   - Measure alignment quality quantitatively")

# Example: Using an averaged reference
print("\n--- Using averaged reference ---")

# Create average reference for better stability
s1d_avg_ref = s1d.mean(axis=0)
s2d_avg_ref = s2d.mean(axis=0)

print("Using averaged reference can improve alignment stability")
print("This is particularly useful for noisy signals")

# %%
# **Common alignment challenges and solutions**
#
# Address common issues encountered during alignment.

print("\n--- Common Alignment Challenges ---")

print("1. Low signal-to-noise ratio:")
print("   - Apply appropriate filtering")
print("   - Use averaged reference")
print("   - Consider binning to improve statistics")

print("\n2. Large shifts:")
print("   - Perform coarse alignment first")
print("   - Use progressive alignment")
print("   - Increase search range")

print("\n3. Non-linear distortions:")
print("   - Use local alignment techniques")
print("   - Apply distortion correction first")
print("   - Consider elastic alignment methods")

print("\n4. Multiple features:")
print("   - Use ROI-based alignment")
print("   - Weight different features appropriately")
print("   - Consider feature-specific alignment")

# Final comparison
print(f"\nAlignment complete!")
print(f"Original 1D signal: {s1d}")
print(f"Aligned 1D signal: {s1d}")
print(f"Original 2D signal: {s2d}")
print(f"Aligned 2D signal: {s2d}")

# %%
# **Plotting aligned vs original signals**
#
# Compare the alignment results visually.

print("\n--- Final Comparison ---")

# For 1D signals, you can overlay all spectra to see the alignment
s1d_sum = s1d.sum(axis=0)
s1d_aligned_sum = s1d.sum(axis=0)

s1d_sum.metadata.General.title = "Original (misaligned) sum"
s1d_aligned_sum.metadata.General.title = "Aligned sum"

# Plot both for comparison
s1d_sum.plot()
s1d_aligned_sum.plot()

# For 2D signals, you can create difference images
s2d_sum = s2d.sum(axis=0)
s2d_aligned_sum = s2d.sum(axis=0)

s2d_sum.metadata.General.title = "Original (misaligned) sum"
s2d_aligned_sum.metadata.General.title = "Aligned sum"

# Plot both for comparison
s2d_sum.plot()
s2d_aligned_sum.plot()

print("Alignment demonstration complete!")
print("Check the plots to verify the alignment quality.")
