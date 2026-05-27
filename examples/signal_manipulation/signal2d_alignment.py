"""
Signal2D image registration and alignment
=========================================

This example demonstrates image registration and alignment functionality
for 2D signals, including shift estimation and alignment with sub-pixel accuracy.
"""

# %%
# Create test data with known shifts
import numpy as np
import hyperspy.api as hs

# Create a test image with distinct features
def create_test_image(shift_x=0, shift_y=0, size=100):
    """Create a test image with circular and rectangular features"""
    x = np.linspace(-5, 5, size) + shift_x
    y = np.linspace(-5, 5, size) + shift_y
    X, Y = np.meshgrid(x, y)
    
    # Create multiple features for alignment
    circle1 = np.exp(-((X-1)**2 + (Y-1)**2)/0.5)  # Circle at (1,1)
    circle2 = np.exp(-((X+2)**2 + (Y-2)**2)/0.8)  # Circle at (-2,2)
    rectangle = np.exp(-np.maximum(np.abs(X+1), np.abs(Y+1))/0.3)  # Rectangle at (-1,-1)
    
    return circle1 + 0.7*circle2 + 0.5*rectangle

# Create an image stack with known shifts
n_images = 5
known_shifts = [(0, 0), (0.5, 0.3), (-0.3, 0.7), (0.8, -0.4), (-0.6, -0.5)]

image_data = np.zeros((n_images, 100, 100))
for i, (sx, sy) in enumerate(known_shifts):
    image_data[i] = create_test_image(sx, sy)
    # Add some noise
    image_data[i] += np.random.normal(0, 0.05, (100, 100))

# Create HyperSpy signal
s = hs.signals.Signal2D(image_data)
s.axes_manager.navigation_axes[0].name = 'Image'
s.axes_manager.signal_axes[0].name = 'Y'
s.axes_manager.signal_axes[0].units = 'μm'
s.axes_manager.signal_axes[0].scale = 0.1
s.axes_manager.signal_axes[1].name = 'X'
s.axes_manager.signal_axes[1].units = 'μm'
s.axes_manager.signal_axes[1].scale = 0.1

print(f"Created image stack: {s}")
print(f"Known shifts: {known_shifts}")

# %%
# Method 1: Estimate shifts only
print("\n--- Estimating shifts ---")
estimated_shifts = s.estimate_shift2D()
print(f"Estimated shifts: {estimated_shifts}")

# Compare with known shifts
print("\nComparison with known shifts:")
for i, (known, estimated) in enumerate(zip(known_shifts, estimated_shifts)):
    error_x = abs(known[0] - estimated[0])
    error_y = abs(known[1] - estimated[1])
    print(f"Image {i}: Known({known[0]:.1f}, {known[1]:.1f}) vs "
          f"Estimated({estimated[0]:.3f}, {estimated[1]:.3f}) - "
          f"Error: ({error_x:.3f}, {error_y:.3f})")

# %%
# Method 2: Estimate and align in separate steps
print("\n--- Separate estimation and alignment ---")
s_copy1 = s.deepcopy()
shifts = s_copy1.estimate_shift2D()
s_copy1.align2D(shifts=shifts)
print("Alignment completed using pre-estimated shifts")

# %%
# Method 3: Estimate and align in single step
print("\n--- Single-step alignment ---")
s_copy2 = s.deepcopy()
s_copy2.align2D()  # This estimates shifts internally
print("Single-step alignment completed")

# %%
# Method 4: Sub-pixel accuracy with upsampling
print("\n--- Sub-pixel accuracy methods ---")

# Using scikit-image upsampling
s_subpixel = s.deepcopy()
shifts_subpixel = s_subpixel.estimate_shift2D(sub_pixel_factor=10)
print(f"Sub-pixel shifts (upsampling): {shifts_subpixel}")

# For multi-dimensional datasets: statistical method
if s.axes_manager.navigation_size > 1:
    s_stat = s.deepcopy()
    shifts_stat = s_stat.estimate_shift2D(reference="stat")
    print(f"Statistical method shifts: {shifts_stat}")

# %%
# Visualize the alignment results using HyperSpy plotting
# Select first 3 images for comparison
original_images = [s.inav[i] for i in range(3)]
aligned_images = [s_copy2.inav[i] for i in range(3)]

# Plot original images
hs.plot.plot_images(original_images,
                   label=[f'Original Image {i}' for i in range(3)],
                   colorbar=True)

# Plot aligned images  
hs.plot.plot_images(aligned_images,
                   label=[f'Aligned Image {i}' for i in range(3)],
                   colorbar=True)

# %%
# Method 5: Parallel alignment for large datasets
print("\n--- Parallel alignment ---")
s_parallel = s.deepcopy()

# Estimate shifts first
shifts_parallel = s_parallel.estimate_shift2D()

# Align using multiple workers
s_parallel.align2D(shifts=shifts_parallel, num_workers=2)
print("Parallel alignment completed with 2 workers")

# %%
# Demonstrate alignment quality assessment
print("\n--- Alignment quality assessment ---")

def calculate_alignment_score(signal):
    """Calculate a simple alignment score based on image correlation"""
    if signal.axes_manager.navigation_size < 2:
        return None
    
    reference = signal.inav[0].data
    scores = []
    
    for i in range(1, signal.axes_manager.navigation_size):
        current = signal.inav[i].data
        # Normalized cross-correlation
        correlation = np.corrcoef(reference.flatten(), current.flatten())[0, 1]
        scores.append(correlation)
    
    return np.mean(scores)

# Compare alignment scores
original_score = calculate_alignment_score(s)
aligned_score = calculate_alignment_score(s_copy2)

if original_score is not None and aligned_score is not None:
    print(f"Original correlation score: {original_score:.4f}")
    print(f"Aligned correlation score: {aligned_score:.4f}")
    print(f"Improvement: {aligned_score - original_score:.4f}")
else:
    print("Alignment score calculation requires multiple images")

# %%
# Demonstrate shift correction analysis
print("\n--- Shift correction analysis ---")

# Calculate cumulative drift
cumulative_drift_x = np.cumsum([shift[0] for shift in estimated_shifts])
cumulative_drift_y = np.cumsum([shift[1] for shift in estimated_shifts])

# Create signals for the shift data to plot with HyperSpy
shift_x_signal = hs.signals.Signal1D([s[0] for s in estimated_shifts])
shift_y_signal = hs.signals.Signal1D([s[1] for s in estimated_shifts])

shift_x_signal.metadata.General.title = "X-direction shifts"
shift_y_signal.metadata.General.title = "Y-direction shifts"

# Use plot_spectra for side-by-side comparison - much cleaner than matplotlib subplots!
print("Plotting individual shifts using plot_spectra...")
hs.plot.plot_spectra([shift_x_signal, shift_y_signal], style='mosaic', legend='auto')

# Create trajectory signal for cumulative drift
trajectory_signal = hs.signals.Signal1D(cumulative_drift_y)
trajectory_signal.axes_manager[0].axis = cumulative_drift_x
trajectory_signal.metadata.General.title = "Drift trajectory (Y vs X cumulative)"
print("Plotting drift trajectory...")
trajectory_signal.plot()

# %%
# Important notes about alignment
print("\n--- Important notes ---")
print("1. align2D() modifies data in-place - use deepcopy() to preserve original")
print("2. Sub-pixel accuracy improves precision but increases computation time")
print("3. Statistical method works best for multi-dimensional datasets")
print("4. Parallel processing helps with large image stacks")
print("5. Always verify alignment quality using correlation or visual inspection")
