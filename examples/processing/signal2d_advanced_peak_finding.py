"""
2D Peak Finding with Different Methods
======================================

This example demonstrates various methods for finding peaks in 2D signals,
such as images and diffraction patterns.
"""

import numpy as np
import hyperspy.api as hs
from scipy import ndimage
import matplotlib.pyplot as plt

# %%
# ## Creating test 2D signals with synthetic peaks
# 
# We'll create synthetic data with known peak locations to test different peak finding methods

# %%
# Create synthetic 2D data with peaks

def create_2d_gaussian_peak(shape, center, width, amplitude=1.0):
    """Create a 2D Gaussian peak."""
    y, x = np.ogrid[:shape[0], :shape[1]]
    peak = amplitude * np.exp(-((x - center[1])**2 + (y - center[0])**2) / (2 * width**2))
    return peak

# Create base signal with noise
base_shape = (100, 100)
noise_level = 0.1
base_data = np.random.random(base_shape) * noise_level

# Add several peaks at known locations
peak_centers = [(25, 30), (70, 40), (30, 75), (80, 85)]
peak_widths = [3, 4, 2.5, 3.5]
peak_amplitudes = [1.0, 0.8, 1.2, 0.9]

for center, width, amplitude in zip(peak_centers, peak_widths, peak_amplitudes):
    base_data += create_2d_gaussian_peak(base_shape, center, width, amplitude)

# Create HyperSpy signal
signal_2d = hs.signals.Signal2D(base_data)
signal_2d.axes_manager[0].name = 'y'
signal_2d.axes_manager[1].name = 'x'
signal_2d.axes_manager[0].units = 'pixel'
signal_2d.axes_manager[1].units = 'pixel'
signal_2d.metadata.General.title = "2D Signal with Peaks"

# **Signal with known peaks created**
#
# Our synthetic 2D signal contains 4 Gaussian peaks at known locations:
# - Peak positions: (25,30), (70,40), (30,75), (80,85)  
# - Different widths and amplitudes for testing robustness
# - Added realistic noise to test algorithm performance

# %%
# ## Method 1: Template matching for peak finding
# 
# Template matching finds peaks by correlating a known peak shape with the image

# Create a template (small Gaussian)
template_size = 15
template_center = template_size // 2
template = create_2d_gaussian_peak((template_size, template_size), 
                                   (template_center, template_center), 
                                   3.0, 1.0)

# Normalize template
template = template / np.sum(template)

# Apply template matching using correlation
from scipy.signal import correlate2d
correlation = correlate2d(signal_2d.data, template, mode='same')

# Find local maxima in correlation
from scipy.ndimage import maximum_filter
local_maxima = maximum_filter(correlation, size=10) == correlation

# Apply threshold
threshold = 0.7 * np.max(correlation)
peaks_template = np.where((local_maxima) & (correlation > threshold))

# Results from template matching
print(f"Template matching found {len(peaks_template[0])} peaks")
for i, (y, x) in enumerate(zip(peaks_template[0], peaks_template[1])):
    print(f"  Peak {i+1}: ({y}, {x})")

# %%
# ## Method 2: Using scikit-image peak finding
# 
# Scikit-image provides robust peak detection algorithms with good parameter control

try:
    from skimage.feature import peak_local_maxima
    from skimage.filters import gaussian
    
    # Apply slight smoothing to reduce noise
    smoothed_data = gaussian(signal_2d.data, sigma=1.0)
    
    # Find local maxima
    coordinates = peak_local_maxima(smoothed_data, 
                                    min_distance=8,  # Minimum distance between peaks
                                    threshold_abs=0.5,  # Absolute threshold
                                    threshold_rel=0.3)  # Relative threshold
    
    print(f"Scikit-image found {len(coordinates)} peaks")
    for i, coord in enumerate(coordinates):
        print(f"  Peak {i+1}: ({coord[0]}, {coord[1]})")
        
    skimage_available = True
    
except ImportError:
    print("Scikit-image not available for peak detection")
    skimage_available = False
    coordinates = np.empty((0, 2), dtype=int)

# %%
# Method 3: Simple threshold-based peak finding

print("\\n=== Method 3: Threshold-based Detection ===")

# Apply Gaussian filter to smooth noise
smoothed = ndimage.gaussian_filter(signal_2d.data, sigma=1.5)

# Find local maxima using maximum filter
local_max = ndimage.maximum_filter(smoothed, size=7) == smoothed

# Apply intensity threshold
intensity_threshold = 0.6  # Adjust based on your data
high_intensity = smoothed > intensity_threshold

# Combine conditions
peaks_threshold = local_max & high_intensity
peak_positions = np.where(peaks_threshold)

print(f"Threshold method found {len(peak_positions[0])} peaks")
for i, (y, x) in enumerate(zip(peak_positions[0], peak_positions[1])):
    print(f"  Peak {i+1}: ({y}, {x}) - Intensity: {smoothed[y, x]:.3f}")

# %%
# Method 4: Using HyperSpy's built-in peak finding

print("\\n=== Method 4: HyperSpy Peak Finding ===")

# HyperSpy provides some peak finding functionality
# Find peaks using a simple approach with HyperSpy tools

# Create a copy for processing
signal_copy = signal_2d.deepcopy()

# Apply some preprocessing
signal_copy.data = ndimage.gaussian_filter(signal_copy.data, sigma=1.0)

# For 2D signals, we can use scipy's peak finding on flattened data or per-line
print("HyperSpy peak finding works best with 1D spectra or spectrum images")
print("For 2D peak detection in images, use the methods above")

# %%
# Visualization of results

print("\\n=== Visualizing Peak Detection Results ===")

# Create a figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('2D Peak Finding Methods Comparison')

# Plot 1: Original data with known peaks
ax1 = axes[0, 0]
im1 = ax1.imshow(signal_2d.data, cmap='viridis', origin='lower')
ax1.scatter([c[1] for c in peak_centers], [c[0] for c in peak_centers], 
           c='red', s=100, marker='x', linewidths=3, label='True peaks')
ax1.set_title('Original Data with True Peaks')
ax1.legend()
plt.colorbar(im1, ax=ax1)

# Plot 2: Template matching results
ax2 = axes[0, 1]
im2 = ax2.imshow(signal_2d.data, cmap='viridis', origin='lower')
ax2.scatter(peaks_template[1], peaks_template[0], 
           c='cyan', s=80, marker='o', alpha=0.7, label='Template matching')
ax2.set_title('Template Matching Results')
ax2.legend()
plt.colorbar(im2, ax=ax2)

# Plot 3: Scikit-image results (if available)
ax3 = axes[1, 0]
im3 = ax3.imshow(signal_2d.data, cmap='viridis', origin='lower')
if skimage_available and len(coordinates) > 0:
    coord_x = [coord[1] for coord in coordinates]
    coord_y = [coord[0] for coord in coordinates]
    ax3.scatter(coord_x, coord_y, 
               c='orange', s=80, marker='s', alpha=0.7, label='Scikit-image')
ax3.set_title('Scikit-image Peak Detection')
ax3.legend()
plt.colorbar(im3, ax=ax3)

# Plot 4: Threshold-based results
ax4 = axes[1, 1]
im4 = ax4.imshow(signal_2d.data, cmap='viridis', origin='lower')
ax4.scatter(peak_positions[1], peak_positions[0], 
           c='magenta', s=80, marker='^', alpha=0.7, label='Threshold-based')
ax4.set_title('Threshold-based Detection')
ax4.legend()
plt.colorbar(im4, ax=ax4)

plt.tight_layout()

# %%
# Comparison and accuracy assessment

print("\\n=== Accuracy Assessment ===")

def calculate_detection_accuracy(true_peaks, detected_peaks, tolerance=5):
    """Calculate how many true peaks were correctly detected."""
    true_peaks = np.array(true_peaks)
    detected_peaks = np.array(detected_peaks).T if len(detected_peaks) == 2 else detected_peaks
    
    matches = 0
    for true_peak in true_peaks:
        distances = np.sqrt(np.sum((detected_peaks - true_peak)**2, axis=1))
        if np.min(distances) <= tolerance:
            matches += 1
    
    return matches, len(true_peaks)

# Assess each method
methods = [
    ("Template Matching", (peaks_template[0], peaks_template[1])),
    ("Threshold-based", (peak_positions[0], peak_positions[1]))
]

if skimage_available and len(coordinates) > 0:
    coord_y = np.array([coord[0] for coord in coordinates])
    coord_x = np.array([coord[1] for coord in coordinates])
    methods.append(("Scikit-image", (coord_y, coord_x)))

for method_name, peaks in methods:
    if len(peaks[0]) > 0:
        matches, total = calculate_detection_accuracy(peak_centers, peaks, tolerance=8)
        accuracy = matches / total * 100
        print(f"{method_name}: {matches}/{total} peaks detected correctly ({accuracy:.1f}%)")
    else:
        print(f"{method_name}: No peaks detected")

print("\\n=== Summary ===")
print("Different peak finding methods have various strengths:")
print("- Template matching: Good for peaks with known shape")
print("- Scikit-image: Robust with good parameter control") 
print("- Threshold-based: Simple and fast, good for clean data")
print("- Choose method based on your data characteristics and noise level")

# Display the plot
plt.show()
