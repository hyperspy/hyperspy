"""
Advanced Signal Processing with Map Method
==========================================

This example demonstrates the powerful capabilities of HyperSpy's `map` method
for applying functions across signal dimensions. The map method enables efficient
parallel processing and seamless integration with external scientific libraries.

"""

import numpy as np
import hyperspy.api as hs
from scipy import ndimage, optimize, signal as sp_signal
from skimage import filters, feature, measure, restoration
import matplotlib.pyplot as plt
import dask.array as da

# %%
# Creating Test Data for Advanced Processing
# ------------------------------------------

# Create a spectrum image with varying spectral features
print("Creating multidimensional test data...")
nav_shape = (8, 6)  # Navigation dimensions
sig_shape = (256, 256)  # Signal dimensions for 2D images
spectrum_length = 512  # For 1D spectral data

# Create 2D signal (images) with different features per navigation position
image_data = np.zeros(nav_shape + sig_shape)
for i in range(nav_shape[0]):
    for j in range(nav_shape[1]):
        x, y = np.meshgrid(np.linspace(-4, 4, sig_shape[0]), 
                          np.linspace(-4, 4, sig_shape[1]))
        
        # Create varying Gaussian features
        center_x = (i - nav_shape[0]/2) * 0.5
        center_y = (j - nav_shape[1]/2) * 0.5
        sigma = 0.5 + 0.3 * (i + j) / (nav_shape[0] + nav_shape[1])
        
        # Main feature
        image_data[i, j] = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
        
        # Add noise and secondary features
        image_data[i, j] += 0.1 * np.random.random(sig_shape)
        image_data[i, j] += 0.3 * np.exp(-((x + 1)**2 + (y - 1)**2) / 0.8)

s_images = hs.signals.Signal2D(image_data)
s_images.axes_manager.signal_axes[0].name = 'Y'
s_images.axes_manager.signal_axes[0].units = 'μm'
s_images.axes_manager.signal_axes[0].scale = 0.1
s_images.axes_manager.signal_axes[0].offset = -12.8
s_images.axes_manager.signal_axes[1].name = 'X'
s_images.axes_manager.signal_axes[1].units = 'μm'
s_images.axes_manager.signal_axes[1].scale = 0.1
s_images.axes_manager.signal_axes[1].offset = -12.8
s_images.axes_manager.navigation_axes[0].name = 'Position_Y'
s_images.axes_manager.navigation_axes[0].units = 'mm'
s_images.axes_manager.navigation_axes[0].scale = 0.5
s_images.axes_manager.navigation_axes[1].name = 'Position_X'
s_images.axes_manager.navigation_axes[1].units = 'mm'
s_images.axes_manager.navigation_axes[1].scale = 0.5
s_images.metadata.General.title = 'Multi-featured image stack'

print(f"Created image signal: {s_images}")

# Create spectrum image with varying spectral features
spectrum_data = np.zeros(nav_shape + (spectrum_length,))
energy_axis = np.linspace(100, 1000, spectrum_length)
for i in range(nav_shape[0]):
    for j in range(nav_shape[1]):
        # Multiple peaks with varying intensity and position
        peak1_pos = 200 + 50 * i / nav_shape[0]
        peak2_pos = 600 + 100 * j / nav_shape[1]
        peak1_int = 1.0 + 0.5 * j / nav_shape[1]
        peak2_int = 0.8 + 0.7 * i / nav_shape[0]
        
        spectrum_data[i, j] = (peak1_int * np.exp(-((energy_axis - peak1_pos) / 30)**2) +
                              peak2_int * np.exp(-((energy_axis - peak2_pos) / 40)**2) +
                              0.1 * np.random.random(spectrum_length))

s_spectra = hs.signals.Signal1D(spectrum_data)
s_spectra.axes_manager.signal_axes[0].name = 'Energy'
s_spectra.axes_manager.signal_axes[0].units = 'eV'
s_spectra.axes_manager.signal_axes[0].scale = 1.76
s_spectra.axes_manager.signal_axes[0].offset = 100
s_spectra.axes_manager.navigation_axes[0].name = 'Y'
s_spectra.axes_manager.navigation_axes[0].units = 'μm'
s_spectra.axes_manager.navigation_axes[0].scale = 2.0
s_spectra.axes_manager.navigation_axes[1].name = 'X'
s_spectra.axes_manager.navigation_axes[1].units = 'μm'
s_spectra.axes_manager.navigation_axes[1].scale = 2.0
s_spectra.metadata.General.title = 'Multi-peak spectrum image'

print(f"Created spectrum signal: {s_spectra}")

# %%
# Basic Map Operations with SciPy
# -------------------------------

# Apply different filters with constant parameters
print("\nApplying basic scipy filters...")

# Gaussian smoothing - preserves signal structure and metadata
s_smooth = s_images.map(ndimage.gaussian_filter, sigma=2.0, inplace=False)
s_smooth.metadata.General.title = 'Gaussian smoothed images'

# Edge detection using Sobel filter
s_edges = s_images.map(ndimage.sobel, inplace=False)
s_edges.metadata.General.title = 'Edge-detected images'

# Morphological operations
s_dilated = s_images.map(ndimage.binary_dilation, 
                        structure=np.ones((5, 5)), 
                        iterations=2, 
                        inplace=False)
s_dilated.metadata.General.title = 'Morphologically processed images'

print(f"Smoothed: {s_smooth}")
print(f"Edges: {s_edges}")
print(f"Dilated: {s_dilated}")

# %%
# Variable Parameters Across Navigation Dimensions
# ------------------------------------------------

print("\nApplying variable parameters across navigation...")

# Create parameter signals that vary across navigation dimensions
nav_size = nav_shape[0] * nav_shape[1]
sigma_values = np.linspace(0.5, 4.0, nav_size).reshape(nav_shape)
angle_values = np.linspace(0, 45, nav_size).reshape(nav_shape)

# Convert to HyperSpy signals for map method
sigma_signal = hs.signals.BaseSignal(sigma_values).T
angle_signal = hs.signals.BaseSignal(angle_values).T

# Apply variable Gaussian smoothing
s_var_smooth = s_images.map(ndimage.gaussian_filter, 
                           sigma=sigma_signal, 
                           inplace=False)
s_var_smooth.metadata.General.title = 'Variable sigma smoothing'

# Apply variable rotation (from scipy.ndimage)
s_rotated = s_images.map(ndimage.rotate, 
                        angle=angle_signal, 
                        reshape=False, 
                        order=1,
                        inplace=False)
s_rotated.metadata.General.title = 'Variable rotation'

print(f"Variable smoothing: {s_var_smooth}")
print(f"Variable rotation: {s_rotated}")

# %%
# Advanced Processing with Scikit-Image
# -------------------------------------

print("\nApplying advanced skimage processing...")

# Edge detection with different algorithms
s_canny = s_images.map(feature.canny, sigma=1.5, low_threshold=0.1, high_threshold=0.3, inplace=False)
s_canny.metadata.General.title = 'Canny edge detection'

# Local binary patterns for texture analysis
s_lbp = s_images.map(feature.local_binary_pattern, 
                    P=8, R=1, method='uniform', 
                    inplace=False)
s_lbp.metadata.General.title = 'Local binary patterns'

# Denoising with different algorithms
s_denoised = s_images.map(restoration.denoise_wavelet, 
                         sigma=0.1, 
                         convert2ycbcr=False,
                         inplace=False)
s_denoised.metadata.General.title = 'Wavelet denoised'

print(f"Canny edges: {s_canny}")
print(f"LBP texture: {s_lbp}")
print(f"Denoised: {s_denoised}")

# %%
# Custom Analysis Functions with Ragged Arrays
# --------------------------------------------

def analyze_image_features(image):
    """
    Custom function that extracts multiple features from each image.
    Returns different sized results, requiring ragged arrays.
    """
    # Threshold to find regions
    threshold = filters.threshold_otsu(image)
    binary = image > threshold
    
    # Label connected regions
    labeled = measure.label(binary)
    props = measure.regionprops(labeled, intensity_image=image)
    
    if len(props) == 0:
        return np.array([0, 0, 0, 0])  # No features found
    
    # Extract features for all regions
    features = []
    for prop in props:
        features.extend([
            prop.area,
            prop.eccentricity,
            prop.mean_intensity,
            prop.major_axis_length
        ])
    
    return np.array(features)

def fit_spectral_peaks(spectrum):
    """
    Custom function to fit Gaussian peaks to spectrum data.
    Returns peak positions and intensities.
    """
    # Find peaks using scipy
    peaks, properties = sp_signal.find_peaks(spectrum, height=0.2, distance=20)
    
    if len(peaks) == 0:
        return np.array([0, 0])  # No peaks found
    
    # Return peak positions and heights
    peak_data = []
    for peak in peaks:
        peak_data.extend([peak, spectrum[peak]])
    
    return np.array(peak_data)

print("\nApplying custom analysis functions...")

# Image feature extraction (ragged arrays due to variable number of features)
image_features = s_images.map(analyze_image_features, 
                             inplace=False, 
                             ragged=True)
image_features.metadata.General.title = 'Image feature analysis'

# Spectral peak fitting (ragged arrays due to variable number of peaks)
spectral_peaks = s_spectra.map(fit_spectral_peaks, 
                              inplace=False, 
                              ragged=True)
spectral_peaks.metadata.General.title = 'Spectral peak analysis'

print(f"Image features: {image_features}")
print(f"Spectral peaks: {spectral_peaks}")

# Examine results
print(f"\nSample image features at [0,0]: {image_features.data[0,0]}")
print(f"Sample spectral peaks at [0,0]: {spectral_peaks.data[0,0]}")

# %%
# Parallel Processing and Performance Optimization
# ------------------------------------------------

print("\nDemonstrating parallel processing capabilities...")

# Create larger dataset for performance testing
large_data = np.random.random((20, 15, 128, 128))
s_large = hs.signals.Signal2D(large_data)

import time

# Sequential processing
start_time = time.time()
result_seq = s_large.map(ndimage.gaussian_filter, 
                        sigma=2.0, 
                        num_workers=1, 
                        inplace=False)
seq_time = time.time() - start_time

# Parallel processing
start_time = time.time()
result_par = s_large.map(ndimage.gaussian_filter, 
                        sigma=2.0, 
                        num_workers=4, 
                        inplace=False)
par_time = time.time() - start_time

speedup = seq_time / par_time if par_time > 0 else float('inf')
print(f"Sequential processing: {seq_time:.3f} seconds")
print(f"Parallel processing (4 cores): {par_time:.3f} seconds")
print(f"Speedup: {speedup:.2f}x")

# %%
# Lazy Computation for Memory Efficiency
# --------------------------------------

print("\nDemonstrating lazy computation...")

# Create very large dataset that would consume too much memory if computed immediately
lazy_data = da.random.random((50, 40, 256, 256), chunks=(5, 5, 256, 256))
s_lazy = hs.signals.Signal2D(lazy_data)

print(f"Lazy signal: {s_lazy}")
print(f"Is lazy: {s_lazy._lazy}")

# Apply expensive operation lazily
s_lazy_processed = s_lazy.map(ndimage.gaussian_filter, 
                             sigma=3.0, 
                             lazy_output=True, 
                             inplace=False)

print(f"Lazy processed signal: {s_lazy_processed}")
print("Computation will be triggered when data is accessed or compute() is called")

# %%
# Integration with Optimization and Fitting
# -----------------------------------------

def fit_gaussian_2d(image):
    """
    Fit a 2D Gaussian to the image data.
    Returns fitted parameters: amplitude, x0, y0, sigma_x, sigma_y, theta
    """
    def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, theta):
        x, y = xy
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        return amplitude * np.exp(-(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2))
    
    # Create coordinate arrays
    y_idx, x_idx = np.indices(image.shape)
    
    # Initial guess
    amplitude_guess = image.max()
    x0_guess, y0_guess = np.unravel_index(image.argmax(), image.shape)
    
    try:
        # Fit the function
        popt, _ = optimize.curve_fit(
            gaussian_2d, 
            (x_idx.ravel(), y_idx.ravel()), 
            image.ravel(),
            p0=[amplitude_guess, x0_guess, y0_guess, 20, 20, 0],
            maxfev=1000
        )
        return popt
    except:
        # Return default values if fitting fails
        return np.array([0, 0, 0, 0, 0, 0])

print("\nApplying Gaussian fitting to each image...")

# Apply Gaussian fitting (this might take a moment)
fitted_params = s_images.inav[:3, :3].map(fit_gaussian_2d, 
                                         inplace=False, 
                                         ragged=False)  # All return same size
fitted_params.metadata.General.title = '2D Gaussian fit parameters'

print(f"Fitted parameters: {fitted_params}")
print(f"Parameter shape: {fitted_params.data.shape}")
print(f"Sample fitted parameters at [0,0]: {fitted_params.data[0,0]}")

# %%
# Best Practices and Tips
# ----------------------

print("\nMap method best practices demonstrated:")
print("✓ Use inplace=False to preserve original data")
print("✓ Set ragged=True when output sizes vary")
print("✓ Use variable parameters with BaseSignal objects")
print("✓ Leverage num_workers for parallel processing")
print("✓ Use lazy_output=True for memory efficiency")
print("✓ Combine with external libraries (scipy, skimage, etc.)")
print("✓ Custom functions can return complex analysis results")
print("✓ Metadata and axis information are preserved")

print("\nAdvanced map method examples completed successfully!")
