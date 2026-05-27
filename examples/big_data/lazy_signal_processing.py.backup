"""
Lazy Signal Processing for Big Data
====================================

This example demonstrates how to handle large datasets using HyperSpy's 
lazy evaluation system, which allows processing of data that doesn't fit 
in memory through chunked operations and delayed computation.

"""

import numpy as np
import hyperspy.api as hs
import dask.array as da

print("Big Data Processing with HyperSpy Lazy Signals")
print("=" * 50)

# Create a large synthetic dataset that would typically not fit in memory
# We'll simulate this by creating a dask array directly
print("\n1. Creating large synthetic dataset...")

# Simulate a large 4D dataset: (100, 100, 256, 256) 
# This would be ~25 GB in memory as float64!
print("Simulating large 4D dataset: (100, 100, 256, 256)")
print("Estimated memory requirement: ~25 GB")

# Create dask array with chunks
chunk_size = (10, 10, 256, 256)  # Process in smaller chunks
print(f"Using chunks of size: {chunk_size}")

# Generate synthetic data using dask
def generate_synthetic_chunk(chunk_shape, offset):
    """Generate a synthetic data chunk with spatial and spectral structure."""
    nav_i, nav_j, sig_i, sig_j = chunk_shape
    offset_i, offset_j, _, _ = offset
    
    # Create synthetic spectrum image with realistic features
    data = np.zeros(chunk_shape)
    
    for i in range(nav_i):
        for j in range(nav_j):
            # Global coordinates
            global_i = offset_i + i
            global_j = offset_j + j
            
            # Create spatial-dependent spectral features
            # Feature 1: Gaussian peak that varies with position
            x_center = 128 + 50 * np.sin(global_i * 0.1)
            y_center = 128 + 50 * np.cos(global_j * 0.1)
            
            # Create 2D Gaussian in signal space
            sig_x, sig_y = np.meshgrid(np.arange(sig_i), np.arange(sig_j), indexing='ij')
            gaussian = np.exp(-((sig_x - x_center)**2 + (sig_y - y_center)**2) / 1000)
            
            # Add some background and noise
            background = 0.1 * np.ones((sig_i, sig_j))
            noise = 0.05 * np.random.random((sig_i, sig_j))
            
            data[i, j, :, :] = gaussian + background + noise
    
    return data

# Create chunks manually for demonstration
print("\n2. Creating lazy signal from chunks...")

# For this example, create a smaller but realistic dataset
nav_shape = (20, 20)  # Navigation dimensions
sig_shape = (64, 64)   # Signal dimensions (image)
total_shape = nav_shape + sig_shape

# Create dask array
chunk_nav = (5, 5)
chunk_sig = (64, 64)
chunks = chunk_nav + chunk_sig

# Generate the dask array with simpler approach
lazy_data = da.random.random(total_shape, chunks=chunks)

# Add some realistic structure using simpler operations
print("Adding spatial-spectral structure...")

# Create a simple structured pattern
# Navigation-dependent signal variation
nav_coords = np.mgrid[0:nav_shape[0], 0:nav_shape[1]]
sig_coords = np.mgrid[0:sig_shape[0], 0:sig_shape[1]]

# Create a simple but realistic pattern
# Use map_blocks for applying functions to dask arrays
def add_structure(block, block_id):
    """Add realistic structure to each block."""
    nav_i_start = block_id[0] * chunks[0]
    nav_j_start = block_id[1] * chunks[1]
    
    # Add a spatial-spectral correlation
    for i in range(block.shape[0]):
        for j in range(block.shape[1]):
            # Global coordinates
            global_i = nav_i_start + i
            global_j = nav_j_start + j
            
            # Add a pattern that varies across navigation space
            center_i = sig_shape[0] // 2 + int(10 * np.sin(global_i * 0.3))
            center_j = sig_shape[1] // 2 + int(10 * np.cos(global_j * 0.3))
            
            # Create a simple 2D Gaussian peak
            for si in range(block.shape[2]):
                for sj in range(block.shape[3]):
                    distance_sq = (si - center_i)**2 + (sj - center_j)**2
                    if distance_sq < 100:  # Only modify pixels near center
                        block[i, j, si, sj] += 2.0 * np.exp(-distance_sq / 50)
    
    return block

# Apply structure to the random data
structured_data = da.map_blocks(add_structure, lazy_data, dtype=lazy_data.dtype)

print(f"Created dask array with shape: {structured_data.shape}")
print(f"Chunk sizes: {structured_data.chunks}")

# Create lazy HyperSpy signal
print("\n3. Creating lazy HyperSpy signal...")

s_lazy = hs.signals.Signal2D(structured_data, lazy=True)

# Set axes properties  
s_lazy.axes_manager.navigation_axes[0].name = 'scan_x'
s_lazy.axes_manager.navigation_axes[0].units = 'μm'
s_lazy.axes_manager.navigation_axes[0].scale = 0.5

s_lazy.axes_manager.navigation_axes[1].name = 'scan_y'
s_lazy.axes_manager.navigation_axes[1].units = 'μm'
s_lazy.axes_manager.navigation_axes[1].scale = 0.5

s_lazy.axes_manager.signal_axes[0].name = 'detector_x'
s_lazy.axes_manager.signal_axes[0].units = 'pixels'

s_lazy.axes_manager.signal_axes[1].name = 'detector_y'
s_lazy.axes_manager.signal_axes[1].units = 'pixels'

s_lazy.metadata.General.title = "Large Lazy Signal"

print(f"Lazy signal created: {s_lazy}")
print(f"Is lazy: {s_lazy._lazy}")

# Demonstrate lazy operations
print("\n4. Performing lazy operations...")

# These operations don't compute immediately - they build a computation graph
print("Creating computation graph for processing...")

# Lazy cropping
print("- Lazy cropping signal dimensions...")
s_cropped = s_lazy.isig[16:48, 16:48]  # Crop signal to center region
print(f"  Cropped signal shape: {s_cropped.data.shape}")

# Lazy mathematical operations
print("- Lazy mathematical operations...")
s_normalized = (s_cropped - s_cropped.min()) / (s_cropped.max() - s_cropped.min())
print(f"  Normalized signal ready for computation")

# Lazy aggregation operations
print("- Lazy aggregation operations...")
s_mean = s_normalized.mean(axis=(0, 1))  # Mean over navigation dimensions
s_sum = s_normalized.sum(axis=(2, 3))    # Sum over signal dimensions

print(f"  Mean signal shape: {s_mean.data.shape}")
print(f"  Sum signal shape: {s_sum.data.shape}")

# Lazy filtering (if we had a 1D signal)
print("- Creating 1D lazy signal for filtering demonstration...")
# Create 1D spectrum image from our 2D data by taking line profiles
s_1d_lazy = s_lazy.mean(axis=3)  # Average over one signal dimension
s_1d_lazy = hs.signals.Signal1D(s_1d_lazy.data, lazy=True)

# Set 1D axes
s_1d_lazy.axes_manager.navigation_axes[0].name = 'scan_x'
s_1d_lazy.axes_manager.navigation_axes[0].units = 'μm'
s_1d_lazy.axes_manager.navigation_axes[0].scale = 0.5

s_1d_lazy.axes_manager.navigation_axes[1].name = 'scan_y'
s_1d_lazy.axes_manager.navigation_axes[1].units = 'μm'
s_1d_lazy.axes_manager.navigation_axes[1].scale = 0.5

s_1d_lazy.axes_manager.signal_axes[0].name = 'detector_x'
s_1d_lazy.axes_manager.signal_axes[0].units = 'pixels'

print(f"  1D lazy signal shape: {s_1d_lazy.data.shape}")

# Apply lazy smoothing
s_smoothed = s_1d_lazy.smooth_savitzky_golay(polynomial_order=2, window_length=5)
print(f"  Smoothed signal ready for computation")

# Demonstrate memory-efficient computation
print("\n5. Computing results in chunks...")

# Compute small result first
print("Computing mean spectrum (small result)...")
mean_result = s_mean  # This is already computed for lazy signals
if hasattr(s_mean, 'compute'):
    mean_result = s_mean.compute()
print(f"Mean spectrum computed: shape {mean_result.data.shape}")

# Compute navigation-space result
print("Computing sum image (navigation space)...")
sum_result = s_sum  # This is already computed for lazy signals  
if hasattr(s_sum, 'compute'):
    sum_result = s_sum.compute()
print(f"Sum image computed: shape {sum_result.data.shape}")

# Save results without loading full dataset into memory
print("\n6. Saving results efficiently...")

# Save lazy signal directly - HyperSpy handles chunked saving
output_file = "lazy_processed_data.hspy"
print(f"Saving lazy signal to {output_file}...")
s_normalized.save(output_file, overwrite=True)
print("Saved successfully using chunked writing!")

# Clean up
import os
if os.path.exists(output_file):
    os.remove(output_file)
    print("Cleaned up temporary file")

# Demonstrate rechunking for different operations
print("\n7. Rechunking strategies...")

print("Original chunks (good for navigation operations):")
print(f"  {s_lazy.data.chunks}")

# Rechunk for signal-space operations
print("Rechunking for signal-space operations...")
if hasattr(s_lazy.data, 'rechunk'):
    rechunked_data = s_lazy.data.rechunk(chunks=(2, 2, 32, 32))
    s_rechunked = hs.signals.Signal2D(rechunked_data, lazy=True)
    print(f"New chunks: {s_rechunked.data.chunks}")
else:
    print("Signal is not lazy - rechunking not applicable")
    s_rechunked = s_lazy

# Show memory usage comparison
print("\n8. Memory usage comparison...")

# Estimate memory usage
def estimate_memory_usage(signal):
    """Estimate memory usage of a signal in MB."""
    total_elements = np.prod(signal.data.shape)
    bytes_per_element = 8  # float64
    total_bytes = total_elements * bytes_per_element
    return total_bytes / (1024**2)  # Convert to MB

original_memory = estimate_memory_usage(s_lazy)
if hasattr(s_lazy.data, 'chunksize'):
    chunk_memory = np.prod(s_lazy.data.chunksize) * 8 / (1024**2)
else:
    # Estimate chunk size from the chunks
    first_chunk_size = tuple(c[0] for c in s_lazy.data.chunks) if hasattr(s_lazy.data, 'chunks') else s_lazy.data.shape
    chunk_memory = np.prod(first_chunk_size) * 8 / (1024**2)

print(f"Full dataset memory requirement: {original_memory:.1f} MB")
print(f"Single chunk memory requirement: {chunk_memory:.1f} MB")
print(f"Memory reduction factor: {original_memory/chunk_memory:.1f}x")

# Demonstrate advanced lazy operations
print("\n9. Advanced lazy operations...")

# Lazy decomposition (would work on much larger datasets)
print("Setting up lazy decomposition (PCA)...")
print("Note: For demonstration only - decomposition on small dataset")

# For very large datasets, you might want to:
# 1. Sample the data first for initial decomposition
# 2. Apply the learned components to the full dataset

# Sample every 2nd pixel in navigation for initial decomposition
s_sampled = s_1d_lazy.inav[::2, ::2]
print(f"Sampled signal shape: {s_sampled.data.shape}")

# This would work for larger datasets too
print("Decomposition could be applied to sampled data first...")
print("Then transform the full lazy dataset using the learned components")

print("\n10. Best practices summary...")
print("✓ Use appropriate chunk sizes for your operations")
print("✓ Chain operations before computing to optimize computation graph")
print("✓ Consider rechunking between different types of operations")
print("✓ Use sampling for initial analysis of very large datasets")
print("✓ Save intermediate results to avoid recomputation")
print("✓ Monitor memory usage and adjust chunk sizes accordingly")

print(f"\nLazy signal processing example completed!")
print("Key benefits:")
print("- Process datasets larger than available RAM")
print("- Automatic parallelization across chunks") 
print("- Memory-efficient computation through chunking")
print("- Optimized computation graphs for chained operations")
