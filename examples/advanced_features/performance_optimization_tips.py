"""
Performance Optimization Tips
=============================

This example demonstrates techniques for optimizing HyperSpy performance when
working with large datasets, including memory management, lazy evaluation,
parallel processing, and efficient computational strategies.
"""

import hyperspy.api as hs
import numpy as np
import time

# %%
# ## Memory Management and Data Types
# 
# Optimize memory usage through appropriate data types and memory-conscious operations

print("Demonstrating memory optimization techniques...")

# Create test dataset
large_data = np.random.random((100, 100, 512)).astype(np.float64)
print(f"Original data size: {large_data.nbytes / 1024**2:.1f} MB (float64)")

# %%
# ### Data Type Optimization

# Convert to appropriate precision for analysis
signal_float64 = hs.signals.Signal1D(large_data)
print(f"Float64 signal memory: {signal_float64.data.nbytes / 1024**2:.1f} MB")

# Use float32 when full precision isn't needed (saves 50% memory)
signal_float32 = hs.signals.Signal1D(large_data.astype(np.float32))
print(f"Float32 signal memory: {signal_float32.data.nbytes / 1024**2:.1f} MB")

# For integer data, use appropriate bit depth
integer_data = (large_data * 1000).astype(np.uint16)
signal_uint16 = hs.signals.Signal1D(integer_data)
print(f"Uint16 signal memory: {signal_uint16.data.nbytes / 1024**2:.1f} MB")

# Memory savings summary
float64_size = signal_float64.data.nbytes / 1024**2
float32_size = signal_float32.data.nbytes / 1024**2
uint16_size = signal_uint16.data.nbytes / 1024**2

print(f"\nMemory optimization results:")
print(f"Float32 saves {((float64_size - float32_size) / float64_size * 100):.1f}% vs float64")
print(f"Uint16 saves {((float64_size - uint16_size) / float64_size * 100):.1f}% vs float64")

# %%
# ## View vs Copy Operations
# 
# Understand which operations create memory copies vs views

print("\nAnalyzing memory copy vs view operations...")

# Operations that typically create views (no memory copy)
view_transpose = signal_float32.T
view_slice = signal_float32.inav[10:90, 20:80]

# Check if operations share memory
shares_memory_transpose = np.shares_memory(signal_float32.data, view_transpose.data)
shares_memory_slice = np.shares_memory(signal_float32.data, view_slice.data)

print(f"Transpose shares memory: {shares_memory_transpose}")
print(f"Slice shares memory: {shares_memory_slice}")

# Operations that may require copying
try:
    # Complex transpose may require copy
    complex_transpose = signal_float32.transpose(signal_axes=[0, 1])
    shares_memory_complex = np.shares_memory(signal_float32.data, complex_transpose.data)
    print(f"Complex transpose shares memory: {shares_memory_complex}")
except:
    print("Complex transpose requires copy")

# Deep copy always creates new memory
deep_copy = signal_float32.deepcopy()
shares_memory_deepcopy = np.shares_memory(signal_float32.data, deep_copy.data)
print(f"Deep copy shares memory: {shares_memory_deepcopy}")

# %%
# ## Lazy Evaluation with Dask
# 
# Use Dask for out-of-core computation with large datasets

print("\nDemonstrating lazy evaluation techniques...")

try:
    import dask.array as da
    
    # Create large dask array that doesn't fit in memory
    print("Creating lazy Dask array...")
    large_dask_array = da.random.random((1000, 1000, 1000), chunks=(100, 100, 100))
    print(f"Dask array size: {large_dask_array.nbytes / 1024**3:.1f} GB (virtual)")
    
    # Create HyperSpy signal with lazy evaluation
    lazy_signal = hs.signals.Signal1D(large_dask_array)
    print(f"Lazy signal created: {lazy_signal}")
    
    # Operations on lazy signals are computed only when needed
    lazy_mean = lazy_signal.mean(axis=(0, 1))
    print(f"Lazy mean computed: {lazy_mean}")
    
    # Force computation by accessing the data
    print("Computing result...")
    start_time = time.time()
    result_data = lazy_mean.data  # This will trigger computation for lazy signals
    compute_time = time.time() - start_time
    print(f"Computation completed in {compute_time:.2f} seconds")
    
except ImportError:
    print("Dask not available - install with: pip install dask")

# %%
# ## Efficient Indexing Strategies
# 
# Optimize data access patterns for better performance

print("\nOptimizing data access patterns...")

# Create test signal for indexing optimization
test_data = np.random.random((200, 200, 1000))
test_signal = hs.signals.Signal1D(test_data)
test_signal.axes_manager[0].name = 'y'
test_signal.axes_manager[0].units = 'nm'
test_signal.axes_manager[0].scale = 0.1

test_signal.axes_manager[1].name = 'x'
test_signal.axes_manager[1].units = 'nm'
test_signal.axes_manager[1].scale = 0.1

test_signal.axes_manager[2].name = 'energy'
test_signal.axes_manager[2].units = 'eV'
test_signal.axes_manager[2].scale = 0.1
test_signal.axes_manager[2].offset = 100

# %%
# ### Efficient vs Inefficient Access Patterns

# EFFICIENT: Process chunks that are contiguous in memory
print("Testing efficient chunk processing...")
start_time = time.time()

# Process spatial chunks (navigation dimensions are contiguous)
chunk_results = []
for y_start in range(0, 200, 50):
    for x_start in range(0, 200, 50):
        chunk = test_signal.inav[y_start:y_start+50, x_start:x_start+50]
        chunk_mean = chunk.mean(axis=(0, 1))
        chunk_results.append(chunk_mean.data.mean())

efficient_time = time.time() - start_time
print(f"Efficient chunking: {efficient_time:.3f} seconds")

# INEFFICIENT: Random access pattern
print("Testing inefficient random access...")
start_time = time.time()

random_results = []
np.random.seed(42)  # For reproducible results
random_indices = np.random.randint(0, 200, size=(100, 2))

for y, x in random_indices:
    spectrum = test_signal.inav[y, x]
    random_results.append(spectrum.data.mean())

inefficient_time = time.time() - start_time
print(f"Inefficient random access: {inefficient_time:.3f} seconds")
print(f"Efficiency improvement: {inefficient_time/efficient_time:.1f}x faster")

# %%
# ## Parallel Processing with map()
# 
# Use HyperSpy's map function for parallel processing

print("\nDemonstrating parallel processing...")

def compute_peak_position(spectrum):
    """Find peak position in spectrum"""
    return np.argmax(spectrum.data)

def compute_peak_statistics(spectrum):
    """Compute multiple statistics efficiently"""
    data = spectrum.data
    return {
        'max_position': np.argmax(data),
        'max_value': np.max(data),
        'mean_value': np.mean(data),
        'std_value': np.std(data)
    }

# Create smaller test signal for demonstration
small_signal = test_signal.inav[::20, ::20]  # Subsample for demo
print(f"Processing signal with shape: {small_signal}")

# %%
# ### Serial vs Parallel Processing Comparison

# Serial processing
print("Serial processing...")
start_time = time.time()
peak_positions_serial = []
for i in range(small_signal.axes_manager.navigation_size):
    spectrum = small_signal.inav[np.unravel_index(i, small_signal.axes_manager.navigation_shape)]
    peak_positions_serial.append(compute_peak_position(spectrum))
serial_time = time.time() - start_time

# Parallel processing using map
print("Parallel processing...")
start_time = time.time()
peak_positions_parallel = small_signal.map(compute_peak_position, 
                                          show_progressbar=False,
                                          inplace=False)
parallel_time = time.time() - start_time

print(f"Serial processing: {serial_time:.3f} seconds")
print(f"Parallel processing: {parallel_time:.3f} seconds")
if serial_time > parallel_time:
    print(f"Parallel speedup: {serial_time/parallel_time:.1f}x faster")

# %%
# ## Memory-Efficient Statistical Operations
# 
# Compute statistics without loading entire dataset into memory

print("\nMemory-efficient statistical operations...")

# For very large signals, compute statistics in chunks
def chunked_statistics(signal, chunk_size=50):
    """Compute statistics in memory-efficient chunks"""
    nav_shape = signal.axes_manager.navigation_shape
    results = {
        'mean_values': [],
        'max_values': [],
        'std_values': []
    }
    
    # Process in chunks
    for y_start in range(0, nav_shape[0], chunk_size):
        for x_start in range(0, nav_shape[1], chunk_size):
            y_end = min(y_start + chunk_size, nav_shape[0])
            x_end = min(x_start + chunk_size, nav_shape[1])
            
            # Extract chunk
            chunk = signal.inav[y_start:y_end, x_start:x_end]
            
            # Compute statistics for chunk
            chunk_mean = chunk.mean(axis=(0, 1))
            chunk_max = chunk.max(axis=(0, 1))
            chunk_std = chunk.std(axis=(0, 1))
            
            results['mean_values'].append(chunk_mean.data.mean())
            results['max_values'].append(chunk_max.data.max())
            results['std_values'].append(chunk_std.data.mean())
    
    return results

# Demonstrate chunked processing
chunk_stats = chunked_statistics(test_signal, chunk_size=50)
print(f"Processed {len(chunk_stats['mean_values'])} chunks")
print(f"Overall mean: {np.mean(chunk_stats['mean_values']):.3f}")
print(f"Overall max: {np.max(chunk_stats['max_values']):.3f}")

# %%
# ## Optimization for Different Analysis Types
# 
# Specific optimization strategies for common analysis workflows

print("\nAnalysis-specific optimizations...")

# %%
# ### Spectrum Image Analysis Optimization

# Efficient axis configuration for spectrum images
spectrum_signal = test_signal.copy()

# Pre-configure axes for efficient access
nav_axes = spectrum_signal.axes_manager.navigation_axes
sig_axis = spectrum_signal.axes_manager.signal_axes[0]

# Batch configure navigation axes
for ax in nav_axes:
    ax.units = 'nm'
    ax.scale = 0.1

print("Optimized spectrum image analysis:")
print("- Batch axis configuration")
print("- Efficient navigation axis access")
print("- Signal axis optimization")

# %%
# ### Image Stack Analysis Optimization

# Convert to image stack for spatial analysis
image_stack = test_signal.as_signal2D(('y', 'x'))
print(f"Image stack shape: {image_stack}")

# Efficient spatial operations
spatial_mean = image_stack.mean(axis=('y', 'x'))
spatial_std = image_stack.std(axis=('y', 'x'))

print("Optimized image stack analysis:")
print("- Appropriate signal type conversion")
print("- Spatial dimension optimization")
print("- Statistical operations across space")

# %%
# ## Performance Monitoring and Profiling
# 
# Tools and techniques for monitoring HyperSpy performance

print("\nPerformance monitoring techniques...")

# Memory usage monitoring
def print_memory_usage(signal, description="Signal"):
    """Print memory usage information"""
    size_mb = signal.data.nbytes / 1024**2
    print(f"{description} memory usage: {size_mb:.1f} MB")
    print(f"Data type: {signal.data.dtype}")
    print(f"Shape: {signal.data.shape}")
    print(f"Owns data: {signal.data.flags.owndata}")

print_memory_usage(test_signal, "Test signal")
print_memory_usage(signal_float32, "Float32 signal")

# %%
# ## Performance Best Practices Summary

print("\nPerformance Optimization Summary:")
print("=================================")

print("\n1. Memory Management:")
print("   ✅ Use appropriate data types (float32 vs float64)")
print("   ✅ Understand view vs copy operations")
print("   ✅ Use change_dtype() for type conversion")
print("   ✅ Monitor memory usage with .nbytes")

print("\n2. Lazy Evaluation:")
print("   ✅ Use Dask arrays for large datasets")
print("   ✅ Compute only when results are needed")
print("   ✅ Chain operations before computing")

print("\n3. Efficient Access Patterns:")
print("   ✅ Process contiguous chunks")
print("   ✅ Avoid random access patterns")
print("   ✅ Use batch operations over loops")
print("   ✅ Configure axes efficiently with .set()")

print("\n4. Parallel Processing:")
print("   ✅ Use .map() with parallel=True")
print("   ✅ Design functions for parallel execution")
print("   ✅ Balance chunk size vs overhead")

print("\n5. Analysis-Specific Optimization:")
print("   ✅ Choose appropriate signal types")
print("   ✅ Use axis-aware operations")
print("   ✅ Pre-configure metadata and axes")
print("   ✅ Monitor performance with timing")

print("\n6. Memory-Efficient Strategies:")
print("   ✅ Process data in chunks")
print("   ✅ Use streaming operations")
print("   ✅ Clear intermediate results")
print("   ✅ Use views instead of copies when possible")

print(f"\nOptimization demonstration completed with test signal: {test_signal}")
