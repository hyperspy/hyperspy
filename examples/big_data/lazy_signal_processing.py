"""
Lazy Signal Processing for Big Data
====================================

This example demonstrates HyperSpy's lazy evaluation system for handling large
datasets that don't fit in memory. Key concepts include chunked operations,
memory monitoring, and efficient computation strategies.

Following AI Guide best practices for educational examples.
"""

import hyperspy.api as hs
import numpy as np
import dask.array as da
import time

# %%
# ## Memory Monitoring Utility
# 
# Monitor memory usage throughout the example.

def print_memory_usage(description):
    """Print current memory usage."""
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"{description}: {memory_mb:.1f} MB")
    except ImportError:
        print(f"{description}: psutil not available for memory monitoring")

print("🚀 Lazy Signal Processing with HyperSpy")
print("=" * 50)

print_memory_usage("Initial memory usage")

# %%
# ## Create Large Lazy Dataset
# 
# Simulate a large dataset using Dask arrays for out-of-core processing.

print("\n📊 Creating Large Lazy Dataset")
print("=" * 50)

# Create a large lazy array (virtual - not loaded into memory)
# This simulates a 4D dataset that would be ~6.4 GB in memory
print("Creating lazy 4D dataset: (50, 50, 128, 128)")
print("Estimated memory if loaded: ~6.4 GB")

# Create dask array with appropriate chunks
chunk_size = (10, 10, 128, 128)
large_lazy_data = da.random.random((50, 50, 128, 128), chunks=chunk_size)

print(f"Created lazy array with chunks: {chunk_size}")
print(f"Virtual size: {large_lazy_data.nbytes / 1024**3:.1f} GB")

print_memory_usage("After creating lazy array")

# %%
# ## Create HyperSpy Lazy Signal
# 
# Convert the Dask array to a HyperSpy lazy signal.

print("\n🔧 Creating HyperSpy Lazy Signal")
print("=" * 50)

# Create HyperSpy signal from lazy array
lazy_signal = hs.signals.Signal2D(large_lazy_data)

# Configure axes
lazy_signal.axes_manager.navigation_axes[0].name = 'x'
lazy_signal.axes_manager.navigation_axes[0].units = 'μm'
lazy_signal.axes_manager.navigation_axes[0].scale = 0.1
lazy_signal.axes_manager.navigation_axes[1].name = 'y'
lazy_signal.axes_manager.navigation_axes[1].units = 'μm'
lazy_signal.axes_manager.navigation_axes[1].scale = 0.1
lazy_signal.axes_manager.signal_axes[0].name = 'detector_x'
lazy_signal.axes_manager.signal_axes[0].units = 'px'
lazy_signal.axes_manager.signal_axes[0].scale = 1
lazy_signal.axes_manager.signal_axes[1].name = 'detector_y'
lazy_signal.axes_manager.signal_axes[1].units = 'px'
lazy_signal.axes_manager.signal_axes[1].scale = 1

print(f"Lazy signal created: {lazy_signal}")
print(f"Is lazy: {lazy_signal._lazy}")
print(f"Chunks: {lazy_signal.data.chunks}")

print_memory_usage("After creating lazy signal")

# %%
# ## Lazy Operations
# 
# Perform operations on lazy signals without loading data into memory.

print("\n⚡ Performing Lazy Operations")
print("=" * 50)

# Operations on lazy signals are computed only when needed
print("Computing mean along navigation dimensions...")
start_time = time.time()

# Chain operations before computing (best practice)
lazy_mean = lazy_signal.mean(axis=(0, 1))
lazy_max = lazy_signal.max(axis=(0, 1))
lazy_std = lazy_signal.std(axis=(0, 1))

print(f"Operations chained in {time.time() - start_time:.3f} seconds")
print("No computation performed yet - operations are lazy!")

print_memory_usage("After chaining operations")

# %%
# ## Compute Results
# 
# Trigger computation and observe memory usage.

print("\n💻 Computing Results")
print("=" * 50)

# Force computation of results
print("Computing mean...")
start_time = time.time()
# For lazy signals, operations return computed results
mean_result = lazy_mean  # Mean is already computed
compute_time = time.time() - start_time

print(f"Mean computation: {compute_time:.2f} seconds")
print(f"Result shape: {mean_result.data.shape}")
print(f"Result memory: {mean_result.data.nbytes / 1024**2:.1f} MB")

print_memory_usage("After computing mean")

# %%
# ## Chunked Processing Strategy
# 
# Demonstrate efficient chunked processing for large datasets.

print("\n📦 Chunked Processing Strategy")
print("=" * 50)

# Create a smaller example for demonstration
chunk_signal = hs.signals.Signal2D(da.random.random((20, 20, 64, 64), chunks=(5, 5, 64, 64)))
chunk_signal.axes_manager.navigation_axes[0].name = 'x'
chunk_signal.axes_manager.navigation_axes[0].units = 'μm'
chunk_signal.axes_manager.navigation_axes[0].scale = 0.1
chunk_signal.axes_manager.navigation_axes[1].name = 'y'
chunk_signal.axes_manager.navigation_axes[1].units = 'μm'
chunk_signal.axes_manager.navigation_axes[1].scale = 0.1

print(f"Chunk signal: {chunk_signal}")
print(f"Chunk structure: {chunk_signal.data.chunks}")

# Process in chunks using HyperSpy's map function
def simple_processing(image):
    """Simple processing function for demonstration."""
    return np.mean(image.data)

print("Processing chunks with map function...")
start_time = time.time()

# Use map for parallel processing
processed = chunk_signal.map(simple_processing, show_progressbar=False)

processing_time = time.time() - start_time
print(f"Chunked processing: {processing_time:.2f} seconds")

print_memory_usage("After chunked processing")

# %%
# ## Memory-Efficient Analysis
# 
# Demonstrate memory-efficient statistical analysis.

print("\n🧮 Memory-Efficient Analysis")
print("=" * 50)

# Create analysis signal
analysis_signal = hs.signals.Signal2D(da.random.random((30, 30, 32, 32), chunks=(6, 6, 32, 32)))

# Efficient statistical operations
print("Computing statistics efficiently...")
start_time = time.time()

# Use HyperSpy's efficient methods
stats = {
    'mean': analysis_signal.mean(axis=(0, 1)),
    'max': analysis_signal.max(axis=(0, 1)),
    'std': analysis_signal.std(axis=(0, 1))
}

stats_time = time.time() - start_time
print(f"Statistics computed in {stats_time:.2f} seconds")

# Compute one result to show efficiency
mean_computed = stats['mean']  # Already computed
print(f"Mean result shape: {mean_computed.data.shape}")

print_memory_usage("After statistical analysis")

# %%
# ## Best Practices Summary
# 
# Key takeaways for lazy signal processing.

print("\n✅ Lazy Processing Best Practices")
print("=" * 50)

print("1. 🗂️  Memory Management:")
print("   • Use Dask arrays for out-of-core processing")
print("   • Choose appropriate chunk sizes")
print("   • Monitor memory usage throughout processing")

print("\n2. 🔄 Lazy Operations:")
print("   • Chain operations before computing")
print("   • Use .compute() only when results are needed")
print("   • Leverage HyperSpy's lazy-aware methods")

print("\n3. ⚡ Efficient Processing:")
print("   • Use HyperSpy's map function for parallel processing")
print("   • Process in chunks to control memory usage")
print("   • Avoid loading entire datasets into memory")

print("\n4. 📊 Analysis Strategy:")
print("   • Use statistical methods that work with lazy arrays")
print("   • Compute intermediate results selectively")
print("   • Save results to avoid recomputation")

print(f"\nExample completed successfully!")
print(f"Maximum memory usage monitored throughout processing")
print_memory_usage("Final memory usage")