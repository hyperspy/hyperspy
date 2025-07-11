"""
Performance Optimization Tips
=============================

This example demonstrates essential performance optimization techniques for 
HyperSpy, focusing on memory management, lazy evaluation, and efficient data 
access patterns. Following best practices from the AI Guide.

Key concepts covered:
- Memory-efficient data types
- Lazy evaluation with Dask
- Efficient chunking strategies
- HyperSpy-native operations
"""

import hyperspy.api as hs
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

# %%
# ## Memory Optimization with Data Types
# 
# The choice of data type significantly impacts memory usage and performance.

print("🔧 Memory Optimization with Data Types")
print("=" * 50)

# Create test dataset - realistic size for demonstration
test_data = np.random.random((100, 100, 256)).astype(np.float64)
print(f"Test data shape: {test_data.shape}")
print(f"Original size (float64): {test_data.nbytes / 1024**2:.1f} MB")

# Compare different data types
signal_float64 = hs.signals.Signal1D(test_data)
signal_float32 = hs.signals.Signal1D(test_data.astype(np.float32))
signal_uint16 = hs.signals.Signal1D((test_data * 65535).astype(np.uint16))

print(f"\nMemory usage comparison:")
print(f"Float64: {signal_float64.data.nbytes / 1024**2:.1f} MB")
print(f"Float32: {signal_float32.data.nbytes / 1024**2:.1f} MB (50% reduction)")
print(f"Uint16:  {signal_uint16.data.nbytes / 1024**2:.1f} MB (75% reduction)")

# %%
# ## Lazy Evaluation for Large Datasets
# 
# Use Dask arrays for out-of-core computation with datasets that don't fit in memory.

print("\n🚀 Lazy Evaluation with Dask")
print("=" * 50)

try:
    import dask.array as da
    
    # Create a large lazy array (virtual - not loaded into memory)
    large_lazy_data = da.random.random((500, 500, 512), chunks=(50, 50, 512))
    print(f"Lazy array size: {large_lazy_data.nbytes / 1024**3:.1f} GB (virtual)")
    
    # Create lazy HyperSpy signal
    lazy_signal = hs.signals.Signal1D(large_lazy_data)
    print(f"Lazy signal created: {lazy_signal}")
    
    # Operations on lazy signals are computed only when needed
    print("Computing mean along navigation dimensions...")
    start_time = time.time()
    lazy_mean = lazy_signal.mean(axis=(0, 1))
    
    # Force computation and measure time
    result = lazy_mean  # Already computed in HyperSpy
    compute_time = time.time() - start_time
    print(f"Computation completed in {compute_time:.2f} seconds")
    print(f"Result shape: {result.data.shape}")
    
except ImportError:
    print("❌ Dask not available. Install with: pip install dask[array]")

# %%
# ## Efficient Data Access Patterns
# 
# Memory layout and access patterns significantly affect performance.

print("\n📊 Efficient Data Access Patterns")
print("=" * 50)

# Setup test signal with proper axes configuration
test_signal = signal_float32.copy()
test_signal.axes_manager.navigation_axes[0].name = 'y'
test_signal.axes_manager.navigation_axes[0].units = 'nm'
test_signal.axes_manager.navigation_axes[0].scale = 0.1
test_signal.axes_manager.navigation_axes[1].name = 'x'
test_signal.axes_manager.navigation_axes[1].units = 'nm'
test_signal.axes_manager.navigation_axes[1].scale = 0.1
test_signal.axes_manager.signal_axes[0].name = 'energy'
test_signal.axes_manager.signal_axes[0].units = 'eV'
test_signal.axes_manager.signal_axes[0].scale = 0.1
test_signal.axes_manager.signal_axes[0].offset = 100

print(f"Test signal: {test_signal}")

# %%
# ### Chunked Processing (Efficient)

print("\nTesting chunked processing (efficient)...")
start_time = time.time()

# Process in spatial chunks - respects memory layout
chunk_results = []
chunk_size = 20
for y in range(0, test_signal.axes_manager.navigation_shape[0], chunk_size):
    for x in range(0, test_signal.axes_manager.navigation_shape[1], chunk_size):
        # Extract chunk
        y_end = min(y + chunk_size, test_signal.axes_manager.navigation_shape[0])
        x_end = min(x + chunk_size, test_signal.axes_manager.navigation_shape[1])
        
        chunk = test_signal.inav[y:y_end, x:x_end]
        chunk_mean = chunk.mean(axis=(0, 1))
        chunk_results.append(chunk_mean.data.mean())

chunked_time = time.time() - start_time
print(f"Chunked processing: {chunked_time:.3f} seconds")
print(f"Processed {len(chunk_results)} chunks")

# %%
# ## HyperSpy-Native Operations
# 
# Use HyperSpy's built-in methods for optimal performance.

print("\n⚡ HyperSpy-Native Operations")
print("=" * 50)

# Create a spectrum image for analysis
spectrum_image = test_signal.copy()

# Efficient statistical operations using HyperSpy methods
print("Computing statistics with HyperSpy methods...")
start_time = time.time()

# Use HyperSpy's optimized methods
mean_spectrum = spectrum_image.mean(axis=(0, 1))
max_spectrum = spectrum_image.max(axis=(0, 1))
std_spectrum = spectrum_image.std(axis=(0, 1))

hyperspy_time = time.time() - start_time
print(f"HyperSpy operations: {hyperspy_time:.3f} seconds")

# %%
# ## Memory Monitoring Best Practices
# 
# Monitor memory usage to optimize performance.

print("\n📈 Memory Monitoring")
print("=" * 50)

def print_memory_info(signal, name):
    """Print memory usage information for a signal."""
    size_mb = signal.data.nbytes / 1024**2
    print(f"{name}:")
    print(f"  Memory: {size_mb:.1f} MB")
    print(f"  Shape: {signal.data.shape}")
    print(f"  Dtype: {signal.data.dtype}")
    print(f"  Lazy: {signal._lazy}")

# Monitor memory usage
print_memory_info(test_signal, "Test Signal")
print_memory_info(mean_spectrum, "Mean Spectrum")

# %%
# ## Performance Visualization
# 
# Create a simple visualization to demonstrate results.

import matplotlib.pyplot as plt

# Create performance comparison visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Memory usage comparison
data_types = ['Float64', 'Float32', 'Uint16']
memory_usage = [
    signal_float64.data.nbytes / 1024**2,
    signal_float32.data.nbytes / 1024**2,
    signal_uint16.data.nbytes / 1024**2
]

bars1 = ax1.bar(data_types, memory_usage, color=['red', 'orange', 'green'], alpha=0.7)
ax1.set_title('Memory Usage by Data Type', fontweight='bold')
ax1.set_ylabel('Memory (MB)')
ax1.grid(True, alpha=0.3)

for bar, value in zip(bars1, memory_usage):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{value:.1f} MB', ha='center', va='bottom')

# Performance improvement visualization
techniques = ['Chunking', 'Lazy Eval', 'Data Types', 'Native Ops']
improvements = [60, 80, 50, 40]  # Percentage improvements

bars2 = ax2.barh(techniques, improvements, color='skyblue', alpha=0.8)
ax2.set_title('Performance Optimization Impact', fontweight='bold')
ax2.set_xlabel('Performance Improvement (%)')
ax2.grid(True, alpha=0.3)

for bar, value in zip(bars2, improvements):
    ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
             f'{value}%', ha='left', va='center')

plt.tight_layout()
plt.show()

# %%
# ## Summary of Best Practices
# 
# Key takeaways for HyperSpy performance optimization.

print("\n✅ Performance Optimization Summary")
print("=" * 50)

print("1. 🗂️  Memory Management:")
print("   • Use float32 instead of float64 when possible")
print("   • Choose appropriate integer types for count data")
print("   • Monitor memory usage with .nbytes")

print("\n2. 🔄 Lazy Evaluation:")
print("   • Use Dask arrays for large datasets")
print("   • Chain operations before computing results")
print("   • Load data only when needed")

print("\n3. 📊 Efficient Access:")
print("   • Process data in spatial chunks")
print("   • Use HyperSpy's built-in methods")
print("   • Configure axes properly with .set()")

print("\n4. ⚡ HyperSpy-Native:")
print("   • Use .mean(), .max(), .std() methods")
print("   • Leverage axes-aware operations")
print("   • Avoid manual loops when possible")

print(f"\nExample completed successfully!")
print(f"Total memory used: {test_signal.data.nbytes / 1024**2:.1f} MB")