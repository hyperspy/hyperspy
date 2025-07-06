"""
Programmatic Axis Navigation
============================

This example demonstrates how to programmatically navigate through
multidimensional datasets using HyperSpy's axes manager. Learn how to
set positions, iterate through data, and control navigation programmatically
for automated analysis workflows.

Key concepts covered:
- Setting navigation indices and coordinates pr        stats = {
            'indices': indices,
            'time': signal.axes_manager[0].value,
            'temperature': signal.axes_manager[1].value,
            'mean': np.mean(image.data),   # Use NumPy for scalar statistics
            'std': np.std(image.data),     # Use NumPy for scalar statistics
            'max': np.max(image.data),     # Use NumPy for scalar statistics
            'min': np.min(image.data)      # Use NumPy for scalar statistics
        }cally
- Iterating through navigation dimensions
- Using index vs coordinate systems
- Automated navigation for batch processing
- Custom navigation patterns

"""

import numpy as np
import hyperspy.api as hs

# %%
# **Creating multidimensional test data**
#
# First, let's create a 4D dataset to demonstrate navigation techniques.
# This represents time-resolved 2D imaging with temperature variation.

# Create a 4D dataset (2 navigation + 2 signal dimensions)
# This could represent time-resolved 2D imaging
nav_time = 10   # Time points
nav_temp = 8    # Temperature points
img_y = 64      # Image height
img_x = 64      # Image width

# Initialize data array
data_4d = np.zeros((nav_time, nav_temp, img_y, img_x))

# Create synthetic data with time and temperature dependencies
for t in range(nav_time):
    for temp in range(nav_temp):
        # Create a pattern that evolves with time and temperature
        y, x = np.ogrid[:img_y, :img_x]
        center_y, center_x = img_y//2, img_x//2
        
        # Pattern shifts and changes with time and temperature
        shift_y = 5 * np.sin(t * 0.6) * (temp / nav_temp)
        shift_x = 5 * np.cos(t * 0.6) * (temp / nav_temp)
        
        # Create Gaussian blob that moves and changes
        pattern = np.exp(-((y - center_y - shift_y)**2 + (x - center_x - shift_x)**2) / 
                        (2 * (10 + 5 * temp)**2))
        
        # Add some noise
        pattern += 0.1 * np.random.randn(img_y, img_x)
        
        data_4d[t, temp] = pattern

# Create HyperSpy signal
s = hs.signals.Signal2D(data_4d)

# Configure axes properly
# Set navigation axes (Time and Temperature)
s.axes_manager.navigation_axes.set(
    name=['Time', 'Temperature'],
    units=['s', '°C'],
    scale=[0.5, 10],
    offset=[0, 300]
)

# Set signal axes (Y and X positions)
s.axes_manager.signal_axes.set(
    name=['Y', 'X'],
    units=['μm', 'μm'],
    scale=[0.1, 0.1]
)

s.metadata.General.title = 'Time-Temperature 2D Imaging'

print(f"Created 4D signal: {s}")
print(f"Navigation axes: {[ax.name for ax in s.axes_manager.navigation_axes]}")
print(f"Signal axes: {[ax.name for ax in s.axes_manager.signal_axes]}")

# %%
# Basic Programmatic Navigation
# -----------------------------

print("\n" + "="*50)
print("BASIC PROGRAMMATIC NAVIGATION")
print("="*50)

# 1. Setting navigation indices
print("\n1. Setting navigation positions using indices:")
print(f"   Initial position: {s.axes_manager.indices}")
print(f"   Current time: {s.axes_manager[0].value:.1f} {s.axes_manager[0].units}")
print(f"   Current temperature: {s.axes_manager[1].value:.1f} {s.axes_manager[1].units}")

# Move to a specific position
s.axes_manager.indices = (7, 3)  # time index 7, temperature index 3
print(f"   After setting indices to (7, 3):")
print(f"   New position: {s.axes_manager.indices}")
print(f"   Current time: {s.axes_manager[0].value:.1f} {s.axes_manager[0].units}")
print(f"   Current temperature: {s.axes_manager[1].value:.1f} {s.axes_manager[1].units}")

# 2. Setting individual axis indices
print("\n2. Setting individual axis indices:")
s.axes_manager[0].index = 2  # Move to time index 2
s.axes_manager[1].index = 5  # Move to temperature index 5
print(f"   After individual axis setting:")
print(f"   Position: {s.axes_manager.indices}")
print(f"   Time: {s.axes_manager[0].value:.1f} {s.axes_manager[0].units}")
print(f"   Temperature: {s.axes_manager[1].value:.1f} {s.axes_manager[1].units}")

# 3. Setting axes by coordinate values
print("\n3. Setting axes by coordinate values:")
s.axes_manager[0].value = 3.0  # Set time to 3.0 seconds
s.axes_manager[1].value = 350  # Set temperature to 350°C
print(f"   After setting by values (3.0s, 350°C):")
print(f"   Position: {s.axes_manager.indices}")
print(f"   Actual values: {s.axes_manager[0].value:.1f}s, {s.axes_manager[1].value:.1f}°C")

# %%
# Systematic Navigation Patterns
# ------------------------------

print("\n" + "="*50)
print("SYSTEMATIC NAVIGATION PATTERNS")
print("="*50)

# 1. Sequential navigation through all positions
print("\n1. Sequential navigation through all positions:")
position_count = 0
for indices in s.axes_manager:
    position_count += 1
    time_val = s.axes_manager[0].value
    temp_val = s.axes_manager[1].value
    
    # Only print first few and last few to avoid clutter
    if position_count <= 3 or position_count > s.axes_manager.navigation_size - 3:
        print(f"   Position {position_count}: indices={indices}, time={time_val:.1f}s, temp={temp_val:.1f}°C")
    elif position_count == 4:
        print("   ... (intermediate positions) ...")

print(f"   Total positions visited: {position_count}")

# 2. Custom navigation pattern - diagonal sweep
print("\n2. Custom navigation pattern - diagonal sweep:")
diagonal_positions = []
for i in range(min(s.axes_manager[0].size, s.axes_manager[1].size)):
    indices = (i, i)
    s.axes_manager.indices = indices
    time_val = s.axes_manager[0].value
    temp_val = s.axes_manager[1].value
    diagonal_positions.append((indices, time_val, temp_val))
    print(f"   Diagonal step {i+1}: indices={indices}, time={time_val:.1f}s, temp={temp_val:.1f}°C")

# 3. Grid sampling pattern
print("\n3. Grid sampling pattern (every 2nd position):")
grid_positions = []
for t_idx in range(0, s.axes_manager[0].size, 2):
    for temp_idx in range(0, s.axes_manager[1].size, 2):
        indices = (t_idx, temp_idx)
        s.axes_manager.indices = indices
        grid_positions.append(indices)

print(f"   Grid sampling: {len(grid_positions)} positions sampled")
print(f"   Sample positions: {grid_positions[:5]}...")

# %%
# Practical Applications
# ----------------------

print("\n" + "="*50)
print("PRACTICAL APPLICATIONS")
print("="*50)

# 1. Find maximum intensity position
print("\n1. Finding position with maximum intensity:")
max_intensity = 0
max_position = None
max_coords = None

for indices in s.axes_manager:
    current_image = s.inav[indices]
    intensity = np.max(current_image.data)  # Use NumPy for scalar maximum of image
    
    if intensity > max_intensity:
        max_intensity = intensity
        max_position = indices
        max_coords = (s.axes_manager[0].value, s.axes_manager[1].value)

s.axes_manager.indices = max_position
print(f"   Maximum intensity: {max_intensity:.3f}")
print(f"   Found at indices: {max_position}")
if max_coords is not None:
    print(f"   Coordinates: {max_coords[0]:.1f}s, {max_coords[1]:.1f}°C")
else:
    print("   Coordinates: Not available")

# 2. Extract time series at specific temperature
print("\n2. Extracting time series at specific temperature:")
target_temp = 330  # Target temperature
s.axes_manager[1].value = target_temp  # Set temperature
actual_temp = s.axes_manager[1].value
temp_index = s.axes_manager[1].index

print(f"   Target temperature: {target_temp}°C")
print(f"   Actual temperature: {actual_temp:.1f}°C (index {temp_index})")

# Extract time series
time_series_max = []
time_values = []

for t_idx in range(s.axes_manager[0].size):
    s.axes_manager[0].index = t_idx
    current_image = s.inav[s.axes_manager.indices]
    max_val = np.max(current_image.data)  # Use NumPy for scalar maximum of image
    time_series_max.append(max_val)
    time_values.append(s.axes_manager[0].value)

print(f"   Extracted {len(time_series_max)} time points")

# 3. Temperature sweep at specific time
print("\n3. Temperature sweep at specific time:")
target_time = 2.0  # Target time
s.axes_manager[0].value = target_time
actual_time = s.axes_manager[0].value
time_index = s.axes_manager[0].index

print(f"   Target time: {target_time}s")
print(f"   Actual time: {actual_time:.1f}s (index {time_index})")

temp_series_max = []
temp_values = []

for temp_idx in range(s.axes_manager[1].size):
    s.axes_manager[1].index = temp_idx
    current_image = s.inav[s.axes_manager.indices]
    max_val = np.max(current_image.data)  # Use NumPy for scalar maximum of image
    temp_series_max.append(max_val)
    temp_values.append(s.axes_manager[1].value)

print(f"   Extracted {len(temp_series_max)} temperature points")

# %%
# Visualization of Navigation Results using HyperSpy plotting
# -----------------------------------

print("\n4. Visualizing navigation results:")

# Create signals for time and temperature series for better plotting
time_signal = hs.signals.Signal1D(time_series_max)
time_signal.axes_manager.signal_axes.set(
    name=['Time'],
    units=['s'],
    scale=[s.axes_manager[0].scale],
    offset=[s.axes_manager[0].offset]
)
time_signal.metadata.General.title = f'Time Series at {actual_temp:.1f}°C'
time_signal.plot()

temp_signal = hs.signals.Signal1D(temp_series_max)
temp_signal.axes_manager.signal_axes.set(
    name=['Temperature'],
    units=['°C'],
    scale=[s.axes_manager[1].scale],
    offset=[s.axes_manager[1].offset]
)
temp_signal.metadata.General.title = f'Temperature Series at {actual_time:.1f}s'
temp_signal.plot()

# Show image at maximum intensity position using HyperSpy
s.axes_manager.indices = max_position
max_image = s.inav[max_position]
if max_coords is not None:
    max_image.metadata.General.title = f'Image at Max Intensity (t={max_coords[0]:.1f}s, T={max_coords[1]:.1f}°C)'
else:
    max_image.metadata.General.title = 'Image at Max Intensity'
max_image.plot()

# Create navigation heatmap as a HyperSpy signal
nav_intensity_map = np.zeros((s.axes_manager[0].size, s.axes_manager[1].size))
for t_idx in range(s.axes_manager[0].size):
    for temp_idx in range(s.axes_manager[1].size):
        s.axes_manager.indices = (t_idx, temp_idx)
        nav_intensity_map[t_idx, temp_idx] = np.max(s.inav[s.axes_manager.indices].data)  # Use NumPy for scalar

heatmap_signal = hs.signals.Signal2D(nav_intensity_map)
heatmap_signal.axes_manager[0].name = 'Time'
heatmap_signal.axes_manager[0].units = 's'
heatmap_signal.axes_manager[0].scale = s.axes_manager[0].scale
heatmap_signal.axes_manager[0].offset = s.axes_manager[0].offset
heatmap_signal.axes_manager[1].name = 'Temperature'
heatmap_signal.axes_manager[1].units = '°C'
heatmap_signal.axes_manager[1].scale = s.axes_manager[1].scale
heatmap_signal.axes_manager[1].offset = s.axes_manager[1].offset
heatmap_signal.metadata.General.title = 'Navigation Space Heatmap (Max Intensity)'
heatmap_signal.plot()

# Add marker for maximum position using HyperSpy marker
if max_coords is not None:
    max_marker = hs.plot.markers.Points(
        offsets=[max_coords[1], max_coords[0]],  # Temperature, Time
        sizes=200,
        color='red'
    )
    heatmap_signal.add_marker(max_marker, plot_marker=True)

# %%
# Advanced Navigation Techniques
# ------------------------------

print("\n" + "="*50)
print("ADVANCED NAVIGATION TECHNIQUES")
print("="*50)

# 1. Custom iterator with specific pattern
print("\n1. Custom navigation iterator:")

def spiral_navigation(axes_manager):
    """Custom iterator that navigates in a spiral pattern."""
    t_size = axes_manager[0].size
    temp_size = axes_manager[1].size
    
    # Start from center
    t_center = t_size // 2
    temp_center = temp_size // 2
    
    visited = set()
    positions = [(t_center, temp_center)]
    
    # Add spiral positions
    for radius in range(1, max(t_size, temp_size)):
        for dt in range(-radius, radius + 1):
            for dtemp in range(-radius, radius + 1):
                if abs(dt) == radius or abs(dtemp) == radius:
                    t_pos = t_center + dt
                    temp_pos = temp_center + dtemp
                    
                    if (0 <= t_pos < t_size and 0 <= temp_pos < temp_size and
                        (t_pos, temp_pos) not in visited):
                        positions.append((t_pos, temp_pos))
                        visited.add((t_pos, temp_pos))
    
    return positions

# Test spiral navigation
spiral_positions = spiral_navigation(s.axes_manager)
print(f"   Spiral navigation pattern: {len(spiral_positions)} positions")
print(f"   First 10 positions: {spiral_positions[:10]}")

# 2. Navigation with condition checking
print("\n2. Conditional navigation (finding positions above threshold):")
threshold = 0.5
high_intensity_positions = []

for indices in s.axes_manager:
    current_image = s.inav[indices]
    max_intensity = np.max(current_image.data)  # Use NumPy for scalar maximum of image
    
    if max_intensity > threshold:
        coords = (s.axes_manager[0].value, s.axes_manager[1].value)
        high_intensity_positions.append((indices, coords, max_intensity))

print(f"   Found {len(high_intensity_positions)} positions above threshold {threshold}")
print("   Top 3 positions:")
sorted_positions = sorted(high_intensity_positions, key=lambda x: x[2], reverse=True)
for i, (indices, coords, intensity) in enumerate(sorted_positions[:3]):
    print(f"   {i+1}. Indices: {indices}, Coords: ({coords[0]:.1f}s, {coords[1]:.1f}°C), Intensity: {intensity:.3f}")

# 3. Batch processing example
print("\n3. Batch processing example - calculate statistics for each position:")

class NavigationStatistics:
    def __init__(self):
        self.stats = []
    
    def process_position(self, signal, indices):
        """Process a single navigation position."""
        signal.axes_manager.indices = indices
        image = signal.inav[indices]
        
        stats = {
            'indices': indices,
            'time': signal.axes_manager[0].value,
            'temperature': signal.axes_manager[1].value,
            'mean': image.mean(axis=None),   # Use HyperSpy method for scalar mean
            'std': image.std(axis=None),     # Use HyperSpy method for scalar std
            'max': image.max(axis=None),     # Use HyperSpy method for scalar max
            'min': image.min(axis=None)      # Use HyperSpy method for scalar min
        }
        self.stats.append(stats)
        return stats

# Process all positions
processor = NavigationStatistics()
for indices in s.axes_manager:
    processor.process_position(s, indices)

print(f"   Processed {len(processor.stats)} positions")
print("   Statistics summary:")
all_means = [stat['mean'] for stat in processor.stats]
all_stds = [stat['std'] for stat in processor.stats]
print(f"   Mean intensity range: {np.min(all_means):.3f} - {np.max(all_means):.3f}")
print(f"   Std deviation range: {np.min(all_stds):.3f} - {np.max(all_stds):.3f}")

# %%
# Best Practices and Tips
# -----------------------

print("\n" + "="*60)
print("BEST PRACTICES FOR PROGRAMMATIC NAVIGATION")
print("="*60)
print("""
1. INDEX vs COORDINATE NAVIGATION:
   • Use indices for exact position control (integer values)
   • Use coordinates for approximate positioning (float values)
   • Coordinate setting finds nearest valid position

2. EFFICIENT ITERATION:
   • Use built-in axes_manager iterator for sequential access
   • Create custom iterators for specific patterns
   • Consider memory usage for large datasets

3. NAVIGATION STATE MANAGEMENT:
   • Always check current position before processing
   • Store important positions for later reference
   • Use try/except for bounds checking

4. PERFORMANCE OPTIMIZATION:
   • Minimize navigation operations in inner loops
   • Cache frequently accessed positions
   • Use vectorized operations when possible

5. COORDINATE SYSTEM AWARENESS:
   • Remember that indices are 0-based
   • Coordinate values depend on scale and offset
   • Check axis properties before navigation
""")

# Example of safe navigation with bounds checking
print("\n4. Safe navigation with bounds checking:")

def safe_navigate(signal, time_coord, temp_coord):
    """Safely navigate to coordinates with bounds checking."""
    # Store original position
    original_indices = signal.axes_manager.indices
    
    try:
        # Attempt navigation
        signal.axes_manager[0].value = time_coord
        signal.axes_manager[1].value = temp_coord
        
        actual_coords = (signal.axes_manager[0].value, signal.axes_manager[1].value)
        success = True
        
    except Exception as e:
        # Restore original position on error
        signal.axes_manager.indices = original_indices
        actual_coords = None
        success = False
        print(f"   Navigation failed: {e}")
    
    return success, actual_coords

# Test safe navigation
test_coords = [(1.5, 340), (10.0, 400), (-1.0, 300)]  # Some valid, some invalid
for time_coord, temp_coord in test_coords:
    success, actual = safe_navigate(s, time_coord, temp_coord)
    if success:
        print(f"   ✓ Navigation to ({time_coord}, {temp_coord}) → {actual}")
    else:
        print(f"   ✗ Failed navigation to ({time_coord}, {temp_coord})")

print("\n5. Navigation performance comparison:")
import time

# Method 1: Sequential index setting
start_time = time.time()
for i in range(min(100, s.axes_manager.navigation_size)):
    s.axes_manager.indices = (i % s.axes_manager[0].size, i % s.axes_manager[1].size)
index_time = time.time() - start_time

# Method 2: Built-in iterator (limited to 100 iterations)
start_time = time.time()
count = 0
for indices in s.axes_manager:
    count += 1
    if count >= 100:
        break
iterator_time = time.time() - start_time

print(f"   Index setting (100 ops): {index_time*1000:.2f} ms")
print(f"   Iterator method (100 ops): {iterator_time*1000:.2f} ms")
print(f"   Performance ratio: {index_time/iterator_time:.1f}x")

print("\nProgrammatic axis navigation examples completed!")
print("Use s.axes_manager.indices and s.axes_manager[i].value for navigation!")
