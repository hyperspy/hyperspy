"""
Interactive Operations for Live Data Analysis
=============================================

This example demonstrates interactive-style operations in HyperSpy that enable
live data analysis and responsive plotting. Interactive operations are crucial
for exploratory data analysis and real-time data processing workflows.
"""

# %%
# **Understanding Interactive Operations**
#
# Interactive operations in HyperSpy allow you to:
# 1. Perform calculations that update automatically when data changes
# 2. Create responsive analysis workflows
# 3. Build custom analysis pipelines
# 4. Maintain consistency between derived quantities and source data

import numpy as np
import hyperspy.api as hs

# %%
# **Creating Test Data for Interactive Analysis**
#
# We'll create a realistic spectral dataset that simulates common scientific measurements.
# This data will have both systematic trends and random variations.

# Create a 2D signal with navigation dimensions (10 positions, 100 energy channels)
np.random.seed(42)  # For reproducible results
positions = np.arange(10)
energies = np.linspace(0, 100, 100)

# Create synthetic spectral data with position-dependent characteristics
data = np.zeros((10, 100))
for i in range(10):
    # Base spectrum with position-dependent peak
    peak_center = 30 + i * 2  # Peak shifts with position
    spectrum = 50 * np.exp(-(energies - peak_center)**2 / 100)  # Gaussian peak
    spectrum += 20 * np.exp(-(energies - 70)**2 / 200)  # Secondary peak
    spectrum += 10 + 5 * np.random.random(100)  # Background + noise
    data[i, :] = spectrum

# Create the HyperSpy signal with proper calibration
test_signal = hs.signals.Signal1D(data)
test_signal.axes_manager.signal_axes[0].name = 'Energy'
test_signal.axes_manager.signal_axes[0].units = 'eV'
test_signal.axes_manager.signal_axes[0].scale = 1.0
test_signal.axes_manager.signal_axes[0].offset = 0.0
test_signal.axes_manager.navigation_axes[0].name = 'Position'
test_signal.axes_manager.navigation_axes[0].units = 'μm'
test_signal.axes_manager.navigation_axes[0].scale = 0.5

print("=== Test Dataset Information ===")
print(f"Signal shape: {test_signal.data.shape}")
print(f"Navigation shape: {test_signal.axes_manager.navigation_shape}")
print(f"Signal shape: {test_signal.axes_manager.signal_shape}")
print(f"Data range: {test_signal.data.min():.1f} to {test_signal.data.max():.1f}")

# %%
# **Basic Interactive Statistical Operations**
#
# These operations compute statistics along specific axes and are fundamental
# for understanding data distributions and trends.

# Compute statistics along the signal axis (energy dimension)
# Results have navigation shape (10,) - one value per position

interactive_max = test_signal.max(axis=-1)      # Maximum intensity per position
interactive_std = test_signal.std(axis=-1)      # Standard deviation per position  
interactive_mean = test_signal.mean(axis=-1)    # Mean intensity per position
interactive_min = test_signal.min(axis=-1)      # Minimum intensity per position

print("\n=== Statistical Operation Results ===")
print(f"Max values shape: {interactive_max.data.shape}")
print(f"Mean values shape: {interactive_mean.data.shape}")
print(f"Std values shape: {interactive_std.data.shape}")

# Analyze the statistical distributions
print(f"\nStatistical Summary:")
print(f"Max intensities: {interactive_max.data.min():.1f} to {interactive_max.data.max():.1f}")
print(f"Mean intensities: {interactive_mean.data.min():.1f} to {interactive_mean.data.max():.1f}")
print(f"Std deviations: {interactive_std.data.min():.1f} to {interactive_std.data.max():.1f}")

# %%
# **Understanding Data Responsiveness**
#
# Interactive operations maintain consistency when source data changes.
# This is crucial for live data analysis and iterative processing.

print("\n=== Demonstrating Data Responsiveness ===")

# Save original statistics for comparison
original_max = interactive_max.data.copy()
original_mean = interactive_mean.data.copy()
original_std = interactive_std.data.copy()

# Modify the signal data (simulate experimental drift or processing)
noise_level = 5
test_signal += np.random.normal(0, noise_level, test_signal.data.shape)

# Recompute statistics with modified data
new_max = test_signal.max(axis=-1)
new_mean = test_signal.mean(axis=-1) 
new_std = test_signal.std(axis=-1)

print(f"Data modification: Added Gaussian noise (σ = {noise_level})")
print(f"Original max range: [{original_max.min():.1f}, {original_max.max():.1f}]")
print(f"New max range: [{new_max.data.min():.1f}, {new_max.data.max():.1f}]")
print(f"Original mean range: [{original_mean.min():.1f}, {original_mean.max():.1f}]")
print(f"New mean range: [{new_mean.data.min():.1f}, {new_mean.data.max():.1f}]")

# Calculate the change in statistics
mean_change = np.abs(new_mean.data - original_mean).mean()
std_change = np.abs(new_std.data - original_std).mean()
print(f"Average change in mean: {mean_change:.2f}")
print(f"Average change in std: {std_change:.2f}")

# Update the interactive results
interactive_max = new_max
interactive_mean = new_mean
interactive_std = new_std

# %%
# **Visualizing Interactive Operations**
#
# Visualization is essential for understanding the behavior of interactive operations
# and validating the analysis results.

print("\n=== Creating Visualizations ===")

# Plot the original signal for reference
sample_spectrum = test_signal.inav[0]
sample_spectrum.metadata.General.title = 'Sample Spectrum (Position 0)'
sample_spectrum.plot()

# Plot the statistical results
interactive_max.metadata.General.title = 'Maximum Intensity vs Position'
interactive_max.plot()

interactive_mean.metadata.General.title = 'Mean Intensity vs Position'
interactive_mean.plot()

interactive_std.metadata.General.title = 'Standard Deviation vs Position'
interactive_std.plot()

# %%
# **Advanced Interactive Analysis Functions**
#
# Custom functions enable domain-specific analysis and complex data processing.
# These functions demonstrate how to build sophisticated analysis workflows.

def compute_signal_to_noise_ratio(signal, noise_region=(0, 10)):
    """
    Compute signal-to-noise ratio for each spectrum.
    
    Parameters
    ----------
    signal : Signal1D
        Input signal
    noise_region : tuple
        Index range for noise estimation (start, end)
        
    Returns
    -------
    Signal1D
        Signal-to-noise ratio for each spectrum
    """
    signal_max = signal.max(axis=-1)
    noise_std = signal.isig[noise_region[0]:noise_region[1]].std(axis=-1)
    # Avoid division by zero
    noise_std.data[noise_std.data == 0] = 1e-10
    return signal_max / noise_std

def compute_peak_characteristics(signal, prominence_threshold=5):
    """
    Analyze peak characteristics for each spectrum.
    
    Parameters
    ----------
    signal : Signal1D
        Input signal
    prominence_threshold : float
        Minimum prominence for peak detection
        
    Returns
    -------
    dict
        Dictionary containing peak position, height, and width information
    """
    peak_positions = signal.indexmax(axis=-1)  # Position of maximum
    peak_heights = signal.max(axis=-1)         # Height of maximum
    
    # Estimate peak width using full width at half maximum (FWHM)
    peak_widths = []
    for i in range(signal.axes_manager.navigation_size):
        spectrum = signal.inav[i].data
        peak_idx = int(peak_positions.data[i])
        peak_height = peak_heights.data[i]
        half_max = peak_height / 2
        
        # Find indices where signal crosses half maximum
        left_idx = peak_idx
        right_idx = peak_idx
        
        # Search left
        while left_idx > 0 and spectrum[left_idx] > half_max:
            left_idx -= 1
        
        # Search right
        while right_idx < len(spectrum) - 1 and spectrum[right_idx] > half_max:
            right_idx += 1
            
        width = right_idx - left_idx
        peak_widths.append(width)
    
    return {
        'positions': peak_positions,
        'heights': peak_heights,
        'widths': hs.signals.Signal1D(np.array(peak_widths))
    }

def compute_integrated_intensity(signal, integration_range=(20, 80)):
    """
    Compute integrated intensity over specified energy range.
    
    Parameters
    ----------
    signal : Signal1D
        Input signal
    integration_range : tuple
        Energy range for integration (start_idx, end_idx)
        
    Returns
    -------
    Signal1D
        Integrated intensity for each spectrum
    """
    return signal.isig[integration_range[0]:integration_range[1]].sum(axis=-1)

# %%
# **Applying Advanced Analysis Functions**
#
# Let's apply these functions to demonstrate sophisticated interactive analysis.

print("\n=== Advanced Interactive Analysis ===")

# Compute signal-to-noise ratio
snr = compute_signal_to_noise_ratio(test_signal, noise_region=(0, 10))
print(f"Signal-to-noise ratio statistics:")
print(f"  Range: {snr.data.min():.1f} to {snr.data.max():.1f}")
print(f"  Mean: {snr.data.mean():.1f}")

# Analyze peak characteristics
peak_info = compute_peak_characteristics(test_signal)
print(f"Peak analysis results:")
print(f"  Position range: {peak_info['positions'].data.min():.0f} to {peak_info['positions'].data.max():.0f}")
print(f"  Height range: {peak_info['heights'].data.min():.1f} to {peak_info['heights'].data.max():.1f}")
print(f"  Width range: {peak_info['widths'].data.min():.0f} to {peak_info['widths'].data.max():.0f}")

# Compute integrated intensity
integrated = compute_integrated_intensity(test_signal, integration_range=(20, 80))
print(f"Integrated intensity statistics:")
print(f"  Range: {integrated.data.min():.1f} to {integrated.data.max():.1f}")
print(f"  Mean: {integrated.data.mean():.1f}")

# %%
# **Visualizing Advanced Analysis Results**
#
# Create comprehensive visualizations to understand the analysis results.

print("\n=== Advanced Analysis Visualizations ===")

# Set up proper metadata for plotting
snr.metadata.General.title = 'Signal-to-Noise Ratio vs Position'
snr.axes_manager.navigation_axes[0].name = 'Position'
snr.axes_manager.navigation_axes[0].units = 'μm'
snr.plot()

peak_info['positions'].metadata.General.title = 'Peak Position vs Position'
peak_info['positions'].axes_manager.navigation_axes[0].name = 'Position'
peak_info['positions'].axes_manager.navigation_axes[0].units = 'μm'
peak_info['positions'].plot()

peak_info['heights'].metadata.General.title = 'Peak Height vs Position'
peak_info['heights'].axes_manager.navigation_axes[0].name = 'Position'
peak_info['heights'].axes_manager.navigation_axes[0].units = 'μm'
peak_info['heights'].plot()

integrated.metadata.General.title = 'Integrated Intensity vs Position'
integrated.axes_manager.navigation_axes[0].name = 'Position'
integrated.axes_manager.navigation_axes[0].units = 'μm'
integrated.plot()

# %%
# **Building Interactive Analysis Workflows**
#
# Demonstrate how to create analysis pipelines that automatically update
# when data changes.

class InteractiveAnalyzer:
    """
    A class for managing interactive analysis workflows.
    """
    
    def __init__(self, signal):
        self.signal = signal
        self.results = {}
    
    def compute_all_statistics(self):
        """Compute all statistical measures."""
        self.results['max'] = self.signal.max(axis=-1)
        self.results['mean'] = self.signal.mean(axis=-1)
        self.results['std'] = self.signal.std(axis=-1)
        self.results['snr'] = compute_signal_to_noise_ratio(self.signal)
        self.results['integrated'] = compute_integrated_intensity(self.signal)
        
    def update_data(self, new_data):
        """Update signal data and recompute statistics."""
        self.signal.data = new_data
        self.compute_all_statistics()
    
    def get_summary(self):
        """Get summary of all computed statistics."""
        if not self.results:
            self.compute_all_statistics()
        
        summary = {}
        for key, result in self.results.items():
            summary[key] = {
                'mean': result.data.mean(),
                'std': result.data.std(),
                'range': (result.data.min(), result.data.max())
            }
        return summary

# %%
# **Demonstration of Interactive Analysis Workflow**

print("\n=== Interactive Analysis Workflow ===")

# Create analyzer instance
analyzer = InteractiveAnalyzer(test_signal)

# Compute initial statistics
analyzer.compute_all_statistics()
initial_summary = analyzer.get_summary()

print("Initial analysis summary:")
for metric, stats in initial_summary.items():
    print(f"  {metric}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")

# Simulate data update (e.g., new measurement)
updated_data = test_signal.data + np.random.normal(0, 2, test_signal.data.shape)
analyzer.update_data(updated_data)
updated_summary = analyzer.get_summary()

print("\nUpdated analysis summary:")
for metric, stats in updated_summary.items():
    print(f"  {metric}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")

# %%
# **Key Concepts and Best Practices**
#
# Interactive operations in HyperSpy enable powerful data analysis workflows:

print("\n=== Key Takeaways ===")
print("1. Statistical Operations:")
print("   - Use max(), min(), mean(), std() for basic statistics")
print("   - Operations along specific axes create derived datasets")
print("   - Results automatically maintain proper dimensionality")

print("\n2. Data Responsiveness:")
print("   - Recompute derived quantities when source data changes")
print("   - Maintain consistency between original and processed data")
print("   - Use interactive patterns for live data analysis")

print("\n3. Custom Analysis Functions:")
print("   - Build domain-specific analysis tools")
print("   - Combine multiple operations for complex workflows")
print("   - Create reusable analysis pipelines")

print("\n4. Visualization Integration:")
print("   - Plot results to validate analysis")
print("   - Use proper metadata for clear visualizations")
print("   - Leverage HyperSpy's native plotting capabilities")

print("\n5. Workflow Management:")
print("   - Organize analysis in classes for complex projects")
print("   - Implement update mechanisms for dynamic data")
print("   - Build summary and reporting functions")

print("\nInteractive operations are essential for:")
print("- Exploratory data analysis")
print("- Real-time data processing")
print("- Quality control and monitoring")
print("- Automated analysis pipelines")
print("- Interactive scientific computing")

# Recompute statistics with modified data
new_max = test_signal.max(axis=-1)
new_mean = test_signal.mean(axis=-1) 
new_std = test_signal.std(axis=-1)

print(f"Original max range: [{original_max.min():.2f}, {original_max.max():.2f}]")
print(f"New max range: [{new_max.data.min():.2f}, {new_max.data.max():.2f}]")
print(f"Original mean range: [{original_mean.min():.2f}, {original_mean.max():.2f}]")
print(f"New mean range: [{new_mean.data.min():.2f}, {new_mean.data.max():.2f}]")

# Update the interactive results for plotting
interactive_max = new_max
interactive_mean = new_mean
interactive_std = new_std

# %%
# Visualize the interactive operations using HyperSpy's native plotting
print("\n--- Visualizing interactive operations ---")

# Original signal (first spectrum)
first_spectrum = test_signal.inav[0]
first_spectrum.metadata.General.title = 'Original signal (first spectrum)'
first_spectrum.plot()

# Interactive operations results - convert to Signal1D for proper plotting
interactive_max.metadata.General.title = 'Maximum along signal axis'
interactive_max.plot()

interactive_std.metadata.General.title = 'Standard deviation along signal axis'
interactive_std.plot()

interactive_mean.metadata.General.title = 'Mean along signal axis'
interactive_mean.plot()

# %%
# Example of interactive-style functions
print("\n--- Custom interactive-style functions ---")

def compute_signal_to_noise(signal):
    """Compute signal-to-noise ratio"""
    signal_mean = signal.mean(axis=-1)
    signal_std = signal.std(axis=-1)
    return signal_mean / signal_std

def compute_peak_position(signal):
    """Find peak position for each spectrum"""
    return signal.indexmax(axis=-1)

def compute_integrated_intensity(signal, start_idx=20, end_idx=80):
    """Compute integrated intensity over a range"""
    return signal.isig[start_idx:end_idx].sum(axis=-1)

# Apply these functions
snr = compute_signal_to_noise(test_signal)
peak_pos = compute_peak_position(test_signal)
integrated = compute_integrated_intensity(test_signal)

print(f"Signal-to-noise ratio - mean: {snr.data.mean():.2f}")
print(f"Peak positions - range: [{peak_pos.data.min()}, {peak_pos.data.max()}]")
print(f"Integrated intensity - mean: {integrated.data.mean():.2f}")

# %%
# Plot the custom functions results using HyperSpy's native plotting
print("\n--- Visualizing custom analysis results ---")

snr.metadata.General.title = 'Signal-to-Noise Ratio'
snr.plot()

peak_pos.metadata.General.title = 'Peak Position'  
peak_pos.plot()

integrated.metadata.General.title = 'Integrated Intensity'
integrated.plot()

print("\nInteractive operations example completed!")
print("Key points:")
print("- Use max(), min(), mean(), std() for basic statistics")
print("- Apply custom functions for domain-specific analysis")
print("- Recompute when data changes to maintain consistency")
print("- Visualize results to understand data behavior")
