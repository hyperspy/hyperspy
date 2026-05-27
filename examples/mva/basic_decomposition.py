"""
Basic Decomposition Example
============================

This example demonstrates the basics of Principal Component Analysis (PCA) 
decomposition in HyperSpy for reducing dimensionality and extracting 
meaningful components from multidimensional data.

"""

import numpy as np
import hyperspy.api as hs

# Create synthetic spectrum image with noise
np.random.seed(42)  # For reproducible results

# Create a 2D navigation space (10x10) with 1D signal space (100 channels)
nav_shape = (10, 10)
sig_shape = (100,)
data_shape = nav_shape + sig_shape

# Generate synthetic data with two main components
energy_axis = np.linspace(0, 10, sig_shape[0])

# Component 1: Gaussian peak at energy 3 eV
component1 = np.exp(-(energy_axis - 3)**2 / 0.5)
# Component 2: Gaussian peak at energy 7 eV  
component2 = np.exp(-(energy_axis - 7)**2 / 0.8)

# Create spatial maps for the components
x, y = np.meshgrid(np.linspace(0, 1, nav_shape[0]), np.linspace(0, 1, nav_shape[1]), indexing='ij')
map1 = np.exp(-((x - 0.3)**2 + (y - 0.3)**2) / 0.1)  # Localized in corner
map2 = np.exp(-((x - 0.7)**2 + (y - 0.7)**2) / 0.15)  # Localized in opposite corner

# Combine components to create full dataset
data = np.zeros(data_shape)
for i in range(nav_shape[0]):
    for j in range(nav_shape[1]):
        data[i, j, :] = (map1[i, j] * component1 + 
                        map2[i, j] * component2 + 
                        0.1 * np.random.random(sig_shape[0]))  # Add noise

print("Creating synthetic spectrum image...")
print(f"Data shape: {data.shape} (navigation: {nav_shape}, signal: {sig_shape})")

# Create HyperSpy signal
s = hs.signals.Signal1D(data)

# Set proper axes
s.axes_manager.navigation_axes[0].name = 'x'
s.axes_manager.navigation_axes[0].units = 'μm'
s.axes_manager.navigation_axes[0].scale = 0.1

s.axes_manager.navigation_axes[1].name = 'y' 
s.axes_manager.navigation_axes[1].units = 'μm'
s.axes_manager.navigation_axes[1].scale = 0.1

s.axes_manager.signal_axes[0].name = 'Energy'
s.axes_manager.signal_axes[0].units = 'eV'
s.axes_manager.signal_axes[0].scale = 0.1

# Set metadata
s.metadata.General.title = "Synthetic Spectrum Image for Decomposition"
s.metadata.Signal.signal_type = "EDS_TEM"

print(f"Signal created: {s}")
print(f"Signal shape: {s.data.shape}")

# Perform PCA decomposition
print("\nPerforming PCA decomposition...")

# Decompose with SVD (Principal Component Analysis implementation in HyperSpy)
s.decomposition(algorithm='SVD', output_dimension=5)

print("Decomposition complete!")
print(f"Explained variance ratio: {s.learning_results.explained_variance_ratio}")

# Plot the scree plot to visualize explained variance
s.plot_explained_variance_ratio(n=10)

# Plot the first few components
print("\nPlotting first 3 components...")
s.plot_decomposition_factors(comp_ids=3, same_window=False)
s.plot_decomposition_loadings(comp_ids=3, same_window=False)

# Print some statistics
print(f"\nFirst 5 explained variance ratios:")
for i, ratio in enumerate(s.learning_results.explained_variance_ratio[:5]):
    print(f"  Component {i}: {ratio:.3f} ({ratio*100:.1f}%)")

cumulative_variance = np.cumsum(s.learning_results.explained_variance_ratio)
print(f"\nCumulative explained variance:")
print(f"  First 2 components: {cumulative_variance[1]:.3f} ({cumulative_variance[1]*100:.1f}%)")
print(f"  First 3 components: {cumulative_variance[2]:.3f} ({cumulative_variance[2]*100:.1f}%)")

# Reconstruct signal using first 2 components
print("\nReconstructing signal using first 2 components...")
s_reconstructed = s.get_decomposition_model(components=2)
s_reconstructed.metadata.General.title = "Reconstructed with 2 components"

print(f"Original signal shape: {s.data.shape}")
print(f"Reconstructed signal shape: {s_reconstructed.data.shape}")

# Show the original and reconstructed signals for comparison
print("Plotting original vs reconstructed signals...")
s.plot(title="Original Signal")
s_reconstructed.plot(title="Reconstructed (2 components)")

print("\nBasic decomposition example completed!")
print("Key takeaways:")
print("- PCA helps identify the main sources of variance in the data")
print("- The scree plot shows how much variance each component explains") 
print("- Components can be interpreted as characteristic spectra and their spatial distributions")
print("- Reconstruction with fewer components reduces noise while preserving main features")
