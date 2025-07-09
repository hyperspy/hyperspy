"""
Decomposition Model Reconstruction Example
==========================================

This example demonstrates how to use decomposition results for signal 
reconstruction, denoising, and component analysis. It shows how to extract
and manipulate decomposition factors and loadings.

"""

import numpy as np
import hyperspy.api as hs
import matplotlib.pyplot as plt

# Create synthetic noisy spectrum image
np.random.seed(123)

print("Creating synthetic noisy spectrum image...")

# Parameters
nav_size = 20
sig_size = 200
energy_axis = np.linspace(0, 20, sig_size)

# Create three distinct spectral components
# Component 1: Sharp peak at 5 eV
comp1_spectrum = 10 * np.exp(-((energy_axis - 5)**2) / 0.2)

# Component 2: Broad background
comp2_spectrum = 3 * np.exp(-energy_axis / 8) + 1

# Component 3: Sharp peak at 15 eV  
comp3_spectrum = 8 * np.exp(-((energy_axis - 15)**2) / 0.3)

# Create spatial distributions for each component
x, y = np.meshgrid(np.linspace(0, 1, nav_size), np.linspace(0, 1, nav_size), indexing='ij')

# Component 1: Concentrated in center
map1 = np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.1)

# Component 2: Uniform background
map2 = np.ones_like(x) * 0.8

# Component 3: Ring pattern
distance = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)
map3 = np.exp(-((distance - 0.3)**2) / 0.05)

# Combine to create full dataset
data = np.zeros((nav_size, nav_size, sig_size))
for i in range(nav_size):
    for j in range(nav_size):
        clean_signal = (map1[i, j] * comp1_spectrum + 
                       map2[i, j] * comp2_spectrum + 
                       map3[i, j] * comp3_spectrum)
        
        # Add significant noise
        noise_level = 0.3 * np.max(clean_signal)
        noisy_signal = clean_signal + noise_level * np.random.random(sig_size)
        
        data[i, j, :] = noisy_signal

print(f"Data shape: {data.shape}")

# Create HyperSpy signal
s = hs.signals.Signal1D(data)

# Set axes properties
s.axes_manager.navigation_axes[0].name = 'x'
s.axes_manager.navigation_axes[0].units = 'nm'
s.axes_manager.navigation_axes[0].scale = 5.0

s.axes_manager.navigation_axes[1].name = 'y'
s.axes_manager.navigation_axes[1].units = 'nm' 
s.axes_manager.navigation_axes[1].scale = 5.0

s.axes_manager.signal_axes[0].name = 'Energy'
s.axes_manager.signal_axes[0].units = 'eV'
s.axes_manager.signal_axes[0].scale = 0.1

s.metadata.General.title = "Noisy Synthetic Spectrum Image"

print(f"Original signal: {s}")

# Perform decomposition
print("\nPerforming SVD decomposition...")
s.decomposition(algorithm='SVD', output_dimension=10)

# Analyze explained variance
explained_var = s.learning_results.explained_variance_ratio
cumulative_var = np.cumsum(explained_var)

print(f"Explained variance ratios (first 5): {explained_var[:5]}")
print(f"Cumulative variance (first 5): {cumulative_var[:5]}")

# Plot explained variance
s.plot_explained_variance_ratio(n=10)

# Determine optimal number of components (e.g., 95% of variance)
n_components_95 = np.argmax(cumulative_var >= 0.95) + 1
print(f"\nComponents needed for 95% variance: {n_components_95}")

# Visualize the factors (characteristic spectra)
print("\nPlotting decomposition factors (characteristic spectra)...")
s.plot_decomposition_factors(comp_ids=5, same_window=True, 
                           title="Decomposition Factors")

# Visualize the loadings (spatial distributions)
print("Plotting decomposition loadings (spatial maps)...")
s.plot_decomposition_loadings(comp_ids=5, same_window=False,
                            title="Decomposition Loadings")

# Reconstruct signals with different numbers of components
print("\nReconstructing signals with different component numbers...")

# Reconstruction with 3 components (should capture main features)
s_recon_3 = s.get_decomposition_model(components=3)
s_recon_3.metadata.General.title = "Reconstructed (3 components)"

# Reconstruction with optimal number of components
s_recon_opt = s.get_decomposition_model(components=n_components_95)
s_recon_opt.metadata.General.title = f"Reconstructed ({n_components_95} components)"

# Calculate reconstruction errors
def calculate_rmse(original, reconstructed):
    """Calculate Root Mean Square Error between signals."""
    return np.sqrt(np.mean((original.data - reconstructed.data)**2))

rmse_3 = calculate_rmse(s, s_recon_3)
rmse_opt = calculate_rmse(s, s_recon_opt)

print(f"\nReconstruction RMSE:")
print(f"  3 components: {rmse_3:.4f}")
print(f"  {n_components_95} components: {rmse_opt:.4f}")

# Compare original vs reconstructed at a specific position
print("\nComparing spectra at center position...")
center_pos = (nav_size//2, nav_size//2)

# Plot comparison using matplotlib directly
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Original signal map (show mean intensity)
original_map = np.mean(s.data, axis=2)  # Mean over energy axis
im1 = axes[0,0].imshow(original_map, cmap='viridis')
axes[0,0].set_title("Original Signal (Mean Intensity)")
plt.colorbar(im1, ax=axes[0,0])

# Reconstructed signal map (3 components)
recon_map_3 = np.mean(s_recon_3.data, axis=2)
im2 = axes[0,1].imshow(recon_map_3, cmap='viridis')
axes[0,1].set_title("Reconstructed (3 components)")
plt.colorbar(im2, ax=axes[0,1])

# Spectrum comparison at center
energy_axis = s.axes_manager.signal_axes[0].axis
axes[1,0].plot(energy_axis, s.inav[center_pos].data, label="Original", linewidth=2)
axes[1,0].plot(energy_axis, s_recon_3.inav[center_pos].data, label="3 components", linewidth=2)
axes[1,0].plot(energy_axis, s_recon_opt.inav[center_pos].data, label=f"{n_components_95} components", linewidth=2)
axes[1,0].legend()
axes[1,0].set_title("Spectrum Comparison (Center)")
axes[1,0].set_xlabel("Energy (eV)")
axes[1,0].set_ylabel("Intensity")

# Difference map
diff_map_data = np.mean((s - s_recon_3).data, axis=2)
im3 = axes[1,1].imshow(diff_map_data, cmap='RdBu_r')
axes[1,1].set_title("Difference (Original - 3 comp.)")
plt.colorbar(im3, ax=axes[1,1])

plt.tight_layout()
plt.show()

# Extract and examine individual factors
print(f"\nExamining decomposition factors:")
factors = s.learning_results.factors
loadings = s.learning_results.loadings

print(f"Factors shape: {factors.shape}")  # (n_components, signal_size)
print(f"Loadings shape: {loadings.shape}")  # (nav_x, nav_y, n_components)

# Plot the first 3 factors individually
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i in range(3):
    axes[i].plot(energy_axis, factors[:, i])
    axes[i].set_title(f"Factor {i}")
    axes[i].set_xlabel("Energy (eV)")
    axes[i].set_ylabel("Intensity")
plt.tight_layout()
plt.show()

# Manual reconstruction example
print("\nDemonstrating manual reconstruction...")
# Manually reconstruct using first 2 components
manual_recon = np.zeros_like(s.data)

# The loadings are flattened (nav_size*nav_size, n_components)
loadings_reshaped = loadings.reshape(nav_size, nav_size, -1)

for i in range(nav_size):
    for j in range(nav_size):
        # Reconstruction: sum of (loading * factor) for each component
        for k in range(2):  # Use first 2 components
            manual_recon[i, j, :] += loadings_reshaped[i, j, k] * factors[:, k]

# Create signal from manual reconstruction
s_manual = hs.signals.Signal1D(manual_recon)
s_manual.axes_manager = s.axes_manager.copy()
s_manual.metadata.General.title = "Manual Reconstruction (2 components)"

# Verify it matches the automatic reconstruction
s_auto_2 = s.get_decomposition_model(components=2)
reconstruction_match = np.allclose(s_manual.data, s_auto_2.data)
print(f"Manual vs automatic reconstruction match: {reconstruction_match}")

print("\nDecomposition model reconstruction example completed!")
print("\nKey insights:")
print("- Decomposition separates signal into characteristic spectra (factors) and spatial maps (loadings)")
print("- Reconstruction with fewer components acts as an effective denoising filter")
print("- The number of components can be chosen based on explained variance")
print("- Manual reconstruction helps understand the mathematical basis of the decomposition")
