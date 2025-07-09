"""
Different Decomposition Algorithms Example
==========================================

This example compares different decomposition algorithms available in HyperSpy:
PCA, ICA, NMF, and demonstrates their strengths and use cases for different
types of data analysis.

"""

import numpy as np
import hyperspy.api as hs
import matplotlib.pyplot as plt

# Create synthetic dataset with known components
np.random.seed(456)

print("Creating synthetic dataset for algorithm comparison...")

# Parameters
nav_size = 15
sig_size = 150
energy_axis = np.linspace(0, 15, sig_size)

# Define three physically meaningful spectral components

# Component 1: Sharp characteristic X-ray peak
comp1_spectrum = 20 * np.exp(-((energy_axis - 4)**2) / 0.05)

# Component 2: Exponential background decay  
comp2_spectrum = 10 * np.exp(-energy_axis / 3) + 2

# Component 3: Broader characteristic peak
comp3_spectrum = 15 * np.exp(-((energy_axis - 10)**2) / 0.4)

# Create realistic spatial distributions
x, y = np.meshgrid(np.linspace(0, 1, nav_size), np.linspace(0, 1, nav_size), indexing='ij')

# Spatial map 1: Particle-like distribution
map1 = np.exp(-((x - 0.3)**2 + (y - 0.7)**2) / 0.05) + \
       0.5 * np.exp(-((x - 0.8)**2 + (y - 0.2)**2) / 0.03)

# Spatial map 2: Background (everywhere but stronger at edges)
map2 = 0.5 + 0.5 * (x**2 + y**2)

# Spatial map 3: Interface-like distribution (diagonal band)
map3 = np.exp(-((x + y - 1)**2) / 0.1)

# Ensure non-negative mixing (important for NMF)
map1 = np.abs(map1)
map2 = np.abs(map2) 
map3 = np.abs(map3)

# Create full dataset
data = np.zeros((nav_size, nav_size, sig_size))
for i in range(nav_size):
    for j in range(nav_size):
        # Mix components with non-negative coefficients
        clean_signal = (map1[i, j] * comp1_spectrum + 
                       map2[i, j] * comp2_spectrum + 
                       map3[i, j] * comp3_spectrum)
        
        # Add moderate noise
        noise_level = 0.15 * np.max(clean_signal)
        noisy_signal = clean_signal + noise_level * np.random.random(sig_size)
        
        # Ensure non-negative (realistic for spectroscopy data)
        data[i, j, :] = np.maximum(noisy_signal, 0)

# Create HyperSpy signal
s = hs.signals.Signal1D(data)

# Set axes
s.axes_manager.navigation_axes[0].name = 'x'
s.axes_manager.navigation_axes[0].units = 'μm'
s.axes_manager.navigation_axes[0].scale = 0.2

s.axes_manager.navigation_axes[1].name = 'y'
s.axes_manager.navigation_axes[1].units = 'μm'
s.axes_manager.navigation_axes[1].scale = 0.2

s.axes_manager.signal_axes[0].name = 'Energy'
s.axes_manager.signal_axes[0].units = 'keV'
s.axes_manager.signal_axes[0].scale = 0.1

s.metadata.General.title = "Synthetic EDS Spectrum Image"
s.metadata.Signal.signal_type = "EDS_TEM"

print(f"Signal shape: {s.data.shape}")
print(f"Signal: {s}")

# Apply different decomposition algorithms
algorithms = ['SVD', 'NMF']  # ICA requires preprocessing with SVD first
results = {}

for alg in algorithms:
    print(f"\n=== Applying {alg} decomposition ===")
    
    # Create a copy for each algorithm
    s_copy = s.deepcopy()
    
    if alg == 'SVD':
        # Singular Value Decomposition - HyperSpy's default PCA implementation
        s_copy.decomposition(algorithm='SVD', output_dimension=5)
        
    elif alg == 'NMF':
        # Non-negative Matrix Factorization - enforces non-negativity
        try:
            s_copy.decomposition(algorithm='NMF', output_dimension=3)
        except ImportError:
            print("  NMF requires scikit-learn. Skipping...")
            continue
    
    results[alg] = s_copy

# Compare decomposition results
print("\n=== Comparing Decomposition Results ===")

# Plot factors comparison
fig, axes = plt.subplots(len(results), 3, figsize=(15, 4*len(results)))
if len(results) == 1:
    axes = axes.reshape(1, -1)

for i, (alg, signal) in enumerate(results.items()):
    for j in range(3):
        if alg == 'ICA':
            factors = signal.learning_results.bss_factors
        else:
            factors = signal.learning_results.factors
        
        # factors is shape (n_features, n_components) in HyperSpy
        if j < factors.shape[1]:
            axes[i, j].plot(energy_axis, factors[:, j])
            axes[i, j].set_title(f"{alg} - Component {j}")
            axes[i, j].set_xlabel("Energy (keV)")
            axes[i, j].set_ylabel("Intensity")
        else:
            axes[i, j].set_visible(False)

plt.tight_layout()
plt.show()

# Plot loadings comparison
print("\nPlotting spatial distributions (loadings)...")
for alg, signal in results.items():
    print(f"\nShowing {alg} spatial loadings:")
    if alg == 'ICA':
        signal.plot_bss_loadings(comp_ids=3, same_window=False)
    else:
        signal.plot_decomposition_loadings(comp_ids=3, same_window=False)

# Quantitative comparison: reconstruction error
print(f"\n=== Reconstruction Quality Comparison ===")

for alg, signal in results.items():
    # Reconstruct with 3 components
    if alg == 'ICA':
        # For ICA, use BSS reconstruction
        reconstructed = signal.get_bss_model(components=3)
    else:
        reconstructed = signal.get_decomposition_model(components=3)
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((s.data - reconstructed.data)**2))
    
    # Calculate correlation coefficient
    correlation = np.corrcoef(s.data.flatten(), reconstructed.data.flatten())[0, 1]
    
    print(f"{alg:>3}: RMSE = {rmse:.4f}, Correlation = {correlation:.4f}")

# Demonstrate specific use cases
print(f"\n=== Algorithm-Specific Analysis ===")

if 'PCA' in results:
    pca_signal = results['PCA']
    print("\nPCA Analysis:")
    print("- Best for: Data compression, noise reduction, identifying main variance sources")
    print("- Components are orthogonal and ordered by explained variance")
    
    # Show cumulative explained variance
    explained_var = pca_signal.learning_results.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    print(f"- Cumulative explained variance: {cumulative_var[:5]}")

if 'ICA' in results:
    ica_signal = results['ICA'] 
    print("\nICA Analysis:")
    print("- Best for: Separating mixed signals, blind source separation")
    print("- Components are statistically independent")
    print("- Often produces more interpretable physical components")

if 'NMF' in results:
    nmf_signal = results['NMF']
    print("\nNMF Analysis:")
    print("- Best for: Non-negative data (spectra, images), parts-based decomposition")
    print("- Enforces non-negativity constraint - more physically realistic")
    print("- Components often correspond to actual physical/chemical phases")

# Show original true components for comparison
print(f"\n=== Original Components (Ground Truth) ===")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

true_components = [comp1_spectrum, comp2_spectrum, comp3_spectrum]
component_names = ["Sharp Peak", "Background", "Broad Peak"]

for i, (comp, name) in enumerate(zip(true_components, component_names)):
    axes[i].plot(energy_axis, comp, 'k-', linewidth=2, label="True")
    axes[i].set_title(f"True Component {i}: {name}")
    axes[i].set_xlabel("Energy (keV)")
    axes[i].set_ylabel("Intensity")
    axes[i].legend()

plt.tight_layout()
plt.show()

print("\nDifferent decomposition algorithms example completed!")
print("\nAlgorithm Summary:")
print("• PCA: Linear, orthogonal, variance-based. Good for data compression and noise reduction.")
print("• ICA: Non-linear, independent components. Good for blind source separation.")  
print("• NMF: Non-negative, parts-based. Good for interpretable, physically meaningful decomposition.")
print("\nChoose algorithm based on:")
print("- Data characteristics (negative values present?)")
print("- Physical assumptions (orthogonality, independence, non-negativity)")
print("- Analysis goals (compression, source separation, physical interpretation)")
