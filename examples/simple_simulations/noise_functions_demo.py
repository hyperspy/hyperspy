"""
Noise Functions in HyperSpy
============================

This example demonstrates HyperSpy's built-in noise functions for adding
realistic noise to signals, essential for simulation and testing.

"""

import hyperspy.api as hs
import numpy as np

# %%
# ## Creating clean test signals
# Start with clean signals to demonstrate noise addition

# Signal for Gaussian noise (requires float dtype)
clean_signal = hs.signals.Signal1D(np.ones(100, dtype='float64') * 1000)
clean_signal.axes_manager[0].scale = 0.1
clean_signal.axes_manager[0].units = 'eV'
clean_signal.axes_manager[0].name = 'Energy'

# Signal for Poissonian noise (works with any numeric dtype)
intensity_signal = hs.signals.Signal1D(np.ones(100, dtype='int32') * 50)
intensity_signal.axes_manager[0].scale = 0.1
intensity_signal.axes_manager[0].units = 'eV'
intensity_signal.axes_manager[0].name = 'Energy'

print("Original signals created")
print(f"Clean signal - dtype: {clean_signal.data.dtype}, mean: {clean_signal.data.mean():.1f}")
print(f"Intensity signal - dtype: {intensity_signal.data.dtype}, mean: {intensity_signal.data.mean():.1f}")

# %%
# ## Adding Gaussian noise
# Gaussian noise is additive and requires float data type

# Make a copy to preserve original
gaussian_noisy = clean_signal.copy()
print(f"\nBefore Gaussian noise: {gaussian_noisy.data[:5]}")

# Add Gaussian noise with standard deviation of 50
gaussian_noisy.add_gaussian_noise(std=50, random_state=42)
print(f"After Gaussian noise (std=50): {gaussian_noisy.data[:5]}")
print(f"Noise standard deviation: {gaussian_noisy.data.std():.1f}")

# %%
# ## Different Gaussian noise levels
# Demonstrate various noise levels

noise_levels = [10, 50, 100]
gaussian_signals = []

for std in noise_levels:
    noisy = clean_signal.copy()
    noisy.add_gaussian_noise(std=std, random_state=42)
    gaussian_signals.append(noisy)
    print(f"Gaussian noise std={std}: signal std={noisy.data.std():.1f}")

# %%
# ## Adding Poissonian noise
# Poissonian noise follows Poisson distribution, good for counting statistics

# Make a copy to preserve original  
poisson_noisy = intensity_signal.copy()
print(f"\nBefore Poissonian noise: {poisson_noisy.data[:5]}")

# Add Poissonian noise (in-place operation)
poisson_noisy.add_poissonian_noise(random_state=42)
print(f"After Poissonian noise: {poisson_noisy.data[:5]}")
print(f"Original mean: {intensity_signal.data.mean():.1f}")
print(f"Noisy mean: {poisson_noisy.data.mean():.1f}")

# %%
# ## Poissonian noise with different intensities
# Higher intensities lead to better signal-to-noise ratio

intensities = [10, 50, 200]
poisson_signals = []

for intensity in intensities:
    signal = hs.signals.Signal1D(np.ones(100, dtype='float64') * intensity)
    signal.add_poissonian_noise(random_state=42)
    poisson_signals.append(signal)
    
    # For Poisson distribution, variance equals mean
    theoretical_std = np.sqrt(intensity)
    actual_std = signal.data.std()
    print(f"Intensity={intensity}: theoretical std={theoretical_std:.1f}, actual std={actual_std:.1f}")

# %%
# ## Poissonian noise with float data
# Poissonian noise works with float data too

float_signal = hs.signals.Signal1D(np.ones(100, dtype='float64') * 50.5)
print(f"\nFloat signal before Poissonian: {float_signal.data[:5]}")
float_signal.add_poissonian_noise(random_state=42)
print(f"Float signal after Poissonian: {float_signal.data[:5]}")

# %%
# ## Combining both noise types
# Realistic signals often have both sources of noise

# Start with a structured signal
x = np.linspace(0, 10, 100)
structured_data = 1000 + 500 * np.exp(-(x-5)**2/2)  # Gaussian peak
combined_signal = hs.signals.Signal1D(structured_data)
combined_signal.change_dtype('float64')  # Use HyperSpy method

print(f"\nStructured signal peak: {combined_signal.data.max():.1f}")

# Add Poissonian noise first (shot noise from detection)
combined_signal.add_poissonian_noise(random_state=42)
print(f"After Poissonian: peak={combined_signal.data.max():.1f}")

# Then add Gaussian noise (electronic noise)
combined_signal.add_gaussian_noise(std=30, random_state=43)
print(f"After both noises: peak={combined_signal.data.max():.1f}")

# %%
# ## Error handling - Gaussian noise with integer data
# Gaussian noise requires float data type

int_signal = hs.signals.Signal1D(np.ones(10, dtype='int32') * 100)
print(f"\nTrying Gaussian noise on integer data (dtype: {int_signal.data.dtype}):")

try:
    int_signal.add_gaussian_noise(std=10)
    print("No error - unexpected!")
except TypeError as e:
    print(f"Expected TypeError: {e}")
    
    # Fix by changing dtype
    int_signal.change_dtype('float64')
    print(f"Changed to dtype: {int_signal.data.dtype}")
    int_signal.add_gaussian_noise(std=10, random_state=42)
    print("Gaussian noise added successfully after dtype change")

# %%
# ## Random state for reproducibility
# Both functions accept random_state parameter for reproducible results

print("\nReproducibility with random_state:")

# Same random state gives identical results
s1 = hs.signals.Signal1D(np.ones(5, dtype='float64') * 100)
s2 = hs.signals.Signal1D(np.ones(5, dtype='float64') * 100)

s1.add_gaussian_noise(std=10, random_state=42)
s2.add_gaussian_noise(std=10, random_state=42)

print(f"Signal 1: {s1.data}")
print(f"Signal 2: {s2.data}")
print(f"Identical: {np.allclose(s1.data, s2.data)}")

# %%
# ## In-place operation behavior
# Both noise functions modify the signal data in-place

original = hs.signals.Signal1D(np.ones(5, dtype='float64') * 100)
original_data_id = id(original.data)

print(f"\nBefore noise: {original.data}")
print(f"Data object ID: {original_data_id}")

original.add_gaussian_noise(std=5, random_state=42)

print(f"After noise: {original.data}")
print(f"Data object ID: {id(original.data)}")
print(f"Same data object: {id(original.data) == original_data_id}")

# %%
# ## Best practices for noise simulation

print("\nBest practices for adding noise:")
print("✓ Use add_poissonian_noise() for shot noise (counting statistics)")
print("✓ Use add_gaussian_noise() for electronic/thermal noise")
print("✓ Ensure float dtype for Gaussian noise")
print("✓ Add Poissonian noise before Gaussian noise for realism")
print("✓ Use random_state parameter for reproducible simulations")
print("✓ Consider signal intensity levels - higher intensity = better SNR")
print("✓ Both functions work in-place - make copies if needed")

# %%
# ## Practical simulation example
# Simulate a realistic EELS spectrum with both noise sources

print("\nPractical example: Simulated EELS spectrum")

# Create core-loss edge
energy = np.linspace(400, 600, 200)  
edge_onset = 450
edge_intensity = 1000
background = 500

# Power-law background + edge
spectrum_data = background * (energy/energy[0])**(-2)  # Background  
edge_data = edge_intensity * np.exp(-(energy - edge_onset)**2 / (2 * 10**2))  # Edge
spectrum_data += edge_data * (energy >= edge_onset)

# Create signal
eels_spectrum = hs.signals.Signal1D(spectrum_data)
eels_spectrum.change_dtype('float64')  # Use HyperSpy method
eels_spectrum.axes_manager[0].axis = energy
eels_spectrum.axes_manager[0].units = 'eV'
eels_spectrum.axes_manager[0].name = 'Energy Loss'

print(f"Simulated spectrum - intensity range: {eels_spectrum.data.min():.0f} to {eels_spectrum.data.max():.0f}")

# Add realistic noise
eels_spectrum.add_poissonian_noise(random_state=42)  # Shot noise
eels_spectrum.add_gaussian_noise(std=20, random_state=43)  # Electronic noise

print(f"After noise - intensity range: {eels_spectrum.data.min():.0f} to {eels_spectrum.data.max():.0f}")
print("Realistic noisy EELS spectrum created!")
