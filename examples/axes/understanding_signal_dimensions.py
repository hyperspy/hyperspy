"""
Signal Dimensions Basics
========================

This example demonstrates the core concept of navigation vs signal dimensions 
in HyperSpy through simple, visual examples.
"""

import numpy as np
import hyperspy.api as hs
import matplotlib.pyplot as plt

# %%
# Create different types of signals to show dimension concepts

# Single spectrum: 0D navigation + 1D signal  
energy = np.linspace(0, 100, 200)
spectrum_data = 1000 * np.exp(-((energy - 50)**2) / 200) + 100
spectrum = hs.signals.Signal1D(spectrum_data)
spectrum.axes_manager[0].name = 'Energy'
spectrum.axes_manager[0].units = 'eV'

# Spectrum image: 2D navigation + 1D signal
spectrum_image_data = np.random.random((8, 10, 200))
spectrum_image = hs.signals.Signal1D(spectrum_image_data)
spectrum_image.axes_manager[0].name = 'y'
spectrum_image.axes_manager[1].name = 'x'
spectrum_image.axes_manager[2].name = 'Energy'

# Image stack: 1D navigation + 2D signal  
image_stack_data = np.random.random((15, 64, 64))
image_stack = hs.signals.Signal2D(image_stack_data)
image_stack.axes_manager[0].name = 'time'
image_stack.axes_manager[1].name = 'x'
image_stack.axes_manager[2].name = 'y'

# %%
# Visualize the dimension concepts

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Single spectrum
axes[0, 0].plot(energy, spectrum_data, 'b-', linewidth=2)
axes[0, 0].set_xlabel('Energy (eV)')
axes[0, 0].set_ylabel('Intensity')
axes[0, 0].set_title('Single Spectrum\n(0D nav + 1D signal)')
axes[0, 0].grid(True, alpha=0.3)

# Spectrum image navigator - simplified to avoid potential issues
navigator_data = np.sum(spectrum_image_data, axis=2)  # Sum along energy axis
axes[0, 1].imshow(navigator_data, origin='lower', cmap='viridis')
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('y') 
axes[0, 1].set_title('Spectrum Image Navigator\n(2D nav + 1D signal)')

# Single image from stack
axes[1, 0].imshow(image_stack_data[7], origin='lower', cmap='gray')
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('y')
axes[1, 0].set_title('Image from Stack\n(1D nav + 2D signal)')

# Show shapes as text
axes[1, 1].text(0.1, 0.8, f'Spectrum: {spectrum.data.shape}', fontsize=12, transform=axes[1, 1].transAxes)
axes[1, 1].text(0.1, 0.6, f'Spectrum Image: {spectrum_image.data.shape}', fontsize=12, transform=axes[1, 1].transAxes)
axes[1, 1].text(0.1, 0.4, f'Image Stack: {image_stack.data.shape}', fontsize=12, transform=axes[1, 1].transAxes)
axes[1, 1].set_title('Data Shapes')
axes[1, 1].set_xlim(0, 1)
axes[1, 1].set_ylim(0, 1)
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()

# %%
# **Signal Dimension Examples Summary**
#
# Understanding how HyperSpy interprets signal dimensions is fundamental to effective data analysis.

# **Spectrum (0D navigation + 1D signal):**
# - Navigation shape: () - no scanning dimensions  
# - Signal shape: (200,) - energy/spectral dimension
# - Interpretation: Single spectrum measurement

# **Spectrum Image (2D navigation + 1D signal):**  
# - Navigation shape: (8, 10) - spatial scanning grid
# - Signal shape: (200,) - energy/spectral dimension  
# - Interpretation: Spectrum acquired at each spatial position

# **Image Stack (1D navigation + 2D signal):**
# - Navigation shape: (15,) - time series or depth dimension
# - Signal shape: (32, 32) - 2D image at each navigation point
# - Interpretation: Series of images (e.g., time-lapse, depth series)
