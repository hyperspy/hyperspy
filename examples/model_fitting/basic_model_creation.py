"""
Basic Model Creation and Fitting
=================================

This example demonstrates how to create and fit models in HyperSpy.
Models are composed of components that represent different physical or 
mathematical features in your data.

Key concepts:
- Creating models from signals
- Adding components to models
- Setting initial parameter values
- Performing the fit
- Evaluating fit quality
"""

import numpy as np
import hyperspy.api as hs
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for examples
import matplotlib.pyplot as plt

# %%
# **Creating synthetic data for model fitting**
#
# We'll create a realistic spectrum with multiple components to demonstrate 
# the model fitting process. This synthetic data includes:
# - A power law background (typical in many analytical datasets)
# - Multiple peaks (Gaussian and Lorentzian)
# - Realistic noise levels

# Create energy axis
energy = np.linspace(100, 800, 1000)

# Create synthetic spectrum with multiple components
np.random.seed(42)

# Background (power law)
background = 5000 * energy**(-2.5)

# Peak 1 (Gaussian)
peak1 = 8000 * np.exp(-0.5 * ((energy - 250) / 20)**2)

# Peak 2 (Gaussian) 
peak2 = 6000 * np.exp(-0.5 * ((energy - 450) / 30)**2)

# Peak 3 (Lorentzian)
peak3 = 4000 * (25**2) / ((energy - 600)**2 + 25**2)

# Add noise
noise = np.random.normal(0, 100, len(energy))

# Combine all components
signal_data = background + peak1 + peak2 + peak3 + noise

# Create HyperSpy signal
s = hs.signals.Signal1D(signal_data)
s.axes_manager.signal_axes[0].name = 'Energy'
s.axes_manager.signal_axes[0].units = 'eV'
s.axes_manager.signal_axes[0].scale = 0.7
s.axes_manager.signal_axes[0].offset = 100.0
s.metadata.General.title = 'Synthetic Spectrum'

# **Signal created successfully**
#
# Our synthetic spectrum is ready for model fitting with realistic spectroscopic features.

# %%
# ## Create and configure the model
# 
# We'll build a model with multiple components to fit the synthetic data

# Create model from signal
m = s.create_model()

# Add background component
background_comp = hs.model.components1D.PowerLaw()
background_comp.name = 'background'
m.append(background_comp)

# Add peak components
peak1_comp = hs.model.components1D.Gaussian()
peak1_comp.name = 'peak1'
m.append(peak1_comp)

peak2_comp = hs.model.components1D.Gaussian()
peak2_comp.name = 'peak2'
m.append(peak2_comp)

peak3_comp = hs.model.components1D.Lorentzian()
peak3_comp.name = 'peak3'
m.append(peak3_comp)

# %%
# **Model structure and components**
#
# The model now contains four components that correspond to our synthetic data:
# - **Background**: PowerLaw component for the continuum
# - **Peak 1**: Gaussian component for sharp symmetric peaks  
# - **Peak 2**: Another Gaussian component
# - **Peak 3**: Lorentzian component for broader asymmetric features

# %%
# ## Set initial parameter estimates
# 
# Good initial guesses improve fitting convergence and accuracy

# Background parameters
background_comp.A.value = 5000
background_comp.r.value = 2.5

# Peak 1 parameters
peak1_comp.centre.value = 250
peak1_comp.sigma.value = 20
peak1_comp.A.value = 8000

# Peak 2 parameters  
peak2_comp.centre.value = 450
peak2_comp.sigma.value = 30
peak2_comp.A.value = 6000

# Peak 3 parameters
peak3_comp.centre.value = 600
peak3_comp.gamma.value = 25
peak3_comp.A.value = 4000

# **Initial parameters configured**
#
# Setting reasonable initial values helps the optimization algorithm converge 
# to the correct solution more reliably.

# %%
# **Performing the model fit**
#
# The fitting process uses non-linear least squares optimization to find 
# the best parameters that minimize the difference between model and data.

# Fit the model
m.fit()

# **Fit completed successfully!**
#
# The optimization algorithm has found the best-fit parameters for all components.

# %%
# Evaluate fit quality
# ====================

# Calculate R-squared
model_values = m.as_signal().data
residuals = s.data - model_values
ss_res = np.sum(residuals**2)
ss_tot = np.sum((s.data - np.mean(s.data))**2)
r_squared = 1 - (ss_res / ss_tot)

print("\nFitted Parameters:")
print("==================")
print(f"Background A: {background_comp.A.value:.2f} (true: 5000)")
print(f"Background r: {background_comp.r.value:.2f} (true: 2.5)")
print(f"Peak 1 centre: {peak1_comp.centre.value:.2f} eV (true: 250)")
print(f"Peak 1 sigma: {peak1_comp.sigma.value:.2f} eV (true: 20)")
print(f"Peak 1 amplitude: {peak1_comp.A.value:.2f} (true: 8000)")
print(f"Peak 2 centre: {peak2_comp.centre.value:.2f} eV (true: 450)")
print(f"Peak 2 sigma: {peak2_comp.sigma.value:.2f} eV (true: 30)")
print(f"Peak 2 amplitude: {peak2_comp.A.value:.2f} (true: 6000)")
print(f"Peak 3 centre: {peak3_comp.centre.value:.2f} eV (true: 600)")
print(f"Peak 3 gamma: {peak3_comp.gamma.value:.2f} eV (true: 25)")
print(f"Peak 3 amplitude: {peak3_comp.A.value:.2f} (true: 4000)")

print(f"\nGoodness of fit (R²): {r_squared:.4f}")

# %%
# Create 2D example
# =================

print("\nCreating 2D example...")

# Create 2D synthetic data
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

# 2D Gaussian peak
data_2d = 1000 * np.exp(-((X - 1)**2 + (Y + 0.5)**2) / (2 * 1.5**2)) + \
          100 + 50 * np.random.random((100, 100))

s2d = hs.signals.Signal2D(data_2d)
s2d.axes_manager.signal_axes[0].name = 'x'
s2d.axes_manager.signal_axes[1].name = 'y'
s2d.axes_manager.signal_axes[0].scale = 0.1
s2d.axes_manager.signal_axes[1].scale = 0.1
s2d.metadata.General.title = 'Synthetic 2D Data'

print(f"Created 2D signal: {s2d}")

# Create 2D model
m2d = s2d.create_model()

# Add 2D Gaussian component
gaussian_2d_comp = hs.model.components2D.Gaussian2D()
gaussian_2d_comp.name = 'gaussian_2d'
m2d.append(gaussian_2d_comp)

# Add 2D constant background using Expression component
background_2d_comp = hs.model.components2D.Expression(
    expression="a + b*x + c*y",
    name="background_2d",
    a=100.0,  # constant term
    b=0.0,    # x gradient
    c=0.0     # y gradient
)
m2d.append(background_2d_comp)

# Set initial guesses for 2D model
gaussian_2d_comp.centre_x.value = 1.0
gaussian_2d_comp.centre_y.value = -0.5
gaussian_2d_comp.sigma_x.value = 1.5
gaussian_2d_comp.sigma_y.value = 1.5
gaussian_2d_comp.A.value = 1000

# Fit the 2D model
print("Fitting 2D model...")
m2d.fit()

print("\n2D Fitted Parameters:")
print("====================")
print(f"Gaussian centre X: {gaussian_2d_comp.centre_x.value:.2f} (true: 1.0)")
print(f"Gaussian centre Y: {gaussian_2d_comp.centre_y.value:.2f} (true: -0.5)")
print(f"Gaussian sigma X: {gaussian_2d_comp.sigma_x.value:.2f} (true: 1.5)")
print(f"Gaussian sigma Y: {gaussian_2d_comp.sigma_y.value:.2f} (true: 1.5)")
print(f"Gaussian amplitude: {gaussian_2d_comp.A.value:.2f} (true: 1000)")

# %%
# Visualization
# =============

print("\nCreating visualization...")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 1D fit
axes[0, 0].plot(s.axes_manager.signal_axes[0].axis, s.data, 'b-', 
                label='Data', alpha=0.7, linewidth=1)
axes[0, 0].plot(s.axes_manager.signal_axes[0].axis, m.as_signal().data, 'r-', 
                label='Total fit', linewidth=2)
axes[0, 0].set_xlabel('Energy (eV)')
axes[0, 0].set_ylabel('Intensity')
axes[0, 0].set_title('1D Model Fit')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Individual components
axes[0, 1].plot(s.axes_manager.signal_axes[0].axis, s.data, 'k-', 
                label='Data', alpha=0.7, linewidth=1)
for component in m:
    # Use function() method to get component values
    comp_values = component.function(s.axes_manager.signal_axes[0].axis)
    axes[0, 1].plot(s.axes_manager.signal_axes[0].axis, comp_values, 
                    '--', label=component.name, linewidth=2)
axes[0, 1].set_xlabel('Energy (eV)')
axes[0, 1].set_ylabel('Intensity')
axes[0, 1].set_title('Individual Model Components')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 2D data
im1 = axes[1, 0].imshow(s2d.data, extent=[-5, 5, -5, 5], origin='lower', cmap='viridis')
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('y')
axes[1, 0].set_title('2D Data')
plt.colorbar(im1, ax=axes[1, 0], label='Intensity')

# 2D fit
im2 = axes[1, 1].imshow(m2d.as_signal().data, extent=[-5, 5, -5, 5], origin='lower', cmap='viridis')
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('y')
axes[1, 1].set_title('2D Model Fit')
plt.colorbar(im2, ax=axes[1, 1], label='Intensity')

plt.tight_layout()
plt.savefig('model_fitting_results.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Summary
# =======

print("\n" + "="*60)
print("MODEL FITTING SUMMARY")
print("="*60)

print(f"""
1D MODEL RESULTS:
• Signal: {s}
• Components: {len(m)} (Power law + 2 Gaussians + 1 Lorentzian)
• R²: {r_squared:.4f}
• Peak positions: {peak1_comp.centre.value:.1f}, {peak2_comp.centre.value:.1f}, {peak3_comp.centre.value:.1f} eV

2D MODEL RESULTS:
• Signal: {s2d}
• Components: {len(m2d)} (2D Gaussian + Expression background)
• Peak center: ({gaussian_2d_comp.centre_x.value:.2f}, {gaussian_2d_comp.centre_y.value:.2f})

AVAILABLE COMPONENTS:
1D Components:
• Gaussian, Lorentzian, Voigt, PseudoVoigt
• PowerLaw, Exponential, Polynomial
• Arctan, Erf, Logistic, and more

2D Components:
• Gaussian2D, Expression

NEXT STEPS:
• Explore parameter bounds and constraints
• Try different optimization algorithms  
• Use smart fitting strategies for complex models
• Apply to real experimental data
• Investigate spectrum image fitting for spatial analysis

OUTPUT FILES:
• model_fitting_results.png
""")

print("Model creation and fitting demonstration complete!")
