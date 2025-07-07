"""
Basic Model Creation and Fitting
=================================

This example demonstrates how to create and fit models in HyperSpy.
Model# Calculate R-squared
model_signal = m.as_signal()
residuals = s - model_signal  # Direct signal arithmetic preserves metadata
ss_res = np.sum(residuals.data**2)        # Use NumPy for final scalar calculation
ss_tot = np.sum((s.data - np.mean(s.data))**2)  # Use NumPy for scalar statistics
r_squared = 1 - (ss_res / ss_tot)composed of components that represent different physical or 
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

# %%
# **Creating synthetic data for model fitting**
#
# We'll create a realistic spectrum using HyperSpy's model framework to demonstrate 
# the model fitting process. This approach follows best practices by using
# HyperSpy models for data generation and then fitting.

# Create empty signal with proper axis calibration
s = hs.signals.Signal1D(np.zeros(1000))
s.axes_manager.signal_axes[0].name = 'Energy'
s.axes_manager.signal_axes[0].units = 'eV'
s.axes_manager.signal_axes[0].scale = 0.7
s.axes_manager.signal_axes[0].offset = 100.0
s.metadata.General.title = 'Synthetic Spectrum'

# Create ground truth model for simulation
m_true = s.create_model()

# Add components to ground truth model (using best practice: name at creation)
background_true = hs.model.components1D.PowerLaw(name="background_true")
m_true.append(background_true)

peak1_true = hs.model.components1D.Gaussian(name="peak1_true")
m_true.append(peak1_true)

peak2_true = hs.model.components1D.Gaussian(name="peak2_true")
m_true.append(peak2_true)

peak3_true = hs.model.components1D.Lorentzian(name="peak3_true")
m_true.append(peak3_true)

# Set parameter values using direct access (recommended for single components)
background_true.A.value = 5000
background_true.r.value = 2.5
background_true.origin.value = 0.0
background_true.left_cutoff.value = 0.0

# For simulation, ensure maps are set
background_true.A.map['values'][:] = 5000
background_true.A.map['is_set'][:] = True
background_true.r.map['values'][:] = 2.5
background_true.r.map['is_set'][:] = True
background_true.origin.map['values'][:] = 0.0
background_true.origin.map['is_set'][:] = True
background_true.left_cutoff.map['values'][:] = 0.0
background_true.left_cutoff.map['is_set'][:] = True

peak1_true.centre.value = 215
peak1_true.sigma.value = 20
peak1_true.A.value = 8000

# For simulation, ensure maps are set
peak1_true.centre.map['values'][:] = 215
peak1_true.centre.map['is_set'][:] = True
peak1_true.sigma.map['values'][:] = 20
peak1_true.sigma.map['is_set'][:] = True
peak1_true.A.map['values'][:] = 8000
peak1_true.A.map['is_set'][:] = True

peak2_true.centre.value = 315
peak2_true.sigma.value = 30
peak2_true.A.value = 6000

# For simulation, ensure maps are set
peak2_true.centre.map['values'][:] = 315
peak2_true.centre.map['is_set'][:] = True
peak2_true.sigma.map['values'][:] = 30
peak2_true.sigma.map['is_set'][:] = True
peak2_true.A.map['values'][:] = 6000
peak2_true.A.map['is_set'][:] = True

peak3_true.centre.value = 450
peak3_true.gamma.value = 25
peak3_true.A.value = 4000

# For simulation, ensure maps are set
peak3_true.centre.map['values'][:] = 450
peak3_true.centre.map['is_set'][:] = True
peak3_true.gamma.map['values'][:] = 25
peak3_true.gamma.map['is_set'][:] = True
peak3_true.A.map['values'][:] = 4000
peak3_true.A.map['is_set'][:] = True

# Generate synthetic data from model
s = m_true.as_signal()
s.set_signal_origin("simulation")

# Store ground truth model
m_true.signal = s
s.models.store(m_true, name="ground_truth")

# Add realistic noise
s.change_dtype('float64')
np.random.seed(42)
s.add_gaussian_noise(std=100, random_state=42)

# **Signal created successfully**
#
# Our synthetic spectrum is ready for model fitting with realistic spectroscopic features.
# The ground truth model is stored and can be accessed via s.models.ground_truth.restore()

# %%
# ## Create and configure the model for fitting
# 
# We'll build a new model to fit the synthetic data (separate from ground truth)

# Create new model from noisy signal for fitting
m = s.create_model()

# Add background component (using best practice: name at creation)
background_comp = hs.model.components1D.PowerLaw(name='background')
m.append(background_comp)

# Add peak components (using best practice: name at creation)
peak1_comp = hs.model.components1D.Gaussian(name='peak1')
m.append(peak1_comp)

peak2_comp = hs.model.components1D.Gaussian(name='peak2')
m.append(peak2_comp)

peak3_comp = hs.model.components1D.Lorentzian(name='peak3')
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
peak1_comp.centre.value = 215  # Match ground truth
peak1_comp.sigma.value = 20
peak1_comp.A.value = 8000

# Peak 2 parameters  
peak2_comp.centre.value = 315  # Match ground truth
peak2_comp.sigma.value = 30
peak2_comp.A.value = 6000

# Peak 3 parameters
peak3_comp.centre.value = 450  # Match ground truth
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
model_signal = m.as_signal()
residuals = s - model_signal  # Direct signal arithmetic preserves metadata
ss_res = np.sum(residuals.data**2)        # Use NumPy for final scalar calculation
ss_tot = np.sum((s.data - np.mean(s.data))**2)  # Use NumPy for scalar statistics
r_squared = 1 - (ss_res / ss_tot)

# %%
# Fitted Parameters
# =================
#
# Let's examine how well the model fitted the true parameters

print(f"Background A: {background_comp.A.value:.2f} (true: 5000)")
print(f"Background r: {background_comp.r.value:.2f} (true: 2.5)")
print(f"Peak 1 centre: {peak1_comp.centre.value:.2f} eV (true: 215)")
print(f"Peak 1 sigma: {peak1_comp.sigma.value:.2f} eV (true: 20)")
print(f"Peak 1 amplitude: {peak1_comp.A.value:.2f} (true: 8000)")
print(f"Peak 2 centre: {peak2_comp.centre.value:.2f} eV (true: 315)")
print(f"Peak 2 sigma: {peak2_comp.sigma.value:.2f} eV (true: 30)")
print(f"Peak 2 amplitude: {peak2_comp.A.value:.2f} (true: 6000)")
print(f"Peak 3 centre: {peak3_comp.centre.value:.2f} eV (true: 450)")
print(f"Peak 3 gamma: {peak3_comp.gamma.value:.2f} eV (true: 25)")
print(f"Peak 3 amplitude: {peak3_comp.A.value:.2f} (true: 4000)")

print(f"Goodness of fit (R²): {r_squared:.4f}")

# %%
# Create 2D example
# =================
#
# Now let's demonstrate model creation for 2D signals

# Create 2D synthetic data using HyperSpy models
s2d_empty = hs.signals.Signal2D(np.zeros((100, 100)))
s2d_empty.axes_manager.signal_axes[0].name = 'x'
s2d_empty.axes_manager.signal_axes[1].name = 'y'
s2d_empty.axes_manager.signal_axes[0].scale = 0.1
s2d_empty.axes_manager.signal_axes[1].scale = 0.1
s2d_empty.axes_manager.signal_axes[0].offset = -5.0
s2d_empty.axes_manager.signal_axes[1].offset = -5.0
s2d_empty.metadata.General.title = 'Synthetic 2D Data'

# Create ground truth 2D model
m2d_true = s2d_empty.create_model()

# Add 2D Gaussian component to ground truth (using best practice: name at creation)
gaussian_2d_true = hs.model.components2D.Gaussian2D(name="gaussian_2d_true")
m2d_true.append(gaussian_2d_true)

# Add 2D background using Expression component
background_2d_true = hs.model.components2D.Expression(
    expression="a + b*x + c*y",
    name="background_2d",
    a=100.0,  # constant term
    b=0.0,    # x gradient
    c=0.0     # y gradient
)
m2d_true.append(background_2d_true)

# Set 2D parameter values using direct access
gaussian_2d_true.centre_x.value = 1.0
gaussian_2d_true.centre_y.value = -0.5
gaussian_2d_true.sigma_x.value = 1.5
gaussian_2d_true.sigma_y.value = 1.5
gaussian_2d_true.A.value = 1000

# For 2D simulation, ensure maps are set
gaussian_2d_true.centre_x.map['values'][:] = 1.0
gaussian_2d_true.centre_x.map['is_set'][:] = True
gaussian_2d_true.centre_y.map['values'][:] = -0.5
gaussian_2d_true.centre_y.map['is_set'][:] = True
gaussian_2d_true.sigma_x.map['values'][:] = 1.5
gaussian_2d_true.sigma_x.map['is_set'][:] = True
gaussian_2d_true.sigma_y.map['values'][:] = 1.5
gaussian_2d_true.sigma_y.map['is_set'][:] = True
gaussian_2d_true.A.map['values'][:] = 1000
gaussian_2d_true.A.map['is_set'][:] = True

# Set Expression component parameters
background_2d_true.a.value = 100.0
background_2d_true.b.value = 0.0
background_2d_true.c.value = 0.0

# For 2D simulation, ensure maps are set
background_2d_true.a.map['values'][:] = 100.0
background_2d_true.a.map['is_set'][:] = True
background_2d_true.b.map['values'][:] = 0.0
background_2d_true.b.map['is_set'][:] = True
background_2d_true.c.map['values'][:] = 0.0
background_2d_true.c.map['is_set'][:] = True

# Generate 2D simulation
s2d = m2d_true.as_signal()
s2d.set_signal_origin("simulation")

# Store ground truth
m2d_true.signal = s2d
s2d.models.store(m2d_true, name="ground_truth")

# Add noise to 2D data
s2d.change_dtype('float64')
s2d.add_gaussian_noise(std=50, random_state=42)

print(f"Created 2D signal: {s2d}")

# Create 2D model
m2d = s2d.create_model()

# Add 2D Gaussian component (using best practice: name at creation)
gaussian_2d_comp = hs.model.components2D.Gaussian2D(name='gaussian_2d')
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
# Visualization using HyperSpy plotting
# =============

print("\nCreating visualization...")

# Plot 1D fit using HyperSpy's model plotting capabilities
m.plot()

# Plot 2D data and fit using HyperSpy's plot_images
hs.plot.plot_images([s2d, m2d.as_signal()],
                   label=['2D Data', '2D Model Fit'],
                   cmap='viridis',
                   colorbar=True)

# %%
# Summary
# =======
#
# Model fitting results and key takeaways

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
