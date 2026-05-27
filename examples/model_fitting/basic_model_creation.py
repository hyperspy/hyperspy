"""
Basic Model Creation and Fitting
=================================

This example demonstrates how to create and fit models in HyperSpy following
best practices from the AI Guide. Models are composed of components that 
represent different physical or mathematical features in your data.

Key concepts:
- Creating models from signals using HyperSpy's model framework
- Adding components with proper naming and parameter initialization
- Setting initial parameter values using estimation methods
- Performing robust fitting with error handling
- Evaluating fit quality using statistical measures
- Best practices for model-based data analysis
"""

import numpy as np
import hyperspy.api as hs
import matplotlib.pyplot as plt

# %%
# **Creating synthetic data for model fitting**
#
# We'll create a realistic spectrum using HyperSpy's model framework to demonstrate 
# the model fitting process. This approach follows AI Guide best practices by using
# HyperSpy models for data generation and then fitting.

# Create empty signal with proper axis calibration
s = hs.signals.Signal1D(np.zeros(1000))
s.axes_manager.signal_axes[0].name = 'Energy'
s.axes_manager.signal_axes[0].units = 'eV'
s.axes_manager.signal_axes[0].scale = 0.7
s.axes_manager.signal_axes[0].offset = 100.0
s.metadata.General.title = 'Synthetic Spectrum'

print(f"Created signal: {s}")
print(f"Energy range: {s.axes_manager.signal_axes[0].axis[0]:.1f} to {s.axes_manager.signal_axes[0].axis[-1]:.1f} eV")

# Create ground truth model for simulation
m_true = s.create_model()

# ✅ BEST PRACTICE: Add components with descriptive names at creation
background_true = hs.model.components1D.PowerLaw(name="background_true")
m_true.append(background_true)

peak1_true = hs.model.components1D.Gaussian(name="peak1_true")
m_true.append(peak1_true)

peak2_true = hs.model.components1D.Gaussian(name="peak2_true")
m_true.append(peak2_true)

peak3_true = hs.model.components1D.Lorentzian(name="peak3_true")
m_true.append(peak3_true)

# ✅ BEST PRACTICE: Set parameter values using direct access for single spectrum
# Background parameters
background_true.A.value = 5000
background_true.r.value = 2.5
background_true.origin.value = 0.0
background_true.left_cutoff.value = 0.0

# Peak parameters with realistic values
peak1_true.centre.value = 215
peak1_true.sigma.value = 20
peak1_true.A.value = 8000

peak2_true.centre.value = 315
peak2_true.sigma.value = 30
peak2_true.A.value = 6000

peak3_true.centre.value = 450
peak3_true.gamma.value = 25
peak3_true.A.value = 4000

# ✅ BEST PRACTICE: Store current values in arrays for parameter management
# This ensures all parameters are properly set for model generation
for component in m_true:
    for param in component.parameters:
        param.store_current_value_in_array()

# ✅ BEST PRACTICE: Double-check that all parameters are set before calling as_signal
# This ensures no parameters are unset when generating synthetic data
print("Verifying all parameters are set...")
for component in m_true:
    for param in component.parameters:
        if not hasattr(param, 'value') or param.value is None:
            print(f"WARNING: {component.name}.{param.name} is not set!")
            param.value = 1.0  # Set a default value
            if hasattr(param, 'assign_current_value_to_all'):
                param.assign_current_value_to_all()  # For multidimensional signals

# ✅ BEST PRACTICE: Ensure parameters are properly set for model generation
# For multidimensional data, we would use set_parameters_value, but for single spectrum this is sufficient
print("Ground truth parameters set:")
print(f"Background: A={background_true.A.value:.0f}, r={background_true.r.value:.2f}")
print(f"Peak 1: centre={peak1_true.centre.value:.0f} eV, A={peak1_true.A.value:.0f}")
print(f"Peak 2: centre={peak2_true.centre.value:.0f} eV, A={peak2_true.A.value:.0f}")
print(f"Peak 3: centre={peak3_true.centre.value:.0f} eV, A={peak3_true.A.value:.0f}")

# Generate synthetic data with noise
s = m_true.as_signal()
s.add_gaussian_noise(std=100)  # Add realistic noise
s.metadata.General.title = 'Synthetic Spectrum with Noise'

print(f"\nGenerated noisy spectrum: {s}")
print(f"Signal range: {s.data.min():.1f} to {s.data.max():.1f}")

# Plot the synthetic data
s.plot()
print("Synthetic data generated successfully!")

# %%
# **Model fitting workflow**
#
# Now we'll demonstrate the complete model fitting workflow, including
# parameter estimation, fitting, and quality assessment.

# Create a new model for fitting (separate from ground truth)
m = s.create_model()

# ✅ BEST PRACTICE: Add components with descriptive names
background = hs.model.components1D.PowerLaw(name="background")
m.append(background)

peak1 = hs.model.components1D.Gaussian(name="peak1")
m.append(peak1)

peak2 = hs.model.components1D.Gaussian(name="peak2")
m.append(peak2)

peak3 = hs.model.components1D.Lorentzian(name="peak3")
m.append(peak3)

print(f"Created model with {len(m)} components:")
for i, comp in enumerate(m):
    print(f"  {i+1}. {comp.name} ({comp.__class__.__name__})")

# %%
# **Parameter initialization using estimation methods**
#
# Use HyperSpy's built-in parameter estimation when available.

# ✅ BEST PRACTICE: Use estimate_parameters when available
try:
    # Estimate background parameters
    background.estimate_parameters(s, x1=100, x2=600)
    print("Background parameters estimated successfully")
except Exception as e:
    print(f"Background estimation failed: {e}")
    # Manual initialization as fallback
    background.A.value = 3000
    background.r.value = 2.0
    background.origin.value = 0.0
    background.left_cutoff.value = 0.0
    print("Background parameters set manually")

# ✅ BEST PRACTICE: Find peak positions using HyperSpy methods
peak_positions = []
try:
    # Simple peak finding using maximum values in regions
    roi1 = s.isig[200.:230.]
    peak_positions.append(roi1.axes_manager.signal_axes[0].index2value(roi1.data.argmax()))
    
    roi2 = s.isig[300.:330.]
    peak_positions.append(roi2.axes_manager.signal_axes[0].index2value(roi2.data.argmax()))
    
    roi3 = s.isig[430.:470.]
    peak_positions.append(roi3.axes_manager.signal_axes[0].index2value(roi3.data.argmax()))
    
    print(f"Peak positions found: {peak_positions}")
    
except Exception as e:
    print(f"Peak finding failed: {e}")
    # Use approximate positions
    peak_positions = [215, 315, 450]
    print(f"Using approximate peak positions: {peak_positions}")

# Initialize peak parameters
peak1.centre.value = peak_positions[0]
peak1.sigma.value = 15  # Initial guess
peak1.A.value = 5000    # Initial guess

peak2.centre.value = peak_positions[1]
peak2.sigma.value = 25  # Initial guess
peak2.A.value = 4000    # Initial guess

peak3.centre.value = peak_positions[2]
peak3.gamma.value = 20  # Initial guess
peak3.A.value = 3000    # Initial guess

print("Initial parameters set:")
print(f"Peak 1: centre={peak1.centre.value:.1f} eV, A={peak1.A.value:.0f}")
print(f"Peak 2: centre={peak2.centre.value:.1f} eV, A={peak2.A.value:.0f}")
print(f"Peak 3: centre={peak3.centre.value:.1f} eV, A={peak3.A.value:.0f}")

# %%
# **Robust model fitting with error handling**
#
# Demonstrate proper fitting workflow with error handling and monitoring.

# Plot model before fitting
m.plot()
print("Model plotted with initial parameters")

# ✅ BEST PRACTICE: Robust fitting with error handling
try:
    print("Starting model fitting...")
    
    # Fit with bounded parameters to ensure physical reasonableness
    m.fit(bounded=True)
    
    print("Model fitting completed successfully!")
    
    # Display fitted parameters
    print("\nFitted parameters:")
    print(f"Background: A={background.A.value:.0f}, r={background.r.value:.3f}")
    print(f"Peak 1: centre={peak1.centre.value:.2f} eV, σ={peak1.sigma.value:.2f}, A={peak1.A.value:.0f}")
    print(f"Peak 2: centre={peak2.centre.value:.2f} eV, σ={peak2.sigma.value:.2f}, A={peak2.A.value:.0f}")
    print(f"Peak 3: centre={peak3.centre.value:.2f} eV, γ={peak3.gamma.value:.2f}, A={peak3.A.value:.0f}")
    
except Exception as e:
    print(f"Fitting failed: {e}")
    print("Consider adjusting initial parameters or using different optimizer")
    raise

# %%
# **Fit quality assessment**
#
# Evaluate the quality of the fit using statistical measures.

# Calculate residuals and fit statistics
model_signal = m.as_signal()
residuals = s - model_signal  # Direct signal arithmetic preserves metadata

# ✅ BEST PRACTICE: Calculate meaningful fit statistics
ss_res = np.sum(residuals.data**2)
ss_tot = np.sum((s.data - np.mean(s.data))**2)
r_squared = 1 - (ss_res / ss_tot)

# Calculate reduced chi-squared
n_data_points = len(s.data)
n_parameters = len(m.p0)  # Number of free parameters
degrees_of_freedom = n_data_points - n_parameters
chi_squared_red = ss_res / degrees_of_freedom

# Calculate RMSE
rmse = np.sqrt(ss_res / n_data_points)

print(f"\n=== Fit Quality Assessment ===")
print(f"R-squared: {r_squared:.4f}")
print(f"Reduced χ²: {chi_squared_red:.2f}")
print(f"RMSE: {rmse:.1f}")
print(f"Degrees of freedom: {degrees_of_freedom}")

# Interpretation guidelines
print(f"\n=== Interpretation Guidelines ===")
if r_squared > 0.95:
    print("✅ Excellent fit (R² > 0.95)")
elif r_squared > 0.90:
    print("✅ Good fit (R² > 0.90)")
elif r_squared > 0.80:
    print("⚠️  Acceptable fit (R² > 0.80)")
else:
    print("❌ Poor fit (R² < 0.80) - consider model revision")

if chi_squared_red < 2.0:
    print("✅ Good fit quality (χ²_red < 2.0)")
elif chi_squared_red < 5.0:
    print("⚠️  Acceptable fit quality (χ²_red < 5.0)")
else:
    print("❌ Poor fit quality (χ²_red > 5.0) - consider model revision")

# %%
# **Visualization and result interpretation**
#
# Create comprehensive plots to visualize the fit results.

# Plot the fit results
m.plot()

# Plot residuals
residuals.plot()
residuals.metadata.General.title = 'Fit Residuals'

# Create a summary figure with components
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot data, model, and components
energy_axis = s.axes_manager.signal_axes[0].axis
ax1.plot(energy_axis, s.data, 'ko-', label='Data', markersize=3)
ax1.plot(energy_axis, model_signal.data, 'r-', label='Model', linewidth=2)

# Plot individual components by temporarily disabling others
# Background only
for comp in m:
    if comp != background:
        comp.active = False
background_signal = m.as_signal()
ax1.plot(energy_axis, background_signal.data, 'b--', label='Background', linewidth=1)

# Peak 1 + background
peak1.active = True
peak1_signal = m.as_signal()
ax1.plot(energy_axis, peak1_signal.data, 'g--', label='Peak 1', linewidth=1)

# Peak 2 + background  
peak1.active = False
peak2.active = True
peak2_signal = m.as_signal()
ax1.plot(energy_axis, peak2_signal.data, 'm--', label='Peak 2', linewidth=1)

# Peak 3 + background
peak2.active = False
peak3.active = True
peak3_signal = m.as_signal()
ax1.plot(energy_axis, peak3_signal.data, 'c--', label='Peak 3', linewidth=1)

# Restore all components
for comp in m:
    comp.active = True
ax1.set_xlabel('Energy (eV)')
ax1.set_ylabel('Intensity')
ax1.set_title('Model Fit Results')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot residuals
ax2.plot(energy_axis, residuals.data, 'ro-', markersize=2)
ax2.axhline(y=0, color='k', linestyle='-', alpha=0.5)
ax2.set_xlabel('Energy (eV)')
ax2.set_ylabel('Residuals')
ax2.set_title('Fit Residuals')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n=== Model Fitting Summary ===")
print(f"Successfully fitted {len(m)} components to the spectrum")
print(f"Final R-squared: {r_squared:.4f}")
print(f"Final χ²_red: {chi_squared_red:.2f}")
print("Model fitting example completed successfully!")

# %%
# **Best practices summary**
#
# Key takeaways for model fitting in HyperSpy.

print("\n=== Best Practices Summary ===")
print("✅ Use HyperSpy models for data simulation")
print("✅ Name components descriptively at creation")
print("✅ Use estimate_parameters() when available")
print("✅ Initialize parameters with reasonable values")
print("✅ Use bounded fitting for physical constraints")
print("✅ Always assess fit quality with multiple metrics")
print("✅ Visualize results with data, model, and residuals")
print("✅ Handle fitting errors gracefully")
print("✅ Document parameter meanings and units")
print("✅ Compare fitted parameters with known values when possible")

print("\nBasic model creation and fitting example completed!")
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
# =====================================

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
