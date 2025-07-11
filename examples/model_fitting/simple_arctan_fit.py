"""
Simple arctan fit
=================

Fit an arctan function.

"""

import numpy as np
import hyperspy.api as hs

# %%
# Generate the data and create spectrum
# -------------------------------------
# Generate the data and make the spectrum
data = np.arctan(np.arange(-500, 500))
s = hs.signals.Signal1D(data)
s.axes_manager[0].offset = -500
s.axes_manager[0].units = ""
s.axes_manager[0].name = "x"
s.metadata.General.title = "Simple arctan fit"
s.set_signal_origin("simulation")

s.add_gaussian_noise(0.1)

# %%
# Create model and add arctan component
# -------------------------------------
# Make the arctan component for use in the model (using best practice: name at creation)
arctan_component = hs.model.components1D.Arctan(name="Arctan_Fit")

# Create the model and add the arctan component
m = s.create_model()
m.append(arctan_component)

# %%
# Fit the model and display results
# ---------------------------------
# Fit the arctan component to the spectrum
m.fit()

# Print the result of the fit
m.print_current_values()

# Plot the spectrum and the model fitting
m.plot()

# %%
# ## Fit Quality Assessment
# 
# Evaluate the quality of the fit using multiple metrics.

print("\n📊 Fit Quality Assessment")
print("=" * 50)

# Calculate R-squared (coefficient of determination)
model_signal = m.as_signal()
residuals = s - model_signal
ss_res = np.sum(residuals.data**2)
ss_tot = np.sum((s.data - np.mean(s.data))**2)
r_squared = 1 - (ss_res / ss_tot)

print(f"R-squared: {r_squared:.4f}")
print(f"Root Mean Square Error: {np.sqrt(ss_res / len(s.data)):.4f}")

# Calculate chi-squared statistic
chi_squared = np.sum((residuals.data / np.sqrt(np.abs(s.data)))**2)
reduced_chi_squared = chi_squared / (len(s.data) - len(m.p0))

print(f"Chi-squared: {chi_squared:.2f}")
print(f"Reduced chi-squared: {reduced_chi_squared:.4f}")

# Residual analysis
print(f"Mean residual: {np.mean(residuals.data):.4f}")
print(f"Std residual: {np.std(residuals.data):.4f}")

# %%
# ## Parameter Uncertainty Analysis
# 
# Estimate uncertainty in fitted parameters.

print("\n🔍 Parameter Uncertainty Analysis")
print("=" * 50)

# Print parameter values with their standard errors
for component in m:
    print(f"\n{component.name} parameters:")
    for param in component.parameters:
        if param.free:
            print(f"  {param.name}: {param.value:.4f} ± {param.std:.4f}")
        else:
            print(f"  {param.name}: {param.value:.4f} (fixed)")

# %%
# ## Memory Monitoring
# 
# Monitor memory usage during fitting.

print("\n💾 Memory Usage")
print("=" * 50)

print(f"Original signal memory: {s.data.nbytes / 1024**2:.2f} MB")
print(f"Model signal memory: {model_signal.data.nbytes / 1024**2:.2f} MB")
print(f"Residuals memory: {residuals.data.nbytes / 1024**2:.2f} MB")

# %%
# ## Visualization with HyperSpy Methods
# 
# Use HyperSpy's plotting capabilities instead of matplotlib.

print("\n🎨 Enhanced Visualization")
print("=" * 50)

# Plot using HyperSpy's model plotting (replaces plt.show())
m.plot()

# Create residuals plot
residuals.metadata.General.title = "Fit Residuals"
residuals.plot()

print("✅ Enhanced arctan fitting with best practices:")
print("• R-squared and chi-squared metrics")
print("• Parameter uncertainty estimation")
print("• Memory usage monitoring")
print("• HyperSpy-native plotting methods")
print("• Residual analysis")
