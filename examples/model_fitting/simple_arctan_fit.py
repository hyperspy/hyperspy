"""Creates a spectrum, and fits an arctan to it."""

import numpy as np

#Generate the data and make the spectrum
x_range = np.array(range(-500,500))
arctanSpectrum = signals.SpectrumSimulation({"data":np.arctan(x_range)})
arctanSpectrum.axes_manager.axes[0].offset = -500
arctanSpectrum.axes_manager.axes[0].units = ""
arctanSpectrum.axes_manager.axes[0].name = "x"
arctanSpectrum.mapped_parameters.title = "Simple arctan fit"
			
arctanSpectrum.add_gaussian_noise(0.1)

#Make the arctan component for use in the model
arctanComponent = components.Arctan()

#Create the model and add the arctan component
arctanModel = create_model(arctanSpectrum)
arctanModel.append(arctanComponent)

#Fit the arctan component to the spectrum
arctanModel.fit()

# Print the result of the fit
arctanModel.print_current_values()

#Plot the spectrum and the model fitting
arctanModel.plot()
