"""Creates a spectrum, and fits an arctan to it."""

import numpy as np

#Generate the data and make the spectrum
x_range = np.array(range(-500,500))
s = signals.SpectrumSimulation(np.arctan(x_range))
s.axes_manager._axes[0].offset = -500
s.axes_manager._axes[0].units = ""
s.axes_manager._axes[0].name = "x"
s.mapped_parameters.title = "Simple arctan fit"
			
s.add_gaussian_noise(0.1)

#Make the arctan component for use in the model
arctan_component = components.Arctan()

#Create the model and add the arctan component
m = create_model(s)
m.append(arctanComponent)

#Fit the arctan component to the spectrum
m.fit()

# Print the result of the fit
m.print_current_values()

#Plot the spectrum and the model fitting
m.plot()
