"""Creates a spectrum, and fits an arctan to it."""

import numpy as np

#Generate the data and make the spectrum
x_range = np.array(range(-500,500))
arctanSpectrum = signals.Spectrum({"data":np.arctan(x_range})

#Make the arctan component for use in the model
arctanComponent = components.Arctan()

#Create the model and add the arctan component
arctanModel = create_model(arctanSpectrum)
arctanModel.append(arctanComponent)

#Fit the arctan component to the spectrum
arctanModel.fit()

#Plot the spectrum and the model fitting
arctanModel.plot()
