""" Loads hyperspy as a regular python library, loads spectrums from files, does curve fitting, and plotting the model and original spectrum to a png file"""

import hyperspy.hspy as hspy
import matplotlib.pyplot as plt

coreLossSpectrumFileName = "coreloss_spectrum.msa"
lowlossSpectrumFileName = "lowloss_spectrum.msa"

s = hspy.load(coreLossSpectrumFileName).to_EELS()
s.add_elements(("Mn", "O"))
s.set_microscope_parameters(
    beam_energy=300,
    convergence_angle=24.6,
    collection_angle=13.6)

ll = hspy.load(lowlossSpectrumFileName).to_EELS()

m = hspy.create_model(s, ll=ll)
m.enable_fine_structure()
m.multifit(kind="smart")
m.plot()

plt.savefig("model_original_spectrum_plot.png")
