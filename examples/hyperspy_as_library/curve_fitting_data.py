""" Loads hyperspy as a regular python library, loads spectrums from files, does curve fitting, and exports the results as hdf5 files"""

import hyperspy.hspy as hspy
import numpy as np
import matplotlib.pyplot as plt

coreLossSpectrumFileName = "test.dm3"

s = hspy.load(coreLossSpectrumFileName).to_EELS()
s.add_elements(("Ti","O","Mn"))
s.set_microscope_parameters(beam_energy=200, convergence_angle=20.0, collection_angle=10.0)

m = hspy.create_model(s)
m.enable_fine_structure()
m.multifit(kind="smart")

m.export_results(format="hdf5")
