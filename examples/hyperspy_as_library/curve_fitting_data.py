""" Loads hyperspy as a regular python library, loads spectrums from files, does curve fitting, and exports the results as hdf5 files"""

import hyperspy.hspy as hspy
import numpy as np
import matplotlib.pyplot as plt

coreLossSpectrumFileName = "test.dm3" #Several spectrums (stack)
lowLossFileName = "lowloss.dm3" #One spectrum

s = hspy.load(coreLossSpectrumFileName).to_EELS()
s.add_elements(("Ti","O","Mn"))
s.set_microscope_parameters(beam_energy=200, convergence_angle=20.0, collection_angle=10.0)

#The low loss spectrum must have the same spatial dimensions as the 
#core loss spectrum. This is achieved by loading the low loss spectrum
#several times and putting them in a stack.
coreLossSpatialDimension = len(s.data) #Assumes one spatial dimension
lowLossFileNameList = [lowLossFileName]*coreLossSpatialDimension 

ll = hspy.load(lowLossFileNameList, stack=True).to_EELS()

m = hspy.create_model(s, ll=ll)
m.enable_fine_structure()
m.multifit(kind="smart")

m.export_results(format="hdf5")
