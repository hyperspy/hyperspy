"""Creates several random spectrum images, then aggregates them.
"""

# If running from hyperspy's interactive the next two imports can be omitted 
# omitted (i.e. the next 2 lines)
import numpy as np
import matplotlib.pyplot as plt

from hyperspy.signals.spectrum import Spectrum
from hyperspy.signals.aggregate import AggregateSpectrum

# create the spectrum objects

s1 = Spectrum({'data' : np.random.random((64, 64, 1024)), 
               'mapped_parameters' : {
                    'name':'file1'}})
s2 = Spectrum({'data' : np.random.random((64, 64, 1024)),
               'mapped_parameters' : {
                    'name':'file2'}})
s3 = Spectrum({'data' : np.random.random((64, 64, 1024)),
               'mapped_parameters' : {
                    'name':'file3'}})

"""
Perhaps a better example (but one that can't be self contained without
including too many data files for you to download):

s1=load('your_file1.dm3')
s2=load('your_file2.dm3')
s3=load('your_file3.dm3')

"""

s=AggregateSpectrum(s1,s2,s3)

"""
An even better way: load the files and create the aggregate in one command:

from glob import glob
s=load('*.dm3')

"""

s.plot()

# If running from hyperspy's interactive console the next line can be 
# omitted
plt.show()
