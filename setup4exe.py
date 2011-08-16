from distutils.core import setup
import py2exe
from glob import glob
import matplotlib
import os

                                    
includes = []
includes.append('numpy')
includes.append('scipy')
includes.append('numpy.core')
includes.append('wx')
includes.append('wx.*')
#includes.append('mdp')
#includes.append('netCDF4')

data_files = [('hyperspy/data', ['d:\\Python/hyperspy/src/data/edges_db.csv']),
              ('hyperspy/data', ['d:\\Python/hyperspy/src/data/microscopes.csv']),
              ('hyperspy/data', ['d:\\Python/hyperspy/src/data/hyperspyrc']),
              ('', ['c:\\Python27/Lib/site-packages/HDF5DLL.DLL']),
              ('', ['c:\\Python27/Lib/site-packages/HDF5_HLDLL.DLL']),
              ('', ['c:\\Python27/Lib/site-packages/NETCDF.DLL']),
              ('', ['c:\\Python27/Lib/site-packages/netCDF4-0.9.3-py2.7.egg-info']),
              ('', ['c:\\Python27/Lib/site-packages/netCDF4.pyd']),
              ('', ['c:\\Python27/Lib/site-packages/netCDF4_utils.py']),
              ('', ['c:\\Python27/Lib/site-packages/SZ.DLL']),
              ('', ['c:\\Python27/Lib/site-packages/ZLIB1.DLL']),
              ('', ['c:\\Python27/Lib/site-packages/netCDF4_utils.pyc']),
              ('', ['c:\\Python27/Lib/site-packages/netCDF4_utils.pyo']),
              ]
data_files.extend([('mdp/utils', ['c:\\Python27/Lib/site-packages/mdp/utils/slideshow.css']),
              ('mdp/hinet', ['c:\\Python27/Lib/site-packages/mdp/hinet/hinet.css']),
              ])

data_files.extend(matplotlib.get_py2exe_datafiles())
includes.extend(["matplotlib.backends", "matplotlib.backends.backend_wxagg",])


setup(console=['hyperspy.py'],
      options = {
          "py2exe": 
              {"includes": includes,
               "skip_archive": True,
               "packages": ['netcdftime',],
               "compressed": False,},
                },
      data_files=data_files)


