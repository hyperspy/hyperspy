#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    import setuptools
except:
    print "Setuptools unavailable.  setup.py develop and test commands not available."

from distutils.core import setup

import distutils.dir_util

import os
import sys
import shutil

import hyperspy.Release as Release
# clean the build directory so we aren't mixing Windows and Linux installations carelessly.
if os.path.exists('build'):
    distutils.dir_util.remove_tree('build')

install_req = ['scipy', 'ipython', 'matplotlib', 'numpy', 'mdp', 'netcdf','nose']

def are_we_building4windows():
    for arg in sys.argv:
        if 'wininst' in arg:
            return True

scripts = ['bin/hyperspy_compile_kica', 'bin/hyperspy']

if are_we_building4windows() or os.name in ['nt','dos']:
    # In the Windows command prompt we can't execute Python scripts 
    # without a .py extension. A solution is to create batch files
    # that runs the different scripts.
    # (code adapted from scitools)
    install_req.append('pyreadline')
    scripts.append('bin/win_post_installation.py')

    batch_files = []
    for script in scripts:
        batch_file = os.path.splitext(script)[0] + '.bat'
        f = open(batch_file, "w")
        f.write('python "%%~dp0\%s" %%*\n' % os.path.split(script)[1])
        f.close()
        batch_files.append(batch_file)
    scripts.extend(batch_files)
    
version = Release.version

setup(
    name = "hyperspy",
    package_dir = {'hyperspy': 'hyperspy'},
    version = version,
    #py_modules = ['', ],
    packages = ['hyperspy', 'hyperspy.components', 'hyperspy.io_plugins', 
                'hyperspy.drawing', 'hyperspy.mva', 'hyperspy.signals', 
                'hyperspy.bss','hyperspy.gui', 'hyperspy.tests', 
                'hyperspy.tests.io'],
    requires = install_req,
    scripts = scripts,
    package_data = 
    {
        'hyperspy': 
            [
                'ipython_profile/*',
                'data/hyperspyrc',
                'data/*.m', 
                'data/*.csv',
                'data/*.tar.gz',
                'data/kica/*.m',
                'data/kica/*.c',
                'data/kica/distributions/*.m',
		'tests/io/dm3_1D_data/*.dm3',
		'tests/io/dm3_2D_data/*.dm3',
		'tests/io/dm3_3D_data/*.dm3',
            ],
    },
#    author = Release.authors['F_DLP'][0],
#    author_email = Release.authors['F_DLP'][1],
    description = Release.description,
    long_description = open('README.txt').read(),
    license = Release.license,
    platforms = Release.platforms,
    url = Release.url,
    test_suite = 'nose.collector',
    )
