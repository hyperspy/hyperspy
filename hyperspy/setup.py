#!/usr/bin/env python
# -*- coding: utf-8 -*-

from distutils.core import setup

from setuptools import setup

install_req = ['hyperspy', 'traitsgui']

setup(
    name = "hyperspy-gui",
    version = "0.1",
    package_dir = {'hyperspy-gui': '.'},
    packages = ['hyperspy-gui', ],
    requires = install_req,
    scripts = ['scripts/*'],
#    package_data = {'hyperspy':['examples/*', 'data/*']},
    author = "Francisco Javier de la Peña",
    author_email = "delapena@lps.u-psud.fr",
    description = "Electron Energy Loss Spectra analysis",
#    license = "GPL",
    platforms='any',
    url = "http://www.hyperspy.org",   
)
