#!/usr/bin/env python
# -*- coding: utf-8 -*-

from distutils.core import setup

from setuptools import setup

install_req = ['eelslab', 'traitsgui']

setup(
    name = "eelslab-gui",
    version = "0.1",
    package_dir = {'eelslab-gui': '.'},
    packages = ['eelslab-gui', ],
    requires = install_req,
    scripts = ['scripts/*'],
#    package_data = {'eelslab':['examples/*', 'data/*']},
    author = "Francisco Javier de la Peña",
    author_email = "delapena@lps.u-psud.fr",
    description = "Electron Energy Loss Spectra analysis",
#    license = "GPL",
    platforms='any',
    url = "http://www.eelslab.org",   
)
