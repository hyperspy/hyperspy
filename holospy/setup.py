#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2016-2023 The pyXem developers
#
# This file is part of pyXem.
#
# pyXem is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pyXem is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pyXem.  If not, see <http://www.gnu.org/licenses/>.

from setuptools import setup

exec(open("pyxem/release_info.py").read())  # grab version info

# Projects with optional features for building the documentation and running
# tests. From setuptools:
# https://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-extras-optional-features-with-their-own-dependencies
extra_feature_requirements = {
    "doc": [
        "furo",
        "nbsphinx                   >= 0.7",
        "sphinx                     >= 3.0.2",
        "sphinx-copybutton          >= 0.2.5",
        "sphinx-autodoc-typehints   >= 1.10.3",
        "sphinx-gallery             >= 0.6",
        "sphinxcontrib-bibtex       >= 1.0",
    ],
    "tests": [
        "pytest     >= 5.0",
        "pytest-cov >= 2.8.1",
        "pytest-xdist",
        "coveralls  >= 1.10",
        "coverage   >= 5.0",
    ],
    "dev": ["black", "pre-commit >=1.16"],
    "gpu": ["cupy >= 9.0.0"],
}


setup(
    name="holospy",
    version="0.0.1",
    description="Holography",
    author="Hyperspy Developers",
    license=license,
    url="https://github.com/hyperspy/holospy",
    long_description=open("README.rst").read(),
    keywords=[
        "data analysis",
        "microscopy",
        "electron microscopy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    install_requires=["hyperspy"],
extras_require = {
	"ipython": ["IPython>7.0, !=8.0", "ipyparallel"],
    "learning": ["scikit-learn>=1.0.1"],
    "gui-jupyter": ["hyperspy_gui_ipywidgets>=1.1.0", "ipympl"],
    # UPDATE BEFORE RELEASE
    "gui-traitsui": ["hyperspy_gui_traitsui @ git+https://github.com/hyperspy/hyperspy_gui_traitsui#egg=hyperspy_gui_traitsui"],
    #"gui-traitsui": ["hyperspy_gui_traitsui>=1.1.0"],
    "tests": [
        "pytest>=3.6",
        "pytest-mpl",
        "pytest-xdist",
        "pytest-rerunfailures",
        "pytest-instafail",
        ],
    "coverage":["pytest-cov"],
    # required to build the docs
    "build-doc": [
        "pydata_sphinx_theme",
        "sphinx>=1.7",
        "sphinx-gallery",
        "sphinx-toggleprompt",
        "sphinxcontrib-mermaid",
        "sphinxcontrib-towncrier>=0.3.0a0",
        "sphinx-design",
        "towncrier",
        ],
}
    python_requires=">=3.7",
    package_data={
        "": ["LICENSE", "README.rst"],
        "holospy": ["*.py", "hyperspy_extension.yaml"],
    },
    entry_points={"hyperspy.extensions": ["holospy = holospy"]},
)