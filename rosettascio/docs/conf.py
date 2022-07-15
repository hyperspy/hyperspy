# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'RosettaSciIO'
copyright = '2022, HyperSpy Developers'
author = 'HyperSpy Developers'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.githubpages',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_theme_options = {
    "icon_links": [
        {
            # Label for this link
            "name": "GitHub",
            # URL where the link will redirect
            "url": "https://github.com/hyperspy/rosettasciio",  # required
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "fab fa-github-square",
            # The type of image to be used (see below for details)
            "type": "fontawesome",
        },
        {
            # Label for this link
            "name": "RosettaSciIO",
            # URL where the link will redirect
            "url": "https://github.com/hyperspy/rosettasciio",  # required
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "",
            # The type of image to be used (see below for details)
            "type": "local",
        }
    ],
    "logo": {
        "text": "RosettaSciIO",
    },
}

# If you’re hosting your documentation on ReadTheDocs, you should consider
# adding an explicit placement for their ethical advertisements. These are
# non-tracking advertisements from ethical companies, and they help
# ReadTheDocs sustain themselves and their free service.
#
# Ethical advertisements are added to your sidebar by default. To ensure
# they are there if you manually update your sidebar, ensure that the
# sidebar-ethical-ads.html template is added to your list. For example:

html_sidebars = {
    "**": ["search-field.html", "sidebar-nav-bs.html", "sidebar-ethical-ads.html"]
}

def setup(app):
    app.add_css_file("custom-styles.css")