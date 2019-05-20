
import logging
import yaml
import os

_ext_f = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "hyperspy_extensions.yaml")
with open(_ext_f, 'r') as stream:
    EXTENSIONS = yaml.safe_load(stream)