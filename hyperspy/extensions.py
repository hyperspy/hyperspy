
import logging
import os
import importlib
import copy
import pkgutil


import pkg_resources
import yaml

import hyperspy.misc.config_dir

_logger = logging.getLogger(__name__)

_ext_f = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "hyperspy_extension.yaml")
with open(_ext_f, 'r') as stream:
    EXTENSIONS = yaml.safe_load(stream)

# External extensions are not integrated into the API and not
# import unless needed
ALL_EXTENSIONS = copy.deepcopy(EXTENSIONS)

_ext_extensions = [
    entry_point.module_name
    for entry_point in pkg_resources.iter_entry_points('hyperspy.extensions')]

for _ext_ext_mod in _ext_extensions:
    _logger.info("Enabling extension %s" % _ext_ext_mod)
    _path = os.path.join(
        os.path.dirname(pkgutil.get_loader(_ext_ext_mod).get_filename()),
        "hyperspy_extension.yaml")

    if os.path.isfile(_path):
        with open(_path, 'r') as stream:
            _ext_ext = yaml.safe_load(stream)
            if "signals" in _ext_ext:
                ALL_EXTENSIONS["signals"].update(_ext_ext["signals"])
            if "components1D" in _ext_ext:
                ALL_EXTENSIONS["components1D"].update(
                    _ext_ext["components1D"])
            if "components2D" in _ext_ext:
                ALL_EXTENSIONS["components2D"].update(
                    _ext_ext["components2D"])
    else:
        _logger.error("Failed to load hyperspy extension from {0}. Please report this issue to the {0} developers".format(_ext_ext_mod))

