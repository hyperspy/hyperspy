
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
EXTENSIONS["GUI"]["widgets"] = {}

# External extensions are not integrated into the API and not
# import unless needed
ALL_EXTENSIONS = copy.deepcopy(EXTENSIONS)

_external_extensions = [
    entry_point.module_name
    for entry_point in pkg_resources.iter_entry_points('hyperspy.extensions')]

for _external_extension_mod in _external_extensions:
    _logger.info("Enabling extension %s" % _external_extension_mod)
    _path = os.path.join(
        os.path.dirname(pkgutil.get_loader(_external_extension_mod).get_filename()),
        "hyperspy_extension.yaml")

    if os.path.isfile(_path):
        with open(_path, 'r') as stream:
            _external_extension = yaml.safe_load(stream)
            if "signals" in _external_extension:
                ALL_EXTENSIONS["signals"].update(_external_extension["signals"])
            if "components1D" in _external_extension:
                ALL_EXTENSIONS["components1D"].update(
                    _external_extension["components1D"])
            if "components2D" in _external_extension:
                ALL_EXTENSIONS["components2D"].update(
                    _external_extension["components2D"])
            if "GUI" in _external_extension:
                if "toolkeys" in _external_extension["GUI"]:
                    ALL_EXTENSIONS["GUI"]["toolkeys"].extend(
                        _external_extension["GUI"]["toolkeys"])
                if "widgets" in _external_extension["GUI"]:
                    for toolkit, specs in _external_extension["GUI"]["widgets"].items():
                        if toolkit not in ALL_EXTENSIONS["GUI"]["widgets"]:
                            ALL_EXTENSIONS["GUI"]["widgets"][toolkit] = {}
                        ALL_EXTENSIONS["GUI"]["widgets"][toolkit].update(specs)

    else:
        _logger.error(
            "Failed to load hyperspy extension from {0}. Please report this issue to the {0} developers".format(_external_extension_mod))
