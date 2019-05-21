
import logging
import yaml
import os
import importlib
import copy

import hyperspy.misc.config_dir

_logger = logging.getLogger(__name__)

_ext_f = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "hyperspy_extension.yaml")
with open(_ext_f, 'r') as stream:
    EXTENSIONS = yaml.safe_load(stream)

# External extensions are not integrated into the API and not
# import unless needed
ALL_EXTENSIONS = copy.deepcopy(EXTENSIONS)

_external_exts_f = os.path.join(
    hyperspy.misc.config_dir.config_path,
    "hspy_extensions.yaml")
with open(_external_exts_f, 'r') as stream:
    external_exts = yaml.safe_load(stream)
    for ext_ext_mod in external_exts:
        _logger.info("Loading extension %s" % ext_ext_mod)
        path = os.path.join(
            os.path.dirname(
                importlib.import_module(ext_ext_mod).__file__),
            "hyperspy_extension.yaml")
        with open(path, 'r') as stream:
            ext_ext = yaml.safe_load(stream)
            if "signals" in ext_ext:
                ALL_EXTENSIONS["signals"].update(ext_ext["signals"])
            if "components1D" in ext_ext:
                ALL_EXTENSIONS["components1D"].update(ext_ext["components1D"])
            if "components2D" in ext_ext:
                ALL_EXTENSIONS["components2D"].update(ext_ext["components2D"])
