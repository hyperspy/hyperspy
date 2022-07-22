# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.

import logging
import yaml
import os

from rsciio.version import __version__

IO_PLUGINS = []
_logger = logging.getLogger(__name__)

here = os.path.abspath(os.path.dirname(__file__))

for sub, _, _ in os.walk(here):
    specsf = os.path.join(sub, "specifications.yaml")
    if os.path.isfile(specsf):
        with open(specsf, 'r') as stream:
            specs = yaml.safe_load(stream)
            specs["api"] = "rsciio.%s.api" % os.path.split(sub)[1]
            IO_PLUGINS.append(specs)


