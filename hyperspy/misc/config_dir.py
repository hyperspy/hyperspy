# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

import shutil
import logging
from pathlib import Path

_logger = logging.getLogger(__name__)

config_files = list()
config_path = Path("~/.hyperspy").expanduser()
config_path.mkdir(parents=True, exist_ok=True)

data_path = Path(__file__).resolve().parents[1].joinpath("data")

for file in config_files:
    templates_file = data_path.joinpath(file)
    config_file = config_path.joinpath(file)
    if not config_file.is_file():
        _logger.info(f"Setting configuration file: {file}")
        shutil.copy(templates_file, config_file)
