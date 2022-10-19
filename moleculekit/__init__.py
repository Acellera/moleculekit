# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
from moleculekit.home import home as __home
import os
import logging.config
from moleculekit import _version

__version__ = _version.get_versions()["version"]

try:
    logging.config.fileConfig(
        os.path.join(__home(), "logging.ini"), disable_existing_loggers=False
    )
except Exception:
    print("MoleculeKit: Logging setup failed")

from . import _version
__version__ = _version.get_versions()['version']
