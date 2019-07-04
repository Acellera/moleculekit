from moleculekit.version import version as _version
from moleculekit.home import home as __home
import os
import logging.config

__version__ = _version()

try:
    logging.config.fileConfig(os.path.join(__home(), 'logging.ini'), disable_existing_loggers=False)
except:
    print("MoleculeKit: Logging setup failed")