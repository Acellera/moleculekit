# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
import os
import logging.config
from importlib.metadata import version, PackageNotFoundError

# from importlib.resources import files

try:
    __version__ = version("moleculekit")
except PackageNotFoundError:
    pass

__curr_dir = os.path.dirname(os.path.abspath(__file__))
# __share_dir = str(files("moleculekit.share").joinpath(""))
__share_dir = os.path.join(__curr_dir, "share")

try:
    logging.config.fileConfig(
        os.path.join(__curr_dir, "logging.ini"), disable_existing_loggers=False
    )
except Exception:
    print("MoleculeKit: Logging setup failed")
