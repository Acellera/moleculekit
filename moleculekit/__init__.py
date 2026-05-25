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
    # Optional environment override for the moleculekit log line format.
    # Set MOLECULEKIT_LOG_FORMAT to any logging.Formatter format string —
    # e.g. "%(name)s - %(levelname)s - %(message)s" to drop the timestamp.
    _log_format = os.environ.get("MOLECULEKIT_LOG_FORMAT")
    if _log_format:
        _formatter = logging.Formatter(_log_format)
        for _h in logging.getLogger("moleculekit").handlers:
            _h.setFormatter(_formatter)
except Exception:
    print("MoleculeKit: Logging setup failed")
