# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
import os

_config = {
    "viewer": None,
    "configfile": os.getenv("HTMD_CONFIG") if os.getenv("HTMD_CONFIG") else None,
    "ncpus": 1,
}


def config(
    viewer: str | None = _config["viewer"],
    configfile: str | None = _config["configfile"],
    ncpus: int = _config["ncpus"],
):
    """
    Function to change HTMD configuration variables.

    Parameters
    ----------
    viewer : str
        Defines the backend viewer for molecular visualization
    configfile : str
        Defines the HTMD configuration file that is called at the beginning of importing
    ncpus : int
        Defines the number of cpus available for several HTMD operations
    """
    _config["viewer"] = viewer
    _config["configfile"] = configfile
    _config["ncpus"] = ncpus
