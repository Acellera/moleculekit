# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
import moleculekit
import os
import sys
import inspect


def home(dataDir=None, shareDir=None):
    """Return the pathname of the moleculekit root directory (or a data subdirectory).

    Parameters
    ----------
    dataDir : str
        If not None, return the path to a specific data directory
    shareDir : bool
        If True, return path to the share directory

    Returns
    -------
    dir : str
        The directory

    Example
    -------
    >>> from moleculekit.home import home
    >>> home()                                 # doctest: +ELLIPSIS
    '.../moleculekit'
    >>> home(dataDir="dhfr")                   # doctest: +ELLIPSIS
    '.../data/dhfr'
    >>> os.path.join(home(dataDir="dhfr"),"dhfr.pdb")  # doctest: +ELLIPSIS
    '.../data/dhfr/dhfr.pdb'
    """

    homeDir = os.path.dirname(inspect.getfile(moleculekit))
    try:
        if sys._MEIPASS:
            homeDir = sys._MEIPASS
    except Exception:
        pass

    if dataDir:
        return os.path.join(homeDir, "test-data", dataDir)
    elif shareDir is not None:
        sharedir = os.path.join(homeDir, "share")
        if not os.path.exists(sharedir):
            raise FileNotFoundError("Could not find moleculekit share directory.")
        return os.path.join(sharedir, shareDir)
    else:
        return homeDir


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    h = home()
    print(h)
