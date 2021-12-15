# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
_viewers = []


def getCurrentViewer():
    import pymol
    import sys

    # Currently only supports a single viewer
    if len(_viewers) == 0:
        _stdouterr = sys.stdout, sys.stderr
        pymol.finish_launching(["pymol", "-q"])
        sys.stdout, sys.stderr = _stdouterr
        _viewers.append(True)
