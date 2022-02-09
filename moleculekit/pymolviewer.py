# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
from moleculekit.molecule import mol_equal
from moleculekit.util import tempname
import numpy as np
import os

_viewers = []
pymolViewingMols = {}
showing = {}
_checkFrequency = 0.5


def getCurrentViewer():
    import pymol
    import sys

    if len(_viewers) == 0:
        import threading

        _viewers.append(True)
        _stdouterr = sys.stdout, sys.stderr
        pymol.finish_launching(["pymol", "-Q"])
        sys.stdout, sys.stderr = _stdouterr
        x = threading.Thread(target=_monitoringThread, args=())
        x.daemon = True
        x.start()


def _monitoringThread():
    global pymolViewingMols
    global showing
    from pymol import cmd
    import time

    def _view(mol, viewname):
        topo = tempname(suffix=".cif")
        xtc = tempname(suffix=".xtc")
        mol.write(topo)
        mol.write(xtc)
        cmd.delete(viewname)
        cmd.load(topo, viewname)
        cmd.load_traj(xtc, viewname, state=1)
        cmd.dss()  # Guess SS
        os.remove(topo)
        os.remove(xtc)
        showing[viewname] = True

    curr_mols = {key: val.copy() for key, val in pymolViewingMols.items()}
    for viewname in curr_mols:
        _view(curr_mols[viewname], viewname)

    while True:
        # print("KEYS", curr_mols.keys())
        new_keys = np.setdiff1d(list(pymolViewingMols.keys()), list(curr_mols.keys()))
        for viewname in new_keys:
            curr_mols[viewname] = pymolViewingMols[viewname].copy()
            _view(curr_mols[viewname], viewname)
            # print("viewed new mol")

        for viewname in curr_mols:
            if not mol_equal(
                curr_mols[viewname], pymolViewingMols[viewname], _logger=False
            ):
                curr_mols[viewname] = pymolViewingMols[viewname].copy()
                _view(curr_mols[viewname], viewname)
                # print("updated existing mol")

        time.sleep(_checkFrequency)

        # Check if molecule which was showing before does not exist in viewer anymore
        # If it doesn't remove it from our lists here to clean-up
        pymol_objects = cmd.get_object_list("all")
        todelete = []
        for key in showing:
            if key not in pymol_objects:
                todelete.append(key)
        for key in todelete:
            del showing[key]
            del curr_mols[key]
            del pymolViewingMols[key]
