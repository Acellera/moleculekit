# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
from moleculekit.molecule import mol_equal
from moleculekit.util import tempname
import os

_viewers = []
viewingMols = {}
showing = {}
_checkFrequency = 0.5


def _pymol_launch():
    import sys

    try:
        import pymol
    except ImportError:
        raise ImportError(
            "Pymol is not installed. You can install it with 'conda install pymol-open-source -c conda-forge'"
        )

    _stdouterr = sys.stdout, sys.stderr
    pymol.finish_launching(["pymol", "-Q"])
    sys.stdout, sys.stderr = _stdouterr


def _pymol_view(mol, unique_id):
    from pymol import cmd

    topo = tempname(suffix=".cif")
    xtc = tempname(suffix=".xtc")
    mol.write(topo)
    mol.write(xtc)
    cmd.delete(unique_id)
    cmd.load(topo, unique_id)
    cmd.load_traj(xtc, unique_id, state=1)
    cmd.dss()  # Guess SS
    os.remove(topo)
    os.remove(xtc)
    showing[unique_id] = True


def _pymol_get_mols():
    from pymol import cmd

    return cmd.get_object_list("all")


def getCurrentPymolViewer():
    getCurrentViewer(_pymol_launch, _pymol_view, _pymol_get_mols)


def getCurrentViewer(launch_fn, view_fn, get_mols_fn):
    if len(_viewers) == 0:
        import threading

        _viewers.append(True)
        launch_fn()

        x = threading.Thread(target=_monitoringThread, args=(view_fn, get_mols_fn))
        x.daemon = True
        x.start()


def _monitoringThread(view_fn, get_mols_fn):
    import time

    curr_mols = {key: val.copy() for key, val in viewingMols.items()}
    for unique_id in curr_mols:
        view_fn(curr_mols[unique_id], unique_id)

    while True:
        try:
            new_keys = [key for key in viewingMols if key not in curr_mols.keys()]
            for unique_id in new_keys:
                curr_mols[unique_id] = viewingMols[unique_id].copy()
                view_fn(curr_mols[unique_id], unique_id)

            for unique_id in curr_mols:
                if not mol_equal(
                    curr_mols[unique_id], viewingMols[unique_id], _logger=False
                ):
                    curr_mols[unique_id] = viewingMols[unique_id].copy()
                    view_fn(curr_mols[unique_id], unique_id)

            time.sleep(_checkFrequency)

            if len(viewingMols) and get_mols_fn is not None:
                # Reap molecules the user closed in the external viewer.
                molnames = get_mols_fn()

                todelete = []
                for key in showing:
                    if key not in molnames:
                        todelete.append(key)

                for key in todelete:
                    del showing[key]
                    del curr_mols[key]
                    del viewingMols[key]
        except Exception:
            import traceback

            print("Failed to view molecule with error:\n", traceback.format_exc())
