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


def _pymol_view(mol, viewname):
    from pymol import cmd

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


def _pymol_get_mols():
    from pymol import cmd

    return cmd.get_object_list("all")


def _launch_pmview_exe(port):
    from moleculekit.util import find_executable, wait_for_port, check_port
    from subprocess import Popen

    if not check_port(port):
        pmview_exe = find_executable("pmview")
        if pmview_exe is None:
            raise FileNotFoundError(
                "Could not find 'pmview' executable in PATH. You can install it with 'conda install pmview -c acellera'"
            )
        Popen([pmview_exe])

        wait_for_port(port)


def _pmview_launch(url):
    from moleculekit.util import check_port

    port = int(url.split(":")[-1])
    if not check_port(port):
        _launch_pmview_exe(port)


def _pmview_view(mol, viewname, bonds=None, url=None):
    from moleculekit.util import check_port
    import requests
    import time

    port = int(url.split(":")[-1])
    if not check_port(port):
        _pmview_launch(url)

    topo = tempname(suffix=".cif")
    traj = tempname(suffix=".xtc")
    mol.write(topo)
    mol.write(traj)

    files = {"topo": (topo, open(topo, "rb")), "traj": (traj, open(traj, "rb"))}
    data = {"label": viewname, "moleculeID": viewname}
    response = requests.post(
        f"{url}/load-molecules", headers={}, data=data, files=files
    )
    response.close()

    os.remove(traj)
    os.remove(topo)

    t = time.time()
    while viewname not in _pmview_get_mols(url):
        # Wait for the mol to appear in the viewer before marking it as showing
        time.sleep(_checkFrequency)
        if time.time() - t > 10:  # Something went bad
            return

    showing[viewname] = True


def _pmview_get_mols(url):
    from moleculekit.util import check_port
    import requests
    import json

    port = int(url.split(":")[-1])
    if not check_port(port):
        return []

    response = requests.get(f"{url}/get-molecules")
    response.close()

    mols = json.loads(response.text)

    return mols


def getCurrentPymolViewer():
    getCurrentViewer(_pymol_launch, _pymol_view, _pymol_get_mols)


def getCurrentPMViewer(url):
    getCurrentViewer(
        lambda: _pmview_launch(url),
        lambda *args: _pmview_view(*args, url=url),
        lambda: _pmview_get_mols(url),
    )


def getCurrentViewer(launch_fn, view_fn, get_mols_fn):
    if len(_viewers) == 0:
        import threading

        _viewers.append(True)
        launch_fn()

        x = threading.Thread(target=_monitoringThread, args=(view_fn, get_mols_fn))
        x.daemon = True
        x.start()


def _monitoringThread(view_fn, get_mols_fn):
    global viewingMols
    global showing

    import time

    curr_mols = {key: val.copy() for key, val in viewingMols.items()}
    for viewname in curr_mols:
        view_fn(curr_mols[viewname], viewname)

    while True:
        # print("KEYS", curr_mols.keys())
        new_keys = np.setdiff1d(list(viewingMols.keys()), list(curr_mols.keys()))
        for viewname in new_keys:
            curr_mols[viewname] = viewingMols[viewname].copy()
            view_fn(curr_mols[viewname], viewname)
            # print("viewed new mol")

        for viewname in curr_mols:
            if not mol_equal(curr_mols[viewname], viewingMols[viewname], _logger=False):
                curr_mols[viewname] = viewingMols[viewname].copy()
                view_fn(curr_mols[viewname], viewname)
                # print("updated existing mol")

        time.sleep(_checkFrequency)

        if len(viewingMols) and get_mols_fn is not None:
            # Check if molecule which was showing before does not exist in viewer anymore
            # If it doesn't remove it from our lists here to clean-up
            molnames = get_mols_fn()

            todelete = []
            for key in showing:
                if key not in molnames:
                    todelete.append(key)

            for key in todelete:
                del showing[key]
                del curr_mols[key]
                del viewingMols[key]
