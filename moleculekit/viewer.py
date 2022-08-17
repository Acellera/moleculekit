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
_molstar_stub = None
_molstar_url = ""


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


def _launch_molkitstar_exe(port):
    from moleculekit.util import find_executable, wait_for_port, check_port
    from subprocess import Popen

    if not check_port(port):
        molkitstarexe = find_executable("molkitstar")
        if molkitstarexe is None:
            raise FileNotFoundError(
                "Could not find 'molkitstar' viewer executable in PATH. Please install it with conda install molkitstar -c acellera"
            )
        Popen([molkitstarexe])

        wait_for_port(port)


def _molstar_launch(url):
    from moleculekit.util import check_port
    from moleculekit.molstar_grpc import molstar_pb2_grpc
    import grpc

    global _molstar_stub
    global _molstar_url

    port = int(url.split(":")[1])
    if not check_port(port):
        _launch_molkitstar_exe(port)

    channel = grpc.insecure_channel(url)
    stub = molstar_pb2_grpc.MoleculeLoaderStub(channel)
    _molstar_stub = stub
    _molstar_url = url


def _molstar_view(mol, viewname, bonds=None, url=None):
    from moleculekit.util import check_port
    from moleculekit.molstar_grpc import molstar_pb2
    from grpc._channel import _InactiveRpcError

    port = int(url.split(":")[1])
    if not check_port(port):
        _molstar_launch(url)

    psf = tempname(suffix=".psf")
    xtc = tempname(suffix=".xtc")
    mol.write(psf, explicitbonds=bonds)
    mol.write(xtc)

    with open(psf) as fh:
        psfblob = fh.read()
    with open(xtc, "rb") as fh:
        xtcbinaryblob = fh.read()

    data = molstar_pb2.MoleculeData(
        name=viewname,
        structure=psfblob,
        # repr_type='ball+stick',
        # repr_selection='hetero',
        traj=xtcbinaryblob,
    )
    try:
        _ = _molstar_stub.LoadMolecule(data)
    except _InactiveRpcError:
        print(f"ERROR: No viewer found at URL: {_molstar_url}")

    os.remove(xtc)
    os.remove(psf)


def _molstar_get_mols():
    # TODO: Write this!
    response = _molstar_stub.GetMolecules()
    print("Molstar server replied: " + response.molecules)
    return response.molecules


def getCurrentPymolViewer():
    getCurrentViewer(_pymol_launch, _pymol_view, _pymol_get_mols)


def getCurrentMolstarViewer(url):
    getCurrentViewer(
        lambda: _molstar_launch(url), lambda *args: _molstar_view(*args, url=url), None
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

        if get_mols_fn is not None:
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
