# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
import numpy as np
import os
import tempfile
import logging


logger = logging.getLogger(__name__)


def tempname(suffix="", create=False):
    if create:
        file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    else:
        file = tempfile.NamedTemporaryFile(delete=True, suffix=suffix)
    file.close()
    return file.name


def ensurelist(tocheck, tomod=None):
    """Convert np.ndarray and scalars to lists.

    Lists and tuples are left as is. If a second argument is given,
    the type check is performed on the first argument, and the second argument is converted.
    """
    if tomod is None:
        tomod = tocheck
    if isinstance(tocheck, np.ndarray):
        return list(tomod)
    if isinstance(tocheck, range):
        return list(tocheck)
    if not isinstance(tocheck, list) and not isinstance(tocheck, tuple):
        return [
            tomod,
        ]
    return tomod


def rotationMatrix(axis, theta):
    """Produces a rotation matrix given an axis and radians

    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.

    Parameters
    ----------
    axis: list
        The axis around which to rotate
    theta: float
        The rotation angle in radians

    Returns
    -------
    M: numpy.ndarray
        The rotation matrix.

    Examples
    --------
    >>> M = rotationMatrix([0, 0, 1], 1.5708)
    >>> M.round(4)
    array([[-0., -1.,  0.],
           [ 1., -0.,  0.],
           [ 0.,  0.,  1.]])

    >>> axis = [4.0, 4., 1.]
    >>> theta = 1.2
    >>> v = [3.0, 5., 0.]
    >>> np.dot(rotationMatrix(axis, theta), v).round(2)
    array([ 2.75,  4.77,  1.92])
    """
    from math import sqrt, sin, cos

    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis / sqrt(np.dot(axis, axis))
    a = cos(theta / 2)
    b, c, d = -axis * sin(theta / 2)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )


def molRMSD(mol, refmol, rmsdsel1, rmsdsel2):
    """Calculates the RMSD between two Molecules

    Parameters
    ----------
    mol : :class:`Molecule <moleculekit.molecule.Molecule>` object
    refmol
    rmsdsel1
    rmsdsel2

    Returns
    -------
    rmsd : float
        The RMSD between the two structures
    """
    dist = mol.coords[rmsdsel1, :, :] - refmol.coords[rmsdsel2, :, :]
    rmsd = np.sqrt(np.mean(np.sum(dist * dist, axis=1), axis=0))
    return np.squeeze(rmsd)


def orientOnAxes(mol, sel="all"):
    """Rotate a molecule so that its main axes are oriented along XYZ.

    The calculation is based on the axes of inertia of the given
    selection, but masses will be ignored. After the operation, the
    main axis will be parallel to the Z axis, followed by Y and X (the
    shortest axis). Only the first frame is oriented.  The reoriented
    molecule is returned.

    Parameters
    ----------
    mol :
        The Molecule to be rotated
    sel : str
        Atom selection string on which the rotation is computed.
        See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__

    Examples
    --------
    >>> mol = Molecule("1kdx")
    >>> mol = orientOnAxes(mol,"chain B")

    """
    if mol.numFrames != 1:
        logger.warning("Only the first frame is considered for the orientation")
    mol = mol.copy()
    sel = mol.atomselect(sel)

    covariance = np.cov(mol.coords[sel, :, 0].T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    logger.info(f"Moments of inertia: {eigenvalues}")
    eigenvectors = eigenvectors.T

    if np.linalg.det(eigenvectors) < 0:
        eigenvectors = -eigenvectors  # avoid inversions

    mol.rotateBy(eigenvectors)
    return mol


def sequenceID(field, prepend=None, step=1):
    """Array of integers which increments at value change of another array

    Parameters
    ----------
    field : np.ndarray or tuple
        An array of values. Once a change in value happens, a new ID will be created in `seq`.
        If a tuple of ndarrays is passed, a change in any of them will cause an increase in `seq`.
    prepend : str
        A string to prepend to the incremental sequence
    step : int
        The step size for incremeting the ID

    Returns
    -------
    seq : np.ndarray
        An array of equal size to `field` containing integers which increment every time there is a change in `field`

    Examples
    --------
    >>> # A change in resid, insertion, chain or segid will cause an increase in the sequence
    >>> sequenceID((mol.resid, mol.insertion, mol.chain, mol.segid))
    array([  1,   1,   1, ..., 285, 286, 287])
    >>> # it is typically used to renumber resids as follows
    >>> mol.set('resid', sequenceID((mol.resid, mol.insertion, mol.chain, mol.segid)))
    """
    if isinstance(field, tuple):
        fieldlen = len(field[0])
    else:
        fieldlen = len(field)

    if fieldlen == 0:
        raise RuntimeError("An empty array was passed to sequenceID")

    if prepend is None:
        seq = np.zeros(fieldlen, dtype=int)
    else:
        seq = np.empty(fieldlen, dtype=object)

    c = int(0)
    if prepend is None:
        seq[0] = c
    else:
        seq[0] = prepend + str(c)

    for i in range(1, fieldlen):
        if isinstance(
            field, tuple
        ):  # Support tuples of multiple fields. Change in any of them will cause an increment
            for t in field:
                if t[i - 1] != t[i]:
                    c += step  # new sequence id
                    break
        elif field[i - 1] != field[i]:
            c += step  # new sequence id
        if prepend is None:
            seq[i] = c
        else:
            seq[i] = prepend + str(c)
    return seq


def _missingChain(mol):
    if mol.chain is None or np.size(mol.chain) == 0:
        raise RuntimeError(
            "Segid fields have to be set for all atoms in the Molecule object before building."
        )
    empty = [True if len(c) == 0 else False for c in mol.chain]
    if np.any(empty):
        idx = np.where(empty)[0]
        if len(idx) == 1:
            raise RuntimeError("Atom " + str(idx) + " does not have a chain defined.")
        elif len(idx) <= 5:
            raise RuntimeError("Atoms " + str(idx) + " do not have a chain defined.")
        else:
            raise RuntimeError(
                "Atoms ["
                + str(idx[0])
                + ","
                + str(idx[1])
                + ",...,"
                + str(idx[-1])
                + "] do not have chain defined."
            )


def _missingSegID(mol):
    if mol.segid is None or np.size(mol.segid) == 0:
        raise RuntimeError(
            "Segid fields have to be set for all atoms in the Molecule object before building."
        )
    empty = [True if len(s) == 0 else False for s in mol.segid]
    if np.any(empty):
        idx = np.where(empty)[0]
        if len(idx) == 1:
            raise RuntimeError("Atom " + str(idx) + " does not have a segid defined.")
        elif len(idx) <= 5:
            raise RuntimeError("Atoms " + str(idx) + " do not have a segid defined.")
        else:
            raise RuntimeError(
                "Atoms ["
                + str(idx[0])
                + ","
                + str(idx[1])
                + ",...,"
                + str(idx[-1])
                + "] do not have segid defined."
            )


def maxDistance(mol, sel="all", origin=None):
    """Calculates the max distance of a set of atoms from an origin

    Parameters
    ----------
    mol : :class:`Molecule <moleculekit.molecule.Molecule>` object
        The molecule containing the atoms
    sel : str
        Atom selection string for atoms for which to calculate distances.
        See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__
    origin : list
        The origin x,y,z coordinates

    Returns
    -------
    maxd : float
        The maximum distance in Angstrom

    Example
    -------
    >>> y = maxDistance(mol, sel='protein', origin=[0, 0, 0])
    >>> print(round(y,2))
    48.39
    """
    from moleculekit.distance import cdist

    if origin is None:
        origin = [0, 0, 0]
    coors = mol.get("coords", sel=sel)
    dists = cdist(np.atleast_2d(coors), np.atleast_2d(origin))
    return np.max(dists)


def boundingBox(mol, sel="all"):
    """Calculates the bounding box of a selection of atoms.

    Parameters
    ----------
    mol : :class:`Molecule <moleculekit.molecule.Molecule>` object
        The molecule containing the atoms
    sel : str
        Atom selection string of atoms. See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__

    Returns
    -------
    bbox : np.ndarray
        The bounding box around the atoms selected in `sel`.

    Example
    -------
    >>> boundingBox(mol, sel='chain A')
    array([[-17.3390007 , -10.43700027,  -1.43900001],
           [ 25.40600014,  27.03800011,  46.46300125]], dtype=float32)

    """
    coords = mol.get("coords", sel=sel)
    maxc = np.squeeze(np.max(coords, axis=0))
    minc = np.squeeze(np.min(coords, axis=0))
    return np.vstack((minc, maxc))


def uniformRandomRotation():
    """
    Return a uniformly distributed rotation 3 x 3 matrix

    The initial description of the calculation can be found in the section 5 of "How to generate random matrices from
    the classical compact groups" of Mezzadri (PDF: https://arxiv.org/pdf/math-ph/0609050.pdf; arXiv:math-ph/0609050;
    and NOTICES of the AMS, Vol. 54 (2007), 592-604). Sample code is provided in that section as the ``haar_measure``
    function.

    Apparently this code can randomly provide flipped molecules (chirality-wise), so a fix found in
    https://github.com/tmadl/sklearn-random-rotation-ensembles/blob/5346f29855eb87241e616f6599f360eba12437dc/randomrotation.py
    was applied.

    Returns
    -------
    M : np.ndarray
        A uniformly distributed rotation 3 x 3 matrix
    """
    q, r = np.linalg.qr(np.random.normal(size=(3, 3)))
    M = np.dot(q, np.diag(np.sign(np.diag(r))))
    if np.linalg.det(M) < 0:  # Fixing the flipping
        M[:, 0] = -M[:, 0]  # det(M)=1
    return M


def writeVoxels(arr, filename, vecMin, vecRes):
    """DEPRECACTED: Use writeCube instead"""
    logger.warning("writeVoxels is deprecated. Please use writeCube instead.")
    writeCube(arr, filename, vecMin, vecRes)


def writeCube(arr, filename, vecMin, vecRes):
    """Writes 3D array to cube file

    Parameters
    ----------
    arr: np.ndarray
        3D array with volumetric data.
    filename: str
        string with the filename of the cubefile
    vecMin: np.ndarray
        3D vector denoting the minimal corner of the grid
    vecRes: np.ndarray
        3D vector denoting the resolution of the grid in each dimension in Angstrom
    """
    vecRes = np.array(vecRes)
    vecMin = np.array(vecMin)

    # conversion to gaussian units
    L = 1 / 0.52917725
    gauss_bin = vecRes * L

    minCorner = L * (vecMin + 0.5 * vecRes)

    ngrid = arr.shape

    # write header
    with open(filename, "w") as outFile:
        outFile.write("CUBE FILE\n")
        outFile.write("OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n")
        outFile.write(
            "%5d %12.6f %12.6f %12.6f\n" % (1, minCorner[0], minCorner[1], minCorner[2])
        )
        outFile.write("%5d %12.6f %12.6f %12.6f\n" % (ngrid[0], gauss_bin[0], 0, 0))
        outFile.write("%5d %12.6f %12.6f %12.6f\n" % (ngrid[1], 0, gauss_bin[1], 0))
        outFile.write("%5d %12.6f %12.6f %12.6f\n" % (ngrid[2], 0, 0, gauss_bin[2]))
        outFile.write(
            "%5d %12.6f %12.6f %12.6f %12.6f\n"
            % (1, 0, minCorner[0], minCorner[1], minCorner[2])
        )

        # main loop
        cont = 0

        for i in range(ngrid[0]):
            for j in range(ngrid[1]):
                for k in range(ngrid[2]):
                    outFile.write(f"{arr[i][j][k]:13.5g}")
                    if np.mod(cont, 6) == 5:
                        outFile.write("\n")
                    cont += 1


def readCube(fname):
    """Read 3D numpy array from CUBE file

    Parameters
    ----------
    fname: str
        CUBE file path

    Returns
    -------
    data: np.ndarray
        3D numpy array with the volumetric data
    meta: dict
        Dictionary with the metadata of the CUBE file
    """

    def _get(f):
        ll = f.readline().strip().split()
        return int(ll[0]), map(float, ll[1:])

    meta = {}
    with open(fname, "r") as f:
        f.readline()
        f.readline()  # ignore comments
        natm, meta["org"] = _get(f)
        nx, meta["xvec"] = _get(f)
        ny, meta["yvec"] = _get(f)
        nz, meta["zvec"] = _get(f)
        meta["atoms"] = [_get(f) for i in range(natm)]
        data = np.zeros((nx * ny * nz))
        idx = 0
        for line in f:
            for val in line.strip().split():
                data[idx] = float(val)
                idx += 1
    data = np.reshape(data, (nx, ny, nz))
    return data, meta


def _get_pdb_entities(pdbids):
    import requests

    pdbids = ensurelist(pdbids)

    pdbstring = '"' + '","'.join(pdbids) + '"'
    query = f"""
        {{
            entries(entry_ids:[{pdbstring}]) {{
                rcsb_id
                rcsb_entry_container_identifiers {{
                    polymer_entity_ids
                }}
            }}
        }}
    """
    resp = requests.post("https://data.rcsb.org/graphql", json={"query": query})
    entries = resp.json()["data"]["entries"]

    results = {}
    for entry in entries:
        rcsb_id = entry["rcsb_id"].upper()
        results[rcsb_id] = entry["rcsb_entry_container_identifiers"][
            "polymer_entity_ids"
        ]
    missing = [pdbid for pdbid in pdbids if pdbid.upper() not in results]
    return results, missing


def _get_pdb_entity_sequences(entities):
    import requests
    import re

    if isinstance(entities, dict):
        entity_list = []
        for pdbid in entities:
            for ent in entities[pdbid]:
                entity_list.append(f"{pdbid}_{ent}")
    else:
        entity_list = ensurelist(entities)

    entitystr = '"' + '","'.join(entity_list) + '"'
    query = f"""
        {{
            polymer_entities(entity_ids:[{entitystr}]) {{
                rcsb_id
                entity_poly {{
                    pdbx_seq_one_letter_code
                }}
            }}
        }}
    """
    resp = requests.post("https://data.rcsb.org/graphql", json={"query": query})
    data = resp.json()["data"]["polymer_entities"]

    results = {}
    for entry in data:
        rcsb_id = entry["rcsb_id"].upper()
        results[rcsb_id] = entry["entity_poly"]["pdbx_seq_one_letter_code"]
        results[rcsb_id] = re.sub(r"\([A-Za-z0-9]+\)", "?", results[rcsb_id])

    return results


def calculateAnglesAndDihedrals(bonds, cyclicdih=False):
    """
    Calculate all angles and dihedrals from a set of bonds.
    """
    from moleculekit.cython_utils import calculateAnglesAndDihedrals as _calculate
    from moleculekit.molecule import Molecule

    _, angles, dihedrals = _calculate(bonds, cyclicdih, np.max(bonds) + 1)

    angles = sorted([sorted([angle, angle[::-1]])[0] for angle in angles])
    angles = np.array(angles, dtype=Molecule._dtypes["angles"])

    dihedrals = sorted(
        [sorted([dihedral, dihedral[::-1]])[0] for dihedral in dihedrals]
    )
    dihedrals = np.array(dihedrals, dtype=Molecule._dtypes["dihedrals"])

    if len(dihedrals) == 0:
        dihedrals = np.zeros((0, 4), dtype=Molecule._dtypes["dihedrals"])
    if len(angles) == 0:
        angles = np.zeros((0, 3), dtype=Molecule._dtypes["angles"])

    return angles, dihedrals


def guessAnglesAndDihedrals(bonds, cyclicdih=False):
    """
    Calculate all angles and dihedrals from a set of bonds.
    """
    logger.warning(
        "guessAnglesAndDihedrals is deprecated. Please use calculateAnglesAndDihedrals instead."
    )
    return calculateAnglesAndDihedrals(bonds, cyclicdih)


def assertSameAsReferenceDir(compareDir, outdir="."):
    """Check if files in refdir are present in the directory given as second argument AND their content matches.

    Raise an exception if not."""

    import filecmp
    import os

    toCompare = os.listdir(compareDir)
    match, mismatch, error = filecmp.cmpfiles(
        outdir, compareDir, toCompare, shallow=False
    )
    if len(mismatch) != 0 or len(error) != 0 or len(match) != len(toCompare):
        logger.error(
            f"Mismatch while checking directory {outdir} versus reference {compareDir}"
        )
        logger.error(f"Files being checked: {toCompare}")
        for f in mismatch:
            logger.error(
                f"    diff {os.path.join(outdir, f)} {os.path.join(compareDir, f)}"
            )
        raise Exception("Mismatch in regression testing.")


def natsorted(items):
    import re

    def natural_sort_key(s, _nsre=re.compile("([0-9]+)")):
        return [
            int(text) if text.isdigit() else text.lower() for text in _nsre.split(s)
        ]

    return sorted(items, key=natural_sort_key)


def file_diff(file, reference):
    import difflib

    with open(reference) as f:
        reflines = f.readlines()
    with open(file) as f:
        newlines = f.readlines()

    diff = difflib.unified_diff(
        reflines, newlines, fromfile=reference, tofile=file, n=1
    )
    diff = list(diff)
    if len(diff):
        raise RuntimeError("".join(diff))


def folder_diff(folder, reference, ignore_ftypes=(".log", ".txt")):
    import filecmp

    def list_files(filepath):
        paths = []
        for root, _, files in os.walk(filepath):
            for ff in files:
                if os.path.splitext(ff)[-1] not in ignore_ftypes:
                    paths.append(os.path.relpath(os.path.join(root, ff), filepath))
        return paths

    files = list_files(reference)
    match, mismatch, error = filecmp.cmpfiles(folder, reference, files, shallow=False)

    if len(mismatch) != 0 or len(error) != 0 or len(match) != len(files):
        for ff in mismatch:
            reffile = os.path.join(folder, ff)
            newfile = os.path.join(reference, ff)
            file_diff(newfile, reffile)


def find_executable(execname):
    import shutil

    exe = shutil.which(execname, mode=os.X_OK)
    if not exe:
        return None

    if os.path.islink(exe):
        if os.path.isabs(os.readlink(exe)):
            exe = os.readlink(exe)
        else:
            exe = os.path.join(os.path.dirname(exe), os.readlink(exe))
    return exe


def wait_for_port(port, host="127.0.0.1", timeout=240.0, _logger=False):
    """Wait until a port starts accepting TCP connections.
    Args:
        port (int): Port number.
        host (str): Host address on which the port should exist.
        timeout (float): In seconds. How long to wait before raising errors.
    Raises:
        TimeoutError: The port isn't accepting connection after time specified in `timeout`.
    """
    import time
    import socket

    if _logger:
        print(f"Waiting for port {host}:{port} to start accepting connections")

    start_time = time.time()
    while True:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                break
        except OSError as ex:
            time.sleep(0.5)
            if time.time() - start_time >= timeout:
                raise TimeoutError(
                    f"Waited too long for the port {port} on host {host} to start accepting connections."
                ) from ex


def check_port(port, host="127.0.0.1", timeout=120):
    import socket

    try:
        with socket.create_connection((host, port), timeout=timeout):
            pass
        return True
    except OSError:
        return False


def string_to_tempfile(content, ext):
    from tempfile import NamedTemporaryFile

    f = NamedTemporaryFile(delete=False, suffix="." + ext)
    f.write(content.encode("ascii", "ignore"))
    f.close()
    return f.name


def opm(pdbid, keep=False, keepaltloc="A", validateElements=True):
    from moleculekit.opm import get_opm_pdb

    logger.warning(
        "Function opm() has been deprecated. Please use moleculekit.opm.get_opm_pdb instead."
    )
    get_opm_pdb(
        pdbid, keep=keep, keepaltloc=keepaltloc, validateElements=validateElements
    )


def rotation_matrix_from_vectors(vec1, vec2):
    """Find the rotation matrix that aligns vec1 to vec2

    Taken from: https://stackoverflow.com/a/67767180

    Parameters
    ----------
    vec1 : np.ndarray
        A 3d "source" vector
    vec2 : np.ndarray
        A 3d "destination" vector

    Returns
    -------
    mat : np.ndarray
        A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    import numpy as np

    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (
        vec2 / np.linalg.norm(vec2)
    ).reshape(3)
    v = np.cross(a, b)
    if any(v):  # if not all zeros then
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
    else:
        return np.eye(3)  # cross of all zeros only occurs on identical directions
