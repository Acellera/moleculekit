# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
import numpy as np
import ctypes as ct
import os
from moleculekit.home import home
import platform
import tempfile
import logging
from unittest import TestCase


logger = logging.getLogger(__name__)

try:
    libdir = home(libDir=True)
    if platform.system() == "Windows":
        tmalignlib = ct.cdll.LoadLibrary(os.path.join(libdir, "tmalign.dll"))
    else:
        tmalignlib = ct.cdll.LoadLibrary(os.path.join(libdir, "tmalign.so"))
except Exception:
    tmalignlib = None


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


def molTMscore(mol, ref, selCAmol, selCAref):
    """Calculates the TMscore between two Molecules

    Parameters
    ----------
    mol : :class:`Molecule <moleculekit.molecule.Molecule>` object
        A Molecule containing a single or multiple frames
    ref : :class:`Molecule <moleculekit.molecule.Molecule>` object
        A reference Molecule containing a single frame. Will automatically keep only ref.frame.
    selCAmol : numpy.ndarray
        An atomselection array of booleans or indexes of the CA atoms for mol
    selCAref : numpy.ndarray
        An atomselection array of booleans or indexes of the CA atoms for ref

    Returns
    -------
    tmscoreRef : numpy.ndarray
        TMscore normalized by length of ref
    rmsd : numpy.ndarray
        RMSD only OF COMMON RESIDUES for all frames. This is not the same as a full protein RMSD!!!

    Examples
    --------
    tmscore, rmsd = molTMscore(mol, ref, mol.atomselect('protein'), ref.atomselect('protein'))
    """
    from moleculekit.util import sequenceID
    from moleculekit.molecule import _residueNameTable

    if tmalignlib is None:
        raise RuntimeError(
            "Failed to load tmalign libs. Check that moleculekit is installed correctly."
        )

    def calculateVariables(currmol):
        res = sequenceID(
            (currmol.resid, currmol.insertion, currmol.segid, currmol.chain)
        )
        caidx = currmol.name == "CA"
        res = np.unique(res)
        reslen = len(res)
        # Calculate the protein sequence
        seq = "".join([_residueNameTable[x] for x in currmol.resname[caidx]])
        seq = ct.c_char_p(seq.encode("utf-8"))

        # Keep only CA coordinates
        coords = currmol.coords[caidx, :, :].copy()
        return reslen, res.astype(np.int32), seq, coords

    mol = mol.copy()
    ref = ref.copy()
    mol.filter(selCAmol, _logger=False)
    ref.filter(selCAref, _logger=False)
    ref.dropFrames(keep=ref.frame)

    reslenMOL, residMOL, seqMOL, coordsMOL = calculateVariables(mol)
    reslenREF, residREF, seqREF, coordsREF = calculateVariables(ref)

    # DLLEXPORT void tmalign(int xlen, int ylen, int* xresno, int* yresno, char* seqx, char* seqy,
    # float* xcoor, float* ycoor, int nframes,
    # double *TM1, double *TM2, double *rmsd)
    # tmalignlib.tmalign.argtypes = [ct.c_int, ct.c_int, ct.POINTER(ct.c_int), ct.POINTER(ct.c_int), ct.c_char_p, ct.c_char_p,
    #                                ct.POINTER(ct.c_float), ct.POINTER(ct.c_float), ct.c_int,
    #                                ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double)]
    resTM1 = (ct.c_double * mol.numFrames)()
    resTM2 = (ct.c_double * mol.numFrames)()
    resRMSD = (ct.c_double * mol.numFrames)()
    tmalignlib.tmalign(
        ct.c_int(reslenREF),
        ct.c_int(reslenMOL),
        residREF.ctypes.data_as(ct.POINTER(ct.c_int32)),
        residMOL.ctypes.data_as(ct.POINTER(ct.c_int32)),
        seqREF,
        seqMOL,
        coordsREF.ctypes.data_as(ct.POINTER(ct.c_float)),
        coordsMOL.ctypes.data_as(ct.POINTER(ct.c_float)),
        ct.c_int(mol.numFrames),
        ct.byref(resTM1),
        ct.byref(resTM2),
        ct.byref(resRMSD),
    )
    resTM1 = np.ctypeslib.as_array(resTM1)
    resRMSD = np.ctypeslib.as_array(resRMSD)
    return resTM1.astype(np.float32), resRMSD.astype(np.float32)


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
    from scipy.spatial.distance import cdist

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
    """Writes grid free energy to cube file

    Parameters
    ----------
    arr: np.ndarray
        3D array with volumetric data.
    filename: str
        string with the filename of the cubefile
    vecMin: np.ndarray
        3D vector denoting the minimal corner of the grid
    vecRes: np.ndarray
        3D vector denoting the resolution of the grid in each dimension
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


def opm(pdb, keep=False, keepaltloc="A"):
    """Download a molecule from the OPM.

    Parameters
    ----------
    pdb: str
        The 4-letter PDB code
    keep: bool
        If False, removes the DUM atoms. If True, it keeps them.

    Returns
    -------
    mol: Molecule
        The oriented molecule

    thickness: float or None
        The bilayer thickness (both layers)

    Examples
    --------
    >>> mol, thickness = opm("1z98")
    >>> mol.numAtoms
    7902
    >>> thickness
    28.2
    >>> _, thickness = opm('4u15')
    >>> thickness is None
    True

    """
    import urllib.request
    import re
    from moleculekit.support import string_to_tempfile
    from moleculekit.molecule import Molecule

    response = urllib.request.urlopen(
        f"https://storage.googleapis.com/opm-assets/pdb/{pdb.lower()}.pdb"
    )
    text = response.read()
    tempfile = string_to_tempfile(text.decode("ascii"), "pdb")

    mol = Molecule(tempfile, keepaltloc=keepaltloc)
    if not keep:
        mol.filter("not resname DUM")

    # Assuming the half-thickness is the last word in the matched line
    # REMARK      1/2 of bilayer thickness:   14.1
    tmp = open(tempfile)
    pattern = re.compile("^REMARK.+thickness")

    thickness = None
    for line in tmp:
        if re.match(pattern, line):
            thickness = 2.0 * float(line.split()[-1])
            break
    tmp.close()
    os.unlink(tempfile)

    return mol, thickness


def guessAnglesAndDihedrals(bonds, cyclicdih=False):
    """
    Generate a guess of angle and dihedral N-body terms based on a list of bond index pairs.
    """

    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(np.unique(bonds))
    g.add_edges_from([tuple(b) for b in bonds])

    angles = []
    for n in g.nodes():
        neighbors = list(g.neighbors(n))
        for e1 in range(len(neighbors)):
            for e2 in range(e1 + 1, len(neighbors)):
                angles.append((neighbors[e1], n, neighbors[e2]))

    angles = sorted([sorted([angle, angle[::-1]])[0] for angle in angles])
    angles = np.array(angles, dtype=np.uint32)

    dihedrals = []
    for a1 in range(len(angles)):
        for a2 in range(a1 + 1, len(angles)):
            a1a = angles[a1]
            a2a = angles[a2]
            a2f = a2a[
                ::-1
            ]  # Flipped a2a. We don't need flipped a1a as it produces the flipped versions of these 4
            if np.all(a1a[1:] == a2a[:2]) and (cyclicdih or (a1a[0] != a2a[2])):
                dihedrals.append(list(a1a) + [a2a[2]])
            if np.all(a1a[1:] == a2f[:2]) and (cyclicdih or (a1a[0] != a2f[2])):
                dihedrals.append(list(a1a) + [a2f[2]])
            if np.all(a2a[1:] == a1a[:2]) and (cyclicdih or (a2a[0] != a1a[2])):
                dihedrals.append(list(a2a) + [a1a[2]])
            if np.all(a2f[1:] == a1a[:2]) and (cyclicdih or (a2f[0] != a1a[2])):
                dihedrals.append(list(a2f) + [a1a[2]])

    dihedrals = sorted(
        [sorted([dihedral, dihedral[::-1]])[0] for dihedral in dihedrals]
    )
    dihedrals = np.array(dihedrals, dtype=np.uint32)

    if len(dihedrals) == 0:
        dihedrals = np.zeros((0, 4), dtype=np.uint32)
    if len(angles) == 0:
        angles = np.zeros((0, 3), dtype=np.uint32)

    return angles, dihedrals


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


class _TestUtils(TestCase):
    @classmethod
    def setUpClass(self):
        from moleculekit.molecule import Molecule

        self.moldiala = Molecule(os.path.join(home(dataDir="pdb"), "alanine.pdb"))

    def test_guessAnglesDihedrals(self):
        from moleculekit.home import home
        from moleculekit.molecule import Molecule

        mol = Molecule(os.path.join(home(dataDir="test-molecule-utils"), "NH4.pdb"))
        angles, dihedrals = guessAnglesAndDihedrals(mol.bonds)

        self.assertTrue(angles.dtype == np.uint32, "Returned wrong dtype for angles")
        self.assertTrue(
            dihedrals.dtype == np.uint32, "Returned wrong dtype for dihedrals"
        )
        self.assertTrue(
            np.all(angles.shape == (6, 3)), "Returned wrong number of angles"
        )
        self.assertTrue(
            np.all(dihedrals.shape == (0, 4)), "Returned wrong number of dihedrals"
        )

    def test_tmscore(self):
        from moleculekit.molecule import Molecule

        expectedTMscore = np.array(
            [
                0.21418524,
                0.2367377,
                0.23433833,
                0.21362964,
                0.20935164,
                0.20279461,
                0.27012895,
                0.22675238,
                0.21230793,
                0.2372011,
            ]
        )
        expectedRMSD = np.array(
            [
                3.70322128,
                3.43637027,
                3.188193,
                3.84455877,
                3.53053882,
                3.46781854,
                2.93777629,
                2.97978692,
                2.70792428,
                2.63051318,
            ]
        )

        mol = Molecule(os.path.join(home(dataDir="tmscore"), "filtered.pdb"))
        mol.read(os.path.join(home(dataDir="tmscore"), "traj.xtc"))
        ref = Molecule(os.path.join(home(dataDir="tmscore"), "ntl9_2hbb.pdb"))
        tmscore, rmsd = molTMscore(
            mol, ref, mol.atomselect("protein"), ref.atomselect("protein")
        )

        self.assertTrue(np.allclose(tmscore, expectedTMscore))
        self.assertTrue(np.allclose(rmsd, expectedRMSD))

    def test_mol_rmsd(self):
        mol = self.moldiala
        mol2 = mol.copy()
        mol2.rotateBy(rotationMatrix([1, 0, 0], np.pi / 3))
        rmsd = molRMSD(mol, mol2, np.arange(mol.numAtoms), np.arange(mol2.numAtoms))

        assert np.allclose(rmsd, 5.4344)

    def test_orientOnAxes(self):
        omol = orientOnAxes(self.moldiala)

        covariance = np.cov(omol.coords[:, :, 0].T)
        _, eigenvectors = np.linalg.eigh(covariance)

        assert np.allclose(np.diag(eigenvectors), np.array([1, 1, 1]))
        assert (
            eigenvectors[~np.eye(eigenvectors.shape[0], dtype=bool)].max() < 1e-8
        )  # off diagonals close to 0

    def test_missingChain(self):
        mol = self.moldiala.copy()

        try:
            _missingChain(mol)
        except RuntimeError:
            pass
        else:
            self.fail("_missingChain() did not raise RuntimeError!")

        mol.chain[:] = "A"
        try:
            _missingChain(mol)
        except RuntimeError:
            self.fail("_missingChain() raised RuntimeError unexpectedly!")

        mol.chain[6] = ""
        try:
            _missingChain(mol)
        except RuntimeError:
            pass
        else:
            self.fail("_missingChain() did not raise RuntimeError!")

    def test_missingSegid(self):
        mol = self.moldiala.copy()

        try:
            _missingSegID(mol)
        except RuntimeError:
            pass
        else:
            self.fail("_missingSegID() did not raise RuntimeError!")

        mol.segid[:] = "A"
        try:
            _missingSegID(mol)
        except RuntimeError:
            self.fail("_missingSegID() raised RuntimeError unexpectedly!")

        mol.segid[6] = ""
        try:
            _missingSegID(mol)
        except RuntimeError:
            pass
        else:
            self.fail("_missingSegID() did not raise RuntimeError!")

    def test_maxDistance(self):
        dist = maxDistance(self.moldiala)
        assert np.allclose(dist, 10.771703745561421)


if __name__ == "__main__":
    import unittest

    unittest.main(verbosity=2)
