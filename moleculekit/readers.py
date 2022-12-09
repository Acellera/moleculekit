# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
import numpy as np
from moleculekit.util import sequenceID
from moleculekit.periodictable import elements_from_masses
from moleculekit.util import ensurelist
from moleculekit.molecule import Molecule, mol_equal
from contextlib import contextmanager
from glob import glob
import unittest
import os
import re
import logging

logger = logging.getLogger(__name__)


@contextmanager
def openFileOrStringIO(strData, mode=None):
    from io import StringIO
    import os

    if isinstance(strData, StringIO):
        try:
            yield strData
        finally:
            strData.close()
    elif os.path.exists(strData):
        try:
            fi = open(strData, mode)
            yield fi
        finally:
            fi.close()


# Pandas NA values taken from https://github.com/pydata/pandas/blob/6645b2b11a82343e5f07b15a25a250f411067819/pandas/io/common.py
# Removed NA because it's natrium!
_NA_VALUES = set(
    [
        "-1.#IND",
        "1.#QNAN",
        "1.#IND",
        "-1.#QNAN",
        "#N/A N/A",
        "#N/A",
        "N/A",
        "#NA",
        "NULL",
        "NaN",
        "-NaN",
        "nan",
        "-nan",
        "",
    ]
)


class FormatError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class Topology(object):
    def __init__(self, pandasdata=None):
        self.record = []
        self.serial = []
        self.name = []
        self.altloc = []
        self.element = []
        self.resname = []
        self.chain = []
        self.resid = []
        self.insertion = []
        self.occupancy = []
        self.beta = []
        self.segid = []
        self.bonds = []
        self.charge = []
        self.masses = []
        self.angles = []
        self.dihedrals = []
        self.impropers = []
        self.atomtype = []
        self.bondtype = []
        self.formalcharge = []
        self.crystalinfo = None

        if pandasdata is not None:
            for field in self.__dict__:
                fielddat = pandasdata.get(field)
                if fielddat is not None and not np.all(fielddat.isnull()):
                    if (
                        fielddat.dtype == object
                    ):  # If not all are NaN replace NaNs with default values
                        pandasdata.loc[fielddat.isnull(), field] = ""
                    else:
                        pandasdata.loc[fielddat.isnull(), field] = 0
                    self.__dict__[field] = fielddat.tolist()

    @property
    def atominfo(self):
        return [
            "record",
            "serial",
            "name",
            "altloc",
            "element",
            "resname",
            "chain",
            "resid",
            "insertion",
            "occupancy",
            "beta",
            "segid",
            "charge",
            "masses",
            "atomtype",
        ]

    def fromMolecule(self, mol):
        for field in self.__dict__:
            data = mol.__dict__[field]
            if data is None:
                continue
            if isinstance(data, np.ndarray):
                self.__dict__[field] = data.tolist()
            self.__dict__[field] = data


class Trajectory(object):
    def __init__(
        self, coords=None, box=None, boxangles=None, fileloc=None, step=None, time=None
    ):
        self.coords = []
        self.box = []
        self.boxangles = []
        self.fileloc = []
        self.step = []
        self.time = []
        if coords is not None:
            if coords.ndim == 2:
                coords = coords[:, :, np.newaxis]
            self.coords = coords

            nframes = self.numFrames
            if box is None:
                self.box = np.zeros((3, nframes), np.float32)
            if boxangles is None:
                self.boxangles = np.zeros((3, nframes), np.float32)
            if step is None:
                self.step = np.arange(nframes, dtype=int)
            if time is None:
                self.time = np.zeros(nframes, dtype=np.float32)
        if box is not None:
            self.box = box
        if boxangles is not None:
            self.boxangles = boxangles
        if fileloc is not None:
            self.fileloc = fileloc
        if step is not None:
            self.step = step
        if time is not None:
            self.time = time

    @property
    def numFrames(self):
        return self.coords.shape[2]


class TopologyInconsistencyError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class MolFactory(object):
    """This class converts Topology and Trajectory data into Molecule objects"""

    @staticmethod
    def construct(
        topos, trajs, filename, frame, validateElements=True, uniqueBonds=False
    ):
        from moleculekit.molecule import Molecule

        topos = ensurelist(topos)
        trajs = ensurelist(trajs)
        if len(topos) != len(trajs):
            raise RuntimeError(
                "Different number of topologies ({}) and trajectories ({}) were read from {}".format(
                    len(topos), len(trajs), filename
                )
            )

        mols = []
        for topo, traj in zip(topos, trajs):
            natoms = MolFactory._getNumAtoms(topo, traj, filename)

            mol = Molecule()
            if topo is not None:
                mol._emptyTopo(natoms)
                MolFactory._parseTopology(
                    mol,
                    topo,
                    filename,
                    validateElements=validateElements,
                    uniqueBonds=uniqueBonds,
                )
            if traj is not None:
                if natoms is not None:
                    mol._emptyTraj(natoms)
                MolFactory._parseTraj(mol, traj, filename, frame)

            mols.append(mol)

        if len(mols) == 1:
            return mols[0]
        else:
            return mols

    @staticmethod
    def _getNumAtoms(topo, traj, filename):
        toponatoms = None
        trajnatoms = None

        if topo is not None:
            natoms = []
            # Checking number of atoms that were read in the topology file for each field are the same
            for field in topo.atominfo:
                if len(topo.__dict__[field]) != 0:
                    natoms.append(len(topo.__dict__[field]))
            natoms = np.unique(natoms)
            if len(natoms) == 0:
                raise RuntimeError(f"No atoms were read from file {filename}.")
            if len(natoms) != 1:
                raise TopologyInconsistencyError(
                    "Different number of atoms read from file {} for different fields: {}.".format(
                        filename, natoms
                    )
                )
            toponatoms = natoms[0]
        if traj is not None and len(traj.coords):
            trajnatoms = traj.coords.shape[0]

        if (
            toponatoms is not None
            and trajnatoms is not None
            and toponatoms != trajnatoms
        ):
            raise TopologyInconsistencyError(
                "Different number of atoms in topology ({}) and trajectory ({}) for "
                "file {}".format(toponatoms, trajnatoms, filename)
            )

        if toponatoms is not None:
            return toponatoms
        elif trajnatoms is not None:
            return trajnatoms

    @staticmethod
    def _elementChecks(mol, filename):
        # Check for non-legit elements and capitalize them correctly

        from moleculekit.periodictable import periodictable

        virtual_sites = ["Vs"]

        misnamed_element_map = {"So": "Na"}  # Map badly guessed elements
        # Don't issue multiple times the same warning for the same element renaming
        issued_warnings = []

        for i in range(mol.numAtoms):
            el = mol.element[i].capitalize()  # Standardize capitalization of elements

            if el in misnamed_element_map:  # Check if element has a common misnaming
                if el not in issued_warnings:
                    logger.warning(
                        f"Element {el} doesn't exist in the periodic table. "
                        f"Assuming it was meant to be element {misnamed_element_map[el]} and renaming it."
                    )
                    issued_warnings.append(el)
                el = misnamed_element_map[el]

            # If it's still not in the periodic table of elements throw error
            if el in virtual_sites:
                if el not in issued_warnings:
                    logger.info(
                        f"Read element {el} which might correspond to a virtual site atom"
                    )
                    issued_warnings.append(el)
            elif el not in periodictable:
                raise RuntimeError(
                    f"Element {el} was read in file {filename} but was not found in the periodictable. "
                    "To disable this check, pass `validateElements=False` to the Molecule constructor or read method."
                )

            mol.element[i] = el  # Set standardized element

    @staticmethod
    def _parseTopology(mol, topo, filename, validateElements=True, uniqueBonds=True):
        from moleculekit.molecule import Molecule
        import io

        for field in topo.__dict__:
            if field == "crystalinfo":
                mol.crystalinfo = topo.crystalinfo
                continue

            # Skip on empty new field data
            if (
                topo.__dict__[field] is None
                or len(topo.__dict__[field]) == 0
                or np.all([x is None for x in topo.__dict__[field]])
            ):
                continue

            # Empty strings in future float dtype arrays cannot be converted to numbers so here we set them to 0
            if (
                np.issubdtype(mol._dtypes[field], np.floating)
                and isinstance(topo.__dict__[field], list)
                and isinstance(topo.__dict__[field][0], str)
            ):
                topo.__dict__[field] = [
                    x if len(x.strip()) else 0 for x in topo.__dict__[field]
                ]

            newfielddata = np.array(topo.__dict__[field], dtype=mol._dtypes[field])

            # Objects could be ints for example but we want them as str
            if mol._dtypes[field] == object and len(newfielddata) != 0:
                newfielddata = np.array([str(x) for x in newfielddata], dtype=object)

            mol.__dict__[field] = newfielddata

        if len(mol.bonds) != 0 and len(topo.bondtype) == 0:
            mol.bondtype = np.empty(
                mol.bonds.shape[0], dtype=Molecule._dtypes["bondtype"]
            )
            mol.bondtype[:] = "un"

        mol.element = mol._guessMissingElements()
        if validateElements:
            MolFactory._elementChecks(mol, filename)

        if uniqueBonds and len(mol.bonds):
            from moleculekit.molecule import calculateUniqueBonds

            mol.bonds, mol.bondtype = calculateUniqueBonds(mol.bonds, mol.bondtype)

        if isinstance(filename, io.StringIO):
            topoloc = "StringIO"
            fileloc = [["StringIO", 0]]
            viewname = "StringIO"
        else:
            topoloc = os.path.abspath(filename)
            fileloc = [[filename, 0]]
            viewname = os.path.basename(filename)

        mol.topoloc = topoloc
        mol.fileloc = fileloc
        mol.viewname = viewname

    @staticmethod
    def _parseTraj(mol, traj, filename, frame):
        from moleculekit.molecule import Molecule
        import io

        ext = (
            os.path.splitext(filename)[1][1:]
            if not isinstance(filename, io.StringIO)
            else ""
        )

        if len(traj.coords):
            assert (
                traj.coords.ndim == 3
            ), f"{ext} reader must return 3D coordinates array for file {filename}"
            assert (
                traj.coords.shape[1] == 3
            ), f"{ext} reader must return 3 values in 2nd dimension for file {filename}"

        for field in ["coords", "box", "boxangles"]:
            # All necessary dtype conversions. Only perform if necessary since it otherwise reallocates memory!
            if getattr(traj, field) is None or len(getattr(traj, field)) == 0:
                continue

            if getattr(traj, field).dtype != Molecule._dtypes[field]:
                setattr(
                    traj, field, getattr(traj, field).astype(Molecule._dtypes["coords"])
                )

            # Check for array contiguity
            if not getattr(traj, field).flags["C_CONTIGUOUS"]:
                setattr(traj, field, np.ascontiguousarray(getattr(traj, field)))

        if len(traj.coords):
            mol.coords = traj.coords

        if traj.box is None:
            mol.box = np.zeros((3, 1), dtype=Molecule._dtypes["box"])
        else:
            mol.box = np.array(traj.box)
            if mol.box.ndim == 1:
                mol.box = mol.box[:, np.newaxis]

        if traj.boxangles is None:
            mol.boxangles = np.zeros((3, 1), dtype=Molecule._dtypes["boxangles"])
        else:
            mol.boxangles = np.array(traj.boxangles)
            if mol.boxangles.ndim == 1:
                mol.boxangles = mol.boxangles[:, np.newaxis]

        # mol.fileloc = traj.fileloc
        mol.step = np.hstack(traj.step).astype(int)
        mol.time = np.hstack(traj.time)

        if ext in _TRAJECTORY_READERS and frame is None and len(traj.coords):
            # Writing hidden index file containing number of frames in trajectory file
            if not isinstance(filename, io.StringIO) and os.path.isfile(filename):
                MolFactory._writeNumFrames(filename, mol.numFrames)
            ff = range(mol.numFrames)
            # tr.step = tr.step + traj[-1].step[-1] + 1
        elif frame is None:
            ff = [0]
        elif frame is not None:
            ff = [frame]
        else:
            raise AssertionError("Should not reach here")

        if isinstance(filename, io.StringIO):
            mol.fileloc = [["StringIO", j] for j in ff]
        else:
            mol.fileloc = [[filename, j] for j in ff]

    @staticmethod
    def _writeNumFrames(filepath, numFrames):
        """Write the number of frames in a hidden file. Allows us to check for trajectory length issues before projecting

        Parameters
        ----------
        filepath : str
            Path to trajectory file
        numFrames : int
            Number of frames in trajectory file
        """
        filepath = os.path.abspath(filepath)
        filedir = os.path.dirname(filepath)
        basename = os.path.basename(filepath)
        numframefile = os.path.join(filedir, f".{basename}.numframes")
        if not os.path.exists(numframefile) or (
            os.path.exists(numframefile)
            and (os.path.getmtime(numframefile) < os.path.getmtime(filepath))
        ):
            try:
                with open(numframefile, "w") as f:
                    f.write(str(numFrames))
            except Exception:
                pass


def XYZread(filename, frame=None, topoloc=None):
    topo = Topology()

    frames = []
    firstconf = True
    with open(filename, "r") as f:
        while True:
            line = f.readline()
            if line == "":
                break
            natoms = int(line.split()[0])
            f.readline()
            coords = []
            for i in range(natoms):
                s = f.readline().split()
                if firstconf:
                    topo.record.append("HETATM")
                    topo.serial.append(i + 1)
                    topo.element.append(s[0])
                    topo.name.append(s[0])
                    topo.resname.append("MOL")
                coords.append(s[1:4])
            frames.append(np.vstack(coords))
            firstconf = False

    coords = np.stack(frames, axis=2)
    traj = Trajectory(coords=coords)
    return MolFactory.construct(topo, traj, filename, frame)


def GJFread(filename, frame=None, topoloc=None):
    # $rungauss
    # %chk=ts_rhf
    # %mem=2000000
    # #T RHF/6-31G(d) TEST
    #
    # C9H8O4
    #
    # 0,1
    # C1,2.23927,-0.379063,0.262961
    # C2,0.842418,1.92307,-0.424949
    # C3,2.87093,0.845574,0.272238

    # import re
    # regex = re.compile('(\w+),([-+]?[0-9]*\.?[0-9]+),([-+]?[0-9]*\.?[0-9]+),([-+]?[0-9]*\.?[0-9]+)')
    import re

    topo = Topology()
    coords = []

    with open(filename, "r") as f:
        for line in f:
            pieces = re.split(r"\s+|,", line.strip())
            if (
                len(pieces) == 4
                and not line.startswith("$")
                and not line.startswith("%")
                and not line.startswith("#")
            ):
                topo.record.append("HETATM")
                topo.element.append(pieces[0])
                topo.name.append(pieces[0])
                topo.resname.append("MOL")
                coords.append([float(s) for s in pieces[1:4]])
        topo.serial = range(len(topo.record))

    coords = np.vstack(coords)[:, :, np.newaxis]
    traj = Trajectory(coords=coords)
    return MolFactory.construct(topo, traj, filename, frame)


def MOL2read(filename, frame=None, topoloc=None, singlemol=True, validateElements=True):
    from moleculekit.periodictable import periodictable

    topologies = []  # Allow reading of multi-mol MOL2 files
    topologies.append(Topology())
    topo = topologies[-1]
    coordinates = [[]]
    coords = coordinates[-1]
    section = None

    molnum = 0
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:  # Ignore empty lines
                continue

            if line.startswith("@<TRIPOS>MOLECULE"):
                section = None
                atom_attr_num = 0
                molnum += 1
                if molnum > 1:  # New Molecule, create new topology
                    if singlemol:
                        break
                    topologies.append(Topology())
                    topo = topologies[-1]
                    coordinates.append([])
                    coords = coordinates[-1]
            if line.startswith("@<TRIPOS>ATOM"):
                section = "atom"
                continue
            if line.startswith("@<TRIPOS>BOND"):
                section = "bond"
                continue
            if line.startswith("@<TRIPOS>UNITY_ATOM_ATTR"):
                section = "atom_attr"
                continue
            if line.startswith("@<TRIPOS>"):  # Skip all other sections
                section = None
                continue

            if section == "atom":
                pieces = line.strip().split()
                topo.record.append("HETATM")
                topo.serial.append(int(pieces[0]))
                topo.name.append(pieces[1])
                coords.append([float(x) for x in pieces[2:5]])
                topo.atomtype.append(pieces[5])
                topo.formalcharge.append(0)
                if len(pieces) > 6:
                    topo.resid.append(int(pieces[6]))
                if len(pieces) > 7:
                    topo.resname.append(pieces[7][:3])
                if len(pieces) > 8:
                    topo.charge.append(float(pieces[8]))

                element = pieces[5].split(".")[0]
                if element == "Du":
                    # Corina, SYBYL and Tripos dummy atoms. Don't catch Du.C which is a dummy carbon and should be recognized as carbon
                    # We are using the PDB convention of left aligning the name in 4 spaces to signify ion/metal
                    topo.name[-1] = f"{topo.name[-1]:<4}"
                    topo.element.append("")
                elif element in periodictable:
                    topo.element.append(element)
                else:
                    topo.element.append("")
            elif section == "bond":
                pieces = line.strip().split()
                if len(pieces) < 4:
                    raise RuntimeError(
                        f"Less than 4 values encountered in bonds definition in line {line}"
                    )
                topo.bonds.append([int(pieces[1]) - 1, int(pieces[2]) - 1])
                topo.bondtype.append(pieces[3])
            elif section == "atom_attr":
                if atom_attr_num == 0:
                    atom_attr_idx, atom_attr_num = map(int, line.strip().split())
                    continue
                if line.startswith("charge"):
                    formalcharge = int(line.strip().split()[-1])
                    idx = topo.serial.index(atom_attr_idx)
                    topo.formalcharge[idx] = formalcharge
                atom_attr_num -= 1

    trajectories = []
    for cc in coordinates:
        trajectories.append(Trajectory(coords=np.vstack(cc)[:, :, np.newaxis]))

    if singlemol:
        if molnum > 1:
            logger.warning(
                f"Mol2 file {filename} contained multiple molecules. Only the first was read."
            )
        return MolFactory.construct(
            topologies[0],
            trajectories[0],
            filename,
            frame,
            validateElements=validateElements,
        )
    else:
        return MolFactory.construct(
            topologies, trajectories, filename, frame, validateElements=validateElements
        )


def MAEread(fname, frame=None, topoloc=None):
    """Reads maestro files.

    Parameters
    ----------
    fname : str
        .mae file

    Returns
    -------
    topo : Topology
    coords : list of lists
    """
    from moleculekit.periodictable import periodictable_by_number

    section = None
    section_desc = False
    section_data = False

    topo = Topology()
    coords = []
    heteros = []

    # Stripping starting and trailing whitespaces which confuse csv reader
    with open(fname, "r") as csvfile:
        stripped = (row.strip() for row in csvfile)

        import csv

        reader = csv.reader(
            stripped, delimiter=" ", quotechar='"', skipinitialspace=True
        )
        for row in reader:
            if len(row) == 0:
                continue

            if row[0].startswith("m_atom"):
                section = "atoms"
                section_desc = True
                section_cols = []
            elif row[0].startswith("m_bond"):
                section = "bonds"
                section_desc = True
                section_cols = []
            elif row[0].startswith("m_PDB_het_residues"):
                section = "hetresidues"
                section_desc = True
                section_cols = []
            elif (
                section_desc and row[0] == ":::"
            ):  # Once the section description has finished create a map from names to columns
                section_dict = dict(zip(section_cols, range(len(section_cols))))
                section_desc = False
                section_data = True
            elif section_data and (row[0] == ":::" or row[0] == "}"):
                section_data = False
            else:  # It's actual data
                if section_desc:
                    section_cols.append(row[0])

                # Reading the data of the atoms section
                if section == "atoms" and section_data:
                    topo.record.append("ATOM")
                    row = np.array(row)
                    if len(row) != len(section_dict):  # TODO: fix the reader
                        raise RuntimeError(
                            "{} has {} fields in the m_atom section description, but {} fields in the "
                            "section data. Please check for missing fields in the mae file.".format(
                                fname, len(section_dict), len(row)
                            )
                        )
                    row[row == "<>"] = 0
                    if "i_pdb_PDB_serial" in section_dict:
                        topo.serial.append(row[section_dict["i_pdb_PDB_serial"]])
                    if "s_m_pdb_atom_name" in section_dict:
                        topo.name.append(row[section_dict["s_m_pdb_atom_name"]].strip())
                    if "i_m_atomic_number" in section_dict:
                        num = int(row[section_dict["i_m_atomic_number"]])
                        topo.element.append(periodictable_by_number[num].symbol)
                    if "s_m_pdb_residue_name" in section_dict:
                        topo.resname.append(
                            row[section_dict["s_m_pdb_residue_name"]].strip()
                        )
                    if "i_m_residue_number" in section_dict:
                        topo.resid.append(int(row[section_dict["i_m_residue_number"]]))
                    if "s_m_chain_name" in section_dict:
                        topo.chain.append(row[section_dict["s_m_chain_name"]])
                    if "s_pdb_segment_id" in section_dict:
                        topo.segid.append(row[section_dict["s_pdb_segment_id"]])
                    if "r_m_pdb_occupancy" in section_dict:
                        topo.occupancy.append(
                            float(row[section_dict["r_m_pdb_occupancy"]])
                        )
                    if "r_m_pdb_tfactor" in section_dict:
                        topo.beta.append(float(row[section_dict["r_m_pdb_tfactor"]]))
                    if "s_m_insertion_code" in section_dict:
                        topo.insertion.append(
                            row[section_dict["s_m_insertion_code"]].strip()
                        )
                    if "" in section_dict:
                        topo.element.append("")  # TODO: Read element
                    if "" in section_dict:
                        topo.altloc.append(
                            ""
                        )  # TODO: Read altloc. Quite complex actually. Won't bother.
                    if "r_m_x_coord" in section_dict:
                        coords.append(
                            [
                                float(row[section_dict["r_m_x_coord"]]),
                                float(row[section_dict["r_m_y_coord"]]),
                                float(row[section_dict["r_m_z_coord"]]),
                            ]
                        )
                    topo.masses.append(0)

                # Reading the data of the bonds section
                if section == "bonds" and section_data:
                    topo.bonds.append(
                        [
                            int(row[section_dict["i_m_from"]]) - 1,
                            int(row[section_dict["i_m_to"]]) - 1,
                        ]
                    )  # -1 to conver to 0 indexing

                # Reading the data of the hetero residue section
                if section == "hetresidues" and section_data:
                    heteros.append(row[section_dict["s_pdb_het_name"]].strip())

    for h in heteros:
        topo.record[topo.resname == h] = "HETATM"

    coords = np.vstack(coords)[:, :, np.newaxis]
    traj = Trajectory(coords=coords)
    return MolFactory.construct(topo, traj, fname, frame)


def _getPDB(pdbid):
    from moleculekit.util import string_to_tempfile
    from moleculekit.home import home

    # Try loading it from the pdb data directory
    tempfile = False
    localpdb = os.path.join(home(dataDir="pdb"), pdbid.lower() + ".pdb")
    if os.path.isfile(localpdb):
        logger.info(f"Using local copy for {pdbid}: {localpdb}")
        filepath = localpdb
    else:
        # or the PDB website
        from moleculekit.rcsb import _getRCSBtext

        logger.info(f"Attempting PDB query for {pdbid}")
        url = f"https://files.rcsb.org/download/{pdbid}.pdb"
        text = _getRCSBtext(url)
        filepath = string_to_tempfile(text.decode("ascii"), "pdb")
        tempfile = True
    return filepath, tempfile


def pdbGuessElementByName(elements, names, onlymissing=True):
    """
    https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/tutorials/pdbintro.html#misalignment which states that elements
    should be right-aligned in columns 13-14 unless it's a 4 letter name when it would end up being left-aligned.
    """
    from moleculekit.periodictable import periodictable
    from collections import defaultdict
    import re

    allelements = [str(el).upper() for el in periodictable]
    newelements = np.array(["" for _ in range(len(elements))], dtype=object)

    alternatives = defaultdict(list)

    if not onlymissing or (elements.dtype == np.float64 and np.all(np.isnan(elements))):
        noelem = np.arange(len(elements))
    else:
        noelem = np.where((elements.str.strip() == "") | elements.isnull())[0]

    uqnames = np.unique(names[noelem])
    for name in uqnames:
        name = (
            re.sub(r"\d", " ", name[0]) + name[1:]
        )  # Remove numbers from first column
        elem = None
        if name[0] == " ":  # If there is no letter in col 13 then col 14 is the element
            elem = name[1]
        else:
            if (
                name[-1] == " "
            ):  # If it's not a 4 letter name then it's a two letter element
                elem = name[0] + name[1].lower()
            else:  # With 4 letter name it could be either a 1 letter element or 2 letter element
                if name[0] in allelements:
                    elem = name[0]
                if name[:2].upper() in allelements:
                    if elem is not None:
                        altelem = name[0] + name[1].lower()
                        alternatives[(elem, altelem)].append(name)
                    else:
                        elem = name[0] + name[1].lower()
        if elem is not None:
            elem = elem.strip()
        if elem is not None and len(elem) != 0 and elem not in periodictable:
            logger.warning(
                f'Element guessing failed for atom with name {name} as the guessed element "{elem}" was not found in '
                "the periodic table. Check for incorrect column alignment in the PDB file or report "
                "to moleculekit issue tracker."
            )
            elem = None
        newelements[names == name] = elem

    for elem, altelem in alternatives:
        names = np.unique(alternatives[(elem, altelem)])
        namestr = '"' + '" "'.join(names) + '"'
        altelemname = periodictable[altelem].name
        logger.warning(
            f"Atoms with names {namestr} were guessed as element {elem} but could also be {altelem} ({altelemname})."
            f" If this is a case, you can correct them with mol.set('element', '{altelem}', sel='name {namestr}')"
        )
    return noelem, newelements[noelem]


def PDBread(
    filename,
    mode="pdb",
    frame=None,
    topoloc=None,
    validateElements=True,
    uniqueBonds=True,
):
    from pandas import read_fwf
    import io

    tempfile = False
    if (
        not isinstance(filename, io.StringIO)
        and not os.path.isfile(filename)
        and len(filename) == 4
    ):  # Could be a PDB id. Try to load it from the PDB website
        filename, tempfile = _getPDB(filename)

    """
    COLUMNS        DATA  TYPE    FIELD        DEFINITION
    -------------------------------------------------------------------------------------
     1 -  6        Record name   "ATOM  "
     7 - 11        Integer       serial       Atom  serial number.
    13 - 16        Atom          name         Atom name.
    17             Character     altLoc       Alternate location indicator.
    18 - 20        Residue name  resName      Residue name.
    22             Character     chainID      Chain identifier.
    23 - 26        Integer       resSeq       Residue sequence number.
    27             AChar         iCode        Code for insertion of residues.
    31 - 38        Real(8.3)     x            Orthogonal coordinates for X in Angstroms.
    39 - 46        Real(8.3)     y            Orthogonal coordinates for Y in Angstroms.
    47 - 54        Real(8.3)     z            Orthogonal coordinates for Z in Angstroms.
    55 - 60        Real(6.2)     occupancy    Occupancy.
    61 - 66        Real(6.2)     tempFactor   Temperature  factor.
    77 - 78        LString(2)    element      Element symbol, right-justified.
    79 - 80        LString(2)    charge       Charge  on the atom.
    """
    if mode == "pdb":
        topocolspecs = [
            (0, 6),
            (6, 11),
            (12, 16),
            (16, 17),
            (17, 21),
            (21, 22),
            (22, 26),
            (26, 27),
            (54, 60),
            (60, 66),
            (72, 76),
            (76, 78),
            (78, 80),
        ]
        toponames = (
            "record",
            "serial",
            "name",
            "altloc",
            "resname",
            "chain",
            "resid",
            "insertion",
            "occupancy",
            "beta",
            "segid",
            "element",
            "formalcharge",
        )
    elif mode == "pdbqt":
        # http://autodock.scripps.edu/faqs-help/faq/what-is-the-format-of-a-pdbqt-file
        # The rigid root contains one or more PDBQT-style ATOM or HETATM records. These records resemble their
        # traditional PDB counterparts, but diverge in columns 71-79 inclusive (where the first character in the line
        # corresponds to column 1). The partial charge is stored in columns 71-76 inclusive (in %6.3f format, i.e.
        # right-justified, 6 characters wide, with 3 decimal places). The AutoDock atom-type is stored in columns 78-79
        # inclusive (in %-2.2s format, i.e. left-justified and 2 characters wide..
        topocolspecs = [
            (0, 6),
            (6, 11),
            (12, 16),
            (16, 17),
            (17, 21),
            (21, 22),
            (22, 26),
            (26, 27),
            (54, 60),
            (60, 66),
            (70, 76),
            (77, 79),
        ]
        toponames = (
            "record",
            "serial",
            "name",
            "altloc",
            "resname",
            "chain",
            "resid",
            "insertion",
            "occupancy",
            "beta",
            "charge",
            "atomtype",
        )
    topodtypes = {
        "record": str,
        "serial": int,
        "name": str,
        "altloc": str,
        "resname": str,
        "chain": str,
        "resid": int,
        "insertion": str,
        "occupancy": np.float32,
        "beta": np.float32,
        "segid": str,
        "element": str,
        "atomtype": str,
        "charge": np.float32,
        "formalcharge": np.int32,
    }
    coordcolspecs = [(30, 38), (38, 46), (46, 54)]
    coordnames = ("x", "y", "z")

    """
    COLUMNS       DATA  TYPE      FIELD        DEFINITION
    -------------------------------------------------------------------------
     1 -  6        Record name    "CONECT"
     7 - 11        Integer        serial       Atom  serial number
    12 - 16        Integer        serial       Serial number of bonded atom
    17 - 21        Integer        serial       Serial  number of bonded atom
    22 - 26        Integer        serial       Serial number of bonded atom
    27 - 31        Integer        serial       Serial number of bonded atom
    """
    bondcolspecs = [(6, 11), (11, 16), (16, 21), (21, 26), (26, 31)]
    bondnames = ("serial1", "serial2", "serial3", "serial4", "serial5")

    """
    COLUMNS       DATA  TYPE    FIELD          DEFINITION
    -------------------------------------------------------------
     1 -  6       Record name   "CRYST1"
     7 - 15       Real(9.3)     a              a (Angstroms).
    16 - 24       Real(9.3)     b              b (Angstroms).
    25 - 33       Real(9.3)     c              c (Angstroms).
    34 - 40       Real(7.2)     alpha          alpha (degrees).
    41 - 47       Real(7.2)     beta           beta (degrees).
    48 - 54       Real(7.2)     gamma          gamma (degrees).
    56 - 66       LString       sGroup         Space  group.
    67 - 70       Integer       z              Z value.
    """
    cryst1colspecs = [
        (6, 15),
        (15, 24),
        (24, 33),
        (33, 40),
        (40, 47),
        (47, 54),
        (55, 66),
        (66, 70),
    ]
    cryst1names = ("a", "b", "c", "alpha", "beta", "gamma", "sGroup", "z")

    """
    Guessing columns for REMARK 290 SMTRY from the example since the specs don't define them
              1         2         3         4         5         6         7
    01234567890123456789012345678901234567890123456789012345678901234567890
    REMARK 290   SMTRY1   1  1.000000  0.000000  0.000000        0.00000
    REMARK 290   SMTRY2   1  0.000000  1.000000  0.000000        0.00000
    REMARK 290   SMTRY3   1  0.000000  0.000000  1.000000        0.00000
    REMARK 290   SMTRY1   2 -1.000000  0.000000  0.000000       36.30027
    REMARK 290   SMTRY2   2  0.000000 -1.000000  0.000000        0.00000
    REMARK 290   SMTRY3   2  0.000000  0.000000  1.000000       59.50256
    REMARK 290   SMTRY1   3 -1.000000  0.000000  0.000000        0.00000
    REMARK 290   SMTRY2   3  0.000000  1.000000  0.000000       46.45545
    REMARK 290   SMTRY3   3  0.000000  0.000000 -1.000000       59.50256
    REMARK 290   SMTRY1   4  1.000000  0.000000  0.000000       36.30027
    REMARK 290   SMTRY2   4  0.000000 -1.000000  0.000000       46.45545
    REMARK 290   SMTRY3   4  0.000000  0.000000 -1.000000        0.00000

    Guessing columns for REMARK 350   BIOMT from the example since the specs don't define them
    REMARK 350   BIOMT1   1 -0.981559  0.191159  0.000000        0.00000
    REMARK 350   BIOMT2   1 -0.191159 -0.981559  0.000000        0.00000
    REMARK 350   BIOMT3   1  0.000000  0.000000  1.000000      -34.13878
    REMARK 350   BIOMT1   2 -0.838088  0.545535  0.000000        0.00000
    REMARK 350   BIOMT2   2 -0.545535 -0.838088  0.000000        0.00000
    REMARK 350   BIOMT3   2  0.000000  0.000000  1.000000      -32.71633
    """
    symmetrycolspecs = [(20, 23), (23, 33), (33, 43), (43, 53), (53, 68)]
    symmetrynames = ("idx", "rot1", "rot2", "rot3", "trans")

    def concatCoords(coords, coorddata):
        if coorddata.tell() != 0:  # Not empty
            coorddata.seek(0)
            parsedcoor = read_fwf(
                coorddata,
                colspecs=coordcolspecs,
                names=coordnames,
                na_values=_NA_VALUES,
                keep_default_na=False,
            )
            if coords is None:
                coords = np.zeros((len(parsedcoor), 3, 0), dtype=np.float32)
            currcoords = np.vstack((parsedcoor.x, parsedcoor.y, parsedcoor.z)).T
            if coords.shape[0] != currcoords.shape[0]:
                logger.warning(
                    "Different number of atoms read in different MODELs in the PDB file. "
                    "Keeping only the first {} model(s)".format(coords.shape[2])
                )
                return coords
            coords = np.append(coords, currcoords[:, :, np.newaxis], axis=2)
        return coords

    teridx = []
    currter = 0
    topoend = False

    cryst1data = io.StringIO()
    topodata = io.StringIO()
    conectdata = io.StringIO()
    coorddata = io.StringIO()
    symmetrydata = io.StringIO()

    coords = None

    with openFileOrStringIO(filename, "r") as f:
        for line in f:
            if line.startswith("CRYST1"):
                cryst1data.write(line)
            if line.startswith("ATOM") or line.startswith("HETATM"):
                coorddata.write(line)
            if (line.startswith("ATOM") or line.startswith("HETATM")) and not topoend:
                topodata.write(line)
                teridx.append(str(currter))
            if line.startswith("TER"):
                currter += 1
            if (mode == "pdb" and line.startswith("END")) or (
                mode == "pdbqt" and line.startswith("ENDMDL")
            ):  # pdbqt should not stop reading at ENDROOT or ENDBRANCH
                topoend = True
            if line.startswith("CONECT"):
                conectdata.write(line)
            if line.startswith("MODEL"):
                coords = concatCoords(coords, coorddata)
                coorddata = io.StringIO()
            if line.startswith(
                "REMARK 290   SMTRY"
            ):  # TODO: Support BIOMT fields. It's a bit more complicated. Can't be done with pandas
                symmetrydata.write(line)

        cryst1data.seek(0)
        topodata.seek(0)
        conectdata.seek(0)
        symmetrydata.seek(0)

        coords = concatCoords(coords, coorddata)

        parsedbonds = read_fwf(
            conectdata,
            colspecs=bondcolspecs,
            names=bondnames,
            na_values=_NA_VALUES,
            keep_default_na=False,
        )
        parsedcryst1 = read_fwf(
            cryst1data,
            colspecs=cryst1colspecs,
            names=cryst1names,
            na_values=_NA_VALUES,
            keep_default_na=False,
        )
        enforceddtypes = {"resname": str, "segid": str}
        parsedtopo = read_fwf(
            topodata,
            colspecs=topocolspecs,
            names=toponames,
            na_values=_NA_VALUES,
            keep_default_na=False,
            dtype=enforceddtypes,
            delimiter="\r\t",
        )  # , dtype=topodtypes)
        parsedsymmetry = read_fwf(
            symmetrydata,
            colspecs=symmetrycolspecs,
            names=symmetrynames,
            na_values=_NA_VALUES,
            keep_default_na=False,
        )

    if len(parsedtopo) == 0:
        raise RuntimeError(f"No atoms could be read from PDB file {filename}")

    # Before stripping guess elements from atomname as the spaces are significant
    if "element" in parsedtopo:
        if parsedtopo.element.dtype != object:
            parsedtopo.element = parsedtopo.element.astype(object)
        idx, newelem = pdbGuessElementByName(parsedtopo.element, parsedtopo.name)
        if len(idx):
            parsedtopo.iloc[idx, parsedtopo.columns.get_loc("element")] = newelem

    for field in topodtypes:
        if (
            field in parsedtopo
            and topodtypes[field] == str
            and parsedtopo[field].dtype == object
        ):
            parsedtopo[field] = parsedtopo[field].str.strip()

    # Fixing PDB format charges which can come after the number
    if "formalcharge" in parsedtopo and parsedtopo.formalcharge.dtype == "object":
        parsedtopo.formalcharge = parsedtopo.formalcharge.str.strip()
        charges = np.zeros(len(parsedtopo.formalcharge), dtype=np.float32)
        for i, c in enumerate(parsedtopo.formalcharge):
            if not isinstance(c, str):
                continue
            if len(c) > 1:
                if c[1] == "-":
                    charges[i] = -1 * float(c[0])
                elif c[1] == "+":
                    charges[i] = float(c[0])
            elif len(c):
                charges[i] = float(c)
        parsedtopo.formalcharge = charges

    # Fixing hexadecimal index and resids
    # Support for reading hexadecimal
    if parsedtopo.serial.dtype == "object":
        logger.warning(
            'Non-integer values were read from the PDB "serial" field. Dropping PDB values and assigning new ones.'
        )
        if len(np.unique(parsedtopo.serial)) == len(parsedtopo.serial):
            parsedtopo.serial = (
                sequenceID(parsedtopo.serial) + 1
            )  # Indexes should start from 1 in PDB
        else:
            parsedtopo.serial = np.arange(1, len(parsedtopo.serial) + 1)
    if parsedtopo.resid.dtype == "object":
        logger.warning(
            'Non-integer values were read from the PDB "resid" field. Dropping PDB values and assigning new ones.'
        )
        parsedtopo.resid = sequenceID(parsedtopo.resid)

    if parsedtopo.insertion.dtype == np.float64 and not np.all(
        np.isnan(parsedtopo.insertion)
    ):
        # Trying to minimize damage from resids overflowing into insertion field
        logger.warning(
            'Integer values detected in "insertion" field. Your resids might be overflowing. Take care'
        )
        parsedtopo.insertion = [
            str(int(x)) if not np.isnan(x) else "" for x in parsedtopo.insertion
        ]

    if len(parsedtopo) > 99999:
        logger.warning(
            "Reading PDB file with more than 99999 atoms. Bond information can be wrong."
        )

    crystalinfo = {}
    if len(parsedcryst1):
        crystalinfo = parsedcryst1.iloc[0].to_dict()
        if isinstance(crystalinfo["sGroup"], str) or not np.isnan(
            crystalinfo["sGroup"]
        ):
            crystalinfo["sGroup"] = crystalinfo["sGroup"].split()
    if len(parsedsymmetry):
        numcopies = int(len(parsedsymmetry) / 3)
        crystalinfo["numcopies"] = numcopies
        crystalinfo["rotations"] = parsedsymmetry[
            ["rot1", "rot2", "rot3"]
        ].values.reshape((numcopies, 3, 3))
        crystalinfo["translations"] = parsedsymmetry["trans"].values.reshape(
            (numcopies, 3)
        )

    topo = Topology(parsedtopo)

    # Bond formatting part
    # TODO: Speed this up. This is the slowest part for large PDB files. From 700ms to 7s
    serials = parsedtopo.serial.values
    # if isinstance(serials[0], str) and np.any(serials == '*****'):
    #     logger.info('Non-integer serials were read. For safety we will discard all bond information and serials will be assigned automatically.')
    #     topo.serial = np.arange(1, len(serials)+1, dtype=int)
    if np.max(parsedbonds.max()) > np.max(serials):
        logger.info(
            "Bond indexes in PDB file exceed atom indexes. For safety we will discard all bond information."
        )
    else:
        mapserials = np.full(np.max(serials) + 1, -1, dtype=np.int64)
        mapserials[serials] = list(range(len(serials)))
        for i in range(len(parsedbonds)):
            row = parsedbonds.loc[i].tolist()
            topo.bonds += [
                [int(row[0]), int(row[b])] for b in range(1, 5) if not np.isnan(row[b])
            ]

        topo.bonds = np.array(topo.bonds, dtype=np.uint32)
        if topo.bonds.size != 0:
            mappedbonds = mapserials[topo.bonds[:]]
            wrongidx, _ = np.where(
                mappedbonds == -1
            )  # Some PDBs have bonds to non-existing serials... go figure
            if len(wrongidx):
                logger.info(
                    f"Discarding {len(wrongidx)} bonds to non-existing indexes in the PDB file."
                )
            mappedbonds = np.delete(mappedbonds, wrongidx, axis=0)
            topo.bonds = np.array(mappedbonds, dtype=np.uint32)

    if (
        len(topo.segid) and np.all(np.array(topo.segid) == "") and currter != 0
    ):  # If no segid was read, use the TER rows to define segments
        topo.segid = teridx

    if tempfile:
        os.unlink(filename)

    topo.crystalinfo = crystalinfo
    traj = Trajectory(coords=coords)
    return MolFactory.construct(
        topo,
        traj,
        filename,
        frame,
        validateElements=validateElements,
        uniqueBonds=uniqueBonds,
    )


def PDBQTread(filename, frame=None, topoloc=None):
    return PDBread(filename, mode="pdbqt", frame=frame, topoloc=topoloc)


def PRMTOPread(filename, frame=None, topoloc=None, validateElements=True):
    with open(filename, "r") as f:
        topo = Topology()
        uqresnames = []
        residx = []
        bondsidx = []
        angleidx = []
        dihedidx = []
        section = None
        for line in f:
            if line.startswith("%FLAG POINTERS"):
                section = "pointers"
            elif line.startswith("%FLAG ATOM_NAME"):
                section = "names"
            elif line.startswith("%FLAG CHARGE"):
                section = "charges"
            elif line.startswith("%FLAG MASS"):
                section = "masses"
            elif line.startswith("%FLAG ATOM_TYPE_INDEX"):
                section = "type"
            elif line.startswith("%FLAG RESIDUE_LABEL"):
                section = "resname"
            elif line.startswith("%FLAG RESIDUE_POINTER"):
                section = "resstart"
            elif line.startswith("%FLAG BONDS_INC_HYDROGEN") or line.startswith(
                "%FLAG BONDS_WITHOUT_HYDROGEN"
            ):
                section = "bonds"
            elif line.startswith("%FLAG ANGLES_INC_HYDROGEN") or line.startswith(
                "%FLAG ANGLES_WITHOUT_HYDROGEN"
            ):
                section = "angles"
            elif line.startswith("%FLAG DIHEDRALS_INC_HYDROGEN") or line.startswith(
                "%FLAG DIHEDRALS_WITHOUT_HYDROGEN"
            ):
                section = "dihedrals"
            elif line.startswith("%FLAG BOX_DIMENSIONS"):
                section = "box"
            elif line.startswith("%FLAG AMBER_ATOM_TYPE"):
                section = "amberatomtype"
            elif line.startswith("%FLAG"):
                section = None

            if line.startswith("%"):
                continue

            if section == "pointers":
                pass
            elif section == "names":
                fieldlen = 4
                topo.name += [
                    line[i : i + fieldlen].strip()
                    for i in range(0, len(line), fieldlen)
                    if len(line[i : i + fieldlen].strip()) != 0
                ]
            elif section == "charges":
                fieldlen = 16
                topo.charge += [
                    float(line[i : i + fieldlen].strip()) / 18.2223
                    for i in range(0, len(line), fieldlen)
                    if len(line[i : i + fieldlen].strip()) != 0
                ]  # 18.2223 = Scaling factor for charges
            elif section == "masses":
                fieldlen = 16
                topo.masses += [
                    float(line[i : i + fieldlen].strip())
                    for i in range(0, len(line), fieldlen)
                    if len(line[i : i + fieldlen].strip()) != 0
                ]
            elif section == "resname":
                fieldlen = 4
                uqresnames += [
                    line[i : i + fieldlen].strip()
                    for i in range(0, len(line), fieldlen)
                    if len(line[i : i + fieldlen].strip()) != 0
                ]
            elif section == "resstart":
                fieldlen = 8
                residx += [
                    int(line[i : i + fieldlen].strip())
                    for i in range(0, len(line), fieldlen)
                    if len(line[i : i + fieldlen].strip()) != 0
                ]
            elif section == "bonds":
                fieldlen = 8
                bondsidx += [
                    int(line[i : i + fieldlen].strip())
                    for i in range(0, len(line), fieldlen)
                    if len(line[i : i + fieldlen].strip()) != 0
                ]
            elif section == "angles":
                fieldlen = 8
                angleidx += [
                    int(line[i : i + fieldlen].strip())
                    for i in range(0, len(line), fieldlen)
                    if len(line[i : i + fieldlen].strip()) != 0
                ]
            elif section == "dihedrals":
                fieldlen = 8
                dihedidx += [
                    int(line[i : i + fieldlen].strip())
                    for i in range(0, len(line), fieldlen)
                    if len(line[i : i + fieldlen].strip()) != 0
                ]
            elif section == "amberatomtype":
                fieldlen = 4
                topo.atomtype += [
                    line[i : i + fieldlen].strip()
                    for i in range(0, len(line), fieldlen)
                    if len(line[i : i + fieldlen].strip()) != 0
                ]

    if len(topo.name) == 0:
        raise FormatError("No atoms read in PRMTOP file. Trying a different reader.")
    # Replicating unique resnames according to their start and end indeces
    residx.append(len(topo.name) + 1)

    """
    NOTE: the atom numbers in the following arrays that describe bonds, angles, and dihedrals are coordinate array 
    indexes for runtime speed. The true atom number equals the absolute value of the number divided by three, plus one. 
    In the case of the dihedrals, if the fourth atom is negative, this implies that the dihedral is an improper. If the 
    third atom is negative, this implies that the end group interations are to be ignored. End group interactions are 
    ignored, for example, in dihedrals of various ring systems (to prevent double counting of 1-4 interactions) and 
    in multiterm dihedrals.
    """

    for i in range(len(residx) - 1):
        numresatoms = residx[i + 1] - residx[i]
        topo.resname += [uqresnames[i]] * numresatoms
        topo.resid += [i + 1] * numresatoms

    # Processing bond triplets
    for i in range(0, len(bondsidx), 3):
        topo.bonds.append([int(bondsidx[i] / 3), int(bondsidx[i + 1] / 3)])

    # Processing angle quads
    for i in range(0, len(angleidx), 4):
        topo.angles.append(
            [int(angleidx[i] / 3), int(angleidx[i + 1] / 3), int(angleidx[i + 2] / 3)]
        )

    # Processing dihedral quints
    for i in range(0, len(dihedidx), 5):
        atoms = [
            int(dihedidx[i] / 3),
            int(dihedidx[i + 1] / 3),
            abs(int(dihedidx[i + 2] / 3)),
            int(dihedidx[i + 3] / 3),
        ]
        if atoms[3] >= 0:
            topo.dihedrals.append(atoms)
        else:
            atoms[3] = abs(atoms[3])
            topo.impropers.append(atoms)

    # Elements from masses
    topo.element = elements_from_masses(topo.masses)
    return MolFactory.construct(
        topo, None, filename, frame, validateElements=validateElements
    )


def PSFread(filename, frame=None, topoloc=None, validateElements=True):
    import re

    residinsertion = re.compile(r"(\d+)([a-zA-Z])")

    topo = Topology()

    with open(filename, "r") as f:
        mode = None

        for line in f:
            if line.strip() == "":
                mode = None

            if mode == "atom":
                ll = line.split()
                topo.serial.append(ll[0])
                topo.segid.append(ll[1])
                match = residinsertion.findall(ll[2])
                if match:
                    resid = int(match[0][0])
                    insertion = match[0][1]
                else:
                    resid = int(ll[2])
                    insertion = ""
                topo.resid.append(resid)
                topo.insertion.append(insertion)
                topo.resname.append(ll[3])
                topo.name.append(ll[4])
                topo.atomtype.append(ll[5] if ll[5] != "NULL" else "")
                topo.charge.append(float(ll[6]))
                topo.masses.append(float(ll[7]))
            elif mode == "bond":
                ll = line.split()
                for x in range(0, len(ll), 2):
                    topo.bonds.append([int(ll[x]) - 1, int(ll[x + 1]) - 1])
            elif mode == "angle":
                ll = line.split()
                for x in range(0, len(ll), 3):
                    topo.angles.append(
                        [int(ll[x]) - 1, int(ll[x + 1]) - 1, int(ll[x + 2]) - 1]
                    )
            elif mode == "dihedral":
                ll = line.split()
                for x in range(0, len(ll), 4):
                    topo.dihedrals.append(
                        [
                            int(ll[x]) - 1,
                            int(ll[x + 1]) - 1,
                            int(ll[x + 2]) - 1,
                            int(ll[x + 3]) - 1,
                        ]
                    )
            elif mode == "improper":
                ll = line.split()
                for x in range(0, len(ll), 4):
                    topo.impropers.append(
                        [
                            int(ll[x]) - 1,
                            int(ll[x + 1]) - 1,
                            int(ll[x + 2]) - 1,
                            int(ll[x + 3]) - 1,
                        ]
                    )

            if "!NATOM" in line:
                mode = "atom"
            elif "!NBOND" in line:
                mode = "bond"
            elif "!NTHETA" in line:
                mode = "angle"
            elif "!NPHI" in line:
                mode = "dihedral"
            elif "!NIMPHI" in line:
                mode = "improper"

    # Elements from masses
    topo.element = elements_from_masses(topo.masses)
    return MolFactory.construct(
        topo, None, filename, frame, validateElements=validateElements
    )


def XTCread(filename, frame=None, topoloc=None):
    from moleculekit.xtc import read_xtc, read_xtc_frames

    if frame is None:
        coords, box, time, step = read_xtc(filename.encode("UTF-8"))
    else:
        frame = ensurelist(frame)
        coords, box, time, step = read_xtc_frames(
            filename.encode("UTF-8"), np.array(frame, dtype=np.int32)
        )

    if np.size(coords, 2) == 0:
        raise RuntimeError(f"Malformed XTC file. No frames read from: {filename}")
    if np.size(coords, 0) == 0:
        raise RuntimeError(f"Malformed XTC file. No atoms read from: {filename}")

    coords *= 10.0  # Convert from nm to Angstrom
    box *= 10.0  # Convert from nm to Angstrom
    time *= 1e3  # Convert from ps to fs. This seems to be ACEMD3 specific. GROMACS writes other units in time
    nframes = coords.shape[2]
    if len(step) != nframes or np.sum(step) == 0:
        step = np.arange(nframes)
    if len(time) != nframes or np.sum(time) == 0:
        time = np.zeros(nframes, dtype=np.float32)
    return MolFactory.construct(
        None, Trajectory(coords=coords, box=box, step=step, time=time), filename, frame
    )


def XSCread(filename, frame=None, topoloc=None):
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            pieces = line.split()
            if len(pieces) != 19:
                raise RuntimeError(
                    "Incorrect XSC file. Line should contain 19 numbers."
                )
            pieces = np.array(list(map(float, pieces)))

            if np.any(pieces[[2, 3, 4, 6, 7, 8]] != 0):
                raise RuntimeError(
                    "Simulation box has to be rectangular for moleculekit to read it"
                )
            step = pieces[0]
            box = pieces[[1, 5, 9]]

    return MolFactory.construct(
        None,
        Trajectory(
            box=box,
            step=[
                step,
            ],
            time=np.zeros(1, dtype=np.float32),
        ),
        filename,
        frame,
    )


def CRDread(filename, frame=None, topoloc=None):
    # default_name
    #  7196
    #  -7.0046035  10.4479194  20.8320000  -7.3970000   9.4310000  20.8320000
    #  -7.0486898   8.9066002  21.7218220  -7.0486899   8.9065995  19.9421780

    with open(filename, "r") as f:
        lines = f.readlines()

        if lines[0].startswith("*"):
            raise FormatError("CRDread failed. Trying other readers.")

        natoms = None
        try:
            natoms = int(lines[1])
        except Exception:
            logger.warning(
                "Didn't find number of atoms in CRD file. Will read until the end."
            )

        coords = []
        fieldlen = 12
        for line in lines[2:]:  # skip first 2 lines
            coords += [
                float(line[i : i + fieldlen].strip())
                for i in range(0, len(line), fieldlen)
                if len(line[i : i + fieldlen].strip()) != 0
            ]
            if natoms is not None and len(coords) == natoms * 3:
                break

    coords = np.vstack([coords[i : i + 3] for i in range(0, len(coords), 3)])[
        :, :, np.newaxis
    ]
    return MolFactory.construct(None, Trajectory(coords=coords), filename, frame)


def CRDCARDread(filename, frame=None, topoloc=None):
    """https://www.charmmtutorial.org/index.php/CHARMM:The_Basics
    title = * WATER
    title = *  DATE:     4/10/07      4:25:51      CREATED BY USER: USER
    title = *
    Number of atoms (NATOM)       = 6
    Atom number (ATOMNO)          = 1 (just an exmaple)
    Residue number (RESNO)        = 1
    Residue name (RESName)        = TIP3
    Atom type (TYPE)              = OH2
    Coordinate (X)                = -1.30910
    Coordinate (Y)                = -0.25601
    Coordinate (Z)                = -0.24045
    Segment ID (SEGID)            = W
    Residue ID (RESID)            = 1
    Atom weight (Weighting)       = 0.00000

    now what that looks like...

    * WATER
    *  DATE:     4/10/07      4:25:51      CREATED BY USER: USER
    *
        6
        1    1 TIP3 OH2   -1.30910  -0.25601  -0.24045 W    1      0.00000
        2    1 TIP3 H1    -1.85344   0.07163   0.52275 W    1      0.00000
        3    1 TIP3 H2    -1.70410   0.16529  -1.04499 W    1      0.00000
        4    2 TIP3 OH2    1.37293   0.05498   0.10603 W    2      0.00000
        5    2 TIP3 H1     1.65858  -0.85643   0.10318 W    2      0.00000
        6    2 TIP3 H2     0.40780  -0.02508  -0.02820 W    2      0.00000
    """
    coords = []
    topo = Topology()
    with open(filename, "r") as f:
        lines = f.readlines()

        if not lines[0].startswith("*"):
            raise FormatError("CRDCARDread failed. Trying other readers.")

        i = 0
        while lines[i].startswith("*"):
            i += 1

        for line in lines[i + 1 :]:
            pieces = line.split()
            topo.resname.append(pieces[2])
            topo.name.append(pieces[3])
            coords.append([float(x) for x in pieces[4:7]])
            topo.segid.append(pieces[7])
            topo.resid.append(int(pieces[8]))
    coords = np.vstack(coords)[:, :, np.newaxis]
    return MolFactory.construct(topo, Trajectory(coords=coords), filename, frame)


def BINCOORread(filename, frame=None, topoloc=None):
    import struct

    with open(filename, "rb") as f:
        dat = f.read(4)
        fmt = "i"
        natoms = struct.unpack(fmt, dat)[0]
        dat = f.read(natoms * 3 * 8)
        fmt = "d" * (natoms * 3)
        coords = struct.unpack(fmt, dat)
        coords = np.array(coords, dtype=np.float32).reshape((natoms, 3, 1))
    return MolFactory.construct(None, Trajectory(coords=coords), filename, frame)


def MDTRAJread(filename, frame=None, topoloc=None, validateElements=True):
    try:
        import mdtraj as md
    except ImportError:
        raise ImportError(
            "To support extension {} please install the `mdtraj` package".format(
                os.path.splitext(filename)[1]
            )
        )

    traj = md.load(filename, top=topoloc)
    coords = np.swapaxes(np.swapaxes(traj.xyz, 0, 1), 1, 2) * 10
    try:
        step = traj.time / traj.timestep
    except Exception:
        step = [0]
    time = traj.time * 1000  # need to go from picoseconds to femtoseconds

    if traj.unitcell_lengths is None:
        box = None
    else:
        box = traj.unitcell_lengths.T.copy() * 10

    if traj.unitcell_angles is None:
        boxangles = None
    else:
        boxangles = traj.unitcell_angles.T.copy()
    traj = Trajectory(
        coords=coords.copy(), box=box, boxangles=boxangles, step=step, time=time
    )  # Copying coords needed to fix MDtraj stride
    return MolFactory.construct(
        None, traj, filename, frame, validateElements=validateElements
    )


def MDTRAJTOPOread(filename, frame=None, topoloc=None, validateElements=True):
    translate = {
        "serial": "serial",
        "name": "name",
        "element": "element",
        "resSeq": "resid",
        "resName": "resname",
        "chainID": "chain",
        "segmentID": "segid",
    }
    try:
        import mdtraj as md
    except ImportError:
        raise ImportError(
            "To support extension {} please install the `mdtraj` package".format(
                os.path.splitext(filename)[1]
            )
        )

    mdstruct = md.load(filename)
    topology = mdstruct.topology
    table, bonds = topology.to_dataframe()

    topo = Topology()
    for k in table.keys():
        topo.__dict__[translate[k]] = table[k].tolist()

    coords = np.array(mdstruct.xyz.swapaxes(0, 1).swapaxes(1, 2) * 10, dtype=np.float32)
    topo.bonds = bonds[:, :2]
    return MolFactory.construct(
        topo,
        Trajectory(coords=coords),
        filename,
        frame,
        validateElements=validateElements,
    )


def _guess_element_from_name(name):
    from moleculekit.periodictable import periodictable

    name = name.strip()
    if len(name) == 1:
        return name
    name = name[:2]
    if name[0] in periodictable:
        if name in periodictable:
            logger.warning(
                f"Atom name {name} could match either element {name[0]} or {name.capitalize()}. Choosing the first."
            )
        return name[0]
    elif name in periodictable:
        return name
    else:
        return name[0]


def GROTOPread(filename, frame=None, topoloc=None):
    # Reader for GROMACS .top file format:
    # http://manual.gromacs.org/online/top.html
    topo = Topology()
    section = None
    atmidx = []
    with open(filename, "r") as f:
        for line in f:
            if line.startswith(";") or line.startswith("#") or len(line.strip()) == 0:
                continue
            if not line.startswith("[") and section == "atoms":
                pieces = line.split()
                atmidx.append(int(pieces[0]))
                topo.atomtype.append(pieces[1])
                topo.resid.append(pieces[2])
                topo.resname.append(pieces[3])
                topo.name.append(pieces[4])
                topo.charge.append(pieces[6])
                atomtype = re.sub("[0-9]", "", pieces[1])
                topo.element.append(_guess_element_from_name(atomtype))
            if not line.startswith("[") and section == "bonds":
                pieces = line.split()
                topo.bonds.append([int(pieces[0]), int(pieces[1])])
            if not line.startswith("[") and section == "angles":
                pieces = line.split()
                topo.angles.append([int(pieces[0]), int(pieces[1]), int(pieces[2])])
            if not line.startswith("[") and section == "dihedrals":
                pieces = line.split()
                topo.dihedrals.append(
                    [int(pieces[0]), int(pieces[1]), int(pieces[2]), int(pieces[3])]
                )
            if not line.startswith("[") and section == "impropers":
                pieces = line.split()
                topo.impropers.append(
                    [int(pieces[0]), int(pieces[1]), int(pieces[2]), int(pieces[3])]
                )

            if "[ atoms ]" in line:
                section = "atoms"
            elif "[ bonds ]" in line:
                section = "bonds"
            elif "[ angles ]" in line:
                section = "angles"
            elif "[ dihedrals ]" in line:
                section = "dihedrals"
            elif "[ impropers ]" in line:
                section = "impropers"
            elif line.startswith("["):
                section = None

    if section is None and len(topo.name) == 0:
        raise FormatError("No atoms read in GROTOP file. Trying a different reader.")

    atmidx = np.array(atmidx, dtype=int)
    atommapping = np.ones(np.max(atmidx) + 1, dtype=int) * -1
    atommapping[atmidx] = np.arange(len(atmidx))
    if len(topo.bonds):
        topo.bonds = atommapping[np.array(topo.bonds)]
    if len(topo.angles):
        topo.angles = atommapping[np.array(topo.angles)]
    if len(topo.dihedrals):
        topo.dihedrals = atommapping[np.array(topo.dihedrals)]
    if len(topo.impropers):
        topo.impropers = atommapping[np.array(topo.impropers)]

    return MolFactory.construct(topo, None, filename, frame)


def CIFread(filename, frame=None, topoloc=None, zerowarning=True):
    from moleculekit.pdbx.reader.PdbxReader import PdbxReader

    myDataList = []
    ifh = open(filename, "r")
    pRd = PdbxReader(ifh)
    pRd.read(myDataList)
    ifh.close()

    # Taken from http://mmcif.wwpdb.org/docs/pdb_to_pdbx_correspondences.html#ATOMP
    atom_site_mapping = {
        "group_PDB": ("record", str),
        "id": ("serial", int),
        "label_alt_id": ("altloc", str),
        "auth_atom_id": ("name", str),
        "auth_comp_id": ("resname", str),
        "auth_asym_id": ("chain", str),
        "auth_seq_id": ("resid", int),
        "pdbx_PDB_ins_code": ("insertion", str),
        "label_entity_id": ("segid", str),
        "type_symbol": ("element", str),
        "occupancy": ("occupancy", float),
        "B_iso_or_equiv": ("beta", float),
        "pdbx_formal_charge": ("formalcharge", int),
    }
    alternatives = {
        "auth_atom_id": "label_atom_id",
        "auth_comp_id": "label_comp_id",
        "auth_asym_id": "label_asym_id",
        "auth_seq_id": "label_seq_id",
    }
    chem_comp_mapping = {
        "comp_id": ("resname", str),
        "atom_id": ("name", str),
        "alt_atom_id": ("atomtype", str),
        "type_symbol": ("element", str),
        "charge": ("formalcharge", int),
        "partial_charge": ("charge", float),
    }
    cryst1_mapping = {
        "length_a": ("a", float),
        "length_b": ("b", float),
        "length_c": ("c", float),
        "angle_alpha": ("alpha", float),
        "angle_beta": ("beta", float),
        "angle_gamma": ("gamma", float),
        "space_group_name_H-M": ("sGroup", str),
        "Z_PDB": ("z", int),
    }
    bondtype_mapping = {
        "SING": "1",
        "DOUB": "2",
        "TRIP": "3",
        "QUAD": "4",
        "AROM": "ar",
    }

    topo = Topology()

    if len(myDataList) > 1:
        logger.warning(
            "Multiple Data objects in mmCIF. Please report this issue to the moleculekit issue tracker"
        )

    dataObj = myDataList[0]

    def fixDefault(val, dtype):
        if val == "?":
            if dtype == float or dtype == int:
                val = 0
            if dtype == str:
                val = ""
        return dtype(val)

    # Parsing CRYST1 data
    cryst = dataObj.getObj("cell")
    if cryst is not None and cryst.getRowCount() == 1:
        row = cryst.getRow(0)
        attrs = cryst.getAttributeList()

        crystalinfo = {}
        for source_field, target in cryst1_mapping.items():
            if source_field not in attrs:
                continue
            target_field, dtype = target
            val = dtype(fixDefault(row[cryst.getAttributeIndex(source_field)], dtype))
            crystalinfo[target_field] = val

        if "sGroup" in crystalinfo and (
            isinstance(crystalinfo["sGroup"], str)
            or not np.isnan(crystalinfo["sGroup"])
        ):
            crystalinfo["sGroup"] = crystalinfo["sGroup"].split()
        topo.crystalinfo = crystalinfo

    # Parsing ATOM and HETATM data
    allcoords = []
    ideal_allcoords = []
    coords = []
    ideal_coords = []
    currmodel = -1
    firstmodel = None
    if "atom_site" in dataObj.getObjNameList():
        atom_site = dataObj.getObj("atom_site")
        mapping = atom_site_mapping
    elif "chem_comp_atom" in dataObj.getObjNameList():
        atom_site = dataObj.getObj("chem_comp_atom")
        mapping = chem_comp_mapping

    attrs = atom_site.getAttributeList()

    for i in range(atom_site.getRowCount()):
        row = atom_site.getRow(i)

        if "pdbx_PDB_model_num" in atom_site.getAttributeList():
            modelid = row[atom_site.getAttributeIndex("pdbx_PDB_model_num")]
        else:
            modelid = -1
        # On a new model, restart coords and append the old ones
        if currmodel != -1 and currmodel != modelid:
            currmodel = modelid
            allcoords.append(coords)
            coords = []
            if len(ideal_coords):
                ideal_allcoords.append(ideal_coords)
                ideal_coords = []

        if currmodel == -1:
            currmodel = modelid
            firstmodel = modelid

        if currmodel == firstmodel:
            for source_field, target in mapping.items():
                if source_field not in attrs:
                    if source_field not in alternatives:
                        continue
                    else:
                        source_field = alternatives[source_field]
                target_field, dtype = target
                val = row[atom_site.getAttributeIndex(source_field)]
                val = dtype(fixDefault(val, dtype))
                if (
                    source_field
                    in (
                        "label_alt_id",
                        "alt_atom_id",
                        "auth_asym_id",
                        "label_asym_id",
                        "pdbx_PDB_ins_code",
                    )
                    and val == "."
                ):  # Atoms without altloc seem to be stored with a dot
                    val = ""
                topo.__dict__[target_field].append(val)

        if "Cartn_x" in attrs:
            coords.append(
                [
                    row[atom_site.getAttributeIndex("Cartn_x")],
                    row[atom_site.getAttributeIndex("Cartn_y")],
                    row[atom_site.getAttributeIndex("Cartn_z")],
                ]
            )
        elif "model_Cartn_x" in attrs:
            coords.append(
                [
                    fixDefault(
                        row[atom_site.getAttributeIndex("model_Cartn_x")], float
                    ),
                    fixDefault(
                        row[atom_site.getAttributeIndex("model_Cartn_y")], float
                    ),
                    fixDefault(
                        row[atom_site.getAttributeIndex("model_Cartn_z")], float
                    ),
                ]
            )
            ideal_coords.append(
                [
                    fixDefault(
                        row[atom_site.getAttributeIndex("pdbx_model_Cartn_x_ideal")],
                        float,
                    ),
                    fixDefault(
                        row[atom_site.getAttributeIndex("pdbx_model_Cartn_y_ideal")],
                        float,
                    ),
                    fixDefault(
                        row[atom_site.getAttributeIndex("pdbx_model_Cartn_z_ideal")],
                        float,
                    ),
                ]
            )

    if "chem_comp_bond" in dataObj.getObjNameList():
        bond_site = dataObj.getObj("chem_comp_bond")
        for i in range(bond_site.getRowCount()):
            row = bond_site.getRow(i)
            name1 = row[bond_site.getAttributeIndex("atom_id_1")]
            name2 = row[bond_site.getAttributeIndex("atom_id_2")]
            bondtype = row[bond_site.getAttributeIndex("value_order")]
            idx1 = topo.name.index(name1)
            idx2 = topo.name.index(name2)
            topo.bonds.append([idx1, idx2])
            topo.bondtype.append(bondtype_mapping[bondtype])

    if len(coords) != 0:
        allcoords.append(coords)
    if len(ideal_coords) != 0:
        ideal_allcoords.append(ideal_coords)

    allcoords = np.stack(allcoords, axis=2).astype(np.float32)
    if len(ideal_allcoords):
        ideal_allcoords = np.stack(ideal_allcoords, axis=2).astype(np.float32)

    coords = allcoords
    if np.any(np.all(allcoords == 0, axis=1)) and len(ideal_allcoords):
        if np.any(np.all(ideal_allcoords == 0, axis=1)) and zerowarning:
            logger.warning(
                "Found [0, 0, 0] coordinates in molecule! Proceed with caution."
            )
        coords = ideal_allcoords

    return MolFactory.construct(topo, Trajectory(coords=coords), filename, frame)


_ATOM_TYPE_REG_EX = re.compile(r"^\S+x\d+$")


def RTFread(filename, frame=None, topoloc=None):
    def _guessElement(name):
        import re

        name = re.sub("[0-9]*$", "", name)
        name = name.capitalize()
        return name

    def _guessMass(element):
        from moleculekit.periodictable import periodictable

        return periodictable[element].mass

    allatomtypes = []
    bonds = []
    improper_indices = []
    # netcharge = 0.
    mass_by_type = dict()
    element_by_type = dict()
    topo = Topology()

    with open(filename, "r") as f:
        for line in f:
            if line.startswith("MASS "):
                pieces = line.split()
                atom_type = pieces[2]
                mass_by_type[atom_type] = float(pieces[3])
                element_by_type[atom_type] = pieces[4]
                allatomtypes.append(atom_type)
            # elif line.startswith("RESI "):
            #     pieces = line.split()
            #     netcharge = float(pieces[2])
            elif line.startswith("ATOM "):
                pieces = line.split()
                topo.name.append(pieces[1])
                topo.atomtype.append(pieces[2])
                topo.charge.append(float(pieces[3]))
            elif line.startswith("BOND "):
                pieces = line.split()
                bonds.append([topo.name.index(pieces[1]), topo.name.index(pieces[2])])
            elif line.startswith("IMPR "):
                pieces = line.split()
                improper_indices.append(
                    [
                        topo.name.index(pieces[1]),
                        topo.name.index(pieces[2]),
                        topo.name.index(pieces[3]),
                        topo.name.index(pieces[4]),
                    ]
                )

    # if there weren't any "MASS" lines, we need to guess them
    for idx in range(len(topo.name)):
        atype = topo.atomtype[idx]
        name = topo.name[idx]
        if atype not in element_by_type:
            element_by_type[atype] = _guessElement(name)
            logger.info(
                "Guessing element {} for atom {} type {}".format(
                    element_by_type[atype], name, atype
                )
            )
        if atype not in mass_by_type:
            mass_by_type[atype] = _guessMass(element_by_type[atype])
        if atype not in allatomtypes:
            allatomtypes.append(atype)

    topo.element = np.array(
        [element_by_type[t].capitalize() for t in topo.atomtype], dtype=object
    )
    topo.masses = np.array([mass_by_type[t] for t in topo.atomtype], dtype=np.float32)
    topo.bonds = np.vstack(bonds)

    improper_indices = np.array(improper_indices).astype(np.uint32)
    if improper_indices.ndim == 1:
        improper_indices = improper_indices[:, np.newaxis]
    topo.impropers = improper_indices

    for type_ in topo.atomtype:
        if re.match(_ATOM_TYPE_REG_EX, type_):
            raise ValueError(
                f'Atom type {type_} is incompatible. It cannot finish with "x" + number!'
            )

    return MolFactory.construct(topo, None, filename, frame)


def PREPIread(filename, frame=None, topoloc=None):
    names = []
    atomtypes = []
    charges = []
    impropers = []

    atominfosection = False
    impropersection = False
    with open(filename, "r") as f:
        for i, line in enumerate(f):
            if (i == 4) and line.split()[1] != "INT":
                raise ValueError("Invalid prepi format line 5")
            if (i == 5) and line.strip() != "CORRECT     OMIT DU   BEG":
                raise ValueError("Invalid prepi format line 6")
            if i == 10:
                atominfosection = True
            if line.startswith("IMPROPER"):
                impropersection = True
                continue

            if atominfosection and line.strip() == "":
                atominfosection = False
            if impropersection and line.strip() == "":
                impropersection = False

            if line.strip() == "":
                continue

            if atominfosection:
                pieces = line.split()
                names.append(pieces[1].upper())
                atomtypes.append(pieces[2])
                charges.append(float(pieces[10]))
            if impropersection:
                impropernames = [impn.upper() for impn in line.split()]
                not_found = np.setdiff1d(impropernames, names)
                if len(not_found):
                    logger.warning(
                        f"Could not find atoms: {', '.join(not_found)}. Skipping reading of improper '{line.strip()}'"
                    )
                    continue
                impropers.append([names.index(impn) for impn in impropernames])

    impropers = np.array(impropers).astype(np.uint32)
    if impropers.ndim == 1:
        impropers = impropers[:, np.newaxis]

    for type_ in atomtypes:
        if re.match(_ATOM_TYPE_REG_EX, type_):
            raise ValueError(
                f'Atom type {type_} is incompatible. It cannot finish with "x" + number!'
            )

    topo = Topology()
    topo.name = np.array(names, dtype=object)
    topo.atomtype = np.array(atomtypes, dtype=object)
    topo.charge = np.array(charges, dtype=np.float32)
    topo.impropers = impropers

    return MolFactory.construct(topo, None, filename, frame)


def SDFread(filename, frame=None, topoloc=None):
    # Some (mostly correct) info here: www.nonlinear.com/progenesis/sdf-studio/v0.9/faq/sdf-file-format-guidance.aspx
    # Format is correctly specified here: https://www.daylight.com/meetings/mug05/Kappler/ctfile.pdf
    chargemap = {"7": -3, "6": -2, "5": -1, "0": 0, "3": 1, "2": 2, "1": 3, "4": 0}

    with open(filename, "r") as f:
        lines = f.readlines()
        molend = False
        for line in lines:
            if "V3000" in line:
                raise RuntimeError(
                    "Moleculekit does not support parsing V3000 SDF files yet."
                )
            if line.strip() == "$$$$":
                molend = True
            elif molend and line.strip() != "":
                logger.warning(
                    "MoleculeKit will only read the first molecule from the SDF file."
                )
                break

        topo = Topology()
        coords = []
        mol_start = 0

        n_atoms = int(lines[mol_start + 3][:3])
        n_bonds = int(lines[mol_start + 3][3:6])

        coord_start = mol_start + 4
        bond_start = coord_start + n_atoms
        bond_end = bond_start + n_bonds
        for n in range(coord_start, bond_start):
            line = lines[n]
            coords.append(
                [
                    float(line[:10].strip()),
                    float(line[10:20].strip()),
                    float(line[20:30].strip()),
                ]
            )
            atom_symbol = line[31:34].strip()
            topo.record.append("HETATM")
            topo.element.append(atom_symbol)
            topo.name.append(atom_symbol)
            topo.serial.append(n - coord_start)
            topo.formalcharge.append(chargemap[line[36:39].strip()])

        for n in range(bond_start, bond_end):
            line = lines[n]
            idx1 = line[:3].strip()
            idx2 = line[3:6].strip()
            bond_type = line[6:9].strip()
            topo.bonds.append([int(idx1) - 1, int(idx2) - 1])
            topo.bondtype.append(bond_type)

        for line in lines[bond_end:]:
            if line.strip() == "$$$$":
                break
            if line.startswith("M  CHG"):  # These charges are more correct
                pairs = line.strip().split()[3:]
                for cc in range(0, len(pairs), 2):
                    topo.formalcharge[int(pairs[cc]) - 1] = int(pairs[cc + 1])

    traj = Trajectory(coords=np.vstack(coords))
    return MolFactory.construct(topo, traj, filename, frame)


def MMTFread(filename, frame=None, topoloc=None, validateElements=True):
    from mmtf import fetch, parse_gzip, parse

    if len(filename) == 4 and not os.path.isfile(filename):
        data = fetch(filename)
    elif filename.endswith(".gz"):
        data = parse_gzip(filename)
    else:
        data = parse(filename)

    topo = Topology()

    gtypes = np.array(data.group_type_list).reshape(data.num_models, -1)
    first_coords_only = False
    if np.any(gtypes != gtypes[0]):
        logger.warning(
            "File contained multiple models with different topologies. Reading only first"
        )
        first_coords_only = True

    a_idx = 0
    g_idx = 0
    # Read only first model topology
    for chain_idx in range(data.chains_per_model[0]):
        n_groups = data.groups_per_chain[chain_idx]
        # Iterate over residues
        for _ in range(n_groups):
            group_first_a_idx = a_idx
            gr = data.group_list[data.group_type_list[g_idx]]
            resid = data.group_id_list[g_idx]
            ins = data.ins_code_list[g_idx]

            # Iterate over atoms in residue
            for name, elem, fchg in zip(
                gr["atomNameList"], gr["elementList"], gr["formalChargeList"]
            ):
                topo.record.append(
                    "ATOM" if gr["singleLetterCode"] != "?" else "HETATM"
                )
                topo.resname.append(gr["groupName"])
                topo.name.append(name)
                topo.element.append(elem)
                topo.formalcharge.append(fchg)
                topo.beta.append(data.b_factor_list[a_idx])
                topo.occupancy.append(data.occupancy_list[a_idx])
                topo.serial.append(data.atom_id_list[a_idx])
                topo.altloc.append(data.alt_loc_list[a_idx].replace("\x00", ""))
                topo.insertion.append(ins.replace("\x00", ""))
                topo.chain.append(data.chain_name_list[chain_idx])
                topo.segid.append(
                    str(chain_idx)
                )  # Set segid as chain since there is no segid in mmtf
                topo.resid.append(resid)
                a_idx += 1

            for b in range(len(gr["bondOrderList"])):
                topo.bonds.append(
                    [
                        gr["bondAtomList"][b * 2] + group_first_a_idx,
                        gr["bondAtomList"][b * 2 + 1] + group_first_a_idx,
                    ]
                )
                topo.bondtype.append(str(gr["bondOrderList"][b]))

            g_idx += 1

    n_atoms = len(topo.name)
    for b in range(len(data.bond_order_list)):
        bond_idx = [data.bond_atom_list[b * 2], data.bond_atom_list[b * 2 + 1]]
        if np.any(bond_idx[0] >= n_atoms or bond_idx[1] >= n_atoms):
            continue
        topo.bonds.append(bond_idx)
        topo.bondtype.append(str(data.bond_order_list[b]))

    if first_coords_only:
        coords = np.array(
            [
                data.x_coord_list[:n_atoms],
                data.y_coord_list[:n_atoms],
                data.z_coord_list[:n_atoms],
            ]
        )
    else:
        coords = np.array(
            [
                data.x_coord_list.reshape(data.num_models, n_atoms),
                data.y_coord_list.reshape(data.num_models, n_atoms),
                data.z_coord_list.reshape(data.num_models, n_atoms),
            ]
        )
        coords = np.transpose(coords, [2, 0, 1])
    traj = Trajectory(coords=coords)
    return MolFactory.construct(
        topo, traj, filename, frame, validateElements=validateElements
    )


def ALPHAFOLDread(
    filename,
    frame=None,
    topoloc=None,
    validateElements=True,
    uri="https://alphafold.ebi.ac.uk/files/AF-{uniprot}-F1-model_v3.cif",
):
    import urllib.request
    import tempfile

    filename = filename[3:].upper()
    with urllib.request.urlopen(uri.format(uniprot=filename)) as f:
        contents = f.read().decode("utf-8")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfile = os.path.join(tmpdir, f"{filename}.cif")
        with open(tmpfile, "w") as f:
            f.write(contents)
        results = CIFread(tmpfile)
    return results


# Register here all readers with their extensions
_TOPOLOGY_READERS = {
    "prmtop": PRMTOPread,
    "prm": PRMTOPread,
    "psf": PSFread,
    "mae": MAEread,
    "mol2": MOL2read,
    "gjf": GJFread,
    "xyz": XYZread,
    "pdb": PDBread,
    "ent": PDBread,
    "pdbqt": PDBQTread,
    "top": [GROTOPread, PRMTOPread],
    "crd": CRDCARDread,
    "cif": CIFread,
    "rtf": RTFread,
    "prepi": PREPIread,
    "sdf": SDFread,
    "mmtf": MMTFread,
    "mmtf.gz": MMTFread,
    "alphafold": ALPHAFOLDread,
}

_MDTRAJ_TOPOLOGY_EXTS = [
    "pdb",
    "pdb.gz",
    "h5",
    "lh5",
    "prmtop",
    "parm7",
    "psf",
    "mol2",
    "hoomdxml",
    "gro",
    "arc",
    "hdf5",
]
for ext in _MDTRAJ_TOPOLOGY_EXTS:
    if ext not in _TOPOLOGY_READERS:
        _TOPOLOGY_READERS[ext] = MDTRAJTOPOread

_TRAJECTORY_READERS = {"xtc": XTCread, "xsc": XSCread}

_COORDINATE_READERS = {"crd": CRDread, "coor": BINCOORread}

_MDTRAJ_TRAJECTORY_EXTS = ("dcd", "binpos", "trr", "nc", "h5", "lh5", "netcdf")
for ext in _MDTRAJ_TRAJECTORY_EXTS:
    if ext not in _TRAJECTORY_READERS:
        _TRAJECTORY_READERS[ext] = MDTRAJread


_ALL_READERS = {}
for k in _TOPOLOGY_READERS:
    if k not in _ALL_READERS:
        _ALL_READERS[k] = []
    _ALL_READERS[k] += ensurelist(_TOPOLOGY_READERS[k])

for k in _TRAJECTORY_READERS:
    if k not in _ALL_READERS:
        _ALL_READERS[k] = []
    _ALL_READERS[k] += ensurelist(_TRAJECTORY_READERS[k])

for k in _COORDINATE_READERS:
    if k not in _ALL_READERS:
        _ALL_READERS[k] = []
    _ALL_READERS[k] += ensurelist(_COORDINATE_READERS[k])


# fmt: off
class _TestReaders(unittest.TestCase):
    def testfolder(self, subname=None):
        from moleculekit.home import home
        if subname is None:
            return home(dataDir='molecule-readers')
        else:
            return os.path.join(home(dataDir='molecule-readers'), subname)

    def test_pdb(self):
        from moleculekit.home import home
        for f in glob(os.path.join(self.testfolder(), '*.pdb')):
            _ = Molecule(f)
        for f in glob(os.path.join(home(dataDir='pdb/'), '*.pdb')):
            _ = Molecule(f)
        
    def test_pdbqt(self):
        _ = Molecule(os.path.join(self.testfolder(), '3ptb.pdbqt'))

    def test_psf(self):
        _ = Molecule(os.path.join(self.testfolder('4RWS'), 'structure.psf'))

    def test_xtc(self):
        _ = Molecule(os.path.join(self.testfolder('4RWS'), 'traj.xtc'))

    def test_combine_topo_traj(self):
        testfolder = self.testfolder('4RWS')
        mol = Molecule(os.path.join(testfolder, 'structure.psf'))
        mol.read(os.path.join(testfolder, 'traj.xtc'))

    def test_prmtop(self):
        testfolder = self.testfolder('3AM6')
        _ = Molecule(os.path.join(testfolder, 'structure.prmtop'))

    def test_crd(self):
        testfolder = self.testfolder('3AM6')
        _ = Molecule(os.path.join(testfolder, 'structure.crd'))

    def test_mol2(self):
        testfolder = self.testfolder('3L5E')
        _ = Molecule(os.path.join(testfolder, 'protein.mol2'))
        _ = Molecule(os.path.join(testfolder, 'ligand.mol2'))

    def test_sdf(self):
        import pickle
        sdf_file = os.path.join(self.testfolder(), 'benzamidine-3PTB-pH7.sdf')
        ref_file = os.path.join(self.testfolder(), 'benzamidine-3PTB-pH7.pkl')
        with open(ref_file, "rb") as f:
            ref_data = pickle.load(f)

        mol = Molecule(sdf_file)
        assert mol.numAtoms == 18
        assert np.all(mol.name == mol.element)
        assert np.array_equal(mol.element, ref_data["element"])
        assert np.array_equal(mol.coords, ref_data["coords"])
        assert np.array_equal(mol.formalcharge, ref_data["formalcharge"])
        assert np.array_equal(mol.bonds, ref_data["bonds"])
        assert np.array_equal(mol.bondtype, ref_data["bondtype"])

        sdf_file = os.path.join(self.testfolder(), 'S98_ideal.sdf')
        ref_file = os.path.join(self.testfolder(), 'S98_ideal.pkl')
        with open(ref_file, "rb") as f:
            ref_data = pickle.load(f)

        mol = Molecule(sdf_file)
        assert mol.numAtoms == 34
        assert np.all(mol.name == mol.element)
        assert np.array_equal(mol.element, ref_data["element"])
        assert np.array_equal(mol.coords, ref_data["coords"])
        assert np.array_equal(mol.formalcharge, ref_data["formalcharge"])
        assert np.array_equal(mol.bonds, ref_data["bonds"])
        assert np.array_equal(mol.bondtype, ref_data["bondtype"])

    def test_gjf(self):
        testfolder = self.testfolder('3L5E')
        _ = Molecule(os.path.join(testfolder, 'ligand.gjf'))

    def test_xyz(self):
        testfolder = self.testfolder('3L5E')
        _ = Molecule(os.path.join(testfolder, 'ligand.xyz'))

    def test_MDTRAJTOPOread(self):
        testfolder = self.testfolder('3L5E')
        _ = Molecule(os.path.join(testfolder, 'ligand.mol2'))

    def test_mae(self):
        for f in glob(os.path.join(self.testfolder(), '*.mae')):
            _ = Molecule(f)

    def test_append_trajectories(self):
        testfolder = self.testfolder('CMYBKIX')
        mol = Molecule(os.path.join(testfolder, 'filtered.pdb'))
        mol.read(glob(os.path.join(testfolder, '*.xtc')))

    def test_missing_crystal_info(self):
        _ = Molecule(os.path.join(self.testfolder(), 'weird-cryst.pdb'))

    def test_missing_occu_beta(self):
        mol = Molecule(os.path.join(self.testfolder(), 'opm_missing_occu_beta.pdb'))
        assert np.all(mol.occupancy[:2141] != 0)
        assert np.all(mol.occupancy[2141:] == 0)
        assert np.all(mol.beta[:2141] != 0)
        assert np.all(mol.beta[2141:] == 0)

    def test_dcd(self):
        mol = Molecule(os.path.join(self.testfolder('dcd'), '1kdx_0.pdb'))
        mol.read(os.path.join(self.testfolder('dcd'), '1kdx.dcd'))

    def test_dcd_into_prmtop(self):
        mol = Molecule(os.path.join(self.testfolder(), 'dialanine', 'structure.prmtop'))
        mol.read(os.path.join(self.testfolder(), 'dialanine', 'traj.dcd'))
        assert mol.numFrames == 2

    def test_dcd_frames(self):
        mol = Molecule(os.path.join(self.testfolder('dcd'), '1kdx_0.pdb'))
        mol.read(os.path.join(self.testfolder('dcd'), '1kdx.dcd'))
        tmpcoo = mol.coords.copy()
        mol.read([os.path.join(self.testfolder('dcd'), '1kdx.dcd')], frames=[8])
        assert np.array_equal(tmpcoo[:, :, 8], np.squeeze(mol.coords)), 'Specific frame reading not working'

    def test_xtc_frames(self):
        mol = Molecule(os.path.join(self.testfolder('4RWS'), 'structure.pdb'))
        mol.read(os.path.join(self.testfolder('4RWS'), 'traj.xtc'))
        tmpcoo = mol.coords.copy()
        mol.read([os.path.join(self.testfolder('4RWS'), 'traj.xtc')], frames=[1])
        assert np.array_equal(tmpcoo[:, :, 1], np.squeeze(mol.coords)), 'Specific frame reading not working'

    def test_acemd3_xtc_fstep(self):
        from moleculekit.util import tempname

        mol = Molecule(os.path.join(self.testfolder(), 'aladipep_traj_4fs_100ps.xtc'))
        refstep = np.array([ 10,  20,  30,  40,  50,  60,  70,  80,  90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200])
        reftime = np.array([ 40.,  80., 120., 160., 200., 240., 280., 320., 360., 400., 440.,
                            480., 520., 560., 600., 640., 680., 720., 760., 800.], dtype=np.float32)
        assert np.array_equal(mol.step, refstep)
        assert np.allclose(mol.time, reftime)
        assert abs(mol.fstep - 4E-5) < 1E-12

        # Test that XTC writing doesn't mess up the times
        tmpfile = tempname(suffix=".xtc")
        mol.write(tmpfile)

        mol2 = Molecule(tmpfile)
        assert mol.fstep == mol2.fstep
        assert np.array_equal(mol.time, mol2.time)

    def test_gromacs_top(self):
        mol = Molecule(os.path.join(self.testfolder(), 'gromacs.top'))
        assert np.array_equal(mol.name, ['C1', 'O2', 'N3', 'H4', 'H5', 'N6', 'H7', 'H8'])
        assert np.array_equal(mol.atomtype, ['C', 'O', 'NT', 'H', 'H', 'NT', 'H', 'H'])
        assert np.all(mol.resid == 1)
        assert np.all(mol.resname == 'UREA')
        assert np.array_equal(mol.charge, np.array([ 0.683, -0.683, -0.622,  0.346,  0.276, -0.622,  0.346,  0.276], np.float32))
        assert np.array_equal(mol.bonds, [[2, 3],
                                            [2, 4],
                                            [5, 6],
                                            [5, 7],
                                            [0, 1],
                                            [0, 2],
                                            [0, 5]])
        assert np.array_equal(mol.angles, [[0, 2, 3],
                                            [0, 2, 4],
                                            [3, 2, 4],
                                            [0, 5, 6],
                                            [0, 5, 7],
                                            [6, 5, 7],
                                            [1, 0, 2],
                                            [1, 0, 5],
                                            [2, 0, 5]])
        assert np.array_equal(mol.dihedrals, [[1, 0, 2, 3],
                                                [5, 0, 2, 3],
                                                [1, 0, 2, 4],
                                                [5, 0, 2, 4],
                                                [1, 0, 5, 6],
                                                [2, 0, 5, 6],
                                                [1, 0, 5, 7],
                                                [2, 0, 5, 7],
                                                [2, 3, 4, 0],
                                                [5, 6, 7, 0],
                                                [0, 2, 5, 1]])
        assert len(mol.impropers) == 0

    def test_mmcif_single_frame(self):
        mol = Molecule(os.path.join(self.testfolder(), '1ffk.cif'))
        assert mol.numAtoms == 64281
        assert mol.numFrames == 1

    def test_mmcif_multi_frame(self):
        mol = Molecule(os.path.join(self.testfolder(), '1j8k.cif'))
        assert mol.numAtoms == 1402
        assert mol.numFrames == 20

    def test_mmcif_ligand(self):
        mol = Molecule(os.path.join(self.testfolder(), 'URF.cif'))
        assert mol.numAtoms == 12, mol.numAtoms
        assert mol.numFrames == 1

        mol = Molecule(os.path.join(self.testfolder(), 'BEN.cif'))
        assert mol.numAtoms == 17, mol.numAtoms
        assert mol.numFrames == 1

        mol = Molecule(os.path.join(self.testfolder(), '33X.cif'))
        assert mol.numAtoms == 16, mol.numAtoms
        assert mol.numFrames == 1

    def test_multiple_file_fileloc(self):
        mol = Molecule(os.path.join(self.testfolder('multi-traj'), 'structure.pdb'))
        mol.read(glob(os.path.join(self.testfolder('multi-traj'), 'data', '*', '*.xtc')))
        # Try to vstack fileloc. This will fail with wrong fileloc shape
        fileloc = np.vstack(mol.fileloc)
        assert fileloc.shape == (12, 2)
        print('Correct fileloc shape with multiple file reading.')

    def test_topo_overwriting(self):
        # Testing overwriting of topology fields
        mol = Molecule(os.path.join(self.testfolder('multi-topo'), 'mol.psf'))
        atomtypes = mol.atomtype.copy()
        charges = mol.charge.copy()
        coords = np.array([[[0.],
                            [0.],
                            [-0.17]],

                           [[0.007],
                            [1.21],
                            [0.523]],

                           [[0.],
                            [0.],
                            [-1.643]],

                           [[-0.741],
                            [-0.864],
                            [-2.296]]], dtype=np.float32)

        mol.read(os.path.join(self.testfolder('multi-topo'), 'mol.pdb'))
        assert np.array_equal(mol.atomtype, atomtypes)
        assert np.array_equal(mol.charge, charges)
        assert np.array_equal(mol.coords, coords)
        print('Merging of topology fields works')

    def test_integer_resnames(self):
        mol = Molecule(os.path.join(self.testfolder(), 'errors.pdb'))
        assert np.unique(mol.resname) == '007'

    def test_star_indexes(self):
        mol = Molecule(os.path.join(self.testfolder(), 'errors.pdb'))
        assert np.all(mol.serial == np.arange(1, mol.numAtoms+1))

    def test_xsc(self):
        mol1 = Molecule(os.path.join(self.testfolder(), 'test1.xsc'))
        ref1box = np.array([[81.67313],[75.81903],[75.02757]], dtype=np.float32)
        ref1step = np.array([2])
        assert np.array_equal(mol1.box, ref1box)
        assert np.array_equal(mol1.step, ref1step)

        mol2 = Molecule(os.path.join(self.testfolder(), 'test2.xsc'))
        ref2box = np.array([[41.0926],[35.99448],[63.51968]], dtype=np.float32)
        ref2step = np.array([18])
        assert np.array_equal(mol2.box, ref2box)
        assert np.array_equal(mol2.step, ref2step)

        # Test reading xsc into existing molecule
        mol1 = Molecule('3ptb')
        mol1.read(os.path.join(self.testfolder(), 'test1.xsc'))
        assert np.array_equal(mol1.box, ref1box)
        assert np.array_equal(mol1.step, ref1step)

    def test_pdb_element_guessing(self):
        mol = Molecule(os.path.join(self.testfolder(), 'errors.pdb'))
        refelem = np.array(['C', 'C', 'C', 'C', 'C', 'C', 'C', 'N', 'Na', 'N', 'H', 'H', 'C', 'Cl', 'Ca'], dtype=object)
        assert np.array_equal(mol.element, refelem)

        mol = Molecule(os.path.join(self.testfolder(), 'dialanine_solute.pdb'))
        refelem = np.array(['H', 'C', 'H', 'H', 'C', 'O', 'N', 'H', 'C', 'H', 'C', 'H', 'H', 'H', 'C', 'O', 'N', 'H',
                            'C', 'H', 'H', 'H'], dtype=object)
        assert np.array_equal(mol.element, refelem)

        mol = Molecule(os.path.join(self.testfolder(), 'cl_na_element.pdb'))
        refelem = np.array(['Cl', 'Na'], dtype=object)
        assert np.array_equal(mol.element, refelem)

        mol = Molecule(os.path.join(self.testfolder(), 'dummy_atoms.mol2'))
        refelem = np.array(['Cd', 'Co', 'Ca', 'Xe', 'Rb', 'Lu', 'Ga', 'Ba', 'Cs', 'Ho', 'Pb', 'Sr', 'Yb', 'Y'], dtype=object)
        assert np.array_equal(mol.element, refelem), f"Failed guessing {refelem}, got {mol.element}"

    def test_pdb_charges(self):
        mol = Molecule(os.path.join(self.testfolder(), 'errors.pdb'))
        refcharge = np.array([-1,  1, -1,  0,  0,  0,  0,  0,  0,  0,  0, 0,  0,  0,  0], dtype=np.int32)
        assert np.array_equal(mol.formalcharge, refcharge)

    def test_prepi(self):
        mol = Molecule(os.path.join(self.testfolder(), 'benzamidine.prepi'))
        assert np.array_equal(mol.name, ['N', 'H18', 'H17', 'C7', 'N14', 'H15', 'H16', 'C', 'C6', 'H12', 'C5', 'H11', 'C4', 'H10', 'C3', 'H9', 'C2', 'H'])
        assert np.array_equal(mol.atomtype, ['NT', 'H', 'H', 'CM', 'NT', 'H', 'H', 'CA', 'CA', 'HA', 'CA', 'HA', 'CA', 'HA', 'CA', 'HA', 'CA', 'HA'])
        assert np.array_equal(mol.charge, np.array([-0.4691,  0.3267,  0.3267,  0.5224, -0.4532,  0.3267,  0.3267,
                                                    -0.2619, -0.082 ,  0.149 , -0.116 ,  0.17  , -0.058 ,  0.17  ,
                                                    -0.116 ,  0.17  , -0.082 ,  0.149 ], dtype=np.float32))
        assert np.array_equal(mol.impropers, np.array([[ 7,  0,  3,  4],
                                                        [ 8, 16,  7,  3],
                                                        [ 7, 10,  8,  9],
                                                        [ 8, 12, 10, 11],
                                                        [10, 14, 12, 13],
                                                        [12, 16, 14, 15],
                                                        [ 7, 14, 16, 17]]))

    def test_rtf(self):
        mol = Molecule(os.path.join(self.testfolder(), 'mol.rtf'))
        assert np.array_equal(mol.element, ['C', 'C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'N', 'N', 'H', 'H', 'H', 'H'])
        assert np.array_equal(mol.name, ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'H8', 'H9', 'H10', 'H11', 'H12', 'N13', 'N14', 'H15', 'H16', 'H17', 'H18'])
        assert np.array_equal(mol.atomtype, ['C261', 'C261', 'C261', 'C261', 'C261', 'C261', 'C2N2', 'HG61', 'HG61', 'HG61', 'HG61', 'HG61', 'N2P1', 'N2P1', 'HGP2', 'HGP2', 'HGP2', 'HGP2'])
        assert np.array_equal(mol.charge, np.array([ 0.062728, -0.046778, -0.061366, -0.062216, -0.061366, -0.046778,
                                                    0.270171,  0.063213,  0.062295,  0.062269,  0.062295,  0.063213,
                                                    -0.28704 , -0.28704 ,  0.3016  ,  0.3016  ,  0.3016  ,  0.3016  ], dtype=np.float32))
        assert np.array_equal(mol.masses, np.array([12.011, 12.011, 12.011, 12.011, 12.011, 12.011, 12.011,  1.008,
                                                    1.008,  1.008,  1.008,  1.008, 14.007, 14.007,  1.008,  1.008,
                                                    1.008,  1.008], dtype=np.float32))
        assert np.array_equal(mol.bonds, np.array([[ 0,  1],
                                                    [ 0,  5],
                                                    [ 0,  6],
                                                    [ 1,  2],
                                                    [ 1,  7],
                                                    [ 2,  3],
                                                    [ 2,  8],
                                                    [ 3,  4],
                                                    [ 3,  9],
                                                    [ 4,  5],
                                                    [ 4, 10],
                                                    [ 5, 11],
                                                    [ 6, 12],
                                                    [ 6, 13],
                                                    [16, 12],
                                                    [17, 12],
                                                    [14, 13],
                                                    [15, 13]]))
        assert np.array_equal(mol.impropers, np.array([[ 6,  0, 12, 13]]))

    def test_mmtf(self):
        from tqdm import tqdm
        mol = Molecule("3hyd")
        assert mol.numAtoms == 125
        assert mol.numBonds == 120

        mol = Molecule("3ptb")
        assert mol.numAtoms == 1701
        assert mol.numBonds == 1675

        mol = Molecule("6a5j")
        assert mol.numAtoms == 260
        assert mol.numBonds == 257

        # with open(os.path.join(self.testfolder(), 'pdbids.txt'), "r") as f:
        #     pdbids = [ll.strip() for ll in f] 
        # # Skip 1USE 2IWN 3ZHI which are wrong in pdb and correct in mmtf

        pdbids = ["3ptb", "3hyd", "6a5j", "5vbl", "7q5b"]
        # Compare with PDB reader
        # nc-aa residues are not marked as hetatm in mmtf
        pdb_exceptions = {"5vbl": ["record"], "7q5b": ["record"], "5v0o": ["record"]}
        # segids don't exist in mmtf. formalcharge doesn't exist (correctly) in pdb
        # serials are not consistent but are irrelevant anyway
        exceptions = ["serial", "segid", "formalcharge"]
        for pdbid in tqdm(pdbids, desc="PDB/MMTF comparison"):
            with self.subTest(pdbid=pdbid):
                mol1 = Molecule(pdbid, type="pdb")
                mol2 = Molecule(pdbid, type="mmtf")
                mol2.dropFrames(keep=0)
                exc = exceptions
                if pdbid in pdb_exceptions:
                    exc += pdb_exceptions[pdbid]
                assert mol_equal(mol1, mol2, exceptFields=exc)
# fmt: on

if __name__ == "__main__":
    import unittest

    unittest.main(verbosity=2)
