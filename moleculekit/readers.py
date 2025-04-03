# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
import numpy as np
from moleculekit.util import sequenceID
from moleculekit.periodictable import elements_from_masses
from moleculekit.util import ensurelist
from moleculekit.molecule import Molecule
from contextlib import contextmanager
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
            if strData.endswith(".gz"):
                import gzip

                fi = gzip.open(strData, mode + "t")
            else:
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
        self.virtualsite = []
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
            "virtualsite",
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
                self.box = np.zeros((3, nframes), Molecule._dtypes["box"])
            if boxangles is None:
                self.boxangles = np.zeros((3, nframes), Molecule._dtypes["boxangles"])
            if step is None:
                self.step = np.arange(nframes, dtype=Molecule._dtypes["step"])
            if time is None:
                self.time = np.zeros(nframes, dtype=Molecule._dtypes["time"])
        if box is not None:
            self.box = np.array(box, dtype=Molecule._dtypes["box"])
            if boxangles is None:
                raise ValueError("boxangles must be provided if box is provided")
        if boxangles is not None:
            self.boxangles = np.array(boxangles, dtype=Molecule._dtypes["boxangles"])
            if box is None:
                raise ValueError("box must be provided if boxangles is provided")
        if fileloc is not None:
            self.fileloc = fileloc
        if step is not None:
            self.step = np.array(step, dtype=Molecule._dtypes["step"])
        if time is not None:
            self.time = np.array(time, dtype=Molecule._dtypes["time"])

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
                f"Different number of topologies ({len(topos)}) and trajectories ({len(trajs)}) were read from {filename}"
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
                    f"Different number of atoms read from file {filename} for different fields: {natoms}."
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
                f"Different number of atoms in topology ({toponatoms}) and trajectory ({trajnatoms}) for "
                f"file {filename}"
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
            if os.path.isfile(filename):
                topoloc = os.path.abspath(filename)
                viewname = os.path.basename(filename)
            else:
                topoloc = filename
                viewname = filename
            fileloc = [[filename, 0]]

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
            mol.coords = np.array(traj.coords, dtype=Molecule._dtypes["coords"])

        if traj.box is None:
            mol.box = np.zeros((3, 1), dtype=Molecule._dtypes["box"])
        else:
            mol.box = np.array(traj.box, dtype=Molecule._dtypes["box"])
            if mol.box.ndim == 1:
                mol.box = mol.box[:, np.newaxis]

        if traj.boxangles is None:
            mol.boxangles = np.zeros((3, 1), dtype=Molecule._dtypes["boxangles"])
        else:
            mol.boxangles = np.array(
                traj.boxangles, dtype=Molecule._dtypes["boxangles"]
            )
            if mol.boxangles.ndim == 1:
                mol.boxangles = mol.boxangles[:, np.newaxis]

        # mol.fileloc = traj.fileloc
        mol.step = np.hstack(traj.step).astype(Molecule._dtypes["step"])
        mol.time = np.hstack(traj.time).astype(Molecule._dtypes["time"])

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
    import gzip
    import re

    natom_line = re.compile(r"^([0-9]+)\s*$")
    coord_line = re.compile(
        r"^\w+\s+([-+]?[0-9]*\.?[0-9]+)\s+([-+]?[0-9]*\.?[0-9]+)\s+([-+]?[0-9]*\.?[0-9]+)\s*$"
    )
    topo = Topology()

    frames = []
    firstconf = True
    coords = None
    i = 0
    gz = filename.endswith(".gz")
    with gzip.open(filename, "rt") if gz else open(filename, "r") as f:
        for line in f:
            if natom_line.match(line):
                if coords is not None:
                    frames.append(coords)
                    firstconf = False
                coords = []
                natoms = int(line.split()[0])
            elif coord_line.match(line):
                pieces = line.split()
                if firstconf:
                    topo.record.append("HETATM")
                    topo.serial.append(i + 1)
                    topo.element.append(pieces[0])
                    topo.name.append(pieces[0])
                    topo.resname.append("MOL")
                    i += 1
                coords.append([float(x) for x in pieces[1:4]])

    frames.append(coords)
    assert len(topo.record) == natoms
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


def _getLocalPDB(fname):
    if "LOCAL_PDB_REPO" in os.environ and os.path.isfile(
        os.path.join(os.environ["LOCAL_PDB_REPO"], fname)
    ):
        filepath = os.path.join(os.environ["LOCAL_PDB_REPO"], fname)
        logger.info(f"Using local copy for {fname}: {filepath}")
        return filepath
    return None


def _getPDB(pdbid):
    from moleculekit.util import string_to_tempfile

    # Try loading it from the pdb data directory
    tempfile = False

    filepath = _getLocalPDB(pdbid.lower() + ".pdb")
    if filepath is None:
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

    edge_cases = {"SOD": "Na"}

    allelements = [str(el).upper() for el in periodictable]
    newelements = np.array(["" for _ in range(len(elements))], dtype=object)

    alternatives = defaultdict(list)

    bad_elements = []
    for el in elements:
        if el.strip() == "" or el.strip().capitalize() not in periodictable:
            bad_elements.append(True)
        else:
            bad_elements.append(False)

    if not onlymissing or all(bad_elements):
        noelem = np.arange(len(elements))
    else:
        noelem = np.where(bad_elements)[0]

    names = np.array(names, dtype=object)
    uqnames = np.unique(names[noelem])
    for name in uqnames:
        name = (
            re.sub(r"\d", " ", name[0]) + name[1:]
        )  # Remove numbers from first column
        elem = None
        if name.strip() in edge_cases:
            elem = edge_cases[name.strip()]
        elif (
            name[0] == " "
        ):  # If there is no letter in col 13 then col 14 is the element
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
                f'Element guessing failed for atom with name "{name}" as the guessed element "{elem}" was not found in '
                "the periodic table. Check for incorrect column alignment in the PDB file or report "
                "to moleculekit issue tracker."
            )
            elem = None
        newelements[names == name] = elem if elem is not None else ""

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
        "occupancy": float,
        "beta": float,
        "segid": str,
        "element": str,
        "atomtype": str,
        "charge": float,
        "formalcharge": int,
    }
    coordcolspecs = [(30, 38), (38, 46), (46, 54)]

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
    # bondnames = ("serial1", "serial2", "serial3", "serial4", "serial5")

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

    topo = Topology()
    crystalinfo = {}
    parsedsymmetry = {prop: [] for prop in symmetrynames}
    parsedbonds = []

    coords = []
    teridx = []
    currter = 0
    topoend = False
    modelcoords = []
    failed_type_conversion = set()
    failed_parsing = set()

    def _fix_formal_charge(val):
        if prop == "formalcharge":
            # Move the minus to the start. Sometimes it's at the end in PDB files
            if "-" in val:
                val = "-" + val.replace("-", "")
            if "+" in val:
                val = val.replace("+", "")
        return val

    with openFileOrStringIO(filename, "r") as f:
        for line in f:
            if line.startswith("CRYST1"):
                for (s, e), prop in zip(cryst1colspecs, cryst1names):
                    try:
                        crystalinfo[prop] = line[s:e].strip()
                        if len(crystalinfo[prop]):
                            if prop == "sGroup":
                                crystalinfo[prop] = crystalinfo[prop].split()
                            elif prop == "z":
                                crystalinfo[prop] = int(crystalinfo[prop])
                            else:
                                crystalinfo[prop] = float(crystalinfo[prop])
                    except Exception as err:
                        logger.warning(
                            f"Failed to parse crystal info with error: {err}"
                        )
                        del crystalinfo[prop]
            if line.startswith("ATOM") or line.startswith("HETATM"):
                modelcoords.append([float(line[s:e]) for s, e in coordcolspecs])
            if (line.startswith("ATOM") or line.startswith("HETATM")) and not topoend:
                teridx.append(str(currter))
                for (s, e), prop in zip(topocolspecs, toponames):
                    dt = topodtypes[prop]
                    val = line[s:e]
                    if dt != str:
                        val = _fix_formal_charge(val.strip())
                        if len(val):
                            try:
                                val = dt(val)
                            except Exception:
                                failed_type_conversion.add(prop)
                        else:
                            val = 0
                    getattr(topo, prop).append(val)
            if line.startswith("TER"):
                currter += 1
            if (mode == "pdb" and line.startswith("END")) or (
                mode == "pdbqt" and line.startswith("ENDMDL")
            ):  # pdbqt should not stop reading at ENDROOT or ENDBRANCH
                topoend = True
                if len(modelcoords):
                    coords.append(modelcoords)
                modelcoords = []
            if line.startswith("CONECT"):
                parsedbonds.append([line[s:e].strip() for s, e in bondcolspecs])
            if line.startswith("MODEL"):
                if len(modelcoords):
                    coords.append(modelcoords)
                modelcoords = []
            if line.startswith(
                "REMARK 290   SMTRY"
            ):  # TODO: Support BIOMT fields. It's a bit more complicated.
                for (s, e), prop in zip(symmetrycolspecs, symmetrynames):
                    try:
                        parsedsymmetry[prop].append(float(line[s:e]))
                    except Exception as err:
                        logger.warning(
                            f"Failed to parse symmetry info with error: {err}"
                        )
                        failed_parsing.add("symmetry")

    if len(modelcoords):
        coords.append(modelcoords)

    natoms = None
    for i, cc in enumerate(coords):
        if natoms is None:
            natoms = len(cc)
        elif len(cc) != natoms:
            logger.warning(
                "Different number of atoms read in different MODELs in the PDB file. "
                f"Keeping only the first {i} model(s)"
            )
            coords = coords[:i]
            break
    coords = np.stack(coords, axis=2)

    if len(topo.name) == 0:
        raise RuntimeError(f"No atoms could be read from PDB file {filename}")

    # Before stripping guess elements from atomname as the spaces are significant
    if mode == "pdb" and validateElements:
        idx, newelem = pdbGuessElementByName(topo.element, topo.name)
        for i, ix in enumerate(idx):
            topo.element[ix] = newelem[i]

    # Now we can safely strip all string fields
    for prop in topodtypes:
        if hasattr(topo, prop) and topodtypes[prop] == str and len(getattr(topo, prop)):
            for i in range(len(topo.name)):
                topo.__dict__[prop][i] = topo.__dict__[prop][i].strip()

    # Fixing hexadecimal index and resids
    # Support for reading hexadecimal
    if "serial" in failed_type_conversion:
        logger.warning(
            'Non-integer values were read from the PDB "serial" field. Dropping PDB values and assigning new ones.'
        )
        if len(np.unique(topo.serial)) == len(topo.serial):
            # Indexes should start from 1 in PDB
            topo.serial = sequenceID(topo.serial) + 1
        else:
            topo.serial = np.arange(1, len(topo.serial) + 1)
    if "resid" in failed_type_conversion:
        logger.warning(
            'Non-integer values were read from the PDB "resid" field. Dropping PDB values and assigning new ones.'
        )
        topo.resid = sequenceID(topo.resid)

    if len(topo.serial) > 99999:
        logger.warning(
            "Reading PDB file with more than 99999 atoms. Bond information can be wrong."
        )

    if len(parsedsymmetry["rot1"]) and "symmetry" not in failed_parsing:
        numcopies = int(len(parsedsymmetry["trans"]) / 3)
        rots = np.vstack(
            (parsedsymmetry["rot1"], parsedsymmetry["rot2"], parsedsymmetry["rot3"])
        ).T
        crystalinfo["numcopies"] = numcopies
        crystalinfo["rotations"] = rots.reshape((numcopies, 3, 3))
        crystalinfo["translations"] = np.array(parsedsymmetry["trans"]).reshape(
            (numcopies, 3)
        )

    # Bond formatting part
    serials = topo.serial
    mapserials = np.full(np.max(serials) + 1, -1, dtype=np.int64)
    mapserials[serials] = list(range(len(serials)))
    for i in range(len(parsedbonds)):
        row = parsedbonds[i]
        topo.bonds += [
            [int(row[0]), int(row[b])] for b in range(1, 5) if row[b].strip() != ""
        ]

    if len(topo.bonds):
        topo.bonds = np.array(topo.bonds, dtype=Molecule._dtypes["bonds"])
        badidx = ~np.all(np.isin(topo.bonds, topo.serial), axis=1)
        if np.any(badidx):
            # Some PDBs have bonds to non-existing serials... go figure
            topo.bonds = topo.bonds[~badidx]
            logger.info(
                f"Discarded {np.sum(badidx)} bonds to non-existing indexes in the PDB file."
            )
        topo.bonds = np.array(
            mapserials[topo.bonds[:]], dtype=Molecule._dtypes["bonds"]
        )

    # If no segid was read, use the TER rows to define segments
    if len(topo.segid) and np.all(np.array(topo.segid) == "") and currter != 0:
        topo.segid = teridx

    if tempfile:
        os.unlink(filename)

    box = None
    boxangles = None
    if "a" in crystalinfo and "b" in crystalinfo and "c" in crystalinfo:
        box = np.array([crystalinfo["a"], crystalinfo["b"], crystalinfo["c"]])
    if "alpha" in crystalinfo and "beta" in crystalinfo and "gamma" in crystalinfo:
        boxangles = np.array(
            [crystalinfo["alpha"], crystalinfo["beta"], crystalinfo["gamma"]]
        )

    topo.crystalinfo = crystalinfo
    traj = Trajectory(coords=coords, box=box, boxangles=boxangles)
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


def PRMTOPread(filename, frame=None, topoloc=None, validateElements=False):
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
    topo.element, topo.virtualsite = elements_from_masses(topo.masses)
    return MolFactory.construct(
        topo, None, filename, frame, validateElements=validateElements
    )


def PSFread(filename, frame=None, topoloc=None, validateElements=False):
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
    topo.element, topo.virtualsite = elements_from_masses(topo.masses)
    return MolFactory.construct(
        topo, None, filename, frame, validateElements=validateElements
    )


def XTCread(filename, frame=None, topoloc=None):
    from moleculekit.xtc import read_xtc, read_xtc_frames
    from moleculekit.unitcell import box_vectors_to_lengths_and_angles

    if frame is None:
        coords, boxvectors, time, step = read_xtc(filename.encode("UTF-8"))
    else:
        coords, boxvectors, time, step = read_xtc_frames(
            filename.encode("UTF-8"), np.array(ensurelist(frame), dtype=np.int32)
        )

    if np.size(coords, 2) == 0:
        raise RuntimeError(f"Malformed XTC file. No frames read from: {filename}")
    if np.size(coords, 0) == 0:
        raise RuntimeError(f"Malformed XTC file. No atoms read from: {filename}")

    time = time.astype(Molecule._dtypes["time"])
    coords *= 10.0  # Convert from nm to Angstrom
    boxvectors *= 10.0  # Convert from nm to Angstrom
    time *= 1e3  # Convert from ps to fs. This seems to be ACEMD3 specific. GROMACS writes other units in time
    nframes = coords.shape[2]
    if len(step) != nframes or np.sum(step) == 0:
        step = np.arange(nframes, dtype=Molecule._dtypes["step"])
    if len(time) != nframes or np.sum(time) == 0:
        time = np.zeros(nframes, dtype=Molecule._dtypes["time"])

    bx, by, bz, alpha, beta, gamma = box_vectors_to_lengths_and_angles(
        boxvectors[0].T, boxvectors[1].T, boxvectors[2].T
    )
    box = np.stack([bx, by, bz], axis=0)
    boxangles = np.stack([alpha, beta, gamma], axis=0)
    return MolFactory.construct(
        None,
        Trajectory(coords=coords, box=box, boxangles=boxangles, step=step, time=time),
        filename,
        frame,
    )


def XSCread(filename, frame=None, topoloc=None):
    from moleculekit.unitcell import box_vectors_to_lengths_and_angles

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

            step = pieces[0]
            bx, by, bz, alpha, beta, gamma = box_vectors_to_lengths_and_angles(
                pieces[1:4], pieces[4:7], pieces[7:10]
            )
            box = np.array([bx, by, bz])
            boxangles = np.array([alpha, beta, gamma])

    return MolFactory.construct(
        None,
        Trajectory(
            box=box,
            boxangles=boxangles,
            step=[step],
            time=[0],
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
        coords = np.array(coords, dtype=Molecule._dtypes["coords"]).reshape(
            (natoms, 3, 1)
        )
    return MolFactory.construct(None, Trajectory(coords=coords), filename, frame)


def MDTRAJread(filename, frame=None, topoloc=None, validateElements=True):
    try:
        import mdtraj as md
    except ImportError:
        raise ImportError(
            f"To support extension {os.path.splitext(filename)[1]} please install the `mdtraj` package"
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
            f"To support extension {os.path.splitext(filename)[1]} please install the `mdtraj` package"
        )

    mdstruct = md.load(filename)
    topology = mdstruct.topology
    table, bonds = topology.to_dataframe()

    topo = Topology()
    for k in table.keys():
        topo.__dict__[translate[k]] = table[k].tolist()

    coords = np.array(
        mdstruct.xyz.swapaxes(0, 1).swapaxes(1, 2) * 10,
        dtype=Molecule._dtypes["coords"],
    )
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


# def CIFread_new(filename, frame=None, topoloc=None, zerowarning=True, data=None):
#     from mmcif.core.mmciflib import ParseCifSimple
#     from mmcif.io.BinaryCifReader import BinaryCifReader
#     from mmcif.core.mmciflib import CifFile

#     # Taken from http://mmcif.wwpdb.org/docs/pdb_to_pdbx_correspondences.html#ATOMP
#     atom_site_mapping = {
#         "group_PDB": ("record", str),
#         "id": ("serial", int),
#         "label_alt_id": ("altloc", str),
#         "auth_atom_id": ("name", str),
#         "auth_comp_id": ("resname", str),
#         "auth_asym_id": ("chain", str),
#         "auth_seq_id": ("resid", int),
#         "pdbx_PDB_ins_code": ("insertion", str),
#         "label_entity_id": ("segid", str),
#         "type_symbol": ("element", str),
#         "occupancy": ("occupancy", float),
#         "B_iso_or_equiv": ("beta", float),
#         "pdbx_formal_charge": ("formalcharge", int),
#     }
#     alternatives = {
#         "auth_atom_id": "label_atom_id",
#         "auth_comp_id": "label_comp_id",
#         "auth_asym_id": "label_asym_id",
#         "auth_seq_id": "label_seq_id",
#     }
#     chem_comp_mapping = {
#         "comp_id": ("resname", str),
#         "atom_id": ("name", str),
#         "alt_atom_id": ("atomtype", str),
#         "type_symbol": ("element", str),
#         "charge": ("formalcharge", int),
#         "partial_charge": ("charge", float),
#     }
#     cryst1_mapping = {
#         "length_a": ("a", float),
#         "length_b": ("b", float),
#         "length_c": ("c", float),
#         "angle_alpha": ("alpha", float),
#         "angle_beta": ("beta", float),
#         "angle_gamma": ("gamma", float),
#         "space_group_name_H-M": ("sGroup", str),
#         "Z_PDB": ("z", int),
#     }
#     bondtype_mapping = {
#         "SING": "1",
#         "DOUB": "2",
#         "TRIP": "3",
#         "QUAD": "4",
#         "AROM": "ar",
#     }

#     topo = Topology()

#     if filename.endswith(".bcif.gz"):
#         reader = BinaryCifReader(storeStringsAsBytes=False)
#         filename = reader.deserialize(filename)
#     reader = ParseCifSimple(filename, False, 0, 255, "?", "/tmp/test.log")
#     blockNameList = []
#     blockNameList = reader.GetBlockNames(blockNameList)
#     if len(blockNameList) > 1:
#         logger.warning(
#             "Multiple Data objects in mmCIF. Please report this issue to the moleculekit issue tracker"
#         )

#     block = reader.GetBlock(blockNameList[0])
#     tableNameList = []
#     tableNameList = block.GetTableNames(tableNameList)

#     for tableName in tableNameList:
#         table = block.GetTable(tableName)
#         columnNameList = table.GetColumnNames()
#         numRows = table.GetNumRows()
#         print(f"Table {tableName} colunms {columnNameList}")

#         rowList = []
#         for iRow in range(0, numRows):
#             row = table.GetRow(iRow)
#             rowList.append(row)
#         print(f"table {tableName} row length {len(rowList)}")

#     def fixDefault(val, dtype):
#         if val in ("?", "."):
#             if dtype == float or dtype == int:
#                 val = 0
#             if dtype == str:
#                 val = ""
#         return dtype(val)

#     # Parsing CRYST1 data
#     if "cell" in tableNameList:
#         cryst = block.GetTable("cell")
#         columns = cryst.GetColumnNames()
#         if cryst is not None and cryst.GetNumRows() == 1:
#             row = cryst.getRow(0)
#             crystalinfo = {}
#             for source_field, target in cryst1_mapping.items():
#                 if source_field not in columns:
#                     continue
#                 target_field, dtype = target
#                 val = dtype(
#                     fixDefault(row[cryst.getAttributeIndex(source_field)], dtype)
#                 )
#                 crystalinfo[target_field] = val

#             if "sGroup" in crystalinfo and (
#                 isinstance(crystalinfo["sGroup"], str)
#                 or not np.isnan(crystalinfo["sGroup"])
#             ):
#                 crystalinfo["sGroup"] = crystalinfo["sGroup"].split()
#             topo.crystalinfo = crystalinfo


def CIFread(
    filename, frame=None, topoloc=None, zerowarning=True, data=None, covalentonly=True
):
    from moleculekit.pdbx.reader.PdbxReader import PdbxReader
    from collections import defaultdict

    if data is not None:
        myDataList = data
    else:
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
        if val in ("?", "."):
            if dtype == float or dtype == int:
                val = 0
            if dtype == str:
                val = ""
        return dtype(val)

    # Parsing CRYST1 data
    cryst = dataObj.getObj("cell")
    crystalinfo = {}
    if cryst is not None and cryst.getRowCount() == 1:
        row = cryst.getRow(0)
        attrs = cryst.getAttributeList()

        for source_field, target in cryst1_mapping.items():
            if source_field not in attrs:
                continue
            target_field, dtype = target
            val = dtype(fixDefault(row[cryst.getAttributeIndex(source_field)], dtype))
            crystalinfo[target_field] = val

        # the sGroup info can exist in the symmetry object as well
        cryst2 = dataObj.getObj("symmetry")
        if cryst2 is not None:
            row = cryst2.getRow(0)
            if "space_group_name_H-M" in cryst2.getAttributeList():
                crystalinfo["sGroup"] = row[
                    cryst2.getAttributeIndex("space_group_name_H-M")
                ]

        if "sGroup" in crystalinfo and (
            isinstance(crystalinfo["sGroup"], str)
            or not np.isnan(crystalinfo["sGroup"])
        ):
            crystalinfo["sGroup"] = crystalinfo["sGroup"].split()

        topo.crystalinfo = crystalinfo

    # Parsing intra-residue bond data
    chem_comp_bond = {}
    if "chem_comp_bond" in dataObj.getObjNameList():
        bond_site = dataObj.getObj("chem_comp_bond")
        for i in range(bond_site.getRowCount()):
            row = bond_site.getRow(i)
            resname = row[bond_site.getAttributeIndex("comp_id")]
            name1 = row[bond_site.getAttributeIndex("atom_id_1")]
            name2 = row[bond_site.getAttributeIndex("atom_id_2")]
            bondtype = row[bond_site.getAttributeIndex("value_order")]
            if resname not in chem_comp_bond:
                chem_comp_bond[resname] = {}
            if name1 not in chem_comp_bond[resname]:
                chem_comp_bond[resname][name1] = []
            if name2 not in chem_comp_bond[resname]:
                chem_comp_bond[resname][name2] = []
            chem_comp_bond[resname][name1].append((name2, bondtype))
            chem_comp_bond[resname][name2].append((name1, bondtype))

    # Parsing inter-residue bond data
    struct_conn = defaultdict(list)
    all_struct_conn_atoms = []
    if "struct_conn" in dataObj.getObjNameList():
        struct_conn_site = dataObj.getObj("struct_conn")
        for i in range(struct_conn_site.getRowCount()):
            row = struct_conn_site.getRow(i)
            conn_type_id = row[struct_conn_site.getAttributeIndex("conn_type_id")]
            if covalentonly and conn_type_id not in (
                "covale",
                "covale_base",
                "covale_phosphate",
                "covale_sugar",
                "modres",
            ):
                # Skip non-covalent bonds like disulfide and metal coordination
                # For types check here: https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v40.dic/Items/_struct_conn_type.id.html
                continue
            chain1 = fixDefault(
                row[struct_conn_site.getAttributeIndex("ptnr1_auth_asym_id")], str
            )
            chain2 = fixDefault(
                row[struct_conn_site.getAttributeIndex("ptnr2_auth_asym_id")], str
            )
            resid1 = int(row[struct_conn_site.getAttributeIndex("ptnr1_auth_seq_id")])
            resid2 = int(row[struct_conn_site.getAttributeIndex("ptnr2_auth_seq_id")])
            name1 = row[struct_conn_site.getAttributeIndex("ptnr1_label_atom_id")]
            name2 = row[struct_conn_site.getAttributeIndex("ptnr2_label_atom_id")]
            insertion1 = fixDefault(
                row[struct_conn_site.getAttributeIndex("pdbx_ptnr1_PDB_ins_code")], str
            )
            insertion2 = fixDefault(
                row[struct_conn_site.getAttributeIndex("pdbx_ptnr2_PDB_ins_code")], str
            )
            bondtype = row[struct_conn_site.getAttributeIndex("pdbx_value_order")]
            if bondtype == "?":
                bondtype = "un"
            else:
                bondtype = bondtype_mapping.get(bondtype.upper(), bondtype.upper())
            id1 = (resid1, insertion1, chain1, name1)
            id2 = (resid2, insertion2, chain2, name2)
            struct_conn[id1].append((id2, bondtype))
            all_struct_conn_atoms += [id1, id2]
    all_struct_conn_atoms = set(all_struct_conn_atoms)

    # Parsing ATOM and HETATM data
    if "atom_site" in dataObj.getObjNameList():
        macromolecule = True
    elif "chem_comp_atom" in dataObj.getObjNameList():
        macromolecule = False
    else:
        raise RuntimeError("No atom site data found in mmCIF file.")

    allcoords = []
    ideal_allcoords = []
    coords = []
    ideal_coords = []
    currmodel = -1
    firstmodel = None
    if macromolecule:
        atom_site = dataObj.getObj("atom_site")
        mapping = atom_site_mapping
    else:
        atom_site = dataObj.getObj("chem_comp_atom")
        mapping = chem_comp_mapping

    attrs = atom_site.getAttributeList()
    curr_residue_atoms = {}  # Track index of atoms in the current residue for bond calc
    bond_atoms_idx = {}  # Track index of inter-residue bond atoms
    prev_residue = None  # Keep track when residue changes to reset curr_residue_atoms
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

            # Define residue and atom unique IDs for bond mapping
            if len(topo.resid):
                uqid = (
                    topo.resid[-1],
                    topo.insertion[-1],
                    topo.chain[-1],
                    topo.segid[-1],
                )
                uqatomid = (
                    topo.resid[-1],
                    topo.insertion[-1],
                    topo.chain[-1],
                    topo.name[-1],
                )
            else:  # Small molecules only have a single residue. No need for ID
                uqid = "smallmol"
                uqatomid = None

            # When residue unique ID changes reset the residue atom dictionary
            if prev_residue is None or prev_residue != uqid:
                prev_residue = uqid
                curr_residue_atoms = {}

            curr_residue_atoms[topo.name[-1]] = len(topo.name) - 1
            # If the atom exists in the intra-residue bond dict check the partners
            if topo.name[-1] in chem_comp_bond.get(topo.resname[-1], {}):
                for bond in chem_comp_bond[topo.resname[-1]][topo.name[-1]]:
                    # If we have also seen the partner of the bond add the bond
                    if bond[0] in curr_residue_atoms:
                        topo.bonds.append(
                            [
                                curr_residue_atoms[topo.name[-1]],
                                curr_residue_atoms[bond[0]],
                            ]
                        )
                        topo.bondtype.append(
                            bondtype_mapping.get(bond[1].upper(), bond[1].upper())
                        )
            # If the atom is part of inter-residue bonds add it to the bond_atoms_idx
            if uqatomid in all_struct_conn_atoms:
                bond_atoms_idx[uqatomid] = len(topo.resid) - 1

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

    # Add the inter-residue bonds
    if len(struct_conn):
        for atm1 in struct_conn:
            for atm2, bt in struct_conn[atm1]:
                if atm1 in bond_atoms_idx and atm2 in bond_atoms_idx:
                    topo.bonds.append([bond_atoms_idx[atm1], bond_atoms_idx[atm2]])
                    topo.bondtype.append(bt)

    if len(coords) != 0:
        allcoords.append(coords)
    if len(ideal_coords) != 0:
        ideal_allcoords.append(ideal_coords)

    allcoords = np.stack(allcoords, axis=2).astype(Molecule._dtypes["coords"])
    if len(ideal_allcoords):
        ideal_allcoords = np.stack(ideal_allcoords, axis=2).astype(
            Molecule._dtypes["coords"]
        )

    coords = allcoords
    if np.any(np.all(allcoords == 0, axis=1)) and len(ideal_allcoords):
        if np.any(np.all(ideal_allcoords == 0, axis=1)) and zerowarning:
            logger.warning(
                "Found [0, 0, 0] coordinates in molecule! Proceed with caution."
            )
        coords = ideal_allcoords

    if not macromolecule:
        topo.record = ["HETATM"] * len(topo.name)

    box = None
    boxangles = None
    if "a" in crystalinfo and "b" in crystalinfo and "c" in crystalinfo:
        box = np.array([crystalinfo["a"], crystalinfo["b"], crystalinfo["c"]])
    if "alpha" in crystalinfo and "beta" in crystalinfo and "gamma" in crystalinfo:
        boxangles = np.array(
            [crystalinfo["alpha"], crystalinfo["beta"], crystalinfo["gamma"]]
        )
    return MolFactory.construct(
        topo, Trajectory(coords=coords, box=box, boxangles=boxangles), filename, frame
    )


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
    topo.masses = np.array(
        [mass_by_type[t] for t in topo.atomtype], dtype=Molecule._dtypes["masses"]
    )
    topo.bonds = np.vstack(bonds).astype(Molecule._dtypes["bonds"])

    improper_indices = np.array(improper_indices).astype(Molecule._dtypes["impropers"])
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

    impropers = np.array(impropers).astype(Molecule._dtypes["impropers"])
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
    topo.charge = np.array(charges, dtype=Molecule._dtypes["charge"])
    topo.impropers = impropers

    return MolFactory.construct(topo, None, filename, frame)


def SDFread(filename, frame=None, topoloc=None, mol_idx=None):
    # Some (mostly correct) info here: www.nonlinear.com/progenesis/sdf-studio/v0.9/faq/sdf-file-format-guidance.aspx
    # Format is correctly specified here: https://www.daylight.com/meetings/mug05/Kappler/ctfile.pdf
    chargemap = {"7": -3, "6": -2, "5": -1, "0": 0, "3": 1, "2": 2, "1": 3, "4": 0}
    bondmap = {
        "1": "1",
        "2": "2",
        "3": "3",
        "4": "ar",  # aromatic
        "5": "un",  # single or double
        "6": "un",  # single or aromatic
        "7": "un",  # double or aromatic
        "8": "un",  # any
    }

    if mol_idx is None:
        mol_idx = 0
        logger.warning(
            "MoleculeKit will only read the first molecule from the SDF file."
        )

    v3000 = False
    with openFileOrStringIO(filename, "r") as f:
        curr_mol = 0
        lines = []
        for line in f:
            lines.append(line)
            if "V3000" in line:
                v3000 = True
            if line.strip() == "$$$$":
                if curr_mol == mol_idx:
                    break
                else:
                    curr_mol += 1
                    lines = []

        if mol_idx >= curr_mol and len(lines) == 0:
            raise RuntimeError(
                f"SDF file contains only {curr_mol} molecules. Cannot read the requested mol_idx {mol_idx}"
            )

        if v3000:
            topo, coords = parseV3000SDF(lines, chargemap, bondmap)
        else:
            topo = Topology()
            coords = []
            mol_start = 0

            molname = lines[0].strip().split()
            if len(molname):
                molname = molname[0]
            if len(molname) == 3:
                resname = molname[:3].upper()
            else:
                resname = "MOL"

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
                topo.resname.append(resname)

            for n in range(bond_start, bond_end):
                line = lines[n]
                idx1 = line[:3].strip()
                idx2 = line[3:6].strip()
                bond_type = line[6:9].strip()
                topo.bonds.append([int(idx1) - 1, int(idx2) - 1])
                topo.bondtype.append(bondmap[bond_type])

            for line in lines[bond_end:]:
                if line.strip() == "$$$$":
                    break
                if line.startswith("M  CHG"):  # These charges are more correct
                    pairs = line.strip().split()[3:]
                    for cc in range(0, len(pairs), 2):
                        topo.formalcharge[int(pairs[cc]) - 1] = int(pairs[cc + 1])

    traj = Trajectory(coords=np.vstack(coords))
    return MolFactory.construct(topo, traj, filename, frame)


def parseV3000SDF(lines, chargemap, bondmap):
    topo = Topology()
    coords = []

    molname = lines[0].strip().split()
    if len(molname):
        molname = molname[0]
    if len(molname) == 3:
        resname = molname[:3].upper()
    else:
        resname = "MOL"

    section = None
    for line in lines[4:]:
        line = line.strip()
        if line == "$$$$":
            break
        if "v30 end" in line.lower() or "m  end" in line.lower():
            section = None
            continue
        if "begin ctab" in line.lower():
            section = "ctab"
            continue
        if "begin atom" in line.lower():
            section = "atom"
            continue
        if "begin bond" in line.lower():
            section = "bond"
            continue

        if section is None:
            continue
        if section == "ctab":
            # M  V30 COUNTS 49 53 0 0 0
            if "v30 counts" in line.lower():
                n_atoms, n_bonds = map(int, line.split()[3:5])
        if section == "atom":
            # M  V30 1 C 42.4990 45.8550 38.9350 0
            pieces = line.split()[2:]
            topo.record.append("HETATM")
            topo.element.append(pieces[1])
            topo.name.append(pieces[1])
            topo.serial.append(int(pieces[0]))
            topo.formalcharge.append(chargemap[pieces[5]])
            topo.resname.append(resname)
            coords.append([float(pieces[2]), float(pieces[3]), float(pieces[4])])
        if section == "bond":
            # M  V30 bond_idx bond_type atom1 atom2
            pieces = line.split()[3:]
            topo.bonds.append([int(pieces[1]) - 1, int(pieces[2]) - 1])
            topo.bondtype.append(bondmap[pieces[0]])

    return topo, coords


def sdf_generator(sdffile):
    """Generates Molecule objects from an SDF file"""
    from io import StringIO

    sdfstr = ""
    with open(sdffile, "r") as f:
        lines = f.readlines()
        for line in lines:
            sdfstr += line
            if line.startswith("$$$$"):
                yield Molecule(StringIO(sdfstr), type="sdf", mol_idx=0)
                sdfstr = ""


# This is a hack to fix the URL fetching of MMTF in pyodide
def get_raw_data_from_url(pdb_id, reduced=False):
    """ " Get the msgpack unpacked data given a PDB id.

    :param pdb_id: the input PDB id
    :return the unpacked data (a dict)"""
    from mmtf.api.default_api import get_url, ungzip_data, _unpack, urllib2

    url = get_url(pdb_id, reduced)
    request = urllib2.Request(url)
    request.add_header("Accept-encoding", "gzip")
    response = urllib2.urlopen(request)
    if response.info().get("Content-Encoding") == "gzip":
        data = ungzip_data(response.read())
    else:
        # Fixed here from original data = response.read()
        data = response
    return _unpack(data)


def MMTFread(filename, frame=None, topoloc=None, validateElements=True):
    from mmtf.api import default_api

    # Monkey-patch the function to fix bug in data = response.read() which should not have read()
    default_api.get_raw_data_from_url = get_raw_data_from_url
    from mmtf import fetch, parse_gzip, parse

    if len(filename) == 4 and not os.path.isfile(filename):
        localpdb = _getLocalPDB(f"{filename.lower()}.mmtf.gz")
        if localpdb is not None:
            data = parse_gzip(localpdb)
        else:
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


def BCIFread(
    filename,
    frame=None,
    topoloc=None,
    zerowarning=True,
    uri="https://models.rcsb.org/{pdbid}.bcif.gz",
    covalentonly=True,
):
    from moleculekit.pdbx.reader.BinaryCifReader import BinaryCifReader

    if len(filename) == 4 and not os.path.isfile(filename):
        localpdb = _getLocalPDB(f"{filename.lower()}.bcif.gz")
        if localpdb is not None:
            filename = localpdb
        else:
            filename = uri.format(pdbid=filename.lower())

    bcr = BinaryCifReader(storeStringsAsBytes=False, defaultStringEncoding="utf-8")
    data = bcr.deserialize(filename)

    return CIFread(
        filename=filename,
        data=data,
        frame=frame,
        topoloc=topoloc,
        zerowarning=zerowarning,
        covalentonly=covalentonly,
    )


def NETCDFread(
    filename,
    frame=None,
    topoloc=None,
    stride=None,
    atom_indices=None,
):
    from moleculekit.fileformats.netcdf import netcdf_file

    _handle = netcdf_file(filename, mode="r", version=2)

    total_n_frames = _handle.variables["coordinates"].shape[0]
    n_atoms = _handle.dimensions["atom"]
    frame_slice = slice(0, total_n_frames, stride)

    if atom_indices is None:
        # get all of the atoms
        atom_slice = slice(None)
    else:
        atom_slice = ensurelist(atom_indices)
        if not np.all(atom_slice < n_atoms):
            raise ValueError(
                "As a zero-based index, the entries in "
                "atom_indices must all be less than the number of atoms "
                "in the trajectory, %d" % n_atoms
            )
        if not np.all(atom_slice >= 0):
            raise ValueError(
                "The entries in atom_indices must be greater " "than or equal to zero"
            )

    if "coordinates" in _handle.variables:
        coordinates = _handle.variables["coordinates"][frame_slice, atom_slice, :]
        coordinates = np.transpose(coordinates, [1, 2, 0])  # nm to Angstroms
    else:
        raise ValueError(
            "No coordinates found in the NetCDF file. The only "
            "variables in the file were %s" % _handle.variables.keys()
        )

    time = None
    step = None
    if "step" in _handle.variables:
        step = _handle.variables["step"][frame_slice]
    if "time" in _handle.variables:
        time = _handle.variables["time"][frame_slice]
        if len(time) > 1 and step is None:
            timestep = time[1] - time[0]
            step = time / timestep

    if "cell_lengths" in _handle.variables:
        cell_lengths = _handle.variables["cell_lengths"][frame_slice].T
    else:
        cell_lengths = None

    if "cell_angles" in _handle.variables:
        cell_angles = _handle.variables["cell_angles"][frame_slice].T
    else:
        cell_angles = None

    if cell_lengths is None and cell_angles is not None:
        logger.warning("cell_lengths were found, but no cell_angles")
    if cell_lengths is not None and cell_angles is None:
        logger.warning("cell_angles were found, but no cell_lengths")

    # scipy.io.netcdf variables are mem-mapped, and are only backed
    # by valid memory while the file handle is open. This is _bad_.
    # because we need to support the user opening the file, reading
    # the coordinates, and then closing it, and still having the
    # coordinates be a valid memory segment.
    # https://github.com/rmcgibbo/mdtraj/issues/440
    if coordinates is not None and not coordinates.flags["WRITEABLE"]:
        coordinates = np.array(coordinates, copy=True)
    if time is not None and not time.flags["WRITEABLE"]:
        time = np.array(time, copy=True)
    if step is not None and not step.flags["WRITEABLE"]:
        step = np.array(step, copy=True)
    if cell_lengths is not None and not cell_lengths.flags["WRITEABLE"]:
        cell_lengths = np.array(cell_lengths, copy=True)
    if cell_angles is not None and not cell_angles.flags["WRITEABLE"]:
        cell_angles = np.array(cell_angles, copy=True)

    _handle.close()  # Close NETCDF file handler

    # Convert to float64 before multiplying by 1000 to avoid precision loss
    if time is not None:
        time = time.astype(np.float64) * 1000  # ps to fs
    return MolFactory.construct(
        None,
        Trajectory(
            coords=coordinates,
            box=cell_lengths,
            boxangles=cell_angles,
            step=step,
            time=time,
        ),
        filename,
        frame,
    )


def DCDread(filename, frame=None, topoloc=None, stride=None, atom_indices=None):
    from moleculekit.fileformats.utils import cast_indices
    from moleculekit.dcd import DCDTrajectoryFile

    atom_indices = cast_indices(atom_indices)
    with DCDTrajectoryFile(str(filename)) as f:
        if frame is not None:
            f.seek(frame)
            n_frames = 1
        else:
            n_frames = None
        xyz, cell_lengths, cell_angles = f.read(
            n_frames=n_frames, stride=stride, atom_indices=atom_indices
        )
        istart, nsavc, delta = f.read_header()
        # Timestep conversion factor found in OpenMM
        delta = np.round(delta * 0.04888821, decimals=8)
        steps = np.arange(istart, (nsavc * xyz.shape[0]) + istart, nsavc)
        if stride is not None:
            steps = steps[::stride]

    xyz = np.transpose(xyz, (1, 2, 0))
    if cell_lengths is not None:
        cell_lengths = cell_lengths.T
    if cell_angles is not None:
        cell_angles = cell_angles.T

    return MolFactory.construct(
        None,
        Trajectory(
            coords=xyz,
            box=cell_lengths,
            boxangles=cell_angles,
            step=steps,
            time=steps * delta * 1000,  # ps to fs
        ),
        filename,
        frame,
    )


def TRRread(filename, frame=None, topoloc=None, stride=None, atom_indices=None):
    from moleculekit.trr import load_trr
    from moleculekit.unitcell import box_vectors_to_lengths_and_angles

    xyz, time, step, boxvectors, _ = load_trr(
        filename, stride=stride, atom_indices=atom_indices, frame=frame
    )
    xyz = np.transpose(xyz, (1, 2, 0)) * 10  # nm to Angstroms

    v1 = boxvectors[:, 0, :]
    v2 = boxvectors[:, 1, :]
    v3 = boxvectors[:, 2, :]
    (
        a_length,
        b_length,
        c_length,
        alpha,
        beta,
        gamma,
    ) = box_vectors_to_lengths_and_angles(v1, v2, v3)
    box = np.vstack((a_length, b_length, c_length)) * 10  # nm to Angstroms
    boxangles = np.vstack((alpha, beta, gamma))

    if len(time) > 1 and (step is None or len(step) != len(time)):
        timestep = time[1] - time[0]
        step = time / timestep

    # Convert to float64 before multiplication to avoid precision loss
    time = time.astype(np.float64)
    time *= 1000  # ps to fs
    return MolFactory.construct(
        None,
        Trajectory(
            coords=xyz,
            box=box,
            boxangles=boxangles,
            step=step,
            time=time,
        ),
        filename,
        frame,
    )


def BINPOSread(filename, frame=None, topoloc=None, stride=None, atom_indices=None):
    from moleculekit.binpos import load_binpos

    xyz = load_binpos(filename, stride=stride, atom_indices=atom_indices, frame=frame)
    xyz = np.transpose(xyz, (1, 2, 0))
    return MolFactory.construct(None, Trajectory(coords=xyz), filename, frame)


def INPCRDread(filename, frame=None, topoloc=None, stride=None, atom_indices=None):
    with open(filename, "r", encoding="ascii") as f:
        # Skip title
        f.readline()
        # Read number of atoms
        natoms = int(f.readline().strip())

        # Read coordinates
        coords = []
        while len(coords) < natoms * 3:  # Read until we have all coordinates
            line = f.readline()
            # Each value uses 12.7f format
            for i in range(0, len(line), 12):
                if i + 12 <= len(line):
                    coords.append(float(line[i : i + 12]))

        coords = np.array(coords).reshape(natoms, 3)

        # Try to read box information if it exists
        box = None
        boxangles = None
        line = f.readline()
        if line:  # If there's an extra line, it contains box info
            boxinfo = [float(line[i : i + 12]) for i in range(0, len(line) - 1, 12)]
            if len(boxinfo) == 6:  # Should contain 3 lengths and 3 angles
                box = np.array(boxinfo[:3])
                boxangles = np.array(boxinfo[3:])

    # Add singleton dimension for frames
    coords = coords.reshape(natoms, 3, 1)
    if box is not None:
        box = box.reshape(3, 1)
        boxangles = boxangles.reshape(3, 1)

    return MolFactory.construct(
        None, Trajectory(coords=coords, box=box, boxangles=boxangles), filename, frame
    )


def JSONread(filename, frame=None, topoloc=None, stride=None, atom_indices=None):
    from moleculekit.molecule import Molecule
    import json

    with open(filename, "r") as f:
        data = json.load(f)

    if "moleculekit_version" not in data:
        raise ValueError(f"This file {filename} is not a MoleculeKit JSON file")
    del data["moleculekit_version"]

    mol = Molecule.fromDict(data)
    mol.topoloc = filename
    return mol


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
    "alphafold": ALPHAFOLDread,
    "bcif": BCIFread,
    "json": JSONread,
    # "cifnew": CIFread_new,
}

_MDTRAJ_TOPOLOGY_EXTS = [
    "h5",
    "lh5",
    "parm7",
    "hoomdxml",
    "gro",
    "arc",
    "hdf5",
]
for ext in _MDTRAJ_TOPOLOGY_EXTS:
    if ext not in _TOPOLOGY_READERS:
        _TOPOLOGY_READERS[ext] = MDTRAJTOPOread

_TRAJECTORY_READERS = {
    "xtc": XTCread,
    "xsc": XSCread,
    "nc": NETCDFread,
    "netcdf": NETCDFread,
    "ncdf": NETCDFread,
    "dcd": DCDread,
    "trr": TRRread,
    "binpos": BINPOSread,
    "h5": MDTRAJread,
    "lh5": MDTRAJread,
}

_COORDINATE_READERS = {"crd": CRDread, "coor": BINCOORread, "inpcrd": INPCRDread}


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
