# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
from moleculekit.projections.projection import Projection
import numpy as np
import logging

logger = logging.getLogger(__name__)


class AtomNotFoundException(Exception):
    pass


class Dihedral:
    """Class to store atoms defining a dihedral angle.

    Example
    -------
    >>> # Using the helper functions to construct a dihedral object
    >>> d1 = Dihedral.phi(mol, 5, 6, segid='P0')
    >>> d2 = Dihedral.chi1(mol, 12, segid='P0') # Defining segid
    >>> d3 = Dihedral.chi1(mol, 38, chain='X') # Defining chain
    >>> # Manual construction
    >>> atom1 = {'name': 'N', 'resid': 5, 'segid': 'P'}
    >>> atom2 = {'name': 'CA', 'resid': 3, 'segid': 'P', 'chain': 'A', 'insertion': 'B'}
    >>> atom3 = {'name': 'C', 'resid': 46, 'segid': 'P', 'chain': 'X'}
    >>> atom4 = {'name': 'O', 'resid': 2, 'segid': 'P'}
    >>> d = Dihedral(atom1, atom2, atom3, atom4)
    """

    def __init__(self, atom1, atom2, atom3, atom4, dihedraltype=None, check_valid=True):
        if check_valid:
            valid_keys = ("name", "resid", "segid", "insertion", "chain")
            default_values = {
                "name": "",
                "resid": 0,
                "segid": "",
                "insertion": "",
                "chain": "",
            }
            for i, a in enumerate([atom1, atom2, atom3, atom4]):
                for k in a.keys():
                    if k not in valid_keys:
                        raise RuntimeError(
                            'Dictionary key can\'t be "{}". Valid keys are: {}'.format(
                                k, valid_keys
                            )
                        )
                for k in np.setdiff1d(valid_keys, tuple(a.keys())):
                    print(np.setdiff1d(valid_keys, tuple(a.keys())))
                    a[k] = default_values[k]
                # frames.append(pd.DataFrame(a, index=[i]))
        # self.atoms = pd.concat(frames)
        self.atoms = [atom1, atom2, atom3, atom4]
        self.dihedraltype = dihedraltype

    def __str__(self):
        descr = ""
        if self.dihedraltype is not None:
            descr = f'"{self.dihedraltype}" dihedral angle including atoms:\n'
        descr += "name\tresid\tinsertion\tchain\tsegid\n"
        for a in self.atoms:
            descr += "{}\t{}\t{}\t\t{}\t{}\n".format(
                a["name"], a["resid"], a["insertion"], a["chain"], a["segid"]
            )
        return descr

    def __repr__(self):
        return self.__str__()

    def _explanation(self):
        descr = ""
        if self.dihedraltype is not None:
            descr += self.dihedraltype + " "
        resids = []
        insertions = []
        for a in self.atoms:
            resids.append(a["resid"])
            insertions.append(a["insertion"])
        uqresids = list(np.unique(resids))
        uqinsertions = list(np.unique(insertions))

        descr += "dihedral of Resid/Insertion/Chain/Segid/ {}/{}/{}/{}".format(
            ",".join(map(str, uqresids)),
            ",".join(uqinsertions),
            self.atoms[0]["chain"],
            self.atoms[0]["segid"],
        )
        return descr

    @staticmethod
    def dihedralsToIndexes(mol, dihedrals, sel="all"):
        """Converts dihedral objects to atom indexes of a given Molecule

        Parameters
        ----------
        mol : :class:`Molecule <moleculekit.molecule.Molecule>` object
            A Molecule object from which to obtain atom information
        dihedrals : list
            A single dihedral or a list of Dihedral objects
        sel : str
            Atom selection string to restrict the application of the selections.
            See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__

        Returns
        -------
        indexes : list of lists
            A list containing a list of atoms that correspond to each dihedral.

        Examples
        --------
        >>> dihs = []
        >>> dihs.append(Dihedral.phi(mol, 1, 2))
        >>> dihs.append(Dihedral.psi(mol, 2, 3))
        >>> indexes = Dihedral.dihedralsToIndexes(mol, dihs)
        """
        selatoms = mol.atomselect(sel)
        from moleculekit.util import ensurelist

        indexes = []
        for dih in ensurelist(dihedrals):
            idx = []
            for a in dih.atoms:
                atomsel = (
                    (mol.name == a["name"])
                    & (mol.resid == a["resid"])
                    & (mol.insertion == a["insertion"])
                    & (mol.chain == a["chain"])
                    & (mol.segid == a["segid"])
                )
                atomsel = atomsel & selatoms
                if np.sum(atomsel) != 1:
                    raise RuntimeError(
                        "Expected one atom from atomselection {}. Got {} instead.".format(
                            a, np.sum(atomsel)
                        )
                    )
                idx.append(np.where(atomsel)[0][0])
            indexes.append(idx)
        return indexes

    @staticmethod
    def _findResidue(mol, resid, insertion=None, chain=None, segid=None):
        idx = mol.resid == resid
        descr = f'Resid "{resid}"'
        if insertion is not None:
            idx &= mol.insertion == insertion
            descr += f' Insertion "{insertion}"'
        if chain is not None:
            idx &= mol.chain == chain
            descr += f' Chain "{chain}"'
        if segid is not None:
            idx &= mol.segid == segid
            descr += f' Segid "{segid}"'

        if np.sum(idx) == 0:
            raise RuntimeError(f"No residues found with description ({descr})")

        idx2 = np.where(idx)[0]
        if not np.array_equal(idx2, np.arange(idx2[0], idx2[-1] + 1)):
            raise RuntimeError(
                "Residue with ({}) has non-continuous indexes in PBD file ({})".format(
                    descr, np.where(idx)[0]
                )
            )

        uqins = np.unique(mol.insertion[idx])
        if len(uqins) > 1:
            raise RuntimeError(
                "Residue with ({}) exists with multiple insertions ({}). Define insertion to disambiguate.".format(
                    descr, uqins
                )
            )
        uqchain = np.unique(mol.chain[idx])
        if len(uqchain) > 1:
            raise RuntimeError(
                "Residue with ({}) exists in multiple chains ({}). Define chain to disambiguate.".format(
                    descr, uqchain
                )
            )
        uqseg = np.unique(mol.segid[idx])
        if len(uqseg) > 1:
            raise RuntimeError(
                "Residue with ({}) exists in multiple segments ({}). Define segid to disambiguate.".format(
                    descr, uqseg
                )
            )
        return {
            "resid": resid,
            "insertion": uqins[0],
            "chain": uqchain[0],
            "segid": uqseg[0],
            "idx": idx,
        }

    @staticmethod
    def _findAtom(mol, name, resdict):
        atomidx = resdict["idx"] & (mol.name == name)
        if np.sum(atomidx) == 0:
            raise AtomNotFoundException(
                f'No atoms found with description ({resdict}) and name "{name}".'
            )
        newresdict = {
            "name": name,
            "resid": resdict["resid"],
            "insertion": resdict["insertion"],
            "chain": resdict["chain"],
            "segid": resdict["segid"],
        }
        return newresdict

    @staticmethod
    def _findResname(mol, resdict):
        residx = (
            (mol.resid == resdict["resid"])
            & (mol.insertion == resdict["insertion"])
            & (mol.chain == resdict["chain"])
            & (mol.segid == resdict["segid"])
        )
        uqresname = np.unique(mol.resname[residx])

        if len(uqresname) > 1:
            raise RuntimeError(f"Multiple resnames ({uqresname}) found in ({resdict})")
        return uqresname[0]

    @staticmethod
    def proteinDihedrals(
        mol, sel="protein or resname ACE NME", dih=("psi", "phi"), ff="amber"
    ):
        """Returns a list of tuples containing the four resid/atom pairs for each dihedral of the protein

        Parameters
        ----------
        mol : :class:`Molecule <moleculekit.molecule.Molecule>` object
            A Molecule object from which to obtain structural information
        sel : str
            Atom selection string to restrict the atoms for which to calculate dihedrals (e.g. only one of many chains).
            See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__
        dih : tuple
            A tuple of the dihedral types we want to calculate (phi, psi, omega, chi1, chi2, chi3, chi4, chi5)

        Returns
        -------
        dihedrals : list of :class:`Dihedral <moleculekit.projections.metricdihedral.Dihedral>` objects
            A list of Dihedral objects
        """
        mol = mol.copy()
        mol.filter(sel, _logger=False)
        segments = []  # Here I consider segments both chains and segments
        residues = []
        ro = io = co = so = None  # "old" values as in previous atom's
        for r, i, c, s in zip(mol.resid, mol.insertion, mol.chain, mol.segid):
            if co is not None and so is not None and ((c != co) or (s != so)):
                segments.append(residues)
                residues = []
            if (r != ro) or (i != io) or (c != co) or (s != so):
                residues.append((r, i, c, s))
                ro, io, co, so = (r, i, c, s)
        if len(residues) != 0:
            segments.append(residues)

        dihedrals = []
        for s in segments:
            residues = s
            for r in range(len(residues)):
                resid, ins, chain, segid = residues[r]
                starting = r == 0
                ending = r == len(residues) - 1
                if "phi" in dih and not starting:
                    resid2, ins2, _, _ = residues[r - 1]
                    dihedrals.append(
                        Dihedral.phi(mol, resid2, resid, segid, chain, ins2, ins, ff)
                    )
                if "psi" in dih and not ending:
                    resid2, ins2, _, _ = residues[r + 1]
                    dihedrals.append(
                        Dihedral.psi(mol, resid, resid2, segid, chain, ins, ins2, ff)
                    )
                if "omega" in dih and not ending:
                    resid2, ins2, _, _ = residues[r + 1]
                    dihedrals.append(
                        Dihedral.omega(mol, resid, resid2, segid, chain, ins, ins2, ff)
                    )
                if "chi1" in dih:
                    dihedrals.append(Dihedral.chi1(mol, resid, segid, chain, ins, ff))
                if "chi2" in dih:
                    dihedrals.append(Dihedral.chi2(mol, resid, segid, chain, ins, ff))
                if "chi3" in dih:
                    dihedrals.append(Dihedral.chi3(mol, resid, segid, chain, ins, ff))
                if "chi4" in dih:
                    dihedrals.append(Dihedral.chi4(mol, resid, segid, chain, ins, ff))
                if "chi5" in dih:
                    dihedrals.append(Dihedral.chi5(mol, resid, segid, chain, ins, ff))
        return [d for d in dihedrals if d is not None]

    # Sidechain dihedral atoms taken from
    # http://www.ccp14.ac.uk/ccp/web-mirrors/garlic/garlic/commands/dihedrals.html
    @staticmethod
    def phi(
        mol,
        res1,
        res2,
        segid=None,
        chain=None,
        insertion1=None,
        insertion2=None,
        ff="amber",
    ):
        """Constructs a Dihedral object corresponding to the phi angle of res1 and res2

        Parameters
        ----------
        mol : :class:`Molecule <moleculekit.molecule.Molecule>` object
            A Molecule object from which to obtain structural information
        res1 : int
            The resid of the first residue containing the C atom
        res2 : int
            The resid of the second residue containing the N CA C atoms
        segid : str
            The segment id of the residues
        chain : str
            The chain letter of the residues
        insertion1 : str
            The insertion letter of residue 1
        insertion2 : str
            The insertion letter of residue 2

        Returns
        -------
        dihedral : :class:`Dihedral <moleculekit.projections.metricdihedral.Dihedral>` object
            A Dihedral object
        """
        res1dict = Dihedral._findResidue(mol, res1, insertion1, chain, segid)
        res2dict = Dihedral._findResidue(mol, res2, insertion2, chain, segid)

        a1 = Dihedral._findAtom(mol, "C", res1dict)
        try:  # Check if backbone atoms exist. Capped terminals don't have them all
            a2 = Dihedral._findAtom(mol, "N", res2dict)
            a3 = Dihedral._findAtom(mol, "CA", res2dict)
            a4 = Dihedral._findAtom(mol, "C", res2dict)
        except AtomNotFoundException:
            return None

        return Dihedral(a1, a2, a3, a4, dihedraltype="phi", check_valid=False)

    @staticmethod
    def psi(
        mol,
        res1,
        res2,
        segid=None,
        chain=None,
        insertion1=None,
        insertion2=None,
        ff="amber",
    ):
        """Constructs a Dihedral object corresponding to the psi angle of res1 and res2

        Parameters
        ----------
        mol : :class:`Molecule <moleculekit.molecule.Molecule>` object
            A Molecule object from which to obtain structural information
        res1 : int
            The resid of the first residue containing the N CA C atoms
        res2 : int
            The resid of the second residue containing the N atom
        segid : str
            The segment id of the residues
        chain : str
            The chain letter of the residues
        insertion1 : str
            The insertion letter of residue 1
        insertion2 : str
            The insertion letter of residue 2

        Returns
        -------
        dihedral : :class:`Dihedral <moleculekit.projections.metricdihedral.Dihedral>` object
            A Dihedral object
        """
        res1dict = Dihedral._findResidue(mol, res1, insertion1, chain, segid)
        res2dict = Dihedral._findResidue(mol, res2, insertion2, chain, segid)

        try:  # Check if backbone atoms exist. Capped terminals don't have them all
            a1 = Dihedral._findAtom(mol, "N", res1dict)
            a2 = Dihedral._findAtom(mol, "CA", res1dict)
            a3 = Dihedral._findAtom(mol, "C", res1dict)
        except AtomNotFoundException:
            return None

        a4 = Dihedral._findAtom(mol, "N", res2dict)

        return Dihedral(a1, a2, a3, a4, dihedraltype="psi", check_valid=False)

    @staticmethod
    def omega(
        mol,
        res1,
        res2,
        segid=None,
        chain=None,
        insertion1=None,
        insertion2=None,
        ff="amber",
    ):
        """Constructs a Dihedral object corresponding to the omega angle of res1 and res2

        Parameters
        ----------
        mol : :class:`Molecule <moleculekit.molecule.Molecule>` object
            A Molecule object from which to obtain structural information
        res1 : int
            The resid of the first residue containing the CA C atoms
        res2 : int
            The resid of the second residue containing the N CA atoms
        segid : str
            The segment id of the residues
        chain : str
            The chain letter of the residues
        insertion1 : str
            The insertion letter of residue 1
        insertion2 : str
            The insertion letter of residue 2

        Returns
        -------
        dihedral : :class:`Dihedral <moleculekit.projections.metricdihedral.Dihedral>` object
            A Dihedral object
        """
        res1dict = Dihedral._findResidue(mol, res1, insertion1, chain, segid)
        res2dict = Dihedral._findResidue(mol, res2, insertion2, chain, segid)

        try:  # Check if backbone atoms exist. Capped terminals don't have them all
            a1 = Dihedral._findAtom(mol, "CA", res1dict)
            a2 = Dihedral._findAtom(mol, "C", res1dict)
            a3 = Dihedral._findAtom(mol, "N", res2dict)
            a4 = Dihedral._findAtom(mol, "CA", res2dict)
        except AtomNotFoundException:
            return None

        return Dihedral(a1, a2, a3, a4, dihedraltype="omega", check_valid=False)

    @staticmethod
    def chi1(mol, res, segid=None, chain=None, insertion=None, ff="amber"):
        """Constructs a Dihedral object corresponding to the chi1 angle of a residue

        Parameters
        ----------
        mol : :class:`Molecule <moleculekit.molecule.Molecule>` object
            A Molecule object from which to obtain structural information
        res : int
            The resid of the residue
        segid : str
            The segment id of the residue
        chain : str
            The chain letter of the residue
        insertion : str
            The insertion letter of the residue

        Returns
        -------
        dihedral : :class:`Dihedral <moleculekit.projections.metricdihedral.Dihedral>` object
            A Dihedral object
        """
        chi1std = ("N", "CA", "CB", "CG")
        chi1 = {
            "ARG": chi1std,
            "ASN": chi1std,
            "ASP": chi1std,
            "CYS": ("N", "CA", "CB", "SG"),
            "GLN": chi1std,
            "GLU": chi1std,
            "HIS": chi1std,
            "ILE": ("N", "CA", "CB", "CG1"),
            "LEU": chi1std,
            "LYS": chi1std,
            "MET": chi1std,
            "PHE": chi1std,
            "PRO": chi1std,
            "SER": ("N", "CA", "CB", "OG"),
            "THR": ("N", "CA", "CB", "OG1"),
            "TRP": chi1std,
            "TYR": chi1std,
            "VAL": ("N", "CA", "CB", "CG1"),
        }

        resdict = Dihedral._findResidue(mol, res, insertion, chain, segid)
        resname = Dihedral._findResname(mol, resdict)
        Dihedral._checkKnownResidues(resname)
        if resname not in chi1:
            return None
        return Dihedral(
            Dihedral._findAtom(mol, chi1[resname][0], resdict),
            Dihedral._findAtom(mol, chi1[resname][1], resdict),
            Dihedral._findAtom(mol, chi1[resname][2], resdict),
            Dihedral._findAtom(mol, chi1[resname][3], resdict),
            dihedraltype="chi1",
            check_valid=False,
        )

    @staticmethod
    def chi2(mol, res, segid=None, chain=None, insertion=None, ff="amber"):
        """Constructs a Dihedral object corresponding to the chi2 angle of a residue

        Parameters
        ----------
        mol : :class:`Molecule <moleculekit.molecule.Molecule>` object
            A Molecule object from which to obtain structural information
        res : int
            The resid of the residue
        segid : str
            The segment id of the residue
        chain : str
            The chain letter of the residue
        insertion : str
            The insertion letter of the residue

        Returns
        -------
        dihedral : :class:`Dihedral <moleculekit.projections.metricdihedral.Dihedral>` object
            A Dihedral object
        """
        if ff.lower() == "amber":
            chi2std = ("CA", "CB", "CG", "CD")
            chi2 = {
                "ARG": chi2std,
                "ASN": ("CA", "CB", "CG", "OD1"),
                "ASP": ("CA", "CB", "CG", "OD1"),
                "GLN": chi2std,
                "GLU": chi2std,
                "HIS": ("CA", "CB", "CG", "ND1"),
                "ILE": ("CA", "CB", "CG1", "CD1"),
                "LEU": ("CA", "CB", "CG", "CD1"),
                "LYS": chi2std,
                "MET": ("CA", "CB", "CG", "SD"),
                "PHE": ("CA", "CB", "CG", "CD1"),
                "PRO": chi2std,
                "TRP": ("CA", "CB", "CG", "CD1"),
                "TYR": ("CA", "CB", "CG", "CD1"),
            }
        if ff.lower() == "charmm":
            chi2std = ("CA", "CB", "CG", "CD")
            chi2 = {
                "ARG": chi2std,
                "ASN": ("CA", "CB", "CG", "OD1"),
                "ASP": ("CA", "CB", "CG", "OD1"),
                "GLN": chi2std,
                "GLU": chi2std,
                "HIS": ("CA", "CB", "CG", "ND1"),
                "ILE": ("CA", "CB", "CG1", "CD"),
                "LEU": ("CA", "CB", "CG", "CD1"),
                "LYS": chi2std,
                "MET": ("CA", "CB", "CG", "SD"),
                "PHE": ("CA", "CB", "CG", "CD1"),
                "PRO": chi2std,
                "TRP": ("CA", "CB", "CG", "CD1"),
                "TYR": ("CA", "CB", "CG", "CD1"),
            }

        resdict = Dihedral._findResidue(mol, res, insertion, chain, segid)
        resname = Dihedral._findResname(mol, resdict)
        Dihedral._checkKnownResidues(resname)
        if resname not in chi2:
            return None
        return Dihedral(
            Dihedral._findAtom(mol, chi2[resname][0], resdict),
            Dihedral._findAtom(mol, chi2[resname][1], resdict),
            Dihedral._findAtom(mol, chi2[resname][2], resdict),
            Dihedral._findAtom(mol, chi2[resname][3], resdict),
            dihedraltype="chi2",
            check_valid=False,
        )

    @staticmethod
    def chi3(mol, res, segid=None, chain=None, insertion=None, ff="amber"):
        """Constructs a Dihedral object corresponding to the chi3 angle of a residue

        Parameters
        ----------
        mol : :class:`Molecule <moleculekit.molecule.Molecule>` object
            A Molecule object from which to obtain structural information
        res : int
            The resid of the residue
        segid : str
            The segment id of the residue
        chain : str
            The chain letter of the residue
        insertion : str
            The insertion letter of the residue

        Returns
        -------
        dihedral : :class:`Dihedral <moleculekit.projections.metricdihedral.Dihedral>` object
            A Dihedral object
        """
        chi3 = {
            "ARG": ("CB", "CG", "CD", "NE"),
            "GLN": ("CB", "CG", "CD", "OE1"),
            "GLU": ("CB", "CG", "CD", "OE1"),
            "LYS": ("CB", "CG", "CD", "CE"),
            "MET": ("CB", "CG", "SD", "CE"),
        }

        resdict = Dihedral._findResidue(mol, res, insertion, chain, segid)
        resname = Dihedral._findResname(mol, resdict)
        Dihedral._checkKnownResidues(resname)
        if resname not in chi3:
            return None
        return Dihedral(
            Dihedral._findAtom(mol, chi3[resname][0], resdict),
            Dihedral._findAtom(mol, chi3[resname][1], resdict),
            Dihedral._findAtom(mol, chi3[resname][2], resdict),
            Dihedral._findAtom(mol, chi3[resname][3], resdict),
            dihedraltype="chi3",
            check_valid=False,
        )

    @staticmethod
    def chi4(mol, res, segid=None, chain=None, insertion=None, ff="amber"):
        """Constructs a Dihedral object corresponding to the chi4 angle of a residue

        Parameters
        ----------
        mol : :class:`Molecule <moleculekit.molecule.Molecule>` object
            A Molecule object from which to obtain structural information
        res : int
            The resid of the residue
        segid : str
            The segment id of the residue
        chain : str
            The chain letter of the residue
        insertion : str
            The insertion letter of the residue

        Returns
        -------
        dihedral : :class:`Dihedral <moleculekit.projections.metricdihedral.Dihedral>` object
            A Dihedral object
        """
        chi4 = {"ARG": ("CG", "CD", "NE", "CZ"), "LYS": ("CG", "CD", "CE", "NZ")}

        resdict = Dihedral._findResidue(mol, res, insertion, chain, segid)
        resname = Dihedral._findResname(mol, resdict)
        Dihedral._checkKnownResidues(resname)
        if resname not in chi4:
            return None
        return Dihedral(
            Dihedral._findAtom(mol, chi4[resname][0], resdict),
            Dihedral._findAtom(mol, chi4[resname][1], resdict),
            Dihedral._findAtom(mol, chi4[resname][2], resdict),
            Dihedral._findAtom(mol, chi4[resname][3], resdict),
            dihedraltype="chi4",
            check_valid=False,
        )

    @staticmethod
    def chi5(mol, res, segid=None, chain=None, insertion=None, ff="amber"):
        """Constructs a Dihedral object corresponding to the chi5 angle of a residue

        Parameters
        ----------
        mol : :class:`Molecule <moleculekit.molecule.Molecule>` object
            A Molecule object from which to obtain structural information
        res : int
            The resid of the residue
        segid : str
            The segment id of the residue
        chain : str
            The chain letter of the residue
        insertion : str
            The insertion letter of the residue

        Returns
        -------
        dihedral : :class:`Dihedral <moleculekit.projections.metricdihedral.Dihedral>` object
            A Dihedral object
        """
        chi5 = {"ARG": ("CD", "NE", "CZ", "NH1")}

        resdict = Dihedral._findResidue(mol, res, insertion, chain, segid)
        resname = Dihedral._findResname(mol, resdict)
        Dihedral._checkKnownResidues(resname)
        if resname not in chi5:
            return None
        return Dihedral(
            Dihedral._findAtom(mol, chi5[resname][0], resdict),
            Dihedral._findAtom(mol, chi5[resname][1], resdict),
            Dihedral._findAtom(mol, chi5[resname][2], resdict),
            Dihedral._findAtom(mol, chi5[resname][3], resdict),
            dihedraltype="chi5",
            check_valid=False,
        )

    @staticmethod
    def _checkKnownResidues(resname):
        knownresnames = [
            "ARG",
            "ASN",
            "ASP",
            "CYS",
            "GLN",
            "GLU",
            "HIS",
            "ILE",
            "LEU",
            "LYS",
            "MET",
            "PHE",
            "PRO",
            "SER",
            "THR",
            "TRP",
            "TYR",
            "VAL",
            "GLY",
            "ALA",
        ]
        if resname not in knownresnames:
            raise RuntimeError(
                "Residue {} not in list of known residues {}. Rename your residues to match these.".format(
                    resname, knownresnames
                )
            )


class MetricDihedral(Projection):
    """Calculates a set of dihedral angles from trajectories

    Parameters
    ----------
    dih : list of :class:`Dihedral <moleculekit.projections.metricdihedral.Dihedral>` object
        You can provide your own list of Dihedral objects. See example.
    sincos : bool, optional
        Set to True to return the dihedral angles as their sine and cosine components. Makes them periodic.
    protsel : str, optional
        Atom selection string for the protein segment for which to calculate dihedral angles. Resids should be unique
        within that segment. See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__

    Examples
    --------
    >>> mol = Molecule('3PTB')
    >>> mol.filter('not insertion A')
    >>> met = MetricDihedral()
    >>> met.project(mol)
    >>> # More complicated example
    >>> dih = []
    >>> dih.append(Dihedral.chi1(mol, 45))
    >>> dih.append(Dihedral.psi(mol, 29, 30))
    >>> met = MetricDihedral(dih, protsel='protein and segid 0')
    >>> met.project(mol)
    >>> met.getMapping(mol)
    """

    def __init__(self, dih=None, sincos=True, protsel="protein or resname ACE NME"):
        super().__init__()

        if dih is not None and not isinstance(dih[0], Dihedral):
            raise RuntimeError(
                "Manually passing dihedrals to MetricDihedral requires use of the Dihedral class. Check the example in the documentation"
            )

        self._protsel = protsel
        self._sincos = sincos
        self._dihedrals = dih

    def project(self, mol):
        """Project molecule.

        Parameters
        ----------
        mol : :class:`Molecule <moleculekit.molecule.Molecule>`
            A :class:`Molecule <moleculekit.molecule.Molecule>` object to project.

        Returns
        -------
        data : np.ndarray
            An array containing the projected data.
        """
        dihedrals = self._getMolProp(mol, "dihedrals")
        return self._calcDihedralAngles(mol, dihedrals, sincos=self._sincos)

    def getMapping(self, mol):
        """Returns the description of each projected dimension.

        Parameters
        ----------
        mol : :class:`Molecule <moleculekit.molecule.Molecule>` object
            A Molecule object which will be used to calculate the descriptions of the projected dimensions.

        Returns
        -------
        map : :class:`DataFrame <pandas.core.frame.DataFrame>` object
            A DataFrame containing the descriptions of each dimension
        """
        dihedrals = self._getMolProp(mol, "dihedrals")

        from pandas import DataFrame

        types = []
        indexes = []
        description = []
        for i, dih in enumerate(dihedrals):
            types += ["dihedral"]
            indexes += [dih]
            mapstr = ""
            for d in dih:
                mapstr += "({} {} {} {} {}) ".format(
                    mol.resname[d],
                    mol.resid[d],
                    mol.name[d],
                    mol.segid[d],
                    mol.chain[d],
                )
            if self._sincos:
                description += ["Sine of angle of " + mapstr]
                description += ["Cosine of angle of " + mapstr]
                types += ["dihedral"]
                indexes += [dih]
            else:
                description += ["Angle of " + mapstr]
        return DataFrame(
            {"type": types, "atomIndexes": indexes, "description": description}
        )

    def _calculateMolProp(self, mol, props="all"):
        from moleculekit.util import ensurelist

        res = {}
        protsel = mol.atomselect(self._protsel)

        if self._dihedrals is None:  # Default phi psi dihedrals
            dihedrals = Dihedral.proteinDihedrals(mol, protsel)
        else:
            dihedrals = ensurelist(self._dihedrals)

        res["dihedrals"] = Dihedral.dihedralsToIndexes(mol, dihedrals, protsel)
        return res

    def _calcDihedralAngles(self, mol, dihedrals, sincos=True):
        from moleculekit.dihedral import dihedralAngle

        metric = np.zeros((np.size(mol.coords, 2), len(dihedrals)))

        for i, dih in enumerate(dihedrals):
            metric[:, i] = np.rad2deg(dihedralAngle(mol.coords[dih, :, :]))

        if sincos:
            sc_metric = np.zeros((np.size(metric, 0), np.size(metric, 1) * 2))
            sc_metric[:, 0::2] = np.sin(metric * np.pi / 180.0)
            sc_metric[:, 1::2] = np.cos(metric * np.pi / 180.0)
            metric = sc_metric
        return metric.astype(np.float32)


import unittest


class _TestMetricDihedral(unittest.TestCase):
    def test_dihedral_traj(self):
        from moleculekit.molecule import Molecule
        from moleculekit.home import home
        from os import path

        mol = Molecule(
            path.join(home(dataDir="test-projections"), "trajectory", "filtered.pdb")
        )
        mol.read(path.join(home(dataDir="test-projections"), "trajectory", "traj.xtc"))

        metr = MetricDihedral(protsel="protein")
        data = metr.project(mol)
        dataref = np.load(
            path.join(home(dataDir="test-projections"), "metricdihedral", "ref.npy")
        )
        assert np.allclose(
            data, dataref, atol=1e-03
        ), "Diherdals calculation gave different results from reference"

    def test_dihedral_5mat(self):
        from moleculekit.molecule import Molecule
        from moleculekit.home import home
        from moleculekit.tools.autosegment import autoSegment
        from os import path

        mol = Molecule("5MAT")
        mol.filter("not insertion A and not altloc A B", _logger=False)
        mol = autoSegment(mol, _logger=False)

        data = MetricDihedral().project(mol)
        dataref = np.load(
            path.join(home(dataDir="test-projections"), "metricdihedral", "5mat.npy")
        )
        assert np.allclose(
            data, dataref, atol=1e-03
        ), "Diherdals calculation gave different results from reference"

        ref_idx = np.load(
            path.join(
                home(dataDir="test-projections"),
                "metricdihedral",
                "5mat_mapping_indexes.npy",
            )
        )
        mapping = MetricDihedral().getMapping(mol)
        mapping_idx = np.vstack(mapping.atomIndexes.to_numpy())

        assert np.array_equal(
            mapping_idx, ref_idx
        ), "Mapping of atom indexes has changed"

    def test_dialanine_ace_nme(self):
        from moleculekit.molecule import Molecule
        from moleculekit.home import home
        from os import path

        mol = Molecule(
            path.join(
                home(dataDir="test-projections"),
                "metricdihedral",
                "dialanine-peptide.pdb",
            )
        )
        data = MetricDihedral().project(mol)

        refarray = np.array(
            [[-0.71247578, -0.70169669, 0.27399951, -0.96172982]], dtype=np.float32
        )
        assert np.allclose(refarray, data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
