# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#

from moleculekit.projections.projection import Projection
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MetricSecondaryStructure(Projection):
    """Calculates the secondary structure of the protein. DSSP implementation and documentation taken from MDtraj.

    Parameters
    ----------
    sel : str
        Atom selection string for the protein.
        See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__
    simplified: bool
        Uses the simplified 3-letter code
    integer : bool
        Use integers instead of letter codes.

    Notes
    -----
    The simplified DSSP codes are:
        - 'C' : Coil. Either of the 'T', 'S' or ' ' codes. Integer code: 0
        - 'E' : Strand. Either of the 'E', or 'B' codes. Integer code: 1
        - 'H' : Helix. Either of the 'H', 'G', or 'I' codes. Integer code: 2

    The full DSSP assignment codes are:
        - 'H' : Alpha helix. Integer code: 3
        - 'B' : Residue in isolated beta-bridge. Integer code: 4
        - 'E' : Extended strand, participates in beta ladder. Integer code: 5
        - 'G' : 3-helix (3/10 helix). Integer code: 6
        - 'I' : 5 helix (pi helix). Integer code: 7
        - 'T' : hydrogen bonded turn. Integer code: 8
        - 'S' : bend. Integer code: 9
        - ' ' : Loops and irregular elements. Integer code: 10

    A special 'NA' code will be assigned to each 'residue' in the topology which
    isn't actually a protein residue (does not contain atoms with the names
    'CA', 'N', 'C', 'O'), such as water molecules that are listed as 'residue's
    in the topology.
    """

    def __init__(self, sel="protein", simplified=True, integer=True):
        super().__init__()

        self.sel = sel
        self.simplified = simplified
        self.integer = integer

    def _precalculate(self, mol):
        (
            self._pc_ca_indices,
            self._pc_nco_indices,
            self._pc_proline_indices,
            self._pc_chain_ids,
        ) = self._calcarrays(mol)

    def _calculateMolProp(self, mol, props="all"):
        from moleculekit.util import sequenceID

        # Calculate all properties at once since it would be too slow otherwise to redo calculations
        res = {}

        mol = mol.copy()
        mol.filter(self.sel, _logger=False)

        residues = sequenceID((mol.resid, mol.chain, mol.insertion))

        backbone = mol.atomselect("backbone")
        ca_indices = np.where(mol.name == "CA")[0].astype(np.int32)
        chainids = mol.chain[ca_indices]
        resnames = mol.resname[ca_indices]
        proline_indices = np.array(resnames == "PRO", dtype=np.int32)

        _, chain_ids = np.unique(chainids, return_inverse=True)
        chain_ids = chain_ids.astype(np.int32)

        nco_indices = np.ones((residues.max() + 1, 3), dtype=np.int32) * -1
        natriums = np.where((mol.name == "N") & backbone)[0]
        carbons = np.where((mol.name == "C") & backbone)[0]
        oxygens = np.where((mol.name == "O") & backbone)[0]
        nco_indices[residues[natriums], 0] = natriums
        nco_indices[residues[carbons], 1] = carbons
        nco_indices[residues[oxygens], 2] = oxygens

        res["ca_indices"] = ca_indices
        res["nco_indices"] = nco_indices
        res["proline_indices"] = proline_indices
        res["chain_ids"] = chain_ids
        return res

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
        mol = mol.copy()
        molprops = self._getMolProp(mol, "all")

        ca_indices = molprops["ca_indices"]
        nco_indices = molprops["nco_indices"]
        proline_indices = molprops["proline_indices"]
        chain_ids = molprops["chain_ids"]

        mol.filter(self.sel, _logger=False)

        xyz = np.swapaxes(np.swapaxes(np.atleast_3d(mol.coords), 1, 2), 0, 1)
        xyz = np.array(xyz.copy(), dtype=np.float32) / 10  # converting to nm

        try:
            from mdtraj.geometry._geometry import _dssp as dssp
        except ImportError:
            raise ImportError(
                "To calculate secondary structure you need to install mdtraj with `conda install mdtraj -c conda-forge`"
            )

        natoms = np.unique(
            (
                nco_indices.shape[0],
                ca_indices.shape[0],
                proline_indices.shape[0],
                chain_ids.shape[0],
            )
        )
        if len(natoms) != 1:
            raise AssertionError(
                "Wrong dimensions in SS data arrays. Report this bug on the issue tracker."
            )
        data = dssp(xyz, nco_indices, ca_indices, proline_indices, chain_ids)

        if self.simplified:
            trans = str.maketrans("HGIEBTS ", "HHHEECCC")
            data = data.translate(trans)

        data = np.fromiter(data, dtype=np.dtype("U2"))
        data = data.reshape(mol.numFrames, len(chain_ids))

        if self.integer:
            if self.simplified:
                data[data == "H"] = 2
                data[data == "E"] = 1
                data[data == "C"] = 0
            else:
                data[data == "H"] = 3
                data[data == "B"] = 4
                data[data == "E"] = 5
                data[data == "G"] = 6
                data[data == "I"] = 7
                data[data == "T"] = 8
                data[data == "S"] = 9
                data[data == " "] = 10
            data = np.array(data, dtype=np.int32)
        # data[:, np.logical_not(protein_indices)] = 'NA'

        return data

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
        idx = mol.atomselect(f"{self.sel} and name CA", indexes=True)
        from pandas import DataFrame

        types = []
        indexes = []
        description = []
        for i in idx:
            types += ["secondary structure"]
            indexes += [i]
            description += [
                "Secondary structure of residue {} {}{}".format(
                    mol.resname[i], mol.resid[i], mol.insertion[i]
                )
            ]
        return DataFrame(
            {"type": types, "atomIndexes": indexes, "description": description}
        )


def _ssmap(sschar):
    # Map secondary structures to integers
    #     G = 3-turn helix (310 helix). Min length 3 residues.
    #     H = 4-turn helix (α helix). Min length 4 residues.
    #     I = 5-turn helix (π helix). Min length 5 residues.
    #     T = hydrogen bonded turn (3, 4 or 5 turn)
    #     E = extended strand in parallel and/or anti-parallel β-sheet conformation. Min length 2 residues.
    #     B = residue in isolated β-bridge (single pair β-sheet hydrogen bond formation)
    #     S = bend (the only non-hydrogen-bond based assignment).
    #     C = coil (residues which are not in any of the above conformations).
    ssnum = np.zeros((len(sschar), len(sschar[0])), dtype=np.uint8)

    for i in range(len(sschar)):
        sschar[i][sschar[i] == "G"] = "H"
        sschar[i][sschar[i] == "I"] = "H"
        sschar[i][sschar[i] == "B"] = "E"
        sschar[i][sschar[i] == "b"] = "E"
        sschar[i][sschar[i] == "S"] = "C"
        sschar[i][sschar[i] == "T"] = "C"
        ssnum[i, np.where(sschar[i] == "C")[0]] = 0
        ssnum[i, np.where(sschar[i] == "E")[0]] = 1
        ssnum[i, np.where(sschar[i] == "H")[0]] = 2
    return ssnum
